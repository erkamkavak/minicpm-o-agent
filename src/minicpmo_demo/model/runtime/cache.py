#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KV-cache, RoPE realignment, and sliding-window helpers."""

import logging
from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from transformers.cache_utils import DynamicCache

from .tensor_ops import rotate_half

logger = logging.getLogger(__name__)


# sliding window
@dataclass
class StreamingWindowConfig:
    text_window_high_tokens: int = 8000
    text_window_low_tokens: int = 6000


@dataclass
class DuplexWindowConfig:
    """duplex sliding window configuration

    sliding window mode:
    - "off": disable sliding window
    - "basic": basic sliding window (trigger by cache length)
    - "context": sliding window with context (trigger by unit number, preserve generated text to previous)
    """

    # sliding window mode
    sliding_window_mode: str = "off"  # "off" / "basic" / "context"

    # basic sliding window parameters
    basic_window_high_tokens: int = 8000  # high watermark: trigger sliding window when exceeded
    basic_window_low_tokens: int = 6000  # low watermark: keep to this value after sliding window

    # context sliding window parameters
    context_previous_max_tokens: int = 500  # previous maximum token number
    context_max_units: int = 24  # maximum unit number (trigger sliding window when exceeded)

    # verification mode (for comparison test)
    verify_mode: bool = False  # whether to enable verification log


def as_dynamic_cache(past_key_values):
    """Convert legacy tuple cache to DynamicCache if needed."""
    if isinstance(past_key_values, DynamicCache):
        return past_key_values

    if isinstance(past_key_values, tuple):
        return DynamicCache.from_legacy_cache(past_key_values)

    return past_key_values


def get_kv_cache_length(cache) -> int:
    """Get the sequence length of a KV cache.

    Args:
        cache: DynamicCache or tuple-based cache

    Returns:
        The number of tokens in the cache
    """
    if cache is None:
        return 0

    if isinstance(cache, DynamicCache):
        if not cache.key_cache or not cache.key_cache[0].numel():
            return 0
        return cache.key_cache[0].shape[-2]

    if isinstance(cache, tuple):
        return cache[0][0].shape[2]

    return 0


def get_rotary_cos_sin(
    head_dim: int,
    positions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    rope_theta: float = 10000.0,
    inv_freq_cache: Optional[Dict[Tuple, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cos and sin components for given positions.

    Args:
        head_dim: Dimension of each attention head
        positions: Position indices tensor
        device: Target device
        dtype: Target dtype
        rope_theta: RoPE base frequency (default 10000.0)
        inv_freq_cache: Optional cache dict for inverse frequencies

    Returns:
        Tuple of (cos, sin) tensors with shape [1, 1, seq_len, head_dim]
    """
    cache_key = (head_dim, device)

    inv_freq = inv_freq_cache.get(cache_key) if inv_freq_cache is not None else None
    if inv_freq is None or inv_freq.device != device or inv_freq.shape[0] != head_dim // 2:
        exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
        inv_freq = 1.0 / (rope_theta**exponent)
        if inv_freq_cache is not None:
            inv_freq_cache[cache_key] = inv_freq

    positions = positions.to(device=device, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", positions, inv_freq)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Use cat instead of repeat_interleave, consistent with model's original RotaryEmbedding
    # Original: emb = torch.cat((freqs, freqs), dim=-1) -> [f0, f1, ..., f_{d/2}, f0, f1, ..., f_{d/2}]
    cos_full = torch.cat([cos, cos], dim=-1).to(dtype=dtype)
    sin_full = torch.cat([sin, sin], dim=-1).to(dtype=dtype)
    cos_full = cos_full.unsqueeze(0).unsqueeze(0)
    sin_full = sin_full.unsqueeze(0).unsqueeze(0)
    return cos_full, sin_full


def realign_rotary_suffix(
    suffix_keys: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    rope_theta: float = 10000.0,
    inv_freq_cache: Optional[Dict[Tuple, torch.Tensor]] = None,
) -> torch.Tensor:
    """Realign RoPE position encoding after cache eviction.

    When tokens are dropped from the middle of a cache, the suffix tokens
    need their RoPE embeddings recalculated with new position indices.

    Args:
        suffix_keys: Key tensor to realign, shape [batch, heads, seq_len, head_dim]
        old_positions: Original position indices
        new_positions: New position indices after eviction
        rope_theta: RoPE base frequency
        inv_freq_cache: Optional cache dict for inverse frequencies

    Returns:
        Realigned key tensor with same shape as input
    """
    if suffix_keys.numel() == 0:
        return suffix_keys

    head_dim = suffix_keys.shape[-1]
    device = suffix_keys.device
    dtype = suffix_keys.dtype

    # Compute old position cos/sin
    cos_old, sin_old = get_rotary_cos_sin(head_dim, old_positions, device, dtype, rope_theta, inv_freq_cache)

    # Inverse transform: recover original key
    base = cos_old * suffix_keys - sin_old * rotate_half(suffix_keys)

    # Compute new position cos/sin
    cos_new, sin_new = get_rotary_cos_sin(head_dim, new_positions, device, dtype, rope_theta, inv_freq_cache)

    # Forward transform: re-encode with new positions
    return cos_new * base + sin_new * rotate_half(base)


def drop_tokens_from_cache(
    cache: Optional[DynamicCache | Tuple],
    length: int,
    preserve: int,
    position_offset: int,
    rope_theta: float = 10000.0,
    inv_freq_cache: Optional[Dict[Tuple, torch.Tensor]] = None,
) -> Tuple[Optional[DynamicCache], int, bool]:
    """Drop tokens from a KV cache while preserving system prompt.

    Removes tokens in the range [preserve, preserve + length) from the cache,
    realigning RoPE embeddings for the suffix.

    Args:
        cache: DynamicCache or tuple-based cache (will be converted to DynamicCache)
        length: Number of tokens to drop
        preserve: Number of tokens to preserve at the start (system prompt)
        position_offset: Current position offset for RoPE calculation
        rope_theta: RoPE base frequency
        inv_freq_cache: Optional cache dict for inverse frequencies

    Returns:
        Tuple of (cache, new_position_offset, success)
        Note: Tuple cache will be converted to DynamicCache. Modification is in-place.
    """
    if cache is None or length <= 0:
        return cache, position_offset, False

    cache = as_dynamic_cache(cache)

    total_len = get_kv_cache_length(cache)
    if total_len <= 0:
        return cache, position_offset, False

    preserve = min(preserve, total_len)
    available = total_len - preserve

    if available < length:
        logger.warning(
            "Cannot drop %d tokens: only %d available (total=%d, preserve=%d)",
            length,
            available,
            total_len,
            preserve,
        )
        return cache, position_offset, False

    suffix_len = total_len - preserve - length
    # note: after RoPE reindex, the position of cache has been compressed (from preserve start)
    # so here should not add position_offset, but use the actual layout of current cache
    suffix_offset = preserve + length  # suffix current position in cache
    prefix_offset = preserve  # suffix new position (follow preserve)

    # Prepare position tensors for RoPE realignment
    old_positions = None
    new_positions = None
    if suffix_len > 0:
        device = cache.key_cache[0].device
        old_positions = torch.arange(
            suffix_offset,
            suffix_offset + suffix_len,
            device=device,
            dtype=torch.long,
        )
        new_positions = torch.arange(
            prefix_offset,
            prefix_offset + suffix_len,
            device=device,
            dtype=torch.long,
        )

    keep_len = total_len - length

    # Process each layer (in-place modification)
    for layer_idx in range(len(cache.key_cache)):
        key_tensor = cache.key_cache[layer_idx]
        value_tensor = cache.value_cache[layer_idx]

        if not key_tensor.numel():
            continue

        # Preserve prefix (system prompt)
        prefix_keys = key_tensor[:, :, :preserve, :]
        prefix_values = value_tensor[:, :, :preserve, :]

        if suffix_len > 0:
            # Keep and realign suffix
            suffix_keys = key_tensor[:, :, preserve + length :, :]
            suffix_values = value_tensor[:, :, preserve + length :, :]

            if old_positions is not None and new_positions is not None and suffix_keys.numel():
                suffix_keys = realign_rotary_suffix(
                    suffix_keys,
                    old_positions,
                    new_positions,
                    rope_theta,
                    inv_freq_cache,
                )

            cache.key_cache[layer_idx] = torch.cat([prefix_keys, suffix_keys], dim=-2).contiguous()
            cache.value_cache[layer_idx] = torch.cat([prefix_values, suffix_values], dim=-2).contiguous()
        else:
            cache.key_cache[layer_idx] = prefix_keys.contiguous()
            cache.value_cache[layer_idx] = prefix_values.contiguous()

    cache.crop(keep_len)
    cache._seen_tokens = max(keep_len, 0)

    new_offset = position_offset + length
    logger.debug("Dropped %d tokens from cache, new length=%d", length, keep_len)

    return cache, new_offset, True
