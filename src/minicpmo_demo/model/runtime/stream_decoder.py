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

"""Duplex stream decoder and context-window cache management."""

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from .cache import DuplexWindowConfig
from .cache import drop_tokens_from_cache
from .cache import realign_rotary_suffix
from .sampling import _validate_sampling_probs
from .sampling import top_k_top_p_filtering

logger = logging.getLogger(__name__)


class StreamDecoder:
    def __init__(self, llm, tokenizer, special_token_ids=None, forbidden_token_ids=None):
        self.m = llm
        self.tokenizer = tokenizer
        self.listen_id = self.tokenizer.eos_token_id

        self.chunk_eos_id = self.tokenizer.convert_tokens_to_ids("<|chunk_eos|>")
        self.chunk_tts_eos_id = self.tokenizer.convert_tokens_to_ids("<|chunk_tts_eos|>")
        self.turn_eos_id = self.tokenizer.convert_tokens_to_ids("<|turn_eos|>")
        self.speak_id = self.tokenizer.convert_tokens_to_ids("<|speak|>")

        self.special_token_ids = special_token_ids if special_token_ids is not None else []

        # cache special tokens (used for context sliding window filtering)
        self._all_special_ids = set()
        self._all_special_tokens_text = set()
        if self.tokenizer:
            if hasattr(self.tokenizer, "all_special_ids"):
                self._all_special_ids = set(self.tokenizer.all_special_ids)
            if hasattr(self.tokenizer, "all_special_tokens"):
                self._all_special_tokens_text = set(self.tokenizer.all_special_tokens)

        custom_special_tokens = [
            "<unit>",
            "</unit>",
            "<image>",
            "</image>",
            "<slice>",
            "</slice>",
            "<|listen|>",
            "<|speak|>",
            "<|tts_bos|>",
            "<|tts_eos|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<|chunk_eos|>",
            "<|chunk_tts_eos|>",
            "<|turn_eos|>",
            "<|audio_start|>",
            "<|audio_end|>",
        ]
        self._all_special_tokens_text.update(custom_special_tokens)
        for token in custom_special_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                self._all_special_ids.add(token_id)

        if forbidden_token_ids is None:
            self.forbidden_token_ids = []
        elif isinstance(forbidden_token_ids, int):
            self.forbidden_token_ids = [self.forbidden_token_ids]
        else:
            self.forbidden_token_ids = forbidden_token_ids
        self.forbidden_token_ids.append(self.chunk_eos_id)

        assert isinstance(self.forbidden_token_ids, list)

        self.cache = None
        self.context = ""
        self.generated_tokens = []  # track generated tokens
        self.generated_special_tokens = []  # track generated special tokens
        self.reset()
        self.embeds = None
        self.system_embeds = None

        # sliding window related states
        self._unit_history: List[Dict[str, Any]] = []
        self._next_unit_id: int = 0
        self._pending_unit_id: Optional[int] = None
        self._pending_unit_start_cache_len: int = 0
        self._system_preserve_length: int = 0
        self._position_offset: int = 0
        self._window_config = DuplexWindowConfig()
        self._window_enabled: bool = True
        self._rope_inv_freq_cache: Dict[Tuple, torch.Tensor] = {}

        # context preserving sliding window states
        # initial cache layout: [prefix] [suffix] [units...]
        # after first sliding window: [prefix] [previous_marker + content] [suffix] [units...]
        #                              fixed     dynamic sliding region      fixed
        self._preserve_prefix_length: int = 0  # original prefix length (fixed)
        self._previous_content_length: int = 0  # previous content length (dynamic, including marker)
        self._suffix_token_ids: List[int] = []  # suffix token ids (e.g. <|im_end|>)

        # previous marker (added dynamically after first sliding window)
        self._previous_marker: str = "\n\nprevious: "  # fixed prefix marker
        self._previous_marker_token_ids: List[int] = []  # marker token ids (initialized)
        self._has_previous: bool = False  # whether previous marker has been added

        # previous content
        self._previous_text: str = ""  # accumulated generated text (without marker)
        self._previous_token_ids: List[int] = []  # previous full token ids (including marker)

        # validation statistics
        self._sliding_event_count: int = 0  # sliding window trigger count
        self._total_dropped_tokens: int = 0  # total dropped token count
        self._total_dropped_units: int = 0  # total dropped unit count

    def sliding_embeds(self):
        # tmp = system_embeds
        # tmp +-》 embeds after 5s
        # reset
        # feed
        pass

    def reset(self):
        self.context = ""
        self.cache = None
        self.generated_tokens = []
        self.generated_special_tokens = []
        self.embeds = None
        self.system_embeds = None

        # sliding window state reset
        old_unit_count = len(self._unit_history) if hasattr(self, "_unit_history") else 0
        self._unit_history = []
        self._next_unit_id = 0
        self._pending_unit_id = None
        self._pending_unit_start_cache_len = 0
        self._system_preserve_length = 0
        self._position_offset = 0
        self._rope_inv_freq_cache = {}

        # context preserving sliding window state reset
        self._preserve_prefix_length = 0
        self._previous_content_length = 0
        self._suffix_token_ids = []
        self._previous_marker = "\n\nprevious: "
        self._previous_marker_token_ids = []
        self._has_previous = False
        self._previous_text = ""
        self._previous_token_ids = []

        # validation statistics
        self._sliding_event_count = 0  # sliding window trigger count
        self._total_dropped_tokens = 0  # total dropped token count
        self._total_dropped_units = 0  # total dropped unit count

    def get_cache_length(self) -> int:
        if self.cache is None:
            return 0
        if isinstance(self.cache, DynamicCache):
            if len(self.cache.key_cache) > 0 and self.cache.key_cache[0].numel() > 0:
                return self.cache.key_cache[0].shape[2]
            return 0
        # Tuple cache format
        return self.cache[0][0].shape[2]

    def get_total_generated_tokens(self) -> int:
        return sum(len(u.get("generated_tokens", [])) for u in self._unit_history)

    def register_unit_start(self) -> int:
        self._pending_unit_id = self._next_unit_id
        self._pending_unit_start_cache_len = self.get_cache_length()
        return self._pending_unit_id

    def register_unit_end(
        self,
        input_type: str,
        generated_tokens: Optional[List[int]] = None,
        is_listen: bool = False,
        generated_text: Optional[str] = None,
    ):
        """Call when unit ends, record unit information

        Should be called after feeding </unit> token

        Args:
            input_type: "audio" / "video" / "omni" / "system"
            generated_tokens: tokens generated by the unit (token ids)
            is_listen: whether the unit is in listen state
            generated_text: text generated by the unit (used for context preserving mode)
        """
        if self._pending_unit_id is None:
            logger.warning("register_unit_end called without register_unit_start")
            return

        # calculate the length of the unit
        current_cache_len = self.get_cache_length()
        unit_len = current_cache_len - self._pending_unit_start_cache_len

        if unit_len > 0:
            entry = {
                "unit_id": self._pending_unit_id,
                "length": unit_len,
                "type": input_type,
                "generated_tokens": generated_tokens or [],
                "generated_text": generated_text or "",  # used for context preserving mode
                "is_listen": is_listen,
            }
            self._unit_history.append(entry)

        self._pending_unit_id = None
        self._pending_unit_start_cache_len = 0
        self._next_unit_id += 1

    def register_system_prompt(self):
        """Call after system prompt prefill, record preserve length"""
        self._system_preserve_length = self.get_cache_length()

    # sliding window core methods

    def _get_rope_theta(self) -> float:
        """get model rope_theta configuration"""
        return float(getattr(self.m.config, "rope_theta", 10000.0))

    def _drop_tokens_from_cache(self, length: int) -> bool:
        """remove specified number of tokens from cache (protect system prompt)

        remove tokens in the range [preserve, preserve + length)
        supports DynamicCache and tuple cache formats
        """
        if self.cache is None or length <= 0:
            return False

        cache_type = "DynamicCache" if isinstance(self.cache, DynamicCache) else "TupleCache"
        cache_len_before = self.get_cache_length()
        offset_before = self._position_offset

        new_cache, new_offset, success = drop_tokens_from_cache(
            cache=self.cache,
            length=length,
            preserve=self._system_preserve_length,
            position_offset=self._position_offset,
            rope_theta=self._get_rope_theta(),
            inv_freq_cache=self._rope_inv_freq_cache,
        )
        if success:
            self.cache = new_cache  # For DynamicCache this is the same object (in-place)
            self._position_offset = new_offset

        return success

    def _drop_unit(self, unit_id: int) -> bool:
        """remove specified unit"""
        entries = [u for u in self._unit_history if u["unit_id"] == unit_id]
        if not entries:
            return False

        total_len = sum(e["length"] for e in entries)
        if total_len <= 0:
            for e in entries:
                self._unit_history.remove(e)
            return False

        if not self._drop_tokens_from_cache(total_len):
            return False

        for e in entries:
            self._unit_history.remove(e)

        return True

    def _drop_next_unit(self) -> bool:
        """remove the earliest non-system unit"""
        for entry in self._unit_history:
            unit_id = entry.get("unit_id")
            if unit_id is None:
                continue
            # skip system type
            if entry.get("type") == "system":
                continue
            if self._drop_unit(unit_id):
                return True
        return False

    def enforce_window(self) -> bool:
        """enforce sliding window strategy (same as single-mode, only look at cache length)

        when cache length exceeds high water line, loop to remove the earliest unit,
        until cache length drops below the low water line.
        """
        if not self._window_enabled:
            return False

        cfg = self._window_config
        cache_len_before = self.get_cache_length()

        if cache_len_before <= cfg.basic_window_high_tokens:
            return False  # not above high water line, no trigger

        dropped_count = 0
        cache_len = cache_len_before
        while cache_len > cfg.basic_window_low_tokens:
            if not self._drop_next_unit():
                break
            dropped_count += 1
            cache_len = self.get_cache_length()

        if dropped_count > 0:
            # update statistics counters
            self._sliding_event_count += 1
            self._total_dropped_tokens += cache_len_before - cache_len
            self._total_dropped_units += dropped_count

            # consistency check
            expected = self._system_preserve_length + sum(u["length"] for u in self._unit_history)
            is_consistent = expected == cache_len
            if not is_consistent:
                logger.error(
                    "CONSISTENCY ERROR! preserve=%d + sum(units)=%d != cache=%d, offset=%d",
                    self._system_preserve_length,
                    sum(u["length"] for u in self._unit_history),
                    cache_len,
                    self._position_offset,
                )

        return dropped_count > 0

    # context preserving sliding window methods

    def register_system_prompt_with_context(
        self,
        suffix_token_ids: Optional[List[int]] = None,
        context_previous_marker: str = "\n\nprevious: ",
    ):
        """register system prompt (with context preserving mode)

        initial cache layout: [prefix] [suffix] [units...]
        after first sliding window: [prefix] [context_previous_marker + content] [suffix] [units...]

        when calling this method, cache should only have prefix (without previous marker)
        suffix will be fed in later

        Args:
            suffix_token_ids: suffix token ids (e.g. id of <|im_end|>)
            context_previous_marker: previous marker prefix, e.g. "\\n\\nprevious: "
        """
        # prefix = current cache content (fixed, without previous marker)
        self._preserve_prefix_length = self.get_cache_length()
        self._previous_content_length = 0  # initially no previous content
        self._suffix_token_ids = suffix_token_ids or []
        # total preserve length = prefix + suffix (initially no previous)
        self._system_preserve_length = self._preserve_prefix_length + len(self._suffix_token_ids)

        # initialize previous related states
        self._previous_marker = context_previous_marker
        self._previous_marker_token_ids = (
            self.tokenizer.encode(context_previous_marker, add_special_tokens=False) if self.tokenizer else []
        )
        self._has_previous = False
        self._previous_text = ""
        self._previous_token_ids = []

    @torch.no_grad()
    def update_system_prompt(
        self,
        new_prefix_token_ids: List[int],
        new_suffix_token_ids: List[int],
        new_ref_audio_embeds: Optional[torch.Tensor] = None,
    ) -> bool:
        """Replace the protected duplex system prompt span while preserving units."""
        if self.cache is None:
            return False

        total_len = self.get_cache_length()
        if total_len <= 0:
            return False

        if self._preserve_prefix_length or self._suffix_token_ids or self._previous_content_length:
            old_system_end = (
                self._preserve_prefix_length
                + self._previous_content_length
                + len(self._suffix_token_ids)
            )
        else:
            old_system_end = self._system_preserve_length
        old_system_end = min(old_system_end, total_len)

        units_len = total_len - old_system_end
        units_cache = self._slice_cache(old_system_end, total_len) if units_len > 0 else None

        embed_parts = []
        if new_prefix_token_ids:
            embed_parts.append(self.embed_tokens(new_prefix_token_ids))

        ref_audio_len = 0
        if new_ref_audio_embeds is not None:
            if new_ref_audio_embeds.dim() == 3 and new_ref_audio_embeds.shape[0] == 1:
                new_ref_audio_embeds = new_ref_audio_embeds.squeeze(0)
            if new_ref_audio_embeds.numel() > 0:
                ref_audio_len = new_ref_audio_embeds.shape[0]
                embed_parts.append(new_ref_audio_embeds.to(device=self.m.device))

        previous_token_ids = list(self._previous_token_ids)
        if previous_token_ids:
            embed_parts.append(self.embed_tokens(previous_token_ids))
        if new_suffix_token_ids:
            embed_parts.append(self.embed_tokens(new_suffix_token_ids))

        new_system_len = (
            len(new_prefix_token_ids)
            + ref_audio_len
            + len(previous_token_ids)
            + len(new_suffix_token_ids)
        )

        new_system_cache = None
        if embed_parts:
            system_embeds = torch.cat(embed_parts, dim=0)
            device = system_embeds.device
            position_ids = torch.arange(0, new_system_len, device=device).unsqueeze(0)
            outputs = self.m(
                inputs_embeds=system_embeds.unsqueeze(0),
                position_ids=position_ids,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
            new_system_cache = outputs.past_key_values

        if units_cache is not None and self._get_cache_len(units_cache) > 0:
            if old_system_end != new_system_len:
                units_cache = self._reindex_rope_for_cache(
                    units_cache,
                    old_start=old_system_end,
                    new_start=new_system_len,
                    length=units_len,
                )
            self.cache = self._concat_caches(new_system_cache, units_cache)
        else:
            self.cache = new_system_cache

        self._preserve_prefix_length = len(new_prefix_token_ids) + ref_audio_len
        self._previous_content_length = len(previous_token_ids)
        self._suffix_token_ids = list(new_suffix_token_ids)
        self._system_preserve_length = new_system_len
        self._position_offset = 0

        logger.info(
            "Updated duplex system prompt cache: old_system=%d, new_system=%d, units=%d",
            old_system_end,
            new_system_len,
            units_len,
        )
        return True

    def _extract_generated_text(self, units: List[Dict[str, Any]]) -> Tuple[str, List[int]]:
        """extract generated text and token ids from units

        Args:
            units: list of units to extract

        Returns:
            (text, token_ids): concatenated text and token ids (filtered out special tokens)
        """
        text_parts = []
        token_ids = []

        for u in units:
            # only keep generated content of non-listen units
            if u.get("is_listen", False):
                continue
            gen_text = u.get("generated_text", "")
            gen_tokens = u.get("generated_tokens", [])

            # filter out special tokens from text
            if gen_text:
                clean_text = gen_text
                for st in self._all_special_tokens_text:
                    clean_text = clean_text.replace(st, "")
                if clean_text.strip():
                    text_parts.append(clean_text)

            # filter out special tokens
            if gen_tokens:
                filtered_tokens = [t for t in gen_tokens if t not in self._all_special_ids]
                token_ids.extend(filtered_tokens)

        return "".join(text_parts), token_ids

    def _rebuild_cache_with_previous(
        self,
        new_previous_tokens: List[int],
        units_to_keep_len: Optional[int] = None,
    ) -> bool:
        """rebuild cache, insert new previous content between prefix and suffix

        cache layout change:
        [prefix] [old_prev] [suffix] [old_units]  →  [prefix] [new_prev] [suffix] [remaining_units]

        Args:
            new_previous_tokens: new previous token ids
            units_to_keep_len: length of units to keep (from cache end backwards)
                                if None, calculate based on unit_history

        Returns:
            whether successful rebuild
        """
        if self.cache is None:
            return False

        old_previous_len = self._previous_content_length
        new_previous_len = len(new_previous_tokens)
        suffix_len = len(self._suffix_token_ids)
        total_cache_len = self.get_cache_length()

        # calculate length of units to keep
        if units_to_keep_len is None:
            units_to_keep_len = sum(u["length"] for u in self._unit_history)

        # special case: if previous is unchanged (new and old are empty), no need to rebuild prefix+suffix part of cache
        # but still need to reindex units RoPE (because a unit was deleted, position changed)
        if new_previous_len == 0 and old_previous_len == 0:
            # cache layout: [prefix(7)] [suffix(1)] [units...]
            # only keep prefix + suffix + remaining_units
            preserve_len = self._preserve_prefix_length + suffix_len

            # simply slice cache: [prefix+suffix] + [remaining_units]
            # remaining_units in cache end
            if units_to_keep_len > 0:
                # [0:preserve_len] + [total-units_to_keep_len:total]
                prefix_suffix_cache = self._slice_cache(0, preserve_len)
                units_cache = self._slice_cache(total_cache_len - units_to_keep_len, None)

                # calculate number of dropped tokens
                dropped_tokens = total_cache_len - preserve_len - units_to_keep_len

                # reindex units RoPE: position from (preserve_len + dropped_tokens) to preserve_len
                # note: no position_offset, because cache position has been compressed (from 0 start)
                if dropped_tokens > 0:
                    old_start = preserve_len + dropped_tokens
                    new_start = preserve_len
                    units_cache = self._reindex_rope_for_cache(units_cache, old_start, new_start, units_to_keep_len)

                self.cache = self._concat_caches(prefix_suffix_cache, units_cache)
            else:
                self.cache = self._slice_cache(0, preserve_len)

            return True

        # 1. get prefix cache (fixed)
        prefix_end = self._preserve_prefix_length
        prefix_cache = self._slice_cache(0, prefix_end)

        # 2. get units cache to keep (from end)
        units_start_in_old_cache = total_cache_len - units_to_keep_len
        units_cache = None
        if units_to_keep_len > 0:
            units_cache = self._slice_cache(units_start_in_old_cache, None)

        # 3. calculate new previous + suffix cache (needs forward)
        # merge previous tokens and suffix tokens
        prev_suffix_tokens = new_previous_tokens + self._suffix_token_ids
        prev_suffix_len = len(prev_suffix_tokens)

        new_prefix_prev_suffix_cache = prefix_cache
        if prev_suffix_len > 0:
            # Embed tokens
            prev_suffix_embeds = self.embed_tokens(prev_suffix_tokens)
            # calculate start position (after prefix)
            start_pos = self._preserve_prefix_length + self._position_offset

            # forward calculate KV cache
            with torch.no_grad():
                device = prev_suffix_embeds.device
                position_ids = torch.arange(
                    start_pos,
                    start_pos + prev_suffix_len,
                    device=device,
                ).unsqueeze(0)

                # use prefix cache as past_key_values
                outputs = self.m(
                    inputs_embeds=(
                        prev_suffix_embeds.unsqueeze(0) if prev_suffix_embeds.dim() == 2 else prev_suffix_embeds
                    ),
                    position_ids=position_ids,
                    past_key_values=prefix_cache,
                    use_cache=True,
                    return_dict=True,
                )
                # new cache contains prefix + new_previous + suffix
                new_prefix_prev_suffix_cache = outputs.past_key_values

        # 4. adjust units cache RoPE
        # new layout: [prefix] [new_prev] [suffix] [units]
        # note: no position_offset, because cache position has been compressed (from 0 start)
        new_system_total = prefix_end + new_previous_len + suffix_len
        if units_cache is not None and self._get_cache_len(units_cache) > 0:
            old_start = units_start_in_old_cache
            new_start = new_system_total

            if old_start != new_start:
                units_cache = self._reindex_rope_for_cache(units_cache, old_start, new_start, units_to_keep_len)

        # 5. concatenate new cache
        if units_cache is not None and self._get_cache_len(units_cache) > 0:
            self.cache = self._concat_caches(new_prefix_prev_suffix_cache, units_cache)
        else:
            self.cache = new_prefix_prev_suffix_cache

        # 6. update length
        self._previous_content_length = new_previous_len
        # total preserve length = prefix + previous + suffix
        self._system_preserve_length = prefix_end + new_previous_len + suffix_len

        # print detailed cache layout information
        prev_text_preview = self._previous_text[:50] + "..." if len(self._previous_text) > 50 else self._previous_text
        suffix_preview = self.tokenizer.decode(self._suffix_token_ids) if self._suffix_token_ids else ""
        return True

    def _slice_cache(self, start: int, end: Optional[int], clone: bool = True):
        """slice cache

        Args:
            start: start position
            end: end position (None means to end)
            clone: whether to clone (default True, to prevent shared memory issues)
        """
        if self.cache is None:
            return None
        if isinstance(self.cache, DynamicCache):
            # DynamicCache
            new_key_cache = [
                k[:, :, start:end, :].clone() if clone else k[:, :, start:end, :] for k in self.cache.key_cache
            ]
            new_value_cache = [
                v[:, :, start:end, :].clone() if clone else v[:, :, start:end, :] for v in self.cache.value_cache
            ]
            new_cache = DynamicCache()
            new_cache.key_cache = new_key_cache
            new_cache.value_cache = new_value_cache
            return new_cache
        else:
            # Tuple cache
            if clone:
                return tuple(
                    (layer[0][:, :, start:end, :].clone(), layer[1][:, :, start:end, :].clone()) for layer in self.cache
                )
            else:
                return tuple((layer[0][:, :, start:end, :], layer[1][:, :, start:end, :]) for layer in self.cache)

    @staticmethod
    def _get_cache_len(cache) -> int:
        if cache is None:
            return 0
        if isinstance(cache, DynamicCache):
            if len(cache.key_cache) > 0 and cache.key_cache[0].numel() > 0:
                return cache.key_cache[0].shape[2]
            return 0

        if cache and cache[0] and cache[0][0] is not None:
            return cache[0][0].shape[2]
        return 0

    @staticmethod
    def _concat_caches(cache1, cache2):
        if cache1 is None:
            return cache2
        if cache2 is None:
            return cache1

        if isinstance(cache1, DynamicCache):
            new_cache = DynamicCache()
            new_cache.key_cache = [torch.cat([k1, k2], dim=2) for k1, k2 in zip(cache1.key_cache, cache2.key_cache)]
            new_cache.value_cache = [
                torch.cat([v1, v2], dim=2) for v1, v2 in zip(cache1.value_cache, cache2.value_cache)
            ]
            return new_cache
        else:
            return tuple(
                (
                    torch.cat([layer1[0], layer2[0]], dim=2),
                    torch.cat([layer1[1], layer2[1]], dim=2),
                )
                for layer1, layer2 in zip(cache1, cache2)
            )

    def _reindex_rope_for_cache(self, cache, old_start: int, new_start: int, length: int):
        """reindex RoPE position for cache"""
        if cache is None or length <= 0:
            return cache

        if isinstance(cache, DynamicCache):
            device = cache.key_cache[0].device if cache.key_cache else None
        else:
            device = cache[0][0].device if cache and cache[0] else None

        if device is None:
            return cache

        old_positions = torch.arange(old_start, old_start + length, device=device, dtype=torch.long)
        new_positions = torch.arange(new_start, new_start + length, device=device, dtype=torch.long)

        rope_theta = self._get_rope_theta()

        if isinstance(cache, DynamicCache):
            new_key_cache = []
            for k in cache.key_cache:
                new_k = realign_rotary_suffix(k, old_positions, new_positions, rope_theta, self._rope_inv_freq_cache)
                new_key_cache.append(new_k)
            cache.key_cache = new_key_cache
            return cache
        else:
            new_cache = []
            for layer in cache:
                new_k = realign_rotary_suffix(
                    layer[0], old_positions, new_positions, rope_theta, self._rope_inv_freq_cache
                )
                new_cache.append((new_k, layer[1]))
            return tuple(new_cache)

    def _update_previous(
        self,
        new_text: str,
        new_tokens: List[int],
        max_tokens: int,
    ) -> None:
        """update previous context (also update cache)

        when first sliding window, dynamically add marker + text, subsequent sliding window append text
        when content exceeds max_tokens, truncate content (keep marker)
        rebuild cache to maintain consistency

        Args:
            new_text: new text
            new_tokens: new token ids
            max_tokens: previous content maximum token count (without marker)
        """
        marker_len = len(self._previous_marker_token_ids)
        tokens_to_drop = 0

        # if no new content, do not add marker, but still need to rebuild cache
        if not new_tokens and not new_text:
            # still need to rebuild cache (because a unit was deleted)
            self._rebuild_cache_with_previous(self._previous_token_ids)
            return

        if not self._has_previous:
            # when first has actual content: add marker + text
            self._previous_text = new_text
            self._previous_token_ids = self._previous_marker_token_ids.copy() + new_tokens
            self._has_previous = True
        else:
            # subsequent sliding window: append text to previous
            self._previous_text += new_text
            self._previous_token_ids.extend(new_tokens)

        # calculate token count of content (without marker)
        content_token_count = len(self._previous_token_ids) - marker_len

        # check if need to truncate content (keep marker)
        if content_token_count > max_tokens:
            # truncate left content, keep marker + latest max_tokens content
            tokens_to_drop = content_token_count - max_tokens
            old_text = self._previous_text
            # keep marker + truncated content
            content_tokens = self._previous_token_ids[marker_len + tokens_to_drop :]
            self._previous_token_ids = self._previous_marker_token_ids.copy() + content_tokens
            # redecode text (only decode content part)
            try:
                self._previous_text = self.tokenizer.decode(
                    content_tokens,
                    skip_special_tokens=True,
                )
            except Exception as e:
                logger.warning("_update_previous: decode failed: %s", e)

        # rebuild cache
        self._rebuild_cache_with_previous(self._previous_token_ids)

    def _drop_unit_with_context(
        self,
        unit_id: int,
        max_previous_tokens: int,
    ) -> Tuple[bool, str, List[int]]:
        """remove specified unit and return its generated content (for context preserving)

        process:
        1. extract generated content of unit
        2. remove unit from cache (without prefix+previous)
        3. append generated content to previous
        4. rebuild cache (in _update_previous)

        Args:
            unit_id: unit ID to remove
            max_previous_tokens: previous maximum token count

        Returns:
            (success, extracted_text, extracted_tokens): whether successful, extracted text and tokens
        """
        entries = [u for u in self._unit_history if u["unit_id"] == unit_id]
        if not entries:
            return False, "", []

        # extract generated content
        extracted_text, extracted_tokens = self._extract_generated_text(entries)

        # calculate total length
        total_len = sum(e["length"] for e in entries)
        if total_len <= 0:
            for e in entries:
                self._unit_history.remove(e)
            return False, extracted_text, extracted_tokens

        cache_before = self.get_cache_length()

        # remove from unit_history (record for later processing)
        for e in entries:
            self._unit_history.remove(e)

        # note: here no longer call _drop_tokens_from_cache
        # because _update_previous will rebuild the entire cache

        # update previous (also rebuild cache)
        self._update_previous(extracted_text, extracted_tokens, max_previous_tokens)

        return True, extracted_text, extracted_tokens

    def _drop_next_unit_with_context(self, max_previous_tokens: int) -> bool:
        """remove the earliest non-system unit (with context preserving)"""
        for entry in self._unit_history:
            unit_id = entry.get("unit_id")
            if unit_id is None:
                continue
            if entry.get("type") == "system":
                continue
            success, _, _ = self._drop_unit_with_context(unit_id, max_previous_tokens)
            if success:
                return True
        return False

    def enforce_window_with_context(self) -> bool:
        """context preserving sliding window execution

        when unit count exceeds max_units, remove the earliest unit,
        and accumulate its generated content to previous.
        Cache will be automatically rebuilt in _update_previous.

        Returns:
            whether sliding window is executed
        """
        if not self._window_enabled:
            return False

        cfg = self._window_config

        if cfg.sliding_window_mode != "context":
            # if not context mode, fallback to basic sliding window
            return self.enforce_window()

        cache_len_before = self.get_cache_length()
        units_before = len(self._unit_history)

        # context preserving mode: only check if unit count exceeds limit
        # (previous exceeds limit in _update_previous will automatically truncate left)
        if units_before <= cfg.context_max_units:
            return False

        # sliding window loop: remove unit until count ≤ max_units
        dropped_count = 0
        while len(self._unit_history) > cfg.context_max_units:
            if not self._drop_next_unit_with_context(cfg.context_previous_max_tokens):
                break

            dropped_count += 1

        cache_len_after = self.get_cache_length()

        if dropped_count > 0:
            # update statistics counter
            self._sliding_event_count += 1
            self._total_dropped_tokens += cache_len_before - cache_len_after
            self._total_dropped_units += dropped_count

            # consistency check
            expected = self._system_preserve_length + sum(u["length"] for u in self._unit_history)

        return dropped_count > 0

    def get_previous_context(self) -> Tuple[str, List[int]]:
        """get current accumulated previous context

        Returns:
            (previous_text, previous_token_ids): current accumulated text and token ids
        """
        return self._previous_text, self._previous_token_ids.copy()

    def get_window_stats(self) -> Dict[str, Any]:
        """get sliding window statistics"""
        unit_lengths = [u["length"] for u in self._unit_history]
        return {
            "cache_length": self.get_cache_length(),
            "unit_count": len(self._unit_history),
            "unit_lengths": unit_lengths,
            "unit_total_length": sum(unit_lengths),
            "system_preserve_length": self._system_preserve_length,
            "position_offset": self._position_offset,
            "window_enabled": self._window_enabled,
            "total_generated_tokens": self.get_total_generated_tokens(),
            "pending_unit_id": self._pending_unit_id,
            "next_unit_id": self._next_unit_id,
            "config": {
                "sliding_window_mode": self._window_config.sliding_window_mode,
                "basic_window_high_tokens": self._window_config.basic_window_high_tokens,
                "basic_window_low_tokens": self._window_config.basic_window_low_tokens,
                "context_previous_max_tokens": self._window_config.context_previous_max_tokens,
                "context_max_units": self._window_config.context_max_units,
            },
            # context preserving related
            "preserve_prefix_length": self._preserve_prefix_length,
            "previous_content_length": self._previous_content_length,
            "suffix_token_count": len(self._suffix_token_ids),
            "previous_text_length": len(self._previous_text),
            "previous_token_count": len(self._previous_token_ids),
            "has_system_template": self._system_prompt_template is not None,
        }

    def _verify_consistency(self) -> bool:
        """verify unit history and cache length consistency"""
        expected = self._system_preserve_length + sum(u["length"] for u in self._unit_history)
        actual = self.get_cache_length()
        return expected == actual

    def print_verification_summary(self) -> Dict[str, Any]:
        """print verification summary (for comparing off/basic/context mode)

        Returns:
            dictionary containing key verification data
        """
        cfg = self._window_config

        # collect all generated text
        all_generated_text = []
        all_generated_tokens = []
        for u in self._unit_history:
            if not u.get("is_listen", False):
                gen_text = u.get("generated_text", "")
                gen_tokens = u.get("generated_tokens", [])
                if gen_text:
                    all_generated_text.append(gen_text)
                if gen_tokens:
                    all_generated_tokens.extend(gen_tokens)

        combined_text = "".join(all_generated_text)

        summary = {
            "mode": cfg.sliding_window_mode,
            "final_cache_length": self.get_cache_length(),
            "final_unit_count": len(self._unit_history),
            "sliding_event_count": self._sliding_event_count,
            "total_dropped_tokens": self._total_dropped_tokens,
            "total_dropped_units": self._total_dropped_units,
            "total_generated_tokens": len(all_generated_tokens),
            "generated_text": combined_text,
            "previous_text": self._previous_text,
            "previous_token_count": len(self._previous_token_ids),
            "position_offset": self._position_offset,
            "system_preserve_length": self._system_preserve_length,
        }

        return summary

    def set_window_config(self, config: DuplexWindowConfig) -> None:
        """set sliding window configuration"""
        self._window_config = config

    def set_window_enabled(self, enabled: bool) -> None:
        """enable/disable sliding window"""
        old_enabled = self._window_enabled
        self._window_enabled = enabled

    def get_context(self):
        return self.context

    def embed_token(self, tid):
        if isinstance(tid, int):
            tid = torch.tensor([tid], device=self.m.device)
        return self.m.model.embed_tokens(tid)

    def embed_tokens(self, token_ids: List[int]) -> torch.Tensor:
        """batch embed multiple tokens

        Args:
            token_ids: list of token ids

        Returns:
            embeddings tensor [L, H]
        """
        if not token_ids:
            return torch.empty(0, self.m.config.hidden_size, device=self.m.device)
        tids = torch.tensor(token_ids, device=self.m.device)
        return self.m.model.embed_tokens(tids)

    @torch.no_grad()
    def feed(self, embeds: torch.Tensor, return_logits: bool = False):
        """
        embeds : [L, H]   —— new embedding sequence fed into model at once
        """
        L = embeds.size(0)
        device = embeds.device

        past_len = self.get_cache_length()
        pos_ids = torch.arange(past_len, past_len + L, device=device).unsqueeze(0)  # [1, L]

        out = self.m(
            inputs_embeds=embeds.unsqueeze(0),  # [1, L, H]
            position_ids=pos_ids,
            past_key_values=self.cache,
            # use_cache = True,
            return_dict=True,
            output_hidden_states=True,
            # attention_mask=attention_mask
        )
        self.cache = out.past_key_values

        if return_logits:
            logits = self.m.lm_head(out.hidden_states[-1])[:, -1]  # [1, vocab]
            return logits, out.hidden_states[-1]

    @torch.no_grad()
    def decode(
        self,
        logits,
        mode: Literal["sampling", "greedy"] = "sampling",
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        listen_top_k=None,
        listen_prob_scale=1.0,
        text_repetition_penalty=1.05,
        text_repetition_window_size=512,
        length_penalty=1.1,
    ):
        """
        Args:
            logits:
            mode: sampling or greedy
            temperature:
            top_k:
            top_p:
            listen_top_k: force listen_id to be in top-k to keep
            listen_prob_scale: multiply listen_id probability by a weight (<1 means decrease, >1 means increase)
            text_repetition_penalty: repetition penalty coefficient, >1.0 means decrease repetition, <1.0 means increase repetition
            text_repetition_window_size: repetition penalty window size

        Sampling strategy:
            1. first sample all tokens with original logits (apply temperature)
            2. if sampled chunk_eos, return directly (keep the original model's decision of when to stop)
            3. if not sampled chunk_eos, mask it (set logit to -inf), continue sampling text tokens
            4. apply repetition penalty, top-k, top-p, etc. to the text tokens for the final sampling
        """

        logits = logits.clone()

        # 0. independently check chunk_eos before sampling
        eos_id = self.chunk_eos_id

        with torch.no_grad():
            if mode == "greedy":
                sampled_token = torch.argmax(logits[0]).item()
            else:
                original_probs = F.softmax(logits[0], dim=-1)
                _validate_sampling_probs(original_probs, context="StreamDecoder.decode.initial_chunk_eos_sample")
                sampled_token = torch.multinomial(original_probs, num_samples=1).item()

            # if sampled chunk_eos, return directly
            if sampled_token == eos_id:
                next_token_id = torch.tensor([eos_id], device=logits.device)
                next_token_str = self.tokenizer.decode(next_token_id)

                return next_token_id

        # if not sampled chunk_eos, set its logit to -inf
        if self.forbidden_token_ids:
            logits[:, self.forbidden_token_ids] = float("-inf")

        # 1. apply repetition penalty
        if text_repetition_penalty != 1.0 and len(self.generated_tokens) > 0:
            # get recent tokens (within window size) considering special tokens and normal tokens
            recent_tokens = self.generated_tokens[-text_repetition_window_size:]

            # make it unique
            recent_tokens = list(set(recent_tokens))

            # apply penalty to repeated tokens
            for token_id in recent_tokens:
                if token_id < logits.size(-1):  # ensure token_id is in vocabulary range
                    if text_repetition_penalty > 1.0:
                        # penalize repetition: decrease logits
                        logits[0, token_id] /= text_repetition_penalty
                    else:
                        # encourage repetition: increase logits
                        logits[0, token_id] *= 1.0 / text_repetition_penalty

        # 2. apply length penalty to turn_eos token
        # higher length_penalty → suppress turn_eos → model 更不容易结束当前 turn，倾向更长输出
        if length_penalty != 1.0:
            turn_eos_id = self.turn_eos_id
            if logits[0, turn_eos_id] > 0:
                logits[0, turn_eos_id] = logits[0, turn_eos_id] / length_penalty
            else:
                logits[0, turn_eos_id] = logits[0, turn_eos_id] * length_penalty

        if listen_prob_scale != 1.0:  # modify listen token logit separately
            logits[0, self.listen_id] *= listen_prob_scale

        listen_rank = (logits[0] > logits[0, self.listen_id]).sum().item()

        if listen_top_k is not None and listen_rank < listen_top_k:  # listen_id is in top-k, return directly
            next_token_id = torch.tensor([self.listen_id], device=logits.device)
            next_token_str = self.tokenizer.decode(next_token_id)

            if next_token_str == "<|listen|>":
                self.context += " "
            else:
                self.context += next_token_str

            return next_token_id

        if mode == "greedy":
            next_token_id = torch.argmax(logits, dim=-1)
        elif mode == "sampling":
            logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            _validate_sampling_probs(probs, context="StreamDecoder.decode.post_filter_sample")
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            raise ValueError(f"Unsupported decode mode: {mode}")

        if next_token_id.item() not in self.special_token_ids:
            self.generated_tokens.append(next_token_id.item())
        else:
            self.generated_special_tokens.append(next_token_id.item())

        return next_token_id
