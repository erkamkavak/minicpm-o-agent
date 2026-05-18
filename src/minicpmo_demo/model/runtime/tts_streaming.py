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

"""Streaming TTS token generation helpers."""

from dataclasses import dataclass
from typing import List
from typing import Union

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from transformers.cache_utils import DynamicCache

from .sampling import _validate_sampling_probs
from .tensor_ops import rotate_half


# tts
@dataclass
class TTSSamplingParams:
    top_p: float = 0.85
    min_p: float = 0.01
    top_k: int = 25
    repetition_penalty: float = 1.05
    temperature: float = 0.8
    win_size: int = 16
    tau_r: float = 0.1


class TTSStreamingGenerator:
    """
    Streaming generator for TTS that processes chunks and yields audio tokens in real-time.

    Supported attention types:
    - full_attention: Full attention, all tokens can attend to each other
    - sliding_window: Sliding window attention, KV cache is truncated to fixed size (token_window_size)
    - sliding_recompute: Sliding recompute, only keep previous chunk and recompute with current chunk
    - reindex: Keep first chunk as sink, reindex sliding window positions via RoPE rotation
    """

    def __init__(
        self,
        model,
        temperature: float,
        eos_token: Union[int, torch.Tensor],
        chunk_size: int = 25,  # s3tokenizer 1s = 25token
        tts_last_turn_tokens: torch.Tensor = None,
        logits_processors=None,
        logits_warpers=None,
    ):
        self.tts = model
        self.device = model.device
        self.temperature = torch.tensor([temperature], dtype=torch.float, device=self.device)
        self.eos_token = (
            torch.tensor(eos_token, device=self.device) if isinstance(eos_token, int) else eos_token.to(self.device)
        )

        self.num_vq = model.num_vq
        self.num_audio_tokens = model.num_audio_tokens
        self.recomputed_chunks = model.recomputed_chunks
        self.emb_code = model.emb_code
        self.head_code = model.head_code

        # Attention type and window sizes
        self.attention_type = model.attention_type  # "full_attention", "sliding_window", "sliding_recompute", "reindex"
        self.chunk_window_size = model.chunk_window_size  # chunk-level window for sliding_recompute (default 2)
        self.token_window_size = model.token_window_size  # token-level window for sliding_window/reindex (default 300)

        # RoPE config (for reindex mode)
        self.rope_theta = model.model.config.rope_theta
        self.head_dim = model.model.config.hidden_size // model.model.config.num_attention_heads

        # Logits processors
        self.logits_processors = logits_processors if logits_processors is not None else []
        # Logits warpers (like TopP/TopK), separate from processors
        self.logits_warpers = logits_warpers if logits_warpers is not None else []

        # initialize state
        self.past_key_values = None
        self.text_start_pos = 0
        self.idx = -1  # start from -1, become 0 when first called
        self.all_conditions = []
        self.all_generated_tokens = []
        self.tts_last_turn_tokens = tts_last_turn_tokens
        self.spk_emb = None

        audio_bos = [self.tts.audio_bos_token_id]
        audio_bos = torch.Tensor(audio_bos).to(self.tts.emb_text.weight.device, dtype=torch.long)

        self.audio_bos_embeds = self.tts.emb_text(audio_bos).unsqueeze(0)
        self.text_eos_embed = self.tts.emb_text(
            torch.tensor(
                [self.tts.config.text_eos_token_id],
                device=self.tts.emb_text.weight.device,
                dtype=torch.long,
            )
        ).unsqueeze(0)

        # buffer related, used to fill up chunk_size and yield to outside
        self.chunk_size = chunk_size
        self._token_buffer: List[torch.Tensor] = []

        # Chunk info tracking for sliding_recompute and reindex
        self._chunk_info: List[dict] = []
        self._total_seq_len = 0

        # Reindex mode: track sink (first chunk) length
        self._sink_kv_len = 0

    def _build_recompute_inputs(self, current_condition: torch.Tensor) -> torch.Tensor:
        """Build recompute inputs for sliding_recompute mode."""
        if len(self._chunk_info) == 0:
            return current_condition

        prev_chunk = self._chunk_info[-1]
        prev_condition = prev_chunk["condition"]
        prev_audio_tokens = prev_chunk["audio_tokens"]

        recompute_list = [prev_condition]
        if len(prev_audio_tokens) > 0:
            prev_audio_embeds = torch.cat([self.emb_code[0](tok) for tok in prev_audio_tokens], dim=1)
            recompute_list.append(prev_audio_embeds)

        recompute_list.append(current_condition)
        return torch.cat(recompute_list, dim=1)

    def _truncate_kv_cache_sliding_window(self):
        """Truncate KV cache for sliding_window mode."""
        if self.past_key_values is None:
            return

        if hasattr(self.past_key_values, "get_seq_length"):
            current_kv_len = self.past_key_values.get_seq_length()
        else:
            current_kv_len = self.past_key_values[0][0].shape[2]

        if current_kv_len <= self.token_window_size:
            return

        new_cache = DynamicCache()
        num_layers = (
            len(self.past_key_values.key_cache)
            if hasattr(self.past_key_values, "key_cache")
            else len(self.past_key_values)
        )

        for layer_idx in range(num_layers):
            if hasattr(self.past_key_values, "key_cache"):
                key = self.past_key_values.key_cache[layer_idx][:, :, -self.token_window_size :, :]
                value = self.past_key_values.value_cache[layer_idx][:, :, -self.token_window_size :, :]
            else:
                key = self.past_key_values[layer_idx][0][:, :, -self.token_window_size :, :]
                value = self.past_key_values[layer_idx][1][:, :, -self.token_window_size :, :]
            new_cache.update(key, value, layer_idx)

        self.past_key_values = new_cache

    @staticmethod
    def _apply_rope_rotation(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation to tensor."""
        return x * cos + rotate_half(x) * sin

    def _compute_rope_cos_sin(self, positions: torch.Tensor, device: torch.device, dtype: torch.dtype):
        """Compute RoPE cos and sin for given positions."""
        dim_half = self.head_dim // 2
        freq_seq = torch.arange(0, dim_half, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.rope_theta ** (freq_seq / dim_half))

        # positions: [seq_len]
        angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # [seq_len, dim_half]
        angles = torch.cat([angles, angles], dim=-1)  # [seq_len, head_dim]

        cos = angles.cos().to(dtype)
        sin = angles.sin().to(dtype)
        return cos, sin

    def _reindex_kv_cache(self):
        """
        Reindex KV cache for reindex mode:
        1. Keep first chunk as attention sink
        2. Keep last chunk
        3. Discard middle chunks
        4. Reindex the last chunk's key positions to be right after sink via RoPE rotation
        """
        if self.past_key_values is None or len(self._chunk_info) < 2:
            return

        # Get current KV cache length
        if hasattr(self.past_key_values, "get_seq_length"):
            current_kv_len = self.past_key_values.get_seq_length()
        else:
            current_kv_len = self.past_key_values[0][0].shape[2]

        # Calculate sink length (first chunk)
        sink_len = self._chunk_info[0]["condition_len"] + self._chunk_info[0]["audio_token_count"]

        # Last chunk length
        last_chunk = self._chunk_info[-1]
        last_chunk_len = last_chunk["condition_len"] + last_chunk["audio_token_count"]

        keep_len = sink_len + last_chunk_len

        if current_kv_len <= keep_len:
            # No need to truncate, but may need to reindex
            return

        # Step 1: Truncate KV cache - keep sink and last chunk
        device = self.past_key_values.key_cache[0].device
        dtype = self.past_key_values.key_cache[0].dtype

        new_cache = DynamicCache()
        num_layers = len(self.past_key_values.key_cache)

        # Calculate position delta for reindexing
        original_start_pos = current_kv_len - last_chunk_len
        new_start_pos = sink_len
        delta_positions = torch.arange(last_chunk_len, device=device) + (new_start_pos - original_start_pos)

        # Compute rotation cos/sin
        cos, sin = self._compute_rope_cos_sin(delta_positions, device, dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

        for layer_idx in range(num_layers):
            key_full = self.past_key_values.key_cache[layer_idx]
            value_full = self.past_key_values.value_cache[layer_idx]

            # Extract sink and last chunk
            key_sink = key_full[:, :, :sink_len, :]
            value_sink = value_full[:, :, :sink_len, :]
            key_last = key_full[:, :, -last_chunk_len:, :]
            value_last = value_full[:, :, -last_chunk_len:, :]

            # Apply RoPE rotation to reindex key positions
            key_last_reindexed = self._apply_rope_rotation(key_last, cos, sin)

            # Concatenate sink and reindexed last chunk
            key = torch.cat([key_sink, key_last_reindexed], dim=2)
            value = torch.cat([value_sink, value_last], dim=2)

            new_cache.update(key, value, layer_idx)

        self.past_key_values = new_cache

        # Update text_start_pos to reflect new positions
        self.text_start_pos = sink_len + last_chunk_len

    @torch.inference_mode()
    def generate_with_buffer(
        self,
        condition: torch.Tensor,
        text_finished: bool = False,
        max_new_token: int = 500,
    ):
        """input a condition embedding chunk, generate audio token each time,
        and accumulate to buffer, only yield when buffer satisfies chunk_size.

        Yields:
            torch.Tensor of shape [chunk_size] (2D: [1, chunk_size])
        """
        self.idx += 1
        self.device = self.tts.device

        # if text finished, first concatenate Text EOS
        if text_finished:
            condition = torch.cat([condition, self.text_eos_embed], dim=1)

        # always concatenate Audio BOS
        condition = torch.cat([condition, self.audio_bos_embeds], dim=1).to(self.device)

        self.all_conditions.append(condition)

        # Initialize current chunk info
        current_chunk_info = {
            "condition_len": condition.shape[1],
            "audio_token_count": 0,
            "condition": condition.clone(),
            "audio_tokens": [],
        }

        # Handle different attention types
        if self.attention_type == "sliding_recompute" and self.idx >= 1:
            # sliding_recompute: discard KV cache, recompute with previous + current chunk
            self.past_key_values = None
            current_condition = self._build_recompute_inputs(condition)
            self.text_start_pos = 0
        elif self.attention_type == "reindex" and self.idx >= 1:
            # reindex: truncate KV cache keeping sink + last chunk, reindex positions via RoPE
            self._reindex_kv_cache()
            current_condition = condition
            # text_start_pos is updated in _reindex_kv_cache
        else:
            current_condition = condition

        condition_length = current_condition.shape[1]
        prefill_len = condition_length
        finished = torch.zeros(1, dtype=torch.bool, device=self.device)
        chunk_generated_tokens = []

        for t in range(max_new_token):
            if t == 0:
                inputs_embeds = current_condition
                pos_ids = torch.arange(
                    self.text_start_pos,
                    self.text_start_pos + condition_length,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
            else:
                last = self.all_generated_tokens[-1]
                # last: [1,1], directly as code id
                inputs_embeds = self.emb_code[0](last)
                pos_ids = torch.tensor(
                    [self.text_start_pos + prefill_len + t - 1],
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

            outputs = self.tts.model(
                position_ids=pos_ids,
                past_key_values=self.past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
            hidden_states = outputs.last_hidden_state

            # Handle KV cache based on attention type
            if self.attention_type == "sliding_window":
                self.past_key_values = outputs.past_key_values
                self._truncate_kv_cache_sliding_window()
            else:
                self.past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            logits = logits[:, -1].float()

            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))

            logits /= self.temperature

            audio_bos = len(self.all_generated_tokens) == 0 and t == 0

            if not audio_bos:
                # use generated tokens (current chunk) as input for processor/warper (align with modeling_minicpmo)
                all_generated_tokens = torch.cat(self.all_generated_tokens, dim=1).to(self.device)  # [1, T]
                for processor in self.logits_processors:
                    logits = processor(all_generated_tokens, logits)

                for warper in self.logits_warpers:
                    logits = warper(all_generated_tokens, logits)
                del all_generated_tokens

            # sample next token (only use first codebook, same as generate)
            scores = F.softmax(logits, dim=-1)
            _validate_sampling_probs(scores, context="AudioTokenGenerator.streaming_generate.sample")
            idx_next = torch.multinomial(scores, num_samples=1)  # [(B*num_vq), 1]
            next_id = idx_next.view(-1, self.num_vq)[:, 0:1]  # only take first codebook → [B, 1]
            del scores

            if next_id.eq(
                self.eos_token
            ).any():  # generated audio eos token, means this chunk is finished, no longer generate new tokens
                finished[:] = True
            else:  # eos token cannot be added to buffer, he does not speak.
                # convert next_id to correct shape [1, 1], no num_vq dimension
                if next_id.dim() == 0:  # if scalar
                    next_tok = next_id.unsqueeze(0).unsqueeze(0)  # [1, 1]
                elif next_id.dim() == 1:  # if 1D [1]
                    next_tok = next_id.unsqueeze(0)  # [1, 1]
                else:
                    next_tok = next_id

                self.all_generated_tokens.append(next_tok)
                chunk_generated_tokens.append(next_tok)

                # Update chunk info for sliding_recompute
                current_chunk_info["audio_tokens"].append(next_tok.clone())
                current_chunk_info["audio_token_count"] += 1

                self._token_buffer.append(next_tok)

            if len(self._token_buffer) == 0:
                # case 1: if last text chunk, yield None
                if text_finished:
                    yield torch.empty(1, 0, dtype=torch.long, device=self.device), True
                    break
                # case 2: if not last text chunk, break directly
                else:
                    break
            else:  # buffer has something
                # case 1: if buffer is larger/equal to chunk_size, yield out
                if len(self._token_buffer) >= self.chunk_size:
                    batch = torch.cat(self._token_buffer[: self.chunk_size], dim=1)  # [1, chunk_size]
                    yield batch, False  # → [1, chunk_size]
                    # discard yielded part
                    self._token_buffer = self._token_buffer[self.chunk_size :]

                # case 2: if buffer is smaller than chunk_size
                else:
                    # if generation finished, and is the last text chunk, yield all remaining tokens, then break
                    if finished.all():
                        if text_finished:
                            batch = torch.cat(self._token_buffer, dim=1)  # [1, chunk_size]
                            yield batch, True  # → [1, chunk_size]
                            self._token_buffer = []
                            break
                        else:
                            # not the last text chunk, need to wait for next text chunk to fill up buffer, then this call ends
                            break
                    else:  # generation of this audio chunk is not finished, continue generating
                        continue

        # Save current chunk info for sliding_recompute
        self._chunk_info.append(current_chunk_info)
        self._total_seq_len += condition.shape[1] + len(chunk_generated_tokens)

        # Update text_start_pos based on attention type
        if self.attention_type == "sliding_recompute":
            self.text_start_pos += prefill_len + len(chunk_generated_tokens)
        else:
            self.text_start_pos += condition.shape[1] + len(chunk_generated_tokens)
        # note: remaining tokens in buffer will be kept, and accumulated next time
