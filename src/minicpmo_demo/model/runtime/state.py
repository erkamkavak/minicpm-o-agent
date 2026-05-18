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

"""Serializable runtime state snapshots."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SpeculativeSnapshot:
    """Speculative snapshot for VAD speculative rollback.

    Used in VAD speculative execution: creates a snapshot after streaming_prefill
    and before streaming_generate. If speculation fails (user continues speaking),
    the state can be restored to continue streaming_prefill.

    Implementation:
    - LLM KV Cache: only record length, restore by truncation (zero extra VRAM)
    - Audio KV Cache: requires cloning, as generate sets it to None
    - Mel processor: save full state snapshot (including buffer)
    """

    # KV Cache length (for truncation recovery)
    llm_cache_length: int
    audio_cache_length: int

    # session state
    new_user_msg: bool
    llm_generated: bool
    llm_generate_completed: bool

    # Round management
    next_round_id: int
    pending_round_id: Optional[int]
    omni_chunk_history_length: int

    # TTS state (requires cloning, but usually small)
    tts_last_turn_tokens: Optional[torch.Tensor]

    # Streaming processor state
    audio_chunk_idx: int

    # Mel processor state snapshot (including buffer)
    mel_processor_snapshot: Optional[dict] = None

    # Audio encoder KV cache (requires cloning to ensure determinism after recovery)
    audio_past_key_values: Optional[tuple] = None

    # timestamp (for debugging)
    timestamp: float = 0.0

    # debug field: for verifying correctness of recovery
    llm_cache_checksum: Optional[float] = None  # LLM KV Cache first layer K sum
    audio_cache_checksum: Optional[float] = None  # Audio KV Cache first layer K sum
    mel_buffer_checksum: Optional[float] = None  # Mel buffer sum

    # RNG state (key: for ensuring determinism of dithering etc. after recovery)
    rng_state_cpu: Optional[torch.Tensor] = None  # torch CPU RNG state
    rng_state_cuda: Optional[torch.Tensor] = None  # torch CUDA RNG state (if on GPU)

    def summary(self) -> str:
        mel_buf_len = 0
        if self.mel_processor_snapshot:
            buf = self.mel_processor_snapshot.get("buffer")
            if buf is not None:
                mel_buf_len = len(buf)
        return (
            f"llm_cache={self.llm_cache_length}, "
            f"audio_cache={self.audio_cache_length}, "
            f"audio_chunk_idx={self.audio_chunk_idx}, "
            f"mel_buffer={mel_buf_len}, "
            f"history_len={self.omni_chunk_history_length}, "
            f"new_user_msg={self.new_user_msg}, "
            f"llm_generated={self.llm_generated}"
        )
