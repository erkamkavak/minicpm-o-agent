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

"""Duplex streaming capability composed into the MiniCPMO model wrapper."""

import logging
import os
import threading
import time
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch

from ..components import gen_logits
from ..processing_minicpmo import MiniCPMOProcessor
from ..runtime.cache import DuplexWindowConfig
from ..runtime.stream_decoder import StreamDecoder
from ..runtime.tensor_ops import torch_clone_recursive

logger = logging.getLogger(__name__)


class DuplexCapability:
    """双工能力组件 - 封装双工对话的全部逻辑
    
    采用组合模式，接受外部传入的 MiniCPMO 实例，不自己加载模型。
    
    使用方式（推荐，使用透传方法）：
        model = MiniCPMO.from_pretrained(...)
        model.init_unified(...)
        
        model.duplex_prepare(...)
        model.duplex_prefill(...)
        result = model.duplex_generate()
        model.duplex_set_break()
    
    使用方式（直接访问）：
        model.duplex.prepare(...)
        model.duplex.streaming_prefill(...)
        result = model.duplex.streaming_generate(...)
    """
    
    def __init__(
        self,
        model: "MiniCPMO",
        generate_audio: bool = True,
        ls_mode: str = "explicit",
        device: str = "cuda",
        **kwargs,
    ):
        """初始化双工能力组件
        
        Args:
            model: 外部传入的 MiniCPMO 实例（已加载好的模型）
            generate_audio: 是否生成音频输出
            ls_mode: Listen/Speak 模式
            device: 设备
            **kwargs: 其他配置参数
        """
        # 使用外部传入的模型（不自己加载！）
        self.model = model
        self.device = device
        
        self.session_logs = []
        self.session_start_time = None
        self.log_file_path = None
        
        self.generate_audio = generate_audio
        self.ls_mode = ls_mode
        
        # 使用 model 的 processor 和 tokenizer
        if not hasattr(self.model, "processor") or self.model.processor is None:
            self.model.processor = MiniCPMOProcessor.from_pretrained(
                self.model.config._name_or_path, trust_remote_code=True
            )
        self.processor = self.model.processor
        self.tokenizer = self.processor.tokenizer
        
        self.break_event = threading.Event()
        self.session_stop_event = threading.Event()
        
        # llm generation_config
        self.max_new_speak_tokens_per_chunk = kwargs.get("max_new_speak_tokens_per_chunk", 20)
        self.text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.05)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_k = kwargs.get("top_k", 20)
        self.top_p = kwargs.get("top_p", 0.8)
        self.text_repetition_window_size = kwargs.get("text_repetition_window_size", 512)
        self.listen_prob_scale = kwargs.get("listen_prob_scale", 1.0)
        self.force_listen_count = kwargs.get("force_listen_count", 0)
        
        # tts generation_config
        tts_temp_value = kwargs.get("tts_temperature", 0.8)
        self.tts_temperature = torch.tensor([tts_temp_value], dtype=torch.float, device=self.device)
        self.tts_repetition_penalty = kwargs.get("tts_repetition_penalty", 1.05)
        
        # stream config
        self.CHUNK_MS = kwargs.get("chunk_ms", 1000)
        self.FIRST_CHUNK_MS = kwargs.get("first_chunk_ms", 1035)
        self.CNN_REDUNDANCY_MS = kwargs.get("cnn_redundancy_ms", 20)
        self.SAMPLE_RATE = kwargs.get("sample_rate", 16000)
        
        self.model.CHUNK_MS = self.CHUNK_MS
        self.model.FIRST_CHUNK_MS = self.FIRST_CHUNK_MS
        self.model.CNN_REDUNDANCY_MS = self.CNN_REDUNDANCY_MS
        self.model.SAMPLE_RATE = self.SAMPLE_RATE
        
        # special tokens
        self.unit_token_id = self.tokenizer.convert_tokens_to_ids("<unit>")
        self.image_start_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids("</image>")
        self.slice_start_token_id = self.tokenizer.convert_tokens_to_ids("<slice>")
        self.slice_end_token_id = self.tokenizer.convert_tokens_to_ids("</slice>")
        
        self.listen_token_id = self.tokenizer.convert_tokens_to_ids("<|listen|>")
        self.speak_token_id = self.tokenizer.convert_tokens_to_ids("<|speak|>")
        self.tts_bos_token_id = self.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
        self.tts_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|tts_eos|>")
        
        self.chunk_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|chunk_eos|>")
        self.chunk_tts_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|chunk_tts_eos|>")
        self.turn_eos_token_id = self.tokenizer.convert_tokens_to_ids("<|turn_eos|>")
        
        self.chunk_terminator_token_ids = [self.listen_token_id, self.chunk_eos_token_id, self.chunk_tts_eos_token_id]
        self.turn_terminator_token_ids = [self.turn_eos_token_id]
        self.chunk_speak_token_ids = [self.speak_token_id]
        
        self.tts_pad_id = self.tokenizer.convert_tokens_to_ids("<|tts_pad|>")
        bad_token_ids = getattr(self.tokenizer, "bad_token_ids", [])
        self.forbidden_token_ids = [self.tts_pad_id] + list(bad_token_ids)
        # self.forbidden_token_ids = [] + list(bad_token_ids)
        
        self.decoder = StreamDecoder(
            llm=self.model.llm, tokenizer=self.tokenizer, forbidden_token_ids=self.forbidden_token_ids
        )
        self._prefix_system_prompt: Optional[str] = None
        self._suffix_system_prompt: Optional[str] = None
        self._ref_audio: Optional[np.ndarray] = None
        
        # 滑窗模式: "off" / "basic" / "context"
        sliding_window_mode = kwargs.get("sliding_window_mode", "off")
        
        # 不带 Context 的滑窗参数
        basic_window_high_tokens = kwargs.get("basic_window_high_tokens", 4000)
        basic_window_low_tokens = kwargs.get("basic_window_low_tokens", 3500)
        
        # 带 Context 的滑窗参数
        context_previous_max_tokens = kwargs.get("context_previous_max_tokens", 500)
        context_max_units = kwargs.get("context_max_units", 24)
        
        self.decoder.set_window_config(
            DuplexWindowConfig(
                sliding_window_mode=sliding_window_mode,
                basic_window_high_tokens=basic_window_high_tokens,
                basic_window_low_tokens=basic_window_low_tokens,
                context_previous_max_tokens=context_previous_max_tokens,
                context_max_units=context_max_units,
            )
        )
        # 根据 mode 设置滑窗开关
        window_enabled = sliding_window_mode != "off"
        self.decoder.set_window_enabled(window_enabled)
        logger.info(
            "[DuplexCapability] Sliding window: mode=%s, high=%d, low=%d, prev_max=%d, max_units=%d",
            sliding_window_mode,
            basic_window_high_tokens,
            basic_window_low_tokens,
            context_previous_max_tokens,
            context_max_units,
        )
        
        self.tts_logits_processors = None
        self.tts_eos_token = None
        if self.generate_audio:
            self.tts_logits_processors = gen_logits(
                num_code=self.model.tts.config.num_audio_tokens,
                repetition_penalty=self.tts_repetition_penalty,
            )
            self.tts_eos_token = torch.tensor(
                [self.model.tts.config.num_audio_tokens - 1],
                dtype=torch.long,
                device=self.device,
            )
        
        self._reset_streaming_state()
        
        logger.info("[DuplexCapability] 初始化完成")

    def set_break_event(self):
        self.break_event.set()

    def clear_break_event(self):
        self.break_event.clear()

    def set_session_stop(self):
        self.session_stop_event.set()
        self.break_event.set()

    def clear_session_stop(self):
        self.session_stop_event.clear()

    def is_break_set(self) -> bool:
        return self.break_event.is_set()

    def is_session_stop_set(self) -> bool:
        return self.session_stop_event.is_set()

    def _init_token2wav_cache(self, prompt_wav_path: str):
        self.model.tts.audio_tokenizer.cache = None
        flow_cache, hift_cache = self.model.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)
        self.flow_cache_base = torch_clone_recursive(flow_cache)
        self.hift_cache_base = torch_clone_recursive(hift_cache)
        self.pre_lookahead = int(self.model.tts.audio_tokenizer.flow.pre_lookahead_len)
        self.token2wav_initialized = True

    def _reset_token2wav_for_new_turn(self):
        if self.token2wav_initialized:
            self.model.tts.audio_tokenizer.stream_cache = torch_clone_recursive(self.flow_cache_base)
            self.model.tts.audio_tokenizer.hift_cache_dict = torch_clone_recursive(self.hift_cache_base)
            self.token2wav_buffer = [4218] * 3  # silence token prefix

    def _reset_streaming_state(self):
        self.audio_chunk_idx = 0
        self.current_turn_ended = True
        self.speak_count = 0
        self.res_ids = []
        self.total_ids = []
        self.total_hidden = []

        # TTS state
        self.tts_text_start_pos = 0
        self.tts_past_key_values = None
        self.tts_current_turn_start_time = None

        # token2wav state
        self.token2wav_initialized = False
        self.token2wav_buffer = []
        self.flow_cache_base = None
        self.hift_cache_base = None

        # Audio prefill state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.pending_logits: Optional[torch.Tensor] = None
        self.current_mode: Optional[str] = None

        # Force listen state
        self._streaming_generate_count = 0

        # Deferred finalize state（必须清理，否则上一个 session 异常退出后
        # _pending_finalize 残留，导致新 session 的第一个 prefill 报错）
        self._pending_finalize = None

        # 禁止连续 chunk 解码 <|tts_pad|>
        self._last_chunk_had_tts_pad = False

        # Schema tracking: 记录完整的 prefill + generate token 序列
        # prefill_schema_tokens: 每个元素是一个 unit 的 prefill token 列表
        # 格式: [[unit0_prefill_tokens], [unit1_prefill_tokens], ...]
        self.prefill_schema_tokens = []
        self._current_unit_prefill_tokens = []

    def prepare(
        self,
        prefix_system_prompt: Optional[str] = None,
        suffix_system_prompt: Optional[str] = None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        context_previous_marker: str = "\n\nprevious: ",
    ):
        self.clear_break_event()
        self.clear_session_stop()

        self.session_start_time = time.time()

        self._reset_streaming_state()
        self.decoder.reset()

        self.model.init_streaming_processor()
        self._prefix_system_prompt = prefix_system_prompt
        self._suffix_system_prompt = suffix_system_prompt
        self._ref_audio = ref_audio

        if prompt_wav_path is not None and prompt_wav_path and self.generate_audio:
            self._init_token2wav_cache(prompt_wav_path)
            self._reset_token2wav_for_new_turn()

        # Prefill system prompt prefix (batch)
        if prefix_system_prompt:
            tokens = self.tokenizer.encode(prefix_system_prompt, add_special_tokens=False)
            if tokens:
                embeds = self.decoder.embed_tokens(tokens)
                self.decoder.feed(embeds)

        # Prefill reference audio
        if ref_audio is not None:
            data = self.processor.process_audio([ref_audio])
            embeds_nested = self.model.get_audio_embedding(data, chunk_length=self.model.config.audio_chunk_length)
            embeds = torch.cat([t for g in embeds_nested for t in g], dim=0) if embeds_nested else None
            if embeds is not None:
                self.decoder.feed(embeds)

        # 注册 system prompt 保护长度（滑窗时保护这部分不被移除）
        if prefix_system_prompt or suffix_system_prompt or ref_audio is not None:
            logger.info("[Duplex] prepare: registering system prompt protection")
            if self.decoder._window_config.sliding_window_mode == "context":
                # Context 保留模式：
                # 初始化时布局: [prefix] [suffix] [units...]
                # 首次滑窗后布局: [prefix] [context_previous_marker + content] [suffix] [units...]
                # 此时先注册 prefix 长度，再 feed suffix
                # 获取 suffix token ids
                suffix_token_ids = []
                if suffix_system_prompt:
                    suffix_token_ids = self.tokenizer.encode(suffix_system_prompt, add_special_tokens=False)

                # 注册（此时 cache 只有 prefix，还没有 suffix，也没有 previous）
                self.decoder.register_system_prompt_with_context(
                    suffix_token_ids=suffix_token_ids,
                    context_previous_marker=context_previous_marker,  # 首次滑窗时动态添加
                )

                # 现在 feed suffix (batch)
                if suffix_token_ids:
                    suffix_embeds = self.decoder.embed_tokens(suffix_token_ids)
                    self.decoder.feed(suffix_embeds)

                logger.info(
                    "[Duplex] prepare: context-preserve mode, prefix=%d, suffix=%d tokens, marker='%s'",
                    self.decoder._preserve_prefix_length,
                    len(suffix_token_ids),
                    context_previous_marker.replace("\n", "\\n"),
                )
            else:
                # 非 context 保留模式：先 feed suffix，再注册总长度
                if suffix_system_prompt:
                    tokens = self.tokenizer.encode(suffix_system_prompt, add_special_tokens=False)
                    if tokens:
                        suffix_embeds = self.decoder.embed_tokens(tokens)
                        self.decoder.feed(suffix_embeds)
                self.decoder.register_system_prompt()

        if prefix_system_prompt or suffix_system_prompt:
            if ref_audio is not None:
                full_prompt = (prefix_system_prompt or "") + "[音频嵌入]" + (suffix_system_prompt or "")
            else:
                full_prompt = (prefix_system_prompt or "") + (suffix_system_prompt or "")

            return full_prompt

        return ""

    @property
    def supports_system_prompt_update(self) -> bool:
        """Whether the current duplex session has a cache that can be updated."""
        return self.decoder.get_cache_length() > 0

    def update_system_prompt(
        self,
        prefix_system_prompt: Optional[str] = None,
        suffix_system_prompt: Optional[str] = None,
        ref_audio: Optional[np.ndarray] = None,
    ) -> bool:
        """Update the protected system prompt span without resetting duplex history."""
        if self.needs_finalize:
            logger.info("[Duplex] update_system_prompt: flushing pending finalize")
            self.finalize_unit()

        if not self.supports_system_prompt_update:
            raise RuntimeError("Duplex session is not prepared; call prepare() first")

        if suffix_system_prompt is None:
            suffix_system_prompt = self._suffix_system_prompt or ""
        if prefix_system_prompt is None:
            prefix_system_prompt = self._prefix_system_prompt or ""

        prefix_token_ids = (
            self.tokenizer.encode(prefix_system_prompt, add_special_tokens=False)
            if prefix_system_prompt
            else []
        )
        suffix_token_ids = (
            self.tokenizer.encode(suffix_system_prompt, add_special_tokens=False)
            if suffix_system_prompt
            else []
        )

        if ref_audio is None:
            ref_audio = self._ref_audio

        ref_embeds = None
        if ref_audio is not None:
            data = self.processor.process_audio([ref_audio])
            embeds_nested = self.model.get_audio_embedding(
                data,
                chunk_length=self.model.config.audio_chunk_length,
            )
            ref_embeds = torch.cat([t for g in embeds_nested for t in g], dim=0) if embeds_nested else None

        updated = self.decoder.update_system_prompt(
            new_prefix_token_ids=prefix_token_ids,
            new_suffix_token_ids=suffix_token_ids,
            new_ref_audio_embeds=ref_embeds,
        )
        if updated:
            self._prefix_system_prompt = prefix_system_prompt
            self._suffix_system_prompt = suffix_system_prompt
            self._ref_audio = ref_audio
            self._reset_token2wav_for_new_turn()
            self.tts_past_key_values = None
            self.tts_text_start_pos = 0
        return updated

    @torch.no_grad()
    def streaming_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[list] = None,
        max_slice_nums: Union[int, List[int]] = 1,
        batch_vision_feed: bool = False,
    ):
        """Streaming prefill - called once per second, processing audio/video data

        Args:
            audio_waveform: audio waveform data
            frame_list: image frame list
            max_slice_nums: maximum number of slices for HD image encoding (default 1, no slicing)
                           Can be an int (same for all images) or a list matching frame_list length
            batch_vision_feed: if True, batch all vision embeddings into a single feed call for better performance.
                              if False (default), feed each embedding individually (original behavior).

        Process:
            0. determine mode based on input: AUDIO / VISION / OMNI
            1. feed <unit> token
            2. get and feed image embed (if frame_list) - return pending logits in VISION MODE
            3. get and feed audio embed (if audio_waveform) - return pending logits in AUDIO/OMNI MODE

        Returns:
            dict with keys:
                - success: bool
                - cost_vision_process: float (image processing time)
                - cost_vision_embed: float (vision embedding time)
                - cost_vision_feed: float (vision feed time)
                - cost_audio_process: float (audio processing time)
                - cost_audio_embed: float (audio embedding time)
                - cost_audio_feed: float (audio feed time)
                - cost_all: float (total time)
        """
        # Fail-fast: 上一轮 finalize 未完成就进入下一轮 prefill
        if self.needs_finalize:
            raise RuntimeError(
                "streaming_prefill called before finalize_unit()! "
                "必须在 streaming_generate 之后调用 finalize_unit() 再进入下一轮 prefill。"
            )

        start_time = time.time()
        cost_vision_process = 0.0
        cost_vision_embed = 0.0
        cost_vision_feed = 0.0
        cost_audio_process = 0.0
        cost_audio_embed = 0.0
        cost_audio_feed = 0.0

        def _make_result(success, reasons=""):
            reason = reasons
            if isinstance(reasons, list):
                reason = "; ".join(reasons)

            return {
                "success": success,
                "reason": reason,
                "cost_vision_process": cost_vision_process,
                "cost_vision_embed": cost_vision_embed,
                "cost_vision_feed": cost_vision_feed,
                "cost_audio_process": cost_audio_process,
                "cost_audio_embed": cost_audio_embed,
                "cost_audio_feed": cost_audio_feed,
                "cost_all": time.time() - start_time,
            }

        if self.is_session_stop_set():
            return _make_result(False)
        # prefill 只检查 session_stop，不检查 break_event
        # Force Listen 通过 per-chunk force_listen_override 参数在 generate() 中处理

        has_frames = frame_list is not None and len(frame_list) > 0
        has_audio = audio_waveform is not None and len(audio_waveform) > 0

        if has_frames and has_audio:
            mode = "OMNI"
        elif has_frames:
            mode = "VISION"
        elif has_audio:
            mode = "AUDIO"
        else:
            return _make_result(False)

        self.pending_logits = None

        # 滑窗：记录 unit 开始位置
        logger.info(
            "[Duplex] streaming_prefill: mode=%s, has_frames=%s, has_audio=%s, starting unit",
            mode,
            has_frames,
            has_audio,
        )
        self.decoder.register_unit_start()

        # Schema tracking: 开始新的 unit，记录 prefill tokens
        self._current_unit_prefill_tokens = []

        # Step 1: Feed <unit> token
        self.decoder.feed(self.decoder.embed_token(self.unit_token_id))
        self._current_unit_prefill_tokens.append(self.unit_token_id)

        # Step 2: process image
        if has_frames:
            t0 = time.time()

            # Normalize max_slice_nums to a list matching frame_list length
            if isinstance(max_slice_nums, int):
                max_slice_nums_list = [max_slice_nums] * len(frame_list)
            else:
                max_slice_nums_list = list(max_slice_nums)
                if len(max_slice_nums_list) != len(frame_list):
                    raise ValueError(
                        f"max_slice_nums list length ({len(max_slice_nums_list)}) "
                        f"must match frame_list length ({len(frame_list)})"
                    )

            # Check if all max_slice_nums are the same (can use batch processing)
            all_same = len(set(max_slice_nums_list)) == 1

            if all_same:
                # All images use the same max_slice_nums, use batch processing
                processed_frames = self.processor.process_image(frame_list, max_slice_nums=max_slice_nums_list[0])
                if self.device:
                    processed_frames = processed_frames.to(self.device)
            else:
                # Different max_slice_nums per image, process individually and merge
                all_pixel_values = []
                all_tgt_sizes = []
                for frame, max_slices in zip(frame_list, max_slice_nums_list):
                    pf = self.processor.process_image([frame], max_slice_nums=max_slices)
                    if self.device:
                        pf = pf.to(self.device)
                    # pf["pixel_values"][0] is the list of slices for this image
                    all_pixel_values.extend(pf["pixel_values"][0])
                    # pf["tgt_sizes"][0] is the array of target sizes for this image's slices
                    if hasattr(pf["tgt_sizes"][0], "tolist"):
                        all_tgt_sizes.extend(pf["tgt_sizes"][0].tolist())
                    else:
                        all_tgt_sizes.extend(list(pf["tgt_sizes"][0]))

                # Reconstruct processed_frames with merged data
                processed_frames = {
                    "pixel_values": [all_pixel_values],
                    "tgt_sizes": [torch.tensor(all_tgt_sizes) if all_tgt_sizes else []],
                }

            cost_vision_process = time.time() - t0

            t0 = time.time()
            # Get vision embeddings for all images (each may have multiple slices)
            # vision_hidden_states is a list, one entry per input image
            # Each entry contains embeddings for [source_image, slice_1, slice_2, ...]
            vision_hidden_states = self.model.get_vision_embedding(processed_frames)
            cost_vision_embed = time.time() - t0

            if vision_hidden_states is not None and len(vision_hidden_states) > 0:
                t0 = time.time()

                # vision_hidden_states[0] contains ALL slices from ALL images (flattened)
                # Shape: [total_slices, 64, D] where total_slices = sum of slices across all images
                # We need to know how many slices each image has to correctly group them

                # Calculate slice counts for each image using get_sliced_grid (lightweight, no actual slicing)
                slice_counts = []  # e.g., [5, 9] means img1 has 5 slices (1 source + 4 HD), img2 has 9
                for frame_idx, frame in enumerate(frame_list):
                    max_slices = max_slice_nums_list[frame_idx]
                    if hasattr(frame, "size"):
                        # get_sliced_grid returns [M, N] grid or None if no slicing needed
                        # Total images = 1 (source) + M * N (HD slices)
                        grid = self.processor.image_processor.get_sliced_grid(
                            frame.size, max_slices, nerver_split=False
                        )
                        if grid is not None:
                            slice_counts.append(1 + grid[0] * grid[1])  # 1 source + M*N slices
                        else:
                            slice_counts.append(1)  # No slicing, only source image
                    else:
                        slice_counts.append(1)  # Default: single image, no slicing

                # Get the flattened embeddings tensor
                # vision_hidden_states is a list with one element (the batch)
                # vision_hidden_states[0] shape: [total_slices, 64, D]
                all_embeds = vision_hidden_states[0]

                # Collect all feed operations first, then execute
                # This allows us to identify the last token for VISION mode logits
                feed_operations = []  # List of (embed, is_last_for_vision_mode, token_id_or_none)

                embed_idx = 0  # Current index in all_embeds
                for img_idx, num_slices in enumerate(slice_counts):
                    if num_slices == 0:
                        continue

                    # First embedding is always the source image (downsampled overview)
                    # Feed <image> token
                    feed_operations.append(
                        (self.decoder.embed_token(self.image_start_token_id), False, self.image_start_token_id)
                    )
                    # Feed source image embedding (shape: [64, D]) - use None to indicate embedding
                    feed_operations.append((all_embeds[embed_idx], False, None))
                    # Feed </image> token
                    feed_operations.append(
                        (self.decoder.embed_token(self.image_end_token_id), False, self.image_end_token_id)
                    )
                    embed_idx += 1

                    # Remaining embeddings are HD slices (if num_slices > 1)
                    if num_slices > 1:
                        for slice_i in range(1, num_slices):
                            # Feed <slice> token
                            feed_operations.append(
                                (self.decoder.embed_token(self.slice_start_token_id), False, self.slice_start_token_id)
                            )
                            # Feed slice embedding (shape: [64, D])
                            feed_operations.append((all_embeds[embed_idx], False, None))
                            # Feed </slice> token
                            feed_operations.append(
                                (self.decoder.embed_token(self.slice_end_token_id), False, self.slice_end_token_id)
                            )
                            embed_idx += 1

                # Mark the last operation for VISION mode logits
                if feed_operations:
                    feed_operations[-1] = (feed_operations[-1][0], True, feed_operations[-1][2])

                # Execute feed operations
                if batch_vision_feed and feed_operations:
                    # Batch mode: concatenate all embeddings and feed at once
                    # This reduces LLM forward passes from N to 1
                    #
                    # NOTE: Batch mode may have slight numerical differences compared to for-loop mode
                    # due to floating-point precision in attention computation. This is expected behavior
                    # for causal attention with incremental vs batch computation.

                    all_embeds_list = []
                    for embed, is_last, token_id in feed_operations:
                        # Ensure all embeddings have shape [L, H]
                        if embed.dim() == 1:
                            embed = embed.unsqueeze(0)
                        all_embeds_list.append(embed)

                    # Concatenate all embeddings
                    # torch.cat requires consistent dtype; embeddings should already be same dtype
                    all_embeds_to_feed = torch.cat(all_embeds_list, dim=0)  # [total_L, H]

                    # Debug: log embedding info for first unit
                    if self.audio_chunk_idx == 0:
                        logger.info(
                            "[Batch Vision Feed] total_L=%d, dtype=%s, device=%s, sum=%.4f",
                            all_embeds_to_feed.shape[0],
                            all_embeds_to_feed.dtype,
                            all_embeds_to_feed.device,
                            all_embeds_to_feed.sum().item(),
                        )

                    if mode == "VISION":
                        # VISION mode needs logits from the last token
                        self.pending_logits, _ = self.decoder.feed(all_embeds_to_feed, return_logits=True)
                    else:
                        # OMNI mode: just feed, wait for audio to get logits
                        self.decoder.feed(all_embeds_to_feed)

                    # Schema tracking: record all token IDs and embedding markers
                    for embed, is_last, token_id in feed_operations:
                        if token_id is not None:
                            self._current_unit_prefill_tokens.append(token_id)
                        else:
                            embed_dim = embed.shape[0] if len(embed.shape) > 1 else 1
                            self._current_unit_prefill_tokens.append(("img", embed_dim))
                else:
                    # Original mode: feed each embedding individually

                    # Debug: log embedding info for first unit
                    if self.audio_chunk_idx == 0:
                        total_len = sum(e.shape[0] if len(e.shape) > 1 else 1 for e, _, _ in feed_operations)
                        embed_sum = sum(e.sum().item() for e, _, _ in feed_operations)
                        logger.info(
                            "[For-loop Vision Feed] total_L=%d, sum=%.4f",
                            total_len,
                            embed_sum,
                        )

                    for embed, is_last, token_id in feed_operations:
                        if mode == "VISION" and is_last:
                            # Get logits from the last token
                            self.pending_logits, _ = self.decoder.feed(embed, return_logits=True)
                        else:
                            self.decoder.feed(embed)
                        # Schema tracking: 记录 token ID 或 embedding 标记
                        if token_id is not None:
                            self._current_unit_prefill_tokens.append(token_id)
                        else:
                            # 用元组标记 image embedding: ("img", dim)
                            embed_dim = embed.shape[0] if len(embed.shape) > 1 else 1
                            self._current_unit_prefill_tokens.append(("img", embed_dim))
                # For OMNI MODE, no pending logits needed here (wait for audio)

                cost_vision_feed = time.time() - t0

        # Step 3: process audio (if any)
        if has_audio:
            # accumulate audio to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_waveform])

            # calculate required audio length
            if self.audio_chunk_idx == 0:
                required_samples = int(self.FIRST_CHUNK_MS * self.SAMPLE_RATE / 1000)
                if len(self.audio_buffer) < required_samples:
                    padding_samples = required_samples - len(self.audio_buffer)
                    padding = np.zeros(padding_samples, dtype=np.float32)
                    self.audio_buffer = np.concatenate([padding, self.audio_buffer])
            else:
                required_samples = int(self.CHUNK_MS * self.SAMPLE_RATE / 1000)

            need_samples = self.processor.get_streaming_chunk_size()
            if len(self.audio_buffer) < need_samples:
                return _make_result(False, f"音频不足: 需要 {need_samples} 样本, 只有 {len(self.audio_buffer)}")

            audio_chunk = self.audio_buffer[:need_samples]

            t0 = time.time()
            batch_feature = self.processor.process_audio_streaming(
                audio_chunk,
                reset=False,
                return_batch_feature=True,
            )

            if batch_feature is None or batch_feature.audio_features.shape[-1] == 0:
                return _make_result(False, "流式音频处理返回空")

            # metadata
            batch_feature.chunk_idx = self.audio_chunk_idx
            batch_feature.use_extra_context = True
            batch_feature.prefix_extra_frames = 0 if self.audio_chunk_idx == 0 else 2
            batch_feature.suffix_extra_frames = 2

            batch_feature = batch_feature.to(self.device)
            cost_audio_process = time.time() - t0

            t0 = time.time()
            embeds_nested = self.model.get_audio_embedding_streaming(
                batch_feature,
                use_extra_context=batch_feature.use_extra_context,
                prefix_extra_frames=batch_feature.prefix_extra_frames,
                suffix_extra_frames=batch_feature.suffix_extra_frames,
            )
            audio_embeds = torch.cat([t for g in embeds_nested for t in g], dim=0)
            cost_audio_embed = time.time() - t0

            t0 = time.time()
            self.pending_logits, _ = self.decoder.feed(audio_embeds, return_logits=True)
            cost_audio_feed = time.time() - t0

            # Schema tracking: 用元组标记 audio embedding: ("audio", dim)
            embed_dim = audio_embeds.shape[0] if len(audio_embeds.shape) > 1 else 1
            self._current_unit_prefill_tokens.append(("audio", embed_dim))

            if self.audio_chunk_idx == 0:
                cfg = self.processor._streaming_mel_processor.get_config()
                consumed_ms = int(cfg.get("effective_first_chunk_ms", self.FIRST_CHUNK_MS))
                consumed_samples = int(consumed_ms * self.SAMPLE_RATE / 1000)
            else:
                consumed_samples = int(self.CHUNK_MS * self.SAMPLE_RATE / 1000)

            self.audio_buffer = self.audio_buffer[consumed_samples:]

            self.audio_chunk_idx += 1

        self.current_mode = mode

        # for VISION mode, need to manually increase chunk count (AUDIO and OMNI modes already increased in _process_audio_buffer)
        if mode == "VISION":
            self.audio_chunk_idx += 1

        # Schema tracking: 保存当前 unit 的 prefill tokens
        self.prefill_schema_tokens.append(self._current_unit_prefill_tokens)

        return _make_result(True)

    def _make_generate_result(
        self,
        start_time: float,
        is_listen: bool = True,
        text: str = "",
        audio_waveform=None,
        end_of_turn: bool = False,
        cost_llm: float = 0.0,
        cost_tts_prep: float = 0.0,
        cost_tts: float = 0.0,
        cost_token2wav: float = 0.0,
        n_tokens: int = 0,
        n_tts_tokens: int = 0,
    ) -> dict:
        """构造 streaming_generate 的标准返回 dict"""
        return {
            "is_listen": is_listen,
            "text": text,
            "audio_waveform": audio_waveform if audio_waveform is not None else self._generate_silence_waveform(),
            "end_of_turn": end_of_turn,
            "current_time": self.audio_chunk_idx,
            "cost_llm": cost_llm,
            "cost_tts_prep": cost_tts_prep,
            "cost_tts": cost_tts,
            "cost_token2wav": cost_token2wav,
            "cost_all": time.time() - start_time,
            "n_tokens": n_tokens,
            "n_tts_tokens": n_tts_tokens,
        }

    @property
    def needs_finalize(self) -> bool:
        """是否有待执行的 finalize（用于调用方检查）"""
        return getattr(self, "_pending_finalize", None) is not None

    @torch.no_grad()
    def streaming_generate(
        self,
        prompt_wav_path=None,
        max_new_speak_tokens_per_chunk=20,
        decode_mode: str = "sampling",
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        listen_prob_scale=1.0,
        listen_top_k=None,
        text_repetition_penalty=1.05,
        text_repetition_window_size=512,
        length_penalty=1.1,
        force_listen_override: bool = False,
    ):
        """生成响应。返回后必须调用 finalize_unit()（除非 needs_finalize 为 False）。

        调用方可以选择调度策略：
        - 模式 A（异步）: generate → 返回结果 → finalize（与网络传输重叠）
        - 模式 B（同步）: generate → finalize → 返回结果
        """
        start_time = time.time()

        if self.is_session_stop_set():
            self._pending_finalize = None  # 无需 finalize
            return self._make_generate_result(start_time, end_of_turn=True)

        # check if there are pending logits to process
        if not hasattr(self, "pending_logits") or self.pending_logits is None:
            self._pending_finalize = None  # 无需 finalize
            return self._make_generate_result(start_time)

        # use pending logits generated in streaming_prefill
        logits = self.pending_logits
        self.pending_logits = None

        # Force listen: initial N calls OR per-chunk force_listen_override from client
        force_listen = self._streaming_generate_count < self.force_listen_count or force_listen_override
        self._streaming_generate_count += 1
        if force_listen:
            _reason = "force_listen_override" if force_listen_override else f"call #{self._streaming_generate_count}"
            print(f"[Duplex] streaming_generate: force_listen=True ({_reason})")

        # Force Listen 前处理：如果模型正在说话，先补 <|turn_eos|> 关闭说话 turn
        # 这样 KV cache 序列合法：... <|turn_eos|> <|listen|> </unit>
        # 放在循环外，避免污染 res_ids / speak_count / total_hidden_in_unit
        if force_listen_override and not self.current_turn_ended:
            self.total_ids.append(self.turn_eos_token_id)
            logits, _ = self.decoder.feed(
                self.decoder.embed_token(self.turn_eos_token_id), return_logits=True
            )
            self.current_turn_ended = True
            self._reset_token2wav_for_new_turn()
            logger.info("[Duplex] force_listen: fed <|turn_eos|> to close speaking turn, reset TTS caches")

        total_hidden_in_unit = []
        total_ids_in_unit = []
        current_time = self.audio_chunk_idx
        is_listen = False
        end_of_turn = False

        # 如果上个 chunk 解码了 <|tts_pad|>，本 chunk 禁止再解码
        _tts_pad_suppressed = False
        if self._last_chunk_had_tts_pad and self.tts_pad_id not in self.decoder.forbidden_token_ids:
            self.decoder.forbidden_token_ids.append(self.tts_pad_id)
            _tts_pad_suppressed = True

        llm_start_time = time.time()
        _token_trace = []  # [DEBUG] 记录每个 token 的详细信息
        _pending_terminator_id = None  # 延迟 feed 的终止符，和 </unit> 合并
        _chunk_has_tts_pad = False

        for j in range(max_new_speak_tokens_per_chunk):
            if j == max_new_speak_tokens_per_chunk - 1:
                if self.ls_mode == "explicit":
                    # 不立即 feed，记录下来和 </unit> 合并
                    _pending_terminator_id = self.chunk_eos_token_id
                    self.total_ids.append(self.chunk_eos_token_id)
                    _tok_str = self.tokenizer.decode([self.chunk_eos_token_id])
                    _token_trace.append(f"  j={j} CHUNK_EOS id={self.chunk_eos_token_id} '{_tok_str}' (deferred)")
                    break

            t_step = time.time()
            if force_listen:
                last_id = torch.tensor([self.listen_token_id], dtype=torch.long, device=self.device)
                _decode_ms = 0.0
            else:
                last_id = self.decoder.decode(
                    logits=logits,
                    mode=decode_mode,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    listen_top_k=listen_top_k,
                    listen_prob_scale=listen_prob_scale,
                    text_repetition_penalty=text_repetition_penalty,
                    text_repetition_window_size=text_repetition_window_size,
                    length_penalty=length_penalty,
                )
                _decode_ms = (time.time() - t_step) * 1000

                # if current turn not ended, not allowed to listen (only check when not force_listen)
                if last_id.item() == self.listen_token_id and (not self.current_turn_ended):
                    last_id = torch.tensor([self.tts_bos_token_id], dtype=torch.long, device=self.device)

            self.total_ids.append(last_id.item())

            if last_id.item() == self.tts_pad_id:
                _chunk_has_tts_pad = True

            is_listen = last_id.item() == self.listen_token_id
            _tok_str = self.tokenizer.decode([last_id.item()])
            _is_special = last_id.item() in self.chunk_terminator_token_ids or last_id.item() in self.chunk_speak_token_ids

            # termination condition detection
            if last_id.item() in self.chunk_terminator_token_ids:
                # 不立即 feed 终止符，记录下来和 </unit> 合并（省一次 LLM forward）
                if self.ls_mode == "explicit":
                    _pending_terminator_id = last_id.item()
                _token_trace.append(f"  j={j} TERM id={last_id.item()} '{_tok_str}' decode={_decode_ms:.1f}ms (deferred)")
                break
            else:
                # normal speak
                self.current_turn_ended = False

                # 在 feed 之前检查字符长度，超限则不 feed、不记录，直接终止
                if j != 0:
                    _test_ids = total_ids_in_unit + [last_id.item()]
                    _chunk_text = self.tokenizer.decode(_test_ids, skip_special_tokens=True)
                    if len(_chunk_text) >= 28:
                        self.total_ids.pop()
                        if self.ls_mode == "explicit":
                            _pending_terminator_id = self.chunk_eos_token_id
                            self.total_ids.append(self.chunk_eos_token_id)
                        _kept_text = self.tokenizer.decode(total_ids_in_unit, skip_special_tokens=True) if total_ids_in_unit else ""
                        _token_trace.append(
                            f"  j={j} CHAR_LIMIT len={len(_chunk_text)}>=20, rejected token id={last_id.item()} '{_tok_str}', "
                            f"kept len={len(_kept_text)} text='{_kept_text}' (forced chunk_eos, not fed to KV)"
                        )
                        break

                if last_id.item() in self.chunk_speak_token_ids:
                    pass
                else:
                    self.res_ids.append(last_id.item())
                    self.speak_count += 1

                t_feed = time.time()
                logits, hidden = self.decoder.feed(self.decoder.embed_token(last_id.item()), return_logits=True)
                _feed_ms = (time.time() - t_feed) * 1000

                assert len(hidden.shape) == 3
                assert hidden.shape[0] == 1
                assert hidden.shape[1] == 1

                end_of_turn = last_id.item() in self.turn_terminator_token_ids

                if end_of_turn:
                    self.current_turn_ended = True

                _kind = "SPECIAL" if _is_special else "TEXT"
                _token_trace.append(f"  j={j} {_kind} id={last_id.item()} '{_tok_str}' decode={_decode_ms:.1f}ms feed={_feed_ms:.1f}ms")

                if j != 0:
                    total_hidden_in_unit.append([last_id.item(), hidden, end_of_turn])
                    total_ids_in_unit.append(last_id.item())

        # 恢复 forbidden list & 更新连续 tts_pad 状态
        if _tts_pad_suppressed:
            self.decoder.forbidden_token_ids.remove(self.tts_pad_id)
            assert self.tts_pad_id not in self.decoder.forbidden_token_ids
        self._last_chunk_had_tts_pad = _chunk_has_tts_pad

        # [DEBUG] 打印完整 token trace
        _trace_str = "\n".join(_token_trace)
        logger.info(
            f"[TokenTrace] t={current_time} is_listen={is_listen} "
            f"text_tokens={len(total_ids_in_unit)} total_steps={len(_token_trace)}"
            f" tts_pad={'suppressed' if _tts_pad_suppressed else 'allowed'}"
            f" had_tts_pad={_chunk_has_tts_pad}\n{_trace_str}"
        )

        # 计算生成的文本（用于滑窗 context 保留，过滤掉特殊 token）
        if os.environ.get("DEBUG_CHUNK_TEXT") == "1":
            generated_text = self.tokenizer.decode(total_ids_in_unit, skip_special_tokens=False) if total_ids_in_unit else ""
            generated_text += f"({len(total_ids_in_unit)},{len(generated_text)})|"
        else:
            generated_text = self.tokenizer.decode(total_ids_in_unit, skip_special_tokens=True) if total_ids_in_unit else ""

        # 存储 finalize 所需状态（延迟到 finalize_unit() 执行）
        input_type = self.current_mode.lower() if self.current_mode else "audio"
        self._pending_finalize = {
            "terminator_id": _pending_terminator_id,
            "total_ids_in_unit": total_ids_in_unit,
            "is_listen": is_listen,
            "generated_text": generated_text,
            "input_type": input_type,
        }

        llm_end_time = time.time()
        cost_llm = llm_end_time - llm_start_time

        if is_listen:
            self.total_hidden.append([])
            return self._make_generate_result(
                start_time, cost_llm=cost_llm,
                n_tokens=len(total_ids_in_unit),
            )

        # 如果 unit 中出现了 tts_pad_id，传空列表给 TTS
        tts_hidden_in_unit = [] if _chunk_has_tts_pad else total_hidden_in_unit
        tts_hidden_in_unit = total_hidden_in_unit

        self.total_hidden.append(total_hidden_in_unit)
        text = generated_text
        if _chunk_has_tts_pad:
            print(f"> speak (tts_pad): {text}, give an empty condition to duplex tts")
        else:
            print(f"> speak: {text}")

        if not self.generate_audio:
            return self._make_generate_result(
                start_time, is_listen=False, text=text,
                end_of_turn=end_of_turn, cost_llm=cost_llm,
                n_tokens=len(total_ids_in_unit),
            )

        # TTS generate
        tts_start_time = time.time()
        tts_prep_start_time = time.time()
        tts_condition = self._convert_results_to_tts_input(tts_hidden_in_unit)
        tts_prep_end_time = time.time()

        max_token_per_chunk = 25 + 1
        min_token_per_chunk = 25 + 1

        if end_of_turn:
            min_token_per_chunk = 0
        force_flush = True
        if self.tts_text_start_pos == 0:  # 这是turn的开始
            min_token_per_chunk = 0  # 可以允许解码<1s的音频
            # min_token_per_chunk = 10 + 1
            force_flush = True

        if self.tts_current_turn_start_time is None:
            self.tts_current_turn_start_time = current_time

        new_tokens, old_kv = self.model.tts.generate_chunk(
            inputs_embeds=tts_condition,
            temperature=self.tts_temperature,
            repetition_penalty=self.tts_repetition_penalty,
            eos_token=self.tts_eos_token,
            force_no_stop=False,
            max_new_token=max_token_per_chunk,
            min_new_tokens=min_token_per_chunk,
            past_key_values=self.tts_past_key_values,
            logits_processors=self.tts_logits_processors,
            text_start_pos=self.tts_text_start_pos,
        )

        tts_end_time = time.time()

        # 更新 TTS 状态（注意：token2wav 的重置必须在音频生成之后，否则会丢失 buffer 中的 tokens）
        if end_of_turn:
            self.tts_text_start_pos = 0
            self.tts_past_key_values = None
            self.tts_current_turn_start_time = None
            # 注意：_reset_token2wav_for_new_turn() 移到下面音频生成之后
        else:
            self.tts_past_key_values = old_kv
            self.tts_text_start_pos += tts_condition.shape[1] + new_tokens.shape[1]

        # Token2Wav 生成（必须在 reset 之前，否则 buffer 中倒数第二个 chunk 的 tokens 会丢失）
        token2wav_start_time = time.time()
        _buf_before = len(self.token2wav_buffer)
        audio_waveform = self._generate_waveform_from_tokens(
            new_tokens, prompt_wav_path, end_of_turn, force_flush=force_flush
        )
        _buf_after = len(self.token2wav_buffer)
        token2wav_end_time = time.time()

        # [DIAG] Token2Wav 诊断：buffer 状态 + 音频产出
        _wav_samples = len(audio_waveform) if audio_waveform is not None else 0
        _wav_dur_ms = (_wav_samples / 24000 * 1000) if _wav_samples > 0 else 0
        _wav_max = float(np.max(np.abs(audio_waveform))) if audio_waveform is not None and _wav_samples > 0 else 0.0
        logger.info(
            f"[Token2Wav] t={current_time} end_of_turn={end_of_turn} force_flush={force_flush} "
            f"tts_tokens={new_tokens.numel()} buf_before={_buf_before} buf_after={_buf_after} "
            f"wav={'None' if audio_waveform is None else f'{_wav_samples}samples/{_wav_dur_ms:.0f}ms'} "
            f"wav_max={_wav_max:.4f}"
        )

        # 在音频生成完成后再重置 token2wav 状态，确保 buffer 中的 tokens 都被处理
        if end_of_turn:
            self._reset_token2wav_for_new_turn()

        return self._make_generate_result(
            start_time, is_listen=False, text=text,
            audio_waveform=audio_waveform, end_of_turn=end_of_turn,
            cost_llm=cost_llm,
            cost_tts_prep=tts_prep_end_time - tts_prep_start_time,
            cost_tts=tts_end_time - tts_start_time,
            cost_token2wav=token2wav_end_time - token2wav_start_time,
            n_tokens=len(total_ids_in_unit),
            n_tts_tokens=new_tokens.numel(),
        )

    @torch.no_grad()
    def finalize_unit(self):
        """完成 streaming_generate 的延迟操作：feed 终止符 + </unit>，注册 unit 结束，执行滑窗。

        必须在 streaming_generate 之后、下一次 streaming_prefill 之前调用。
        设计为可异步调度：调用方可以先返回结果给前端，再在后台执行 finalize。
        """
        state = getattr(self, "_pending_finalize", None)
        if state is None:
            logger.warning("[Duplex] finalize_unit called but no pending finalize state")
            return

        t_start = time.time()

        # 1. 合并 feed：终止符（如有）+ </unit>
        unit_end_id = self.tokenizer.convert_tokens_to_ids("</unit>")
        terminator_id = state["terminator_id"]
        if terminator_id is not None:
            self.decoder.feed(self.decoder.embed_tokens([terminator_id, unit_end_id]))
        else:
            self.decoder.feed(self.decoder.embed_token(unit_end_id))
        self.total_ids.append(unit_end_id)

        # 2. 注册 unit 结束
        self.decoder.register_unit_end(
            input_type=state["input_type"],
            generated_tokens=state["total_ids_in_unit"],
            is_listen=state["is_listen"],
            generated_text=state["generated_text"],
        )

        # 3. 滑窗
        if self.decoder._window_config.sliding_window_mode == "context":
            self.decoder.enforce_window_with_context()
        elif self.decoder._window_config.sliding_window_mode == "basic":
            self.decoder.enforce_window()

        self._pending_finalize = None

        finalize_ms = (time.time() - t_start) * 1000
        logger.info(
            f"[Duplex] finalize_unit: {state['input_type']} is_listen={state['is_listen']} "
            f"finalize={finalize_ms:.0f}ms"
        )

    def get_session_schema(self, include_embeddings: bool = True) -> str:
        """获取当前 session 的完整 schema（包含 prefill 和 generate 阶段）

        Args:
            include_embeddings: 是否包含 embedding 占位符 (如 [img_embed_64], [audio_embed_50])

        Returns:
            完整的 schema 字符串，每个 unit 的格式为:
            <unit><image>[img_embed_64]</image>[audio_embed_50]<|listen|或|speak|>生成内容</unit>
        """
        if not hasattr(self, "prefill_schema_tokens") or not hasattr(self, "total_ids"):
            return ""

        # 获取 </unit> token id 用于分割 generate tokens
        unit_end_token_id = self.tokenizer.convert_tokens_to_ids("</unit>")

        # 分割 generate tokens 为每个 unit
        generate_units = []
        current_unit = []
        for tid in self.total_ids:
            current_unit.append(tid)
            if tid == unit_end_token_id:
                generate_units.append(current_unit)
                current_unit = []

        # 构建完整 schema
        full_schema_parts = []
        num_units = max(len(self.prefill_schema_tokens), len(generate_units))

        for unit_idx in range(num_units):
            unit_schema = ""

            # Prefill 部分
            if unit_idx < len(self.prefill_schema_tokens):
                prefill_tokens = self.prefill_schema_tokens[unit_idx]
                for item in prefill_tokens:
                    if isinstance(item, tuple):
                        # 元组表示 embedding: ("img", dim) 或 ("audio", dim)
                        embed_type, embed_dim = item
                        if include_embeddings:
                            unit_schema += f"[{embed_type}_embed_{embed_dim}]"
                    else:
                        # 正常 token ID
                        unit_schema += self.tokenizer.decode([item], skip_special_tokens=False)

            # Generate 部分
            if unit_idx < len(generate_units):
                unit_schema += self.tokenizer.decode(generate_units[unit_idx], skip_special_tokens=False)

            full_schema_parts.append(unit_schema)

        return "".join(full_schema_parts)

    def get_unit_schemas(self, include_embeddings: bool = True) -> list:
        """获取每个 unit 的 schema 列表

        Returns:
            每个 unit 的 schema 字符串列表
        """
        if not hasattr(self, "prefill_schema_tokens") or not hasattr(self, "total_ids"):
            return []

        unit_end_token_id = self.tokenizer.convert_tokens_to_ids("</unit>")

        # 分割 generate tokens 为每个 unit
        generate_units = []
        current_unit = []
        for tid in self.total_ids:
            current_unit.append(tid)
            if tid == unit_end_token_id:
                generate_units.append(current_unit)
                current_unit = []

        # 构建每个 unit 的 schema
        unit_schemas = []
        num_units = max(len(self.prefill_schema_tokens), len(generate_units))

        for unit_idx in range(num_units):
            unit_schema = ""

            # Prefill 部分
            if unit_idx < len(self.prefill_schema_tokens):
                prefill_tokens = self.prefill_schema_tokens[unit_idx]
                for item in prefill_tokens:
                    if isinstance(item, tuple):
                        # 元组表示 embedding: ("img", dim) 或 ("audio", dim)
                        embed_type, embed_dim = item
                        if include_embeddings:
                            unit_schema += f"[{embed_type}_embed_{embed_dim}]"
                    else:
                        # 正常 token ID
                        unit_schema += self.tokenizer.decode([item], skip_special_tokens=False)

            # Generate 部分
            if unit_idx < len(generate_units):
                unit_schema += self.tokenizer.decode(generate_units[unit_idx], skip_special_tokens=False)

            unit_schemas.append(unit_schema)

        return unit_schemas

    def _convert_results_to_tts_input(self, results):
        """convert LLM hidden states to TTS input"""
        if len(results) == 0:
            audio_bos = self.model.tts.emb_text(
                torch.tensor(
                    [self.model.tts.audio_bos_token_id],
                    device=self.model.tts.emb_text.weight.device,
                    dtype=torch.long,
                )
            )
            return audio_bos.unsqueeze(0)

        llm_tokens = []
        llm_hidden = []
        for hidden in results:
            llm_tokens.append(hidden[0])
            llm_hidden.append(hidden[1].squeeze(0))

        llm_tokens_tensor = torch.Tensor(llm_tokens).to(self.device, dtype=torch.long)
        llm_embeds = self.model.tts.emb_text(llm_tokens_tensor)

        llm_hidden_tensor = torch.cat(llm_hidden, dim=0)
        llm_hidden_tensor = self.model.tts.projector_semantic(llm_hidden_tensor)
        llm_hidden_tensor = torch.nn.functional.normalize(llm_hidden_tensor, p=2, dim=-1)

        tts_embeds = llm_embeds + llm_hidden_tensor

        audio_bos = self.model.tts.emb_text(
            torch.tensor(
                [self.model.tts.audio_bos_token_id],
                device=self.model.tts.emb_text.weight.device,
                dtype=torch.long,
            )
        )

        tts_embeds = torch.cat([tts_embeds, audio_bos], dim=0)
        return tts_embeds.unsqueeze(0)

    def _generate_waveform_from_tokens(
        self,
        new_tokens: torch.Tensor,
        prompt_wav_path: Optional[str],
        is_last_chunk: bool = False,
        force_flush: bool = False,
    ) -> Optional[np.ndarray]:
        """从 audio tokens 生成波形"""
        if not self.token2wav_initialized:
            print("⚠️ token2wav 未初始化")
            return None

        CHUNK_SIZE = 25

        # 将新 tokens 添加到 buffer
        token_ids = torch.reshape(new_tokens, (-1,)).tolist()
        self.token2wav_buffer += token_ids

        # 检测是否有 chunk_eos token
        has_chunk_eos = any(tid in self.chunk_terminator_token_ids for tid in token_ids)

        self._log(
            "AUDIO",
            f"Token2Wav buffer size: {len(self.token2wav_buffer)}, new tokens: {len(token_ids)}, has_chunk_eos: {has_chunk_eos}",
        )

        pcm_bytes_list = []

        # process enough tokens
        # if there is chunk_eos, try to flush more content
        if has_chunk_eos or force_flush:
            # when there is chunk_eos, try to flush more content
            while len(self.token2wav_buffer) >= self.pre_lookahead + 5:  # at least keep some lookahead
                chunk_to_process = min(CHUNK_SIZE + self.pre_lookahead, len(self.token2wav_buffer))
                pcm_bytes = self.model.tts.audio_tokenizer.stream(
                    self.token2wav_buffer[:chunk_to_process],
                    prompt_wav=prompt_wav_path,
                )
                pcm_bytes_list.append(pcm_bytes)
                self.token2wav_buffer = self.token2wav_buffer[min(CHUNK_SIZE, chunk_to_process - self.pre_lookahead) :]
        else:
            while len(self.token2wav_buffer) >= CHUNK_SIZE + self.pre_lookahead:
                pcm_bytes = self.model.tts.audio_tokenizer.stream(
                    self.token2wav_buffer[: CHUNK_SIZE + self.pre_lookahead],
                    prompt_wav=prompt_wav_path,
                )
                pcm_bytes_list.append(pcm_bytes)
                self.token2wav_buffer = self.token2wav_buffer[CHUNK_SIZE:]

        # if is the last chunk, flush remaining tokens
        if is_last_chunk and len(self.token2wav_buffer) > 0:
            self._log("AUDIO", f"Flushing final {len(self.token2wav_buffer)} tokens")
            pcm_bytes = self.model.tts.audio_tokenizer.stream(
                self.token2wav_buffer,
                prompt_wav=prompt_wav_path,
                last_chunk=True,
            )
            pcm_bytes_list.append(pcm_bytes)
            self.token2wav_buffer = []

        if not pcm_bytes_list:
            return None

        # merge PCM and convert to numpy array (24kHz, int16 -> float32)
        all_pcm = b"".join(pcm_bytes_list)
        if len(all_pcm) == 0:
            self._log("AUDIO", "No audio bytes generated")
            return None

        pcm_np = np.frombuffer(all_pcm, dtype="<i2")
        audio_waveform = pcm_np.astype(np.float32) / 32768.0

        # left pad with zeros if audio is less than 1 second (24kHz), skip for last chunk
        min_samples = 24000  # 1 second at 24kHz
        if not is_last_chunk and len(audio_waveform) < min_samples:
            pad_length = min_samples - len(audio_waveform)
            audio_waveform = np.pad(audio_waveform, (pad_length, 0), mode="constant", constant_values=0)
            self._log("AUDIO", f"Left padded {pad_length} samples ({pad_length/24000:.3f}s) to reach 1s minimum")

        # record generated audio information
        duration_sec = len(audio_waveform) / 24000
        self._log(
            "AUDIO",
            f"Generated audio: {duration_sec:.2f}s @ 24kHz, {len(pcm_bytes_list)} chunks, remaining buffer: {len(self.token2wav_buffer)} tokens",
        )

        return audio_waveform

    @staticmethod
    def _generate_silence_waveform(duration_sec: float = 1.0) -> np.ndarray:
        """generate silence waveform (24kHz)"""
        sample_rate = 24000
        num_samples = int(duration_sec * sample_rate)
        return np.zeros(num_samples, dtype=np.float32)

    def get_generated_text(self) -> str:
        return self.tokenizer.decode(self.res_ids)

    def get_current_time(self) -> int:
        return self.audio_chunk_idx

    def _log(self, event_type: str, message: str, data: Optional[dict] = None):
        if self.session_start_time is None:
            self.session_start_time = time.time()

        timestamp = time.time() - self.session_start_time
        log_entry = {"timestamp": timestamp, "type": event_type, "message": message}
        if data:
            log_entry["data"] = data

        self.session_logs.append(log_entry)

        prefix = f"[{timestamp:6.3f}s]"
        if event_type == "FEED":
            print(f"{prefix} [FEED] {message}")
        elif event_type == "DECODE":
            print(f"{prefix} [DECODE] {message}")
        elif event_type == "SYS":
            print(f"{prefix} [SYS] {message}")
        elif event_type == "AUDIO":
            print(f"{prefix} [AUDIO] {message}")
        else:
            print(f"{prefix} [{event_type}] {message}")
