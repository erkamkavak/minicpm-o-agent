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

import json
import logging
import math
import os
import tempfile
import types
from copy import deepcopy
from threading import Thread
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import Qwen3ForCausalLM
from transformers import Qwen3PreTrainedModel
from transformers import TextIteratorStreamer
from transformers.cache_utils import DynamicCache

from enum import Enum

# 相对导入（同目录）
from .configuration_minicpmo import MiniCPMOConfig
from .capabilities import DuplexCapability
from .services import DuplexProxyMixin
from .services import SpeculativeSnapshotMixin
from .services import UnifiedOperationsMixin
from .components import gen_logits
from .components import MiniCPMTTS
from .components import MiniCPMWhisperEncoder
from .components import MultiModalProjector
from .components import prepare_inputs_for_generation
from .components import Resampler
from .components.vision_encoder import SiglipVisionTransformer
from .processing_minicpmo import MiniCPMOProcessor
from .runtime.cache import as_dynamic_cache
from .runtime.cache import drop_tokens_from_cache
from .runtime.cache import get_kv_cache_length
from .runtime.cache import realign_rotary_suffix
from .runtime.cache import StreamingWindowConfig
from .runtime.tensor_ops import torch_clone_recursive
from .runtime.text_generation import ChunkPrefillChunkGenerate
from .runtime.text_generation import streaming_token_decoder
from .runtime.tts_streaming import TTSSamplingParams
from .runtime.tts_streaming import TTSStreamingGenerator

logger = logging.getLogger(__name__)


class ProcessorMode(Enum):
    """处理器模式枚举"""
    CHAT = "chat"           # 单工对话（非流式 TTS）
    STREAMING = "streaming" # 流式对话（流式 TTS）
    DUPLEX = "duplex"       # 双工对话（流式 TTS + 双工组件）


class MiniCPMOPreTrainedModel(Qwen3PreTrainedModel):
    config_class = MiniCPMOConfig


class MiniCPMO(
    UnifiedOperationsMixin,
    SpeculativeSnapshotMixin,
    DuplexProxyMixin,
    MiniCPMOPreTrainedModel,
):
    def __init__(self, config):
        super().__init__(config)

        self.llm = Qwen3ForCausalLM(config)
        self.embed_dim = self.llm.config.hidden_size
        self.llm.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, self.llm)  # patch llm

        # init vision module
        if self.config.init_vision:
            self.vpm = self.init_vision_module()
            self.vision_dim = self.vpm.embed_dim
            self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

        # init audio module
        if self.config.init_audio:
            self.apm = self.init_audio_module()
            audio_output_dim = int(self.apm.config.encoder_ffn_dim // 4)
            self.audio_avg_pooler = nn.AvgPool1d(self.config.audio_pool_step, stride=self.config.audio_pool_step)
            self.audio_projection_layer = MultiModalProjector(in_dim=audio_output_dim, out_dim=self.embed_dim)
            self.audio_encoder_layer = -1

        # init tts module
        if self.config.init_tts:
            self.tts = self.init_speech_decoder()

        self.terminators = ["<|im_end|>", "<|endoftext|>"]

        self.think_str = ""
        if self.llm.__class__.__name__ == "Qwen3ForCausalLM":
            self.think_str = "<think>\\n\\n</think>\\n\\n"

        self.default_tts_chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n"
            + self.think_str
            + "<|tts_bos|>' }}{% endif %}"
        )

        # for streaming
        self.reset_session(reset_token2wav_cache=True)

        # streaming audio processing constants
        self.SAMPLE_RATE = 16000
        self.CHUNK_MS = 1000  # regular chunk length (ms)
        self.FIRST_CHUNK_MS = 1035  # first chunk length (ms)
        self.CNN_REDUNDANCY_MS = 0  # CNN redundancy (ms)

        # for sliding window
        self.streaming_window_config = StreamingWindowConfig()
        self.streaming_require_system_prompt = True
        self.streaming_window_enabled = True
        self.force_rope_reindex = False  # RoPE reindex testing switch

        # ========== 统一模式支持（新增）==========
        self._current_mode: Optional[ProcessorMode] = None
        self._unified_initialized = False
        
        # 双 TTS 缓存（用于毫秒级切换）
        self._tts_streaming = None      # Token2wav (streaming TTS)
        self._tts_non_streaming = None  # CosyVoice2 (non-streaming TTS)
        
        # 双工能力组件（组合模式，通过 init_unified 初始化）
        self.duplex: Optional["DuplexCapability"] = None
        
        # 双工生成配置（传递给 DuplexCapability）
        self._duplex_config = {
            "generate_audio": True,
            "ls_mode": "explicit",
            "max_new_speak_tokens_per_chunk": 20,
            "text_repetition_penalty": 1.05,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
            "text_repetition_window_size": 512,
            "listen_prob_scale": 1.0,
            "force_listen_count": 0,
            "tts_temperature": 0.8,
            "tts_repetition_penalty": 1.05,
        }

    def _ensure_asset_dir(self, asset_subpath: str, model_dir: Optional[str] = None) -> str:
        """Ensure asset directory exists, downloading from HF if needed."""
        model_dir = model_dir or os.path.join(self.config._name_or_path, asset_subpath)
        if not os.path.exists(model_dir):
            from huggingface_hub import snapshot_download

            repo_dir = snapshot_download(
                repo_id="openbmb/MiniCPM-o-4_5",
                allow_patterns=[f"{asset_subpath}/**"],
            )
            model_dir = os.path.join(repo_dir, asset_subpath)
        assert os.path.exists(model_dir), f"Asset directory not found: {model_dir}"
        return model_dir

    def init_streaming_processor(self):
        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        if hasattr(self.processor, "set_streaming_mode"):
            self.processor.set_streaming_mode(
                mode="exact",
                chunk_ms=self.CHUNK_MS,
                first_chunk_ms=self.FIRST_CHUNK_MS,
                cnn_redundancy_ms=self.CNN_REDUNDANCY_MS,
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )
            self.processor.reset_streaming()
            self.audio_chunk_idx = 0

    def reset_session(self, reset_token2wav_cache=True):
        self.llm_past_key_values = None
        self.audio_past_key_values = None
        self.tts_last_turn_tokens = None
        self.llm_generated = False  # last turn generated by llm or not
        self.llm_generate_completed = False
        self.new_user_msg = True

        self.session_id = None

        if reset_token2wav_cache:
            self.token2wav_cache = None

        # for sliding window
        self.streaming_text_preserve = 0
        self.streaming_position_offset = 0

        self._rope_inv_freq_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

        self._next_round_id = 0
        self._pending_round_id = None

        self._omni_chunk_history: List[Dict[str, Union[str, int]]] = []
        self._round_history: List[Dict[str, Union[int, str, torch.Tensor, Optional[int]]]] = []

    def init_vision_module(self):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.vision_config._attn_implementation = "flash_attention_2"
        else:
            self.config.vision_config._attn_implementation = "eager"
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True,
        )

    def init_audio_module(self):
        if self.config._attn_implementation == "eager":
            self.config.audio_config._attn_implementation = "eager"
        else:
            # using flash_attention_2 will cause: RuntimeError: cu_seqlens_q must have shape (batch_size + 1)
            self.config.audio_config._attn_implementation = "sdpa"

        return MiniCPMWhisperEncoder(self.config.audio_config)

    def init_speech_decoder(self):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.tts_config.attn_implementation = "flash_attention_2"
        else:
            self.config.tts_config.attn_implementation = "eager"

        return MiniCPMTTS(config=self.config.tts_config, audio_tokenizer=None)

    def init_token2wav(self, streaming=False, model_dir=None, enable_float16=False, n_timesteps=5):
        if streaming:
            if self.config.tts_config.audio_tokenizer_type != "s3tokenizer_step_audio":
                logger.warning("audio tokenizer type is set to s3tokenizer_step_audio")
                self.tts.config.audio_tokenizer_type = "s3tokenizer_step_audio"

            try:
                from stepaudio2 import Token2wav
            except ImportError:
                raise ImportError(f"please install Token2wav via: pip install stepaudio2-minicpmo")

            model_dir = self._ensure_asset_dir("assets/token2wav", model_dir)
            logger.info(f"Token2wav model_dir: {model_dir}, enable_float16: {enable_float16}, n_timesteps: {n_timesteps}")
            self.tts.audio_tokenizer = Token2wav(model_dir, float16=enable_float16, n_timesteps=n_timesteps)
            return self.tts.audio_tokenizer
        else:
            if self.config.tts_config.audio_tokenizer_type != "s3tokenizer":
                logger.warning("audio tokenizer type is set to s3tokenizer")
                self.tts.config.audio_tokenizer_type = "s3tokenizer"

            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2
            except ImportError:
                raise ImportError(f"please install cosyvoice via: pip install cosyvoice-minicpmo")

            model_dir = self._ensure_asset_dir("assets/CosyVoice2-0.5B", model_dir)
            self.tts.audio_tokenizer = CosyVoice2(model_dir=model_dir, load_jit=False, load_trt=False, fp16=False)
            return self.tts.audio_tokenizer

    # ==================== 统一模式方法（新增）====================
    
    def init_unified(
        self,
        pt_path: Optional[str] = None,
        preload_both_tts: bool = True,
        duplex_config: Optional[dict] = None,
        device: str = "cuda",
        chat_vocoder: str = "token2wav",
    ):
        """Unified initialization — load once, hot-switch between three modes.

        Args:
            pt_path: Optional extra .pt weights to override base model weights.
                Typical usage: load base model via from_pretrained, then overlay
                fine-tuned weights via pt_path.
            preload_both_tts: Whether to preload both TTS vocoders (recommended
                True — trades ~0.5 GB VRAM for millisecond-level switching).
            duplex_config: Duplex configuration dict.
            device: Target device.
            chat_vocoder: Vocoder for Chat (non-streaming) mode.
                "token2wav" = Step Audio Token2Wav (default, lightweight);
                "cosyvoice2" = CosyVoice2-0.5B (requires extra dependencies).
                When set to "token2wav", CosyVoice2 is not loaded, saving
                ~0.5 GB VRAM and its dependencies.

        After this call the following mode switches are available:
        - set_mode(ProcessorMode.CHAT)
        - set_mode(ProcessorMode.STREAMING)
        - set_mode(ProcessorMode.DUPLEX)
        """
        logger.info("Initializing unified mode...")
        self._chat_vocoder = chat_vocoder
        logger.info(f"Chat vocoder config: {chat_vocoder}")

        # Load extra .pt weights to override base model (if provided)
        if pt_path is not None:
            logger.info(f"Loading extra weights: {pt_path}")
            state_dict = torch.load(pt_path, map_location="cpu")
            info = self.load_state_dict(state_dict, strict=False)
            logger.info(f"Weights loaded — missing: {len(info.missing_keys)}, unexpected: {len(info.unexpected_keys)}")
            if info.unexpected_keys:
                logger.warning(f"Unexpected keys: {info.unexpected_keys[:5]}...")
            del state_dict

        # Update duplex config
        if duplex_config:
            self._duplex_config.update(duplex_config)

        # Preload TTS vocoder
        self.init_token2wav(streaming=True)

        # Create DuplexCapability instance (composition pattern)
        self.duplex = DuplexCapability(
            model=self,
            device=device,
            **self._duplex_config,
        )

        self._unified_initialized = True

        # Default to Streaming mode
        self.set_mode(ProcessorMode.STREAMING)

        logger.info("Unified mode initialization complete")

    def set_mode(self, mode: ProcessorMode) -> None:
        """Set the current processor mode (millisecond-level switch).

        Args:
            mode: Target mode (CHAT / STREAMING / DUPLEX).
        """
        if mode == self._current_mode:
            return

        logger.info(f"Switching mode: {self._current_mode} -> {mode}")

        # Reset session state
        self.reset_session(reset_token2wav_cache=True)

        # Extra reset for duplex mode
        if mode == ProcessorMode.DUPLEX and hasattr(self, 'duplex') and self.duplex is not None:
            self.duplex._reset_streaming_state()
            self.duplex.decoder.reset()

        self._current_mode = mode

    # Operational helpers are provided by UnifiedOperationsMixin.

    @property
    def current_mode(self) -> Optional[ProcessorMode]:
        """当前模式"""
        return self._current_mode
    
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    @staticmethod
    def get_sys_prompt(ref_audio=None, mode="default", language="en", ref_audio_max_ms=None):
        if ref_audio is not None:
            if isinstance(ref_audio, str):
                if ref_audio == "assets/demo.wav":
                    import librosa

                    duration = ref_audio_max_ms / 1000.0 if ref_audio_max_ms else None
                    ref_audio, _ = librosa.load(ref_audio, sr=16000, mono=True, duration=duration)
                else:
                    import os

                    import librosa

                    if os.path.isfile(ref_audio) and os.path.exists(ref_audio):
                        duration = ref_audio_max_ms / 1000.0 if ref_audio_max_ms else None
                        ref_audio, _ = librosa.load(ref_audio, sr=16000, mono=True, duration=duration)
                    else:
                        logger.error(f"Could not find {ref_audio}")
                        ref_audio = None

            assert isinstance(ref_audio, np.ndarray), "ref_audio error"

        if mode == "omni":
            if language == "zh":
                sys_prompt = ""
                vc_prompt_prefix = "模仿音频样本的音色并生成新的内容。"
                vc_prompt_suffix = (
                    "请用这种声音风格来为用户提供帮助。 请认真、高质量地回复用户的问题。 请用高自然度的方式和用户聊天。"
                )
            else:
                sys_prompt = ""
                vc_prompt_prefix = sys_prompt + "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "As an assistant, you will speak using this voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                sys_msgs = {"role": "system", "content": [sys_prompt]}

            return sys_msgs
        elif mode == "audio_assistant":
            if language == "zh":
                vc_prompt_prefix = "模仿音频样本的音色并生成新的内容。"
                vc_prompt_suffix = "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
            else:
                vc_prompt_prefix = "Use the voice in the audio prompt to synthesize new content."
                vc_prompt_suffix = "You are a helpful assistant with the above voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

            else:
                logger.warning(
                    "Warning: ref_audio is None, speech generation will be performed based on the default voice."
                )
                sys_msgs = {"role": "system", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "audio_roleplay":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "假装你是上述音频中的人物，与我进行对话。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "Try to role-play the character based on the audio prompt above."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                print("Warning: ref_audio is None, speech generation will be performed based on the default voice.")
                sys_msgs = {"role": "system", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "voice_cloning":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio]}
            else:
                raise ValueError("ref_audio con't be None in voice_cloning mode.")

            return sys_msgs
        else:
            sys_prompt = "You are a helpful assistant. You can accept audio and text input and output voice and text."
            sys_msgs = {"role": "system", "content": [sys_prompt]}

            return sys_msgs

    @staticmethod
    def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        """Create mask for subsequent steps (size, size) with chunk size,
        this is for streaming encoder

        Args:
            size (int): size of mask
            chunk_size (int): size of chunk
            num_left_chunks (int): number of left chunks
                <0: use full chunk
                >=0: use num_left_chunks
            device (torch.device): "cpu" or "cuda" or torch.Tensor.device
            num_lookhead:

        Returns:
            torch.Tensor: mask
        """
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        for i in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead, size)
            ret[i, start:ending] = True
        return ret

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Computes the output length of the convolutional layers and the output length of the audio encoder"""
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_vision_embedding(self, data):
        if "vision_hidden_states" not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx],
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values,
                        patch_attention_mask=patch_attn_mask,
                        tgt_sizes=tgt_sizes,
                    ).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                    tgt_sizes = torch.Tensor(
                        [
                            [
                                (224 // self.config.patch_size),
                                math.ceil(224 / self.config.patch_size),
                            ]
                        ]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)
        else:
            vision_hidden_states = data["vision_hidden_states"]

        return vision_hidden_states

    def get_vllm_embedding(self, data):
        vision_hidden_states = self.get_vision_embedding(data)

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def get_audio_embedding_streaming(
        self,
        data,
        use_extra_context=False,
        prefix_extra_frames=1,
        suffix_extra_frames=1,
        return_debug=False,
        cnn_min_length=None,
    ):
        """Extract audio embeddings in a streaming manner using cached key-value pairs.

        This method processes incoming audio features incrementally and stores/updates `past_key_values`
        for faster inference on subsequent audio frames. It only supports batch_size=1 and is intended
        for streaming scenarios.

        Args:
            data (dict):
                - **"audio_features"** (`torch.FloatTensor`): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
                - **"audio_feature_lens"** (List[List[int]]): Lengths of each audio segment for each item in the batch.
            use_extra_context (bool): If True, assumes input contains extra frames for CNN context.
            prefix_extra_frames (int): Number of prefix extra frames.
            suffix_extra_frames (int): Number of suffix extra frames.
            return_debug (bool): Whether to return debug information.
            cnn_min_length (int): Minimum length for CNN input padding.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

        # exist audio
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            assert batch_size == 1
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # whisper's past_key_values management (core)
            if self.audio_past_key_values is not None:
                cache_length = self.audio_past_key_values[0][0].shape[2]
                apm_max_len = self.apm.embed_positions.weight.shape[0]
                if cache_length + max_seq_len >= apm_max_len:
                    logger.warning(
                        f"audio_past_key_values length {cache_length + max_seq_len} exceed {apm_max_len}, reset."
                    )
                    self.audio_past_key_values = None

            # build attention mask (bidirectional attention, same as offline mode)
            batch_size, _, max_mel_seq_len = wavforms.shape
            current_seq_len = (max_mel_seq_len - 1) // 2 + 1
            # if use extra context, need to adjust sequence length
            if use_extra_context:
                # calculate actual sequence length after removing redundancy
                # conv2's stride=2, so the mapping from mel frames to output frames is ceil(x/2)
                prefix_to_remove = (prefix_extra_frames + 1) // 2 if prefix_extra_frames > 0 else 0
                suffix_to_remove = (suffix_extra_frames + 1) // 2 if suffix_extra_frames > 0 else 0
                current_seq_len = current_seq_len - prefix_to_remove - suffix_to_remove
            # calculate history length (if there is KV cache)
            if self.audio_past_key_values is not None:
                past_len = self.audio_past_key_values[0][0].shape[2]  # get history sequence length
                total_seq_len = past_len + current_seq_len
            else:
                past_len = 0
                total_seq_len = current_seq_len
            # create bidirectional attention mask (full attention)
            audio_attention_mask = torch.zeros(
                (batch_size, 1, current_seq_len, total_seq_len),
                dtype=self.apm.conv1.weight.dtype,
                device=wavforms.device,
            )

            debug_info = {} if return_debug else None

            # DEBUG: 打印输入和 past_key_values 状态
            if hasattr(self, "_debug_prefill") and self._debug_prefill:
                wavform_sum = wavforms.sum().item()
                past_kv_info = (
                    "None" if self.audio_past_key_values is None else f"len={self.audio_past_key_values[0][0].shape[2]}"
                )
                print(
                    f"[DEBUG audio_embed] wavforms sum={wavform_sum:.6f}, shape={wavforms.shape}, past_kv={past_kv_info}"
                )

            # Step 1: APM processing
            ret_apm = self.apm(
                wavforms,
                past_key_values=self.audio_past_key_values,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=audio_attention_mask,
                use_extra_context=use_extra_context,
                prefix_extra_frames=prefix_extra_frames,
                suffix_extra_frames=suffix_extra_frames,
                return_debug=return_debug,
                cnn_min_length=cnn_min_length,
            )
            if return_debug:
                audio_outputs, debug_info["apm_debug"] = ret_apm
            else:
                audio_outputs = ret_apm

            if hasattr(self, "audio_encoder_layer"):
                audio_states = audio_outputs.hidden_states[self.audio_encoder_layer]
            else:
                audio_states = audio_outputs.last_hidden_state

            # DEBUG: 打印 apm 输出的 checksum
            if hasattr(self, "_debug_prefill") and self._debug_prefill:
                apm_out_sum = audio_outputs.last_hidden_state.sum().item()
                audio_states_sum = audio_states.sum().item()
                print(f"[DEBUG audio_embed] apm_output sum={apm_out_sum:.6f}, audio_states sum={audio_states_sum:.6f}")

            if return_debug:
                debug_info["apm_output"] = audio_outputs.last_hidden_state.clone()
                debug_info["audio_states_before_proj"] = audio_states.clone()
                debug_info["attention_mask"] = audio_attention_mask.clone()
                debug_info["use_extra_context"] = use_extra_context
                debug_info["prefix_extra_frames"] = prefix_extra_frames
                debug_info["suffix_extra_frames"] = suffix_extra_frames

            self.audio_past_key_values = audio_outputs.past_key_values

            # Step 2: Projection
            audio_embeds = self.audio_projection_layer(audio_states)

            # DEBUG: 打印 projection 和 pooling 后的 checksum
            if hasattr(self, "_debug_prefill") and self._debug_prefill:
                proj_sum = audio_embeds.sum().item()
                print(f"[DEBUG audio_embed] after_projection sum={proj_sum:.6f}, shape={audio_embeds.shape}")

            if return_debug:
                debug_info["after_projection"] = audio_embeds.clone()

            # Step 3: Pooling
            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            # DEBUG: 打印 pooling 后的 checksum
            if hasattr(self, "_debug_prefill") and self._debug_prefill:
                pool_sum = audio_embeds.sum().item()
                print(f"[DEBUG audio_embed] after_pooling sum={pool_sum:.6f}, shape={audio_embeds.shape}")

            if return_debug:
                debug_info["after_pooling"] = audio_embeds.clone()

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)

            if return_debug:
                debug_info["final_embeddings"] = final_audio_embeds
                return final_audio_embeds, debug_info
            else:
                return final_audio_embeds
        else:
            return [] if not return_debug else ([], {})

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        dtype = self.apm.embed_positions.weight.dtype
        device = self.apm.embed_positions.weight.device

        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(
                    0,
                    max_seq_len,
                    dtype=audio_feature_lens.dtype,
                    device=audio_feature_lens.device,
                )
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand  # 1 for padded values

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.apm.conv1.weight.dtype, device=self.apm.conv1.weight.device
            )

            if chunk_length > 0:
                chunk_num_frame = int(chunk_length * 50)
                chunk_mask = self.subsequent_chunk_mask(
                    size=max_seq_len,
                    chunk_size=chunk_num_frame,
                    num_left_chunks=-1,
                    device=audio_attention_mask_.device,
                )
                audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))

            audio_attention_mask[audio_attention_mask_] = float("-inf")
            audio_states = self.apm(
                wavforms, output_hidden_states=True, attention_mask=audio_attention_mask
            ).hidden_states[self.audio_encoder_layer]
            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        elif self.training and dummy:
            dummy_wavs = torch.zeros((1, 80, 100), device=device, dtype=dtype)
            audio_states = self.apm(dummy_wavs, output_hidden_states=True).hidden_states[self.audio_encoder_layer]

            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)
            return [audio_embeds]
        else:
            return []

    def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        """
        Args:
            data:
            input_embeddings:
            chunk_length: whisper use full attention or chunk attention
            stream_input: use streaming audio embedding or not

        Returns:
            final embeddings with audio feature
        """
        if stream_input:
            audio_embeddings = self.get_audio_embedding_streaming(data)
        else:
            audio_embeddings = self.get_audio_embedding(data, chunk_length)

        bs = len(input_embeddings)
        if len(data.get("audio_features", [])) > 0:
            assert len(audio_embeddings) == len(input_embeddings)

            if len(audio_embeddings) > 0:
                audio_bounds = data["audio_bounds"]

                if self.config.stream_input:
                    assert bs == 1, "audio stream_input mode only support batch size 1"
                    for i in range(bs):
                        audio_embs = torch.cat(audio_embeddings[i], dim=0).to(
                            device=input_embeddings.device, dtype=input_embeddings.dtype
                        )
                        audio_start_pos = 0
                        for bound in audio_bounds[i]:
                            audio_len = bound[1] - bound[0]
                            input_embeddings[i, bound[0] : bound[1]] = audio_embs[
                                audio_start_pos : audio_start_pos + audio_len, :
                            ]
                            audio_start_pos += audio_len
                else:
                    for i in range(bs):
                        audio_embs = audio_embeddings[i]
                        bounds = audio_bounds[i]
                        for embs, bound in zip(audio_embs, bounds):
                            audio_indices = torch.arange(bound[0], bound[1], dtype=torch.long).to(
                                input_embeddings.device
                            )

                            if embs.shape[0] != len(audio_indices):
                                logger.error(f"Sample {i}:")
                                logger.error(f"  Bounds: {bound}, Indices Length: {len(audio_indices)}")
                                logger.error(f"  Embeddings Shape: {embs.shape}")
                                logger.error(
                                    f"  Input Embedding Shape at Indices: {input_embeddings[i, audio_indices].shape}"
                                )
                                raise ValueError(
                                    f"Shape mismatch: Trying to assign embeddings of shape {embs.shape} "
                                    f"to input indices of length {len(audio_indices)}"
                                )
                            input_embeddings[i, audio_indices] = embs.to(input_embeddings.dtype)
        elif self.training:
            for i in range(bs):
                # dummy audio_embedings
                input_embeddings += audio_embeddings[0].mean() * 0

        return input_embeddings

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        vllm_embedding = self.get_omni_embedding(
            data,
            input_embeddings=vllm_embedding,
            chunk_length=self.config.audio_chunk_length,
        )

        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs,
        )

    def _decode(self, inputs_embeds, tokenizer, attention_mask, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        return outputs

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_config = {
            "inputs_embeds": inputs_embeds,
            "pad_token_id": 0,
            "eos_token_id": terminators,
            "streamer": streamer,
        }
        generation_config.update(kwargs)
        thread = Thread(target=self.llm.generate, kwargs=generation_config)
        thread.start()
        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result))
        return result_text

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        audio_features=None,
        audio_feature_lens=None,
        image_bound=None,
        audio_bounds=None,
        spk_bounds=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        stream=False,
        **kwargs,
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "image_bound": image_bound,
            "audio_bounds": audio_bounds,
            "spk_bounds": spk_bounds,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["tgt_sizes"] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
            model_inputs["inputs_embeds"] = self.get_omni_embedding(
                model_inputs,
                input_embeddings=model_inputs["inputs_embeds"],
                chunk_length=self.config.audio_chunk_length,
            )

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
                outputs = {}  # if stream return TextIteratorStreamer and output is empty
            else:
                outputs = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, **kwargs)
                result = self._decode_text(outputs.sequences, tokenizer)

        return result, outputs

    def _build_streaming_mask(self, tts_tokens_len):
        tts_sequence_full_length = 1 + self.tts.streaming_text_reserved_len + 1
        streaming_attention_mask = torch.zeros(tts_sequence_full_length, dtype=torch.int8)
        streaming_attention_mask[0 : 1 + 1 + tts_tokens_len + 1] = 1
        streaming_attention_mask[-1] = 1
        return streaming_attention_mask

    def _generate_mel_spec(self, inputs, outputs, text, output_chunk_size=25, tts_max_new_tokens=2048):
        spk_embeds = self._get_last_spk_embeds(inputs, outputs)

        text = text.split("<|tts_bos|>")[-1]
        gen_text = text.split("<|tts_eos|>")[0]
        tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
        tts_inputs = self.tts_processor.text_tokenizer.encode(tts_text, add_special_tokens=False)
        tts_input_ids = torch.Tensor(tts_inputs).unsqueeze(0).to(self.device, dtype=torch.long)
        streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=626,
            top_p=self.tts.top_p,
            top_k=self.tts.top_k,
            repetition_penalty=self.tts.repetition_penalty,
        )

        condition_length = 1 + self.tts.streaming_text_reserved_len + 1

        dtype = self.tts.emb_text.weight.dtype
        emb = torch.zeros(1, condition_length, self.tts.num_vq, dtype=dtype, device=self.tts.device)
        past_key_values = [
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]

        audio_input_ids = torch.zeros(
            1,
            condition_length,
            self.tts.num_vq,
            dtype=torch.long,
            device=self.tts.device,
        )

        eos_lab = False
        for chunk_idx in range(math.ceil(emb.shape[1] / self.tts.streaming_text_chunk_size)):
            if chunk_idx == 0:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 0
                end = (chunk_idx + 1) * self.tts.streaming_text_chunk_size + 1
            else:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 1
                end = min(
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size + 1,
                    condition_length - 1,
                )

            if end - begin > 0:
                text_input_ids = tts_input_ids[:, begin:end]
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                if begin == 0:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        lm_spk_emb_last_hidden_states=spk_embeds,
                    )
                else:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                    )

            outputs = self.tts.generate(
                input_ids=audio_input_ids,
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_tts_text_mask,
                max_new_token=output_chunk_size,
                force_no_stop=self.force_no_stop,
                temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
            )
            audio_input_ids = outputs.audio_input_ids
            past_key_values = outputs.past_key_values

            if outputs.finished:
                logger.debug("Generation finished.")
                eos_lab = True
                break

        if not eos_lab:
            logger.debug("eos_lab False, Generation continue.")
            while True:
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=self.force_no_stop,
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )

                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values

                if outputs.finished:
                    logger.debug("Generation finished.")
                    break
                if outputs.new_ids.shape[1] > tts_max_new_tokens:
                    logger.debug(f"Generation length > {tts_max_new_tokens}, stopped.")
                    break

    @staticmethod
    def prepare_generation_config(do_sample, max_new_tokens=50, min_new_tokens=0, **kwargs):
        num_beams = kwargs.get("num_beams", 3)
        generation_config = {
            "num_beams": num_beams,
            "top_p": 0.8,
            "top_k": 100,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.02,
        }

        if do_sample:
            generation_config.update(
                {
                    "top_p": 0.8,
                    "top_k": 100,
                    "temperature": 0.7,
                    "do_sample": True,
                    "repetition_penalty": 1.02,
                }
            )
        elif num_beams > 1:
            generation_config.update({"num_beams": num_beams, "repetition_penalty": 1.2, "do_sample": False})
        else:
            generation_config.update({"do_sample": False, "repetition_penalty": 1.02})

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())
        generation_config["min_new_tokens"] = min_new_tokens
        generation_config["max_new_tokens"] = max_new_tokens

        return generation_config

    @torch.inference_mode()
    def chat(
        self,
        image=None,
        msgs=None,
        tokenizer=None,  # deprecated
        processor=None,  # deprecated
        vision_hidden_states=None,
        max_new_tokens=4096,
        min_new_tokens=0,
        do_sample=True,
        sampling=None,  # deprecated, please use do_sample
        max_inp_length=8192,
        stream=False,
        stream_input=False,
        max_slice_nums=None,
        use_image_id=None,
        enable_thinking=False,
        use_tts_template=False,
        generate_audio=False,
        output_audio_path=None,
        output_tts_inputs_embeds_path=None,
        # add
        omni_mode=False,
        omni_input=None,  # deprecated, please use omni_mode
        teacher_forcing=False,
        return_prompt=False,
        tts_proj_layer=-1,
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
        merge_audio_from_same_content=True,
        tts_ref_audio: Optional[np.ndarray] = None,
        **kwargs,
    ):
        # todo: deprecated
        if sampling is not None:
            do_sample = sampling
        if omni_input is not None:
            omni_mode = omni_input

        batched = isinstance(msgs[0], list)
        msgs_list = msgs
        images_list = image

        if not batched:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []

        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert do_sample or not stream, "if use stream mode, make sure do_sample=True"

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            audios = []
            audio_parts = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["system", "user", "assistant"]
                if i == 0:
                    assert role in ["user", "system"], "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("<image>./</image>")
                    elif isinstance(c, np.ndarray):  # audio
                        audios.append(c)
                        audio_parts.append(i)
                        cur_msgs.append("<audio>./</audio>")
                        use_tts_template = True
                    elif isinstance(c, str):
                        cur_msgs.append(c)

                if omni_mode or stream_input:
                    msg["content"] = "".join(cur_msgs)
                else:
                    msg["content"] = "\n".join(cur_msgs)

            prompts_lists.append(
                self.processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=False if teacher_forcing else True,
                    use_tts_template=use_tts_template,
                    enable_thinking=enable_thinking,
                )
            )
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        if not merge_audio_from_same_content:
            audio_parts_list = None

        inputs = self.processor(
            prompts_lists,
            input_images_list,
            input_audios_list,
            audio_parts_list,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            stream_input=stream_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)

        generation_config = self.prepare_generation_config(
            do_sample=do_sample, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, **kwargs
        )
        generation_config.pop("max_new_tokens", None)

        inputs.pop("image_sizes")

        # teacher_forcing = True => generate audio with given text
        with torch.inference_mode():
            res, outputs = self.generate(
                **inputs,
                tokenizer=self.processor.tokenizer,
                max_new_tokens=1 if teacher_forcing else max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                **generation_config,
            )

        # spk bound and tts bound
        tts_bos_token = self.processor.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
        tts_eos_token = self.processor.tokenizer.convert_tokens_to_ids("<|tts_eos|>")

        # Combine input_ids and generated sequences to get complete sequence
        input_ids = inputs["input_ids"][0]
        generated_ids = outputs.sequences[0]
        # Combine by concatenating input_ids with the new tokens from generated sequence
        full_sequence = torch.cat([input_ids, generated_ids])
        # Update the sequences in outputs
        full_sequences = full_sequence.unsqueeze(0)

        outputs["full_sequences"] = full_sequences

        # 存储 token 统计，供 ChatView 等外部消费者读取
        # input_tokens: tokenizer 级别（含 audio/image 占位符，不含 embedding 展开）
        # generated_tokens: LLM 实际生成的 token 数
        self._last_chat_token_stats = {
            "input_tokens": len(input_ids),
            "generated_tokens": len(generated_ids),
        }

        tts_bos_indices = []
        tts_eos_indices = []
        for i, x in enumerate(full_sequences[0]):
            if x == tts_bos_token:
                tts_bos_indices.append(i + 1)  # tts_bos + 1 才是第一个tts的位置，这样方便直接给tts去slice hidden states
            elif x == tts_eos_token:
                if teacher_forcing and i == len(full_sequences[0]) - 1:
                    continue
                tts_eos_indices.append(i)

        tts_bos_idx = tts_bos_indices[-1] if tts_bos_indices else -1
        # Use None instead of -1 when no EOS token found, so that slice [start:None]
        # means "to the end" rather than [start:-1] which excludes the last element
        tts_eos_idx = tts_eos_indices[-1] if tts_eos_indices else None

        tts_bound = (tts_bos_idx, tts_eos_idx)

        answer = res[0]
        if answer is not None:
            answer = answer.rstrip("<|tts_eos|>")

        generated_waveform = None
        if use_tts_template and generate_audio:
            try:
                # TTS ref audio 优先级：
                # 1. tts_ref_audio（显式指定的 TTS 参考音频，与 LLM ref audio 分离）
                # 2. input_audios_list[0][0]（messages 中第一个音频，即 LLM ref audio）
                # 3. None（无参考音频）
                if tts_ref_audio is not None:
                    _tts_audio_prompt = tts_ref_audio
                    logger.info(f"[Chat TTS] Using separate tts_ref_audio: {len(tts_ref_audio)} samples ({len(tts_ref_audio)/16000:.1f}s)")
                elif len(input_audios_list) > 0 and len(input_audios_list[0]) > 0:
                    _tts_audio_prompt = input_audios_list[0][0]
                    logger.info(f"[Chat TTS] Using LLM ref audio from messages: {len(_tts_audio_prompt)} samples ({len(_tts_audio_prompt)/16000:.1f}s)")
                else:
                    _tts_audio_prompt = None
                    logger.warning("[Chat TTS] No ref audio available for TTS")

                generated_waveform = self._generate_speech_non_streaming(
                    outputs=outputs,
                    tts_bound=tts_bound,
                    tts_proj_layer=tts_proj_layer,
                    audio_prompt=_tts_audio_prompt,
                    output_tts_inputs_embeds_path=output_tts_inputs_embeds_path,
                    tts_sampling_params=tts_sampling_params,
                )
                # 统一为 numpy array
                if isinstance(generated_waveform, torch.Tensor):
                    generated_waveform = generated_waveform.cpu().numpy()

                # 如果指定了保存路径，也保存到文件
                if output_audio_path and generated_waveform is not None:
                    import soundfile as sf
                    sf.write(output_audio_path, generated_waveform, samplerate=24000)
                    logger.debug(f"audio saved to {output_audio_path}")
            except:
                import traceback
                traceback.print_exc()
                generated_waveform = None

        if return_prompt:
            return answer, prompts_lists[0], generated_waveform
        elif generated_waveform is not None:
            return answer, generated_waveform
        else:
            return answer

    @torch.inference_mode()
    def _generate_speech_non_streaming(
        self,
        outputs,
        tts_bound,
        tts_proj_layer,
        audio_prompt,
        output_tts_inputs_embeds_path=None,
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
    ):
        last_hidden_states = [hs[tts_proj_layer] for hs in outputs.hidden_states]
        last_hidden_states = torch.vstack([i[0] for i in last_hidden_states])
        
        # FIX: 某些 pt 权重可能导致 hidden_states 和 sequences 长度不一致
        # 以 full_sequences 为准（这是实际要用的 tokens）
        full_seq_len = len(outputs["full_sequences"][0])
        if last_hidden_states.shape[0] != full_seq_len:
            logger.warning(f"TTS: hidden_states({last_hidden_states.shape[0]}) != full_sequences({full_seq_len}), truncating")
            last_hidden_states = last_hidden_states[:full_seq_len]

        spk_embeds = (
            torch.ones([0, self.tts.config.hidden_size]).to(last_hidden_states.device).to(last_hidden_states.dtype)
        )

        if self.tts.condition_type == "hidden_text_merge":
            llm_tokens = outputs["full_sequences"][0][tts_bound[0] : tts_bound[1]]
            llm_tokens = torch.tensor(llm_tokens, device=self.tts.emb_text.weight.device, dtype=torch.long)
            llm_embeds = self.tts.emb_text(llm_tokens)  # make sure emb_text is compatible with llm vocab size

            hidden_embeds = last_hidden_states[tts_bound[0] : tts_bound[1]]
            hidden_embeds = self.tts.projector_semantic(hidden_embeds)

            if self.tts.config.normalize_projected_hidden:
                hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

            tts_embeds = llm_embeds + hidden_embeds
            if self.tts.interleaved:
                chunks = []
                cond_length = tts_embeds.shape[0]
                for i in range(0, cond_length, 10):
                    chunks.append(tts_embeds[i : i + 10])
                tts_embeds = chunks
        else:
            raise NotImplementedError

        audio_bos = [self.tts.audio_bos_token_id]
        audio_bos = torch.tensor(audio_bos, device=self.tts.emb_text.weight.device, dtype=torch.long)

        audio_bos_embeds = self.tts.emb_text(audio_bos)

        text_eos_embed = self.tts.emb_text(
            torch.tensor(
                [self.tts.config.text_eos_token_id],
                device=self.tts.emb_text.weight.device,
                dtype=torch.long,
            )
        )

        if self.tts.interleaved:

            tts_embeds[-1] = torch.cat([tts_embeds[-1], text_eos_embed], dim=0)
            for i in range(len(tts_embeds)):
                tts_embeds[i] = torch.cat([tts_embeds[i], audio_bos_embeds], dim=0).unsqueeze(0)
            outputs = self.tts.interleaved_generate(
                spk_embeds=spk_embeds,
                conditions=tts_embeds,
                temperature=0.8,
                repetition_penalty=1.05,
                eos_token=torch.tensor(
                    [self.tts.config.num_audio_tokens - 1],
                    dtype=torch.long,
                    device=self.tts.device,
                ),
            )
        else:
            if self.tts.condition_type == "tts_token":
                inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos_embed, audio_bos_embeds], dim=0).unsqueeze(
                    0
                )
            elif self.tts.condition_type == "tts_token_streaming":
                tts_embeds[1] = spk_embeds.squeeze(0)  # apply speaker embedding
                inputs_embeds = tts_embeds.unsqueeze(0)
            else:  # modern case
                inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos_embed, audio_bos_embeds], dim=0).unsqueeze(
                    0
                )

            # save inputs_embeds to file
            if output_tts_inputs_embeds_path:
                torch.save(inputs_embeds, output_tts_inputs_embeds_path)

            outputs = self.tts.generate(
                inputs_embeds=inputs_embeds,
                sampling_params=tts_sampling_params,
                eos_token=torch.tensor(
                    [self.tts.config.num_audio_tokens - 1],
                    dtype=torch.long,
                    device=self.tts.device,
                ),
            )

        if self.tts.config.audio_tokenizer_type == "s3tokenizer":
            # ========== CosyVoice2 vocoder 路径 ==========
            generated_tokens = outputs.new_ids.squeeze(-1)
            reference_audio = audio_prompt
            if reference_audio is not None:
                logger.debug("use reference audio in data to generate waveform")
                prompt_speech_16k = torch.tensor(reference_audio).unsqueeze(0)

            if self.tts.config.s3_stream_generate:
                waveform_pred = self.tts.audio_tokenizer.inference_token2wav(
                    speech_tokens=generated_tokens,
                    prompt_speech_16k=prompt_speech_16k,
                    prompt_speech=None,
                    stream=True,
                    n_timesteps=self.tts.config.s3_stream_n_timesteps,
                    code_chunk_size=self.tts.config.s3_stream_chunk_size,
                    chunk_prelook_size=self.tts.config.s3_stream_prelook_size,
                    use_attn_idx=False,
                )
                return waveform_pred[0]
            else:
                for i, j in enumerate(
                    self.tts.audio_tokenizer.token2wav(
                        speech_token=generated_tokens,
                        speech_token_len=torch.tensor([generated_tokens.shape[1]], device=generated_tokens.device),
                        prompt_speech_16k=prompt_speech_16k,
                        stream=False,
                    )
                ):
                    waveform_pred = j["tts_speech"]
                    waveform_sample_rate = self.tts.audio_tokenizer.sample_rate  # 24000 here, not 16000 input.
                return waveform_pred[0]

        elif self.tts.config.audio_tokenizer_type == "s3tokenizer_step_audio":
            # ========== Token2Wav vocoder 路径（非流式批量转换）==========
            generated_tokens = outputs.new_ids.squeeze(-1)
            token_ids = generated_tokens.reshape(-1).tolist()

            if not token_ids:
                logger.warning("Token2Wav non-streaming: 无 audio tokens 可转换")
                return None

            # Token2Wav 需要文件路径作为 prompt，将 ref audio 写入临时文件
            reference_audio = audio_prompt
            prompt_wav_path = None
            temp_file = None

            if reference_audio is not None:
                logger.debug("use reference audio in data to generate waveform (Token2Wav)")
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="t2w_ref_")
                sf.write(temp_file.name, reference_audio, 16000)
                prompt_wav_path = temp_file.name

            try:
                # 初始化 Token2Wav stream cache
                # set_stream_cache() 返回 (flow_cache, hift_cache)，需手动设置到实例属性
                self.tts.audio_tokenizer.cache = None
                flow_cache, hift_cache = self.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)
                self.tts.audio_tokenizer.stream_cache = flow_cache
                self.tts.audio_tokenizer.hift_cache_dict = hift_cache

                # Token2Wav 的 flow 模型有固定大小的 attention cache buffer，
                # 必须分块喂入（与 streaming 路径一致），否则长文本会溢出。
                CHUNK_SIZE = 25
                pre_lookahead = 3
                waveform_chunks = []
                buffer = list(token_ids)
                pos = 0

                while pos + CHUNK_SIZE + pre_lookahead <= len(buffer):
                    chunk = buffer[pos : pos + CHUNK_SIZE + pre_lookahead]
                    is_last = (pos + CHUNK_SIZE + pre_lookahead >= len(buffer))
                    wav_chunk = self.tts.audio_tokenizer.stream(
                        chunk, prompt_wav=prompt_wav_path,
                        last_chunk=is_last, return_waveform=True,
                    )
                    if wav_chunk is not None:
                        waveform_chunks.append(wav_chunk.squeeze())
                    pos += CHUNK_SIZE

                # flush 剩余 tokens
                if pos < len(buffer):
                    remaining = buffer[pos:]
                    wav_chunk = self.tts.audio_tokenizer.stream(
                        remaining, prompt_wav=prompt_wav_path,
                        last_chunk=True, return_waveform=True,
                    )
                    if wav_chunk is not None:
                        waveform_chunks.append(wav_chunk.squeeze())

                if waveform_chunks:
                    waveform = np.concatenate(waveform_chunks)
                    logger.info(
                        f"Token2Wav non-streaming: {len(token_ids)} tokens → "
                        f"{len(waveform)} samples ({len(waveform)/24000:.2f}s), "
                        f"{len(waveform_chunks)} chunks"
                    )
                    return waveform
                else:
                    logger.warning("Token2Wav non-streaming: 所有 chunks 返回空")
                    return None
            finally:
                # 清理临时文件
                if temp_file is not None:
                    try:
                        os.unlink(temp_file.name)
                    except OSError:
                        pass
        else:
            raise NotImplementedError(
                f"不支持的 audio_tokenizer_type: {self.tts.config.audio_tokenizer_type}"
            )

    @torch.inference_mode()
    def init_token2wav_cache(self, prompt_speech_16k):
        if hasattr(self.tts.audio_tokenizer, "set_stream_cache"):
            self.tts.audio_tokenizer.cache = None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                prompt_wav_path = tmp_wav.name
                sf.write(prompt_wav_path, prompt_speech_16k, 16000)
                flow_cache_base, hift_cache_base = self.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)

            self.token2wav_cache = {
                "flow_cache_base": torch_clone_recursive(flow_cache_base),
                "hift_cache_base": torch_clone_recursive(hift_cache_base),
            }
        else:
            model_input = self.tts.audio_tokenizer.frontend.frontend_token2wav(
                speech_tokens=torch.zeros(1, 1, dtype=torch.long, device=self.tts.device),
                speech_16k=None,
                prompt_speech_16k=prompt_speech_16k,
                resample_rate=self.tts.audio_tokenizer.sample_rate,
                prompt_speech=None,
            )

            prompt_token = model_input["flow_prompt_speech_token"]
            prompt_feat = model_input["prompt_speech_feat"]
            embedding = model_input["flow_embedding"]

            if self.tts.audio_tokenizer.fp16:
                prompt_feat = prompt_feat.to(torch.half)
                embedding = embedding.to(torch.half)

            prepared_cache = self.tts.audio_tokenizer.model.prepare_cache_from_prompt(
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                n_timesteps=self.tts.config.s3_stream_n_timesteps,
                code_chunk_size=self.tts.config.s3_stream_chunk_size,
                chunk_prelook_size=self.tts.config.s3_stream_prelook_size,
                use_attn_idx=False,
            )

            self.token2wav_cache = prepared_cache

    # for sliding window
    def _ensure_dynamic_cache(self):
        cache = self.llm_past_key_values
        if cache is None:
            return None

        cache = as_dynamic_cache(cache)
        if isinstance(cache, DynamicCache):
            self.llm_past_key_values = cache
            return cache

        return None

    def _get_kv_cache_length(self, cache=None):
        cache = cache if cache is not None else self.llm_past_key_values
        return get_kv_cache_length(cache)

    # todo: not-used del?
    def _rebuild_cache_from_history(self):
        preserved_ids: List[torch.Tensor] = []
        for entry in self._omni_chunk_history:
            ids = entry.get("input_ids")
            if ids is None or not isinstance(ids, torch.Tensor) or ids.numel() == 0:
                continue
            preserved_ids.append(ids.to(self.device))
        if not preserved_ids:
            self.llm_past_key_values = None
            self.streaming_position_offset = 0
            self._rope_inv_freq_cache.clear()
            return

        concat_ids = torch.cat(preserved_ids, dim=1)
        attention_mask = torch.ones((1, concat_ids.shape[1]), dtype=torch.bool, device=self.device)
        outputs = self.llm(
            input_ids=concat_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        self.llm_past_key_values = outputs.past_key_values
        self.streaming_position_offset = 0
        self._rope_inv_freq_cache.clear()

    def _get_rope_theta(self) -> float:
        return float(getattr(self.llm.config, "rope_theta", 10000.0))

    def _realign_rotary_suffix(
        self,
        suffix_keys: torch.Tensor,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Wrapper for realign_rotary_suffix using instance's rope_theta and cache."""
        return realign_rotary_suffix(
            suffix_keys,
            old_positions,
            new_positions,
            rope_theta=self._get_rope_theta(),
            inv_freq_cache=self._rope_inv_freq_cache,
        )

    def _encode_text(self, tokenizer, text) -> Optional[torch.Tensor]:
        if tokenizer is None or not text:
            return None
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids.to(self.device)

    @staticmethod
    def _safe_decode(tokenizer, input_ids):
        if tokenizer is None or input_ids is None:
            return None
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.cpu().tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        else:
            ids = input_ids
        try:
            return tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            return None

    def _finalize_round(
        self, round_id: Optional[int], cache_before: int, assistant_input_ids: Optional[torch.Tensor] = None
    ):
        if round_id is None:
            self._pending_round_id = None
            return
        cache_after = self._get_kv_cache_length()
        if assistant_input_ids is not None:
            assistant_len = assistant_input_ids.shape[1]
        else:
            assistant_len = max(cache_after - cache_before, 0)
        if assistant_len > 0:
            self._register_chunk(
                assistant_len,
                "assistant",
                round_id=round_id,
                input_ids=assistant_input_ids,
                tokenizer=self.processor.tokenizer if hasattr(self, "processor") else None,
            )
        logger.info(
            "Finalized round=%s cache len before=%s after=%s assistant_len=%s",
            round_id,
            cache_before,
            cache_after,
            assistant_len,
        )
        self._pending_round_id = None
        self._next_round_id += 1

    def _register_chunk(
        self,
        seq_len: int,
        chunk_type: str,
        *,
        round_id: int,
        input_ids=None,
        tokenizer=None,
    ) -> None:
        if seq_len <= 0:
            return
        entry = {"length": int(seq_len), "type": chunk_type, "round": round_id}
        if input_ids is not None:
            entry["input_ids"] = input_ids.clone().detach()
            entry["decoded"] = self._safe_decode(tokenizer, entry["input_ids"])
        else:
            entry["input_ids"] = None
            entry["decoded"] = None
        self._omni_chunk_history.append(entry)
        logger.info(
            "Registered chunk round=%s type=%s len=%s decoded=%s",
            round_id,
            chunk_type,
            entry["length"],
            entry["decoded"],
        )
        if chunk_type == "system":
            self.streaming_text_preserve = max(self.streaming_text_preserve, entry["length"])

    def _drop_tokens_from_cache(self, length: int, cache: DynamicCache) -> bool:
        """Drop tokens from cache using the utility function."""
        _, new_offset, success = drop_tokens_from_cache(
            cache=cache,
            length=length,
            preserve=self.streaming_text_preserve,
            position_offset=self.streaming_position_offset,
            rope_theta=self._get_rope_theta(),
            inv_freq_cache=self._rope_inv_freq_cache,
        )
        if success:
            self.streaming_position_offset = new_offset
        return success

    def _drop_next_round(self, cache: DynamicCache) -> bool:
        seen_rounds = set()
        for entry in self._omni_chunk_history:
            round_id = entry.get("round")
            if round_id is None or round_id in seen_rounds:
                continue
            seen_rounds.add(round_id)
            round_entries = [e for e in self._omni_chunk_history if e.get("round") == round_id]
            if any(e.get("type") == "system" for e in round_entries):
                continue
            if self._drop_round(round_id, cache):
                return True
        return False

    def _drop_round(self, round_id: int, cache: DynamicCache) -> bool:
        entries = [e for e in self._omni_chunk_history if e.get("round") == round_id]
        if not entries:
            return False
        total_len = sum(e["length"] for e in entries)
        if total_len <= 0:
            for e in entries:
                self._omni_chunk_history.remove(e)
            return False
        if not self._drop_tokens_from_cache(total_len, cache):
            return False
        for e in entries:
            logger.info(
                "Dropped round=%s chunk type=%s len=%s decoded=%s",
                round_id,
                e["type"],
                e["length"],
                e.get("decoded"),
            )
            self._omni_chunk_history.remove(e)
        return True

    def _enforce_text_window(self) -> None:
        if not self.streaming_window_enabled:
            return
        cache = self._ensure_dynamic_cache()
        if cache is None:
            return
        high_limit = max(0, int(self.streaming_window_config.text_window_high_tokens))
        low_limit = max(0, int(self.streaming_window_config.text_window_low_tokens))
        if high_limit <= 0:
            return
        target = max(0, low_limit)
        total_len = self._get_kv_cache_length(cache)
        if total_len <= high_limit:
            return
        dropped_any = False
        while total_len > target:
            if not self._drop_next_round(cache):
                break
            dropped_any = True
            total_len = self._get_kv_cache_length(cache)

    # for sliding window

    # Speculative snapshot helpers are provided by SpeculativeSnapshotMixin.

    @torch.inference_mode()
    def non_streaming_prefill(
        self,
        session_id,
        msgs,
        image=None,
        omni_mode=False,
        max_slice_nums=None,
        use_image_id=None,
        use_tts_template=False,
        enable_thinking=False,
        stream_input=False,
        max_inp_length=8192,
        merge_audio_from_same_content=True,
    ):
        """一次性 prefill 所有消息到 KV cache（非流式，复用 chat 的消息解析逻辑）

        与 streaming_prefill 的区别：
        - streaming_prefill 每次处理 1 条 msg，需要调用多次
        - non_streaming_prefill 一次处理所有 msgs，只调用一次

        两者都不加 generation_prompt，都设置好 KV cache 状态，
        之后统一用 streaming_generate() 或 non_streaming_generate() 做解码。

        Args:
            session_id: 会话 ID
            msgs: 消息列表 [{role, content}, ...]，content 可含 PIL.Image / np.ndarray / str
            image: 兼容 chat() 的 image 参数（一般传 None，图像集成到 msgs 中）
            omni_mode: 是否为 omni 模式（视频输入时为 True）
            max_slice_nums: HD 图像最大切片数
            use_image_id: 是否使用图像 ID
            use_tts_template: 是否使用 TTS 模板
            enable_thinking: 是否启用思考模式
            stream_input: 音频输入模式（False=完整音频）
            max_inp_length: 最大输入长度
            merge_audio_from_same_content: 是否合并同一 content 中的音频

        Returns:
            str: 构建的 prompt 字符串
        """
        assert session_id is not None, "session_id cannot be None"

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        # ── 1. 消息解析（复用 chat() 的逻辑） ──

        if isinstance(msgs, str):
            msgs = json.loads(msgs)

        copy_msgs = deepcopy(msgs)
        assert len(copy_msgs) > 0, "msgs is empty"

        if image is not None and isinstance(copy_msgs[0]["content"], str):
            copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

        images = []
        audios = []
        audio_parts = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["system", "user", "assistant"]
            if i == 0:
                assert role in ["user", "system"], "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("<image>./</image>")
                elif isinstance(c, np.ndarray):
                    audios.append(c)
                    audio_parts.append(i)
                    cur_msgs.append("<audio>./</audio>")
                    use_tts_template = True
                elif isinstance(c, str):
                    cur_msgs.append(c)

            if omni_mode or stream_input:
                msg["content"] = "".join(cur_msgs)
            else:
                msg["content"] = "\n".join(cur_msgs)

        prompt = self.processor.tokenizer.apply_chat_template(
            copy_msgs,
            tokenize=False,
            add_generation_prompt=False,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
        )

        if not merge_audio_from_same_content:
            audio_parts = None

        # ── 2. Tokenize + 预处理 ──

        inputs = self.processor(
            [prompt],
            [images],
            [audios],
            [audio_parts] if audio_parts is not None else None,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            stream_input=stream_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)

        inputs.pop("image_sizes", None)

        # ── 3. Session 状态初始化（与 streaming_prefill 对齐） ──

        self.reset_session(reset_token2wav_cache=False)
        self.session_id = session_id
        self.init_streaming_processor()

        # ── 4. Embedding 计算 ──

        model_inputs = {
            "input_ids": inputs["input_ids"],
            "audio_features": inputs.get("audio_features"),
            "audio_feature_lens": inputs.get("audio_feature_lens"),
            "image_bound": inputs.get("image_bound"),
            "audio_bounds": inputs.get("audio_bounds"),
            "spk_bounds": inputs.get("spk_bounds"),
        }

        if "pixel_values" in inputs:
            model_inputs["pixel_values"] = inputs["pixel_values"]
            model_inputs["tgt_sizes"] = inputs.get("tgt_sizes")

        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(model_inputs)
        inputs_embeds = self.get_omni_embedding(
            model_inputs,
            input_embeddings=model_inputs["inputs_embeds"],
            chunk_length=self.config.audio_chunk_length,
        )

        # ── 5. KV Cache Prefill ──

        round_id = self._next_round_id
        self._pending_round_id = round_id
        seq_len = inputs_embeds.shape[1]
        self._enforce_text_window()
        cache_length = self._get_kv_cache_length()

        attention_mask = torch.ones(
            (1, cache_length + inputs_embeds.shape[1]), dtype=torch.bool, device=self.device
        )

        outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )

        self.llm_past_key_values = as_dynamic_cache(outputs["past_key_values"])
        self._register_chunk(
            seq_len,
            "user",
            round_id=round_id,
            input_ids=inputs["input_ids"],
            tokenizer=self.processor.tokenizer,
        )
        self._enforce_text_window()
        if self.force_rope_reindex:
            self._force_reindex_all_cache()

        logger.info(
            f"non_streaming_prefill done: session={session_id}, "
            f"prompt_len={seq_len}, kv_cache_len={self._get_kv_cache_length()}"
        )

        return prompt

    @torch.inference_mode()
    def non_streaming_generate(
        self,
        session_id,
        max_new_tokens=256,
        do_sample=True,
        min_new_tokens=0,
        generate_audio=False,
        use_tts_template=True,
        enable_thinking=False,
        tts_ref_audio=None,
        tts_sampling_params=None,
        output_audio_path=None,
        length_penalty=1.1,
        tts_proj_layer=-1,
    ):
        """基于已有 KV cache 做非流式 HF generate + 可选 TTS

        必须在 non_streaming_prefill() 之后调用。
        """
        assert self.llm_past_key_values is not None, \
            "KV cache is empty — call non_streaming_prefill() first"

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(
                self.config._name_or_path, trust_remote_code=True
            )
        tokenizer = self.processor.tokenizer

        # 1. 构建 bos string（与 streaming_generate 对齐）
        bos_input = "".join([
            "<|im_end|>\n<|im_start|>assistant\n",
            "" if enable_thinking else self.think_str.replace("\\n", "\n"),
            "<|tts_bos|>" if use_tts_template else "",
        ])

        bos_input_ids = tokenizer.encode(bos_input)
        bos_input_ids = torch.tensor(
            bos_input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        bos_embeds = self.llm.get_input_embeddings()(bos_input_ids)

        # 2. bos prefill（注入 KV cache）
        cache_length = self._get_kv_cache_length()
        attention_mask = torch.ones(
            (1, cache_length + bos_embeds.shape[1]),
            dtype=torch.bool, device=self.device,
        )

        bos_outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=bos_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )
        self.llm_past_key_values = as_dynamic_cache(bos_outputs["past_key_values"])

        bos_seq_len = bos_embeds.shape[1]
        round_id = self._next_round_id
        self._pending_round_id = round_id
        self._register_chunk(
            bos_seq_len, "assistant", round_id=round_id,
            input_ids=bos_input_ids, tokenizer=tokenizer,
        )

        # 3. HF generate（基于 KV cache）
        generation_config = self.prepare_generation_config(
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            length_penalty=length_penalty,
        )
        generation_config.pop("max_new_tokens", None)

        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]

        cache_length_for_gen = self._get_kv_cache_length()
        gen_attention_mask = torch.ones(
            (1, cache_length_for_gen + 1),
            dtype=torch.bool, device=self.device,
        )

        last_logits = bos_outputs.logits[:, -1:, :]
        next_token = torch.argmax(last_logits, dim=-1)

        outputs = self.llm.generate(
            input_ids=next_token,
            past_key_values=self.llm_past_key_values,
            attention_mask=gen_attention_mask,
            pad_token_id=0,
            eos_token_id=terminators,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **generation_config,
        )

        # 4. 文本提取
        generated_ids = outputs.sequences[0]
        full_sequence = torch.cat([bos_input_ids[0], generated_ids])
        full_sequences = full_sequence.unsqueeze(0)
        outputs["full_sequences"] = full_sequences

        self._last_chat_token_stats = {
            "input_tokens": cache_length + bos_seq_len,
            "generated_tokens": len(generated_ids),
        }

        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        )
        for term_str in self.terminators:
            text = text.replace(term_str, "")
        text = text.rstrip("<|tts_eos|>").strip()

        # 更新 KV cache 状态
        self.llm_past_key_values = as_dynamic_cache(outputs.past_key_values) \
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None \
            else self.llm_past_key_values
        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = True

        # 5. TTS（可选）
        generated_waveform = None
        if use_tts_template and generate_audio:
            try:
                tts_bos_token = tokenizer.convert_tokens_to_ids("<|tts_bos|>")
                tts_eos_token = tokenizer.convert_tokens_to_ids("<|tts_eos|>")

                tts_bos_indices = []
                tts_eos_indices = []
                for i, x in enumerate(full_sequences[0]):
                    if x == tts_bos_token:
                        tts_bos_indices.append(i + 1)
                    elif x == tts_eos_token:
                        tts_eos_indices.append(i)

                tts_bos_idx = tts_bos_indices[-1] if tts_bos_indices else -1
                tts_eos_idx = tts_eos_indices[-1] if tts_eos_indices else None
                tts_bound = (tts_bos_idx, tts_eos_idx)

                _tts_audio_prompt = tts_ref_audio
                if _tts_audio_prompt is not None:
                    logger.info(f"[non_streaming_generate TTS] ref_audio: {len(_tts_audio_prompt)} samples")
                else:
                    logger.warning("[non_streaming_generate TTS] No ref audio")

                if tts_sampling_params is None:
                    tts_sampling_params = TTSSamplingParams()

                generated_waveform = self._generate_speech_non_streaming(
                    outputs=outputs,
                    tts_bound=tts_bound,
                    tts_proj_layer=tts_proj_layer,
                    audio_prompt=_tts_audio_prompt,
                    tts_sampling_params=tts_sampling_params,
                )
                if isinstance(generated_waveform, torch.Tensor):
                    generated_waveform = generated_waveform.cpu().numpy()

                if output_audio_path and generated_waveform is not None:
                    import soundfile as sf
                    sf.write(output_audio_path, generated_waveform, samplerate=24000)
            except:
                import traceback
                traceback.print_exc()
                generated_waveform = None

        logger.info(
            f"non_streaming_generate done: session={session_id}, "
            f"generated_tokens={len(generated_ids)}, "
            f"kv_cache_len={self._get_kv_cache_length()}, "
            f"has_audio={generated_waveform is not None}"
        )

        if generated_waveform is not None:
            return text, generated_waveform
        return text

    @torch.inference_mode()
    def streaming_prefill(
        self,
        session_id,
        msgs,
        tokenizer=None,  # deprecated
        omni_mode=True,
        max_slice_nums=None,
        use_tts_template=True,
        enable_thinking=False,
        is_last_chunk=False,  # for audio chunk, if is the last chunk, set to True
        stream_input=None,  # None=auto (is_not_system_prefill), False=完整音频, True=实时流式音频(双工)
        **kwargs,
    ):
        assert session_id is not None, "session_id cannot be None"
        self.is_first = self.session_id is None or session_id != self.session_id

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        images = []
        audios = []

        assert len(msgs) == 1
        copy_msgs = deepcopy(msgs)
        msg = copy_msgs[0]

        assert msg["role"] in ["system", "user", "assistant"]
        is_not_system_prefill = msg["role"] != "system"

        content = msg["content"]
        cur_msgs = []
        for j, c in enumerate(content):
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("<image>./</image>")
            elif isinstance(c, np.ndarray):
                audios.append(c)
                cur_msgs.append("<audio>./</audio>")
            elif isinstance(c, str):
                cur_msgs.append(c)
            else:
                logger.error(f"Invalid content type: {c}, ignore it.")

        cur_contents = "".join(cur_msgs) if omni_mode else "\n".join(cur_msgs)

        if msg["role"] in ["system", "assistant"]:
            self.new_user_msg = True
            self.audio_past_key_values = None

        if self.is_first:
            self.reset_session(reset_token2wav_cache=False)
            self.session_id = session_id

            self.init_streaming_processor()

            if msg["role"] == "user":
                # 没有 system prefill，第一个 user turn 的第一个 segment
                # 不使用 apply_chat_template，手动构建 prompt 以避免自动添加 <|im_end|>
                prompt = "<|im_start|>user\n" + cur_contents
                self.new_user_msg = False  # 标记后续 segments 不需要再添加 user 前缀
            else:
                # system 或 assistant prefill，使用 apply_chat_template
                msg["content"] = cur_contents
                prompt = self.processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    use_tts_template=use_tts_template,
                    enable_thinking=enable_thinking,
                )
            add_special_tokens = True  # add bos
        else:
            # 非首次 prefill
            if self.new_user_msg and msg["role"] == "user":
                # 新的 user turn 的第一个 segment
                if self.llm_generated:
                    # todo: when to set llm_generate_completed?
                    if self.llm_generate_completed:
                        prompt = "<|im_end|>\n<|im_start|>user\n" + cur_contents
                    else:
                        prompt = "<|tts_eos|><|im_end|>\n<|im_start|>user\n" + cur_contents
                else:
                    prompt = "<|im_start|>user\n" + cur_contents
                self.new_user_msg = False
            else:
                # 同一个 turn 的后续 segments，直接使用内容
                prompt = cur_contents
            add_special_tokens = False

        # when first user audio prefill, ensure audio length satisfies FIRST_CHUNK_MS requirements
        if is_not_system_prefill and len(audios) > 0 and self.audio_chunk_idx == 0:
            assert len(audios) == 1, f"streaming mode only supports single audio, currently {len(audios)}"
            first_chunk_samples = int(self.FIRST_CHUNK_MS * self.SAMPLE_RATE / 1000)
            if len(audios[0]) < first_chunk_samples:
                pad_len = first_chunk_samples - len(audios[0])
                audios[0] = np.concatenate([np.zeros(pad_len, dtype=audios[0].dtype), audios[0]])

        # stream_input: None=auto, False=完整音频, True=实时流式（双工）
        _stream_input = stream_input if stream_input is not None else is_not_system_prefill

        # online_streaming: 控制 processor 是否使用流式 mel 处理
        # 完整音频（stream_input=False）时不用流式 mel
        _online_streaming = is_not_system_prefill if _stream_input else False

        model_inputs = self.processor(
            [prompt],
            [images],
            [audios],
            max_slice_nums=1 if max_slice_nums is None else max_slice_nums,
            use_image_id=False,
            chunk_input=True,
            return_tensors="pt",
            max_length=None,
            sampling_rate=16000,
            add_special_tokens=add_special_tokens,
            online_streaming=_online_streaming,
            audio_chunk_idx=self.audio_chunk_idx,
            is_last_chunk=is_last_chunk,
        ).to(self.device)

        # DEBUG: 打印 mel 特征的 checksum（用于诊断 rollback 不一致问题）
        if len(audios) > 0 and is_not_system_prefill and hasattr(self, "_debug_prefill") and self._debug_prefill:
            audio_feats = model_inputs.get("audio_features", None)
            if audio_feats is not None and hasattr(audio_feats, "sum"):
                mel_sum = audio_feats.sum().item()
                mel_shape = audio_feats.shape
                print(
                    f"[DEBUG prefill] audio_chunk_idx={self.audio_chunk_idx}, mel_sum={mel_sum:.6f}, mel_shape={mel_shape}"
                )
            else:
                print(f"[DEBUG prefill] audio_chunk_idx={self.audio_chunk_idx}, audio_feats type={type(audio_feats)}")

        if len(audios) > 0 and is_not_system_prefill:
            self.audio_chunk_idx += 1

        # 1. prepare input embeddings
        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(model_inputs)
        # get audio embedding with audio_past_key_values
        # todo: should pass chunk_length=self.config.audio_chunk_length ?
        inputs_embeds = self.get_omni_embedding(
            model_inputs, input_embeddings=model_inputs["inputs_embeds"], stream_input=_stream_input
        )

        # DEBUG: 打印 inputs_embeds 的 checksum
        if len(audios) > 0 and is_not_system_prefill and hasattr(self, "_debug_prefill") and self._debug_prefill:
            embed_sum = inputs_embeds.sum().item()
            embed_shape = inputs_embeds.shape
            print(f"[DEBUG prefill] inputs_embeds sum={embed_sum:.6f}, shape={embed_shape}")

        if self.is_first:
            self.audio_past_key_values = None  # clean audio_past_key_values after first prefill

        round_id = self._next_round_id
        self._pending_round_id = round_id
        chunk_type = "system" if msg["role"] == "system" else ("user" if msg["role"] == "user" else "assistant")
        seq_len = inputs_embeds.shape[1]
        self._enforce_text_window()
        cache_length = self._get_kv_cache_length()

        attention_mask = torch.ones((1, cache_length + inputs_embeds.shape[1]), dtype=torch.bool, device=self.device)

        # 2. do prefill
        outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )

        self.llm_past_key_values = as_dynamic_cache(outputs["past_key_values"])
        self._register_chunk(
            seq_len,
            chunk_type,
            round_id=round_id,
            input_ids=model_inputs["input_ids"],
            tokenizer=self.processor.tokenizer,
        )
        self._enforce_text_window()
        if self.force_rope_reindex:
            self._force_reindex_all_cache()

        return prompt

    @torch.inference_mode()
    def streaming_generate(
        self,
        session_id,
        tokenizer=None,  # deprecated
        bos_input=None,
        generate_audio=True,
        audio_token_chunk_size=25,  # 25 token/s
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
        max_new_tokens=256,
        fn="chunk_generate",
        enable_thinking=False,
        use_tts_template=True,
        do_sample=True,
        enable_speculative_snapshot=False,
        **kwargs,
    ):
        # 保存抢跑快照（在修改任何状态之前）
        # 用于 VAD 抢跑场景：如果抢跑失败，可调用 restore_speculative_snapshot() 恢复
        # enable_speculative_snapshot=True 时启用，False 时跳过（节省少量开销）
        if enable_speculative_snapshot:
            self._speculative_snapshot = self._save_speculative_snapshot()

        # reset buf
        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = False
        self.audio_past_key_values = None

        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        # reset current turn generated token IDs
        if hasattr(self, "_streaming_generated_token_ids"):
            del self._streaming_generated_token_ids
        # reset full generated text
        if hasattr(self, "_last_streaming_text"):
            del self._last_streaming_text

        cache = self._ensure_dynamic_cache()
        cache_length = self._get_kv_cache_length(cache)
        host_round_id = self._pending_round_id
        logger.info("streaming_generate kv cache length before= %s", cache_length)

        ## 单工情况每调用一次 streaming_generate 需要重新初始化 streaming_processor, 进入下一个 turn
        self.init_streaming_processor()

        # 1) llm generate token and hidden states per chunk=10, 2) tts generate audio token chunk per chunk=25, 3) yield 1 chunk audio token
        def audio_chunk_generator(
            bos_input,
            tokenizer,
            generate_audio,
            tts_sampling_params,
            max_new_tokens,
            do_sample,
            **kwargs,
        ):
            generate_chunk_size = 10

            if bos_input is None:
                bos_input = "".join(
                    [
                        "<|im_end|>\n<|im_start|>assistant\n",
                        "" if enable_thinking else self.think_str.replace("\\n", "\n"),
                        "<|tts_bos|>" if use_tts_template else "",
                    ]
                )

            bos_input_ids = tokenizer.encode(bos_input)
            bos_input_ids = torch.tensor(bos_input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            # DEBUG: 打印生成开始时的状态
            _cache_len = self._get_kv_cache_length()
            _cache_sum = self.llm_past_key_values.key_cache[0].sum().item() if self.llm_past_key_values else 0
            # 检查 KV Cache 最后几个位置的值
            _k_last = (
                self.llm_past_key_values.key_cache[0][0, 0, -5:, :3].flatten().tolist()
                if self.llm_past_key_values
                else []
            )
            print(f"[DEBUG streaming_generate] cache_len={_cache_len}, cache_sum={_cache_sum:.6f}, k_last={_k_last}")

            bos_input_embeds = self.llm.get_input_embeddings()(bos_input_ids)

            generation_inputs_embeds = bos_input_embeds
            generated_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)

            num_chunks_decode = (max_new_tokens + generate_chunk_size - 1) // generate_chunk_size

            conditions = []

            # generate chunk by chunk, each chunk has 10 tokens, each chunk takes last hidden states, and pass tokens to tts
            llm_streaming_generator = ChunkPrefillChunkGenerate(
                model=self.llm,
                tokenizer=tokenizer,
                terminators=["<|tts_eos|>", "<|im_end|>", "</s>"],
            )

            if generate_audio:
                logits_warpers, logits_processors = gen_logits(
                    num_code=self.tts.config.num_audio_tokens,
                    repetition_penalty=tts_sampling_params.repetition_penalty,
                    top_p=tts_sampling_params.top_p,
                    top_k=tts_sampling_params.top_k,
                )

                tts_streaming_generator = TTSStreamingGenerator(
                    model=self.tts,
                    temperature=tts_sampling_params.temperature,
                    eos_token=torch.tensor(
                        [self.tts.config.num_audio_tokens - 1],
                        dtype=torch.long,
                        device=self.tts.device,
                    ),
                    chunk_size=audio_token_chunk_size,  # s3tokenizer 1s = 25token
                    tts_last_turn_tokens=self.tts_last_turn_tokens,
                    logits_processors=logits_processors,
                    logits_warpers=logits_warpers,
                )

            # LLM chunk generate outer loop
            for chunk_idx in range(num_chunks_decode):
                is_first_generate_chunk = chunk_idx == 0

                output = llm_streaming_generator.chunk_generate(
                    inputs_embeds=generation_inputs_embeds,
                    past_key_values=self.llm_past_key_values,
                    is_first_generate_chunk=is_first_generate_chunk,
                    return_hidden_states=True,
                    chunk_size=generate_chunk_size + 1 * is_first_generate_chunk,
                    do_sample=do_sample,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.8),
                    top_k=kwargs.get("top_k", 100),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.02),
                    length_penalty=kwargs.get("length_penalty", 1.0),
                    all_input_ids=generated_ids,
                    suppress_forbidden_tokens=generate_audio,
                )

                if output.chunk_token_ids is None:
                    break

                # DEBUG: 打印第一个 chunk 生成的 token
                if chunk_idx == 0:
                    print(f"[DEBUG streaming_generate] first_chunk_tokens={output.chunk_token_ids.tolist()}")

                if is_first_generate_chunk:
                    if generate_audio:
                        spk_emb = torch.empty(
                            (bos_input_embeds.shape[0], 0, bos_input_embeds.shape[2]),
                            dtype=bos_input_embeds.dtype,
                            device=bos_input_embeds.device,
                        )
                        tts_streaming_generator.spk_emb = spk_emb

                    if output.finished:
                        yield_chunk_token_ids = output.chunk_token_ids
                    else:
                        # the first chunk generated chunk_size + 1 tokens, we only take the first chunk_size tokens,
                        # the last token is not prefilled, and last hidden states is not obtained
                        yield_chunk_token_ids = output.chunk_token_ids[:, :-1]

                elif output.finished:
                    yield_chunk_token_ids = torch.cat([generated_ids[:, -1:], output.chunk_token_ids], dim=1)
                else:
                    # in the chunk that is not the first chunk, we need to add the token at the end of the previous chunk,
                    # it is not prefilled into the model to get last hidden states
                    # similarly, the last generated token of subsequent chunks is not prefilled, and last hidden states is not obtained,
                    # so it is not passed out
                    yield_chunk_token_ids = torch.cat([generated_ids[:, -1:], output.chunk_token_ids[:, :-1]], dim=1)

                if not generate_audio:
                    chunk_generated_text = tokenizer.decode(yield_chunk_token_ids[0])
                    yield yield_chunk_token_ids, output.finished
                else:
                    # TTS inner loop
                    # dense connection here is hardcoded to use text-hidden merged as condition
                    llm_embeds = self.tts.emb_text(yield_chunk_token_ids)
                    hidden_embeds = output.last_hidden_states
                    hidden_embeds = self.tts.projector_semantic(hidden_embeds)
                    if self.tts.config.normalize_projected_hidden:  # default should be opened
                        hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

                    tts_embeds = llm_embeds + hidden_embeds
                    conditions.append(tts_embeds)

                    # Store token IDs instead of decoded text to avoid UTF-8 multi-byte character truncation
                    if not hasattr(self, "_streaming_generated_token_ids"):
                        self._streaming_generated_token_ids = []
                    self._streaming_generated_token_ids.extend(yield_chunk_token_ids[0].tolist())

                    # there is buffer generated, each time exactly returns 25 audio tokens,
                    # the last audio chunk returns audio tokens of variable length, length [0, 25]
                    tts_generator = tts_streaming_generator.generate_with_buffer(
                        condition=tts_embeds, text_finished=output.finished
                    )

                    for audio_token_chunk, is_last_audio_chunk in tts_generator:
                        yield audio_token_chunk, is_last_audio_chunk

                generated_ids = torch.cat([generated_ids, output.chunk_token_ids], dim=1)
                generation_inputs_embeds = output.current_inputs_embeds
                self.llm_past_key_values = output.past_key_values

                if output.finished:
                    if generate_audio:
                        self.tts_last_turn_tokens = tts_streaming_generator.tts_last_turn_tokens
                    break

            # IMPORTANT: Flush remaining TTS buffer when LLM generation ends
            # This handles BOTH cases:
            # 1. LLM finished with terminator (output.finished=True) - buffer may still have tokens
            # 2. LLM hit max chunks limit (output.finished=False) - buffer definitely has tokens
            if generate_audio:
                if len(tts_streaming_generator._token_buffer) > 0:
                    batch = torch.cat(tts_streaming_generator._token_buffer, dim=1)
                    yield batch, True
                    tts_streaming_generator._token_buffer = []

            if generate_audio:
                if hasattr(self, "_streaming_generated_token_ids"):
                    try:
                        self._last_streaming_text = tokenizer.decode(self._streaming_generated_token_ids)
                        assistant_input_ids = self._encode_text(tokenizer=tokenizer, text=self._last_streaming_text)
                        self._finalize_round(
                            round_id=host_round_id, cache_before=cache_length, assistant_input_ids=assistant_input_ids
                        )
                    except Exception:
                        self._last_streaming_text = None
                else:
                    self._last_streaming_text = None

                yield None, None
            else:
                return

        # iter for generating text chunk and audio chunk
        audio_chunk_generator_iter = audio_chunk_generator(
            bos_input=bos_input,
            tokenizer=self.processor.tokenizer,
            generate_audio=generate_audio,
            tts_sampling_params=tts_sampling_params,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

        if generate_audio:
            if self.tts.config.audio_tokenizer_type == "s3tokenizer_step_audio":
                self.tts.audio_tokenizer.stream_cache = torch_clone_recursive(self.token2wav_cache["flow_cache_base"])
                self.tts.audio_tokenizer.hift_cache_dict = torch_clone_recursive(
                    self.token2wav_cache["hift_cache_base"]
                )

                # pre-insert 3-5 prefix 4218 silence tokens, each token corresponds to 0.04s,
                # adding 5 tokens means introducing 0.2s of silence
                buffer = [4218] * 3
                pre_lookahead = 3
                CHUNK_SIZE = 25
                chunk_idx = 0
                prev_text_len = 0  # track text position for streaming text output
                for audio_token_chunk, is_last_audio_chunk in audio_chunk_generator_iter:
                    if audio_token_chunk is None:
                        break

                    buffer += audio_token_chunk.reshape(-1).tolist()

                    if len(buffer) >= CHUNK_SIZE + pre_lookahead:
                        waveform_chunk = self.tts.audio_tokenizer.stream(
                            buffer[: CHUNK_SIZE + pre_lookahead],
                            prompt_wav=None,
                            last_chunk=is_last_audio_chunk,
                            return_waveform=True,
                        )

                        waveform_chunk = torch.from_numpy(waveform_chunk)

                        # get new text chunk corresponding to this waveform
                        # Decode from accumulated token IDs to avoid UTF-8 multi-byte truncation
                        new_text = ""
                        if hasattr(self, "_streaming_generated_token_ids"):
                            current_text = self.processor.tokenizer.decode(self._streaming_generated_token_ids)
                            # Filter out trailing replacement characters (incomplete UTF-8 sequences)
                            safe_end = len(current_text)
                            while safe_end > 0 and current_text[safe_end - 1] == "\ufffd":
                                safe_end -= 1
                            safe_text = current_text[:safe_end]
                            new_text = safe_text[prev_text_len:]
                            prev_text_len = len(safe_text)

                        yield waveform_chunk, new_text

                        buffer = buffer[CHUNK_SIZE:]
                        chunk_idx += 1

                # flush rest
                if len(buffer) > 0:
                    waveform_chunk = self.tts.audio_tokenizer.stream(
                        buffer,
                        prompt_wav=None,
                        last_chunk=True,
                        return_waveform=True,
                    )

                    waveform_chunk = torch.from_numpy(waveform_chunk)

                    # get remaining new text for the final chunk
                    # Final chunk: decode all remaining text without filtering
                    new_text = ""
                    if hasattr(self, "_streaming_generated_token_ids"):
                        current_text = self.processor.tokenizer.decode(self._streaming_generated_token_ids)
                        new_text = current_text[prev_text_len:]
                        prev_text_len = len(current_text)

                    yield waveform_chunk, new_text

                # maybe the buffer is empty, and text is not empty, should we flush text without wave?
            else:
                raise NotImplementedError(f"not supported audio tokenizer: {self.tts.config.audio_tokenizer_type}")
        else:
            # For text-only generation, decode tokens and handle partial multi-byte characters
            yield from streaming_token_decoder(
                audio_chunk_generator_iter,
                self.processor.tokenizer,
                skip_special_tokens=False,
            )

    # Duplex convenience methods are provided by DuplexProxyMixin.
