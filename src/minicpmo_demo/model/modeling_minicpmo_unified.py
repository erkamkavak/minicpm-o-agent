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

import logging
import os
import types
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
from transformers import Qwen3ForCausalLM
from transformers import Qwen3PreTrainedModel

from enum import Enum

# 相对导入（同目录）
from .configuration_minicpmo import MiniCPMOConfig
from .capabilities import DuplexCapability
from .services import DuplexProxyMixin
from .services import ChatGenerationMixin
from .services import MediaEmbeddingMixin
from .services import StreamingGenerationMixin
from .services import StreamingSessionMixin
from .services import SpeculativeSnapshotMixin
from .services import UnifiedOperationsMixin
from .components import MiniCPMTTS
from .components import MiniCPMWhisperEncoder
from .components import MultiModalProjector
from .components import prepare_inputs_for_generation
from .components import Resampler
from .components.vision_encoder import SiglipVisionTransformer
from .processing_minicpmo import MiniCPMOProcessor
from .runtime.cache import StreamingWindowConfig

logger = logging.getLogger(__name__)


class ProcessorMode(Enum):
    """处理器模式枚举"""
    CHAT = "chat"           # 单工对话（非流式 TTS）
    STREAMING = "streaming" # 流式对话（流式 TTS）
    DUPLEX = "duplex"       # 双工对话（流式 TTS + 双工组件）


class MiniCPMOPreTrainedModel(Qwen3PreTrainedModel):
    config_class = MiniCPMOConfig
    # Transformers 5.x expects this mapping during load finalization. The
    # wrapper delegates tied embedding/head behavior to the nested Qwen3 model.
    all_tied_weights_keys = {}


class MiniCPMO(
    UnifiedOperationsMixin,
    MediaEmbeddingMixin,
    StreamingSessionMixin,
    ChatGenerationMixin,
    StreamingGenerationMixin,
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

        # Preload TTS vocoder only when audio generation is enabled. Text/logit
        # benchmark paths can run without the optional Token2Wav dependency.
        if self._duplex_config.get("generate_audio", True):
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

    # Media prompt and embedding helpers are provided by MediaEmbeddingMixin.

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

    # Chat and speech generation helpers are provided by ChatGenerationMixin.

    # Streaming session/cache helpers are provided by StreamingSessionMixin.

    # Streaming and non-streaming conversation APIs are provided by StreamingGenerationMixin.
