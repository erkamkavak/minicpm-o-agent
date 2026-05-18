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

"""MiniCPM-o TTS model and audio-token generation components."""

import logging
import math
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from transformers import LlamaConfig
from transformers import LlamaModel
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.generation.logits_process import TopKLogitsWarper
from transformers.generation.logits_process import TopPLogitsWarper
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import ModelOutput

from ..configuration_minicpmo import MiniCPMTTSConfig
from ..runtime.tts_streaming import TTSSamplingParams

logger = logging.getLogger(__name__)


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, audio_features):
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_dim = config.llm_hidden_size
        self.out_dim = config.hidden_size
        self.intermediate_size = config.llm_intermediate_size
        self.gate_proj = nn.Linear(self.in_dim, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.in_dim, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.out_dim, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


@dataclass
class MiniCPMTTSGenerationOutput(ModelOutput):
    """
    Output class for MiniCPMTTS generation.

    Args:
        new_ids (torch.LongTensor): Newly generated audio code sequence, shape (batch_size, sequence_length, num_vq).
        audio_input_ids (torch.LongTensor): Updated input IDs including condition and generated audio codes, shape (batch_size, full_sequence_length, num_vq).
        past_key_values (Tuple[Tuple[torch.FloatTensor]]): Tuple containing pre-computed keys and values used for attention mechanism. Each element has shape (batch_size, num_heads, sequence_length, embed_size_per_head).
        finished (bool): Boolean indicating whether generation is complete.
    """

    new_ids: torch.LongTensor = None
    audio_input_ids: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_input_ids: Optional[torch.LongTensor] = None
    finished: bool = None


def make_streaming_chunk_mask_inference(
    tts_text_scope: List[int],
    tts_text_mask: torch.Tensor,
    streaming_audio_chunk_size: int = 50,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    max_sequence_length: int = 4096,
):
    """
    Example:
    Input sequence:
    [t1, t2, t3, t4, t5, [Ptts], a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, ...]
    Output 4D causal mask:
    ------- text positions -------
    [0] <- here is [Stts]
    [0,   0] <- here is [spk_emb] * N
    [0,   0,   0]
    [0,   0,   0,    0]
    [0,   0,   0,    0,    0]
    ------- audio positions --------
    [0,    0, -inf, -inf, -inf, 0] <- here is [Ptts], [Ptts]'s last hidden state should predict the first audio token
                                v- here is [Ptts]
    [0,    0, -inf, -inf, -inf, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0, 0, 0] # end of first 1s audio chunk
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """

    # Create a complete attention mask for input embeds [batch_size, seq_len], without considering audio mask as audio is always at the end

    assert tts_text_mask.dtype == torch.int8

    padding_mask = torch.ones(max_sequence_length, dtype=torch.int8, device=device)
    padding_mask[tts_text_scope[0] : tts_text_scope[1]] = tts_text_mask

    # Initialize a standard upper triangular causal mask
    min_dtype = torch.finfo(dtype).min

    causal_mask = torch.full(
        (max_sequence_length, max_sequence_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if max_sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    else:
        raise ValueError("max_sequence_length of tts could not be 1.")

    # For each data sample
    audio_token_start = tts_text_scope[1]
    audio_duration = max_sequence_length - tts_text_scope[1]

    # Record which text chunk the current audio chunk can see up to
    text_pivot = 0
    num_valid_text_tokens = torch.sum(tts_text_mask).item() - 1  # [Ptts] excluded
    # How many audio chunks are in total, the num of buckets should be smaller as possible

    num_text_tokens_per_audio_chunk = 10

    # For each chunk of audio
    for chunk_idx in range(math.ceil(audio_duration / streaming_audio_chunk_size)):
        audio_chunk_start = audio_token_start + chunk_idx * streaming_audio_chunk_size
        audio_chunk_end = audio_token_start + (chunk_idx + 1) * streaming_audio_chunk_size
        # New text seen by this new audio chunk
        new_text_this_chunk = num_text_tokens_per_audio_chunk
        # The right bound of visible text tokens
        text_pivot = min(new_text_this_chunk + text_pivot, num_valid_text_tokens)
        # Mask all text chunks after the visible ones
        # -> [text_pivot, len(tts_text_scope)-1] excluding [Ptts]
        # print("audio_chunk_start-1", audio_chunk_start-1, "audio_chunk_end-1", audio_chunk_end-1, "tts_text_scope[0] + text_pivot", tts_text_scope[0] + text_pivot, "tts_text_scope[1] - 1", tts_text_scope[1] - 1)
        causal_mask[
            audio_chunk_start - 1 : audio_chunk_end - 1,
            # tts_text_scope[0] + text_pivot: tts_text_scope[1],
            tts_text_scope[0] + text_pivot : tts_text_scope[1] - 1,
        ] = min_dtype

    # Mask the padding parts in tts_text_masks (no position will attend to it)
    causal_mask[:, padding_mask == 0] = min_dtype

    # Add extra dimensions, [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask


class MiniCPMTTS(PreTrainedModel):
    config_class = MiniCPMTTSConfig

    def __init__(self, config: MiniCPMTTSConfig, audio_tokenizer: None):
        super().__init__(config)

        self.use_llm_hidden_state = config.use_llm_hidden_state

        self.use_text = config.use_text
        self.streaming = config.streaming
        self.streaming_text_chunk_min = config.streaming_text_chunk_min
        self.streaming_text_chunk_max = config.streaming_text_chunk_max
        self.streaming_audio_chunk_size = config.streaming_audio_chunk_size
        self.streaming_text_reserved_len = config.streaming_text_reserved_len
        # streaming tts
        self.streaming_text_chunk_size = config.streaming_text_chunk_max
        self.audio_bos_token_id = config.audio_bos_token_id
        self.num_mel_bins = config.num_mel_bins
        self.num_vq = config.num_vq
        self.num_audio_tokens = config.num_audio_tokens

        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty

        self.interleaved = config.interleaved
        self.attention_type = config.attention_type
        self.recomputed_chunks = config.recomputed_chunks
        
        # Two different window size concepts:
        # 1. chunk_window_size: number of chunks for sliding_recompute mode (default 2)
        # 2. token_window_size: number of tokens for sliding_window mode (default 300)
        self.chunk_window_size = config.window_size  # chunk-level window for sliding_recompute
        self.token_window_size = config.streaming_sliding_window_audio_window_size  # token-level window for sliding_window
        
        # Legacy aliases (for backward compatibility with existing code)
        self.window_size = self.chunk_window_size  # used in generate_streaming for sliding_recompute
        self.sliding_window_size = self.token_window_size  # used in TTSStreamingGenerator for sliding_window
        
        if self.attention_type == "sliding_recompute" and self.chunk_window_size <= self.recomputed_chunks:
            raise ValueError(
                f"sliding_recompute requires chunk_window_size > recomputed_chunks, "
                f"but got chunk_window_size={self.chunk_window_size} and recomputed_chunks={self.recomputed_chunks}"
            )

        if config.backbone_model == "llama":
            model_config = LlamaConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_hidden_layers=config.num_hidden_layers,
                num_key_value_heads=config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                attn_implementation=config.attn_implementation,
            )

            self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)

            model = LlamaModel(model_config)
            self.model = model
        else:
            raise ValueError(f"Unsupported backbone model: {config.backbone_model}")

        self.projector_spk = self.create_projector(config)
        self.projector_semantic = self.create_projector(config)

        self.audio_tokenizer = audio_tokenizer

        self.emb_code = nn.ModuleList(
            [nn.Embedding(config.num_audio_tokens, config.hidden_size) for _ in range(config.num_vq)]
        )

        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False),
                    name="weight",
                )
                for _ in range(config.num_vq)
            ]
        )

        self.condition_type = config.condition_type

        return

    @staticmethod
    def create_projector(config):
        if config.projector_type == "mlp":
            return MultiModalProjector(config.llm_dim, config.hidden_size)
        elif config.projector_type == "minicpm":
            return MiniCPMMLP(config)
        elif config.projector_type == "default":
            return nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        else:
            raise ValueError(f"Unsupported projector type: {config.projector_type}")

    # non-streaming
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        min_new_token=50,
        max_new_token=2048,
        show_tqdm=True,
        streaming=False,
        text_lengths=None,
        sampling_params: TTSSamplingParams = TTSSamplingParams(),
    ):
        temperature = torch.tensor(
            [sampling_params.temperature] * self.config.num_vq,
            dtype=torch.float,
            device=self.device,
        )
        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        condition_length = inputs_embeds.shape[1]
        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        if streaming:
            raise NotImplementedError("this kind of streaming is not supported yet")

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        past_key_values = None

        for t in range(max_new_token):
            audio_bos = False
            # If this is the first audio token, the case is special
            if t == 0:
                audio_bos = True
                inputs_embeds = inputs_embeds
                position_ids = torch.tensor(
                    list(range(0, condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                if streaming:
                    raise NotImplementedError("this kind of streaming is not supported yet")
                else:
                    causal_mask_4d = None

            else:
                code_emb = []
                for q in range(self.num_vq):
                    x = self.emb_code[q](new_tokens[:, t - 1 : t, q])
                    code_emb.append(x)

                inputs_embeds = torch.stack(code_emb, 3).sum(3)

                position_ids = torch.tensor([condition_length + t - 1], dtype=torch.long, device=self.device).unsqueeze(
                    0
                )

                if streaming:
                    raise NotImplementedError("this kind of streaming is not supported yet")
                else:
                    causal_mask_4d = None

            if self.config.backbone_model == "llama":
                outputs: BaseModelOutputWithPast = self.model(
                    position_ids=position_ids,
                    cache_position=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask_4d,
                    use_cache=True,
                    output_attentions=False,
                    # return_dict=True,  # Add this to ensure returns dict with past_key_values
                )
            else:
                raise ValueError(f"Unsupported backbone model: {self.config.backbone_model}")

            del position_ids
            del inputs_embeds

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

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

            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens
                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

                del logits_token

            if t < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits

            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)

            del scores

            idx_next = idx_next.view(-1, self.num_vq)

            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

        genrated_input_ids = new_tokens[:, 0:t, :]

        return MiniCPMTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=None,  # for update purpose
            past_key_values=None,  # for update purpose
            past_input_ids=None,  # for update purpose
            finished=finish.all(),
        )

    # fake streaming
    @torch.inference_mode()
    def generate_mock_legacy_streaming(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        min_new_token=50,
        max_new_token=2048,
        show_tqdm=True,
        streaming=False,
        text_lengths=None,
        sampling_params: TTSSamplingParams = TTSSamplingParams(),
        valid_text_length=None,
    ):
        assert valid_text_length is not None, "valid_text_length should be not None"

        tts_text_scope = [0, inputs_embeds.shape[1]]
        tts_text_mask = torch.zeros(inputs_embeds.shape[1], dtype=torch.int8, device=inputs_embeds.device)
        tts_text_mask[0:valid_text_length] = 1
        tts_text_mask[-1] = 1  # [Ptts]

        streaming_mask_4d_full = make_streaming_chunk_mask_inference(
            tts_text_scope=tts_text_scope,
            tts_text_mask=tts_text_mask,
            dtype=torch.bfloat16,
            device=self.device,
            streaming_audio_chunk_size=50,
            max_sequence_length=4096,
        )

        temperature = torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.device)
        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        condition_length = inputs_embeds.shape[1]
        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        past_key_values = None

        for t in range(max_new_token):
            audio_bos = False
            if t == 0:
                audio_bos = True
                inputs_embeds = inputs_embeds
                position_ids = torch.tensor(
                    list(range(0, condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                causal_mask_4d = streaming_mask_4d_full[:, :, :condition_length, :condition_length]
            else:
                code_emb = []
                for q in range(self.num_vq):
                    x = self.emb_code[q](new_tokens[:, t - 1 : t, q])
                    code_emb.append(x)

                inputs_embeds = torch.stack(code_emb, 3).sum(3)

                position_ids = torch.tensor([condition_length + t - 1], dtype=torch.long, device=self.device).unsqueeze(
                    0
                )

                causal_mask_4d = streaming_mask_4d_full[
                    :,
                    :,
                    condition_length + t : condition_length + t + 1,
                    : condition_length + t,
                ]

                # get length of past_key_values
                past_key_values_length = past_key_values[0][0].shape[2]

                assert causal_mask_4d.shape[-1] == (past_key_values_length + 1)

            if self.config.backbone_model == "llama":
                outputs: BaseModelOutputWithPast = self.model(
                    position_ids=position_ids,
                    cache_position=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask_4d,
                    use_cache=True,
                    output_attentions=False,
                    # return_dict=True,  # Add this to ensure returns dict with past_key_values
                )
            else:
                raise ValueError(f"Unsupported backbone model: {self.config.backbone_model}")

            del position_ids
            del inputs_embeds

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

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
            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens

                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

                del logits_token

            if t < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits
            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)

            del scores

            idx_next = idx_next.view(-1, self.num_vq)
            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

        genrated_input_ids = new_tokens[:, 0:t, :]

        return MiniCPMTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=None,  # for update purpose
            past_key_values=None,  # for update purpose
            past_input_ids=None,  # for update purpose
            finished=finish.all(),
        )

    # non-streaming, interleave
    @torch.inference_mode()
    def generate_chunk(
        self,
        inputs_embeds: torch.Tensor,
        temperature: torch.Tensor,
        repetition_penalty: float,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        max_new_token=500,
        min_new_tokens=0,
        past_key_values=None,
        logits_processors=None,
        text_start_pos=None,
    ):
        """For inputs_embeds, it should be like [bs=1, seq_len, hidden_dim], its content is like:
        |Text BOS|Spk embeds|Text-Hidden states Interleave (if applicable)|Audio BOS|
        where the last position is the audio BOS token.
        So, the first iteration in generation directly forward the model with inputs_embeds, and
        the last hidden states of the last position (Audio BOS) will be decoded to get the first audio token.
        """
        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens, repetition_penalty=repetition_penalty
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        condition_length = inputs_embeds.shape[1]

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        for t in range(max_new_token):
            audio_bos = False

            # If this is the first audio token, the case is special
            if t == 0:
                audio_bos = True
                inputs_embeds_ = inputs_embeds
                position_ids = torch.tensor(
                    list(range(text_start_pos, text_start_pos + condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
            else:
                # Generate the following audio tokens, it is applicable to all other cases, including second and the following calling of `generate`
                inputs_embeds_ = self.emb_code[0](new_tokens[:, t - 1 : t, 0])

                position_ids = torch.tensor(
                    [text_start_pos + condition_length + t - 1],  # 把上一个token prefill进去
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

            outputs: BaseModelOutputWithPast = self.model(
                position_ids=position_ids,
                # cache_position=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds_,
                use_cache=True,
                output_attentions=False,
                # return_dict=True,  # Add this to ensure returns dict with past_key_values
            )

            del position_ids
            del inputs_embeds_

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

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

            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens

                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                del logits_token

            if force_no_stop or t < min_new_tokens:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)
            del logits

            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)
            del scores

            idx_next = idx_next.view(-1, self.num_vq)

            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

        if not finish.all():
            logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

        # 最新生成的那个token不在此次返回的范围内，如果是eos token，不返回，如果是其他正常的token，也不返回。正常！
        genrated_input_ids = new_tokens[:, 0:t, :]

        return genrated_input_ids, past_key_values

    @torch.inference_mode()
    def interleaved_generate(
        self,
        spk_embeds: torch.Tensor,
        conditions: List[torch.Tensor],
        temperature: torch.Tensor,
        repetition_penalty: float,
        eos_token: Union[int, torch.Tensor],
        **kwargs,
    ):
        """
        For inputs_embeds, it should be like [bs=1, seq_len, hidden_dim], its content is like:
        |Text BOS|Spk embeds|Text-Hidden states Interleave (if applicable)|Audio BOS|
        where the last position is the audio BOS token.
        So, the first iteration in generation directly forward the model with inputs_embeds, and the last hidden states of the last position (Audio BOS) will be decoded to get the first audio token.
        """
        temperature = torch.tensor([temperature], dtype=torch.float, device=self.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=repetition_penalty,
        )

        eos_token = eos_token.to(conditions[0].device)

        num_chunks = len(conditions)
        text_start_pos = 0
        last_window_size = 0
        past_key_values = None

        for idx in range(num_chunks):
            condition = conditions[idx].to(conditions[0].device)
            if self.attention_type == "sliding_recompute":
                recomputed_conditions = []

                if (
                    idx >= self.window_size
                    and (idx - self.recomputed_chunks) % (self.window_size - self.recomputed_chunks) == 0
                ):
                    for i in range(self.recomputed_chunks):
                        recomputed_conditions.append(conditions[idx - self.recomputed_chunks + i])
                        recomputed_conditions.append(
                            self.emb_code[0](generated_tokens[-self.recomputed_chunks + i][:, :, 0])
                        )
                    recomputed_conditions.append(condition)
                    condition = torch.cat(recomputed_conditions, dim=1)

                    text_start_pos = 0
                    new_tokens, old_kv = self.generate_chunk(
                        inputs_embeds=condition,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        eos_token=eos_token,
                        force_no_stop=False,
                        max_new_token=500,
                        past_key_values=None,
                        logits_processors=logits_processors,
                        text_start_pos=text_start_pos,
                    )

                else:
                    new_tokens, old_kv = self.generate_chunk(
                        inputs_embeds=condition,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        eos_token=eos_token,
                        force_no_stop=False,
                        max_new_token=500,
                        past_key_values=past_key_values,
                        logits_processors=logits_processors,
                        text_start_pos=text_start_pos,
                    )
            else:
                new_tokens, old_kv = self.generate_chunk(
                    inputs_embeds=condition,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    eos_token=eos_token,
                    force_no_stop=False,
                    max_new_token=500,
                    past_key_values=past_key_values,
                    logits_processors=logits_processors,
                    text_start_pos=text_start_pos,
                )

            past_key_values = []
            if self.attention_type == "sliding_window" and idx >= 1:
                for layer_idx in range(len(old_kv)):
                    past_key_values.append(
                        (
                            old_kv[layer_idx][0][:, :, last_window_size:, :],
                            old_kv[layer_idx][1][:, :, last_window_size:, :],
                        )
                    )
            else:
                past_key_values = old_kv

            last_window_size = condition.shape[1] + new_tokens.shape[1]
            text_start_pos += last_window_size

            if idx == 0:
                generated_tokens = [new_tokens]
            else:
                generated_tokens.append(new_tokens)

        return MiniCPMTTSGenerationOutput(new_ids=torch.cat(generated_tokens, dim=1), finished=True)


class CustomRepetitionPenaltyLogitsProcessorRepeat:
    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        if freq.size(0) > self.max_input_ids:
            freq.narrow(0, self.max_input_ids, freq.size(0) - self.max_input_ids).zero_()
        alpha = torch.pow(self.penalty, freq)
        scores = scores.contiguous()
        inp = scores.multiply(alpha)
        oth = scores.divide(alpha)
        con = scores < 0
        out = torch.where(con, inp, oth)
        del inp, oth, scores, con, alpha
        return out


def gen_logits(num_code: int, top_p=0.7, top_k=20, repetition_penalty=1.0):
    logits_warpers = []

    if top_p is not None:
        logits_warpers.append(TopPLogitsWarper(top_p, min_tokens_to_keep=3))

    if top_k is not None:
        logits_warpers.append(TopKLogitsWarper(top_k, min_tokens_to_keep=3))

    logits_processors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        logits_processors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16))

    return logits_warpers, logits_processors
