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

"""Streaming-capable Whisper audio encoder wrappers."""

import logging
from typing import Optional

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

logger = logging.getLogger(__name__)


class MiniCPMWhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model
        try:
            # compatible old transformers
            from transformers.models.whisper.modeling_whisper import WHISPER_ATTENTION_CLASSES

            self.self_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
                layer_idx=layer_idx,
            )
        except:
            from transformers.models.whisper.modeling_whisper import WhisperAttention

            self.self_attn = WhisperAttention(
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
                layer_idx=layer_idx,
            )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


# Copied from from transformers.models.whisper.modeling_whisper.WhisperEncoder and add use_cache for streaming inference
class MiniCPMWhisperEncoder(WhisperEncoder):

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniCPMWhisperEncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers)]
        )

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        use_extra_context: Optional[bool] = False,
        prefix_extra_frames: Optional[int] = 1,
        suffix_extra_frames: Optional[int] = 1,
        return_debug: Optional[bool] = False,
        cnn_min_length: Optional[int] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 初始化debug字典
        debug_info = {} if return_debug else None

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        # 可选：将短输入pad到最小长度以确保CNN计算一致性
        original_length = input_features.shape[2]
        padded_for_cnn = False
        if cnn_min_length is not None and original_length < cnn_min_length:
            padded_features = torch.zeros(
                input_features.shape[0],
                input_features.shape[1],
                cnn_min_length,
                dtype=input_features.dtype,
                device=input_features.device,
            )
            padded_features[:, :, :original_length] = input_features
            input_features = padded_features
            padded_for_cnn = True
            if return_debug:
                debug_info["padded_for_cnn"] = True
                debug_info["original_length"] = original_length
                debug_info["padded_length"] = cnn_min_length

        if return_debug:
            debug_info["input_features"] = input_features.clone()

        conv1_output = self.conv1(input_features)
        if return_debug:
            debug_info["after_conv1"] = conv1_output.clone()

        inputs_embeds = nn.functional.gelu(conv1_output)
        if return_debug:
            debug_info["after_gelu1"] = inputs_embeds.clone()

        conv2_output = self.conv2(inputs_embeds)
        if return_debug:
            debug_info["after_conv2"] = conv2_output.clone()

        inputs_embeds = nn.functional.gelu(conv2_output)
        if return_debug:
            debug_info["after_gelu2"] = inputs_embeds.clone()

        # 如果之前进行了padding，现在需要移除padding的影响
        if padded_for_cnn:
            # Conv1: stride=1, 输出长度=输入长度
            # Conv2: stride=2, 输出长度=(输入长度+1)//2
            actual_cnn_output_length = (original_length + 1) // 2
            inputs_embeds = inputs_embeds[:, :, :actual_cnn_output_length]
            if return_debug:
                debug_info["after_unpad"] = inputs_embeds.clone()

        # 如果使用额外上下文，CNN操作后需要移除冗余帧
        # conv2的stride=2，所以输入的冗余帧在输出中会变成一半（向上取整）
        if use_extra_context:
            # 输入有prefix_extra_frames前缀帧和suffix_extra_frames后缀帧
            # conv2 stride=2，输出长度 = ceil(输入长度 / 2)
            # 对于2帧冗余，输出是1帧（ceil(2/2) = 1）
            prefix_to_remove = (prefix_extra_frames + 1) // 2 if prefix_extra_frames > 0 else 0  # 向上取整
            suffix_to_remove = (suffix_extra_frames + 1) // 2 if suffix_extra_frames > 0 else 0  # 向上取整

            # 移除前后冗余帧 (batch, channels, time)
            if prefix_to_remove > 0:
                inputs_embeds = inputs_embeds[:, :, prefix_to_remove:]
            if 0 < suffix_to_remove < inputs_embeds.shape[2]:
                inputs_embeds = inputs_embeds[:, :, :-suffix_to_remove]

        if return_debug and use_extra_context:
            debug_info["after_redundancy_removal"] = inputs_embeds.clone()

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        if return_debug:
            debug_info["after_permute"] = inputs_embeds.clone()

        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, list):
                past_key_values = EncoderDecoderCache(DynamicCache.from_legacy_cache(past_key_values), DynamicCache())
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            else:
                pass
            past_key_values_length = past_key_values.self_attention_cache.get_usable_length(inputs_embeds.shape[1])
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                logger.warning("seems the audio is longer than 30s. repeating the last part of the audio")
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1] - embed_pos.shape[0] + past_key_values_length,
                            dim=0,
                        ),
                    )
                )
            else:
                embed_pos = embed_pos[past_key_values_length : inputs_embeds.shape[1] + past_key_values_length, :]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        if return_debug:
            debug_info["positional_embedding"] = embed_pos.clone()
            debug_info["past_key_values_length"] = past_key_values_length

        hidden_states = inputs_embeds + embed_pos
        if return_debug:
            debug_info["after_add_pos_embed"] = hidden_states.clone()

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if return_debug:
            debug_info["after_dropout"] = hidden_states.clone()

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                        past_key_values,
                        use_cache,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )

                hidden_states = layer_outputs[0]

                if return_debug and idx < 3:  # 只记录前3层以节省内存
                    debug_info[f"after_layer_{idx}"] = hidden_states.clone()

            if use_cache:
                next_encoder_cache = layer_outputs[2 if output_attentions else 1]
            else:
                next_encoder_cache = None

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if return_debug:
            debug_info["after_layer_norm"] = hidden_states.clone()

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            result = tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
            if return_debug:
                return result, debug_info
            return result
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            past_key_values=next_encoder_cache,
        )

        if return_debug:
            return result, debug_info
        return result

