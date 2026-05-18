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

"""Chunked text prefill/generation helpers."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .sampling import _validate_sampling_probs


# text
@dataclass
class GenerateChunkOutput:
    chunk_token_ids: torch.Tensor
    current_inputs_embeds: torch.Tensor
    input_last_hidden_states: Optional[torch.Tensor]  # for tts use_speaker_embedding
    last_hidden_states: Optional[torch.Tensor]  # for tts input feature (projector_semantic)
    past_key_values: Optional[torch.Tensor]
    finished: bool


class ChunkPrefillChunkGenerate:
    def __init__(self, model, tokenizer, terminators):
        self.tokenizer = tokenizer
        self.model = model
        self.terminators = terminators
        self.terminators_ids = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        self.embedding_layer = self.model.get_input_embeddings()

        self.forbidden_tokens = [
            ":",
            "：",
            "；",
            "#",
            "“",
            "”",
            "‘",
            "’",
            "@",
            "*",
            "【",
            "】",
            "「",
            "」",
            "(",
            ")",
            "（",
            "）",
            "[",
            "]",
            "&",
            "/",
            "$",
        ]

        self.forbidden_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in self.forbidden_tokens]
        bad_token_ids = getattr(tokenizer, "bad_token_ids", [])
        if bad_token_ids:
            self.forbidden_token_ids.extend(bad_token_ids)

    @staticmethod
    def prepare_generation_config(do_sample, max_new_tokens=50, min_new_tokens=0, **kwargs):
        num_beams = kwargs.get("num_beams", 3)
        generation_config = {
            "num_beams": num_beams,
            "top_p": 0.8,
            "top_k": 100,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.05,
        }

        if do_sample:
            generation_config.update(
                {
                    "top_p": 0.8,
                    "top_k": 100,
                    "temperature": 0.7,
                    "do_sample": True,
                    "repetition_penalty": 1.05,
                }
            )
        elif num_beams > 1:
            generation_config.update({"num_beams": num_beams, "repetition_penalty": 1.2, "do_sample": False})
        else:
            generation_config.update({"do_sample": False, "repetition_penalty": 1.05})

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())
        generation_config["min_new_tokens"] = min_new_tokens
        generation_config["max_new_tokens"] = max_new_tokens

        return generation_config

    def chunk_generate(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values,
        is_first_generate_chunk: bool,
        chunk_size: int,
        return_hidden_states: bool,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float = 1.05,
        length_penalty: float = 1.0,
        all_input_ids: Optional[torch.Tensor] = None,
        suppress_forbidden_tokens: bool = True,
    ) -> GenerateChunkOutput:
        """
        Args:
            inputs_embeds: [1, seq_len, hidden_dim], Input embeddings of current chunk.
            past_key_values: [num_layers, 2, batch_size, num_heads, seq_len, head_dim], Past key values for llm.
            is_first_generate_chunk: bool, Whether this is the first generate chunk.
            chunk_size: int, The size of the current chunk, default is 10, and it is fixed during training.
            return_hidden_states: bool Whether to return the hidden states, default is True.
            do_sample: bool Whether to sample from the model, default is True.
            temperature: float The temperature for the model, default is 0.7.
            top_p: float The top-p for the model, default is 0.8.
            top_k: int The top-k for the model, default is 100.
            repetition_penalty: float, The repetition penalty for the model, default is 1.05.
            length_penalty: float, The length penalty for the model, default is 1.0. Higher value means more detailed generation.
            all_input_ids: Optional[torch.Tensor], The input ids for the current chunk.
        """

        finished = False
        current_inputs_embeds = inputs_embeds.clone()
        input_last_hidden_states = []
        last_hidden_states = []
        generated_tokens = []

        for token_idx in range(chunk_size):
            if is_first_generate_chunk and token_idx == 0:
                # first generate chunk, prefill inputs_embeds
                model_inputs = {
                    "inputs_embeds": current_inputs_embeds,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "output_hidden_states": return_hidden_states,
                }
            else:  # for all other cases: prefill the latest generated token
                model_inputs = {
                    "inputs_embeds": current_inputs_embeds[:, -1:, :],
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "output_hidden_states": return_hidden_states,
                }

            with torch.no_grad():
                outputs = self.model(**model_inputs)

            # last token's logits
            logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=inputs_embeds.device)

            # forbid specific tokens decoding = model.generate@suppress_tokens
            if suppress_forbidden_tokens and self.forbidden_token_ids:
                logits[:, self.forbidden_token_ids] = float("-inf")

            past_key_values = outputs.past_key_values

            PENALTY_WINDOW_SIZE = 128

            # apply repetition penalty
            if repetition_penalty != 1.0:
                # get token ids for repetition penalty
                if all_input_ids is not None:
                    # use global input ids (including original input and generated part)
                    if len(generated_tokens) > 0:
                        generated_token_ids = torch.cat(generated_tokens, dim=1)
                        current_sequence = torch.cat(
                            [
                                all_input_ids[:, -PENALTY_WINDOW_SIZE:],
                                generated_token_ids,
                            ],
                            dim=1,
                        )
                    else:
                        current_sequence = all_input_ids[:, -PENALTY_WINDOW_SIZE:]
                    unique_token_ids = torch.unique(current_sequence.squeeze(0))
                elif len(generated_tokens) > 0:
                    # revert to original logic: only use generated tokens
                    generated_token_ids = torch.cat(generated_tokens, dim=1).squeeze(0)
                    unique_token_ids = torch.unique(generated_token_ids)
                else:
                    unique_token_ids = torch.tensor([], dtype=torch.long, device=logits.device)

                # apply repetition penalty
                for token_id in unique_token_ids:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] = logits[0, token_id] / repetition_penalty
                    else:
                        logits[0, token_id] = logits[0, token_id] * repetition_penalty

            # apply length penalty, higher value means more detailed generation
            if length_penalty != 1.0:
                for eos_token_id in self.terminators_ids:
                    if logits[0, eos_token_id] > 0:
                        logits[0, eos_token_id] = logits[0, eos_token_id] / length_penalty
                    else:
                        logits[0, eos_token_id] = logits[0, eos_token_id] * length_penalty

            # apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits_filtered = torch.full_like(logits, float("-inf"))
                    logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                    logits = logits_filtered

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # remove tokens with cumulative probability greater than top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")

                # sampling
                probs = F.softmax(logits, dim=-1)
                _validate_sampling_probs(probs, context="ChunkPrefillChunkGenerate.generate.sample")
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if return_hidden_states:
                if is_first_generate_chunk and token_idx == 0:
                    input_last_hidden_states.append(outputs.hidden_states[-1])
                else:
                    last_hidden_states.append(outputs.hidden_states[-1])

            # if terminator token, stop generating
            if next_token.item() in self.terminators_ids:
                finished = True
                break

            generated_tokens.append(next_token)

            # convert new token to embeddings and concatenate
            next_token_embed = self.embedding_layer(next_token)

            # update inputs_embeds, add one
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embed], dim=1)

        if len(generated_tokens) > 0:
            chunk_token_ids = torch.cat(generated_tokens, dim=1)
        else:
            # special case: if last chunk and first predict is eos token, return last token of previous chunk. return a tensor with shape (1, 0)
            if finished:
                chunk_token_ids = torch.zeros((1, 0), dtype=torch.long, device=current_inputs_embeds.device)
            else:
                raise Exception("this should not happen")

        if len(last_hidden_states) > 0:
            last_hidden_states = torch.cat(last_hidden_states, dim=1)
        else:
            # special case: if last chunk, return last token of previous chunk.
            if finished:
                last_hidden_states = torch.cat(last_hidden_states, dim=1)
            else:
                raise Exception("this should not happen")

        if len(input_last_hidden_states) > 0:
            input_last_hidden_states = torch.cat(input_last_hidden_states, dim=1)
        else:
            input_last_hidden_states = None

        return GenerateChunkOutput(
            chunk_token_ids=chunk_token_ids,
            current_inputs_embeds=current_inputs_embeds,
            input_last_hidden_states=input_last_hidden_states,
            last_hidden_states=last_hidden_states,
            past_key_values=past_key_values,
            finished=finished,
        )


def streaming_token_decoder(token_iterator, tokenizer, skip_special_tokens=False):
    """
    Incrementally decode tokens from an iterator, handling partial multi-byte characters.

    When streaming tokens, multi-byte characters (like Chinese) may be split across multiple
    tokens. Decoding partial tokens results in replacement characters (U+FFFD). This function
    buffers tokens and only yields complete characters.

    Args:
        token_iterator: An iterator yielding (token_ids, is_finished) tuples.
                       token_ids can be torch.Tensor or any iterable of integers.
        tokenizer: The tokenizer to use for decoding.
        skip_special_tokens: Whether to skip special tokens during decoding.

    Yields:
        (decoded_text, is_finished) tuples where decoded_text is the new text since last yield.
    """
    accumulated_token_ids = []
    yielded_text_len = 0

    for token_ids, is_finished in token_iterator:
        # Accumulate token IDs
        if torch.is_tensor(token_ids):
            accumulated_token_ids.extend(token_ids.reshape(-1).tolist())
        else:
            accumulated_token_ids.extend(list(token_ids) if hasattr(token_ids, "__iter__") else [token_ids])

        # Decode all accumulated tokens
        full_decoded = tokenizer.decode(accumulated_token_ids, skip_special_tokens=skip_special_tokens)

        if is_finished:
            # Final chunk - yield all remaining text
            new_text = full_decoded[yielded_text_len:]
            yield new_text, is_finished
        else:
            # Find safe prefix without incomplete multi-byte characters
            # The replacement character '�' (U+FFFD) indicates incomplete decoding
            new_text = full_decoded[yielded_text_len:]

            # Hold back text ending with replacement character (incomplete UTF-8 sequence)
            safe_end = len(new_text)
            while safe_end > 0 and new_text[safe_end - 1] == "\ufffd":
                safe_end -= 1

            safe_text = new_text[:safe_end] if safe_end > 0 else ""
            yielded_text_len += len(safe_text)
            yield safe_text, is_finished

