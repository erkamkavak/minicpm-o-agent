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

"""Sampling filters and probability validation helpers."""

import torch
import torch.nn.functional as F


class InvalidSamplingProbabilitiesError(RuntimeError):
    """Raised before multinomial sampling when probabilities are unsafe."""


def _validate_sampling_probs(
    probs: torch.Tensor,
    *,
    context: str,
) -> None:
    """Fail before torch.multinomial can trigger a CUDA device-side assert."""
    invalid = (~torch.isfinite(probs)).any() | (probs < 0).any()
    if invalid.item():
        raise InvalidSamplingProbabilitiesError(
            f"{context}: invalid probabilities before multinomial "
            f"(shape={tuple(probs.shape)}, dtype={probs.dtype}, device={probs.device})"
        )


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    logits = logits.clone()

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # keep the first token that exceeds top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = filter_value

    return logits
