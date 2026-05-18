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

"""Small tensor helpers shared by model runtime modules."""

import torch


def torch_clone_recursive(obj):
    """Recursively clone nested containers of torch.Tensors.

    Supported container types: dict, list, tuple. Non-container non-Tensor
    objects are returned as-is.
    """
    if torch.is_tensor(obj):
        return obj.clone()
    elif isinstance(obj, dict):
        return {k: torch_clone_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [torch_clone_recursive(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(torch_clone_recursive(v) for v in obj)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    return torch.cat((-x2, x1), dim=-1)
