# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from .base import BaseSteerVectorAlgorithm
from .registry import register_algorithm


@register_algorithm("linear")
class LinearTransformAlgorithm(BaseSteerVectorAlgorithm):
    """Linear transformation: h' = W @ h + b.

    Payload: canonical LinearMap 'weight' and optional 'bias' tensors per layer.
    """

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        weight = params["weight"]
        bias = params.get("bias")
        scale_factor = params.get("scale_factor", 1.0)

        device = hidden_state.device
        dtype = hidden_state.dtype
        weight = weight.to(device).to(dtype)

        transformed = torch.matmul(hidden_state, weight.T)
        if bias is not None:
            transformed = transformed + bias.to(device).to(dtype)
        if scale_factor != 1.0:
            transformed = transformed * scale_factor
        return transformed
