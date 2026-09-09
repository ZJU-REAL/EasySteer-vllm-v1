# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from .base import BaseSteerVectorAlgorithm, wire_tensor_rank
from .registry import register_algorithm


@register_algorithm("lm_steer")
class LMSteerAlgorithm(BaseSteerVectorAlgorithm):
    """LM-Steer: h' = h + α * ((h @ P1) @ P2^T).

    Payload: canonical LowRankProjector 'projector1' and 'projector2'
    matrices of shape [hidden_size, rank] per layer.
    """

    graph_family = "lowrank"

    @staticmethod
    def graph_lower(payload, scale):
        # The lowrank family's bias vanishes: delta = (xP1)(αP2)^T.
        p1 = payload["projector1"]
        p2 = payload["projector2"]
        return {"A": p1, "Rout": p2 * scale}

    @staticmethod
    def wire_rank(wire):
        return wire_tensor_rank(wire, "projector1")

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        P1 = params["projector1"]
        P2 = params["projector2"]
        scale_factor = params.get("scale_factor", 1.0)

        device = hidden_state.device
        dtype = hidden_state.dtype
        P1 = P1.to(device).to(dtype)
        P2 = P2.to(device).to(dtype)

        transformed = torch.matmul(hidden_state, P1)  # [..., rank]
        transformed = torch.matmul(
            transformed, P2.transpose(-2, -1)
        )  # [..., hidden_dim]
        return hidden_state + scale_factor * transformed
