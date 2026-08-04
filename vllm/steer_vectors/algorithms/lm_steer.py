# SPDX-License-Identifier: Apache-2.0

import torch

from .base import wire_tensor_rank
from .factory import register_algorithm
from .template import AlgorithmTemplate


@register_algorithm("lm_steer")
class LMSteerAlgorithm(AlgorithmTemplate):
    """LM-Steer: h' = h + α * ((h @ P1) @ P2^T).

    Payload: dict with 'projector1' and 'projector2' low-rank projection
    matrices per layer, loaded from a .pt file.
    """

    graph_family = "lowrank"

    @staticmethod
    def graph_lower(payload, scale):
        # The lowrank family's Rin/b terms vanish: delta = (xP1)(αP2)^T.
        p1 = payload["projector1"]
        p2 = payload["projector2"]
        if p1.dim() > 2:
            p1 = p1[0]
        if p2.dim() > 2:
            p2 = p2[0]
        return {"A": p1, "Rout": p2 * scale}

    @staticmethod
    def wire_rank(wire):
        return wire_tensor_rank(wire, "projector1")

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        P1 = params["projector1"]
        P2 = params["projector2"]
        scale_factor = params.get("scale_factor", 1.0)

        # Multi-vector checkpoints stack steer vectors; use the first.
        if P1.dim() > 2:
            P1 = P1[0]
        if P2.dim() > 2:
            P2 = P2[0]

        device = hidden_state.device
        dtype = hidden_state.dtype
        P1 = P1.to(device).to(dtype)
        P2 = P2.to(device).to(dtype)

        transformed = torch.matmul(hidden_state, P1)  # [..., rank]
        transformed = torch.matmul(
            transformed, P2.transpose(-2, -1)
        )  # [..., hidden_dim]
        return hidden_state + scale_factor * transformed
