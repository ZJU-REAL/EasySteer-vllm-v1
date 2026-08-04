# SPDX-License-Identifier: Apache-2.0
import torch

from .base import BaseSteerVectorAlgorithm, wire_tensor_rank
from .factory import register_algorithm


@register_algorithm("loreft")
class LoReFTAlgorithm(BaseSteerVectorAlgorithm):
    """LoReFT: h' = h + R^T(Wh + b - Rh).

    Payload: dict with 'rotate_layer' (R), 'learned_source_weight' (W)
    and optional 'learned_source_bias' (b) per layer, loaded from a
    pyreft checkpoint directory.
    """

    graph_family = "lowrank"

    @staticmethod
    def graph_lower(payload, scale):
        # delta = (xW^T + b - xR) @ (sR)^T = (x(W^T - R) + b) @ (sR)^T
        rotate = payload["rotate_layer"]
        weight = payload["learned_source_weight"]
        return {
            "A": weight.T - rotate,
            "Rout": rotate * scale,
            "b": payload.get("learned_source_bias"),
        }

    @staticmethod
    def wire_rank(wire):
        return wire_tensor_rank(wire, "rotate_layer")

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        rotate_layer = params["rotate_layer"]
        learned_source_weight = params["learned_source_weight"]
        learned_source_bias = params["learned_source_bias"]
        scale_factor = params.get("scale_factor", 1.0)

        device = hidden_state.device
        dtype = hidden_state.dtype
        rotate_layer = rotate_layer.to(device).to(dtype)
        learned_source_weight = learned_source_weight.to(device).to(dtype)
        if learned_source_bias is not None:
            learned_source_bias = learned_source_bias.to(device).to(dtype)

        rotated_base = torch.matmul(hidden_state, rotate_layer)  # Rh
        learned_output = torch.matmul(hidden_state, learned_source_weight.T)  # Wh
        if learned_source_bias is not None:
            learned_output = learned_output + learned_source_bias

        delta = (
            torch.matmul(learned_output - rotated_base, rotate_layer.T) * scale_factor
        )
        return hidden_state + delta
