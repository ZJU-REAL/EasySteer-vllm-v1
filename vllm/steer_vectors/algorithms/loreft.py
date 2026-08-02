# SPDX-License-Identifier: Apache-2.0
import torch

from .factory import register_algorithm
from .loading import find_reft_checkpoint
from .template import AlgorithmTemplate


@register_algorithm("loreft")
class LoReFTAlgorithm(AlgorithmTemplate):
    """LoReFT: h' = h + R^T(Wh + b - Rh).

    Payload: dict with 'rotate_layer' (R), 'learned_source_weight' (W)
    and optional 'learned_source_bias' (b) per layer, loaded from a
    pyreft checkpoint directory.
    """

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

    @classmethod
    def load_from_path(
        cls,
        path: str,
        device: str,
        *,
        config,
        target_layers: list[int] | None = None,
        **kwargs,
    ) -> dict:
        bin_file_path, layer_idx = find_reft_checkpoint(path, target_layers)
        state_dict = torch.load(bin_file_path, map_location=device)

        dtype = config.adapter_dtype
        rotate_layer, learned_source_weight, learned_source_bias = None, None, None
        for key, value in state_dict.items():
            if "rotate_layer" in key:
                if "parametrizations.weight.original" in key or key.endswith(
                    "rotate_layer"
                ):
                    rotate_layer = value.to(dtype)
            elif "learned_source" in key:
                if key.endswith("weight") and "parametrizations" not in key:
                    learned_source_weight = value.to(dtype)
                elif key.endswith("bias"):
                    learned_source_bias = value.to(dtype)
            elif key == "weight":
                learned_source_weight = value.to(dtype)
            elif key == "bias":
                learned_source_bias = value.to(dtype)

        if rotate_layer is None or learned_source_weight is None:
            raise ValueError(
                f"Could not find all required LoReFT params in {bin_file_path}. Keys: "
                f"{list(state_dict.keys())}"
            )

        return {
            "layer_payloads": {
                layer_idx: {
                    "rotate_layer": rotate_layer,
                    "learned_source_weight": learned_source_weight,
                    "learned_source_bias": learned_source_bias,
                }
            }
        }
