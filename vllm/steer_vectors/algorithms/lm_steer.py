# SPDX-License-Identifier: Apache-2.0

import torch

from .factory import register_algorithm
from .template import AlgorithmTemplate


@register_algorithm("lm_steer")
class LMSteerAlgorithm(AlgorithmTemplate):
    """LM-Steer: h' = h + α * ((h @ P1) @ P2^T).

    Payload: dict with 'projector1' and 'projector2' low-rank projection
    matrices per layer, loaded from a .pt file.
    """

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
        if not target_layers:
            raise ValueError(
                "LMSteerAlgorithm requires target_layers: the .pt file holds "
                "one projector pair, applied to each listed layer"
            )

        # weights_only=False: checkpoints may embed argparse.Namespace etc.
        state_dict = torch.load(path, map_location=device, weights_only=False)

        # gpt2.pt-style checkpoints are a list with the params at index 1.
        if isinstance(state_dict, list) and len(state_dict) > 1:
            state_dict = state_dict[1]

        if not isinstance(state_dict, dict) or not (
            "projector1" in state_dict and "projector2" in state_dict
        ):
            raise ValueError(f"Projector matrices not found in pt file: {path}")

        projector1 = state_dict["projector1"].to(
            device=device, dtype=config.adapter_dtype
        )
        projector2 = state_dict["projector2"].to(
            device=device, dtype=config.adapter_dtype
        )

        payload = {"projector1": projector1, "projector2": projector2}
        return {"layer_payloads": {layer_idx: payload for layer_idx in target_layers}}
