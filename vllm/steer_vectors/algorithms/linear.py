# SPDX-License-Identifier: Apache-2.0

import pickle

import numpy as np
import torch

from .factory import register_algorithm
from .template import AlgorithmTemplate


@register_algorithm("linear")
class LinearTransformAlgorithm(AlgorithmTemplate):
    """Linear transformation: h' = W @ h + b.

    Payload: dict with 'weight' and optional 'bias' tensors per layer,
    loaded from a pickle file with A_ (weight) and B_ (bias) entries.
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
                "LinearTransformAlgorithm requires target_layers: the pkl file "
                "holds one transform, applied to each listed layer"
            )

        with open(path, "rb") as f:
            data = pickle.load(f)

        # LinearTransport objects expose A_/B_ attributes; plain dicts use keys.
        if isinstance(data, dict):
            weight = data.get("A_")
            bias = data.get("B_")
        else:
            weight = getattr(data, "A_", None)
            bias = getattr(data, "B_", None)

        if weight is None:
            raise ValueError(
                f"Weight matrix (A_) not found in pkl file (data type {type(data)})"
            )

        if not isinstance(weight, np.ndarray):
            weight = np.array(weight, dtype=np.float32)
        if bias is not None and not isinstance(bias, np.ndarray):
            bias = np.array(bias, dtype=np.float32)

        weight_tensor = torch.tensor(weight, device=device, dtype=config.adapter_dtype)
        bias_tensor = (
            torch.tensor(bias, device=device, dtype=config.adapter_dtype)
            if bias is not None
            else None
        )

        payload = {"weight": weight_tensor, "bias": bias_tensor}
        return {"layer_payloads": {layer_idx: payload for layer_idx in target_layers}}
