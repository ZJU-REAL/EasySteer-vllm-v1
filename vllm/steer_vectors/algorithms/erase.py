# SPDX-License-Identifier: Apache-2.0

import os

import torch

from .factory import register_algorithm
from .loading import read_gguf_directions, require_extension
from .template import AlgorithmTemplate


@register_algorithm("erase")
class EraseAlgorithm(AlgorithmTemplate):
    """Erase: h' = h - proj_{h1}(h) = h - (h · h1 / ||h1||²) * h1.

    Removes the component of the hidden state along direction h1,
    leaving the orthogonal complement. Payload: a single direction
    tensor per layer (GGUF only).
    """

    def _transform(
        self, hidden_state: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        h1 = params
        if h1.dim() == 1:
            h1 = h1.unsqueeze(0)  # [1, hidden_dim]

        h1_norm_sq = torch.sum(h1 * h1, dim=-1, keepdim=True)  # [1, 1]
        dot_product = torch.sum(hidden_state * h1, dim=-1, keepdim=True)  # [batch, 1]
        proj_scalar = dot_product / (h1_norm_sq + 1e-8)
        h_perp = hidden_state - proj_scalar * h1

        if self.normalize:
            return self._renormalize(hidden_state, h_perp)
        return h_perp

    @classmethod
    def load_from_path(
        cls,
        path: str,
        device: str,
        *,
        config,
        target_layers: list[int] | None = None,
    ) -> dict:
        require_extension(path, ".gguf", cls.__name__)
        return {
            "layer_payloads": read_gguf_directions(path, device, config.adapter_dtype)
        }
