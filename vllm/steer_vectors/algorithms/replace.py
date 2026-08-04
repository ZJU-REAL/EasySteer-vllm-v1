# SPDX-License-Identifier: Apache-2.0

import os

import torch

from .factory import register_algorithm
from .loading import read_gguf_directions, require_extension
from .base import BaseSteerVectorAlgorithm


@register_algorithm("replace")
class ReplaceAlgorithm(BaseSteerVectorAlgorithm):
    """Replace: h' = vector.

    Replaces the hidden state with the payload vector. Payload: a
    single tensor per layer (GGUF only).
    """

    graph_family = "replace"

    @staticmethod
    def graph_lower(payload, scale):
        return {"V": payload * scale}

    def _transform(
        self, hidden_state: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        if params.dim() == 1 and hidden_state.dim() == 2:
            replaced = params.unsqueeze(0).expand_as(hidden_state)
        else:
            replaced = params
        if self.normalize:
            return self._renormalize(hidden_state, replaced)
        return replaced

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
