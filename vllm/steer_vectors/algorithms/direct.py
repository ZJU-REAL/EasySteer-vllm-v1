# SPDX-License-Identifier: Apache-2.0

import torch

from .factory import register_algorithm
from .loading import read_gguf_directions, require_extension
from .template import AlgorithmTemplate


@register_algorithm("direct")
class DirectAlgorithm(AlgorithmTemplate):
    """Direct addition: h' = h + vector.

    Payload: a single direction tensor per layer. Loads EasySteer's
    GGUF export; other formats pass through ``VectorSpec(data=...)``
    (see ``easysteer.vectors`` adapters).
    """

    graph_family = "additive"

    @staticmethod
    def graph_lower(payload, scale):
        return {"V": payload * scale}

    def _transform(
        self, hidden_state: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        transformed = hidden_state + params
        if self.normalize:
            return self._renormalize(hidden_state, transformed)
        return transformed

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
