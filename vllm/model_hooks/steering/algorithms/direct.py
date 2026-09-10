# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from .base import BaseSteerVectorAlgorithm
from .registry import register_algorithm


@register_algorithm("direct")
class DirectAlgorithm(BaseSteerVectorAlgorithm):
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


@register_algorithm("attention_add")
class AttentionAddAlgorithm(DirectAlgorithm):
    """Add a concatenated head direction before the attention output projection."""
