# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from .base import BaseSteerVectorAlgorithm
from .registry import register_algorithm


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
