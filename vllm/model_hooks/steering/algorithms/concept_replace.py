# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-preserving concept substitution.

Replaces one concept (h1) with another (h2) in the hidden state, scaled
by the projection strength:

    λ = h · h1 / ||h1||²           # projection coefficient of h onto h1
    h_new = h + λ(h2 - h1)         # = (h - λ·h1) + λ·h2

The amount of h2 injected equals the amount of h1 erased, preventing
hallucination from over-modification and coverage failures from
under-modification.
"""

import torch

from .base import BaseSteerVectorAlgorithm
from .registry import register_algorithm


@register_algorithm("concept_replace")
class ConceptReplaceAlgorithm(BaseSteerVectorAlgorithm):
    """Concept Replace: h_new = h + λ(h2 - h1) with λ = (h·h1)/||h1||².

    Payload: dict with explicitly named 'h1' and 'h2' tensors per layer.
    """

    graph_family = "projection"

    @staticmethod
    def graph_lower(payload, scale):
        # Like _transform, the substitution ignores the scale.
        h1 = payload["h1"].to(torch.float32).reshape(-1)
        h2 = payload["h2"].to(torch.float32).reshape(-1)
        return {"B": h1 / (h1.pow(2).sum() + 1e-8), "C": h2 - h1}

    def _transform(
        self, hidden_state: torch.Tensor, params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        h1 = params["h1"]
        h2 = params["h2"]
        if h1.dim() == 1:
            h1 = h1.unsqueeze(0)  # [1, hidden_dim]
        if h2.dim() == 1:
            h2 = h2.unsqueeze(0)

        h1_norm_sq = torch.sum(h1 * h1, dim=-1, keepdim=True)  # [1, 1]
        dot_product = torch.sum(hidden_state * h1, dim=-1, keepdim=True)  # [batch, 1]
        lambda_coef = dot_product / (h1_norm_sq + 1e-8)
        h_new = hidden_state + lambda_coef * (h2 - h1)

        if self.normalize:
            return self._renormalize(hidden_state, h_new)
        return h_new
