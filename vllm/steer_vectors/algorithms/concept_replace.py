# SPDX-License-Identifier: Apache-2.0
"""Projection-preserving concept substitution.

Replaces one concept (h1) with another (h2) in the hidden state, scaled
by the projection strength:

    λ = h · h1 / ||h1||²           # projection coefficient of h onto h1
    h_new = h + λ(h2 - h1)         # = (h - λ·h1) + λ·h2

The amount of h2 injected equals the amount of h1 erased, preventing
hallucination from over-modification and coverage failures from
under-modification.
"""

import glob
import os

import torch

from vllm.logger import init_logger

from .factory import register_algorithm
from .loading import read_gguf_directions
from .template import AlgorithmTemplate

logger = init_logger(__name__)


@register_algorithm("concept_replace")
class ConceptReplaceAlgorithm(AlgorithmTemplate):
    """Concept Replace: h_new = h + λ(h2 - h1) with λ = (h·h1)/||h1||².

    Payload: dict with 'h1' and 'h2' tensors per layer, loaded from a
    directory containing two .gguf files (h1.gguf/h2.gguf, *_h1/*_h2,
    or the first two alphabetically).
    """

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
        if not os.path.isdir(path):
            raise ValueError(
                f"ConceptReplaceAlgorithm requires a directory path, got: {path}"
            )

        gguf_files = sorted(glob.glob(os.path.join(path, "*.gguf")))
        if len(gguf_files) < 2:
            raise ValueError(
                "ConceptReplaceAlgorithm requires at least 2 .gguf files in "
                f"directory, found {len(gguf_files)}"
            )

        h1_path = None
        h2_path = None
        for f in gguf_files:
            basename = os.path.basename(f).lower()
            if basename == "h1.gguf" or "_h1" in basename:
                h1_path = f
            elif basename == "h2.gguf" or "_h2" in basename:
                h2_path = f
        # Fallback: first two files alphabetically.
        if h1_path is None:
            h1_path = gguf_files[0]
        if h2_path is None:
            h2_path = next(f for f in gguf_files if f != h1_path)

        logger.debug("Loading concept vectors: h1=%s, h2=%s", h1_path, h2_path)
        h1_weights = read_gguf_directions(h1_path, device, config.adapter_dtype)
        h2_weights = read_gguf_directions(h2_path, device, config.adapter_dtype)

        layer_payloads = {}
        for layer_idx in set(h1_weights) | set(h2_weights):
            if layer_idx not in h1_weights:
                raise ValueError(f"Layer {layer_idx} found in h2 but not in h1")
            if layer_idx not in h2_weights:
                raise ValueError(f"Layer {layer_idx} found in h1 but not in h2")
            layer_payloads[layer_idx] = {
                "h1": h1_weights[layer_idx],
                "h2": h2_weights[layer_idx],
            }
        return {"layer_payloads": layer_payloads}
