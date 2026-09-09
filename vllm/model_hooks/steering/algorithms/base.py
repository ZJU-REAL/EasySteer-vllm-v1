# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The single base class every steering algorithm extends.

An algorithm is a pure transformation over selected token rows: it
provides `_transform` (the core math) over canonical payloads. Controllers
route pre-resolved token positions independently of the algorithm.

Full-graph (Tier-1) support is declared on the class: `graph_family`
names a kernel family in graph.kernels.GRAPH_FAMILIES (None =
split-tier only, rejected at admission under graph_mode=in_graph), and
`graph_lower` maps this algorithm's (payload, scale) onto that family's
slot tensors — colocated with the eager `_transform` whose math it must
reproduce.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch


def wire_tensor_rank(wire, field: str) -> int | None:
    """Rank (second dim) of a named 2-D tensor in an inline wire payload."""
    if not isinstance(wire, dict):
        return None
    tensor = wire.get("tensors", {}).get(field)
    if isinstance(tensor, dict):
        shape = tensor.get("shape")
        if isinstance(shape, list) and len(shape) == 2:
            return int(shape[1])
    return None


class BaseSteerVectorAlgorithm(ABC):
    """Base class for steering algorithms (see module docstring)."""

    graph_family: str | None = None

    @classmethod
    def graph_payload_problem(cls, request=None) -> str | None:
        """Payload restriction, or its condition when only a name is known."""
        return None

    @staticmethod
    def graph_lower(payload: Any, scale: float) -> dict[str, Any]:
        """Lower one layer's (payload, scale) to family slot tensors.

        Returns {table_key: tensor-or-None}; None entries keep the
        table's zero default (e.g. an absent bias).
        """
        raise NotImplementedError

    @staticmethod
    def wire_rank(wire) -> int | None:
        """Rank of an inline wire payload (low-rank families only)."""
        return None

    def __init__(self, *, normalize: bool = False):
        # Payload of any type (Tensor, dict, ...); format is defined by
        # the canonical payload kind and consumed by _transform.
        self.payload: Any | None = None

        # Rescale transformed rows back to their original norm
        # (see _renormalize); honored by the dense-vector algorithms.
        self.normalize = normalize

    def set_payload(self, payload: Any, scale_factor: float = 1.0) -> None:
        """Set a private scaled payload without mutating cached source tensors."""
        if payload is None:
            raise ValueError(f"{self.__class__.__name__} requires a payload")

        if isinstance(payload, torch.Tensor):
            payload = payload * scale_factor
        elif isinstance(payload, dict):
            payload = {**payload, "scale_factor": scale_factor}

        self.payload = payload

    def _is_valid(self, params: Any) -> bool:
        """Check params is not None (rarely overridden)."""
        return params is not None

    @abstractmethod
    def _transform(self, hidden_state: torch.Tensor, params: Any) -> torch.Tensor:
        """Transform the selected token rows (the algorithm's core math).

        Args:
            hidden_state: [num_positions, hidden_dim] selected rows
            params: this intervention's scaled payload
        """
        pass

    def _renormalize(
        self, original: torch.Tensor, transformed: torch.Tensor
    ) -> torch.Tensor:
        """Rescale `transformed` rows to the norms of `original` rows.

        Computed in float32: hidden-state norms can reach ~1e4, so the
        intermediate product overflows float16 (max 65504).
        """
        norm_pre = torch.norm(original, dim=-1, keepdim=True).float()
        norm_post = torch.norm(transformed, dim=-1, keepdim=True).float()
        scaled = transformed.float() * norm_pre / (norm_post + 1e-8)
        return scaled.to(original.dtype)

    def _batch_transform_tensor(
        self, hidden_states, positions_tensor, params, residual=None
    ):
        """
        Apply transformation using position tensor.

        Performs direct tensor operations without GPU-CPU synchronization.

        When `residual` is given, the transform sees the complete hidden
        state (hidden + residual) of the selected rows, but only `hidden`
        is written back — in delta form, so identity transforms leave
        `hidden` bit-exact and the residual stream flows on untouched.

        Args:
            hidden_states: [total_tokens, hidden_dim]
            positions_tensor: [num_positions] GPU tensor of indices
            params: Algorithm parameters
            residual: optional [total_tokens, hidden_dim] residual stream

        Returns:
            hidden_states: Transformed hidden states
        """
        original_dtype = hidden_states.dtype

        selected = hidden_states.index_select(0, positions_tensor)
        if residual is not None:
            complete = selected + residual.index_select(0, positions_tensor)
        else:
            complete = selected

        transformed = self._transform(complete, params).to(original_dtype)
        if residual is not None:
            transformed = selected + (transformed - complete)

        hidden_states.index_copy_(0, positions_tensor, transformed)

        return hidden_states
