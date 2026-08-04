# SPDX-License-Identifier: Apache-2.0
"""The single base class every steering algorithm extends.

An algorithm is a pure transformation over selected token rows: it
provides `_transform` (the core math) plus optional `load_from_path`
(engine-loaded file formats). Where a vector applies lives in
`self.clause` (see clause.py), fully decoupled from the math.

Full-graph (Tier-1) support is declared on the class: `graph_family`
names a kernel family in graph_kernels.GRAPH_FAMILIES (None =
piecewise only, rejected at admission under graph_mode=full), and
`graph_lower` maps this algorithm's (payload, scale) onto that family's
slot tensors — colocated with the eager `_transform` whose math it must
reproduce.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from .clause import ApplyClause


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

    def __init__(self, layer_id: int | None = None, normalize: bool = False, **kwargs):
        self.layer_id = layer_id
        self.clause = ApplyClause()

        # Payload of any type (Tensor, dict, ...); format is defined by
        # the algorithm's load_from_path and consumed by _transform.
        self._payload: Any | None = None

        # Rescale transformed rows back to their original norm
        # (see _renormalize); honored by the dense-vector algorithms.
        self.normalize = normalize

    @classmethod
    def load_from_path(
        cls,
        path: str,
        device: str,
        *,
        config,
        target_layers: list[int] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Load steer vector data from a source file.

        Only formats whose schema EasySteer itself defines are loaded
        engine-side (its GGUF export; the moe_router JSON config).
        Third-party checkpoint formats are interpreted client-side —
        load the file yourself (or use an ``easysteer.vectors``
        adapter) and pass ``VectorSpec(data=...)``.

        Args:
            path: Vector file or directory path.
            device: Device to load tensors on.
            config: The engine's SteerVectorConfig.
            target_layers: Optional layer restriction from the request.
            **kwargs: Algorithm-specific parameters (e.g. moe_mode).

        Returns:
            ``{"layer_payloads": {layer_idx: payload}}`` where each
            payload is whatever this algorithm's ``_transform`` consumes
            (tensor, dict, ...).
        """
        raise ValueError(
            f"algorithm {cls.__name__} loads no source files; pass its "
            "payload via VectorSpec(data=...) — see "
            "vllm.steer_vectors.payloads and easysteer.vectors"
        )

    def set_payload(self, payload: Any, scale_factor: float = 1.0) -> None:
        """Store this intervention's payload.

        Tensor payloads are pre-scaled and copied into an existing
        buffer when shapes match, so the tensor address stays constant
        across reloads (CUDA-graph replay safe). Dict payloads get the
        scale factor merged in; other types are stored as-is.
        """
        if payload is None:
            raise ValueError(f"{self.__class__.__name__} requires a payload")

        if isinstance(payload, torch.Tensor):
            scaled = payload * scale_factor
            existing = self._payload
            if (
                isinstance(existing, torch.Tensor)
                and existing.shape == scaled.shape
                and existing.dtype == scaled.dtype
                and existing.device == scaled.device
            ):
                existing.copy_(scaled)
                return  # buffer address unchanged
            payload = scaled.clone()
        elif isinstance(payload, dict):
            payload = {**payload, "scale_factor": scale_factor}

        self._payload = payload

    def _get_params(self) -> Any:
        """Return the payload as-is (rarely overridden)."""
        return self._payload

    def _is_valid(self, params: Any) -> bool:
        """Check params is not None (rarely overridden)."""
        return params is not None

    @abstractmethod
    def _transform(self, hidden_state: torch.Tensor, params: Any) -> torch.Tensor:
        """Transform the selected token rows (the algorithm's core math).

        Args:
            hidden_state: [num_positions, hidden_dim] selected rows
            params: this intervention's payload (from _get_params)
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

        # Select positions to transform
        selected = hidden_states.index_select(0, positions_tensor)
        if residual is not None:
            complete = selected + residual.index_select(0, positions_tensor)
        else:
            complete = selected

        # Apply transformation
        transformed = self._transform(complete, params).to(original_dtype)
        if residual is not None:
            transformed = selected + (transformed - complete)

        # Write back transformed values
        hidden_states.index_copy_(0, positions_tensor, transformed)

        return hidden_states
