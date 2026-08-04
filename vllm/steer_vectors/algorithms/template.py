# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

import torch

from .base import BaseSteerVectorAlgorithm
from .triggers import TriggerController


class AlgorithmTemplate(BaseSteerVectorAlgorithm, ABC):
    """
    Steer vector algorithm template class.

    Algorithm implementations provide two methods: `_transform` (the
    core math over selected token rows) and `load_from_path` (file
    format -> per-layer payloads). Trigger state (where to apply) lives
    in `self.triggers`, fully decoupled from the algorithm logic;
    `_get_params`/`_is_valid` have sensible defaults and are rarely
    overridden.
    """

    def __init__(self, layer_id: int | None = None, normalize: bool = False, **kwargs):
        super().__init__(layer_id)
        self.triggers = TriggerController()

        # Payload of any type (Tensor, dict, ...); format is defined by
        # the algorithm's load_from_path and consumed by _transform.
        self._payload: Any | None = None

        # Rescale transformed rows back to their original norm
        # (see _renormalize); honored by the dense-vector algorithms.
        self.normalize = normalize

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
