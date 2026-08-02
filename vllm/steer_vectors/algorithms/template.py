# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any

import torch

from .base import BaseSteerVectorAlgorithm
from .triggers import TriggerController
from vllm.steer_vectors.discovery import extract_samples_info

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


class AlgorithmTemplate(BaseSteerVectorAlgorithm, ABC):
    """
    Steer vector algorithm template class.

    Provides a clean template for implementing new algorithms. Algorithm developers
    only need to focus on 3 core methods:

    1. _get_params(): Return algorithm parameters (vectors, matrices, etc.)
    2. _is_valid(): Check if parameters are valid
    3. _transform(): Core transformation logic

    Parameter management (triggers, exclusions, etc.) is handled by
    TriggerController,
    completely decoupled from algorithm logic.
    """

    def __init__(self, layer_id: int | None = None, normalize: bool = False, **kwargs):
        super().__init__(layer_id)
        # Intervention parameters - directly exposed for clean access
        self.params = TriggerController()

        # Payload of any type (Tensor, dict, ...); format is defined by
        # the algorithm's load_from_path and consumed by _transform.
        self._payload: Any | None = None

        self.normalize = normalize  # Direct algorithm uses this

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
        """
        Transform hidden state (MUST be implemented by subclass).

        This is the core logic of your algorithm - the only truly required method.
        """
        pass

    def _get_forward_context_and_samples(self, hidden_states: torch.Tensor):
        """
        Get forward context and sample information.

        This is a shared helper that extracts forward context, current tokens,
        and sample boundaries - common operations for both single-vector and
        multi-vector interventions.

        Args:
            hidden_states: [total_tokens, hidden_dim]

        Returns:
            tuple: (forward_ctx, samples_info, current_tokens) or None if unavailable
        """
        # Get forward context
        if get_forward_context is None:
            return None

        forward_ctx = get_forward_context()
        if forward_ctx is None:
            return None

        current_tokens = forward_ctx.current_tokens
        attn_metadata = forward_ctx.attn_metadata

        if current_tokens is None or attn_metadata is None:
            return None

        # Flatten tokens if needed
        if current_tokens.dim() == 2:
            current_tokens = current_tokens.flatten()

        # Extract sample boundaries using GPU batch operations
        samples_info = extract_samples_info(attn_metadata)

        if samples_info is None:
            # In vLLM V1, query_start_loc should always be available
            raise RuntimeError(
                "Cannot extract sample information from attention metadata. "
                "This should not happen in vLLM V1 with standard attention backends. "
                "Please report this issue with your configuration details."
            )

        return (forward_ctx, samples_info, current_tokens)

    def _batch_transform_tensor(self, hidden_states, positions_tensor, params):
        """
        Apply transformation using position tensor.

        Performs direct tensor operations without GPU-CPU synchronization.

        Args:
            hidden_states: [total_tokens, hidden_dim]
            positions_tensor: [num_positions] GPU tensor of indices
            params: Algorithm parameters

        Returns:
            hidden_states: Transformed hidden states
        """
        original_dtype = hidden_states.dtype

        # Select positions to transform
        selected = hidden_states.index_select(0, positions_tensor)

        # Apply transformation
        transformed = self._transform(selected, params).to(original_dtype)

        # Write back transformed values
        hidden_states.index_copy_(0, positions_tensor, transformed)

        return hidden_states
