# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any
import torch
from abc import ABC, abstractmethod

from .base import BaseSteerVectorAlgorithm
from .utils import extract_samples_info
from .parameter_control import InterventionController

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
    
    Parameter management (triggers, exclusions, etc.) is handled by InterventionController,
    completely decoupled from algorithm logic.
    """
    
    def __init__(self, layer_id: Optional[int] = None, normalize: bool = False, **kwargs):
        super().__init__(layer_id)
        # Intervention parameters - directly exposed for clean access
        self.params = InterventionController()
        
        # Universal payload storage - can store ANY type (Tensor, dict, list, etc.)
        # Algorithms don't need to manage storage - just implement _transform and load_from_path
        self._payloads: dict[int, Any] = {}
        self._active_payload: Optional[Any] = None
        
        # Common parameters - all algorithms inherit these, but only use what they need
        self.normalize = normalize  # Direct algorithm uses this
        # Future common parameters can be added here:
        # self.clamp_range = kwargs.get('clamp_range', None)
        # self.dropout_rate = kwargs.get('dropout_rate', 0.0)

    def set_steer_vector(self, index: int, **kwargs) -> None:
        """
        Universal implementation: Store payload of any type.

        For Tensor payloads, uses an in-place buffer strategy so the tensor
        address stays constant across reloads.  This is critical for CUDA
        graph replay: the graph captures the buffer address, so ``copy_()``
        into the same buffer lets us change the scale at runtime without
        graph re-capture.

        Algorithms don't need to override this - just define what payload format
        they need in load_from_path, and use it in _transform.
        """
        payload = kwargs.get("payload")
        scale_factor = kwargs.get("scale_factor", 1.0)

        if payload is None:
            raise ValueError(f"{self.__class__.__name__} requires 'payload' in kwargs")

        # Handle scale_factor for different payload types
        if isinstance(payload, torch.Tensor):
            scaled = payload * scale_factor
            # Reuse existing buffer if shapes match (CUDA graph safe)
            existing = self._payloads.get(index)
            if (
                isinstance(existing, torch.Tensor)
                and existing.shape == scaled.shape
                and existing.dtype == scaled.dtype
                and existing.device == scaled.device
            ):
                existing.copy_(scaled)
                return  # buffer address unchanged — graph replay safe
            # First load or shape mismatch: allocate a new contiguous buffer
            payload = scaled.clone()
        elif isinstance(payload, dict):
            # For dict payload: add scale_factor to the dict
            payload = {**payload, "scale_factor": scale_factor}
        # For other types: store as-is (algorithms handle scaling themselves)

        self._payloads[index] = payload
    
    def set_active_tensor(self, index: int) -> None:
        """
        Universal implementation: Activate stored payload.

        When the payload is a Tensor and the active buffer already has the
        right shape, copies into the existing buffer (CUDA graph safe).
        Algorithms don't need to override this.
        """
        new_payload = self._payloads.get(index)
        # In-place copy when both are compatible tensors (preserves address)
        if (
            isinstance(new_payload, torch.Tensor)
            and isinstance(self._active_payload, torch.Tensor)
            and new_payload.shape == self._active_payload.shape
            and new_payload.dtype == self._active_payload.dtype
            and new_payload.device == self._active_payload.device
        ):
            self._active_payload.copy_(new_payload)
        else:
            self._active_payload = new_payload
    
    def reset_steer_vector(self, index: int) -> None:
        """
        Universal implementation: Remove payload.
        
        Algorithms don't need to override this.
        """
        if index in self._payloads:
            del self._payloads[index]
    
    def _get_params(self) -> Any:
        """
        Universal implementation: Return active payload as-is.
        
        Algorithms don't need to override this - payload format is defined
        by the algorithm's load_from_path method.
        """
        return self._active_payload
    
    def _is_valid(self, params: Any) -> bool:
        """
        Universal implementation: Check params is not None.
        
        Algorithms rarely need to override this.
        """
        return params is not None
    
    @abstractmethod
    def _transform(self, hidden_state: torch.Tensor, params: Any) -> torch.Tensor:
        """
        Transform hidden state (MUST be implemented by subclass).
        
        This is the core logic of your algorithm - the only truly required method.
        """
        pass
    
    # ========== Helper Methods ==========
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