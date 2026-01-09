# SPDX-License-Identifier: Apache-2.0

"""Steer Vector Request classes for vLLM V1."""

import msgspec
from typing import Optional, List


class VectorConfig(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
    frozen=False,  # type: ignore[call-arg]
):  # type: ignore[call-arg]
    """
    Configuration for a single vector in multi-vector mode.
    
    Args:
        path: Local path to the vector file
        scale: Scale factor for this vector (default: 1.0)
        target_layers: List of layer indices to apply this vector to. If None, apply to all layers
        prefill_trigger_tokens: List of token IDs that trigger vector application in prefill phase.
                               Use [-1] to apply to ALL tokens in prefill phase.
        prefill_trigger_positions: List of token positions that trigger vector application in prefill phase.
                                 Supports negative indexing (e.g., -1 for last token).
        prefill_exclude_tokens: List of token IDs to exclude from vector application in prefill phase.
                               Exclude has higher priority than trigger tokens.
        prefill_exclude_positions: List of token positions to exclude from vector application in prefill phase.
                                  Supports negative indexing. Exclude has higher priority than trigger positions.
        generate_trigger_tokens: List of token IDs that trigger vector application in generate phase.
                                Use [-1] to apply to ALL tokens in generate phase.
        generate_first_k_tokens: Only apply to the first k generated tokens (0-indexed: tokens 0, 1, ..., k-1).
                                Mutually exclusive with generate_after_k_tokens.
        generate_after_k_tokens: Start applying from the k-th generated token onwards (skip first k tokens).
                                Mutually exclusive with generate_first_k_tokens.
        algorithm: Vector algorithm to use: 'direct' (default) or 'loreft'
        normalize: Whether to normalize the vector (default: False, only applies to 'direct' algorithm)
    """
    path: str
    scale: float = 1.0
    target_layers: List[int] | None = None
    prefill_trigger_tokens: List[int] | None = None
    prefill_trigger_positions: List[int] | None = None
    prefill_exclude_tokens: List[int] | None = None
    prefill_exclude_positions: List[int] | None = None
    generate_trigger_tokens: List[int] | None = None
    generate_first_k_tokens: int | None = None
    generate_after_k_tokens: int | None = None
    algorithm: str = "direct"
    normalize: bool = False


class SteerVectorRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
    frozen=False,  # type: ignore[call-arg]
):  # type: ignore[call-arg]
    """
    Request for a Steer Vector adapter.
    Supports both single-vector mode (backward compatible) and multi-vector mode.

    Args:
        steer_vector_name: Name of the steer vector
        steer_vector_int_id: Unique ID for the steer vector (must be > 0)
        debug: Whether to print debug information during forward pass (default: False)
        conflict_resolution: How to handle conflicts when multiple vectors target the same position.
                           'error': raise an error when conflicts occur
                           'priority': use the first vector and ignore others (default)
                           'sequential': apply all vectors in sequence (effects stack)
        
        Single-vector mode (backward compatible):
        steer_vector_local_path: Local path to the steer vector file
        scale: Scale factor for the steer vector (default: 1.0)
        target_layers: List of layer indices to apply the steer vector to. If None, apply to all layers
        prefill_trigger_tokens: List of token IDs that trigger steer vector application in prefill phase.
        prefill_trigger_positions: List of token positions that trigger steer vector application in prefill phase.
        prefill_exclude_tokens: List of token IDs to exclude from steer vector application in prefill phase.
        prefill_exclude_positions: List of token positions to exclude from steer vector application in prefill phase.
        generate_trigger_tokens: List of token IDs that trigger steer vector application in generate phase.
        generate_first_k_tokens: Only apply to the first k generated tokens (0-indexed: tokens 0, 1, ..., k-1).
                                Mutually exclusive with generate_after_k_tokens.
        generate_after_k_tokens: Start applying from the k-th generated token onwards (skip first k tokens).
                                Mutually exclusive with generate_first_k_tokens.
        algorithm: Steer vector algorithm to use: 'direct' (default) or 'loreft'
        normalize: Whether to normalize the steer vector (default: False, only applies to 'direct' algorithm)
        
        Multi-vector mode:
        vector_configs: List of VectorConfig objects for multi-vector control
    """

    steer_vector_name: str
    steer_vector_int_id: int
    steer_vector_local_path: str = ""
    debug: bool = False
    conflict_resolution: str = "priority"
    
    # === Single-vector mode (backward compatible) ===
    scale: float = 1.0
    target_layers: List[int] | None = None
    prefill_trigger_tokens: List[int] | None = None
    prefill_trigger_positions: List[int] | None = None
    prefill_exclude_tokens: List[int] | None = None
    prefill_exclude_positions: List[int] | None = None
    generate_trigger_tokens: List[int] | None = None
    generate_first_k_tokens: int | None = None
    generate_after_k_tokens: int | None = None
    algorithm: str = "direct"
    normalize: bool = False
    
    # === Multi-vector mode ===
    vector_configs: List[VectorConfig] | None = None
    
    # === MoE-specific parameters (for moe_router algorithm) ===
    moe_expert_ids: List[int] | None = None  # Expert IDs to intervene
    moe_mode: str = "boost"                   # Intervention mode: 'boost', 'suppress', 'soft', or 'soft_topk'
    moe_lambda: float = 0.5                   # Lambda parameter for 'soft' modes (z'_k = z_k + lambda * std(z))
    moe_topk: int = 8                         # Top-K parameter for 'soft_topk' mode (only intervene if expert is in top-k)

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.steer_vector_int_id < 1:
            raise ValueError(
                f"steer_vector_int_id must be > 0, got {self.steer_vector_int_id}"
            )
        
        if self.conflict_resolution not in ["error", "priority", "sequential"]:
            raise ValueError(
                f"conflict_resolution must be 'error', 'priority', or 'sequential', "
                f"got '{self.conflict_resolution}'"
            )
        
        # Validate generate position control parameters
        if self.generate_first_k_tokens is not None and self.generate_after_k_tokens is not None:
            raise ValueError(
                "Cannot specify both generate_first_k_tokens and generate_after_k_tokens. "
                "They are mutually exclusive."
            )
        
        if self.generate_first_k_tokens is not None and self.generate_first_k_tokens < 0:
            raise ValueError(
                f"generate_first_k_tokens must be >= 0, got {self.generate_first_k_tokens}"
            )
        
        if self.generate_after_k_tokens is not None and self.generate_after_k_tokens < 0:
            raise ValueError(
                f"generate_after_k_tokens must be >= 0, got {self.generate_after_k_tokens}"
            )
            
        if self.is_multi_vector:
            if self.steer_vector_local_path:
                raise ValueError(
                    "Cannot specify both steer_vector_local_path and vector_configs"
                )
            if not self.vector_configs:
                raise ValueError(
                    "vector_configs cannot be empty in multi-vector mode"
                )
        else:
            # Special case: moe_router algorithm can work without a file path
            # It uses moe_expert_ids, moe_scale, moe_mode from the request directly
            if self.algorithm != "moe_router" and not self.steer_vector_local_path:
                raise ValueError(
                    "Must specify steer_vector_local_path in single-vector mode "
                    "(except for moe_router algorithm)"
                )

    @property
    def is_multi_vector(self) -> bool:
        """Check if this is a multi-vector request."""
        return self.vector_configs is not None

    @property
    def steer_vector_id(self) -> int:
        """Alias for steer_vector_int_id (backward compatibility)."""
        return self.steer_vector_int_id

    @property
    def local_path(self) -> str | None:
        """Get the local path for single-vector mode."""
        if self.is_multi_vector:
            return None  # Multi-vector mode doesn't have a single path
        return self.steer_vector_local_path

    @property
    def scale_factor(self) -> float:
        """Backward compatibility property."""
        if self.is_multi_vector:
            return 1.0  # Multi-vector mode uses individual scales
        return self.scale

    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare SteerVectorRequest
        instances based on steer_vector_name. This allows for identification
        and comparison of steer vector adapters across engines.
        """
        return isinstance(value, self.__class__) and self.steer_vector_name == value.steer_vector_name

    def __hash__(self) -> int:
        """
        Overrides the hash method to hash SteerVectorRequest instances
        based on steer_vector_name. This ensures that SteerVectorRequest instances
        can be used in hash-based collections such as sets and dictionaries,
        identified by their names across engines.
        """
        return hash(self.steer_vector_name)
