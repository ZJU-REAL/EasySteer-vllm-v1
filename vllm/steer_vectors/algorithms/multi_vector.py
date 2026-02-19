# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List, Dict, Set, Tuple, Any
import torch
from dataclasses import dataclass

from .template import AlgorithmTemplate
from .factory import register_algorithm, create_algorithm
from .utils import extract_samples_info

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


@dataclass
class VectorInstance:
    """Represents a vector instance to be applied."""
    vector_idx: int
    algorithm: AlgorithmTemplate
    scale: float = 1.0


@register_algorithm("multi_vector")
class MultiVectorAlgorithm(AlgorithmTemplate):
    """Multi-vector control algorithm implementation, supports applying multiple vectors at the same layer."""
    
    def __init__(self, layer_id: Optional[int] = None):
        super().__init__(layer_id)
        # Store algorithm instance for each vector index
        self.vector_algorithms: Dict[int, AlgorithmTemplate] = {}
        # Store scale_factor for each vector index
        self.vector_scales: Dict[int, float] = {}
        # Conflict resolution strategy
        self.conflict_resolution: str = "priority"  # 'error', 'priority', or 'sequential'
        
    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> Dict[str, Any]:
        """
        MultiVectorAlgorithm is a container and does not load from a single path.
        This method is implemented to satisfy the abstract base class contract.
        """
        return {}
        
    def set_conflict_resolution(self, conflict_resolution: str) -> None:
        """Set conflict resolution strategy."""
        self.conflict_resolution = conflict_resolution

    def add_vector(self, vector_idx: int, algorithm_type: str, **kwargs) -> None:
        """Add a vector to multi-vector manager."""
        # Extract constructor arguments (e.g., normalize)
        init_kwargs = {}
        if "normalize" in kwargs:
            init_kwargs["normalize"] = kwargs.get("normalize")
        
        # Use factory to create algorithm instance
        algo = create_algorithm(algorithm_type, layer_id=self.layer_id, **init_kwargs)
        
        # Prepare unified parameters for set_steer_vector
        set_vector_kwargs = {}
        if "payload" in kwargs:
            set_vector_kwargs["payload"] = kwargs["payload"]
        if "scale_factor" in kwargs:
            set_vector_kwargs["scale_factor"] = kwargs.get("scale_factor", 1.0)

        if set_vector_kwargs:
            algo.set_steer_vector(0, **set_vector_kwargs)  # Use index 0 for internal storage
        
        # Store scale_factor separately (single-vector algorithms don't have scale attribute)
        self.vector_scales[vector_idx] = kwargs.get("scale_factor", 1.0)
        
        # Set trigger configuration using batch interface
        algo.params.configure_from_dict(kwargs)
            
        # Store the algorithm instance
        self.vector_algorithms[vector_idx] = algo
        
    def remove_vector(self, vector_idx: int) -> None:
        """Remove a vector."""
        if vector_idx in self.vector_algorithms:
            del self.vector_algorithms[vector_idx]
        if vector_idx in self.vector_scales:
            del self.vector_scales[vector_idx]

    def _all_vectors_global(self) -> bool:
        """Check if all sub-vectors are globally configured (both phases use -1 trigger).
        
        When True, the fast path can be used: apply _transform directly on the full
        hidden_states tensor without any index_select/index_copy or GPU-CPU sync.
        """
        return all(
            algo.params.is_global_only_config()
            for algo in self.vector_algorithms.values()
        )

    def apply_intervention(
        self, 
        hidden_states: torch.Tensor,
        context_info: Optional[Tuple[torch.Tensor, dict]] = None
    ) -> torch.Tensor:
        """Apply multi-vector intervention.
        
        In V1 continuous batching, a batch may contain both decode and prefill samples.
        We need to handle them separately based on their individual lengths (not batch-level phase).
        
        Args:
            hidden_states: Input tensor to transform
            context_info: Optional tuple of (current_tokens, samples_info).
                         If provided, use this instead of fetching from forward context.
        """
        if not self.vector_algorithms:
            return hidden_states

        # ========== Fast Path: All vectors are globally configured ==========
        # When all sub-vectors use prefill_trigger_tokens=[-1] AND generate_trigger_tokens=[-1]
        # with no exclusions, we can apply each _transform directly on the full tensor.
        # This matches the single-vector fast path: zero GPU-CPU sync, no index ops.
        if self._all_vectors_global():
            if self.params.debug:
                print(f"[MultiVector] ✨ Fast Path: All {len(self.vector_algorithms)} vectors "
                      f"are globally configured, applying direct transforms to ALL "
                      f"{hidden_states.shape[0]} tokens")
            original_dtype = hidden_states.dtype
            for vector_idx in sorted(self.vector_algorithms.keys()):
                algo = self.vector_algorithms[vector_idx]
                algo.set_active_tensor(0)
                params = algo._get_params()
                if algo._is_valid(params):
                    hidden_states = algo._transform(hidden_states, params).to(original_dtype)
                    if self.params.debug:
                        scale = self.vector_scales.get(vector_idx, 1.0)
                        print(f"[MultiVector]   Applied vector {vector_idx} (scale={scale})")
            return hidden_states

        # ========== Normal Path: Token-Level Control ==========
        # Get context information - either from provided context_info or fetch from forward context
        if context_info is not None:
            # Use provided context (e.g., from MoE layer wrapper)
            current_tokens, samples_info = context_info
            if self.params.debug:
                print(f"[MultiVector] Using provided context info")
        else:
            # Get forward context and samples info using helper from template
            # Use the first algorithm's helper method (they all inherit from template)
            first_algo = next(iter(self.vector_algorithms.values()))
            ctx_info = first_algo._get_forward_context_and_samples(hidden_states)
            if ctx_info is None:
                return hidden_states
            
            forward_ctx, samples_info, current_tokens = ctx_info
        
        # Debug: Show batch composition using helper
        # Note: multi_vector has its own params controller, so use self._debug_print_batch_info
        if self.params.debug:
            self._debug_print_batch_info(samples_info, class_name="MultiVector")
        
        # ========== Step 1: Collect all target positions for each vector (GPU tensors) ==========
        # Keep positions as GPU tensors to avoid GPU-CPU sync in this step.
        sorted_vector_indices = sorted(self.vector_algorithms.keys())
        vector_to_positions_tensor: Dict[int, torch.Tensor] = {}

        for vector_idx in sorted_vector_indices:
            algo = self.vector_algorithms[vector_idx]
            
            # Prepare algorithm parameters
            algo.set_active_tensor(0)
            params = algo._get_params()
            if not algo._is_valid(params):
                continue
            
            # Collect intervention positions - result stays on GPU
            positions_tensor = algo.params.collect_intervention_positions(
                hidden_states=hidden_states,
                current_tokens=current_tokens,
                samples_info=samples_info
            )
            
            if positions_tensor is not None and positions_tensor.numel() > 0:
                vector_to_positions_tensor[vector_idx] = positions_tensor

        # ========== Step 2: Conflict resolution ==========
        if self.conflict_resolution == "error":
            # Check for conflicts using GPU set operations (one sync for reporting only)
            for i, vi in enumerate(sorted_vector_indices):
                if vi not in vector_to_positions_tensor:
                    continue
                for vj in sorted_vector_indices[i + 1:]:
                    if vj not in vector_to_positions_tensor:
                        continue
                    # GPU intersection check
                    pi = vector_to_positions_tensor[vi]
                    pj = vector_to_positions_tensor[vj]
                    conflicts = torch.isin(pi, pj)
                    if conflicts.any():
                        # Sync only to report error (exceptional path)
                        conflict_positions = pi[conflicts].tolist()
                        raise ValueError(
                            f"Multiple vectors conflict at positions {conflict_positions}: "
                            f"vectors [{vi}, {vj}]. "
                            f"Set conflict_resolution='priority' to use the first vector, "
                            f"or 'sequential' to apply all vectors in sequence."
                        )

        elif self.conflict_resolution == "priority":
            # For each lower-priority vector, remove positions already claimed by
            # higher-priority vectors. All operations stay on GPU.
            claimed: Optional[torch.Tensor] = None
            for vector_idx in sorted_vector_indices:
                if vector_idx not in vector_to_positions_tensor:
                    continue
                positions = vector_to_positions_tensor[vector_idx]
                if claimed is not None and claimed.numel() > 0:
                    # Remove positions already claimed by higher-priority vectors
                    keep_mask = ~torch.isin(positions, claimed)
                    positions = positions[keep_mask]
                    if positions.numel() == 0:
                        del vector_to_positions_tensor[vector_idx]
                        continue
                    vector_to_positions_tensor[vector_idx] = positions
                # Accumulate claimed positions
                claimed = positions if claimed is None else torch.cat([claimed, positions])

            if self.params.debug and claimed is not None:
                print(f"[MultiVector] Priority mode: {claimed.numel()} total unique positions claimed")

        elif self.conflict_resolution == "sequential":
            # No filtering needed, all vectors will be applied in order
            pass

        else:
            raise ValueError(f"Unknown conflict resolution strategy: {self.conflict_resolution}")
        
        # ========== Step 3: Apply vectors in order (GPU tensors, no sync) ==========
        for vector_idx in sorted_vector_indices:
            if vector_idx not in vector_to_positions_tensor:
                continue

            indices_tensor = vector_to_positions_tensor[vector_idx]
            algo = self.vector_algorithms[vector_idx]
            
            # Prepare algorithm parameters
            algo.set_active_tensor(0)
            params = algo._get_params()
            if algo._is_valid(params):
                # Use helper method from template for batch transformation
                hidden_states = algo._batch_transform_tensor(hidden_states, indices_tensor, params)
                
                if self.params.debug:
                    scale = self.vector_scales.get(vector_idx, 1.0)
                    # Sync only for debug output
                    num_positions = indices_tensor.numel()
                    positions_preview = indices_tensor[:10].tolist()
                    print(f"[MultiVector] Applied vector {vector_idx} (scale={scale}) "
                          f"to {num_positions} positions: "
                          f"{positions_preview}{'...' if num_positions > 10 else ''}")
        
        return hidden_states
    
    # Abstract method implementations required by template (not directly used in multi-vector mode)
    def _get_params(self) -> Any:
        """Not used in multi-vector mode."""
        return None

    def _is_valid(self, params: Any) -> bool:
        """Not used in multi-vector mode."""
        return False

    def _transform(self, hidden_state: torch.Tensor, params: Any) -> torch.Tensor:
        """Not used in multi-vector mode."""
        return hidden_state

    # Methods to comply with BaseSteerVectorAlgorithm interface
    def set_steer_vector(self, index: int, **kwargs) -> None:
        """Not directly used in multi-vector mode."""
        pass

    def reset_steer_vector(self, index: int) -> None:
        """Reset all vectors."""
        self.vector_algorithms.clear()
        self.vector_scales.clear()

    def set_active_tensor(self, index: int) -> None:
        """Not directly used in multi-vector mode."""
        pass 