# SPDX-License-Identifier: Apache-2.0
"""
MoE Layer Wrappers for Steer Vector Intervention

These wrappers intercept MoE router logits and apply interventions before
expert selection. This is separate from the capture mechanism in hidden_states/moe_wrapper.py.
"""

from typing import Optional, Dict, Any
import torch
from torch import nn

from .algorithms import BaseSteerVectorAlgorithm, create_algorithm

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


def extract_moe_layer_id_from_name(module_name: str) -> Optional[int]:
    """
    Extract layer ID from MoE module name.
    
    Args:
        module_name: Module name like 'model.layers.10.block_sparse_moe'
        
    Returns:
        Layer ID as integer, or None if not found
    """
    parts = module_name.split('.')
    
    for i, part in enumerate(parts):
        if part.isdigit():
            if i > 0:
                prev_part = parts[i - 1].lower()
                if prev_part in ['layers', 'h', 'layer', 'blocks', 'block']:
                    return int(part)
    
    return None


class MoELayerWithSteerVector(nn.Module):
    """
    MoE layer wrapper that supports router logits intervention.
    
    This wrapper intercepts the gate/router output and applies transformations
    before expert selection. Uses lazy loading for algorithm instances.
    
    Architecture compatibility:
    - Qwen3-MoE / Qwen3-VL-MoE: self.experts(hidden_states, router_logits)
    - Mixtral: experts called internally, requires gate patching
    - DeepSeek-V2: similar to Mixtral
    """
    
    def __init__(self, base_layer: nn.Module, layer_name: str = "") -> None:
        super().__init__()
        self.base_layer = base_layer
        self.layer_name = layer_name
        self.layer_id: Optional[int] = extract_moe_layer_id_from_name(layer_name)
        
        # Identify gate/router and experts modules
        self.gate = getattr(base_layer, 'gate', None)
        self.router = getattr(base_layer, 'router', None)
        self.experts = getattr(base_layer, 'experts', None)
        
        # Use gate or router (different models use different names)
        self.routing_module = self.gate if self.gate is not None else self.router
        
        # Detect MoE architecture
        self.architecture = self._detect_architecture()
        
        # Algorithm management (lazy loading)
        self.active_algorithm_name: str = "moe_router"
        self.algorithms: Dict[str, BaseSteerVectorAlgorithm] = {}
    
    def _detect_architecture(self) -> str:
        """
        Detect MoE architecture type for proper intervention.
        
        Returns:
            'qwen3' | 'mixtral' | 'deepseek_v2' | 'deepseek_v1' | 'kimi' | 'glm4' | 'llama4' | 'unknown'
        """
        class_name = type(self.base_layer).__name__
        
        # Llama4MoE: SharedFusedMoE with shared_expert, returns tuple (shared_out, routed_out)
        # Uses custom_routing_function with sigmoid, no routed_scaling_factor
        if 'Llama4MoE' in class_name:
            return 'llama4'
        
        # DeepseekV2MoE: SharedFusedMoE, returns tuple (shared_output, final_hidden_states)
        # Used by: Kimi-VL-A3B (uses DeepseekV2Model internally)
        if 'DeepseekV2MoE' in class_name:
            return 'deepseek_v2'
        
        # DeepseekMoE (V1): uses fused_topk + fused_experts directly, no experts module forward
        if class_name == 'DeepseekMoE':
            return 'deepseek_v1'
        
        # KimiMoE (kimi_linear.py): separate shared_experts, FusedMoE returns Tensor
        if 'Kimi' in class_name:
            return 'kimi'
        
        # Glm4MoE: similar to DeepseekV2, SharedFusedMoE returns tuple
        if 'Glm4MoE' in class_name:
            return 'glm4'
        
        # Qwen3/Qwen2-MoE: experts() accepts router_logits parameter, returns Tensor
        if 'Qwen3' in class_name or 'Qwen2' in class_name:
            return 'qwen3'
        
        # Mixtral: need to patch gate temporarily
        if 'Mixtral' in class_name:
            return 'mixtral'
        
        # Fallback detection by checking experts signature
        if self.experts is not None:
            import inspect
            sig = inspect.signature(self.experts.forward)
            if 'router_logits' in sig.parameters:
                return 'qwen3'
        
        return 'unknown'
    
    def _get_or_create_algorithm(self, name: str, **kwargs) -> BaseSteerVectorAlgorithm:
        """Lazy load or get algorithm instance by name."""
        if name not in self.algorithms:
            self.algorithms[name] = create_algorithm(name, layer_id=self.layer_id, **kwargs)
        return self.algorithms[name]
    
    def set_layer_id(self, layer_id: int) -> None:
        """Set layer ID for all created algorithms."""
        self.layer_id = layer_id
        for algo in self.algorithms.values():
            algo.layer_id = layer_id
    
    def set_steer_vector(self, index: int, **kwargs):
        """Set steer vector parameters for the specified algorithm."""
        # Determine algorithm
        algorithm_name = kwargs.pop("algorithm_name", "moe_router")
        self.active_algorithm_name = algorithm_name
        
        # Get or create algorithm
        algo = self._get_or_create_algorithm(algorithm_name)
        
        # Set core vector parameters and intervention parameters
        algo.set_steer_vector(index, **kwargs)
        algo.params.configure_from_dict(kwargs)
    
    def reset_steer_vector(self, index: int):
        """Reset the vector at specified index in all algorithms."""
        for algo in self.algorithms.values():
            algo.reset_steer_vector(index)
    
    def set_active_tensor(self, index: int):
        """Set the active tensor for the currently active algorithm."""
        algo = self._get_or_create_algorithm(self.active_algorithm_name)
        algo.set_active_tensor(index)
    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        Forward pass with router logits intervention.
        
        The intervention happens between gate/router and experts:
        1. Get forward context at layer level (for token-level control)
        2. Call gate to get router_logits
        3. Apply intervention algorithm with context info
        4. Pass modified router_logits to experts
        """
        # Get active algorithm
        active_algo = self._get_or_create_algorithm(self.active_algorithm_name)
        
        # Step 1: Try to get forward context at layer level
        # This context will be passed to the algorithm for token-level intervention control
        context_info = None
        if get_forward_context is not None:
            try:
                forward_ctx = get_forward_context()
                if forward_ctx is not None:
                    current_tokens = forward_ctx.current_tokens
                    attn_metadata = forward_ctx.attn_metadata
                    
                    if current_tokens is not None and attn_metadata is not None:
                        # Flatten tokens if needed
                        if current_tokens.dim() == 2:
                            current_tokens = current_tokens.flatten()
                        
                        # Extract sample boundaries
                        from .algorithms.parameter_control import extract_samples_info
                        samples_info = extract_samples_info(attn_metadata)
                        
                        if samples_info is not None:
                            # Package context info for algorithm
                            context_info = (current_tokens, samples_info)
            except Exception:
                # Context extraction failed, will use fallback in algorithm
                pass
        
        # Step 2: Extract router logits from gate/router
        router_logits = self._extract_router_logits(hidden_states)
        
        if router_logits is None:
            # Failed to extract, use original forward
            return self.base_layer(hidden_states, *args, **kwargs)
        
        # Step 3: Apply intervention to router logits with context info
        # If context_info is None, algorithm will try to fetch it itself (backward compatible)
        modified_router_logits = active_algo.apply_intervention(
            router_logits,
            context_info=context_info
        )
        
        # Step 4: Call experts with modified router logits (architecture-specific)
        return self._forward_with_modified_logits(
            hidden_states, modified_router_logits, *args, **kwargs
        )
    
    def _extract_router_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract router logits from the gate/router module.
        
        Args:
            hidden_states: Input to MoE layer
            
        Returns:
            Router logits (num_tokens, n_experts) or None
        """
        if self.routing_module is None:
            return None
        
        try:
            # Flatten if needed
            orig_shape = hidden_states.shape
            if hidden_states.ndim > 2:
                hidden_size = hidden_states.shape[-1]
                hidden_states_flat = hidden_states.view(-1, hidden_size)
            else:
                hidden_states_flat = hidden_states
            
            # Call gate/router
            gate_output = self.routing_module(hidden_states_flat)
            
            if isinstance(gate_output, tuple):
                router_logits = gate_output[0]
            elif isinstance(gate_output, torch.Tensor):
                router_logits = gate_output
            else:
                return None
            
            # Ensure 2D
            if router_logits.ndim == 1:
                router_logits = router_logits.unsqueeze(0)
            
            return router_logits
            
        except Exception:
            return None
    
    def _forward_with_modified_logits(
        self, 
        hidden_states: torch.Tensor,
        modified_router_logits: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with modified router logits.
        
        Architecture-specific implementation to inject modified logits.
        """
        if self.architecture == 'llama4':
            # Llama4MoE: SharedFusedMoE with shared_expert, returns tuple
            return self._forward_llama4(hidden_states, modified_router_logits)
        
        elif self.architecture == 'deepseek_v2':
            # DeepseekV2MoE: SharedFusedMoE returns tuple, complex handling
            # Used by Kimi-VL-A3B (DeepseekV2Model)
            return self._forward_deepseek_v2(hidden_states, modified_router_logits)
        
        elif self.architecture == 'glm4':
            # Glm4MoE: similar to DeepseekV2, SharedFusedMoE returns tuple
            return self._forward_glm4(hidden_states, modified_router_logits)
        
        elif self.architecture == 'kimi':
            # KimiMoE: has separate shared_experts and routed_scaling_factor
            return self._forward_kimi(hidden_states, modified_router_logits)
        
        elif self.architecture == 'qwen3':
            # Qwen3-MoE: experts() accepts router_logits parameter
            return self._forward_qwen3(hidden_states, modified_router_logits)
        
        elif self.architecture in ['mixtral', 'deepseek_v1']:
            # Mixtral/DeepSeek-V1: need to patch gate temporarily
            return self._forward_with_gate_patch(hidden_states, modified_router_logits, *args, **kwargs)
        
        else:
            # Unknown architecture: try Qwen3 style first, fall back to patching
            try:
                return self._forward_qwen3(hidden_states, modified_router_logits)
            except:
                try:
                    return self._forward_with_gate_patch(hidden_states, modified_router_logits, *args, **kwargs)
                except:
                    # Give up, use original forward
                    return self.base_layer(hidden_states, *args, **kwargs)
    
    def _forward_kimi(
        self,
        hidden_states: torch.Tensor,
        modified_router_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward for KimiMoE architecture.
        
        KimiMoE.forward:
        1. hidden_states = hidden_states.view(-1, hidden_size)
        2. if has shared_experts: shared_output = self.shared_experts(hidden_states)
        3. router_logits, _ = self.gate(hidden_states)
        4. final_hidden_states = self.experts(hidden_states, router_logits) * routed_scaling_factor
        5. if shared_output: final_hidden_states += shared_output
        6. if tp_size > 1: tensor_model_parallel_all_reduce
        7. return final_hidden_states.view(num_tokens, hidden_size)
        """
        base = self.base_layer
        assert hasattr(base, 'experts'), "Missing experts module"
        
        # Save original shape
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        
        # Step 1: Compute shared_output if shared_experts exists
        shared_output = None
        num_shared_experts = getattr(base, 'num_shared_experts', None)
        if num_shared_experts is not None and hasattr(base, 'shared_experts'):
            shared_output = base.shared_experts(hidden_states)
        
        # Step 2: Call experts with modified router_logits
        final_hidden_states = base.experts(
            hidden_states=hidden_states,
            router_logits=modified_router_logits
        )
        
        # Step 3: Apply routed_scaling_factor
        routed_scaling_factor = getattr(base, 'routed_scaling_factor', 1.0)
        final_hidden_states = final_hidden_states * routed_scaling_factor
        
        # Step 4: Add shared_output
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        
        # Step 5: tensor_model_parallel_all_reduce if needed
        tp_size = getattr(base, 'tp_size', 1)
        if tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        
        return final_hidden_states.view(num_tokens, hidden_size)
    
    def _forward_llama4(
        self,
        hidden_states: torch.Tensor,
        modified_router_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward for Llama4MoE architecture.
        
        Llama4MoE.forward:
        1. num_tokens = hidden_states.shape[0]
        2. if is_sequence_parallel: hidden_states = sequence_parallel_chunk(hidden_states)
        3. router_logits, _ = self.router(hidden_states)
        4. shared_out, routed_out = self.experts(hidden_states, router_logits)
        5. experts_out = routed_out + shared_out
        6. if is_sequence_parallel: tensor_model_parallel_all_gather
        7. elif tp_size > 1: maybe_all_reduce_tensor_model_parallel
        
        Key differences from DeepseekV2:
        - Uses router instead of gate
        - No routed_scaling_factor
        - experts returns (shared_out, routed_out), order matters
        """
        base = self.base_layer
        assert hasattr(base, 'experts'), "Missing experts module"
        
        num_tokens = hidden_states.shape[0]
        
        # Handle sequence parallel
        is_sequence_parallel = getattr(base, 'is_sequence_parallel', False)
        if is_sequence_parallel:
            from vllm.model_executor.models.utils import sequence_parallel_chunk
            hidden_states = sequence_parallel_chunk(hidden_states)
            # CRITICAL: modified_router_logits was computed on the original hidden_states
            # (before chunking), but in Llama4MoE.forward, self.router() is called
            # AFTER sequence_parallel_chunk, so we need to chunk router_logits too.
            modified_router_logits = sequence_parallel_chunk(modified_router_logits)
        
        # Call experts with modified router_logits
        # Llama4MoE uses SharedFusedMoE which returns (shared_out, routed_out)
        shared_out, routed_out = base.experts(
            hidden_states=hidden_states,
            router_logits=modified_router_logits,
        )
        
        # Combine shared and routed outputs (no scaling factor in Llama4)
        experts_out = routed_out + shared_out
        
        # Handle sequence parallel or tensor parallel
        if is_sequence_parallel:
            from vllm.distributed import tensor_model_parallel_all_gather
            experts_out = tensor_model_parallel_all_gather(experts_out, 0)
            experts_out = experts_out[:num_tokens]
        else:
            tp_size = getattr(base, 'tp_size', 1)
            if tp_size > 1:
                experts_out = base.experts.maybe_all_reduce_tensor_model_parallel(
                    experts_out
                )
        
        return experts_out
    
    def _forward_deepseek_v2(
        self,
        hidden_states: torch.Tensor,
        modified_router_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward for DeepseekV2MoE architecture.
        
        DeepseekV2MoE.forward:
        1. hidden_states = hidden_states.view(-1, hidden_dim)
        2. if is_sequence_parallel: hidden_states = sequence_parallel_chunk(hidden_states)
        3. if is_internal_router: experts(hidden_states, hidden_states)
           else: router_logits = gate(hidden_states); experts(hidden_states, router_logits)
        4. shared_output, final_hidden_states = fused_moe_out  # TUPLE!
        5. Apply routed_scaling_factor (complex FP16 handling)
        6. if shared_experts: final_hidden_states += shared_output
        7. Handle sequence_parallel or tp_size > 1
        
        Used by: Kimi-VL-A3B (uses DeepseekV2Model internally)
        
        IMPORTANT: When is_internal_router=True, FusedMoE.forward_impl will 
        override the passed router_logits with self.gate(hidden_states).
        We must temporarily disable the internal gate to use our modified logits.
        """
        base = self.base_layer
        assert hasattr(base, 'experts'), "Missing experts module"
        
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Handle sequence parallel
        is_sequence_parallel = getattr(base, 'is_sequence_parallel', False)
        if is_sequence_parallel:
            from vllm.model_executor.layers.moe.fused_moe_glu import sequence_parallel_chunk
            hidden_states_flat = sequence_parallel_chunk(hidden_states_flat)
            # CRITICAL: modified_router_logits was computed on original hidden_states
            # (before chunking), but in DeepseekV2MoE.forward, self.gate() is called
            # AFTER sequence_parallel_chunk, so we need to chunk router_logits too.
            modified_router_logits = sequence_parallel_chunk(modified_router_logits)
        
        # CRITICAL FIX: When is_internal_router=True, FusedMoE.forward_impl 
        # will override our modified_router_logits with self.gate(hidden_states).
        # We must temporarily disable the internal gate to ensure our modified 
        # router_logits are actually used.
        experts = base.experts
        original_gate = None
        if getattr(experts, 'is_internal_router', False):
            # Temporarily disable internal gate
            original_gate = experts._gate
            experts._gate = None
        
        try:
            # Call experts with modified router_logits
            fused_moe_out = experts(
                hidden_states=hidden_states_flat,
                router_logits=modified_router_logits
            )
        finally:
            # Restore the original gate
            if original_gate is not None:
                experts._gate = original_gate
        
        # DeepseekV2MoE.experts (SharedFusedMoE) returns tuple: (shared_output, final_hidden_states)
        shared_output, final_hidden_states = fused_moe_out
        
        # Apply routed_scaling_factor with FP16 overflow handling
        # NOTE: Check is_rocm_aiter_moe_enabled() to avoid double scaling,
        # since ROCM AITER applies routed_scaling_factor internally.
        routed_scaling_factor = getattr(base, 'routed_scaling_factor', 1.0)
        shared_experts = getattr(base, 'shared_experts', None)
        
        # Import the check function
        try:
            from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
                is_rocm_aiter_moe_enabled
            )
        except ImportError:
            # Fallback if not available
            def is_rocm_aiter_moe_enabled():
                return False
        
        if hidden_states.dtype != torch.float16:
            # Only apply scaling if ROCM AITER is not enabled (it handles scaling internally)
            if not is_rocm_aiter_moe_enabled():
                final_hidden_states = final_hidden_states * routed_scaling_factor
        elif shared_experts is not None and shared_output is not None:
            # FP16 overflow fix: scale down shared_output instead
            shared_output = shared_output * (1.0 / routed_scaling_factor)
        
        # Add shared_output
        if shared_experts is not None and shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        
        # Handle sequence_parallel or tensor parallel
        if is_sequence_parallel:
            from vllm.distributed import tensor_model_parallel_all_gather
            final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]
        else:
            tp_size = getattr(base, 'tp_size', 1)
            if tp_size > 1:
                final_hidden_states = base.experts.maybe_all_reduce_tensor_model_parallel(
                    final_hidden_states
                )
        
        return final_hidden_states.view(num_tokens, hidden_dim)
    
    def _forward_glm4(
        self,
        hidden_states: torch.Tensor,
        modified_router_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward for Glm4MoE architecture.
        
        Similar to DeepseekV2MoE but with different routed_scaling_factor handling:
        - Uses SharedFusedMoE, returns (shared_output, final_hidden_states) or just Tensor
        - routed_scaling_factor applied differently
        """
        base = self.base_layer
        assert hasattr(base, 'experts'), "Missing experts module"
        
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Call experts with modified router_logits
        fused_moe_out = base.experts(
            hidden_states=hidden_states_flat,
            router_logits=modified_router_logits
        )
        
        # Glm4MoE.experts may return tuple or Tensor depending on shared_experts
        shared_experts = getattr(base, 'shared_experts', None)
        routed_scaling_factor = getattr(base, 'routed_scaling_factor', 1.0)
        
        if shared_experts is not None:
            shared_output, final_hidden_states = fused_moe_out
            final_hidden_states = (
                final_hidden_states * routed_scaling_factor + shared_output
            )
        else:
            final_hidden_states = fused_moe_out * routed_scaling_factor
        
        # Handle tensor parallel
        tp_size = getattr(base, 'tp_size', 1)
        if tp_size > 1:
            final_hidden_states = base.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )
        
        return final_hidden_states.view(num_tokens, hidden_dim)
    
    def _forward_qwen3(
        self, 
        hidden_states: torch.Tensor,
        modified_router_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward for Qwen3-MoE architecture.
        
        Qwen3MoeSparseMoeBlock.forward:
        1. hidden_states = hidden_states.view(-1, hidden_dim)
        2. router_logits, _ = self.gate(hidden_states)
        3. final_hidden_states = self.experts(hidden_states, router_logits)
        4. return final_hidden_states
        """
        # Replicate Qwen3's forward logic with modified logits
        assert hasattr(self.base_layer, 'experts'), "Missing experts module"
        
        # Handle 1D/2D input
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Handle sequence parallel if enabled
        if hasattr(self.base_layer, 'is_sequence_parallel') and self.base_layer.is_sequence_parallel:
            from vllm.model_executor.layers.moe.fused_moe_glu import sequence_parallel_chunk
            from vllm.distributed import tensor_model_parallel_all_gather
            
            hidden_states = sequence_parallel_chunk(hidden_states)
            
            # CRITICAL: modified_router_logits was computed on the original hidden_states
            # (before chunking), but in the original Qwen3MoeSparseMoeBlock.forward,
            # self.gate() is called AFTER sequence_parallel_chunk, so router_logits
            # has shape matching chunked hidden_states. We need to chunk router_logits too.
            modified_router_logits = sequence_parallel_chunk(modified_router_logits)
            
            # Call experts with modified logits
            final_hidden_states = self.base_layer.experts(
                hidden_states=hidden_states,
                router_logits=modified_router_logits
            )
            
            # Gather results
            final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]
        else:
            # Call experts directly with modified router_logits
            final_hidden_states = self.base_layer.experts(
                hidden_states=hidden_states,
                router_logits=modified_router_logits
            )
        
        # Return to 1D if input was 1D
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states
    
    def _forward_with_gate_patch(
        self,
        hidden_states: torch.Tensor,
        modified_router_logits: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward with temporary gate patching for Mixtral/DeepSeek.
        
        This temporarily replaces the gate's forward method to return
        our modified router logits.
        """
        if self.routing_module is None:
            return self.base_layer(hidden_states, *args, **kwargs)
        
        # Save original forward
        original_forward = self.routing_module.forward
        
        # Create patched forward that returns modified logits
        def patched_forward(*args, **kwargs):
            # Return modified logits in same format as original gate
            return (modified_router_logits, None)
        
        try:
            # Temporarily replace forward
            self.routing_module.forward = patched_forward
            
            # Call base layer (will use patched gate)
            output = self.base_layer(hidden_states, *args, **kwargs)
            
            return output
            
        finally:
            # Always restore original forward
            self.routing_module.forward = original_forward
    
    def __getattr__(self, name):
        """Delegate attribute access to the base layer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)


# List of MoE layer class names for identification
MOE_LAYER_CLASSES = [
    # Qwen family
    'Qwen2MoeSparseMoeBlock',
    'Qwen3MoeSparseMoeBlock',
    'Qwen3NextSparseMoeBlock',
    # Mixtral / Llama family
    'MixtralMoE',
    'Llama4MoE',
    'PhiMoE',
    # DeepSeek family
    'DeepseekMoE',
    'DeepseekV2MoE',
    # Kimi
    'KimiMoE',
    # GLM
    'Glm4MoE',
    # Ernie
    'Ernie4_5_MoeMoE',
    'Ernie4_5_VLMoeMoE',
    # Others
    'DbrxExperts',
    'DbrxMoE',
    'ArcticMoE',
    'JambaMoE',
    'Grok1MoE',
    'GraniteMoeMoE',
    'MiniMaxText01MoE',
    'MiniMaxM2MoE',
    'MiniCPMMoE',
    'OlmoeMoE',
    'FlexOlmoMoE',
    'NemotronHMoE',
    'BailingMoE',
    'Dots1MoE',
    'NomicMoE',
]


def is_moe_layer(module: nn.Module, module_name: str = "") -> bool:
    """
    Check if a module is an MoE layer that should be wrapped for steering.
    
    Args:
        module: The module to check
        module_name: Full module name
        
    Returns:
        True if this is an MoE layer
    """
    # Check by class name
    class_name = type(module).__name__
    if class_name in MOE_LAYER_CLASSES:
        return True
    
    # Check by module name pattern + attributes
    module_name_lower = module_name.lower()
    moe_patterns = ['block_sparse_moe', 'moe_layer', 'sparse_moe']
    
    for pattern in moe_patterns:
        if pattern in module_name_lower:
            # Verify it has gate/router and experts
            has_gate = hasattr(module, 'gate') or hasattr(module, 'router')
            has_experts = hasattr(module, 'experts')
            if has_gate and has_experts:
                return True
    
    return False

