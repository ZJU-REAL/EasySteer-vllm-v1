# SPDX-License-Identifier: Apache-2.0
"""
V1 Worker Mixin for Capture Features (Hidden States, MoE Router Logits, etc.)

This unified mixin integrates all capture/extraction features into vLLM V1's GPUModelRunner.
Currently supports:
- Hidden states capture from transformer layers
- MoE router logits capture from MoE layers

Future extensibility:
- Attention weights capture
- Gradient capture
- Activation capture from specific layers
"""

from typing import Dict
import torch
from torch import nn

from vllm.hidden_states.wrapper import VLLMTransformerLayerWrapper
from vllm.hidden_states.storage import HiddenStatesStore
from vllm.hidden_states.moe_wrapper import (
    VLLMMoELayerWrapper,
    is_moe_layer,
    extract_moe_layer_id_from_name,
)
from vllm.hidden_states.moe_storage import MoERouterLogitsStore


class CaptureModelRunnerMixin:
    """
    Unified mixin for all capture/extraction features in GPUModelRunner.
    
    This mixin manages:
    1. Hidden states capture from transformer layers
    2. MoE router logits capture from MoE layers
    3. (Future) Other capture features like attention weights, gradients, etc.
    
    All capture features share common patterns:
    - Lazy initialization to avoid MRO issues
    - Wrapping during load_model()
    - Enable/disable/get/clear RPC interfaces
    - Multi-batch support
    
    Note: Attributes are created on first access, not in __init__.
    """
    
    # ========================================================================
    # Hidden States Capture
    # ========================================================================
    
    def _wrap_model_for_hidden_states(self, model: nn.Module) -> nn.Module:
        """
        Wrap model layers for hidden states capture.
        
        This should be called during load_model() to prepare the model.
        
        Args:
            model: The model to wrap
            
        Returns:
            The wrapped model
        """
        # Initialize store if not exists
        if not hasattr(self, 'hidden_states_store'):
            self.hidden_states_store = HiddenStatesStore()
            self.hidden_states_capture_enabled = False
            self._hidden_states_wrapped = False
        
        # Skip if already wrapped
        if self._hidden_states_wrapped:
            return model
        
        # Import here to avoid circular dependency
        from vllm.steer_vectors.config import SUPPORTED_DECODER_LAYERS
        
        layer_id = 0
        wrapped_count = 0
        
        def wrap_module(module: nn.Module, name: str = "") -> nn.Module:
            nonlocal layer_id, wrapped_count
            
            # Get class name
            class_name = module.__class__.__name__
            
            # Check if this is a decoder layer
            if class_name in SUPPORTED_DECODER_LAYERS:
                # Check if already wrapped
                if not isinstance(module, VLLMTransformerLayerWrapper):
                    wrapped_layer = VLLMTransformerLayerWrapper(
                        module, layer_id, name, self.hidden_states_store
                    )
                    layer_id += 1
                    wrapped_count += 1
                    return wrapped_layer
            
            # Recursively process child modules
            for child_name, child_module in list(module.named_children()):
                full_child_name = f"{name}.{child_name}" if name else child_name
                wrapped_child = wrap_module(child_module, full_child_name)
                if wrapped_child is not child_module:
                    setattr(module, child_name, wrapped_child)
            
            return module
        
        # Wrap the model and return
        wrapped_model = wrap_module(model)
        self._hidden_states_wrapped = True
        
        if wrapped_count > 0:
            from vllm.logger import init_logger
            logger = init_logger(__name__)
            logger.info(f"[Capture] Wrapped {wrapped_count} decoder layers for hidden states capture")
        else:
            import warnings
            warnings.warn("No decoder layers were wrapped for hidden states capture")
        
        return wrapped_model
    
    def enable_hidden_states_capture(self):
        """Enable hidden states capture."""
        if not hasattr(self, 'hidden_states_store'):
            raise RuntimeError(
                "Hidden states store not initialized. "
                "This should not happen if the model was loaded properly."
            )
        
        if not self._hidden_states_wrapped:
            raise RuntimeError(
                "Model not wrapped for hidden states capture. "
                "This should not happen if the model was loaded properly."
            )
        
        self.hidden_states_store.enable_capture()
        self.hidden_states_store.clear()
        self.hidden_states_store.enable_multi_batch_mode()
        self.hidden_states_capture_enabled = True
    
    def disable_hidden_states_capture(self):
        """Disable hidden states capture"""
        if hasattr(self, 'hidden_states_store') and self.hidden_states_store:
            self.hidden_states_store.disable_capture()
            self.hidden_states_capture_enabled = False
    
    def get_captured_hidden_states(self) -> Dict[int, Dict[str, any]]:
        """
        Get captured hidden states from the store.
        
        Returns tensors as serializable dictionaries for RPC transmission.
        Uses bytes serialization for much faster transmission of large tensors.
        """
        if hasattr(self, 'hidden_states_store') and self.hidden_states_store:
            self.hidden_states_store.finalize_multi_batch()
            
            result = {}
            for layer_id, tensor in self.hidden_states_store.hidden_states.items():
                cpu_tensor = tensor.cpu() if tensor.device.type != 'cpu' else tensor
                
                # Convert bfloat16 to float32 for numpy compatibility
                if cpu_tensor.dtype == torch.bfloat16:
                    cpu_tensor = cpu_tensor.to(torch.float32)
                
                np_array = cpu_tensor.numpy()
                
                result[layer_id] = {
                    'data': np_array.tobytes(),  # Much faster than .tolist()
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype)
                }
            return result
        return {}
    
    def get_hidden_states_debug_info(self) -> Dict[str, any]:
        """Get debug information about hidden states capture"""
        if hasattr(self, 'hidden_states_store') and self.hidden_states_store:
            store = self.hidden_states_store
            return {
                'capture_enabled': store.capture_enabled,
                'multi_batch_mode': store.multi_batch_mode,
                'finalized': store.finalized,
                'num_batches_captured': len(store.batch_hidden_states),
                'num_layers': store.get_layer_count(),
            }
        return {}
    
    def clear_hidden_states(self):
        """Clear stored hidden states"""
        if hasattr(self, 'hidden_states_store') and self.hidden_states_store:
            self.hidden_states_store.clear()
    
    # ========================================================================
    # MoE Router Logits Capture
    # ========================================================================
    
    def _wrap_model_for_moe_capture(self, model: nn.Module) -> nn.Module:
        """
        Wrap model MoE layers for router logits capture.
        
        This should be called during load_model() to prepare the model.
        
        Args:
            model: The model to wrap
            
        Returns:
            The wrapped model
        """
        # Initialize store if not exists
        if not hasattr(self, 'moe_router_logits_store'):
            self.moe_router_logits_store = MoERouterLogitsStore()
            self.moe_capture_enabled = False
            self._moe_wrapped = False
        
        # Skip if already wrapped
        if self._moe_wrapped:
            return model
        
        moe_layer_id = 0
        wrapped_count = 0
        
        def wrap_module(module: nn.Module, name: str = "") -> nn.Module:
            nonlocal moe_layer_id, wrapped_count
            
            # Check if this is an MoE layer
            if is_moe_layer(module, name):
                # Check if already wrapped
                if not isinstance(module, VLLMMoELayerWrapper):
                    # Extract layer ID from module name
                    layer_id = extract_moe_layer_id_from_name(name)
                    if layer_id is None:
                        # If can't extract from name, use sequential ID
                        layer_id = moe_layer_id
                    
                    wrapped_layer = VLLMMoELayerWrapper(
                        module, layer_id, name, self.moe_router_logits_store
                    )
                    moe_layer_id += 1
                    wrapped_count += 1
                    return wrapped_layer
            
            # Recursively process child modules
            for child_name, child_module in list(module.named_children()):
                full_child_name = f"{name}.{child_name}" if name else child_name
                wrapped_child = wrap_module(child_module, full_child_name)
                if wrapped_child is not child_module:
                    setattr(module, child_name, wrapped_child)
            
            return module
        
        # Wrap the model and return
        wrapped_model = wrap_module(model)
        self._moe_wrapped = True
        
        if wrapped_count > 0:
            from vllm.logger import init_logger
            logger = init_logger(__name__)
            logger.info(f"[Capture] Wrapped {wrapped_count} MoE layers for router logits capture")
        # Don't warn if no MoE layers - this is normal for non-MoE models
        
        return wrapped_model
    
    def enable_moe_router_logits_capture(self):
        """Enable MoE router logits capture."""
        if not hasattr(self, 'moe_router_logits_store'):
            raise RuntimeError(
                "MoE router logits store not initialized. "
                "This should not happen if the model was loaded properly."
            )
        
        if not self._moe_wrapped:
            # This is OK if the model doesn't have MoE layers
            import warnings
            warnings.warn(
                "Model not wrapped for MoE router logits capture. "
                "This might mean the model doesn't have MoE layers."
            )
            return
        
        self.moe_router_logits_store.enable_capture()
        self.moe_router_logits_store.clear()
        self.moe_router_logits_store.enable_multi_batch_mode()
        self.moe_capture_enabled = True
    
    def disable_moe_router_logits_capture(self):
        """Disable MoE router logits capture"""
        if hasattr(self, 'moe_router_logits_store') and self.moe_router_logits_store:
            self.moe_router_logits_store.disable_capture()
            self.moe_capture_enabled = False
    
    def get_moe_router_logits(self) -> Dict[int, Dict[str, any]]:
        """
        Get captured MoE router logits from the store.
        
        Returns tensors as serializable dictionaries for RPC transmission.
        Uses bytes serialization for much faster transmission of large tensors.
        """
        if hasattr(self, 'moe_router_logits_store') and self.moe_router_logits_store:
            self.moe_router_logits_store.finalize_multi_batch()
            
            result = {}
            for layer_id, tensor in self.moe_router_logits_store.router_logits.items():
                cpu_tensor = tensor.cpu() if tensor.device.type != 'cpu' else tensor
                
                # Convert bfloat16 to float32 for numpy compatibility
                if cpu_tensor.dtype == torch.bfloat16:
                    cpu_tensor = cpu_tensor.to(torch.float32)
                
                np_array = cpu_tensor.numpy()
                
                result[layer_id] = {
                    'data': np_array.tobytes(),  # Much faster than .tolist()
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype)
                }
            return result
        return {}
    
    def get_moe_debug_info(self) -> Dict[str, any]:
        """Get debug information about MoE router logits capture"""
        if hasattr(self, 'moe_router_logits_store') and self.moe_router_logits_store:
            store = self.moe_router_logits_store
            return {
                'capture_enabled': store.capture_enabled,
                'multi_batch_mode': store.multi_batch_mode,
                'finalized': store.finalized,
                'num_moe_layers_captured': store.get_layer_count(),
                'layer_ids': list(store.router_logits.keys()),
                'layer_names': store.get_layer_info(),
            }
        return {}
    
    def clear_moe_router_logits(self):
        """Clear stored MoE router logits"""
        if hasattr(self, 'moe_router_logits_store') and self.moe_router_logits_store:
            self.moe_router_logits_store.clear()
    
    # ========================================================================
    # Unified Capture Management (Future)
    # ========================================================================
    
    def get_all_capture_status(self) -> Dict[str, any]:
        """
        Get status of all capture features.
        
        Returns:
            Dictionary with status of all capture features
        """
        return {
            'hidden_states': {
                'enabled': getattr(self, 'hidden_states_capture_enabled', False),
                'wrapped': getattr(self, '_hidden_states_wrapped', False),
                'num_layers': self.hidden_states_store.get_layer_count() if hasattr(self, 'hidden_states_store') else 0,
            },
            'moe_router_logits': {
                'enabled': getattr(self, 'moe_capture_enabled', False),
                'wrapped': getattr(self, '_moe_wrapped', False),
                'num_layers': self.moe_router_logits_store.get_layer_count() if hasattr(self, 'moe_router_logits_store') else 0,
            },
            # Future: attention_weights, gradients, etc.
        }
    
    def clear_all_captures(self):
        """Clear all captured data"""
        self.clear_hidden_states()
        self.clear_moe_router_logits()
        # Future: clear other captures

