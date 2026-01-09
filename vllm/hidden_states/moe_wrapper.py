# SPDX-License-Identifier: Apache-2.0
"""
vLLM MoE Layer Wrappers

Wrappers for MoE layers to capture router logits from gate modules.
"""

from typing import Optional
import torch
from torch import nn


# MoE layer class names for different architectures
MOE_LAYER_CLASSES = [
    # Qwen family
    'Qwen2MoeSparseMoeBlock',
    'Qwen3MoeSparseMoeBlock',
    'Qwen3NextSparseMoeBlock',
    'QwenMoE',
    'Qwen2MoE',
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
    'GLMMoE',
    # Ernie
    'Ernie4MoE',
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


class VLLMMoELayerWrapper(nn.Module):
    """
    Wrapper for vLLM MoE layers to capture router logits.
    
    This wrapper intercepts the gate/router output to capture routing decisions
    before the experts process the tokens.
    """

    def __init__(self, base_layer: nn.Module, layer_id: int, layer_name: str = "", 
                 store=None) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.layer_id = layer_id
        self.layer_name = layer_name or f"moe_layer_{layer_id}"
        self.store = store
        
        # Identify gate/router and experts modules
        self.gate = getattr(base_layer, 'gate', None)
        self.router = getattr(base_layer, 'router', None)
        self.experts = getattr(base_layer, 'experts', None)
        
        # Use gate or router (different models use different names)
        self.routing_module = self.gate if self.gate is not None else self.router

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        Forward pass that captures router logits
        
        This method intercepts the MoE forward to capture router logits
        before experts processing.
        """
        # Try to capture router logits if we have the routing module
        if self.store is not None and self.routing_module is not None:
            router_logits = self._extract_router_logits(hidden_states)
            
            if router_logits is not None:
                self.store.store_router_logits(
                    self.layer_id,
                    router_logits,
                    self.layer_name
                )
        
        # Call original layer's forward method
        output = self.base_layer(hidden_states, *args, **kwargs)
        return output

    def _extract_router_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract router logits from the gate/router module
        
        Args:
            hidden_states: Input hidden states to the MoE layer
            
        Returns:
            Router logits tensor (num_tokens, n_experts) or None
        """
        if self.routing_module is None:
            return None
        
        try:
            # Flatten hidden states if needed (some MoE layers expect 2D input)
            orig_shape = hidden_states.shape
            if hidden_states.ndim > 2:
                hidden_size = hidden_states.shape[-1]
                hidden_states_flat = hidden_states.view(-1, hidden_size)
            else:
                hidden_states_flat = hidden_states
            
            # Call gate/router to get logits
            # Most gates return (logits, bias) or just logits
            gate_output = self.routing_module(hidden_states_flat)
            
            if isinstance(gate_output, tuple):
                # (router_logits, bias) format
                router_logits = gate_output[0]
            elif isinstance(gate_output, torch.Tensor):
                # Direct tensor output
                router_logits = gate_output
            else:
                return None
            
            # Ensure router_logits is 2D (num_tokens, n_experts)
            if router_logits.ndim == 1:
                router_logits = router_logits.unsqueeze(0)
            
            return router_logits
            
        except Exception as e:
            # If extraction fails, silently return None
            # (We don't want to break the forward pass)
            return None

    def __getattr__(self, name):
        """Delegate attribute access to the base layer"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)


def is_moe_layer(module: nn.Module, module_name: str = "") -> bool:
    """
    Check if a module is an MoE layer
    
    Args:
        module: The module to check
        module_name: Full module name (e.g., 'model.layers.10.block_sparse_moe')
        
    Returns:
        True if this is an MoE layer
    """
    # Check by class name
    class_name = type(module).__name__
    if class_name in MOE_LAYER_CLASSES:
        return True
    
    # Check by module name pattern
    module_name_lower = module_name.lower()
    moe_patterns = [
        'block_sparse_moe',
        'moe_layer',
        'moe',
        'sparse_moe',
        'experts',
    ]
    
    for pattern in moe_patterns:
        if pattern in module_name_lower:
            # Additional check: make sure it has gate/router and experts
            has_gate = hasattr(module, 'gate') or hasattr(module, 'router')
            has_experts = hasattr(module, 'experts')
            if has_gate and has_experts:
                return True
    
    return False


def extract_moe_layer_id_from_name(module_name: str) -> Optional[int]:
    """
    Extract layer ID from MoE module name
    
    Args:
        module_name: Module name like 'model.layers.10.block_sparse_moe'
        
    Returns:
        Layer ID as integer, or None if not found
        
    Examples:
        'model.layers.10.block_sparse_moe' -> 10
        'transformer.h.12.mlp' -> 12
        'model.moe_layer' -> None (no layer number)
    """
    parts = module_name.split('.')
    
    # Look for numeric parts
    for i, part in enumerate(parts):
        if part.isdigit():
            # Check if previous part suggests this is a layer index
            if i > 0:
                prev_part = parts[i - 1].lower()
                if prev_part in ['layers', 'h', 'layer', 'blocks', 'block']:
                    return int(part)
    
    return None

