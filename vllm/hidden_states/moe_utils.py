# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for MoE Router Logits Capture
"""

import torch
import numpy as np
from typing import Dict, Any


def deserialize_moe_router_logits(serialized_data: Dict[int, Dict[str, Any]]) -> Dict[int, torch.Tensor]:
    """
    Deserialize MoE router logits from RPC-transferred format back to tensors.
    
    Args:
        serialized_data: Dictionary mapping layer_id to serialized tensor info:
            {
                'data': bytes - binary tensor data in float32 format,
                'shape': list of ints,
                'dtype': str (e.g., 'torch.bfloat16') - original dtype
            }
    
    Returns:
        Dictionary mapping layer_id to torch.Tensor with original dtype
    
    Example:
        >>> results = llm.llm_engine.engine_core.collective_rpc("get_moe_router_logits")
        >>> serialized = results[0]
        >>> router_logits = deserialize_moe_router_logits(serialized)
        >>> # Now router_logits[layer_id] is a real tensor with correct dtype
    """
    tensors = {}
    
    for layer_id, tensor_info in serialized_data.items():
        # Extract info
        buffer = tensor_info['data']  # bytes object
        shape = tuple(tensor_info['shape'])
        dtype_str = tensor_info['dtype']
        
        # Convert dtype string to torch dtype
        dtype_map = {
            'torch.float32': torch.float32,
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.float64': torch.float64,
            'torch.int32': torch.int32,
            'torch.int64': torch.int64,
        }
        original_dtype = dtype_map.get(dtype_str, torch.float32)
        
        # Directly create numpy array from bytes (much faster than from list!)
        # Data is always in float32 format (for numpy compatibility)
        np_array = np.frombuffer(buffer, dtype=np.float32).reshape(shape)
        
        # Copy to avoid potential issues with buffer lifecycle
        tensor = torch.from_numpy(np_array.copy())
        
        # Convert back to original dtype if needed
        if original_dtype != tensor.dtype:
            tensor = tensor.to(original_dtype)
        
        tensors[layer_id] = tensor
    
    return tensors


def print_moe_router_logits_summary(router_logits: Dict[int, torch.Tensor]):
    """
    Print a summary of captured MoE router logits.
    
    Args:
        router_logits: Dictionary mapping layer_id to torch.Tensor
    """
    print(f"📊 Captured {len(router_logits)} MoE layers:")
    for layer_id in sorted(router_logits.keys()):
        tensor = router_logits[layer_id]
        num_tokens, n_experts = tensor.shape
        print(f"  Layer {layer_id:2d}: {num_tokens} tokens × {n_experts} experts, "
              f"dtype {tensor.dtype}, device {tensor.device}")

