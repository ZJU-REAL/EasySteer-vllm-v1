# SPDX-License-Identifier: Apache-2.0
"""
MoE Router Logits Storage for vLLM

Storage functionality for capturing and managing router logits from MoE layers.
"""

from typing import Dict, List, Optional, Any
import torch
import threading


class MoERouterLogitsStore:
    """Storage for router logits from MoE layers"""

    def __init__(self):
        self.router_logits: Dict[int, torch.Tensor] = {}  # layer_id -> router_logits
        self.layer_names: Dict[int, str] = {}  # layer_id -> layer_name
        self.capture_enabled = False
        self.lock = threading.Lock()
        
        # Multi-batch support (similar to HiddenStatesStore)
        self.batch_router_logits: List[Dict[int, torch.Tensor]] = []
        self.multi_batch_mode = False
        self.finalized = False
        self.first_moe_layer_call_count = 0  # Track first MoE layer calls

    def clear(self):
        """Clear all stored router logits"""
        with self.lock:
            self.router_logits.clear()
            self.layer_names.clear()
            self.batch_router_logits.clear()
            self.multi_batch_mode = False
            self.finalized = False
            self.first_moe_layer_call_count = 0
            
            # Note: torch.cuda.empty_cache() can be very slow (seconds to minutes)
            # especially with multi-GPU setups and large memory allocations.
            # Since router logits are stored on CPU, we don't need to clear GPU cache.
            # The GPU cache will be managed by PyTorch automatically.

    def enable_capture(self):
        """Enable router logits capture"""
        with self.lock:
            self.capture_enabled = True

    def disable_capture(self):
        """Disable router logits capture"""
        with self.lock:
            self.capture_enabled = False

    def store_router_logits(self, layer_id: int, router_logits: torch.Tensor, layer_name: str = ""):
        """
        Store router logits for a specific MoE layer
        
        Args:
            layer_id: MoE layer ID
            router_logits: Router logits tensor (num_tokens, n_experts)
            layer_name: Optional layer name
        """
        if not (self.capture_enabled and isinstance(router_logits, torch.Tensor)):
            return
            
        with self.lock:
            if self.finalized:
                return
            
            # Detect new forward pass in multi-batch mode
            # We track first MoE layer calls
            first_moe_layer_id = min(self.router_logits.keys()) if self.router_logits else layer_id
            if self.multi_batch_mode and layer_id == first_moe_layer_id:
                self.first_moe_layer_call_count += 1
                
                # If this is the 2nd+ call to first MoE layer, we're starting a new batch
                if self.first_moe_layer_call_count > 1 and self.router_logits:
                    self.finish_current_batch()
            
            # Move to CPU and clone to avoid modifications
            cpu_router_logits = router_logits.detach().cpu().clone()
            self.router_logits[layer_id] = cpu_router_logits
            self.layer_names[layer_id] = layer_name

    def get_all_router_logits(self, device: str = 'cpu') -> Dict[int, torch.Tensor]:
        """
        Get all router logits in a dictionary mapping layer_id to tensor
        
        Args:
            device: Target device for tensors
            
        Returns:
            Dict mapping layer_id to router_logits tensor
        """
        with self.lock:
            result = {}
            target_device = torch.device(device)
            
            for layer_id, tensor in self.router_logits.items():
                if device != 'cpu' and tensor.device != target_device:
                    tensor = tensor.to(target_device)
                result[layer_id] = tensor
                
            return result

    def get_router_logits(self, layer_id: int) -> Optional[torch.Tensor]:
        """Get router logits for a specific layer"""
        with self.lock:
            return self.router_logits.get(layer_id)

    def get_layer_count(self) -> int:
        """Get the number of captured MoE layers"""
        with self.lock:
            return len(self.router_logits)

    def get_layer_info(self) -> Dict[int, str]:
        """Get layer ID to name mapping"""
        with self.lock:
            return self.layer_names.copy()
    
    def enable_multi_batch_mode(self):
        """Enable multi-batch capture mode"""
        with self.lock:
            self.multi_batch_mode = True
            self.first_moe_layer_call_count = 0
    
    def finish_current_batch(self):
        """Mark the current batch as finished"""
        if self.multi_batch_mode and self.router_logits:
            self.batch_router_logits.append(self.router_logits.copy())
            self.router_logits.clear()
            self.first_moe_layer_call_count = 0
    
    def finalize_multi_batch(self):
        """Finalize multi-batch capture by combining all batches"""
        with self.lock:
            if not self.multi_batch_mode:
                return
                
            if self.router_logits:
                self.finish_current_batch()
            
            if not self.batch_router_logits:
                return
            
            # If only one batch, just use it directly
            if len(self.batch_router_logits) == 1:
                self.router_logits = self.batch_router_logits[0]
            else:
                # Combine multiple batches
                combined_router_logits = {}
                all_layer_ids = set()
                for batch in self.batch_router_logits:
                    all_layer_ids.update(batch.keys())
                
                # Concatenate directly on CPU (much faster for multi-batch)
                # Avoids costly CPU<->GPU transfers
                for layer_id in sorted(all_layer_ids):
                    layer_tensors = []
                    for batch in self.batch_router_logits:
                        if layer_id in batch:
                            tensor = batch[layer_id]
                            # Ensure on CPU
                            if tensor.device.type != 'cpu':
                                tensor = tensor.cpu()
                            layer_tensors.append(tensor)
                    
                    if layer_tensors:
                        # Concatenate on CPU along token dimension (dim=0)
                        combined_tensor = torch.cat(layer_tensors, dim=0)
                        combined_router_logits[layer_id] = combined_tensor
                        del layer_tensors
                
                self.router_logits = combined_router_logits
            
            self.batch_router_logits.clear()
            self.multi_batch_mode = False
            
            # Don't call empty_cache() here - it's extremely slow and unnecessary
            # since we're working with CPU tensors
            
            self.finalized = True

