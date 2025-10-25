# SPDX-License-Identifier: Apache-2.0
"""
Hidden States Storage for vLLM

Storage functionality for capturing and managing hidden states.
"""

from typing import Dict, List, Optional, Any
import torch
import threading


class HiddenStatesStore:
    """Storage for hidden states from transformer layers"""

    def __init__(self):
        self.hidden_states: Dict[int, torch.Tensor] = {}  # layer_id -> hidden_state
        self.layer_names: Dict[int, str] = {}  # layer_id -> layer_name
        self.capture_enabled = False
        self.lock = threading.Lock()
        self.pooling_metadata = None
        
        # Multi-batch support
        self.batch_hidden_states: List[Dict[int, torch.Tensor]] = []
        self.batch_pooling_metadata: List[Any] = []
        self.multi_batch_mode = False
        self.finalized = False
        self.layer_0_call_count = 0  # Track layer 0 calls to detect new batches

    def clear(self):
        """Clear all stored hidden states"""
        with self.lock:
            self.hidden_states.clear()
            self.layer_names.clear()
            self.pooling_metadata = None
            self.batch_hidden_states.clear()
            self.batch_pooling_metadata.clear()
            self.multi_batch_mode = False
            self.finalized = False
            self.layer_0_call_count = 0  # Reset counter
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def enable_capture(self):
        """Enable hidden states capture"""
        with self.lock:
            self.capture_enabled = True

    def disable_capture(self):
        """Disable hidden states capture"""
        with self.lock:
            self.capture_enabled = False

    def store_hidden_state(self, layer_id: int, hidden_state: torch.Tensor, layer_name: str = ""):
        """Store hidden state for a specific layer"""
        if not (self.capture_enabled and isinstance(hidden_state, torch.Tensor)):
            return
            
        with self.lock:
            if self.finalized:
                return
            
            # Detect new forward pass in multi-batch mode
            # We track layer 0 calls: if it's called again, it means a new batch started
            if self.multi_batch_mode and layer_id == 0:
                self.layer_0_call_count += 1
                
                # If this is the 2nd+ call to layer 0, we're starting a new batch
                if self.layer_0_call_count > 1 and self.hidden_states:
                    self.finish_current_batch()
            
            # Move to CPU and clone to avoid modifications
            cpu_hidden_state = hidden_state.detach().cpu().clone()
            self.hidden_states[layer_id] = cpu_hidden_state
            self.layer_names[layer_id] = layer_name

    def get_all_hidden_states(self, device: str = 'cpu') -> List[torch.Tensor]:
        """Get all hidden states in layer order"""
        with self.lock:
            sorted_layers = sorted(self.hidden_states.keys())
            result = []
            
            target_device = torch.device(device)
            for layer_id in sorted_layers:
                tensor = self.hidden_states[layer_id]
                if device != 'cpu' and tensor.device != target_device:
                    tensor = tensor.to(target_device)
                result.append(tensor)
                
            return result

    def get_hidden_state(self, layer_id: int) -> Optional[torch.Tensor]:
        """Get hidden state for a specific layer"""
        with self.lock:
            return self.hidden_states.get(layer_id)

    def get_layer_count(self) -> int:
        """Get the number of captured layers"""
        with self.lock:
            return len(self.hidden_states)

    def get_layer_info(self) -> Dict[int, str]:
        """Get layer ID to name mapping"""
        with self.lock:
            return self.layer_names.copy()
    
    def get_pooling_metadata(self):
        """Get captured pooling metadata"""
        with self.lock:
            return self.pooling_metadata
    
    def set_pooling_metadata(self, metadata):
        """Set pooling metadata"""
        with self.lock:
            if not self.finalized:
                self.pooling_metadata = metadata
    
    def enable_multi_batch_mode(self):
        """Enable multi-batch capture mode"""
        with self.lock:
            self.multi_batch_mode = True
            self.layer_0_call_count = 0  # Initialize counter
    
    def finish_current_batch(self):
        """Mark the current batch as finished"""
        if self.multi_batch_mode and self.hidden_states:
            self.batch_hidden_states.append(self.hidden_states.copy())
            self.batch_pooling_metadata.append(self.pooling_metadata)
            self.hidden_states.clear()
            self.pooling_metadata = None
            # Reset layer 0 counter for the new batch
            self.layer_0_call_count = 0
    
    def finalize_multi_batch(self):
        """Finalize multi-batch capture by combining all batches"""
        with self.lock:
            if not self.multi_batch_mode:
                return
                
            if self.hidden_states:
                self.finish_current_batch()
            
            if not self.batch_hidden_states:
                return
            
            if len(self.batch_hidden_states) == 1:
                self.hidden_states = self.batch_hidden_states[0]
                self.pooling_metadata = self.batch_pooling_metadata[0]
            else:
                combined_hidden_states = {}
                all_layer_ids = set()
                for batch in self.batch_hidden_states:
                    all_layer_ids.update(batch.keys())
                
                device = None
                for layer_id in sorted(all_layer_ids):
                    layer_tensors = []
                    for batch in self.batch_hidden_states:
                        if layer_id in batch:
                            tensor = batch[layer_id]
                            if device is None:
                                if torch.cuda.is_available():
                                    device = torch.device('cuda')
                                else:
                                    device = tensor.device
                            layer_tensors.append(tensor.to(device))
                    
                    if layer_tensors:
                        combined_tensor = torch.cat(layer_tensors, dim=0)
                        combined_hidden_states[layer_id] = combined_tensor.cpu()
                        del layer_tensors
                        if device and device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                self.hidden_states = combined_hidden_states
                
                if self.batch_pooling_metadata:
                    for metadata in self.batch_pooling_metadata:
                        if metadata is not None:
                            self.pooling_metadata = metadata
                            break
            
            self.batch_hidden_states.clear()
            self.batch_pooling_metadata.clear()
            self.multi_batch_mode = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.finalized = True

