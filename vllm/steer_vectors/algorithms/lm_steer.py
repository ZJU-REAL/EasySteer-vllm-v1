# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any, Tuple
import torch
import logging

from .template import AlgorithmTemplate
from .factory import register_algorithm

logger = logging.getLogger(__name__)


@register_algorithm("lm_steer")
class LMSteerAlgorithm(AlgorithmTemplate):
    """
    LM-Steer algorithm.
    
    Uses low-rank optimization form: h = h + α*((h·P1)·P2^T)
    where P1 and P2 are low-rank projection matrices, · denotes matrix multiplication.
    """

    def __init__(self, layer_id=None, normalize=False):
        super().__init__(layer_id)
        # normalize is accepted for signature consistency, but not used.
        self.projector1 = {}  # Store projection matrix P1
        self.projector2 = {}  # Store projection matrix P2
        self.scale_factors = {}  # Store scale factor α
        self.active_tensor_index = None
        self.active_params: Optional[dict] = None

    def set_steer_vector(self, index: int, **kwargs):
        """Set projection matrices and scale factor."""
        payload = kwargs.get("payload")
        scale_factor = kwargs.get("scale_factor", 1.0)
        
        if not payload:
            logger.warning(f"Missing payload for layer {self.layer_id}")
            return
            
        # Check if required projector matrices are present
        if "projector1" in payload and "projector2" in payload:
            self.projector1[index] = payload["projector1"]
            self.projector2[index] = payload["projector2"]
            # logger.info(f"Set projector matrices for index {index}")
        else:
            logger.warning(f"Missing required 'projector1'/'projector2' in payload for layer {self.layer_id}")
            return

        self.scale_factors[index] = scale_factor

    def reset_steer_vector(self, index: int):
        """Reset vector at specific index."""
        if index in self.projector1:
            del self.projector1[index]
        if index in self.projector2:
            del self.projector2[index]
        if index in self.scale_factors:
            del self.scale_factors[index]
        if self.active_tensor_index == index:
            self.active_tensor_index = None

    def set_active_tensor(self, index: int):
        """Set currently active tensor index."""
        self.active_tensor_index = index
        if index is not None and index in self.projector1 and index in self.projector2:
            P1 = self.projector1[index]
            P2 = self.projector2[index]
            alpha = self.scale_factors.get(index, 1.0)
            
            # Select the first steer vector (index 0) if multi-dimensional
            if P1.dim() > 2:
                P1_active = P1[0]  # shape: [embed_dim, rank]
            else:
                P1_active = P1
                
            if P2.dim() > 2:
                P2_active = P2[0]  # shape: [embed_dim, rank]
            else:
                P2_active = P2
            
            self.active_params = {
                "P1": P1_active,
                "P2": P2_active,
                "alpha": alpha
            }
        else:
            self.active_params = None

    # Implement abstract methods required by algorithm template
    def _get_params(self) -> Optional[dict]:
        """Get currently active algorithm parameters."""
        return self.active_params

    def _is_valid(self, params: Any) -> bool:
        """Check if algorithm parameters are valid."""
        return (params is not None and 
                isinstance(params, dict) and 
                "P1" in params and 
                "P2" in params)

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply LM-Steer transformation to single token: h = h + α*((h·P1)·P2^T)."""
        P1 = params["P1"]
        P2 = params["P2"]
        alpha = params.get("alpha", 1.0)
        
        # Ensure data types match
        device = hidden_state.device
        dtype = hidden_state.dtype
        
        P1 = P1.to(device).to(dtype)
        P2 = P2.to(device).to(dtype)
        
        # Apply low-rank transformation: (h·P1)·P2^T
        transformed = torch.matmul(hidden_state, P1)  # [..., rank]
        transformed = torch.matmul(transformed, P2.transpose(-2, -1))  # [..., hidden_dim]
        
        # Add original hidden state: h = h + α*((h·P1)·P2^T)
        return hidden_state + alpha * transformed

    @classmethod
    def load_from_path(cls, file_path: str, device: str, config=None, target_layers=None):
        """Load LM-Steer parameters from pt file."""
        import os
        
        try:
            # Load pt file, set weights_only=False to allow loading argparse.Namespace and other objects
            state_dict = torch.load(file_path, map_location=device, weights_only=False)
            
            # Extract projection matrices
            projector1 = None
            projector2 = None
            
            # Check if it's a list structure (handles gpt2.pt special structure)
            if isinstance(state_dict, list) and len(state_dict) > 1:
                # logger.info(f"Detected list structure, trying to extract parameters from element[1]")
                params_dict = state_dict[1]
                
                if isinstance(params_dict, dict) and 'projector1' in params_dict and 'projector2' in params_dict:
                    # This is the low-rank optimization form
                    # logger.info(f"Found low-rank optimization projector1 and projector2 parameters")
                    projector1 = params_dict['projector1']
                    projector2 = params_dict['projector2']
            # Check if it's a dictionary structure
            elif isinstance(state_dict, dict):
                if "projector1" in state_dict and "projector2" in state_dict:
                    # This is the low-rank optimization form
                    # logger.info(f"Found low-rank optimization projector1 and projector2 parameters")
                    projector1 = state_dict["projector1"]
                    projector2 = state_dict["projector2"]
            
            # If projection matrices not found, raise error
            if projector1 is None or projector2 is None:
                logger.error(f"Could not find projector matrices in file {file_path}")
                raise ValueError(f"Projector matrices not found in pt file")
            
            # Get data type from config, with default value
            adapter_dtype = config.adapter_dtype if hasattr(config, 'adapter_dtype') else torch.float16
                
            # Create payload for each target layer
            layer_payloads = {}
            
            # If target layers are not specified, assume apply to all layers
            if target_layers is None:
                # Try to get number of layers from config
                if hasattr(config, 'num_hidden_layers'):
                    target_layers = list(range(config.num_hidden_layers))
                else:
                    # Default to 32 layers
                    target_layers = list(range(32))
            
            # Ensure it's a tensor and convert data type
            projector1_tensor = projector1.to(device=device, dtype=adapter_dtype)
            projector2_tensor = projector2.to(device=device, dtype=adapter_dtype)
            
            for layer_idx in target_layers:
                layer_payloads[layer_idx] = {
                    "projector1": projector1_tensor,
                    "projector2": projector2_tensor
                }
            # logger.info(f"Loaded low-rank projection matrices P1: {projector1_tensor.shape}, P2: {projector2_tensor.shape}")
                
            return {"layer_payloads": layer_payloads}
            
        except Exception as e:
            logger.error(f"Failed to load LM-Steer parameters from {file_path}: {e}")
            raise RuntimeError(f"Failed to load LM-Steer parameters") from e 