# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Dict, Any, Tuple
import torch
import logging

from .template import AlgorithmTemplate
from .factory import register_algorithm

logger = logging.getLogger(__name__)


@register_algorithm("linear")
class LinearTransformAlgorithm(AlgorithmTemplate):
    """
    Linear transformation algorithm: wh+b
    
    Applies a linear transformation to hidden states, including weight matrix 
    multiplication and bias vector addition.
    Uses the same weight matrix and bias vector for all target layers.
    """

    def __init__(self, layer_id=None, normalize=False):
        super().__init__(layer_id)
        # normalize is accepted for signature consistency, but not used.
        self.weights = {}  # Store weight matrix W
        self.biases = {}   # Store bias vector b
        self.scale_factors = {}  # Store scale factors
        self.active_tensor_index = None
        self.active_params: Optional[dict] = None

    def set_steer_vector(self, index: int, **kwargs):
        """Set weight matrix and bias vector."""
        payload = kwargs.get("payload")
        scale_factor = kwargs.get("scale_factor", 1.0)
        
        if not payload or "weight" not in payload:
            logger.warning(f"Missing required 'weight' in payload for layer {self.layer_id}")
            return

        # Save weight matrix and bias vector
        self.weights[index] = payload["weight"]
        self.biases[index] = payload.get("bias", None)  # Bias is optional
        self.scale_factors[index] = scale_factor

    def reset_steer_vector(self, index: int):
        """Reset vector at specific index."""
        if index in self.weights:
            del self.weights[index]
        if index in self.biases:
            del self.biases[index]
        if index in self.scale_factors:
            del self.scale_factors[index]
        if self.active_tensor_index == index:
            self.active_tensor_index = None

    def set_active_tensor(self, index: int):
        """Set currently active tensor index."""
        self.active_tensor_index = index
        if index is not None and index in self.weights:
            weight = self.weights[index]
            bias = self.biases.get(index, None)
            scale = self.scale_factors.get(index, 1.0)
            
            self.active_params = {
                "weight": weight,
                "bias": bias,
                "scale": scale
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
                "weight" in params)

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply linear transformation to single token: wh+b * scale."""
        weight = params["weight"]
        bias = params.get("bias", None)
        scale = params.get("scale", 1.0)
        
        # Ensure data types match
        device = hidden_state.device
        dtype = hidden_state.dtype
        
        weight = weight.to(device).to(dtype)
        if bias is not None:
            bias = bias.to(device).to(dtype)
        
        # Check dimension matching
        if weight.shape[0] != hidden_state.shape[-1]:
            logger.error(f"Dimension mismatch: weight shape={weight.shape}, hidden_state shape={hidden_state.shape}")
            return hidden_state
        
        # Apply weight matrix: matrix multiplication
        transformed = torch.matmul(hidden_state, weight.T)
        
        # Add bias if present
        if bias is not None:
            transformed = transformed + bias
            
        # Apply scale factor
        if scale != 1.0:
            transformed = transformed * scale
            
        return transformed

    @classmethod
    def load_from_path(cls, file_path: str, device: str, config=None, target_layers=None):
        """Load linear transformation parameters from pkl file."""
        import pickle
        import numpy as np
        import os
        
        try:
            # Load pkl file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract weight and bias
            # Check if it's a LinearTransport object, access attributes directly instead of using get method
            if hasattr(data, 'A_') and hasattr(data, 'B_'):
                # Access attributes directly
                weight = data.A_
                bias = data.B_
            elif isinstance(data, dict):
                # If it's a dictionary, use get method
                weight = data.get("A_", None)
                bias = data.get("B_", None)
            else:
                # Try direct attribute access
                try:
                    weight = getattr(data, "A_", None)
                    bias = getattr(data, "B_", None)
                except AttributeError:
                    logger.error(f"Cannot extract A_ and B_ from data type: {type(data)}")
                    raise ValueError(f"Unsupported data format. Neither a dict nor has A_/B_ attributes: {type(data)}")
            
            if weight is None:
                logger.error(f"Failed to find weight (A_) in data of type {type(data)}")
                raise ValueError(f"Weight matrix (A_) not found in pkl file")
                
            # Ensure data is numpy array or convert to numpy array
            if not isinstance(weight, np.ndarray):
                weight = np.array(weight, dtype=np.float32)
                
            if bias is not None and not isinstance(bias, np.ndarray):
                bias = np.array(bias, dtype=np.float32)
            
            # Get data type from config, with default value
            adapter_dtype = config.adapter_dtype if hasattr(config, 'adapter_dtype') else torch.float16
                
            # Convert to torch tensor and set correct dtype
            weight_tensor = torch.tensor(weight, device=device, dtype=adapter_dtype)
            bias_tensor = torch.tensor(bias, device=device, dtype=adapter_dtype) if bias is not None else None
            
            # Create payload for each target layer, using the same weight and bias
            layer_payloads = {}
            
            # If target layers are not specified, assume apply to all layers
            # We assume 48 layers here, based on configuration seen in error messages
            if target_layers is None:
                target_layers = list(range(48))  # Assume model has 48 layers
                
            for layer_idx in target_layers:
                layer_payloads[layer_idx] = {
                    "weight": weight_tensor,
                    "bias": bias_tensor
                }
                
            return {"layer_payloads": layer_payloads}
            
        except Exception as e:
            logger.error(f"Failed to load linear transform parameters from {file_path}: {e}")
            raise RuntimeError(f"Failed to load linear transform parameters") from e 