# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any
import torch
import numpy as np

from .template import AlgorithmTemplate
from .factory import register_algorithm
import logging
logger = logging.getLogger(__name__)

@register_algorithm("erase")
class EraseAlgorithm(AlgorithmTemplate):
    """Erase algorithm: h' = h - proj_{h1}(h)
    
    This algorithm erases the component of hidden state in the direction of h1.
    Formula: h_perp = h - (h · h1 / ||h1||^2) * h1
    
    The result is the component of h that is orthogonal to h1.
    - Only 2 methods needed: _transform and load_from_path
    - All parameter management is handled by AlgorithmTemplate
    - Payload is a simple Tensor (the direction vector h1)
    - Only supports GGUF file format
    """

    def _transform(self, hidden_state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply erasure: h' = h - proj_{h1}(h) (with optional normalization).
        
        Args:
            hidden_state: [batch, hidden_dim] or [hidden_dim]
            params: [hidden_dim] the direction vector h1 to erase
            
        Returns:
            The component of hidden_state orthogonal to params (h1)
        """
        # Ensure params is the right shape for computation
        h1 = params
        if h1.dim() == 1:
            h1 = h1.unsqueeze(0)  # [1, hidden_dim]
        
        # Compute ||h1||^2
        h1_norm_sq = torch.sum(h1 * h1, dim=-1, keepdim=True)  # [1, 1]
        
        # Compute h · h1 (dot product along last dimension)
        # hidden_state: [batch, hidden_dim], h1: [1, hidden_dim]
        dot_product = torch.sum(hidden_state * h1, dim=-1, keepdim=True)  # [batch, 1]
        
        # Compute projection scalar: (h · h1) / ||h1||^2
        proj_scalar = dot_product / (h1_norm_sq + 1e-8)  # [batch, 1]
        
        # Compute projection vector: proj_scalar * h1
        proj_vector = proj_scalar * h1  # [batch, hidden_dim]
        
        # Compute h_perp = h - proj_{h1}(h)
        h_perp = hidden_state - proj_vector
        
        if self.normalize:
            # Preserve original norm
            norm_pre = torch.norm(hidden_state, dim=-1, keepdim=True)
            norm_post = torch.norm(h_perp, dim=-1, keepdim=True)
            return h_perp * norm_pre / (norm_post + 1e-8)
        else:
            return h_perp

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict:
        """Load Erase direction vector from GGUF file."""
        import os
        
        config = kwargs.get("config")
        if config is None:
            raise ValueError("EraseAlgorithm.load_from_path requires 'config' in kwargs")

        file_ext = os.path.splitext(path)[1].lower()
        
        if file_ext != '.gguf':
            raise ValueError(f"EraseAlgorithm only supports .gguf files, got: {file_ext}")
        
        return cls._load_from_gguf(path, device, **kwargs)
    
    @classmethod
    def _load_from_gguf(cls, path: str, device: str, **kwargs) -> dict:
        """Load Erase direction vector from GGUF file."""
        import gguf
        import numpy as np
        
        config = kwargs.get("config")
        
        reader = gguf.GGUFReader(path)
        
        # Validate file type
        archf = reader.get_field("general.architecture")
        if archf and len(archf.parts):
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "steervector" and arch != "controlvector":
                # Only log, don't enforce
                # logger.warning(".gguf file with arch %s may not be a steer vector", arch)
                pass

        sv_weights = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except (ValueError, IndexError) as e:
                raise ValueError(f".gguf file has invalid direction field name: {tensor.name}") from e
            
            np_copy = np.array(tensor.data, copy=True)
            sv_weights[layer] = torch.from_numpy(np_copy).to(device).to(config.adapter_dtype)
            
        return {"layer_payloads": sv_weights}

