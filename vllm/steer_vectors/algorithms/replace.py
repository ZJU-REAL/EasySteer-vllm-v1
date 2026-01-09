# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Any
import torch
import numpy as np

from .template import AlgorithmTemplate
from .factory import register_algorithm
import logging
logger = logging.getLogger(__name__)

@register_algorithm("replace")
class ReplaceAlgorithm(AlgorithmTemplate):
    """Replace algorithm: h' = vector
    
    This algorithm replaces the hidden state with the given vector.
    - Only 2 methods needed: _transform and load_from_path
    - All parameter management is handled by AlgorithmTemplate
    - Payload is a simple Tensor
    - Only supports GGUF file format
    """

    def _transform(self, hidden_state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply replacement: h' = vector (with optional normalization)."""
        if self.normalize:
            norm_pre = torch.norm(hidden_state, dim=-1, keepdim=True)
            # Expand params to match the batch dimension if needed
            if params.dim() == 1 and hidden_state.dim() == 2:
                # params: [hidden_dim], hidden_state: [batch, hidden_dim]
                replaced = params.unsqueeze(0).expand_as(hidden_state)
            else:
                replaced = params
            norm_post = torch.norm(replaced, dim=-1, keepdim=True)
            return replaced * norm_pre / (norm_post + 1e-8)
        else:
            # Expand params to match the batch dimension if needed
            if params.dim() == 1 and hidden_state.dim() == 2:
                return params.unsqueeze(0).expand_as(hidden_state)
            return params

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict:
        """Load Replace steer vector from GGUF file."""
        import os
        
        config = kwargs.get("config")
        if config is None:
            raise ValueError("ReplaceAlgorithm.load_from_path requires 'config' in kwargs")

        file_ext = os.path.splitext(path)[1].lower()
        
        if file_ext != '.gguf':
            raise ValueError(f"ReplaceAlgorithm only supports .gguf files, got: {file_ext}")
        
        return cls._load_from_gguf(path, device, **kwargs)
    
    @classmethod
    def _load_from_gguf(cls, path: str, device: str, **kwargs) -> dict:
        """Load Replace steer vector from GGUF file."""
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

