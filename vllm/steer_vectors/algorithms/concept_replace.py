# SPDX-License-Identifier: Apache-2.0
"""
Concept Replace Algorithm: Projection-Preserving Concept Substitution

This algorithm replaces one concept (h1) with another (h2) in the hidden state,
using adaptive scaling based on projection strength.

Core formula:
    h_new = h + (h · h1 / ||h1||²) * (h2 - h1)
    
Which can be decomposed as:
    λ = h · h1 / ||h1||²           # projection coefficient of h onto h1
    h_new = (h - λ*h1) + λ*h2      # erase h1 component, inject equal amount of h2

This ensures "projection preservation" - the amount of h2 injected equals
the amount of h1 erased, preventing hallucination from over-modification
or concept coverage failures from under-modification.
"""

from typing import Optional, Any, Dict
import torch
import numpy as np
import os
import glob

from .template import AlgorithmTemplate
from .factory import register_algorithm
import logging

logger = logging.getLogger(__name__)


@register_algorithm("concept_replace")
class ConceptReplaceAlgorithm(AlgorithmTemplate):
    """Concept Replace algorithm: h_new = h + λ(h2 - h1) where λ = (h·h1)/||h1||²
    
    This algorithm implements projection-preserving concept substitution:
    - Detects how much of concept h1 exists in the current hidden state
    - Replaces that exact amount with concept h2
    - Prevents hallucination from over-injection
    - Prevents coverage failure from under-injection
    
    Payload format: dict with 'h1' and 'h2' tensors
    Load from: directory containing two .gguf files (h1.gguf, h2.gguf or similar)
    """

    def _transform(self, hidden_state: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply concept replacement: h_new = h + λ(h2 - h1).
        
        Args:
            hidden_state: [batch, hidden_dim] or [hidden_dim]
            params: dict with 'h1' and 'h2' tensors, each [hidden_dim]
            
        Returns:
            Transformed hidden state with h1 concept replaced by h2
        """
        h1 = params['h1']
        h2 = params['h2']
        
        # Ensure h1, h2 are the right shape for computation
        if h1.dim() == 1:
            h1 = h1.unsqueeze(0)  # [1, hidden_dim]
        if h2.dim() == 1:
            h2 = h2.unsqueeze(0)  # [1, hidden_dim]
        
        # Compute ||h1||²
        h1_norm_sq = torch.sum(h1 * h1, dim=-1, keepdim=True)  # [1, 1]
        
        # Compute λ = (h · h1) / ||h1||²
        # hidden_state: [batch, hidden_dim], h1: [1, hidden_dim]
        dot_product = torch.sum(hidden_state * h1, dim=-1, keepdim=True)  # [batch, 1]
        lambda_coef = dot_product / (h1_norm_sq + 1e-8)  # [batch, 1]
        
        # Compute h_new = h + λ(h2 - h1)
        # This is equivalent to: h_new = (h - λ*h1) + λ*h2
        direction_diff = h2 - h1  # [1, hidden_dim]
        h_new = hidden_state + lambda_coef * direction_diff  # [batch, hidden_dim]
        
        if self.normalize:
            # Preserve original norm
            norm_pre = torch.norm(hidden_state, dim=-1, keepdim=True)
            norm_post = torch.norm(h_new, dim=-1, keepdim=True)
            return h_new * norm_pre / (norm_post + 1e-8)
        else:
            return h_new

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict:
        """Load two concept vectors from a directory containing .gguf files.
        
        Expected directory structure:
            path/
                h1.gguf (or *_h1.gguf, or first alphabetically)
                h2.gguf (or *_h2.gguf, or second alphabetically)
        
        The algorithm looks for files in this order:
        1. Files explicitly named h1.gguf and h2.gguf
        2. Files containing '_h1' and '_h2' in their names
        3. First two .gguf files sorted alphabetically
        """
        import os
        
        config = kwargs.get("config")
        if config is None:
            raise ValueError("ConceptReplaceAlgorithm.load_from_path requires 'config' in kwargs")

        if not os.path.isdir(path):
            raise ValueError(f"ConceptReplaceAlgorithm requires a directory path, got: {path}")
        
        return cls._load_from_directory(path, device, **kwargs)
    
    @classmethod
    def _load_from_directory(cls, path: str, device: str, **kwargs) -> dict:
        """Load h1 and h2 vectors from directory containing two .gguf files."""
        import gguf
        import numpy as np
        
        config = kwargs.get("config")
        
        # Find all .gguf files in directory
        gguf_files = sorted(glob.glob(os.path.join(path, "*.gguf")))
        
        if len(gguf_files) < 2:
            raise ValueError(f"ConceptReplaceAlgorithm requires at least 2 .gguf files in directory, found {len(gguf_files)}")
        
        # Determine which file is h1 and which is h2
        h1_path = None
        h2_path = None
        
        for f in gguf_files:
            basename = os.path.basename(f).lower()
            if basename == "h1.gguf" or "_h1" in basename:
                h1_path = f
            elif basename == "h2.gguf" or "_h2" in basename:
                h2_path = f
        
        # Fallback: use first two files alphabetically
        if h1_path is None:
            h1_path = gguf_files[0]
        if h2_path is None:
            # Find the first file that isn't h1_path
            for f in gguf_files:
                if f != h1_path:
                    h2_path = f
                    break
        
        logger.debug(f"Loading concept vectors: h1={h1_path}, h2={h2_path}")
        
        # Load h1
        h1_weights = cls._load_single_gguf(h1_path, device, config)
        
        # Load h2
        h2_weights = cls._load_single_gguf(h2_path, device, config)
        
        # Merge into layer_payloads with both h1 and h2
        layer_payloads = {}
        
        # Get all layer indices from both files
        all_layers = set(h1_weights.keys()) | set(h2_weights.keys())
        
        for layer_idx in all_layers:
            if layer_idx not in h1_weights:
                raise ValueError(f"Layer {layer_idx} found in h2 but not in h1")
            if layer_idx not in h2_weights:
                raise ValueError(f"Layer {layer_idx} found in h1 but not in h2")
            
            layer_payloads[layer_idx] = {
                'h1': h1_weights[layer_idx],
                'h2': h2_weights[layer_idx]
            }
        
        return {"layer_payloads": layer_payloads}
    
    @classmethod
    def _load_single_gguf(cls, path: str, device: str, config) -> Dict[int, torch.Tensor]:
        """Load a single .gguf file and return layer->tensor mapping."""
        import gguf
        import numpy as np
        
        reader = gguf.GGUFReader(path)
        
        weights = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except (ValueError, IndexError) as e:
                raise ValueError(f".gguf file has invalid direction field name: {tensor.name}") from e
            
            np_copy = np.array(tensor.data, copy=True)
            weights[layer] = torch.from_numpy(np_copy).to(device).to(config.adapter_dtype)
        
        return weights

