# SPDX-License-Identifier: Apache-2.0
"""
MoE Router Logits Intervention Algorithm

This algorithm allows steering MoE model behavior by modifying router logits,
which control expert selection probabilities.
"""

from typing import Optional, Any, List
import torch
import json

from .template import AlgorithmTemplate
from .factory import register_algorithm
import logging

logger = logging.getLogger(__name__)


@register_algorithm("moe_router")
class MoERouterAlgorithm(AlgorithmTemplate):
    """
    MoE Router Logits intervention algorithm.
    
    This algorithm modifies router logits to boost or suppress specific experts.
    The intervention happens BEFORE softmax, directly manipulating the raw logits.
    
    Supported modes:
    - 'boost': Set logits[expert_ids] = max(all_logits) - amplify specified experts
    - 'suppress': Set logits[expert_ids] = min(all_logits) - dampen specified experts
    - 'soft': z'_k = z_k + lambda * std(z) - soft intervention based on logits std
    - 'soft_random': z'_k = z_k + lambda * std(z) - soft intervention on random experts (same count as expert_ids)
    - 'soft_hard': Set logits[expert_ids] = max(all_logits) + small_random - boost with random perturbation to avoid ties
    
    Payload format (dict):
    {
        'expert_ids': [1, 5, 10],  # List of expert IDs to intervene (for soft_random: determines count only)
        'mode': 'boost',            # 'boost', 'suppress', 'soft', 'soft_topk', 'soft_random', or 'soft_hard'
        'lambda': 0.5,              # (Optional) For 'soft'/'soft_random' modes: intervention strength
    }
    """
    
    def __init__(self, layer_id: Optional[int] = None, **kwargs):
        # MoE router doesn't use normalize parameter - remove it from kwargs if present
        kwargs.pop('normalize', None)
        super().__init__(layer_id=layer_id, normalize=False, **kwargs)
    
    # Inherit apply_intervention from AlgorithmTemplate - it has full trigger support!
    # We only need to implement _transform for MoE-specific logic
    
    def _transform(self, router_logits: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Apply intervention to router logits.
        
        Args:
            router_logits: (num_tokens, n_experts) - raw logits from gate
            params: Intervention parameters dict with keys:
                - expert_ids: List[int] - expert indices to modify (or count for soft_random)
                - mode: str - 'boost', 'suppress', 'soft', 'soft_topk', 'soft_random', or 'soft_hard'
                - lambda: float - (for 'soft' modes) intervention strength
                - topk: int - (for 'soft_topk' mode) only intervene if expert is NOT in top-k
                
        Returns:
            Modified router_logits with same shape
        """
        expert_ids = params.get('expert_ids', [])
        mode = params.get('mode', 'boost')
        lambda_param = params.get('lambda', 0.5)  # Default lambda for soft modes
        topk_param = params.get('topk', 8)  # Default top-k for soft_topk mode
        
        if not expert_ids:
            # No experts specified, return original
            return router_logits
        
        # Validate expert_ids
        n_experts = router_logits.shape[-1]
        expert_ids = [eid for eid in expert_ids if 0 <= eid < n_experts]
        
        if not expert_ids:
            logger.warning(f"No valid expert IDs found in range [0, {n_experts})")
            return router_logits
        
        # Clone to avoid modifying original
        modified_logits = router_logits.clone()
        
        if mode == 'boost':
            # Boost: Set specified experts' logits to the maximum logit value
            # This makes them most likely to be selected
            max_logits = modified_logits.max(dim=-1, keepdim=True)[0]
            modified_logits[:, expert_ids] = max_logits
            
        elif mode == 'suppress':
            # Suppress: Set specified experts' logits to the minimum logit value
            # This makes them least likely to be selected
            min_logits = modified_logits.min(dim=-1, keepdim=True)[0]
            modified_logits[:, expert_ids] = min_logits
            
        elif mode == 'soft':
            # Soft intervention: z'_k = z_k + lambda * std(z)
            # Calculate standard deviation of logits for each token
            # std(z) shape: (num_tokens, 1)
            logits_std = modified_logits.std(dim=-1, keepdim=True)
            
            # Apply intervention: add (or subtract if lambda < 0) lambda * std(z)
            # This adjusts logits proportionally to their distribution spread
            # Batch operation: use advanced indexing to modify all expert_ids at once
            # Broadcasting: logits_std (num_tokens, 1) -> (num_tokens, len(expert_ids))
            modified_logits[:, expert_ids] += lambda_param * logits_std
            
        elif mode == 'soft_topk':
            # Conditional soft intervention: only intervene if expert is NOT in top-k
            # This boosts experts that are not already selected, encouraging diversity
            
            # Get top-k expert indices for each token
            # topk_indices shape: (num_tokens, topk_param)
            topk_indices = modified_logits.topk(topk_param, dim=-1)[1]
            
            # Create tensor of target expert IDs on same device
            expert_ids_tensor = torch.tensor(expert_ids, device=modified_logits.device, dtype=torch.long)
            
            # Check which experts are in top-k for each token
            # Broadcast comparison: (num_tokens, topk_param, 1) == (1, 1, len(expert_ids))
            # Result: (num_tokens, topk_param, len(expert_ids))
            # .any(dim=1): collapse topk dimension -> (num_tokens, len(expert_ids))
            in_topk_mask = (topk_indices.unsqueeze(-1) == expert_ids_tensor.view(1, 1, -1)).any(dim=1)
            
            # Calculate soft intervention delta
            logits_std = modified_logits.std(dim=-1, keepdim=True)
            delta = lambda_param * logits_std  # (num_tokens, 1)
            
            # Apply intervention only where expert is NOT in top-k
            # Multiply delta by inverted mask (broadcast to match expert_ids dimension)
            # ~in_topk_mask: (num_tokens, len(expert_ids)) - True where expert is NOT in top-k
            # delta: (num_tokens, 1) -> broadcasts to (num_tokens, len(expert_ids))
            # Convert mask to same dtype as delta to avoid type mismatch
            modified_logits[:, expert_ids] += delta * (~in_topk_mask).to(delta.dtype)
            
        elif mode == 'soft_random':
            # Random soft intervention: randomly select same number of experts and apply soft intervention
            # This mode uses expert_ids only to determine the COUNT of experts to randomly select
            
            num_experts_to_select = len(expert_ids)
            n_experts = modified_logits.shape[-1]
            
            # Randomly select experts for each token
            # For each token, we randomly pick num_experts_to_select experts without replacement
            num_tokens = modified_logits.shape[0]
            
            # Generate random expert indices for each token
            # random_expert_ids shape: (num_tokens, num_experts_to_select)
            random_expert_ids = torch.stack([
                torch.randperm(n_experts, device=modified_logits.device)[:num_experts_to_select]
                for _ in range(num_tokens)
            ])
            
            # Calculate soft intervention delta
            logits_std = modified_logits.std(dim=-1, keepdim=True)  # (num_tokens, 1)
            delta = lambda_param * logits_std
            
            # Apply intervention to randomly selected experts
            # Use advanced indexing to modify different experts for each token
            # batch_indices: [0, 1, 2, ..., num_tokens-1] repeated for each expert
            batch_indices = torch.arange(num_tokens, device=modified_logits.device).unsqueeze(1).expand(-1, num_experts_to_select)
            
            # Flatten indices for scatter operation
            batch_flat = batch_indices.flatten()
            expert_flat = random_expert_ids.flatten()
            delta_flat = delta.expand(-1, num_experts_to_select).flatten()
            
            # Apply delta to randomly selected experts
            modified_logits[batch_flat, expert_flat] += delta_flat
            
        elif mode == 'soft_hard':
            # Soft-hard intervention: set expert logits to max + small random perturbation
            # This is similar to 'boost' but adds small random values to avoid ties
            # when multiple experts are specified
            
            # Get maximum logit for each token
            max_logits = modified_logits.max(dim=-1, keepdim=True)[0]  # (num_tokens, 1)
            
            num_experts = len(expert_ids)
            if num_experts > 1:
                # Add small random perturbations to avoid identical logits
                # Random values in range [0, 0.0001] to ensure diversity without changing order significantly
                # Each expert gets a unique random value for each token
                
                # Generate random perturbations for each token and each expert
                # Shape: (num_tokens, num_experts)
                # torch.rand generates values in [0, 1), multiply by 0.0001 to get [0, 0.0001)
                # Match dtype with modified_logits to avoid BFloat16/Float mismatch
                random_perturbations = torch.rand(
                    modified_logits.shape[0], 
                    num_experts, 
                    device=modified_logits.device,
                    dtype=modified_logits.dtype
                ) * 0.0001
                
                # Set expert logits to max + random perturbation
                # Use advanced indexing: modified_logits[:, expert_ids] has shape (num_tokens, num_experts)
                modified_logits[:, expert_ids] = max_logits + random_perturbations
            else:
                # Single expert: just set to max (no need for perturbation)
                modified_logits[:, expert_ids] = max_logits
            
        else:
            logger.warning(f"Unknown intervention mode: {mode}, must be 'boost', 'suppress', 'soft', 'soft_topk', 'soft_random', or 'soft_hard'")
            return router_logits
        
        return modified_logits
    
    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict:
        """
        Load MoE router intervention config from JSON file.
        
        File format:
        {
            "layer_configs": {
                "15": {
                    "expert_ids": [1, 5, 10],
                    "mode": "boost"  # Optional, defaults to "boost"
                },
                "20": {
                    "expert_ids": [0, 2],
                    "mode": "suppress"
                },
                "25": {
                    "expert_ids": [3, 7, 12],
                    "mode": "soft",
                    "lambda": 0.5  # Optional, for 'soft' mode, defaults to 0.5
                },
                "30": {
                    "expert_ids": [1, 3, 5, 7],
                    "mode": "soft_topk",
                    "lambda": 0.8,  # Optional, defaults to request.moe_lambda or 0.5
                    "topk": 8       # Optional, only intervene if expert is NOT in top-8, defaults to 8
                },
                "35": {
                    "expert_ids": [1, 5, 10],  # Only count matters, not specific IDs
                    "mode": "soft_random",
                    "lambda": 0.5  # Optional, defaults to request.moe_lambda or 0.5
                },
                "40": {
                    "expert_ids": [2, 4, 6],
                    "mode": "soft_hard"  # Set to max logits with small random perturbations
                }
            }
        }
        
        Args:
            path: Path to JSON config file
            device: Target device (not used for config loading)
            **kwargs: Additional arguments (target_layers, moe_lambda, moe_topk, etc.)
            
        Returns:
            Dict with 'layer_payloads' key mapping layer_id to intervention params
        """
        import os
        
        # Extract default parameters from kwargs (from SteerVectorRequest)
        default_mode = kwargs.get('moe_mode', None)  # None means use JSON value
        default_lambda = kwargs.get('moe_lambda', 0.5)
        default_topk = kwargs.get('moe_topk', 8)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"MoE config file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            # Extract layer configurations
            layer_configs = config.get("layer_configs", {})
            
            # Convert string keys to int and validate
            layer_payloads = {}
            for layer_str, params in layer_configs.items():
                try:
                    layer_id = int(layer_str)
                    
                    # Validate required fields
                    if 'expert_ids' not in params:
                        logger.warning(f"Layer {layer_id} missing 'expert_ids', skipping")
                        continue
                    
                    # Set mode: SteerVectorRequest > JSON > default 'boost'
                    # Priority: 1. default_mode from request (if specified)
                    #           2. mode from JSON file
                    #           3. fallback to 'boost'
                    if default_mode is not None:
                        mode = default_mode
                        logger.info(f"Layer {layer_id}: Using mode={mode} from SteerVectorRequest")
                    else:
                        mode = params.get('mode', 'boost')
                    
                    intervention_params = {
                        'expert_ids': params.get('expert_ids', []),
                        'mode': mode,
                    }
                    
                    # Add lambda parameter:
                    # 1. If present in JSON, use JSON value
                    # 2. Otherwise, if mode is 'soft', 'soft_topk', or 'soft_random', use default_lambda from request
                    if 'lambda' in params:
                        intervention_params['lambda'] = params['lambda']
                    elif mode in ['soft', 'soft_topk', 'soft_random']:
                        intervention_params['lambda'] = default_lambda
                        logger.info(f"Layer {layer_id}: Using default lambda={default_lambda} for {mode} mode")
                    
                    # Add topk parameter for soft_topk mode:
                    # 1. If present in JSON, use JSON value
                    # 2. Otherwise, if mode is 'soft_topk', use default_topk from request
                    if 'topk' in params:
                        intervention_params['topk'] = params['topk']
                    elif mode == 'soft_topk':
                        intervention_params['topk'] = default_topk
                        logger.info(f"Layer {layer_id}: Using default topk={default_topk} for soft_topk mode")
                    
                    layer_payloads[layer_id] = intervention_params
                    
                except ValueError:
                    logger.warning(f"Invalid layer ID: {layer_str}, skipping")
                    continue
            
            if not layer_payloads:
                raise ValueError(f"No valid layer configurations found in {path}")
            
            return {"layer_payloads": layer_payloads}
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON config: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load MoE config: {e}") from e
    
    def _is_valid(self, params: Any) -> bool:
        """Check if intervention parameters are valid."""
        if params is None:
            return False
        
        if not isinstance(params, dict):
            return False
        
        # Must have expert_ids
        expert_ids = params.get('expert_ids', [])
        if not expert_ids or not isinstance(expert_ids, list):
            return False
        
        return True

