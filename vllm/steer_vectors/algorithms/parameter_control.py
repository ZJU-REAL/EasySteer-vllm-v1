# SPDX-License-Identifier: Apache-2.0
"""
Parameter-driven intervention control for steer vector algorithms.

This module provides:
1. InterventionController class - Manages all intervention parameters
2. GPU-optimized functions - Determine WHERE to apply interventions based on parameters
"""

from typing import Optional, Dict
import torch


class InterventionController:
    """
    Centralized controller for intervention parameters.
    
    Manages all parameters that control WHERE interventions are applied,
    including trigger tokens, positions, exclusion rules, and debug settings.
    
    This class decouples parameter management from algorithm implementation,
    allowing algorithm developers to focus only on transformation logic.
    """
    
    def __init__(self):
        """Initialize with no triggers configured."""
        # Trigger parameters
        self.prefill_trigger_tokens: Optional[set[int]] = None
        self.prefill_trigger_positions: Optional[list[int]] = None
        self.prefill_exclude_tokens: Optional[set[int]] = None
        self.prefill_exclude_positions: Optional[list[int]] = None
        self.generate_trigger_tokens: Optional[set[int]] = None
        
        # Debug mode
        self.debug: bool = False
    
    # ========== Parameter Setters ==========
    
    def set_debug(self, debug: bool) -> None:
        """Set debug mode."""
        self.debug = debug
    
    def set_prefill_trigger_tokens(self, token_ids: Optional[list[int]]) -> None:
        """Set trigger tokens for prefill phase."""
        self.prefill_trigger_tokens = set(token_ids) if token_ids is not None else None
    
    def set_prefill_trigger_positions(self, positions: Optional[list[int]]) -> None:
        """Set trigger positions for prefill phase."""
        self.prefill_trigger_positions = positions
    
    def set_prefill_exclude_tokens(self, token_ids: Optional[list[int]]) -> None:
        """Set tokens to exclude during prefill phase."""
        self.prefill_exclude_tokens = set(token_ids) if token_ids is not None else None
    
    def set_prefill_exclude_positions(self, positions: Optional[list[int]]) -> None:
        """Set positions to exclude during prefill phase."""
        self.prefill_exclude_positions = positions
    
    def set_generate_trigger_tokens(self, token_ids: Optional[list[int]]) -> None:
        """Set trigger tokens for generation phase."""
        self.generate_trigger_tokens = set(token_ids) if token_ids is not None else None
    
    def configure_from_dict(self, config: dict) -> None:
        """
        Batch configure intervention parameters from a dictionary.
        
        This method provides a unified interface for setting all intervention parameters,
        eliminating the need for wrapper layers to know individual parameter names.
        Only processes intervention-related parameters (triggers, exclusions, debug),
        ignoring algorithm-specific parameters.
        
        Args:
            config: Dictionary containing intervention parameters
        """
        if "prefill_trigger_tokens" in config:
            self.set_prefill_trigger_tokens(config["prefill_trigger_tokens"])
        if "prefill_trigger_positions" in config:
            self.set_prefill_trigger_positions(config["prefill_trigger_positions"])
        if "prefill_exclude_tokens" in config:
            self.set_prefill_exclude_tokens(config["prefill_exclude_tokens"])
        if "prefill_exclude_positions" in config:
            self.set_prefill_exclude_positions(config["prefill_exclude_positions"])
        if "generate_trigger_tokens" in config:
            self.set_generate_trigger_tokens(config["generate_trigger_tokens"])
        if "debug" in config:
            self.set_debug(config["debug"])
    
    # ========== Parameter Queries ==========
    
    def should_apply_to_all_prefill_tokens(self) -> bool:
        """Check if steer vector should be applied to all prefill tokens."""
        return self.prefill_trigger_tokens is not None and -1 in self.prefill_trigger_tokens
    
    def should_apply_to_all_generate_tokens(self) -> bool:
        """Check if steer vector should be applied to all generation tokens."""
        return self.generate_trigger_tokens is not None and -1 in self.generate_trigger_tokens
    
    def has_prefill_triggers(self) -> bool:
        """Check if prefill triggers are configured."""
        return (self.prefill_trigger_tokens is not None or
                self.prefill_trigger_positions is not None)
    
    def has_any_triggers(self) -> bool:
        """Check if any triggers are configured."""
        return (self.prefill_trigger_tokens is not None or 
                self.generate_trigger_tokens is not None or
                self.prefill_trigger_positions is not None)
    
    def is_global_only_config(self) -> bool:
        """
        Check if this is a global-only configuration.
        
        A global-only configuration means interventions are applied to ALL tokens
        without any position-based or exclusion filters, which enables the fast path
        that avoids index_select/index_copy overhead.
        
        Returns:
            True if configured for global application only, False otherwise
        """
        # No position-based or exclusion triggers
        has_no_filters = (
            self.prefill_trigger_positions is None and
            self.prefill_exclude_tokens is None and
            self.prefill_exclude_positions is None
        )
        
        if not has_no_filters:
            return False
        
        # Check if trigger configuration is global (-1 means apply to all)
        prefill_is_global = (self.prefill_trigger_tokens == {-1})
        generate_is_global = (self.generate_trigger_tokens == {-1})
        prefill_is_none = (self.prefill_trigger_tokens is None)
        generate_is_none = (self.generate_trigger_tokens is None)
        
        # One of the following must be true:
        # 1. Only prefill global trigger
        # 2. Only generate global trigger  
        # 3. Both are global triggers
        return (
            (prefill_is_global and generate_is_none) or
            (generate_is_global and prefill_is_none) or
            (prefill_is_global and generate_is_global)
        )
    
    # ========== Core Functionality ==========
    
    def collect_intervention_positions(
        self,
        hidden_states: torch.Tensor,
        current_tokens: torch.Tensor,
        samples_info: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Collect all intervention positions based on configured parameters.
        
        This is the main entry point that uses all configured parameters
        to determine which token positions should receive interventions.
        
        Args:
            hidden_states: [total_tokens, hidden_dim]
            current_tokens: [total_tokens] token IDs
            samples_info: Dict with 'query_start_loc', 'num_computed', 'is_decode_mask'
            
        Returns:
            positions_tensor: [num_positions] GPU tensor of positions to transform
            or None if no positions to apply
        """
        return collect_positions_gpu_batch(
            hidden_states=hidden_states,
            current_tokens=current_tokens,
            samples_info=samples_info,
            prefill_trigger_tokens=self.prefill_trigger_tokens,
            prefill_trigger_positions=self.prefill_trigger_positions,
            prefill_exclude_tokens=self.prefill_exclude_tokens,
            prefill_exclude_positions=self.prefill_exclude_positions,
            generate_trigger_tokens=self.generate_trigger_tokens,
            has_prefill_triggers=self.has_prefill_triggers()
        )


# ========== GPU-Optimized Helper Functions ==========


def get_decode_mask_gpu(
    current_tokens: torch.Tensor,
    sample_ids: torch.Tensor,
    is_decode_mask: torch.Tensor,
    generate_trigger_tokens: Optional[set],
    device: torch.device
) -> torch.Tensor:
    """
    Generate mask for decode sample positions using GPU batch operations.
    
    Args:
        current_tokens: [total_tokens] token IDs
        sample_ids: [total_tokens] sample ID for each token
        is_decode_mask: [num_samples] boolean tensor (True for decode samples)
        generate_trigger_tokens: Set of trigger token IDs for generation phase, or None
        device: Device to create tensors on
        
    Returns:
        mask: [total_tokens] boolean tensor indicating positions to apply intervention
    """
    # Get decode sample IDs
    decode_sample_ids = torch.nonzero(is_decode_mask, as_tuple=False).squeeze(-1)
    
    if decode_sample_ids.numel() == 0:
        return torch.zeros(len(sample_ids), dtype=torch.bool, device=device)
    
    # Apply to all decode tokens if configured (-1 means apply to all)
    if generate_trigger_tokens is not None and -1 in generate_trigger_tokens:
        mask = torch.isin(sample_ids, decode_sample_ids)
        return mask
    
    # Apply to specific decode tokens based on token IDs
    trigger_tensor = torch.tensor(
        list(generate_trigger_tokens),
        dtype=current_tokens.dtype,
        device=device
    )
    token_match = torch.isin(current_tokens, trigger_tensor)
    sample_match = torch.isin(sample_ids, decode_sample_ids)
    
    return token_match & sample_match


def apply_exclusions_gpu(
    mask: torch.Tensor,
    current_tokens: torch.Tensor,
    sample_ids: torch.Tensor,
    relative_positions: torch.Tensor,
    num_computed: Optional[torch.Tensor],
    prefill_sample_ids: torch.Tensor,
    prefill_exclude_positions: Optional[list],
    prefill_exclude_tokens: Optional[set],
    device: torch.device
) -> torch.Tensor:
    """
    Apply exclusion rules using GPU batch operations.
    
    Args:
        mask: [total_tokens] boolean tensor to apply exclusions to
        current_tokens: [total_tokens] token IDs
        sample_ids: [total_tokens] sample ID for each token
        relative_positions: [total_tokens] relative position within each sample
        num_computed: [num_samples] cached token counts or None
        prefill_sample_ids: [num_prefill_samples] IDs of prefill samples
        prefill_exclude_positions: List of positions to exclude or None
        prefill_exclude_tokens: Set of token IDs to exclude or None
        device: Device to create tensors on
        
    Returns:
        mask: Updated boolean tensor with exclusions applied
    """
    # ========== 1. Exclude by position ==========
    if prefill_exclude_positions is not None:
        # Compute original positions (accounting for cache)
        if num_computed is not None:
            num_computed_expanded = num_computed[sample_ids]
            original_positions = relative_positions + num_computed_expanded
        else:
            original_positions = relative_positions
        
        exclude_positions_tensor = torch.tensor(
            list(prefill_exclude_positions),
            dtype=original_positions.dtype,
            device=device
        )
        exclude_position_match = torch.isin(original_positions, exclude_positions_tensor)
        
        # Only exclude from prefill samples
        exclude_position_match &= torch.isin(sample_ids, prefill_sample_ids)
        
        # Remove excluded positions
        mask &= ~exclude_position_match
    
    # ========== 2. Exclude by token ==========
    if prefill_exclude_tokens is not None:
        exclude_tokens_tensor = torch.tensor(
            list(prefill_exclude_tokens),
            dtype=current_tokens.dtype,
            device=device
        )
        exclude_token_match = torch.isin(current_tokens, exclude_tokens_tensor)
        
        # Remove excluded tokens
        mask &= ~exclude_token_match
    
    return mask


def get_prefill_mask_gpu(
    current_tokens: torch.Tensor,
    sample_ids: torch.Tensor,
    relative_positions: torch.Tensor,
    is_decode_mask: torch.Tensor,
    num_computed: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    prefill_trigger_tokens: Optional[set],
    prefill_trigger_positions: Optional[list],
    prefill_exclude_tokens: Optional[set],
    prefill_exclude_positions: Optional[list],
    device: torch.device
) -> torch.Tensor:
    """
    Generate mask for prefill sample positions using GPU batch operations.
    
    Handles:
    - Position-based triggers (prefill_trigger_positions)
    - Token-based triggers (prefill_trigger_tokens)
    - Exclusions (prefill_exclude_positions, prefill_exclude_tokens)
    - Prefix caching (num_computed)
    
    Args:
        current_tokens: [total_tokens] token IDs
        sample_ids: [total_tokens] sample ID for each token
        relative_positions: [total_tokens] relative position within each sample
        is_decode_mask: [num_samples] boolean tensor (True for decode samples)
        num_computed: [num_samples] cached token counts or None
        query_start_loc: [num_samples+1] sample boundaries
        prefill_trigger_tokens: Set of trigger token IDs for prefill phase or None
        prefill_trigger_positions: List of trigger positions for prefill phase or None
        prefill_exclude_tokens: Set of token IDs to exclude or None
        prefill_exclude_positions: List of positions to exclude or None
        device: Device to create tensors on
        
    Returns:
        mask: [total_tokens] boolean tensor indicating positions to apply intervention
    """
    total_tokens = len(sample_ids)
    
    # Get prefill sample IDs
    prefill_sample_ids = torch.nonzero(~is_decode_mask, as_tuple=False).squeeze(-1)
    
    if prefill_sample_ids.numel() == 0:
        return torch.zeros(total_tokens, dtype=torch.bool, device=device)
    
    # Check if applying to all prefill tokens (-1 means apply to all)
    if prefill_trigger_tokens is not None and -1 in prefill_trigger_tokens:
        mask = torch.isin(sample_ids, prefill_sample_ids)
        return mask
    
    # Initialize mask for prefill samples
    mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    
    # ========== 1. Position-based triggers ==========
    if prefill_trigger_positions is not None:
        # Compute original positions accounting for prefix caching
        if num_computed is not None:
            num_computed_expanded = num_computed[sample_ids]
            original_positions = relative_positions + num_computed_expanded
        else:
            original_positions = relative_positions
        
        # Match trigger positions
        trigger_positions_tensor = torch.tensor(
            list(prefill_trigger_positions),
            dtype=original_positions.dtype,
            device=device
        )
        position_match = torch.isin(original_positions, trigger_positions_tensor)
        
        # Only apply to prefill samples
        position_match &= torch.isin(sample_ids, prefill_sample_ids)
        
        # Filter out cached positions (only process current forward)
        if num_computed is not None:
            num_computed_expanded = num_computed[sample_ids]
            uncached_mask = original_positions >= num_computed_expanded
            position_match &= uncached_mask
        
        mask |= position_match
    
    # ========== 2. Token-based triggers ==========
    if prefill_trigger_tokens is not None:
        trigger_tokens_tensor = torch.tensor(
            list(prefill_trigger_tokens),
            dtype=current_tokens.dtype,
            device=device
        )
        token_match = torch.isin(current_tokens, trigger_tokens_tensor)
        
        # Only apply to prefill samples
        token_match &= torch.isin(sample_ids, prefill_sample_ids)
        
        mask |= token_match
    
    # ========== 3. Apply exclusions ==========
    if mask.any():  # Only if we have positions to exclude from
        mask = apply_exclusions_gpu(
            mask=mask,
            current_tokens=current_tokens,
            sample_ids=sample_ids,
            relative_positions=relative_positions,
            num_computed=num_computed,
            prefill_sample_ids=prefill_sample_ids,
            prefill_exclude_positions=prefill_exclude_positions,
            prefill_exclude_tokens=prefill_exclude_tokens,
            device=device
        )
    
    return mask


def collect_positions_gpu_batch(
    hidden_states: torch.Tensor,
    current_tokens: torch.Tensor,
    samples_info: Dict[str, torch.Tensor],
    prefill_trigger_tokens: Optional[set],
    prefill_trigger_positions: Optional[list],
    prefill_exclude_tokens: Optional[set],
    prefill_exclude_positions: Optional[list],
    generate_trigger_tokens: Optional[set],
    has_prefill_triggers: bool
) -> Optional[torch.Tensor]:
    """
    Collect all intervention positions using GPU batch operations.
    
    Performs vectorized position collection with minimal GPU-CPU synchronization.
    
    Args:
        hidden_states: [total_tokens, hidden_dim]
        current_tokens: [total_tokens] token IDs
        samples_info: Dict with 'query_start_loc', 'num_computed', 'is_decode_mask'
        prefill_trigger_tokens: Set of trigger token IDs for prefill phase or None
        prefill_trigger_positions: List of trigger positions for prefill phase or None
        prefill_exclude_tokens: Set of token IDs to exclude or None
        prefill_exclude_positions: List of positions to exclude or None
        generate_trigger_tokens: Set of trigger token IDs for generation phase or None
        has_prefill_triggers: Whether prefill triggers are configured
        
    Returns:
        positions_tensor: [num_positions] GPU tensor of positions to transform
        or None if no positions to apply
    """
    query_start_loc = samples_info['query_start_loc']
    num_computed = samples_info['num_computed']
    is_decode_mask = samples_info['is_decode_mask']
    
    device = hidden_states.device
    total_tokens = hidden_states.shape[0]
    num_samples = len(is_decode_mask)
    
    # Normalize num_computed to torch.Tensor for consistent indexing
    if num_computed is not None and not isinstance(num_computed, torch.Tensor):
        num_computed = torch.tensor(num_computed, device=device, dtype=torch.long)
    
    # Step 1: Assign each token to its sample using binary search
    all_positions = torch.arange(total_tokens, device=device)
    # Use right=True to correctly assign boundary tokens
    # For position at query_start_loc[i], it should belong to sample i (not i-1)
    sample_ids = torch.searchsorted(query_start_loc, all_positions, right=True) - 1
    
    # Step 2: Compute relative positions within each sample
    starts = query_start_loc[:-1]
    starts_expanded = starts[sample_ids]
    relative_positions = all_positions - starts_expanded
    
    # Step 3: Initialize position mask for collecting trigger positions
    position_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    
    # Step 4: Handle decode samples
    if generate_trigger_tokens is not None and torch.any(is_decode_mask):
        decode_mask = get_decode_mask_gpu(
            current_tokens=current_tokens,
            sample_ids=sample_ids,
            is_decode_mask=is_decode_mask,
            generate_trigger_tokens=generate_trigger_tokens,
            device=device
        )
        position_mask |= decode_mask
    
    # Step 5: Handle prefill samples
    if has_prefill_triggers and torch.any(~is_decode_mask):
        prefill_mask = get_prefill_mask_gpu(
            current_tokens=current_tokens,
            sample_ids=sample_ids,
            relative_positions=relative_positions,
            is_decode_mask=is_decode_mask,
            num_computed=num_computed,
            query_start_loc=query_start_loc,
            prefill_trigger_tokens=prefill_trigger_tokens,
            prefill_trigger_positions=prefill_trigger_positions,
            prefill_exclude_tokens=prefill_exclude_tokens,
            prefill_exclude_positions=prefill_exclude_positions,
            device=device
        )
        position_mask |= prefill_mask
    
    # Step 6: Extract final positions from mask
    positions_tensor = torch.nonzero(position_mask, as_tuple=False).squeeze(-1)
    
    if positions_tensor.numel() == 0:
        return None
    
    return positions_tensor

