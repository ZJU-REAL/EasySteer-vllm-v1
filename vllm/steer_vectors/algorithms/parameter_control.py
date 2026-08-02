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
        self.generate_first_k_tokens: Optional[int] = None
        self.generate_after_k_tokens: Optional[int] = None
        
        # Debug mode
        self.debug: bool = False
    
    # ========== Parameter Configuration ==========

    def set_debug(self, debug: bool) -> None:
        """Set debug mode."""
        self.debug = debug

    def configure_from_dict(self, config: dict) -> None:
        """
        Batch configure intervention parameters from a dictionary.

        Driven by the canonical field registry: trigger fields present in
        the dict are applied (token-id lists become sets), everything
        else (payloads, algorithm parameters) is ignored.
        """
        from vllm.steer_vectors.request import (
            STEER_TOKEN_SET_FIELDS,
            STEER_TRIGGER_FIELDS,
        )

        for name in STEER_TRIGGER_FIELDS + ("debug",):
            if name not in config:
                continue
            value = config[name]
            if name in STEER_TOKEN_SET_FIELDS and value is not None:
                value = set(value)
            setattr(self, name, value)

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
                self.prefill_trigger_positions is not None or
                self.generate_first_k_tokens is not None or
                self.generate_after_k_tokens is not None)
    
    def is_global_only_config(self) -> bool:
        """
        Check if this is a global-only configuration.
        
        A global-only configuration means interventions are applied to ALL tokens
        in BOTH prefill and generate phases, without any position-based or exclusion
        filters. This enables the fast path that avoids index_select/index_copy overhead.
        
        Design rationale:
        - Fast path requires BOTH phases to be configured as global
        - Single-phase global configs must use the normal path because:
          * Fast path operates on the entire hidden_states tensor
          * Phase information (prefill vs generate) is only available via forward context
          * Mixed batches are common in continuous batching scenarios
        - The -1 token ID is a special marker indicating "apply to all tokens in this phase"
        
        Requirements:
        - prefill_trigger_tokens must contain -1
        - generate_trigger_tokens must contain -1
        - No exclusions (prefill_exclude_tokens = None, prefill_exclude_positions = None)
        
        Note: 
        - Additional token IDs can coexist with -1 (e.g., {1234, -1}), as -1 takes 
          precedence and matches all tokens in the normal path.
        - prefill_trigger_positions is NOT checked because when -1 is present in 
          trigger_tokens, the normal path returns immediately without processing positions.
        
        Returns:
            True if BOTH phases are configured for global application, False otherwise
        """
        # Only exclusion filters matter (-1 in trigger_tokens overrides position triggers)
        has_no_exclusions = (
            self.prefill_exclude_tokens is None and
            self.prefill_exclude_positions is None
        )
        
        if not has_no_exclusions:
            return False
        
        # Check if BOTH trigger configurations contain -1 (global marker)
        prefill_is_global = (
            self.prefill_trigger_tokens is not None and 
            -1 in self.prefill_trigger_tokens
        )
        generate_is_global = (
            self.generate_trigger_tokens is not None and 
            -1 in self.generate_trigger_tokens
        )
        
        # Both phases must be global for fast path
        return prefill_is_global and generate_is_global
    
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
            generate_first_k_tokens=self.generate_first_k_tokens,
            generate_after_k_tokens=self.generate_after_k_tokens,
            has_prefill_triggers=self.has_prefill_triggers()
        )


# ========== Position Collection: Mask Algebra ==========
#
# The final position set is a composition of small per-token boolean
# masks over the flat batch:
#
#   positions = (decode_part | prefill_part) & generate_window
#
#   decode_part   = decode_tokens & token_trigger(generate)
#   prefill_part  = prefill_tokens & (position_trigger | token_trigger)
#                   & ~(position_exclude | token_exclude)
#   generate_window = ~decode_tokens | first_k/after_k condition
#
# Preserved legacy semantics (verified by the equivalence fuzz test
# against scripts/_legacy_position_collector.py):
# - a -1 token id means "all tokens of that phase"; for prefill it also
#   bypasses the exclusion rules entirely
# - token exclusions are not phase-restricted (they only ever see
#   prefill positions in practice because they apply to prefill_part)
# - position triggers/exclusions use absolute positions within the
#   request (prefix-cache offset included); negative indices are
#   Python-style from the end of the total sequence
# - first_k/after_k compare against the number of tokens already
#   generated and only constrain decode tokens


def _isin_token_set(tokens: torch.Tensor, ids) -> torch.Tensor:
    """[total_tokens] mask of tokens whose id is in `ids`."""
    ids_tensor = torch.tensor(
        list(ids), dtype=tokens.dtype, device=tokens.device
    )
    return torch.isin(tokens, ids_tensor)


def _match_positions(
    abs_positions: torch.Tensor,
    positions: list,
    total_len_per_sample: torch.Tensor,
    sample_ids: torch.Tensor,
) -> torch.Tensor:
    """[total_tokens] mask of tokens at the given absolute positions.

    Positive entries match absolute positions directly; negative entries
    are Python-style indices from each sample's total length (prompt +
    cached + generated so far).
    """
    mask = torch.zeros_like(abs_positions, dtype=torch.bool)
    positive = [p for p in positions if p >= 0]
    negative = [p for p in positions if p < 0]
    if positive:
        mask |= torch.isin(
            abs_positions,
            torch.tensor(
                positive, dtype=abs_positions.dtype,
                device=abs_positions.device,
            ),
        )
    if negative:
        totals = total_len_per_sample[sample_ids]
        for neg_idx in negative:
            mask |= abs_positions == totals + neg_idx
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
    generate_first_k_tokens: Optional[int],
    generate_after_k_tokens: Optional[int],
    has_prefill_triggers: bool
) -> Optional[torch.Tensor]:
    """
    Collect intervention positions for the whole batch on the GPU.

    Returns a [num_positions] tensor of flat token indices, or None when
    nothing matches.
    """
    query_start_loc = samples_info['query_start_loc']
    num_computed = samples_info['num_computed']
    is_decode_mask = samples_info['is_decode_mask']

    device = hidden_states.device
    # Size the masks from current_tokens, not hidden_states: under
    # piecewise cudagraphs hidden_states is padded to the graph bucket,
    # while current_tokens/query_start_loc always cover the real tokens.
    # Padding rows are never steered.
    total_tokens = current_tokens.shape[0]

    if num_computed is not None and not isinstance(num_computed, torch.Tensor):
        num_computed = torch.tensor(num_computed, device=device, dtype=torch.long)

    # --- batch geometry ---
    all_positions = torch.arange(total_tokens, device=device)
    sample_ids = torch.searchsorted(query_start_loc, all_positions, right=True) - 1
    relative_positions = all_positions - query_start_loc[:-1][sample_ids]
    if num_computed is not None:
        abs_positions = relative_positions + num_computed[sample_ids]
        total_len = (query_start_loc[1:] - query_start_loc[:-1]) + num_computed
    else:
        abs_positions = relative_positions
        total_len = query_start_loc[1:] - query_start_loc[:-1]
    is_decode_token = is_decode_mask[sample_ids]
    is_prefill_token = ~is_decode_token

    mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)

    # --- decode part ---
    # first_k/after_k without explicit trigger tokens means "all decode
    # tokens" (the window is applied below).
    effective_gtt = generate_trigger_tokens
    if effective_gtt is None and (
        generate_first_k_tokens is not None
        or generate_after_k_tokens is not None
    ):
        effective_gtt = {-1}
    if effective_gtt is not None:
        if -1 in effective_gtt:
            mask |= is_decode_token
        else:
            mask |= is_decode_token & _isin_token_set(
                current_tokens, effective_gtt
            )

    # --- prefill part ---
    if has_prefill_triggers:
        if prefill_trigger_tokens is not None and -1 in prefill_trigger_tokens:
            # "All prefill tokens" bypasses the exclusion rules.
            mask |= is_prefill_token
        else:
            prefill_part = torch.zeros(
                total_tokens, dtype=torch.bool, device=device
            )
            if prefill_trigger_positions is not None:
                prefill_part |= is_prefill_token & _match_positions(
                    abs_positions, prefill_trigger_positions,
                    total_len, sample_ids,
                )
            if prefill_trigger_tokens is not None:
                prefill_part |= is_prefill_token & _isin_token_set(
                    current_tokens, prefill_trigger_tokens
                )
            if prefill_part.any():
                if prefill_exclude_positions is not None:
                    prefill_part &= ~(is_prefill_token & _match_positions(
                        abs_positions, prefill_exclude_positions,
                        total_len, sample_ids,
                    ))
                if prefill_exclude_tokens is not None:
                    prefill_part &= ~_isin_token_set(
                        current_tokens, prefill_exclude_tokens
                    )
            mask |= prefill_part

    # --- generate window (constrains decode tokens only) ---
    if (
        generate_first_k_tokens is not None
        or generate_after_k_tokens is not None
    ):
        num_output_tokens = samples_info.get('num_output_tokens')
        if num_output_tokens is not None:
            gen_counts = num_output_tokens.to(device)[sample_ids]
            if generate_first_k_tokens is not None:
                in_window = gen_counts < generate_first_k_tokens
            else:
                in_window = gen_counts >= generate_after_k_tokens
            mask &= is_prefill_token | in_window

    positions_tensor = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if positions_tensor.numel() == 0:
        return None
    return positions_tensor
