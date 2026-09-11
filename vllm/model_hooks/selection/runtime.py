# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared selection keys and tensor row matching for model hooks."""

import torch

from vllm.model_hooks.selection.batch import BatchView
from vllm.model_hooks.selection.schema import (
    APPLY_SPEC_KEYS,
    EXCLUDE_SELECTOR_KEYS,
    INCLUDE_SELECTOR_KEYS,
)


def resolve_batch_positions(
    batch: BatchView,
    total_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the step's flat token indices to per-sample coordinates.

    Returns ``(sample_ids, abs_positions)`` for the first
    `total_tokens` rows of the batch: each token's sample index, its
    absolute position in that sample (chunk-relative position plus the
    sample's cached-token count).
    """
    query_start_loc = batch.query_start_loc
    all_positions = torch.arange(total_tokens, device=device)
    sample_ids = torch.searchsorted(query_start_loc, all_positions, right=True) - 1
    relative_positions = all_positions - query_start_loc[:-1][sample_ids]
    abs_positions = relative_positions + batch.num_computed[sample_ids]
    return sample_ids, abs_positions


def _canon(value):
    if isinstance(value, (list, tuple)):
        return tuple(_canon(v) for v in value)
    return value


def clause_cache_key(apply_spec: dict | None) -> tuple | None:
    """Hashable identity of a where-clause.

    Position resolution is layer-invariant (clauses match tokens,
    positions and phases — never hidden states), so one resolution per
    step serves every layer. The worker-side resolver and the layer
    hooks must derive keys from this one function or lookups drift.
    """
    if apply_spec is None:
        return None
    return tuple((key, _canon(apply_spec.get(key))) for key in APPLY_SPEC_KEYS)


def selects_all_tokens(apply_spec: dict) -> bool:
    """Whether the clause selects every token of both phases.

    Include selectors are unions with the phase-wide ``all`` selectors, so
    they are redundant when both phases are already selected.  Only an
    exclusion can make such a clause partial.  This enables the fast path
    that skips position collection and steers all of the slot's token rows
    directly, while leaving the original clause in the config fingerprint.
    """
    return (
        apply_spec.get("prompt") == "all"
        and apply_spec.get("generation") == "all"
        and all(apply_spec.get(key) is None for key in EXCLUDE_SELECTOR_KEYS)
    )


def _isin_token_set(tokens: torch.Tensor, ids) -> torch.Tensor:
    """[total_tokens] mask of tokens whose id is in `ids`."""
    ids_tensor = torch.tensor(list(ids), dtype=tokens.dtype, device=tokens.device)
    return torch.isin(tokens, ids_tensor)


def _match_positions(
    abs_positions: torch.Tensor,
    positions: list,
    total_len_per_sample: torch.Tensor,
    sample_ids: torch.Tensor,
    is_decode_token: torch.Tensor,
) -> torch.Tensor:
    """[total_tokens] mask of prompt tokens at the given positions.

    Negative entries are Python-style indices from each sample's prompt
    length in `total_len_per_sample` (stable across prefill chunks).
    Positive entries past the prompt end clamp to the last prompt token
    (the admission-time check warns about it); decode tokens never
    match.
    """
    mask = torch.zeros_like(abs_positions, dtype=torch.bool)
    totals = total_len_per_sample[sample_ids]
    for p in positions:
        if p < 0:
            mask |= abs_positions == totals + p
        else:
            mask |= abs_positions == torch.clamp(totals - 1, max=p)
    return mask & ~is_decode_token


def _match_prompt_window(
    abs_positions: torch.Tensor,
    window: tuple,
    neg_base: torch.Tensor,
    sample_ids: torch.Tensor,
    is_decode_token: torch.Tensor,
) -> torch.Tensor:
    """[total_tokens] mask of prompt tokens inside the half-open window.

    Negative bounds resolve from each sample's prompt length; stop=None
    means the prompt end.
    """
    totals = neg_base[sample_ids]
    start, stop = window
    lo = totals + start if start < 0 else start
    hi = totals if stop is None else (totals + stop if stop < 0 else stop)
    return (~is_decode_token) & (abs_positions >= lo) & (abs_positions < hi)


def _match_generation_steps(
    gen_idx: torch.Tensor,
    is_decode_token: torch.Tensor,
    steps: list | None,
    window: tuple | None,
) -> torch.Tensor:
    """[total_tokens] mask of generation tokens at the given 0-based
    decode steps and/or inside the half-open decode-step window."""
    mask = torch.zeros_like(is_decode_token)
    if steps is not None:
        mask |= torch.isin(
            gen_idx,
            torch.tensor(list(steps), dtype=gen_idx.dtype, device=gen_idx.device),
        )
    if window is not None:
        start, stop = window
        in_window = gen_idx >= start
        if stop is not None:
            in_window &= gen_idx < stop
        mask |= in_window
    return mask & is_decode_token


def collect_positions_apply_spec(
    current_tokens: torch.Tensor,
    batch: BatchView,
    spec: dict,
    *,
    coordinates: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Collect intervention positions for an `apply_spec` where-clause."""
    device = current_tokens.device
    # Size the masks from current_tokens: under piecewise cudagraphs the
    # hidden states are padded to the graph bucket, while current_tokens
    # and query_start_loc always cover the real tokens.
    total_tokens = current_tokens.shape[0]

    if coordinates is None:
        coordinates = resolve_batch_positions(batch, total_tokens, device)
    sample_ids, abs_positions = coordinates
    is_decode_token = abs_positions >= batch.num_prompt[sample_ids]
    generation_indices = torch.where(
        batch.is_decode[sample_ids],
        batch.num_output[sample_ids] - 1,
        abs_positions - batch.num_prompt[sample_ids],
    )

    def _selector_mask(
        prompt_tokens,
        prompt_positions,
        prompt_window,
        generation_tokens,
        generation_positions,
        generation_window,
    ) -> torch.Tensor:
        matched = torch.zeros(total_tokens, dtype=torch.bool, device=device)
        if prompt_tokens is not None:
            matched |= _isin_token_set(current_tokens, prompt_tokens) & ~is_decode_token
        if generation_tokens is not None:
            matched |= (
                _isin_token_set(current_tokens, generation_tokens) & is_decode_token
            )
        if prompt_positions is not None:
            matched |= _match_positions(
                abs_positions,
                prompt_positions,
                batch.num_prompt,
                sample_ids,
                is_decode_token,
            )
        if prompt_window is not None:
            matched |= _match_prompt_window(
                abs_positions,
                prompt_window,
                batch.num_prompt,
                sample_ids,
                is_decode_token,
            )
        if generation_positions is not None or generation_window is not None:
            # Decode step j processes the generated token counted by output j + 1.
            matched |= _match_generation_steps(
                generation_indices,
                is_decode_token,
                generation_positions,
                generation_window,
            )
        return matched

    mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    if spec.get("prompt") == "all":
        mask |= ~is_decode_token
    if spec.get("generation") == "all":
        mask |= is_decode_token

    includes = tuple(spec.get(key) for key in INCLUDE_SELECTOR_KEYS)
    if any(value is not None for value in includes):
        mask |= _selector_mask(*includes)

    excludes = tuple(spec.get(key) for key in EXCLUDE_SELECTOR_KEYS)
    if any(value is not None for value in excludes):
        mask &= ~_selector_mask(*excludes)

    positions_tensor = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if positions_tensor.numel() == 0:
        return None
    return positions_tensor
