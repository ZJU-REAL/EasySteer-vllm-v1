# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NumPy selector evaluation shared by admission and batch routing.

Matches the tensor selector semantics in selection.runtime.
"""

import numpy as np


def _match_positions_np(
    abs_pos: np.ndarray, positions, neg_base: np.ndarray, is_dec: np.ndarray
) -> np.ndarray:
    """Match prompt positions, resolving negative indices from the prompt end.

    Positive indices beyond the prompt clamp to its last token. Decode
    tokens never match.
    """
    mask = np.zeros(abs_pos.shape[0], dtype=bool)
    for p in positions:
        if p < 0:
            mask |= abs_pos == neg_base + p
        else:
            mask |= abs_pos == np.minimum(neg_base - 1, p)
    return mask & ~is_dec


def _match_prompt_window_np(
    abs_pos: np.ndarray, window, neg_base: np.ndarray, is_dec: np.ndarray
) -> np.ndarray:
    """Match a half-open prompt window.

    Negative bounds resolve from the prompt end; stop=None means that end.
    """
    start, stop = window
    lo = neg_base + start if start < 0 else start
    hi = neg_base if stop is None else (neg_base + stop if stop < 0 else stop)
    return ~is_dec & (abs_pos >= lo) & (abs_pos < hi)


def _match_generation_steps_np(
    gen_idx: np.ndarray, is_dec: np.ndarray, steps, window
) -> np.ndarray:
    """Match zero-based generation steps or a half-open generation window."""
    mask = np.zeros(gen_idx.shape[0], dtype=bool)
    if steps is not None:
        mask |= np.isin(gen_idx, np.asarray(list(steps), dtype=np.int64))
    if window is not None:
        start, stop = window
        in_window = gen_idx >= start
        if stop is not None:
            in_window &= gen_idx < stop
        mask |= in_window
    return mask & is_dec


def clause_mask(
    clause: dict,
    is_dec: np.ndarray,
    abs_pos: np.ndarray,
    neg_base: np.ndarray,
    gen_idx: np.ndarray,
    token_ids,
) -> np.ndarray:
    """Union include selectors, then subtract exclusions over a slot's tokens.

    `token_ids` is a thunk: only token-id filters pay for the host copy.
    """
    n = is_dec.shape[0]

    def _selector_mask(
        prompt_tokens,
        prompt_positions,
        prompt_window,
        generation_tokens,
        generation_positions,
        generation_window,
    ) -> np.ndarray:
        matched = np.zeros(n, dtype=bool)
        if prompt_tokens is not None:
            matched |= np.isin(token_ids(), np.asarray(list(prompt_tokens))) & ~is_dec
        if generation_tokens is not None:
            matched |= (
                np.isin(token_ids(), np.asarray(list(generation_tokens))) & is_dec
            )
        if prompt_positions is not None:
            matched |= _match_positions_np(abs_pos, prompt_positions, neg_base, is_dec)
        if prompt_window is not None:
            matched |= _match_prompt_window_np(abs_pos, prompt_window, neg_base, is_dec)
        if generation_positions is not None or generation_window is not None:
            matched |= _match_generation_steps_np(
                gen_idx, is_dec, generation_positions, generation_window
            )
        return matched

    mask = np.zeros(n, dtype=bool)
    if clause.get("prompt") == "all":
        mask |= ~is_dec
    if clause.get("generation") == "all":
        mask |= is_dec

    from vllm.model_hooks.selection.schema import (
        EXCLUDE_SELECTOR_KEYS,
        INCLUDE_SELECTOR_KEYS,
    )

    includes = tuple(clause.get(key) for key in INCLUDE_SELECTOR_KEYS)
    if any(value is not None for value in includes):
        mask |= _selector_mask(*includes)

    excludes = tuple(clause.get(key) for key in EXCLUDE_SELECTOR_KEYS)
    if any(value is not None for value in excludes):
        mask &= ~_selector_mask(*excludes)
    return mask
