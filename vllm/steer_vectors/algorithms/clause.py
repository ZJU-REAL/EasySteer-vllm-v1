# SPDX-License-Identifier: Apache-2.0
"""Where-clause state and position matching for steering algorithms.

`ApplyClause` holds one intervention's `apply_spec` (the wire form
of the user-facing ApplySpec: phases, token/position filters, exclusions
and the generation window); `collect_positions_apply_spec` turns it into
flat token positions for the current step. Algorithms stay pure
transformations over the positions computed here.

Semantics (see steer_vectors/api.py):
  candidates = tokens of the selected phases
  if any include selector given (prompt_tokens, prompt_positions,
    prompt_window, generation_tokens, generation_positions,
    generation_window):
      candidates &= union of the include selectors' matches
  candidates &= ~(union of the exclude selectors' matches)

Include and exclude selectors are symmetric twins evaluated by the
same matchers; windows are half-open (start, stop). Every selector is
phase-scoped: the prompt_* family matches only prompt tokens (negative
prompt_positions resolve from the prompt end; positive ones past the
prompt end clamp to the last prompt token), the generation_* family
matches only decode steps, counted 0-based. There are no sentinel
values and nothing bypasses exclusions — where include and exclude
overlap, the exclusion wins.
"""

import torch

from vllm.steer_vectors.geometry import resolve_batch_positions

_CLAUSE_KEYS = (
    "phases",
    "prompt_tokens",
    "prompt_positions",
    "prompt_window",
    "generation_tokens",
    "generation_positions",
    "generation_window",
    "exclude_prompt_tokens",
    "exclude_prompt_positions",
    "exclude_prompt_window",
    "exclude_generation_tokens",
    "exclude_generation_positions",
    "exclude_generation_window",
)

# The symmetric selector twins, in matcher-argument order.
_INCLUDE_KEYS = (
    "prompt_tokens",
    "prompt_positions",
    "prompt_window",
    "generation_tokens",
    "generation_positions",
    "generation_window",
)
_EXCLUDE_KEYS = tuple(f"exclude_{key}" for key in _INCLUDE_KEYS)


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
    return tuple((key, _canon(apply_spec.get(key))) for key in _CLAUSE_KEYS)


def selects_all_tokens(apply_spec: dict) -> bool:
    """Whether the clause selects every token of both phases.

    Enables the fast path that skips position collection and steers
    all of the slot's token rows directly.
    """
    return len(apply_spec["phases"]) == 2 and all(
        apply_spec.get(key) is None for key in _CLAUSE_KEYS if key != "phases"
    )


class ApplyClause:
    """Holds one intervention's where-clause."""

    def __init__(self):
        self.apply_spec: dict | None = None
        self.cache_key: tuple | None = None

    def configure_from_dict(self, config: dict) -> None:
        """Configure from a canonical steering-parameter dict.

        Driven by the canonical field registry: where-clause fields
        present in the dict are applied, everything else (payloads,
        algorithm parameters) is ignored.
        """
        from vllm.steer_vectors.request import STEER_CLAUSE_FIELDS

        for name in STEER_CLAUSE_FIELDS:
            if name in config:
                setattr(self, name, config[name])
        self.cache_key = clause_cache_key(self.apply_spec)

    def has_clause(self) -> bool:
        """Whether a where-clause is configured."""
        return self.apply_spec is not None

    def selects_all_tokens(self) -> bool:
        """Whether the clause selects every token of both phases."""
        return self.apply_spec is not None and selects_all_tokens(self.apply_spec)


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
    samples_info: dict[str, torch.Tensor],
    spec: dict,
) -> torch.Tensor | None:
    """Collect intervention positions for an `apply_spec` where-clause."""
    query_start_loc = samples_info["query_start_loc"]
    is_decode_mask = samples_info["is_decode_mask"]
    device = current_tokens.device
    # Size the masks from current_tokens: under piecewise cudagraphs the
    # hidden states are padded to the graph bucket, while current_tokens
    # and query_start_loc always cover the real tokens.
    total_tokens = current_tokens.shape[0]

    sample_ids, abs_positions, num_computed = resolve_batch_positions(
        samples_info, total_tokens, device
    )
    chunk_len = query_start_loc[1:] - query_start_loc[:-1]
    total_len = chunk_len if num_computed is None else chunk_len + num_computed
    num_prompt = samples_info.get("num_prompt_tokens")
    neg_base = total_len if num_prompt is None else num_prompt.to(device)
    is_decode_token = is_decode_mask[sample_ids]

    def _gen_idx() -> torch.Tensor:
        num_output_tokens = samples_info.get("num_output_tokens")
        if num_output_tokens is None:
            raise RuntimeError(
                "apply_spec selects decode steps but the runner did not "
                "provide num_output_tokens in samples_info"
            )
        # The decode step processing generated token j has
        # num_output_tokens == j + 1 (the prompt's last position, which
        # produces the first generated token, is a prompt-phase token).
        return num_output_tokens.to(device)[sample_ids] - 1

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
            matched |= (
                _isin_token_set(current_tokens, prompt_tokens) & ~is_decode_token
            )
        if generation_tokens is not None:
            matched |= (
                _isin_token_set(current_tokens, generation_tokens)
                & is_decode_token
            )
        if prompt_positions is not None:
            matched |= _match_positions(
                abs_positions, prompt_positions, neg_base, sample_ids,
                is_decode_token,
            )
        if prompt_window is not None:
            matched |= _match_prompt_window(
                abs_positions, prompt_window, neg_base, sample_ids, is_decode_token
            )
        if generation_positions is not None or generation_window is not None:
            matched |= _match_generation_steps(
                _gen_idx(), is_decode_token, generation_positions, generation_window
            )
        return matched

    phases = spec["phases"]
    mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    if "prompt" in phases:
        mask |= ~is_decode_token
    if "generation" in phases:
        mask |= is_decode_token

    includes = tuple(spec.get(key) for key in _INCLUDE_KEYS)
    if any(value is not None for value in includes):
        mask &= _selector_mask(*includes)

    excludes = tuple(spec.get(key) for key in _EXCLUDE_KEYS)
    if any(value is not None for value in excludes):
        mask &= ~_selector_mask(*excludes)

    positions_tensor = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if positions_tensor.numel() == 0:
        return None
    return positions_tensor
