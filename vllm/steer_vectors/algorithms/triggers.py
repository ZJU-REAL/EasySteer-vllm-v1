# SPDX-License-Identifier: Apache-2.0
"""Where-clause state and position matching for steering algorithms.

`TriggerController` holds one intervention's `apply_spec` (the wire form
of the user-facing ApplySpec: phases, token/position filters, exclusions
and the generation window); `collect_positions_apply_spec` turns it into
flat token positions for the current step. Algorithms stay pure
transformations over the positions computed here.

Semantics (see steer_vectors/api.py):
  candidates = tokens of the selected phases
  if tokens/positions given: candidates &= (token match | position match)
  candidates &= ~(exclude_tokens match | exclude_positions match)
  generation tokens are additionally constrained to the half-open window
  over 0-based decode steps (step j iff start <= j < stop).

There are no sentinel values and nothing bypasses exclusions.
"""

import torch

from vllm.steer_vectors.discovery import resolve_batch_positions


class TriggerController:
    """Holds one intervention's where-clause and debug flag."""

    def __init__(self):
        self.apply_spec: dict | None = None
        self.debug: bool = False

    def set_debug(self, debug: bool) -> None:
        """Set debug mode."""
        self.debug = debug

    def configure_from_dict(self, config: dict) -> None:
        """Configure from a canonical steering-parameter dict.

        Driven by the canonical field registry: where-clause fields
        present in the dict are applied, everything else (payloads,
        algorithm parameters) is ignored.
        """
        from vllm.steer_vectors.request import STEER_TRIGGER_FIELDS

        for name in STEER_TRIGGER_FIELDS + ("debug",):
            if name in config:
                setattr(self, name, config[name])

    def has_any_triggers(self) -> bool:
        """Whether a where-clause is configured."""
        return self.apply_spec is not None

    def is_global_only_config(self) -> bool:
        """Whether the clause selects every token of both phases.

        Enables the fast path that skips position collection and steers
        all of the slot's token rows directly.
        """
        if self.apply_spec is None:
            return False
        spec = self.apply_spec
        return len(spec["phases"]) == 2 and all(
            spec.get(key) is None
            for key in (
                "tokens",
                "positions",
                "exclude_tokens",
                "exclude_positions",
                "window",
            )
        )

    def collect_intervention_positions(
        self,
        hidden_states: torch.Tensor,
        current_tokens: torch.Tensor,
        samples_info: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Positions to steer for the current step.

        Args:
            hidden_states: [total_tokens, hidden_dim] (device source only)
            current_tokens: [total_tokens] token IDs
            samples_info: batch geometry (query_start_loc, num_computed,
                is_decode_mask, num_output_tokens, num_prompt_tokens)

        Returns:
            [num_positions] tensor of flat token indices, or None when
            nothing matches.
        """
        if self.apply_spec is None:
            return None
        return collect_positions_apply_spec(
            current_tokens=current_tokens,
            samples_info=samples_info,
            spec=self.apply_spec,
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
) -> torch.Tensor:
    """[total_tokens] mask of tokens at the given absolute positions.

    Positive entries match absolute positions directly; negative entries
    are Python-style indices from each sample's prompt length in
    `total_len_per_sample` (stable across prefill chunks).
    """
    mask = torch.zeros_like(abs_positions, dtype=torch.bool)
    positive = [p for p in positions if p >= 0]
    negative = [p for p in positions if p < 0]
    if positive:
        mask |= torch.isin(
            abs_positions,
            torch.tensor(
                positive,
                dtype=abs_positions.dtype,
                device=abs_positions.device,
            ),
        )
    if negative:
        totals = total_len_per_sample[sample_ids]
        for neg_idx in negative:
            mask |= abs_positions == totals + neg_idx
    return mask


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

    phases = spec["phases"]
    mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    if "prompt" in phases:
        mask |= ~is_decode_token
    if "generation" in phases:
        mask |= is_decode_token

    tokens = spec.get("tokens")
    positions = spec.get("positions")
    if tokens is not None or positions is not None:
        trigger = torch.zeros(total_tokens, dtype=torch.bool, device=device)
        if tokens is not None:
            trigger |= _isin_token_set(current_tokens, tokens)
        if positions is not None:
            trigger |= _match_positions(abs_positions, positions, neg_base, sample_ids)
        mask &= trigger

    exclude_tokens = spec.get("exclude_tokens")
    if exclude_tokens is not None:
        mask &= ~_isin_token_set(current_tokens, exclude_tokens)
    exclude_positions = spec.get("exclude_positions")
    if exclude_positions is not None:
        mask &= ~_match_positions(
            abs_positions, exclude_positions, neg_base, sample_ids
        )

    window = spec.get("window")
    if window is not None:
        num_output_tokens = samples_info.get("num_output_tokens")
        if num_output_tokens is None:
            raise RuntimeError(
                "apply_spec has a generation window but the runner did not "
                "provide num_output_tokens in samples_info"
            )
        # The decode step processing generated token j has
        # num_output_tokens == j + 1 (the prompt's last position, which
        # produces the first generated token, is a prompt-phase token).
        gen_idx = num_output_tokens.to(device)[sample_ids] - 1
        start, stop = window
        in_window = gen_idx >= start
        if stop is not None:
            in_window &= gen_idx < stop
        mask &= (~is_decode_token) | in_window

    positions_tensor = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if positions_tensor.numel() == 0:
        return None
    return positions_tensor
