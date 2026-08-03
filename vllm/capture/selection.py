# SPDX-License-Identifier: Apache-2.0
"""Row selection, reduction and labelling for capture streams."""

import torch

from vllm.capture.store import StreamStore
from vllm.logger import init_logger

logger = init_logger(__name__)


def prepare_rows(
    tensor: torch.Tensor,
    store: StreamStore,
    stream: str = "",
    request_selects: "dict[str, dict[str, dict]] | None" = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Select/reduce a [tokens, dim] step tensor and label the rows.

    Returns ``(rows, meta)``: the rows to store and an int32
    ``[rows, 3]`` label tensor of (req_idx, position, token_id), where
    req_idx indexes the store's req_table. ``rows`` is None when
    nothing matches this step. ``meta`` is None when batch geometry or
    request identity is unavailable (fallback: all rows, unlabeled).

    Selection resolves the stream's `select` clause with the same
    trigger collector steering uses, so the clause semantics are
    identical. Reductions:
    - 'last': one row per sample per step; under chunked prefill only
      samples whose chunk is final contribute (one row per logical
      step, not one per chunk). Labelled with the row's own position
      and token.
    - 'mean': one synthesized row per sample per step, labelled with
      position -1 and token_id -1. Continuation chunks produce
      per-chunk means (warned once) — use 'all' and reduce client-side
      when chunked prompts need exact means.
    """
    from vllm.forward_context import get_forward_context
    from vllm.steer_vectors.algorithms.triggers import (
        collect_positions_apply_spec,
    )
    from vllm.steer_vectors.discovery import (
        extract_samples_info,
        resolve_batch_positions,
    )

    config = store.config
    ctx = get_forward_context()
    samples_info = (
        extract_samples_info(ctx.attn_metadata)
        if ctx is not None and ctx.attn_metadata is not None
        else None
    )
    geo = getattr(ctx, "batch_geometry", None) if ctx else None
    current_tokens = geo.token_ids if geo is not None else None
    ctx_req_ids = geo.req_ids if geo is not None else None

    # Per-request selection overrides for this stream, keyed by the
    # batch sample index (request ids in the context are the
    # engine-internal ids, the same namespace add_request sees).
    overrides: dict[int, dict] = {}
    if request_selects and ctx_req_ids is not None:
        for i, rid in enumerate(ctx_req_ids):
            per_req = request_selects.get(rid)
            if per_req is not None and stream in per_req:
                overrides[i] = per_req[stream]

    needs_selection = config.selects_rows or bool(overrides)
    if samples_info is None or (needs_selection and current_tokens is None):
        if needs_selection or config.reduce != "all":
            logger.warning_once(
                "Capture selection/reduction requested but batch "
                "geometry is unavailable; keeping all rows unlabelled "
                "for this step."
            )
        return tensor, None

    device = tensor.device
    qsl = samples_info["query_start_loc"].to(device)
    total = int(qsl[-1].item())
    total = min(total, tensor.shape[0])
    num_prompt = samples_info.get("num_prompt_tokens")

    # Kept alongside resolve_batch_positions: the "all" path below uses
    # this exact object as a keep-every-row sentinel to skip the gather.
    all_positions = torch.arange(total, device=device)
    sample_ids, abs_positions, num_computed = resolve_batch_positions(
        samples_info, total, device
    )

    if overrides:
        # Rows of samples WITHOUT an override follow the global config
        # ("all" or a global select clause; reductions cannot combine
        # with overrides, enforced at add_request). Each override group
        # replaces its samples' selection with its own clause.
        tokens_dev = current_tokens[:total].to(device)
        mask = torch.zeros(total, dtype=torch.bool, device=device)
        if config.selects_rows:
            base = collect_positions_apply_spec(
                current_tokens=tokens_dev,
                samples_info=samples_info,
                spec=config.select,
            )
            if base is not None:
                mask[base] = True
        else:
            mask[:] = True
        override_samples = torch.tensor(
            sorted(overrides), dtype=sample_ids.dtype, device=device
        )
        mask &= ~torch.isin(sample_ids, override_samples)
        groups: dict[str, tuple[dict, list[int]]] = {}
        for i, wire in overrides.items():
            key = repr(sorted(wire.items()))
            groups.setdefault(key, (wire, []))[1].append(i)
        for wire, samples in groups.values():
            idx = collect_positions_apply_spec(
                current_tokens=tokens_dev,
                samples_info=samples_info,
                spec=wire,
            )
            if idx is None:
                continue
            gmask = torch.zeros(total, dtype=torch.bool, device=device)
            gmask[idx] = True
            gmask &= torch.isin(
                sample_ids,
                torch.tensor(samples, dtype=sample_ids.dtype, device=device),
            )
            mask |= gmask
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return None, None
    elif config.selects_rows:
        indices = collect_positions_apply_spec(
            current_tokens=current_tokens[:total].to(device),
            samples_info=samples_info,
            spec=config.select,
        )
        if indices is None:
            return None, None
    elif config.reduce == "last":
        starts = qsl[:-1].clamp(max=total)
        ends = qsl[1:].clamp(max=total)
        indices = (ends - 1).clamp(min=0)
        if num_computed is not None and num_prompt is not None:
            # Keep decode samples and final prefill chunks only.
            final = (num_computed.to(ends.device) + (ends - starts)) >= (
                num_prompt.to(ends.device)
            )
            indices = indices[final]
    elif config.reduce == "mean":
        return _mean_reduce(tensor, store, ctx, qsl, total, num_computed, num_prompt)
    else:  # "all"
        indices = all_positions

    rows = tensor[:total][indices] if indices is not all_positions else tensor[:total]
    meta = _row_labels(
        store, ctx, indices, sample_ids, abs_positions, current_tokens, total
    )
    return rows, meta


def _row_labels(
    store: StreamStore,
    ctx,
    indices: torch.Tensor,
    sample_ids: torch.Tensor,
    abs_positions: torch.Tensor,
    current_tokens: torch.Tensor | None,
    total: int,
) -> torch.Tensor | None:
    """Build int32 [rows, 3] (req_idx, position, token_id) labels."""
    geo = getattr(ctx, "batch_geometry", None) if ctx else None
    req_ids = geo.req_ids if geo is not None else None
    if req_ids is None or current_tokens is None:
        logger.warning_once(
            "Capture cannot label rows: the runner did not provide "
            "req_ids/current_tokens for this step."
        )
        return None
    device = sample_ids.device
    step_req = torch.tensor(
        [store.req_index(r) for r in req_ids],
        dtype=torch.int32,
        device=device,
    )
    sel_samples = sample_ids[indices]
    return torch.stack(
        [
            step_req[sel_samples],
            abs_positions[indices].to(torch.int32),
            current_tokens[:total].to(device)[indices].to(torch.int32),
        ],
        dim=1,
    )


def _mean_reduce(
    tensor: torch.Tensor,
    store: StreamStore,
    ctx,
    qsl: torch.Tensor,
    total: int,
    num_computed,
    num_prompt,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One mean row per sample; labels use position/token sentinel -1."""
    starts = qsl[:-1].clamp(max=total)
    ends = qsl[1:].clamp(max=total)
    if (
        num_computed is not None
        and num_prompt is not None
        and bool(((num_computed > 0) & (num_computed < num_prompt)).any())
    ):
        logger.warning_once(
            "Capture 'mean' reduction under chunked prefill produces one "
            "mean per chunk, not per prompt; use reduce='all' and "
            "reduce client-side for chunked prompts."
        )
    rows = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        rows.append(tensor[s:e].mean(dim=0) if e > s else torch.zeros_like(tensor[0]))
    stacked = torch.stack(rows)
    geo = getattr(ctx, "batch_geometry", None) if ctx else None
    req_ids = geo.req_ids if geo is not None else None
    if req_ids is None:
        logger.warning_once(
            "Capture cannot label rows: the runner did not provide "
            "req_ids for this step."
        )
        return stacked, None
    step_req = torch.tensor(
        [store.req_index(r) for r in req_ids],
        dtype=torch.int32,
        device=stacked.device,
    )
    sentinel = torch.full_like(step_req, -1)
    return stacked, torch.stack([step_req, sentinel, sentinel], dim=1)
