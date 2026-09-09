# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row selection, reduction and labelling for capture streams."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_hooks.capture.store import StreamStore
from vllm.model_hooks.selection.batch import BatchView

if TYPE_CHECKING:
    from vllm.model_hooks.selection.batch import BatchGeometry

logger = init_logger(__name__)


@dataclass
class _RowPlan:
    # None indices means every real row, without a gather.
    indices: torch.Tensor | None
    meta: torch.Tensor
    mean_spans: list[tuple[int, int]] | None = None


def _may_select_prompt_head(select: dict) -> bool:
    return (
        select.get("prompt") == "all"
        or select.get("prompt_tokens") is not None
        or select.get("prompt_window") is not None
        or any(position != -1 for position in select.get("prompt_positions") or ())
    )


def selects_skipped_prompt_rows(
    select: dict | None,
    prompt_token_ids: list[int],
    num_computed_tokens: int,
    *,
    reduce: str = "all",
) -> bool:
    """Whether a cache hit omitted rows required by an effective selection.

    Resolve on CPU at admission with the actual skipped prefix and original
    prompt length. The shared collector preserves token matching, exclusions,
    negative positions, windows and clamping exactly as in the capture hooks.
    """
    skipped = min(num_computed_tokens, len(prompt_token_ids))
    if skipped <= 0:
        return False
    if reduce == "last":
        return skipped == len(prompt_token_ids)
    if reduce == "mean" or select is None:
        return True
    if skipped < len(prompt_token_ids) and not _may_select_prompt_head(select):
        return False

    from vllm.model_hooks.selection.runtime import collect_positions_apply_spec

    batch = BatchView(
        query_start_loc=torch.tensor([0, skipped], device="cpu"),
        num_computed=torch.tensor([0], device="cpu"),
        num_prompt=torch.tensor([len(prompt_token_ids)], device="cpu"),
        num_output=torch.tensor([0], device="cpu"),
    )
    tokens = torch.tensor(prompt_token_ids[:skipped], device="cpu")
    return collect_positions_apply_spec(tokens, batch, select) is not None


def needs_prompt_recompute(
    select: dict | None,
    reduce: str = "all",
    prompt_token_ids: list[int] | None = None,
) -> bool:
    """Whether possible prefix hits can omit wanted rows of this prompt.

    vLLM leaves at least the final prompt token for local computation. Once
    tokenized, check the largest possible skipped prefix. Before tokenization,
    only selections provably limited to that final token or decode are safe.
    """
    if prompt_token_ids is not None:
        return selects_skipped_prompt_rows(
            select, prompt_token_ids, len(prompt_token_ids) - 1, reduce=reduce
        )
    if reduce == "last":
        return False
    if reduce == "mean" or select is None:
        return True
    return _may_select_prompt_head(select)


def may_select_step_rows(
    select: dict | None,
    num_computed: int,
    num_scheduled: int,
    prompt_len: int,
    is_prefilling: bool,
    *,
    reduce: str = "all",
) -> bool:
    """Conservatively decide on CPU whether this request needs capture.

    Scheduling counts may overestimate adaptive verification's actual rows.
    Token IDs are not synchronized to CPU for dispatch: includes are widened
    to the whole phase and token exclusions are ignored. A false result proves
    the actual collector selects nothing; an uncertain result keeps eager.
    """
    if num_scheduled <= 0:
        return False
    if reduce == "last":
        return num_computed + num_scheduled >= prompt_len
    if reduce == "mean" or select is None:
        return True
    phases = (
        ("prompt", "generation")
        if is_prefilling and num_computed + num_scheduled > prompt_len
        else ("prompt" if is_prefilling else "generation",)
    )
    if not any(
        select.get(phase) == "all"
        or any(
            select.get(f"{phase}_{field}") is not None
            for field in ("tokens", "positions", "window")
        )
        for phase in phases
    ):
        return False
    upper = dict(select)
    for name in ("prompt", "generation"):
        if upper.pop(f"{name}_tokens", None) is not None:
            upper[name] = "all"
        upper.pop(f"exclude_{name}_tokens", None)
    if any(
        upper.get(phase) == "all"
        and not any(
            upper.get(f"exclude_{phase}_{field}") is not None
            for field in ("positions", "window")
        )
        for phase in phases
    ):
        return True

    from vllm.model_hooks.selection.runtime import collect_positions_apply_spec

    batch = BatchView(
        query_start_loc=torch.tensor([0, num_scheduled], device="cpu"),
        num_computed=torch.tensor([num_computed], device="cpu"),
        num_prompt=torch.tensor([prompt_len], device="cpu"),
        num_output=torch.tensor(
            [0 if is_prefilling else num_computed - prompt_len + 1], device="cpu"
        ),
    )
    tokens = torch.zeros(num_scheduled, dtype=torch.long, device="cpu")
    return collect_positions_apply_spec(tokens, batch, upper) is not None


def count_selected_rows(
    geo: "BatchGeometry",
    store: StreamStore,
    stream: str,
    request_selects: dict[str, dict[str, dict]],
) -> int:
    """Count budget drops without reading or copying model activations.

    Positional selection uses existing host geometry. Only token predicates
    need the device collector; token ids are never copied back for accounting.
    """
    from vllm.model_hooks.selection.runtime import collect_positions_apply_spec

    offsets = geo.query_start_loc_cpu
    if offsets[-1] == 0:
        return 0
    if store.config.reduce == "mean":
        return len(geo.req_ids)
    if store.config.reduce == "last":
        sizes = offsets[1:] - offsets[:-1]
        return int(
            (
                (sizes > 0)
                & (geo.num_computed.numpy() + sizes >= geo.num_prompt.numpy())
            ).sum()
        )
    count = 0
    for index, req_id in enumerate(geo.req_ids):
        start, end = int(offsets[index]), int(offsets[index + 1])
        if end <= start:
            continue
        spec = request_selects.get(req_id, {}).get(stream, store.config.select)
        if spec is None:
            count += end - start
            continue
        token_predicate = any(
            key.endswith("_tokens") and value is not None for key, value in spec.items()
        )
        device = geo.token_ids.device if token_predicate else torch.device("cpu")
        tokens = (
            geo.token_ids[start:end]
            if token_predicate
            else torch.zeros(end - start, dtype=torch.long, device=device)
        )
        batch = BatchView(
            query_start_loc=torch.tensor([0, end - start], device=device),
            num_computed=geo.num_computed[index : index + 1].to(device),
            num_prompt=geo.num_prompt[index : index + 1].to(device),
            num_output=geo.num_output[index : index + 1].to(device),
        )
        indices = collect_positions_apply_spec(tokens, batch, spec)
        if indices is not None:
            count += indices.numel()
    return count


def prepare_rows(
    tensor: torch.Tensor,
    store: StreamStore,
    layer_id: int,
    stream: str = "",
    request_selects: "dict[str, dict[str, dict]] | None" = None,
    tensor_owned: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Select/reduce step rows with a plan shared by this stream's layers.

    Plans contain only indices, labels and reduction boundaries, never layer
    values. The runner supplies fresh geometry each step; stream configuration
    and request selections stay fixed within that forward pass.
    """
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    geo = ctx.batch_geometry
    if geo is None:
        raise RuntimeError(
            "Capture requires the runner's BatchGeometry and selected token ids; "
            "cannot capture correctly labelled rows without them"
        )
    total = min(int(geo.query_start_loc_cpu[-1]), tensor.shape[0])
    if total == 0:
        return None, None
    if (
        store._row_plan_geometry is not geo
        or store._row_plan_request_selects is not request_selects
    ):
        store._row_plan_geometry = geo
        store._row_plan_request_selects = request_selects
        store._row_plans.clear()
    key = (stream, tensor.device, total)
    if key not in store._row_plans:
        store._row_plans[key] = _build_row_plan(
            geo, store, stream, request_selects, total, tensor.device
        )
    plan = store._row_plans[key]
    if plan is None:
        return None, None
    count = plan.meta.shape[0]
    keep = store.limit_rows(layer_id, count)
    if keep == 0:
        return None, None
    if plan.mean_spans is not None:
        rows = torch.stack(
            [
                tensor[start:end].mean(dim=0)
                if end > start
                else torch.zeros_like(tensor[0])
                for start, end in plan.mean_spans[:keep]
            ]
        )
    elif plan.indices is None:
        # Staged rows must survive subsequent model operations until flush().
        rows = tensor[:keep] if tensor_owned else tensor[:keep].clone()
    else:
        rows = tensor[:total][plan.indices[:keep]]
    return rows, plan.meta if keep == count else plan.meta[:keep]


def _build_row_plan(
    geo: "BatchGeometry",
    store: StreamStore,
    stream: str,
    request_selects: "dict[str, dict[str, dict]] | None",
    total: int,
    device: torch.device,
) -> _RowPlan | None:
    from vllm.model_hooks.selection.runtime import (
        collect_positions_apply_spec,
        resolve_batch_positions,
    )

    config = store.config
    overrides = {}
    if request_selects and config.reduce == "all":
        for i, rid in enumerate(geo.req_ids):
            per_req = request_selects.get(rid)
            if per_req is not None and stream in per_req:
                overrides[i] = per_req[stream]

    if config.reduce == "mean" and not overrides:
        return _mean_plan(geo, _request_indices(geo, store, device), total)

    batch = geo.device_view(device)
    coordinates = resolve_batch_positions(batch, total, device)
    sample_ids, abs_positions = coordinates
    tokens = geo.token_ids[:total].to(device)

    def collect(spec):
        return collect_positions_apply_spec(
            current_tokens=tokens,
            batch=batch,
            spec=spec,
            coordinates=coordinates,
        )

    indices = None
    if overrides:
        mask = torch.zeros(total, dtype=torch.bool, device=device)
        if config.selects_rows:
            base = collect(config.select)
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
            idx = collect(wire)
            if idx is None:
                continue
            group_mask = torch.zeros(total, dtype=torch.bool, device=device)
            group_mask[idx] = True
            group_mask &= torch.isin(
                sample_ids,
                torch.tensor(samples, dtype=sample_ids.dtype, device=device),
            )
            mask |= group_mask
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return None
    elif config.selects_rows:
        indices = collect(config.select)
        if indices is None:
            return None
    elif config.reduce == "last":
        # Host offsets are the authoritative snapshot even when adaptive
        # verification changes the device layout. Avoid per-layer GPU reads.
        qsl = geo.query_start_loc_cpu.clip(max=total)
        starts, ends = qsl[:-1], qsl[1:]
        final = (ends > starts) & (
            geo.num_computed.numpy() + ends - starts >= geo.num_prompt.numpy()
        )
        positions = (ends - 1).clip(min=0)
        indices = torch.as_tensor(positions[final], dtype=torch.long, device=device)
        if indices.numel() == 0:
            return None

    if indices is not None:
        sample_ids = sample_ids[indices]
        abs_positions = abs_positions[indices]
        tokens = tokens[indices]
    step_req = _request_indices(geo, store, device)
    meta = torch.stack(
        [step_req[sample_ids], abs_positions.to(torch.int32), tokens.to(torch.int32)],
        dim=1,
    )
    return _RowPlan(indices, meta)


def _request_indices(
    geo: "BatchGeometry", store: StreamStore, device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [store.req_index(r) for r in geo.req_ids], dtype=torch.int32, device=device
    )


def _mean_plan(geo: "BatchGeometry", step_req: torch.Tensor, total: int) -> _RowPlan:
    """One mean per sample/chunk, with synthetic position/token labels."""
    computed, prompt = geo.num_computed.numpy(), geo.num_prompt.numpy()
    if ((computed > 0) & (computed < prompt)).any():
        logger.warning_once(
            "Capture 'mean' reduction under chunked prefill produces one "
            "mean per chunk, not per prompt; use reduce='all' and "
            "reduce client-side for chunked prompts."
        )
    offsets = geo.query_start_loc_cpu.clip(max=total).tolist()
    sentinel = torch.full_like(step_req, -1)
    meta = torch.stack([step_req, sentinel, sentinel], dim=1)
    return _RowPlan(None, meta, list(zip(offsets[:-1], offsets[1:])))
