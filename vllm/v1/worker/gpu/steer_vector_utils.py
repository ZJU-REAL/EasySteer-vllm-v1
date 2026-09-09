# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steer vector support for the V2 GPU model runner."""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.model_hooks.selection.host import clause_mask
from vllm.model_hooks.steering import trace
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.model_hooks.selection.batch import BatchGeometry


def _batch_request_slots(
    input_batch: InputBatch,
    manager,
) -> np.ndarray:
    """Config-slot routing in the current request order."""
    num_reqs = input_batch.num_reqs
    return np.fromiter(
        (
            -1 if (slot := manager.slot_for_request(req_id)) is None else slot
            for req_id in input_batch.req_ids
        ),
        dtype=np.int32,
        count=num_reqs,
    )


def resolve_slot_positions(
    slot_clauses: dict[int, list[dict | None]],
    active_slots: list[int],
    request_slots_np: np.ndarray,
    device: torch.device,
    geo,
    *,
    slot_groups: dict | None = None,
    errors: dict[str, str] | None = None,
) -> dict[tuple, torch.Tensor | None]:
    """Resolve every active clause's steered positions, once per step.

    Where-clauses are layer-invariant, so this single resolution serves
    every decoder/MoE-gate hook (and the Tier-1 mask filler). Keys are
    (slot, clause_cache_key), or (slot, intervention_group, index) when
    resolving compositions; None means no selected token this step.

    Resolution runs host-side in one numpy pass: clauses match phases,
    positions and windows — all host-known geometry — so each slot's
    clauses are evaluated only over that slot's own token rows and the
    matched positions ship to the device in a single copy. Per-step cost
    scales with the batch's tokens, not with the number of distinct live
    configurations. Only token-id filters read the input ids (one cached
    device-to-host copy per step).
    """
    from vllm.model_hooks.selection.runtime import (
        clause_cache_key,
        selects_all_tokens,
    )

    resolved: dict[tuple, torch.Tensor | None] = {}
    if not active_slots:
        return resolved

    qsl = geo.query_start_loc_cpu
    num_computed = geo.num_computed.numpy()
    num_prompt = geo.num_prompt.numpy()
    num_output = geo.num_output.numpy()
    lens = (qsl[1:] - qsl[:-1]).astype(np.int64)
    starts_all = qsl[:-1].astype(np.int64)
    is_decode_req = num_output > 0

    # Empty segments contribute no rows. Actual offsets still determine
    # each request's row range after adaptive draft redistribution.
    active = set(active_slots)
    slot_reqs: dict[int, list[int]] = {}
    for r in np.flatnonzero(lens > 0):
        s = int(request_slots_np[r])
        if s in active:
            slot_reqs.setdefault(s, []).append(r)

    for slot in active_slots:
        reqs = slot_reqs.get(slot)
        clauses = slot_clauses.get(slot, [])
        if not reqs:
            for clause in clauses:
                key = clause_cache_key(clause)
                if key is not None:
                    resolved.setdefault((slot, key), None)
            continue
        reqs_np = np.asarray(reqs, dtype=np.int64)
        seg_lens = lens[reqs_np]
        n = int(seg_lens.sum())
        samp = np.repeat(reqs_np, seg_lens)
        within = np.arange(n, dtype=np.int64) - np.repeat(
            np.cumsum(seg_lens) - seg_lens, seg_lens
        )
        tok_idx = np.repeat(starts_all[reqs_np], seg_lens) + within
        abs_pos = within + num_computed[samp]
        is_dec = abs_pos >= num_prompt[samp]
        generation_indices = np.where(
            is_decode_req[samp], num_output[samp] - 1, abs_pos - num_prompt[samp]
        )

        for clause in clauses:
            key = clause_cache_key(clause)
            if clause is None or key is None or (slot, key) in resolved:
                continue
            if selects_all_tokens(clause):
                pos_np = tok_idx
            else:
                mask = clause_mask(
                    clause,
                    is_dec,
                    abs_pos,
                    num_prompt[samp],
                    generation_indices,
                    lambda tok_idx=tok_idx: geo.token_ids_cpu()[tok_idx],
                )
                pos_np = tok_idx[mask]
            if pos_np.shape[0] == 0:
                resolved[(slot, key)] = None
            else:
                resolved[(slot, key)] = pos_np  # placeholder, replaced below

    if slot_groups is not None:
        grouped = {}
        failed_rows = []
        for slot in active_slots:
            for group in slot_groups.get(slot, ()):
                mode, clause_keys = group
                claimed = np.empty(0, dtype=np.int64)
                for index, clause_key in enumerate(clause_keys):
                    positions = resolved.get((slot, clause_key))
                    if (
                        positions is not None
                        and mode != "sequential"
                        and len(clause_keys) > 1
                    ):
                        overlap = np.isin(positions, claimed)
                        if mode == "error" and overlap.any():
                            failed_rows.append(positions[overlap])
                        if mode == "priority":
                            positions = positions[~overlap]
                        claimed = np.union1d(claimed, positions)
                    grouped[(slot, group, index)] = positions
        resolved = grouped
        if failed_rows:
            assert errors is not None, "Conflict errors require per-request reporting"
            failed_reqs = np.unique(
                np.searchsorted(qsl[1:], np.concatenate(failed_rows), side="right")
            )
            for index in failed_reqs:
                errors[geo.req_ids[index]] = (
                    "Steering vectors conflict at selected token positions; "
                    "use conflict_resolution='priority' or 'sequential'."
                )
            for key, positions in resolved.items():
                if positions is not None:
                    owners = np.searchsorted(qsl[1:], positions, side="right")
                    resolved[key] = positions[~np.isin(owners, failed_reqs)]

    keys = []
    chunks = []
    for key, positions in resolved.items():
        if positions is None or positions.size == 0:
            resolved[key] = None
        else:
            keys.append(key)
            chunks.append(positions)

    if chunks:
        flat = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        flat_t = torch.from_numpy(flat).to(device, non_blocking=True)
        offset = 0
        for key, chunk in zip(keys, chunks):
            size = chunk.shape[0]
            resolved[key] = flat_t[offset : offset + size]
            offset += size
    return resolved


def make_steering_forward_kwargs(
    input_batch: InputBatch,
    manager,
    *,
    geometry: "BatchGeometry",
    errors: dict[str, str] | None = None,
) -> dict:
    """Resolve the routing fields consumed by split/eager steering hooks."""
    slots_np = _batch_request_slots(input_batch, manager)
    active_slots = sorted({int(s) for s in slots_np if s >= 0})
    positions = resolve_slot_positions(
        manager.slot_clauses(),
        active_slots,
        slots_np,
        input_batch.input_ids.device,
        geometry,
        slot_groups=manager.slot_groups(),
        errors=errors,
    )

    if trace.enabled():
        trace.begin_step(
            req_ids=input_batch.req_ids,
            slots=slots_np.tolist(),
            query_start_loc=geometry.query_start_loc_cpu.tolist(),
            token_ids=geometry.token_ids_cpu().tolist(),
            num_computed=geometry.num_computed.tolist(),
            num_output=geometry.num_output.tolist(),
        )
    return {
        "steer_active_slots": active_slots,
        "steer_slot_positions": positions,
    }


def fill_graph_steer_buffers(
    input_batch: InputBatch,
    manager,
    geometry: "BatchGeometry | None" = None,
) -> None:
    """Fill Tier-1 persistent buffers for this step (full-graph mode).

    Writes each token's vector-table row into the shared row buffer and
    sets the per-layer trigger masks to 1 at steered positions. The
    captured kernel `hidden += mask * vectors[row_tok]` then applies the
    right configs without any per-step graph work; row 0 / mask 0 keep
    unsteered and padding tokens untouched. Positions come from the same
    resolver the layer hooks use (resolve_slot_positions).
    """
    from vllm.model_hooks.selection.runtime import clause_cache_key

    padded_tokens = input_batch.num_tokens_after_padding
    manager.zero_graph_masks(padded_tokens)
    row_buf = manager.token_rows_buf
    row_buf[:padded_tokens].zero_()
    entries = manager.graph_batch_entries()
    if not entries:
        return

    assert geometry is not None, "Active graph steering requires batch geometry"
    geo = geometry
    slots_np = _batch_request_slots(input_batch, manager)
    rows_np = np.fromiter(
        (entries[s][0] if s in entries else 0 for s in slots_np),
        dtype=np.int64,
        count=slots_np.shape[0],
    )
    num_scheduled = np.diff(geo.query_start_loc_cpu)
    token_rows_np = np.repeat(rows_np, num_scheduled)
    n = token_rows_np.shape[0]
    if n == 0:
        return
    device = row_buf.device
    row_buf[:n].copy_(torch.from_numpy(token_rows_np), non_blocking=True)

    batch_slots = set(slots_np.tolist())
    active_slots = sorted(s for s in entries if s in batch_slots)
    resolved = resolve_slot_positions(
        manager.slot_clauses(), active_slots, slots_np, device, geo
    )
    from vllm.model_hooks.steering.algorithms import get_algorithm
    from vllm.model_hooks.steering.graph.kernels import graph_family_mask_attr

    # Controllers' masks are views of one allocation. Combine their writes
    # into one scatter instead of launching cat/scatter for every layer.
    mask_writes: dict[tuple[int, str], tuple] = {}
    for slot in active_slots:
        _, request, controllers = entries[slot]
        vector = request.vectors[0]
        positions = resolved[(slot, clause_cache_key(vector.apply_spec))]
        if positions is None:
            continue
        # Families whose delta a zero table row cannot neutralize (e.g.
        # replace) carry their own mask; see GRAPH_FAMILY_MASKS.
        mask_attr = graph_family_mask_attr(get_algorithm(vector.algorithm).graph_family)
        for module in controllers:
            mask_writes.setdefault((id(module), mask_attr), (module, mask_attr, []))[
                2
            ].append(positions)
    if mask_writes:
        positions = []
        offsets = []
        for module, mask_attr, position_list in mask_writes.values():
            positions.extend(position_list)
            offsets.extend(
                [getattr(module, mask_attr).storage_offset()] * len(position_list)
            )
        flat_positions = positions[0] if len(positions) == 1 else torch.cat(positions)
        row_offsets = np.repeat(
            np.asarray(offsets, dtype=np.int64), [p.numel() for p in positions]
        )
        indices = flat_positions + torch.from_numpy(row_offsets).to(
            device, non_blocking=True
        )
        manager.graph_masks_buf.view(-1)[indices] = 1.0
