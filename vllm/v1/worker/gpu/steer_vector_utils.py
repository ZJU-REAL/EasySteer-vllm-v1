# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steer vector support for the V2 GPU model runner."""

import numpy as np
import torch

from vllm.steer_vectors import trace
from vllm.steer_vectors.request import SteerVectorRequest
from vllm.v1.worker.gpu.input_batch import InputBatch


class SteerVectorState:
    """Per-request steer vector bookkeeping for the V2 model runner.

    Each live request resolves to a config slot at admission time
    (payload loading + layer distribution happen there, never in the
    forward pass).
    """

    def __init__(self) -> None:
        self._requests: dict[str, SteerVectorRequest] = {}
        self._slots: dict[str, int] = {}

    def add_request(
        self,
        req_id: str,
        steer_vector_request: SteerVectorRequest | None,
        manager,
    ) -> None:
        if steer_vector_request is None:
            return
        if manager is None:
            # Admission should have rejected this; routing the request to
            # the server config (or to nothing) would silently steer with
            # the wrong vector.
            raise RuntimeError(
                f"request {req_id} carries a steering config but this "
                "worker has no steer vector manager (engine launched "
                "without enable_steer_vector=True)"
            )
        self._requests[req_id] = steer_vector_request
        self._slots[req_id] = manager.acquire_config(req_id, steer_vector_request)

    def remove_request(self, req_id: str, manager) -> None:
        if self._requests.pop(req_id, None) is None:
            return
        self._slots.pop(req_id, None)
        if manager is not None:
            manager.release_config(req_id)

    def slot_of(self, req_id: str) -> int:
        return self._slots.get(req_id, -1)

    def has_routed(self) -> bool:
        return bool(self._slots)


def build_batch_geometry(input_batch: InputBatch) -> "BatchGeometry":
    """Build the per-step BatchGeometry from the runner's InputBatch.

    The single producer of batch geometry: steering triggers, capture
    row selection/labels, and the full-graph buffer filler all consume
    this object (directly or via `geometry_samples_info`).
    """
    from vllm.forward_context import BatchGeometry

    num_reqs = input_batch.num_reqs
    num_computed = input_batch.num_computed_tokens_np[:num_reqs]
    prefill_len = input_batch.prefill_len_np[:num_reqs]
    is_prefilling = input_batch.is_prefilling_np[:num_reqs]
    # While prefilling, nothing has been generated for this request yet.
    # During decode, the scheduler has computed prefill_len + (k - 1) tokens
    # when the k-th output token is being generated, matching the V1
    # semantics of len(output_token_ids) at execute time.
    num_output = np.where(is_prefilling, 0, num_computed - prefill_len + 1).astype(
        np.int32
    )
    return BatchGeometry(
        query_start_loc=input_batch.query_start_loc[: num_reqs + 1],
        num_computed=torch.from_numpy(np.ascontiguousarray(num_computed)),
        num_prompt=torch.from_numpy(np.ascontiguousarray(prefill_len)),
        num_output=torch.from_numpy(num_output),
        req_ids=list(input_batch.req_ids[:num_reqs]),
        token_ids=input_batch.input_ids[: input_batch.num_tokens],
        query_start_loc_cpu=input_batch.query_start_loc_np[: num_reqs + 1],
    )


def _batch_token_slots(
    input_batch: InputBatch, state: SteerVectorState, default_slot: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-request and per-token config-slot routing for this step."""
    num_reqs = input_batch.num_reqs
    slots_np = np.fromiter(
        (
            slot if (slot := state.slot_of(req_id)) >= 0 else default_slot
            for req_id in input_batch.req_ids
        ),
        dtype=np.int32,
        count=num_reqs,
    )
    token_slots_np = np.repeat(slots_np, input_batch.num_scheduled_tokens[:num_reqs])
    return slots_np, token_slots_np


def _match_positions_np(
    abs_pos: np.ndarray, positions, neg_base: np.ndarray
) -> np.ndarray:
    """Mask of tokens at the given absolute positions (numpy mirror of
    clause._match_positions): negative entries index from each sample's
    prompt length."""
    mask = np.zeros(abs_pos.shape[0], dtype=bool)
    positive = [p for p in positions if p >= 0]
    if positive:
        mask |= np.isin(abs_pos, np.asarray(positive, dtype=np.int64))
    for p in positions:
        if p < 0:
            mask |= abs_pos == neg_base + p
    return mask


def _clause_mask_np(
    clause: dict,
    is_dec: np.ndarray,
    abs_pos: np.ndarray,
    neg_base: np.ndarray,
    gen_idx: np.ndarray,
    token_ids,
) -> np.ndarray:
    """Evaluate one where-clause over a slot's tokens (numpy mirror of
    clause.collect_positions_apply_spec — same intersection semantics).

    `token_ids` is a thunk: only token-id filters pay for the host copy.
    """
    n = is_dec.shape[0]
    mask = np.zeros(n, dtype=bool)
    phases = clause["phases"]
    if "prompt" in phases:
        mask |= ~is_dec
    if "generation" in phases:
        mask |= is_dec

    tokens = clause.get("tokens")
    positions = clause.get("positions")
    if tokens is not None or positions is not None:
        trigger = np.zeros(n, dtype=bool)
        if tokens is not None:
            trigger |= np.isin(token_ids(), np.asarray(list(tokens)))
        if positions is not None:
            trigger |= _match_positions_np(abs_pos, positions, neg_base)
        mask &= trigger

    exclude_tokens = clause.get("exclude_tokens")
    if exclude_tokens is not None:
        mask &= ~np.isin(token_ids(), np.asarray(list(exclude_tokens)))
    exclude_positions = clause.get("exclude_positions")
    if exclude_positions is not None:
        mask &= ~_match_positions_np(abs_pos, exclude_positions, neg_base)

    window = clause.get("window")
    if window is not None:
        start, stop = window
        in_window = gen_idx >= start
        if stop is not None:
            in_window &= gen_idx < stop
        mask &= ~is_dec | in_window
    return mask


def resolve_slot_positions(
    slot_clauses: dict[int, list[dict | None]],
    active_slots: list[int],
    token_slots_np: np.ndarray,
    device: torch.device,
    geo,
) -> dict[tuple, torch.Tensor | None]:
    """Resolve every active clause's steered positions, once per step.

    Where-clauses are layer-invariant, so this single resolution serves
    every decoder/MoE-gate hook (and the Tier-1 mask filler). Keys are
    (slot, clause_cache_key); a None value means the clause matched no
    token this step.

    Resolution runs host-side in one numpy pass: clauses match phases,
    positions and windows — all host-known geometry — so each slot's
    clauses are evaluated only over that slot's own token rows and the
    matched positions ship to the device in a single copy. Per-step cost
    scales with the batch's tokens, not with the number of distinct live
    configurations. Only token-id filters read the input ids (one cached
    device-to-host copy per step).
    """
    from vllm.steer_vectors.algorithms.clause import (
        clause_cache_key,
        selects_all_tokens,
    )

    resolved: dict[tuple, torch.Tensor | None] = {}
    if not active_slots:
        return resolved

    qsl = geo.query_start_loc_cpu
    assert qsl is not None, "BatchGeometry is missing its host query_start_loc"
    num_computed = geo.num_computed.numpy()
    num_prompt = geo.num_prompt.numpy()
    num_output = geo.num_output.numpy()
    lens = (qsl[1:] - qsl[:-1]).astype(np.int64)
    starts_all = qsl[:-1].astype(np.int64)
    is_decode_req = num_output > 0

    # Group batch requests by routing slot (a request's slot is its
    # first token's slot; all its tokens share it).
    active = set(active_slots)
    slot_reqs: dict[int, list[int]] = {}
    for r, s in enumerate(token_slots_np[starts_all].tolist()):
        if s in active:
            slot_reqs.setdefault(s, []).append(r)

    keys: list[tuple[int, tuple]] = []
    chunks: list[np.ndarray] = []

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
        is_dec = is_decode_req[samp]

        for clause in clauses:
            key = clause_cache_key(clause)
            if key is None or (slot, key) in resolved:
                continue
            if selects_all_tokens(clause):
                pos_np = tok_idx
            else:
                mask = _clause_mask_np(
                    clause,
                    is_dec,
                    abs_pos,
                    num_prompt[samp],
                    num_output[samp] - 1,
                    lambda: geo.token_ids_cpu()[tok_idx],
                )
                pos_np = tok_idx[mask]
            if pos_np.shape[0] == 0:
                resolved[(slot, key)] = None
            else:
                resolved[(slot, key)] = pos_np  # placeholder, replaced below
                keys.append((slot, key))
                chunks.append(pos_np)

    if chunks:
        flat = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        flat_t = torch.from_numpy(flat).to(device, non_blocking=True)
        offset = 0
        for key, chunk in zip(keys, chunks):
            size = chunk.shape[0]
            resolved[key] = flat_t[offset : offset + size]
            offset += size
    return resolved


def make_steer_vector_forward_kwargs(
    input_batch: InputBatch,
    state: SteerVectorState | None = None,
    default_slot: int = -1,
    manager=None,
) -> dict:
    """Build the ForwardContext fields consumed by steering and capture.

    - batch_geometry: the per-step BatchGeometry (see build_batch_geometry)
    - steer_token_slots / steer_active_slots: per-request config routing
      (only when routed configs are live)
    - steer_slot_positions: per-clause steered positions, resolved once
      here and consumed by every layer hook (see resolve_slot_positions)

    `default_slot` is the server-level config's slot (-1 when absent);
    requests without their own steering config are routed to it.
    """
    num_reqs = input_batch.num_reqs
    geo = build_batch_geometry(input_batch)
    kwargs = {"batch_geometry": geo}

    if state is not None and (state.has_routed() or default_slot >= 0):
        slots_np, token_slots_np = _batch_token_slots(input_batch, state, default_slot)
        token_slots = torch.from_numpy(token_slots_np).to(
            input_batch.input_ids.device, non_blocking=True
        )
        active_slots = sorted({int(s) for s in slots_np if s >= 0})
        kwargs["steer_token_slots"] = token_slots
        kwargs["steer_active_slots"] = active_slots
        if manager is None:
            raise RuntimeError(
                "steering slots are routed but no worker manager was passed "
                "to make_steer_vector_forward_kwargs"
            )
        kwargs["steer_slot_positions"] = resolve_slot_positions(
            manager.slot_clauses(),
            active_slots,
            token_slots_np,
            token_slots.device,
            geo,
        )

        if trace.enabled():
            trace.begin_step(
                req_ids=input_batch.req_ids,
                slots=slots_np.tolist(),
                query_start_loc=input_batch.query_start_loc_np[: num_reqs + 1].tolist(),
                token_ids=geo.token_ids.cpu().tolist(),
                num_computed=geo.num_computed.tolist(),
                num_output=geo.num_output.tolist(),
            )
    return kwargs


def fill_graph_steer_buffers(
    input_batch: InputBatch,
    state: SteerVectorState | None,
    manager,
) -> None:
    """Fill Tier-1 persistent buffers for this step (full-graph mode).

    Writes each token's vector-table row into the shared row buffer and
    sets the per-layer trigger masks to 1 at steered positions. The
    captured kernel `hidden += mask * vectors[row_tok]` then applies the
    right configs without any per-step graph work; row 0 / mask 0 keep
    unsteered and padding tokens untouched. Positions come from the same
    resolver the layer hooks use (resolve_slot_positions).
    """
    from vllm.steer_vectors.algorithms.clause import clause_cache_key

    manager.zero_graph_masks()
    row_buf = manager.token_rows_buf
    row_buf.zero_()
    entries = manager.graph_batch_entries()
    if not entries or state is None:
        return

    slots_np, token_slots_np = _batch_token_slots(
        input_batch, state, manager.server_slot
    )
    rows_np = np.fromiter(
        (entries[s][0] if s in entries else 0 for s in slots_np),
        dtype=np.int64,
        count=slots_np.shape[0],
    )
    num_scheduled = input_batch.num_scheduled_tokens[: slots_np.shape[0]]
    token_rows_np = np.repeat(rows_np, num_scheduled)
    n = token_rows_np.shape[0]
    if n == 0:
        return
    device = row_buf.device
    row_buf[:n].copy_(torch.from_numpy(token_rows_np).to(device, non_blocking=True))

    # Batch geometry for the trigger collector: the same object the
    # forward context carries, from the single producer.
    geo = build_batch_geometry(input_batch)
    batch_slots = set(slots_np.tolist())
    active_slots = sorted(s for s in entries if s in batch_slots)
    resolved = resolve_slot_positions(
        manager.slot_clauses(), active_slots, token_slots_np, device, geo
    )
    from vllm.steer_vectors.algorithms import get_algorithm
    from vllm.steer_vectors.graph_kernels import graph_family_mask_attr

    # One scatter per (module, mask attr), not per slot: slots sharing a
    # layer contribute to the same mask write, so the launch count scales
    # with steered layers, not with live configurations.
    mask_writes: dict[tuple[int, str], tuple] = {}
    for slot in active_slots:
        _, request, controllers = entries[slot]
        positions = resolved[(slot, clause_cache_key(request.apply_spec))]
        if positions is None:
            continue
        # Families whose delta a zero table row cannot neutralize (e.g.
        # replace) carry their own mask; see GRAPH_FAMILY_MASKS.
        mask_attr = graph_family_mask_attr(
            get_algorithm(request.algorithm).graph_family
        )
        for module in controllers:
            mask_writes.setdefault(
                (id(module), mask_attr), (module, mask_attr, [])
            )[2].append(positions)
    for module, mask_attr, position_list in mask_writes.values():
        positions = (
            position_list[0]
            if len(position_list) == 1
            else torch.cat(position_list)
        )
        getattr(module, mask_attr)[positions] = 1.0
