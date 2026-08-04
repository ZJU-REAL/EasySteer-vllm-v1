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


def resolve_slot_positions(
    slot_clauses: dict[int, list[dict | None]],
    active_slots: list[int],
    token_slots_np: np.ndarray,
    token_slots: torch.Tensor,
    geo,
) -> dict[tuple, torch.Tensor | None]:
    """Resolve every active clause's steered positions, once per step.

    Where-clauses are layer-invariant, so this single resolution serves
    every decoder/MoE-gate hook (and the Tier-1 mask filler). Keys are
    (slot, clause_cache_key); a None value means the clause matched no
    token this step. Global-only clauses resolve on the CPU (no device
    sync); others run the trigger collector once on the GPU.
    """
    from vllm.steer_vectors.algorithms.triggers import (
        clause_cache_key,
        collect_positions_apply_spec,
        is_global_only_spec,
    )

    resolved: dict[tuple, torch.Tensor | None] = {}
    device = token_slots.device
    samples_info = None
    for slot in active_slots:
        for clause in slot_clauses.get(slot, []):
            key = clause_cache_key(clause)
            if key is None or (slot, key) in resolved:
                continue
            if is_global_only_spec(clause):
                pos_np = np.nonzero(token_slots_np == slot)[0]
                positions = (
                    torch.from_numpy(pos_np).to(device, non_blocking=True)
                    if pos_np.size
                    else None
                )
            else:
                if samples_info is None:
                    samples_info = geo.samples_info()
                positions = collect_positions_apply_spec(
                    current_tokens=geo.token_ids,
                    samples_info=samples_info,
                    spec=clause,
                )
                if positions is not None:
                    positions = positions[token_slots[positions] == slot]
                    if positions.numel() == 0:
                        positions = None
            resolved[(slot, key)] = positions
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
            manager.slot_clauses(), active_slots, token_slots_np, token_slots, geo
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
    from vllm.steer_vectors.algorithms.triggers import clause_cache_key

    manager.zero_graph_masks()
    row_buf = manager.row_tok_buf
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
    token_slots = torch.from_numpy(token_slots_np).to(device, non_blocking=True)

    # Batch geometry for the trigger collector: the same object the
    # forward context carries, from the single producer.
    geo = build_batch_geometry(input_batch)
    batch_slots = set(slots_np.tolist())
    active_slots = sorted(s for s in entries if s in batch_slots)
    resolved = resolve_slot_positions(
        manager.slot_clauses(), active_slots, token_slots_np, token_slots, geo
    )
    for slot in active_slots:
        _, request, controllers = entries[slot]
        positions = resolved[(slot, clause_cache_key(request.apply_spec))]
        if positions is None:
            continue
        for module in controllers:
            module.graph_mask[positions] = 1.0
