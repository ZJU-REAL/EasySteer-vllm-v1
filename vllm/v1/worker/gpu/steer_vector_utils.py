# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steer vector support for the V2 GPU model runner."""

import numpy as np
import torch

from vllm.steer_vectors import trace
from vllm.steer_vectors.request import SteerVectorRequest, steer_params_dict
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
        self._requests[req_id] = steer_vector_request
        if manager is not None:
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


def make_steer_vector_forward_kwargs(
    input_batch: InputBatch,
    state: SteerVectorState | None = None,
    default_slot: int = -1,
) -> dict:
    """Build the ForwardContext fields consumed by steering algorithms.

    All arrays are in batch order, matching query_start_loc boundaries:
    - current_tokens: flat token ids of the (unpadded) batch
    - num_computed_tokens_cpu: cached/computed tokens per request
    - num_output_tokens_cpu: tokens generated so far per request
    - query_start_loc: per-request token boundaries
    - steer_token_slots / steer_active_slots: per-request config routing
      (only when routed configs are live)

    `default_slot` is the server-level config's slot (-1 when absent);
    requests without their own steering config are routed to it.
    """
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
    kwargs = {
        "current_tokens": input_batch.input_ids[: input_batch.num_tokens],
        "num_computed_tokens_cpu": torch.from_numpy(np.ascontiguousarray(num_computed)),
        "num_output_tokens_cpu": torch.from_numpy(num_output),
        "num_prompt_tokens_cpu": torch.from_numpy(np.ascontiguousarray(prefill_len)),
        "query_start_loc": input_batch.query_start_loc[: num_reqs + 1],
        # Capture labels each stored row with its owning request.
        "req_ids": list(input_batch.req_ids[:num_reqs]),
    }

    if state is not None and (state.has_routed() or default_slot >= 0):
        slots_np = np.fromiter(
            (
                slot if (slot := state.slot_of(req_id)) >= 0 else default_slot
                for req_id in input_batch.req_ids
            ),
            dtype=np.int32,
            count=num_reqs,
        )
        num_scheduled = input_batch.num_scheduled_tokens[:num_reqs]
        token_slots_np = np.repeat(slots_np, num_scheduled)
        kwargs["steer_token_slots"] = torch.from_numpy(token_slots_np).to(
            input_batch.input_ids.device, non_blocking=True
        )
        kwargs["steer_active_slots"] = sorted({int(s) for s in slots_np if s >= 0})

        if trace.enabled():
            trace.begin_step(
                req_ids=input_batch.req_ids,
                slots=slots_np.tolist(),
                query_start_loc=input_batch.query_start_loc_np[: num_reqs + 1].tolist(),
                token_ids=kwargs["current_tokens"].cpu().tolist(),
                num_computed=num_computed.tolist(),
                num_output=num_output.tolist(),
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
    unsteered and padding tokens untouched.
    """
    from vllm.steer_vectors.algorithms.triggers import (
        TriggerController,
    )

    manager.zero_graph_masks()
    row_buf = manager.row_tok_buf
    row_buf.zero_()
    entries = manager.graph_batch_entries()
    if not entries or state is None:
        return

    num_reqs = input_batch.num_reqs
    default_slot = manager.server_slot
    slots_np = np.fromiter(
        (
            slot if (slot := state.slot_of(req_id)) >= 0 else default_slot
            for req_id in input_batch.req_ids
        ),
        dtype=np.int64,
        count=num_reqs,
    )
    rows_np = np.fromiter(
        (entries[s][0] if s in entries else 0 for s in slots_np),
        dtype=np.int64,
        count=num_reqs,
    )
    num_scheduled = input_batch.num_scheduled_tokens[:num_reqs]
    token_rows_np = np.repeat(rows_np, num_scheduled)
    n = token_rows_np.shape[0]
    if n == 0:
        return
    device = row_buf.device
    row_buf[:n].copy_(torch.from_numpy(token_rows_np).to(device, non_blocking=True))
    token_slots = torch.from_numpy(np.repeat(slots_np, num_scheduled)).to(
        device, non_blocking=True
    )

    # Batch geometry for the trigger collector, from scheduler ground
    # truth (is_prefilling_np), matching extract_samples_info.
    current_tokens = input_batch.input_ids[: input_batch.num_tokens]
    num_computed_np = input_batch.num_computed_tokens_np[:num_reqs]
    prefill_len_np = input_batch.prefill_len_np[:num_reqs]
    is_prefilling_np = input_batch.is_prefilling_np[:num_reqs]
    num_output_np = np.where(
        is_prefilling_np, 0, num_computed_np - prefill_len_np + 1
    ).astype(np.int32)
    samples_info = {
        "query_start_loc": input_batch.query_start_loc[: num_reqs + 1],
        "num_computed": torch.from_numpy(np.ascontiguousarray(num_computed_np)).to(
            device, non_blocking=True
        ),
        "is_decode_mask": torch.from_numpy(np.ascontiguousarray(~is_prefilling_np)).to(
            device, non_blocking=True
        ),
        "num_output_tokens": torch.from_numpy(num_output_np).to(
            device, non_blocking=True
        ),
        "num_prompt_tokens": torch.from_numpy(np.ascontiguousarray(prefill_len_np)).to(
            device, non_blocking=True
        ),
    }

    batch_slots = set(slots_np.tolist())
    for slot, (row, request, controllers) in entries.items():
        if slot not in batch_slots:
            continue
        ctrl = TriggerController()
        ctrl.configure_from_dict(steer_params_dict(request))
        if ctrl.is_global_only_config():
            positions = (token_slots == slot).nonzero(as_tuple=False).squeeze(-1)
        else:
            # current_tokens doubles as the hidden_states arg: the
            # collector only takes the device from it.
            positions = ctrl.collect_intervention_positions(
                hidden_states=current_tokens,
                current_tokens=current_tokens,
                samples_info=samples_info,
            )
            if positions is None or positions.numel() == 0:
                continue
            positions = positions[token_slots[positions] == slot]
        if positions.numel() == 0:
            continue
        for module in controllers:
            module.graph_mask[positions] = 1.0
