# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steer vector support for the V2 GPU model runner (eager mode only)."""

import numpy as np
import torch

from vllm.steer_vectors import trace
from vllm.steer_vectors.request import SteerVectorRequest
from vllm.v1.worker.gpu.input_batch import InputBatch


class SteerVectorState:
    """Per-request steer vector bookkeeping for the V2 model runner.

    Each live request resolves to a config slot at admission time
    (payload loading + layer distribution happen there, never in the
    forward pass). Requests whose configs cannot be routed per-request
    yet (multi-vector, moe_router) fall back to the legacy
    globally-activated path.
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
        slot = None
        if manager is not None:
            slot = manager.acquire_config(req_id, steer_vector_request)
        if slot is not None:
            self._slots[req_id] = slot

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

    def legacy_requests(self) -> set[SteerVectorRequest]:
        """Requests using the legacy globally-activated path."""
        return {
            req
            for req_id, req in self._requests.items()
            if req_id not in self._slots
        }


def make_steer_vector_forward_kwargs(
    input_batch: InputBatch,
    state: SteerVectorState | None = None,
) -> dict:
    """Build the ForwardContext fields consumed by steering algorithms.

    All arrays are in batch order, matching query_start_loc boundaries:
    - current_tokens: flat token ids of the (unpadded) batch
    - num_computed_tokens_cpu: cached/computed tokens per request
    - num_output_tokens_cpu: tokens generated so far per request
    - query_start_loc: per-request token boundaries
    - steer_token_slots / steer_active_slots: per-request config routing
      (only when routed configs are live)
    """
    num_reqs = input_batch.num_reqs
    num_computed = input_batch.num_computed_tokens_np[:num_reqs]
    prefill_len = input_batch.prefill_len_np[:num_reqs]
    is_prefilling = input_batch.is_prefilling_np[:num_reqs]
    # While prefilling, nothing has been generated for this request yet.
    # During decode, the scheduler has computed prefill_len + (k - 1) tokens
    # when the k-th output token is being generated, matching the V1
    # semantics of len(output_token_ids) at execute time.
    num_output = np.where(
        is_prefilling, 0, num_computed - prefill_len + 1
    ).astype(np.int32)
    kwargs = {
        "current_tokens": input_batch.input_ids[: input_batch.num_tokens],
        "num_computed_tokens_cpu": torch.from_numpy(
            np.ascontiguousarray(num_computed)
        ),
        "num_output_tokens_cpu": torch.from_numpy(num_output),
        "query_start_loc": input_batch.query_start_loc[: num_reqs + 1],
    }

    if state is not None and state.has_routed():
        slots_np = np.fromiter(
            (state.slot_of(req_id) for req_id in input_batch.req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_scheduled = input_batch.num_scheduled_tokens[:num_reqs]
        token_slots_np = np.repeat(slots_np, num_scheduled)
        kwargs["steer_token_slots"] = torch.from_numpy(token_slots_np).to(
            input_batch.input_ids.device, non_blocking=True
        )
        kwargs["steer_active_slots"] = sorted(
            {int(s) for s in slots_np if s >= 0}
        )

        if trace.enabled():
            trace.begin_step(
                req_ids=input_batch.req_ids,
                slots=slots_np.tolist(),
                query_start_loc=input_batch.query_start_loc_np[
                    : num_reqs + 1
                ].tolist(),
                token_ids=kwargs["current_tokens"].cpu().tolist(),
                num_computed=num_computed.tolist(),
                num_output=num_output.tolist(),
            )
    return kwargs
