# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steer vector support for the V2 GPU model runner (eager mode only)."""

import numpy as np
import torch

from vllm.steer_vectors.request import SteerVectorRequest
from vllm.v1.worker.gpu.input_batch import InputBatch


class SteerVectorState:
    """Per-request steer vector bookkeeping for the V2 model runner.

    V2 analogue of the V1 input-batch steer vector mapping: tracks which
    live request uses which SteerVectorRequest so the runner can hot-swap
    the active vector set each step.
    """

    def __init__(self) -> None:
        self._requests: dict[str, SteerVectorRequest] = {}

    def add_request(
        self,
        req_id: str,
        steer_vector_request: SteerVectorRequest | None,
    ) -> None:
        if steer_vector_request is not None:
            self._requests[req_id] = steer_vector_request

    def remove_request(self, req_id: str) -> None:
        self._requests.pop(req_id, None)

    def make_steer_vector_inputs(self) -> set[SteerVectorRequest]:
        """Set of steer vector requests for all live requests."""
        return set(self._requests.values())


def make_steer_vector_forward_kwargs(
    input_batch: InputBatch,
) -> dict[str, torch.Tensor]:
    """Build the ForwardContext fields consumed by steering algorithms.

    All arrays are in batch order, matching query_start_loc boundaries:
    - current_tokens: flat token ids of the (unpadded) batch
    - num_computed_tokens_cpu: cached/computed tokens per request
    - num_output_tokens_cpu: tokens generated so far per request
    - query_start_loc: per-request token boundaries
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
    return {
        "current_tokens": input_batch.input_ids[: input_batch.num_tokens],
        "num_computed_tokens_cpu": torch.from_numpy(
            np.ascontiguousarray(num_computed)
        ),
        "num_output_tokens_cpu": torch.from_numpy(num_output),
        "query_start_loc": input_batch.query_start_loc[: num_reqs + 1],
    }
