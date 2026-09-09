# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt V2 batches and CUDA graph execution to model hooks."""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_hooks.capture.graph import CaptureGraphState
from vllm.model_hooks.selection.batch import BatchGeometry
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.capture_model_runner_mixin import CaptureModelRunnerMixin
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def release_capture_graph(runner: "CaptureModelRunnerMixin") -> None:
    """Release the native graph before its fixed output buffers."""
    runner.capture_graph_manager = None
    runner.capture_session.release_graph()


def prepare_capture_graph(
    runner: "GPUModelRunner",
    num_reqs: int,
    num_tokens: int,
    uniform_tok_count: int | None,
    num_active_loras: int,
    max_query_len: int,
) -> "ModelCudaGraphManager | None":
    """Select and warm up a capture variant before staging real batch inputs."""
    idle = runner.cudagraph_manager
    steer = runner.vllm_config.steer_vector_config
    if not (
        idle is not None
        and idle.cudagraph_mode.has_full_cudagraphs()
        and runner.parallel_config.world_size_across_dp == 1
        and runner.speculative_config is None
        and runner.lora_config is None
        and (steer is None or steer.graph_mode == "in_graph")
    ):
        return None
    candidate = idle.dispatch(
        num_reqs, num_tokens, uniform_tok_count, num_active_loras, max_query_len
    )
    if candidate.cg_mode != CUDAGraphMode.FULL:
        return None

    session = runner.capture_session
    signature = session.graph_signature()
    state = session.graph_state
    if state is not None and state.signature != signature:
        release_capture_graph(runner)
        state = None
    if state is None:
        state = CaptureGraphState(signature)
        session.graph_state = state
        initialized = False
        try:
            runner.capture_graph_manager = _initialize_capture_graph(runner, state)
            initialized = True
        finally:
            if not initialized:
                release_capture_graph(runner)
    assert runner.capture_graph_manager is not None
    return runner.capture_graph_manager


def _initialize_capture_graph(
    runner: "GPUModelRunner", state: CaptureGraphState
) -> "ModelCudaGraphManager":
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    idle = runner.cudagraph_manager
    assert idle is not None
    mode = (
        CUDAGraphMode.FULL
        if idle.cudagraph_mode == CUDAGraphMode.FULL
        else CUDAGraphMode.FULL_DECODE_ONLY
    )
    manager = ModelCudaGraphManager(
        runner.vllm_config, runner.device, mode, runner.decode_query_len
    )
    # Capture outputs must never overlap the ordinary graph pool.
    manager.pool = current_platform.graph_pool_handle()
    allocated_before = torch.accelerator.memory_allocated(runner.device)
    with state.record_outputs(runner.model):
        manager.capture(
            runner.model,
            runner.model_state,
            runner.input_buffers,
            runner.intermediate_tensors,
            runner.block_tables,
            runner.attn_groups,
            runner.kv_cache_config,
            skip_compiled=True,
            progress_bar_desc="Capturing activation extraction graphs",
        )
    state.allocation_bytes = max(
        0, torch.accelerator.memory_allocated(runner.device) - allocated_before
    )
    return manager


def build_batch_geometry(
    input_batch: InputBatch,
    prompt_lengths: np.ndarray,
    *,
    query_start_loc_cpu: np.ndarray | None = None,
) -> "BatchGeometry":
    """Build the per-step BatchGeometry from the runner's InputBatch.

    The single producer of batch geometry: steering triggers, capture
    row selection/labels, and the full-graph buffer filler all consume
    this object.
    """
    num_reqs = input_batch.num_reqs
    num_computed = input_batch.num_computed_tokens_np[:num_reqs]
    prompt_lengths = prompt_lengths[:num_reqs]
    is_prefilling = input_batch.is_prefilling_np[:num_reqs]
    # Re-prefill can include previously generated tokens. Keep its execution
    # phase separate from the original prompt boundary used by selectors.
    num_output = np.where(is_prefilling, 0, num_computed - prompt_lengths + 1).astype(
        np.int32
    )
    return BatchGeometry(
        query_start_loc=input_batch.query_start_loc[: num_reqs + 1],
        num_computed=torch.from_numpy(np.ascontiguousarray(num_computed)),
        num_prompt=torch.from_numpy(np.ascontiguousarray(prompt_lengths)),
        num_output=torch.from_numpy(num_output),
        req_ids=list(input_batch.req_ids[:num_reqs]),
        token_ids=input_batch.input_ids[: input_batch.num_tokens],
        query_start_loc_cpu=(
            input_batch.query_start_loc_np[: num_reqs + 1]
            if query_start_loc_cpu is None
            else query_start_loc_cpu[: num_reqs + 1]
        ),
    )
