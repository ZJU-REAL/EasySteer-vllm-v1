# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


def _executor_with_response_queues(response_mqs):
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda message: None)
    executor.response_mqs = response_mqs
    executor.futures_queue = deque()
    return executor


@pytest.mark.parametrize("failed_ranks", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize(
    "method",
    [
        "start_capture",
        "stop_capture",
        "fetch_captured",
        "clear_captured",
        "capture_status",
    ],
)
def test_failed_capture_control_drains_workers_before_next_rpc(method, failed_ranks):
    """A recoverable worker error must not become the next call's response."""
    success = WorkerProc.ResponseStatus.SUCCESS
    failure = WorkerProc.ResponseStatus.FAILURE
    queues = [
        deque(
            [(failure if rank in failed_ranks else success, "invalid"), (success, rank)]
        )
        for rank in range(2)
    ]
    executor = _executor_with_response_queues(
        [
            SimpleNamespace(dequeue=lambda timeout, queue=queue: queue.popleft())
            for queue in queues
        ]
    )
    with pytest.raises(RuntimeError, match="invalid"):
        executor.collective_rpc(method)
    assert executor.collective_rpc("stop_capture") == [0, 1]
    assert all(not queue for queue in queues)


@pytest.mark.parametrize(
    "method",
    [
        "compile_or_warm_up_model",
        "determine_available_memory",
        "execute_model",
        pytest.param(lambda worker: None, id="callable"),
    ],
)
def test_model_rpc_failure_does_not_wait_for_rank_blocked_in_collective(method):
    """Initialization failures must surface without waiting for other ranks."""

    def blocked_rank_reply(*, timeout):
        pytest.fail("Must not wait for a rank blocked in a model collective")

    executor = _executor_with_response_queues(
        [
            SimpleNamespace(
                dequeue=lambda timeout: (
                    WorkerProc.ResponseStatus.FAILURE,
                    "rank 0 failed",
                )
            ),
            SimpleNamespace(dequeue=blocked_rank_reply),
        ]
    )
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        executor.collective_rpc(method)
