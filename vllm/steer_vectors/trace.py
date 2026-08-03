# SPDX-License-Identifier: Apache-2.0
"""Steering trace: records where steering was actually applied.

Enabled by setting the VLLM_STEER_TRACE_DIR environment variable to a
directory. Each worker process appends JSONL records:

- {"type": "step", "step": N, "req_ids": [...], "slots": [...],
   "query_start_loc": [...], "token_ids": [...],
   "num_computed": [...], "num_output": [...]}
  emitted once per engine step by the model runner (batch composition).

- {"type": "apply", "step": N, "layer": L, "slot": S, "algo": name,
   "positions": [...]}
  emitted by each steered layer with the exact flat token positions that
  were transformed for that slot.

Joining the two record types gives, for every step: which requests were
in the batch, their token ranges and token ids, and exactly which
positions each config steered at each layer. Used by the Phase C trigger
validation tests; also useful for debugging steering behavior. Tracing
is off unless the env var is set; overhead (GPU syncs) only occurs when
enabled.
"""

import json
import os
import threading

_lock = threading.Lock()
_file = None
_step = 0


def enabled() -> bool:
    return bool(os.environ.get("VLLM_STEER_TRACE_DIR"))


def _get_file():
    global _file
    if _file is None:
        trace_dir = os.environ["VLLM_STEER_TRACE_DIR"]
        os.makedirs(trace_dir, exist_ok=True)
        _file = open(
            os.path.join(trace_dir, f"steer_trace_{os.getpid()}.jsonl"),
            "a",
            buffering=1,
        )
    return _file


def _emit(record: dict) -> None:
    with _lock:
        _get_file().write(json.dumps(record) + "\n")


def begin_step(
    req_ids,
    slots,
    query_start_loc,
    token_ids,
    num_computed,
    num_output,
) -> None:
    """Record batch composition for one engine step (runner side)."""
    global _step
    _step += 1
    _emit(
        {
            "type": "step",
            "step": _step,
            "req_ids": list(req_ids),
            "slots": [int(s) for s in slots],
            "query_start_loc": [int(x) for x in query_start_loc],
            "token_ids": [int(t) for t in token_ids],
            "num_computed": [int(x) for x in num_computed],
            "num_output": [int(x) for x in num_output],
        }
    )


def record_apply(layer_id, slot, algo, positions) -> None:
    """Record the positions a slot's algorithm transformed at a layer."""
    _emit(
        {
            "type": "apply",
            "step": _step,
            "layer": -1 if layer_id is None else int(layer_id),
            "slot": int(slot),
            "algo": algo,
            "positions": [int(p) for p in positions],
        }
    )
