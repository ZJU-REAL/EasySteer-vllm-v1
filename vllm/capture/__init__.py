# SPDX-License-Identifier: Apache-2.0
"""
Capture of intermediate model state for vLLM.

Hook-based capture of decoder-layer hidden states and MoE router logits
(see vllm.capture.session.CaptureSession). Worker-side RPCs live in
vllm.v1.worker.capture_model_runner_mixin.
"""

from vllm.capture.serde import (
    CaptureMeta,
    deserialize_captured,
    match_capture_request_id,
)
from vllm.capture.session import (
    HIDDEN_STATES,
    ROUTER_LOGITS,
    CaptureSession,
)
from vllm.capture.store import StreamConfig, StreamStore

__all__ = [
    "HIDDEN_STATES",
    "ROUTER_LOGITS",
    "CaptureSession",
    "StreamConfig",
    "StreamStore",
    "CaptureMeta",
    "deserialize_captured",
    "match_capture_request_id",
]

__version__ = "2.0.0"
