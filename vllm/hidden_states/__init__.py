# SPDX-License-Identifier: Apache-2.0
"""
Capture of intermediate model state for vLLM.

Hook-based capture of decoder-layer hidden states and MoE router logits
(see vllm.hidden_states.capture.CaptureSession). Worker-side RPCs live
in vllm.v1.worker.capture_model_runner_mixin.
"""

from vllm.hidden_states.capture import (
    HIDDEN_STATES,
    ROUTER_LOGITS,
    CaptureSession,
    StreamConfig,
)
from vllm.hidden_states.request import (
    HiddenStatesCaptureRequest,
    MoERouterLogitsCaptureRequest,
)
from vllm.hidden_states.utils import (
    deserialize_hidden_states,
    deserialize_moe_router_logits,
    print_hidden_states_summary,
    print_moe_router_logits_summary,
)

__all__ = [
    "HIDDEN_STATES",
    "ROUTER_LOGITS",
    "CaptureSession",
    "StreamConfig",
    "HiddenStatesCaptureRequest",
    "MoERouterLogitsCaptureRequest",
    "deserialize_hidden_states",
    "print_hidden_states_summary",
    "deserialize_moe_router_logits",
    "print_moe_router_logits_summary",
]

__version__ = "2.0.0"
