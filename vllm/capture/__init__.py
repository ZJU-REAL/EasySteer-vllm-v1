# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public capture selection and labelled result API."""

from importlib import import_module

_EXPORTS = {
    "SelectSpec": "vllm.model_hooks.selection.spec",
    "HIDDEN_STATES": "vllm.model_hooks.components.registry",
    "ROUTER_LOGITS": "vllm.model_hooks.components.registry",
    "CaptureMeta": "vllm.model_hooks.capture.serialization",
    "deserialize_captured": "vllm.model_hooks.capture.serialization",
    "match_capture_request_id": "vllm.model_hooks.capture.serialization",
    "CaptureSession": "vllm.model_hooks.capture.session",
    "StreamConfig": "vllm.model_hooks.capture.store",
    "StreamStore": "vllm.model_hooks.capture.store",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
