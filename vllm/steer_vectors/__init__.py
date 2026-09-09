# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public steering specifications and payload types."""

from importlib import import_module

_EXPORTS = {
    "ApplySpec": "api",
    "SelectSpec": "api",
    "VectorSpec": "api",
    "SteeringSpec": "api",
    "to_engine_request": "api",
    "Payload": "payloads",
    "DirectionVector": "payloads",
    "LinearMap": "payloads",
    "LowRankProjector": "payloads",
    "ReftIntervention": "payloads",
    "ConceptPair": "payloads",
    "RouterConfig": "payloads",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"vllm.model_hooks.steering.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
