# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared authoring and preload validation without reading source files."""

from typing import Any

from vllm.model_hooks.steering.capabilities import (
    ALGORITHM_CAPABILITIES,
    ROUTER_OVERRIDE_PARAMS,
    validate_normalize,
)


def validate_params(params, allowed: tuple[str, ...], context: str) -> dict:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError("steering parameters must be a dictionary")
    if not all(isinstance(key, str) for key in params):
        raise ValueError("steering parameter names must be strings")
    if unknown := set(params) - set(allowed):
        raise ValueError(
            f"unknown params for {context}: {sorted(unknown)} "
            f"(allowed: {sorted(allowed) or 'none'})"
        )
    return params


def validate_vector_input(
    source: str | None,
    data: Any,
    algorithm: str,
    layers: list[int] | None,
    params: dict | None,
    *,
    normalize: bool = False,
) -> dict:
    """Validate one input and return its algorithm parameters."""
    capability = ALGORITHM_CAPABILITIES.get(algorithm)
    if capability is None:
        raise ValueError(f"Unknown steering algorithm: {algorithm!r}")
    validate_normalize(algorithm, normalize)
    if layers is not None and not layers:
        raise ValueError("layers must be None or non-empty (None disables the filter)")
    if source is not None:
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty path string")
        if "|" in source:
            raise ValueError("Use a plain path and the separate algorithm field")
        if data is not None:
            raise ValueError("source and data are mutually exclusive")

    allowed = capability.params
    if algorithm == "moe_router" and (source is not None or data is not None):
        allowed = ROUTER_OVERRIDE_PARAMS
    params = validate_params(params, allowed, f"algorithm {algorithm!r}")
    if algorithm == "moe_router" and "mode" in params:
        from vllm.model_hooks.steering.payloads import validate_router_mode

        validate_router_mode(params["mode"])

    if data is not None:
        from vllm.model_hooks.steering.payloads import (
            Payload,
            is_broadcast_kind,
            validate_wire,
        )

        if isinstance(data, Payload):
            kind = data.kind
        elif isinstance(data, dict):
            kind = validate_wire(data)
        else:
            raise ValueError("data must be a Payload or its wire dictionary")
        if kind != capability.payload_kind:
            raise ValueError(
                f"algorithm {algorithm!r} requires a {capability.payload_kind!r} "
                f"payload, got {kind!r}"
            )
        if is_broadcast_kind(kind) and not layers:
            raise ValueError(
                f"a {kind!r} payload holds one map applied to each "
                "target layer; layers is required"
            )
    elif source is not None:
        if capability.source == "none":
            raise ValueError(
                f"algorithm {algorithm!r} loads no source files; use an explicit "
                "easysteer.vectors adapter and VectorSpec(data=...)"
            )
        if capability.source == "gguf" and not source.lower().endswith(".gguf"):
            raise ValueError(
                f"algorithm {algorithm!r} only loads .gguf sources; "
                "use VectorSpec(data=...) for other formats"
            )
    elif algorithm == "moe_router":
        if not params.get("expert_ids"):
            raise ValueError(
                "moe_router without a source or data requires params['expert_ids']"
            )
        if not layers:
            raise ValueError("inline moe_router parameters require layers")
    else:
        raise ValueError(
            f"algorithm {algorithm!r} requires either a source file "
            "or an in-memory data payload"
        )
    return params
