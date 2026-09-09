# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Internal steering request structs for vLLM V1.

The user-facing API is `vllm.steer_vectors` (SteeringSpec /
VectorSpec / ApplySpec); specs translate into
these msgspec structs at admission via `to_engine_request()`. The
where-clause travels as one canonical `apply_spec` dict (wire form of
ApplySpec).
"""

import msgspec

from vllm.logger import init_logger

logger = init_logger(__name__)

# --- Canonical steering parameter schema ---
#
# Single source of truth for "what configures how a vector is applied".
# The engine structs below must declare these fields (enforced by
# _assert_schema_complete at import time), and every conversion/copy
# site iterates these tuples instead of spelling the fields out.

STEER_CLAUSE_FIELDS: tuple[str, ...] = ("apply_spec",)

STEER_APPLY_FIELDS: tuple[str, ...] = (
    "scale",
    "target_layers",
    *STEER_CLAUSE_FIELDS,
    "algorithm",
    "normalize",
)


def steer_params_dict(obj, fields: tuple = STEER_APPLY_FIELDS) -> dict:
    """Extract canonical steering parameters from a schema object."""
    return {name: getattr(obj, name) for name in fields}


def is_prompt_length_sensitive(request, prompt_len: int | None = None) -> bool:
    """Whether the config's effect on a token depends on the request's
    prompt length (and not just the token's absolute position).

    True for negative prompt positions (resolve from the prompt end),
    prompt windows with an end-relative bound (negative, or stop=None),
    and generation-step selectors. Positive prompt positions are
    absolute — length-sensitive only when they reach past this
    request's prompt end and clamp to its last token; pass `prompt_len`
    for that precise check (without it, any positive entry counts as
    potentially clamping). Used by prefix-cache block hashing: sensitive
    configs can only share KV blocks between requests with equal prompt
    lengths.
    """

    def _positions_sensitive(values) -> bool:
        if not values:
            return False
        if any(p < 0 for p in values):
            return True
        if prompt_len is None:
            return True
        return max(values) >= prompt_len

    def _end_relative_window(window) -> bool:
        if window is None:
            return False
        start, stop = window
        return start < 0 or stop is None or stop < 0

    def _sensitive(obj) -> bool:
        spec = obj.apply_spec
        if spec is None:
            return False
        return (
            _positions_sensitive(spec.get("prompt_positions"))
            or _positions_sensitive(spec.get("exclude_prompt_positions"))
            or _end_relative_window(spec.get("prompt_window"))
            or _end_relative_window(spec.get("exclude_prompt_window"))
            or spec.get("generation_positions") is not None
            or spec.get("exclude_generation_positions") is not None
            or spec.get("generation_window") is not None
            or spec.get("exclude_generation_window") is not None
        )

    return any(_sensitive(vector) for vector in request.vectors)


def warn_clamped_prompt_positions(request, prompt_len: int, request_id: str) -> None:
    """Warn when positive prompt_positions lie past the prompt end.

    The clause matchers clamp such entries to the last prompt token;
    admission is the one place the prompt length is cheaply known, so
    the warning is emitted here instead of from the per-step hot path.
    """

    def _check(obj) -> None:
        spec = obj.apply_spec
        if spec is None:
            return
        for key in ("prompt_positions", "exclude_prompt_positions"):
            over = [p for p in (spec.get(key) or []) if p >= prompt_len]
            if over:
                logger.warning(
                    "Request %s: %s entries %s are past the prompt end "
                    "(prompt length %d); clamping to the last prompt "
                    "token.",
                    request_id,
                    key,
                    over,
                    prompt_len,
                )

    for vector in request.vectors:
        _check(vector)


def validate_apply_spec(spec: dict) -> None:
    """Structurally validate a wire-format `apply_spec` dict.

    Delegates to `SelectSpec.from_wire` — the single implementation of
    the clause rules — so authoring-side and engine-side validation
    cannot drift.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"apply_spec must be a dict, got {type(spec).__name__}")
    from vllm.model_hooks.steering.api import SelectSpec

    SelectSpec.from_wire(spec)


def _validate_vector(vector) -> None:
    from vllm.model_hooks.steering.capabilities import (
        ALGORITHM_CAPABILITIES,
        validate_normalize,
    )
    from vllm.model_hooks.steering.payloads import validate_wire

    validate_normalize(vector.algorithm, vector.normalize)
    capability = ALGORITHM_CAPABILITIES.get(vector.algorithm)
    if capability is None:
        raise ValueError(f"Unknown steering algorithm: {vector.algorithm!r}")
    if vector.apply_spec is None:
        raise ValueError(
            "ResolvedVector has no apply_spec and would steer no tokens; "
            "build requests through SteeringSpec/ApplySpec"
        )
    validate_apply_spec(vector.apply_spec)
    kind = validate_wire(vector.payload)
    if kind != capability.payload_kind:
        raise ValueError(
            f"algorithm {vector.algorithm!r} requires a {capability.payload_kind!r} "
            f"payload, got {kind!r}"
        )


def _assert_schema_complete(cls, field_names, required) -> None:
    missing = set(required) - set(field_names)
    assert not missing, (
        f"{cls.__name__} is missing canonical steering fields: {sorted(missing)}"
    )


_steer_vector_id_counter = 0


def _next_steer_vector_id() -> int:
    """Generate a unique positive integer ID for steer vectors."""
    global _steer_vector_id_counter
    _steer_vector_id_counter += 1
    if _steer_vector_id_counter > 2147483647:
        _steer_vector_id_counter = 1
    return _steer_vector_id_counter


# --- Engine-level request types (msgspec) ---


class ResolvedVector(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
    frozen=False,  # type: ignore[call-arg]
):  # type: ignore[call-arg]
    """One admitted intervention with a canonical content snapshot.

    Source is provenance only. Workers consume payload and application fields.
    """

    payload: dict
    source: str = ""
    scale: float = 1.0
    target_layers: list[int] | None = None
    apply_spec: dict | None = None
    algorithm: str = "direct"
    normalize: bool = False

    def __post_init__(self) -> None:
        _validate_vector(self)

    @property
    def payload_sha256(self) -> str:
        """Content identity, stored once in the payload snapshot."""
        return self.payload["sha256"]


class SteeringRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
    frozen=False,  # type: ignore[call-arg]
):  # type: ignore[call-arg]
    """An ordered intervention list on the engine wire.

    Produced by ``to_engine_request``; labels support reporting while payload
    content and application fields determine execution identity.
    """

    steer_vector_name: str
    steer_vector_int_id: int
    vectors: list[ResolvedVector]
    conflict_resolution: str = "priority"

    def __post_init__(self) -> None:
        if self.steer_vector_int_id < 1:
            raise ValueError(
                f"steer_vector_int_id must be > 0, got {self.steer_vector_int_id}"
            )
        if not self.vectors:
            raise ValueError("vectors cannot be empty")
        if self.conflict_resolution not in ("error", "priority", "sequential"):
            raise ValueError(
                "conflict_resolution must be 'error', 'priority', or 'sequential', "
                f"got {self.conflict_resolution!r}"
            )


_assert_schema_complete(
    ResolvedVector, ResolvedVector.__struct_fields__, STEER_APPLY_FIELDS
)


def config_fingerprint(request: SteeringRequest) -> str:
    """Stable identity of a steering *configuration* (not just the vector).

    Requests with the same fingerprint share one layer slot; the vector
    payload itself is deduplicated separately by the PayloadCache. Built
    from the canonical field registry so new parameters participate
    automatically.
    """

    def _canon_value(value):
        if isinstance(value, list):
            return tuple(_canon_value(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((k, _canon_value(v)) for k, v in value.items()))
        return value

    def _canon(obj, fields):
        return [_canon_value(getattr(obj, name)) for name in fields]

    values = [request.conflict_resolution if len(request.vectors) > 1 else None]
    for vector in request.vectors:
        values.append(vector.payload_sha256)
        values.extend(_canon(vector, STEER_APPLY_FIELDS))
    return repr(tuple(values))
