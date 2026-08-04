# SPDX-License-Identifier: Apache-2.0

"""Internal steering request structs for vLLM V1.

The user-facing API is `vllm.steer_vectors.api` (SteeringSpec /
VectorSpec / ApplySpec, see STEERING_API_V2.md); specs translate into
these msgspec structs at admission via `to_engine_request()`. The
where-clause travels as one canonical `apply_spec` dict (wire form of
ApplySpec).
"""

import msgspec

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

STEER_MOE_FIELDS: tuple[str, ...] = (
    "moe_expert_ids",
    "moe_mode",
    "moe_lambda",
    "moe_topk",
)


def steer_params_dict(obj, fields: tuple = STEER_APPLY_FIELDS) -> dict:
    """Extract canonical steering parameters from a schema object."""
    return {name: getattr(obj, name) for name in fields}


def build_server_request(steer_config) -> "SteerVectorRequest":
    """The canonical engine-default steering request for a config.

    Single source of truth for what engine-default steering does (used
    by the worker install path, the runtime update endpoint, and the
    prefix-cache salt). Name/id are fixed so every construction site
    produces an identical (same-fingerprint) request.
    """
    if steer_config.steering_config is None:
        raise ValueError(
            "build_server_request called without an engine-default "
            "steering config (steering_config is None)"
        )
    from vllm.steer_vectors.api import SteeringSpec, to_engine_request

    spec = SteeringSpec.model_validate_json(steer_config.steering_config)
    return to_engine_request(spec, name="__server__", int_id=1)


def is_prompt_length_sensitive(request) -> bool:
    """Whether the config's effect on a token depends on the request's
    prompt length (and not just the token's absolute position).

    True for negative positions (resolved from the end of the prompt)
    and generation windows (relative to the prompt end). Used by
    prefix-cache block hashing: such configs can only share KV blocks
    between requests with equal prompt lengths.
    """

    def _sensitive(obj) -> bool:
        spec = obj.apply_spec
        if spec is None:
            return False
        return (
            any(p < 0 for p in (spec.get("positions") or []))
            or any(p < 0 for p in (spec.get("exclude_positions") or []))
            or spec.get("window") is not None
        )

    if _sensitive(request):
        return True
    return any(_sensitive(vc) for vc in request.vector_configs or [])


def validate_apply_spec(spec: dict) -> None:
    """Structurally validate a wire-format `apply_spec` dict.

    Delegates to `SelectSpec.from_wire` — the single implementation of
    the clause rules — so authoring-side and engine-side validation
    cannot drift.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"apply_spec must be a dict, got {type(spec).__name__}")
    from vllm.steer_vectors.api import SelectSpec

    try:
        SelectSpec.from_wire(spec)
    except ValueError:
        raise
    except Exception as exc:  # pydantic ValidationError -> ValueError
        raise ValueError(f"invalid apply_spec: {exc}") from exc


def _validate_where_clause(obj, context: str) -> None:
    if obj.apply_spec is None:
        raise ValueError(
            f"{context} has no apply_spec and would steer no tokens; "
            "build requests through the v2 API (SteeringSpec/ApplySpec)"
        )
    validate_apply_spec(obj.apply_spec)


def _validate_inline_payload(obj, path: str, context: str) -> None:
    """Check the path/payload alternative on an engine struct."""
    if obj.inline_payload is None:
        if obj.payload_sha256 is not None:
            raise ValueError(
                f"{context} has payload_sha256 without inline_payload"
            )
        return
    if path:
        raise ValueError(
            f"{context} has both a file path and an inline payload; "
            "they are mutually exclusive"
        )
    from vllm.steer_vectors.payloads import validate_wire

    validate_wire(obj.inline_payload)
    if obj.payload_sha256 != obj.inline_payload.get("sha256"):
        raise ValueError(
            f"{context}: payload_sha256 does not match the inline payload"
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


class VectorConfig(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
    frozen=False,  # type: ignore[call-arg]
):  # type: ignore[call-arg]
    """One vector of a multi-vector steering request.

    Args:
        path: Local path to the vector file
        scale: Scale factor for this vector
        target_layers: Layer indices to apply to (None = file decides)
        apply_spec: Canonical where-clause dict (ApplySpec.to_wire())
        algorithm: Steering algorithm registry key
        normalize: Normalize the vector before applying
    """

    path: str
    scale: float = 1.0
    target_layers: list[int] | None = None
    apply_spec: dict | None = None
    algorithm: str = "direct"
    normalize: bool = False
    # In-memory payload (vllm.steer_vectors.payloads wire dict) as the
    # alternative to `path`; sha256 is its identity for slot dedup.
    inline_payload: dict | None = None
    payload_sha256: str | None = None


class SteerVectorRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
    frozen=False,  # type: ignore[call-arg]
):  # type: ignore[call-arg]
    """A steering configuration on the engine wire.

    Produced by `vllm.steer_vectors.api.to_engine_request`; not part of
    the public API. Single-vector mode uses the flat fields; multi-vector
    mode uses `vector_configs`.
    """

    steer_vector_name: str
    steer_vector_int_id: int
    steer_vector_local_path: str = ""
    debug: bool = False
    conflict_resolution: str = "priority"

    # === Single-vector mode ===
    scale: float = 1.0
    target_layers: list[int] | None = None
    apply_spec: dict | None = None
    algorithm: str = "direct"
    normalize: bool = False
    # In-memory payload (vllm.steer_vectors.payloads wire dict) as the
    # alternative to a file path; sha256 is its identity for slot dedup.
    inline_payload: dict | None = None
    payload_sha256: str | None = None

    # === Multi-vector mode ===
    vector_configs: list[VectorConfig] | None = None

    # === MoE-specific parameters (for moe_router algorithm) ===
    moe_expert_ids: list[int] | None = None
    moe_mode: str | None = (
        # Intervention mode override ('activate', 'deactivate', 'soft', ...); None = use
        # the config file's per-layer mode (falls back to 'activate')
        None
    )
    moe_lambda: float = (
        0.5  # Lambda parameter for 'soft' modes (z'_k = z_k + lambda * std(z))
    )
    moe_topk: int = (
        8  # Top-K parameter for 'soft_topk' mode (only intervene if expert is in top-k)
    )

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.steer_vector_int_id < 1:
            raise ValueError(
                f"steer_vector_int_id must be > 0, got {self.steer_vector_int_id}"
            )

        if self.conflict_resolution not in ["error", "priority", "sequential"]:
            raise ValueError(
                f"conflict_resolution must be 'error', 'priority', or 'sequential', "
                f"got '{self.conflict_resolution}'"
            )

        if self.is_multi_vector:
            if self.steer_vector_local_path:
                raise ValueError(
                    "Cannot specify both steer_vector_local_path and vector_configs"
                )
            if not self.vector_configs:
                raise ValueError("vector_configs cannot be empty in multi-vector mode")
            for i, vc in enumerate(self.vector_configs):
                context = f"vector_configs[{i}]"
                _validate_where_clause(vc, context)
                _validate_inline_payload(vc, vc.path, context)
                if not vc.path and vc.inline_payload is None:
                    raise ValueError(
                        f"{context} has neither a path nor an inline "
                        "payload"
                    )
        else:
            _validate_inline_payload(
                self, self.steer_vector_local_path, "Steering request"
            )
            # moe_router can work without a file path: it takes
            # expert_ids/mode from the request fields directly.
            if (
                self.algorithm != "moe_router"
                and not self.steer_vector_local_path
                and self.inline_payload is None
            ):
                raise ValueError(
                    "Single-vector mode requires steer_vector_local_path "
                    "or an inline payload (except for moe_router)"
                )
            if self.algorithm == "moe_router" and self.moe_mode is not None:
                from vllm.steer_vectors.algorithms.moe_router import (
                    MoERouterAlgorithm,
                )

                MoERouterAlgorithm.validate_mode(self.moe_mode)
            if self.algorithm == "moe_router" and not self.steer_vector_local_path:
                if not self.moe_expert_ids:
                    raise ValueError(
                        "moe_router without a config file requires moe_expert_ids"
                    )
                if not self.target_layers:
                    raise ValueError(
                        "moe_router without a config file requires target_layers "
                        "(the layers whose experts to steer); without it the "
                        "request would steer nothing"
                    )
            _validate_where_clause(self, "Steering request")

    @property
    def is_multi_vector(self) -> bool:
        """Check if this is a multi-vector request."""
        return self.vector_configs is not None

    @property
    def steer_vector_id(self) -> int:
        """Alias for steer_vector_int_id (backward compatibility)."""
        return self.steer_vector_int_id

    @property
    def local_path(self) -> str | None:
        """Get the local path for single-vector mode."""
        if self.is_multi_vector:
            return None  # Multi-vector mode doesn't have a single path
        return self.steer_vector_local_path

    @property
    def scale_factor(self) -> float:
        """Backward compatibility property."""
        if self.is_multi_vector:
            return 1.0  # Multi-vector mode uses individual scales
        return self.scale

    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare SteerVectorRequest
        instances based on steer_vector_name. This allows for identification
        and comparison of steering configurations across engines.
        """
        return (
            isinstance(value, self.__class__)
            and self.steer_vector_name == value.steer_vector_name
        )

    def __hash__(self) -> int:
        """
        Overrides the hash method to hash SteerVectorRequest instances
        based on steer_vector_name. This ensures that SteerVectorRequest instances
        can be used in hash-based collections such as sets and dictionaries,
        identified by their names across engines.
        """
        return hash(self.steer_vector_name)


_assert_schema_complete(
    VectorConfig, VectorConfig.__struct_fields__, STEER_APPLY_FIELDS
)
_assert_schema_complete(
    SteerVectorRequest,
    SteerVectorRequest.__struct_fields__,
    STEER_APPLY_FIELDS + STEER_MOE_FIELDS,
)
