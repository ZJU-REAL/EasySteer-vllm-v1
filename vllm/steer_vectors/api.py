# SPDX-License-Identifier: Apache-2.0

"""User-facing steering API (v2).

Three concepts replace the v1 request surface:

- `ApplySpec`: where and when a vector applies (phases, token/position
  filters, generation window). Replaces the seven v1 trigger fields and
  the `-1` sentinel.
- `VectorSpec`: one vector — source, algorithm, scale, layers,
  normalize, algorithm-specific params, and its apply clause.
- `SteeringSpec`: an ordered list of vectors plus a conflict policy.
  Attached per request (`llm.generate(..., steering=spec)`, HTTP
  `"steering"`) or as the engine default (`--steering-config`).

Specs are backend-independent: eager, piecewise and full CUDA-graph
engines accept the same spec; a backend that cannot run a spec rejects
it at admission. `to_engine_request()` translates a spec into the
internal engine struct; the apply clause travels as a canonical
`apply_spec` dict executed natively by the trigger controller.

Semantics (differences from v1 are deliberate):

- Exclusions always subtract; nothing bypasses them.
- `generation_window=(start, stop)` is half-open over 0-based decode
  steps: `(0, k)` steers exactly the first k decode steps.
- Negative `positions` resolve from the end of the prompt (`-1` = last
  prompt token), stable across prefill chunks.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

Phase = Literal["prompt", "generation"]

PHASES = ("prompt", "generation")

# The canonical wire keys of an apply/select clause (`to_wire()` dict).
# Single source of truth — engine-side validators import this.
APPLY_SPEC_KEYS: tuple[str, ...] = (
    "phases",
    "tokens",
    "positions",
    "exclude_tokens",
    "exclude_positions",
    "window",
)

# Allowed `VectorSpec.params` keys per algorithm; algorithms not listed
# accept no params. (moe defaults match the v1 request defaults.)
ALGORITHM_PARAMS: dict[str, tuple[str, ...]] = {
    "moe_router": ("expert_ids", "mode", "lambda", "topk"),
}


def _require_nonempty(name: str, value: list | None) -> None:
    if value is not None and len(value) == 0:
        raise ValueError(f"{name} must be None or non-empty (None disables the filter)")


class SelectSpec(BaseModel):
    """The shared token-selection language (where-clause).

    One selection semantics used by both steering (`ApplySpec`, "which
    tokens to steer") and capture ("which rows to extract"): resolved
    identically by the trigger collector, so a clause means the same
    thing in both systems.

    Args:
        phases: Which token kinds to select ("prompt", "generation").
            Required and non-empty; with no other filters the clause
            selects every token of the listed phases.
        tokens: Token-id allowlist (real ids, >= 0).
        positions: Absolute sequence positions; negative values are
            Python-style from the end of the prompt.
        exclude_tokens: Token ids to never select.
        exclude_positions: Positions to never select (same convention).
        generation_window: Half-open (start, stop) over 0-based decode
            steps; stop=None means unbounded. Requires "generation" in
            phases.

    Within the selected phases, `tokens` and `positions` select the
    union of their matches; exclusions always subtract.
    """

    model_config = ConfigDict(extra="forbid")

    phases: list[Phase]
    tokens: list[int] | None = None
    positions: list[int] | None = None
    exclude_tokens: list[int] | None = None
    exclude_positions: list[int] | None = None
    generation_window: tuple[int, int | None] | None = None

    @model_validator(mode="after")
    def _validate(self) -> "SelectSpec":
        if not self.phases:
            raise ValueError("phases must be non-empty")
        if len(set(self.phases)) != len(self.phases):
            raise ValueError(f"phases has duplicates: {self.phases}")
        for name in ("tokens", "positions", "exclude_tokens", "exclude_positions"):
            _require_nonempty(name, getattr(self, name))
        for name in ("tokens", "exclude_tokens"):
            ids = getattr(self, name)
            if ids is not None and any(t < 0 for t in ids):
                raise ValueError(
                    f"{name} must contain real token ids (>= 0); the v1 "
                    "-1 sentinel is replaced by the phases field"
                )
        if self.generation_window is not None:
            start, stop = self.generation_window
            if "generation" not in self.phases:
                raise ValueError(
                    "generation_window requires 'generation' in phases"
                )
            if start < 0:
                raise ValueError(f"generation_window start must be >= 0, got {start}")
            if stop is not None and stop <= start:
                raise ValueError(
                    f"generation_window must be a half-open (start, stop) "
                    f"with stop > start or stop=None, got {self.generation_window}"
                )
        return self

    def to_wire(self) -> dict[str, Any]:
        """Canonical dict carried on the engine struct (`apply_spec`).

        Fixed key order and list-only containers so the config
        fingerprint and msgspec round-trips are deterministic.
        """
        window = None
        if self.generation_window is not None:
            window = [self.generation_window[0], self.generation_window[1]]
        return {
            "phases": [p for p in PHASES if p in self.phases],
            "tokens": self.tokens,
            "positions": self.positions,
            "exclude_tokens": self.exclude_tokens,
            "exclude_positions": self.exclude_positions,
            "window": window,
        }

    @classmethod
    def from_wire(cls, wire: dict[str, Any]) -> "SelectSpec":
        """Validate and rebuild a spec from its `to_wire()` dict.

        Used engine-side to reject malformed selection clauses at
        enable time instead of failing mid-forward.
        """
        unknown = set(wire) - set(APPLY_SPEC_KEYS)
        if unknown:
            raise ValueError(f"unknown selection fields: {sorted(unknown)}")
        window = wire.get("window")
        return cls(
            phases=wire.get("phases", []),
            tokens=wire.get("tokens"),
            positions=wire.get("positions"),
            exclude_tokens=wire.get("exclude_tokens"),
            exclude_positions=wire.get("exclude_positions"),
            generation_window=tuple(window) if window is not None else None,
        )


class ApplySpec(SelectSpec):
    """Where and when a steering vector applies.

    The selection language itself lives in `SelectSpec` (shared with
    capture); `ApplySpec` is the steering-facing name.
    """


class VectorSpec(BaseModel):
    """One steering vector and how it applies.

    Args:
        source: Path to a vector file in a format EasySteer itself
            defines (its GGUF export; moe_router JSON). For third-party
            checkpoint formats, load the file yourself (or use an
            `easysteer.vectors` adapter) and pass `data` instead.
        data: An in-memory payload (`vllm.steer_vectors.payloads`, or
            its wire dict) — the canonical way to steer with tensors
            you constructed or loaded yourself. Mutually exclusive with
            source.
        algorithm: Steering algorithm name (registry key).
        scale: Scale factor.
        layers: Layer indices to apply to; None lets the source decide.
        normalize: Normalize the vector before applying.
        apply: Where/when the vector applies (required).
        params: Algorithm-specific parameters; validated per algorithm,
            unknown keys are rejected.
        name: Optional label used in logs only (not identity).
    """

    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    data: Any = None
    algorithm: str = "direct"
    scale: float = 1.0
    layers: list[int] | None = None
    normalize: bool = False
    apply: ApplySpec
    params: dict[str, Any] = {}
    name: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "VectorSpec":
        _require_nonempty("layers", self.layers)
        if self.source is not None and "|" in self.source:
            raise ValueError(
                "source must be a plain path; set the algorithm via the "
                "algorithm field instead of embedding it as 'path|algo'"
            )
        allowed = ALGORITHM_PARAMS.get(self.algorithm, ())
        unknown = set(self.params) - set(allowed)
        if unknown:
            raise ValueError(
                f"unknown params for algorithm '{self.algorithm}': "
                f"{sorted(unknown)} (allowed: {sorted(allowed) or 'none'})"
            )
        if self.algorithm == "moe_router" and "mode" in self.params:
            from vllm.steer_vectors.algorithms.moe_router import (
                MoERouterAlgorithm,
            )

            MoERouterAlgorithm.validate_mode(self.params["mode"])
        if self.data is not None:
            from vllm.steer_vectors.payloads import (
                ALGORITHM_PAYLOADS,
                Payload,
                is_broadcast_kind,
                validate_wire,
            )

            if self.source is not None:
                raise ValueError("source and data are mutually exclusive")
            expected = ALGORITHM_PAYLOADS.get(self.algorithm)
            if expected is None:
                raise ValueError(
                    f"algorithm {self.algorithm!r} does not accept "
                    "in-memory payloads"
                )
            if isinstance(self.data, Payload):
                kind = self.data.kind
            elif isinstance(self.data, dict):
                kind = validate_wire(self.data)
            else:
                raise ValueError(
                    "data must be a vllm.steer_vectors.payloads object "
                    f"or its wire dict, got {type(self.data).__name__}"
                )
            if kind != expected:
                raise ValueError(
                    f"algorithm {self.algorithm!r} requires a "
                    f"{expected!r} payload, got {kind!r}"
                )
            if is_broadcast_kind(kind) and not self.layers:
                raise ValueError(
                    f"a {kind!r} payload holds one map applied to each "
                    "target layer; layers is required"
                )
        if self.algorithm == "moe_router" and self.source is None:
            if not self.params.get("expert_ids"):
                raise ValueError(
                    "moe_router without a source file requires "
                    "params['expert_ids']"
                )
            if not self.layers:
                raise ValueError(
                    "moe_router without a source file requires layers "
                    "(the layers whose experts to steer)"
                )
        if (
            self.algorithm != "moe_router"
            and self.source is None
            and self.data is None
        ):
            raise ValueError(
                f"algorithm '{self.algorithm}' requires either a source "
                "file or an in-memory data payload"
            )
        # Reject wrong source formats at admission: a load failure inside
        # the worker would take down the EngineCore.
        if self.source is not None:
            if self.algorithm in ("linear", "lm_steer", "loreft"):
                raise ValueError(
                    f"algorithm {self.algorithm!r} loads no source files; "
                    "load the checkpoint yourself (see easysteer.vectors "
                    "adapters) and pass data=..."
                )
            gguf_only = ("direct", "erase", "replace")
            if self.algorithm in gguf_only and not self.source.lower().endswith(
                ".gguf"
            ):
                raise ValueError(
                    f"algorithm {self.algorithm!r} accepts only .gguf "
                    f"sources (EasySteer's export format), got "
                    f"{self.source!r}; for other formats pass data=..."
                )
        return self


class SteeringSpec(BaseModel):
    """A complete steering configuration: ordered vectors + conflict policy.

    Args:
        vectors: The vectors to apply (non-empty; order matters for
            'sequential'/'priority' conflict resolution).
        conflict: What to do when several vectors target one position:
            'priority' (first wins), 'sequential' (stack), 'error'.
        debug: Verbose logging during the forward pass.
    """

    model_config = ConfigDict(extra="forbid")

    vectors: list[VectorSpec]
    conflict: Literal["priority", "sequential", "error"] = "priority"
    debug: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "SteeringSpec":
        if not self.vectors:
            raise ValueError("vectors must be non-empty")
        if len(self.vectors) > 1 and any(
            v.algorithm == "moe_router" for v in self.vectors
        ):
            raise ValueError(
                "moe_router is not supported in multi-vector specs yet; "
                "use a single-vector spec"
            )
        return self


def to_engine_request(
    spec: SteeringSpec,
    *,
    name: str | None = None,
    int_id: int | None = None,
):
    """Translate a v2 SteeringSpec into the internal engine struct.

    `name`/`int_id` exist for deterministic internal identities (the
    server default uses "__server__"/1); normally both are generated.
    """
    from vllm.steer_vectors.request import (
        SteerVectorRequest,
        VectorConfig,
        _next_steer_vector_id,
    )

    if name is None:
        from vllm.utils import random_uuid

        name = spec.vectors[0].name or f"steering-{random_uuid()[:8]}"
    if int_id is None:
        int_id = _next_steer_vector_id()

    def _payload_wire(v: VectorSpec) -> dict[str, Any] | None:
        if v.data is None:
            return None
        return v.data if isinstance(v.data, dict) else v.data.to_wire()

    if len(spec.vectors) == 1:
        v = spec.vectors[0]
        moe: dict[str, Any] = {}
        if v.algorithm == "moe_router":
            moe = {
                "moe_expert_ids": v.params.get("expert_ids"),
                "moe_mode": v.params.get("mode"),
                "moe_lambda": v.params.get("lambda", 0.5),
                "moe_topk": v.params.get("topk", 8),
            }
        wire = _payload_wire(v)
        return SteerVectorRequest(
            steer_vector_name=name,
            steer_vector_int_id=int_id,
            steer_vector_local_path=v.source or "",
            inline_payload=wire,
            payload_sha256=wire["sha256"] if wire else None,
            debug=spec.debug,
            scale=v.scale,
            target_layers=v.layers,
            algorithm=v.algorithm,
            normalize=v.normalize,
            apply_spec=v.apply.to_wire(),
            **moe,
        )

    def _vector_config(v: VectorSpec) -> "VectorConfig":
        wire = _payload_wire(v)
        return VectorConfig(
            path=v.source or "",
            inline_payload=wire,
            payload_sha256=wire["sha256"] if wire else None,
            scale=v.scale,
            target_layers=v.layers,
            algorithm=v.algorithm,
            normalize=v.normalize,
            apply_spec=v.apply.to_wire(),
        )

    return SteerVectorRequest(
        steer_vector_name=name,
        steer_vector_int_id=int_id,
        debug=spec.debug,
        conflict_resolution=spec.conflict,
        vector_configs=[_vector_config(v) for v in spec.vectors],
    )
