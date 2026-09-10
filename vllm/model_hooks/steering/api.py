# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""User-facing steering API.

Three authoring concepts:

- `ApplySpec`: where and when a vector applies (per-phase "all" and
  token/position/window selectors).
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

Selection semantics:

- Each phase is selected independently: `prompt="all"` /
  `generation="all"` select the whole phase, and each phase's include
  selectors select the union of their matches. A phase with neither
  "all" nor a selector is untouched — there is no separate phase gate.
- Every include selector has an exclude twin; exclusions union and
  always subtract. When an include and an exclude overlap, the
  exclusion wins.
- Windows are half-open `(start, stop)`: `generation_window=(0, k)`
  selects exactly the first k decode steps; `prompt_window` bounds may
  be negative (resolved from the end of the prompt).
- Every selector is phase-scoped and named for it: `prompt_tokens` /
  `prompt_positions` / `prompt_window` select prompt tokens,
  `generation_tokens` / `generation_positions` / `generation_window`
  select decode steps. Negative `prompt_positions` resolve from the end
  of the prompt (`-1` = last prompt token), stable across prefill
  chunks; positive ones past the prompt end clamp to the last prompt
  token (with a warning at admission).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from vllm.model_hooks.selection.spec import SelectSpec
from vllm.model_hooks.steering.input_validation import validate_vector_input


class ApplySpec(SelectSpec):
    """Where and when a steering vector applies.

    The selection language itself lives in `SelectSpec` (shared with
    capture); `ApplySpec` is the steering-facing name.
    """


class VectorSpec(BaseModel):
    """One steering vector and how it applies.

    Attributes:
        source: Path to a vector file in a format EasySteer itself
            defines (its GGUF export; moe_router JSON). For third-party
            checkpoint formats, load the file yourself (or use an
            `easysteer.vectors` adapter) and pass `data` instead.
        data: An in-memory payload (`vllm.model_hooks.steering.payloads`, or
            its wire dict) — the canonical way to steer with tensors
            you constructed or loaded yourself. Mutually exclusive with
            source.
        algorithm: Steering algorithm name (registry key).
        scale: Scale factor.
        layers: Layer indices to apply to; None lets the source decide.
        normalize: Rescale the transformed hidden state to its original norm
            for direct, erase, replace, and concept_replace. Other algorithms
            reject True.
        apply: Where/when the vector applies (required).
        params: Algorithm-specific parameters; validated per algorithm,
            unknown keys are rejected.
        name: Optional request label; used as the engine-side request
            name when set (a generated id otherwise).
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
        validate_vector_input(
            self.source,
            self.data,
            self.algorithm,
            self.layers,
            self.params,
            normalize=self.normalize,
        )
        return self


class SteeringSpec(BaseModel):
    """A complete steering configuration: ordered vectors + conflict policy.

    Attributes:
        vectors: The vectors to apply (non-empty; order matters for
            'sequential'/'priority' conflict resolution).
        conflict: What to do when several vectors target one position:
            'priority' (first wins), 'sequential' (stack), 'error'.
    """

    model_config = ConfigDict(extra="forbid")

    vectors: list[VectorSpec]
    conflict: Literal["priority", "sequential", "error"] = "priority"

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
    """Resolve an authoring spec into the engine's ordered intervention list.

    Every source is frozen before admission. Name and integer ID are labels;
    payload content and application fields define execution identity.
    """
    from vllm.model_hooks.steering.loading import resolve_vector_payload
    from vllm.model_hooks.steering.request import (
        ResolvedVector,
        SteeringRequest,
        _next_steer_vector_id,
    )

    if name is None:
        from vllm.utils import random_uuid

        name = spec.vectors[0].name or f"steering-{random_uuid()[:8]}"
    if int_id is None:
        int_id = _next_steer_vector_id()

    vectors = [
        ResolvedVector(
            source=vector.source or "",
            payload=resolve_vector_payload(
                vector.source,
                vector.data,
                vector.algorithm,
                vector.layers,
                vector.params,
            ),
            scale=vector.scale,
            target_layers=list(vector.layers) if vector.layers is not None else None,
            algorithm=vector.algorithm,
            normalize=vector.normalize,
            apply_spec=vector.apply.to_wire(),
        )
        for vector in spec.vectors
    ]
    return SteeringRequest(
        steer_vector_name=name,
        steer_vector_int_id=int_id,
        vectors=vectors,
        conflict_resolution=spec.conflict,
    )
