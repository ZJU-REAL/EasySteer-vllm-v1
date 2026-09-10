# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared token-selection specification, independent of capture and steering."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from vllm.model_hooks.selection.schema import (
    APPLY_SPEC_KEYS,
    EXCLUDE_SELECTOR_KEYS,
    INCLUDE_SELECTOR_KEYS,
)


def _require_nonempty(name: str, value: list | None) -> None:
    if value is not None and len(value) == 0:
        raise ValueError(f"{name} must be None or non-empty (None disables the filter)")


class SelectSpec(BaseModel):
    """The shared token-selection language (where-clause).

    One selection semantics used by both steering (`ApplySpec`, "which
    tokens to steer") and capture ("which rows to extract"): resolved
    identically by the trigger collector, so a clause means the same
    thing in both systems.

    Each phase is selected independently — `prompt="all"` selects every
    prompt token, `prompt_positions=[-1]` selects one, and a phase with
    neither "all" nor a selector is untouched. "Last prompt token plus
    the whole generation" is therefore one clause:
    `SelectSpec(prompt_positions=[-1], generation="all")`.

    Attributes:
        prompt: "all" selects every prompt token — the widest prompt
            include selector; unions with the others like any include.
        generation: "all" selects every generated token — the widest
            generation include selector.
        prompt_tokens: Token-id allowlist over prompt tokens (real
            ids, >= 0).
        prompt_positions: Prompt positions; negative values are
            Python-style from the end of the prompt (`-1` = last prompt
            token). Positive values past the prompt end clamp to the
            last prompt token (warned at admission).
        prompt_window: Half-open (start, stop) over prompt positions;
            negative bounds resolve from the end of the prompt
            (`(-5, None)` = the last five prompt tokens).
        generation_tokens: Token-id allowlist over generated tokens
            (real ids, >= 0).
        generation_positions: 0-based decode step indices (`[0]` = the
            first generated token).
        generation_window: Half-open (start, stop) over 0-based decode
            steps; stop=None means unbounded.
        exclude_prompt_tokens: Prompt token ids to never select.
        exclude_prompt_positions: Prompt positions to never select
            (same convention as `prompt_positions`).
        exclude_prompt_window: Prompt window to never select (same
            convention as `prompt_window`).
        exclude_generation_tokens: Generated token ids to never select.
        exclude_generation_positions: Decode steps to never select.
        exclude_generation_window: Decode-step window to never select.

    Selection = union of the include selectors' matches ("all" being
    the widest selector of its phase); the exclude selectors union and
    always subtract — where an include and an exclude overlap, the
    exclusion wins. Exclusions require their phase to be covered, and a
    clause that selects nothing is rejected.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: Literal["all"] | None = None
    generation: Literal["all"] | None = None
    prompt_tokens: list[int] | None = None
    prompt_positions: list[int] | None = None
    prompt_window: tuple[int, int | None] | None = None
    generation_tokens: list[int] | None = None
    generation_positions: list[int] | None = None
    generation_window: tuple[int, int | None] | None = None
    exclude_prompt_tokens: list[int] | None = None
    exclude_prompt_positions: list[int] | None = None
    exclude_prompt_window: tuple[int, int | None] | None = None
    exclude_generation_tokens: list[int] | None = None
    exclude_generation_positions: list[int] | None = None
    exclude_generation_window: tuple[int, int | None] | None = None

    def _covers(self, phase: str) -> bool:
        """Whether the clause selects anything in the given phase."""
        if getattr(self, phase) == "all":
            return True
        return any(
            getattr(self, key) is not None
            for key in INCLUDE_SELECTOR_KEYS
            if key.startswith(phase)
        )

    @model_validator(mode="after")
    def _validate(self) -> "SelectSpec":
        if not self._covers("prompt") and not self._covers("generation"):
            raise ValueError(
                "the clause selects nothing: set prompt='all' / "
                "generation='all' or at least one include selector"
            )
        for phase in ("prompt", "generation"):
            excludes = [
                key
                for key in EXCLUDE_SELECTOR_KEYS
                if key.startswith(f"exclude_{phase}") and getattr(self, key) is not None
            ]
            if excludes and not self._covers(phase):
                raise ValueError(
                    f"{sorted(excludes)} exclude {phase} tokens, but the "
                    f"clause selects none: set {phase}='all' or a "
                    f"{phase}_* selector"
                )
        for name in (
            "prompt_tokens",
            "prompt_positions",
            "generation_tokens",
            "generation_positions",
            "exclude_prompt_tokens",
            "exclude_prompt_positions",
            "exclude_generation_tokens",
            "exclude_generation_positions",
        ):
            _require_nonempty(name, getattr(self, name))
        for name in (
            "prompt_tokens",
            "generation_tokens",
            "exclude_prompt_tokens",
            "exclude_generation_tokens",
        ):
            ids = getattr(self, name)
            if ids is not None and any(t < 0 for t in ids):
                raise ValueError(f"{name} must contain real token ids (>= 0)")
        for name in ("generation_positions", "exclude_generation_positions"):
            ids = getattr(self, name)
            if ids is not None and any(j < 0 for j in ids):
                raise ValueError(
                    f"{name} must contain 0-based decode steps (>= 0); "
                    "the generation length is not known up front, so "
                    "end-relative steps cannot resolve"
                )
        for name in ("prompt_window", "exclude_prompt_window"):
            window = getattr(self, name)
            if window is None:
                continue
            start, stop = window
            if stop is not None and (start < 0) == (stop < 0) and stop <= start:
                raise ValueError(
                    f"{name} must be a half-open (start, stop) with "
                    f"stop > start (or stop=None for the prompt end), "
                    f"got {window}"
                )
        for name in ("generation_window", "exclude_generation_window"):
            window = getattr(self, name)
            if window is None:
                continue
            start, stop = window
            if start < 0:
                raise ValueError(f"{name} start must be >= 0, got {start}")
            if stop is not None and stop <= start:
                raise ValueError(
                    f"{name} must be a half-open (start, stop) with "
                    f"stop > start or stop=None, got {window}"
                )
        return self

    def to_wire(self) -> dict[str, Any]:
        """Canonical dict carried on the engine struct (`apply_spec`).

        Fixed key order and list-only containers so the config
        fingerprint and msgspec round-trips are deterministic.
        """

        return self.model_dump(mode="json")

    @classmethod
    def from_wire(cls, wire: dict[str, Any]) -> "SelectSpec":
        """Validate and rebuild a spec from its `to_wire()` dict.

        Used engine-side to reject malformed selection clauses at
        enable time instead of failing mid-forward.
        """
        unknown = set(wire) - set(APPLY_SPEC_KEYS)
        if unknown:
            raise ValueError(f"unknown selection fields: {sorted(unknown)}")

        values = dict(wire)
        for key in (*INCLUDE_SELECTOR_KEYS, *EXCLUDE_SELECTOR_KEYS):
            if key.endswith("_window") and values.get(key) is not None:
                values[key] = tuple(values[key])
        return cls.model_validate(values)
