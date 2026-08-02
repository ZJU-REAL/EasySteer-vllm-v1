# SPDX-License-Identifier: Apache-2.0

"""Steer Vector Request classes for vLLM V1."""

import msgspec
from pydantic import BaseModel

# --- Canonical steering parameter schema ---
#
# Single source of truth for "what configures how a vector is applied".
# The API models and engine structs below must declare these fields
# (enforced by _assert_schema_complete at import time), and every
# conversion/copy site iterates these tuples instead of spelling the
# fields out. To add a parameter: add it here, add the field to the four
# schema classes, and consume it where needed — nothing else to keep in
# sync.

STEER_TRIGGER_FIELDS: tuple[str, ...] = (
    "prefill_trigger_tokens",
    "prefill_trigger_positions",
    "prefill_exclude_tokens",
    "prefill_exclude_positions",
    "generate_trigger_tokens",
    "generate_first_k_tokens",
    "generate_after_k_tokens",
)

STEER_APPLY_FIELDS: tuple[str, ...] = (
    "scale",
    "target_layers",
    *STEER_TRIGGER_FIELDS,
    "algorithm",
    "normalize",
)

STEER_MOE_FIELDS: tuple[str, ...] = (
    "moe_expert_ids",
    "moe_mode",
    "moe_lambda",
    "moe_topk",
)

# Trigger fields holding token-id lists; the TriggerController
# stores these as sets for O(1) membership tests.
STEER_TOKEN_SET_FIELDS: frozenset[str] = frozenset(
    ("prefill_trigger_tokens", "prefill_exclude_tokens", "generate_trigger_tokens")
)


def steer_params_dict(obj, fields: tuple = STEER_APPLY_FIELDS) -> dict:
    """Extract canonical steering parameters from any schema object."""
    return {name: getattr(obj, name) for name in fields}


def _has_triggers(obj) -> bool:
    """Whether any canonical trigger field is set on a schema object."""
    return any(getattr(obj, name) is not None for name in STEER_TRIGGER_FIELDS)


def build_server_request(steer_config) -> "SteerVectorRequest":
    """The canonical server-level steering request for a SteerVectorConfig.

    Single source of truth for what server-level steering does (used by
    the worker install path, the runtime scale-update endpoint, and the
    prefix-cache salt): the configured vector applied globally to all
    tokens of every request.
    """
    return SteerVectorRequest(
        steer_vector_name="__server__",
        steer_vector_int_id=1,
        steer_vector_local_path=steer_config.server_vector_path,
        scale=steer_config.server_scale,
        target_layers=steer_config.server_target_layers,
        algorithm=steer_config.server_algorithm,
        normalize=steer_config.server_normalize,
        prefill_trigger_tokens=[-1],
        generate_trigger_tokens=[-1],
    )


def is_prompt_length_sensitive(request) -> bool:
    """Whether the config's effect on a token depends on the request's
    prompt length (and not just the token's absolute position).

    True for negative trigger/exclude positions (resolved from the end
    of the prompt) and first_k/after_k windows (relative to the end of
    the prompt). Used by prefix-cache block hashing: such configs can
    only share KV blocks between requests with equal prompt lengths.
    """

    def _sensitive(obj) -> bool:
        return (
            any(p < 0 for p in (obj.prefill_trigger_positions or []))
            or any(p < 0 for p in (obj.prefill_exclude_positions or []))
            or obj.generate_first_k_tokens is not None
            or obj.generate_after_k_tokens is not None
        )

    if _sensitive(request):
        return True
    return any(_sensitive(vc) for vc in request.vector_configs or [])


def _assert_schema_complete(cls, field_names, required) -> None:
    missing = set(required) - set(field_names)
    assert not missing, (
        f"{cls.__name__} is missing canonical steering fields: {sorted(missing)}"
    )


# --- OpenAI-compatible API parameter types ---

_steer_vector_id_counter = 0


def _next_steer_vector_id() -> int:
    """Generate a unique positive integer ID for steer vectors."""
    global _steer_vector_id_counter
    _steer_vector_id_counter += 1
    if _steer_vector_id_counter > 2147483647:
        _steer_vector_id_counter = 1
    return _steer_vector_id_counter


class VectorConfigParam(BaseModel):
    """Configuration for a single vector in multi-vector mode."""

    path: str
    scale: float = 1.0
    target_layers: list[int] | None = None
    prefill_trigger_tokens: list[int] | None = None
    prefill_trigger_positions: list[int] | None = None
    prefill_exclude_tokens: list[int] | None = None
    prefill_exclude_positions: list[int] | None = None
    generate_trigger_tokens: list[int] | None = None
    generate_first_k_tokens: int | None = None
    generate_after_k_tokens: int | None = None
    algorithm: str = "direct"
    normalize: bool = False


class SteerVectorRequestParam(BaseModel):
    """
    Steer Vector request parameters for OpenAI-compatible API.

    Supports both single-vector mode and multi-vector mode.
    """

    steer_vector_name: str | None = None
    steer_vector_int_id: int | None = None

    steer_vector_local_path: str = ""
    debug: bool = False
    conflict_resolution: str = "priority"

    scale: float = 1.0
    target_layers: list[int] | None = None
    prefill_trigger_tokens: list[int] | None = None
    prefill_trigger_positions: list[int] | None = None
    prefill_exclude_tokens: list[int] | None = None
    prefill_exclude_positions: list[int] | None = None
    generate_trigger_tokens: list[int] | None = None
    generate_first_k_tokens: int | None = None
    generate_after_k_tokens: int | None = None
    algorithm: str = "direct"
    normalize: bool = False

    vector_configs: list[VectorConfigParam] | None = None

    moe_expert_ids: list[int] | None = None
    moe_mode: str | None = None
    moe_lambda: float = 0.5
    moe_topk: int = 8


# --- Engine-level request types (msgspec) ---


class VectorConfig(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
    frozen=False,  # type: ignore[call-arg]
):  # type: ignore[call-arg]
    """
    Configuration for a single vector in multi-vector mode.

    Args:
        path: Local path to the vector file
        scale: Scale factor for this vector (default: 1.0)
        target_layers: List of layer indices to apply this vector to. If None, apply to
        all layers
        prefill_trigger_tokens: List of token IDs that trigger vector application in
        prefill phase.
                               Use [-1] to apply to ALL tokens in prefill phase.
        prefill_trigger_positions: List of token positions that trigger vector
        application in prefill phase.
                                 Supports negative indexing (e.g., -1 for last token).
        prefill_exclude_tokens: List of token IDs to exclude from vector application in
        prefill phase.
                               Exclude has higher priority than trigger tokens.
        prefill_exclude_positions: List of token positions to exclude from vector
        application in prefill phase.
                                  Supports negative indexing. Exclude has higher
                                  priority than trigger positions.
        generate_trigger_tokens: List of token IDs that trigger vector application in
        generate phase.
                                Use [-1] to apply to ALL tokens in generate phase.
        generate_first_k_tokens: Only apply to the first k generated tokens (0-indexed:
        tokens 0, 1, ..., k-1).
                                Mutually exclusive with generate_after_k_tokens.
        generate_after_k_tokens: Start applying from the k-th generated token onwards
        (skip first k tokens).
                                Mutually exclusive with generate_first_k_tokens.
        algorithm: Vector algorithm to use: 'direct' (default) or 'loreft'
        normalize: Whether to normalize the vector (default: False, only applies to
        'direct' algorithm)
    """

    path: str
    scale: float = 1.0
    target_layers: list[int] | None = None
    prefill_trigger_tokens: list[int] | None = None
    prefill_trigger_positions: list[int] | None = None
    prefill_exclude_tokens: list[int] | None = None
    prefill_exclude_positions: list[int] | None = None
    generate_trigger_tokens: list[int] | None = None
    generate_first_k_tokens: int | None = None
    generate_after_k_tokens: int | None = None
    algorithm: str = "direct"
    normalize: bool = False


class SteerVectorRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
    frozen=False,  # type: ignore[call-arg]
):  # type: ignore[call-arg]
    """
    Request to apply a steering configuration.
    Supports both single-vector mode (backward compatible) and multi-vector mode.

    Args:
        steer_vector_name: Name of the steer vector
        steer_vector_int_id: Unique ID for the steer vector (must be > 0)
        debug: Whether to print debug information during forward pass (default: False)
        conflict_resolution: How to handle conflicts when multiple vectors target the
        same position.
                           'error': raise an error when conflicts occur
                           'priority': use the first vector and ignore others (default)
                           'sequential': apply all vectors in sequence (effects stack)

        Single-vector mode (backward compatible):
        steer_vector_local_path: Local path to the steer vector file
        scale: Scale factor for the steer vector (default: 1.0)
        target_layers: List of layer indices to apply the steer vector to. If None,
        apply to all layers
        prefill_trigger_tokens: List of token IDs that trigger steer vector application
        in prefill phase.
        prefill_trigger_positions: List of token positions that trigger steer vector
        application in prefill phase.
        prefill_exclude_tokens: List of token IDs to exclude from steer vector
        application in prefill phase.
        prefill_exclude_positions: List of token positions to exclude from steer vector
        application in prefill phase.
        generate_trigger_tokens: List of token IDs that trigger steer vector application
        in generate phase.
        generate_first_k_tokens: Only apply to the first k generated tokens (0-indexed:
        tokens 0, 1, ..., k-1).
                                Mutually exclusive with generate_after_k_tokens.
        generate_after_k_tokens: Start applying from the k-th generated token onwards
        (skip first k tokens).
                                Mutually exclusive with generate_first_k_tokens.
        algorithm: Steer vector algorithm to use: 'direct' (default) or 'loreft'
        normalize: Whether to normalize the steer vector (default: False, only applies
        to 'direct' algorithm)

        Multi-vector mode:
        vector_configs: List of VectorConfig objects for multi-vector control
    """

    steer_vector_name: str
    steer_vector_int_id: int
    steer_vector_local_path: str = ""
    debug: bool = False
    conflict_resolution: str = "priority"

    # === Single-vector mode (backward compatible) ===
    scale: float = 1.0
    target_layers: list[int] | None = None
    prefill_trigger_tokens: list[int] | None = None
    prefill_trigger_positions: list[int] | None = None
    prefill_exclude_tokens: list[int] | None = None
    prefill_exclude_positions: list[int] | None = None
    generate_trigger_tokens: list[int] | None = None
    generate_first_k_tokens: int | None = None
    generate_after_k_tokens: int | None = None
    algorithm: str = "direct"
    normalize: bool = False

    # === Multi-vector mode ===
    vector_configs: list[VectorConfig] | None = None

    # === MoE-specific parameters (for moe_router algorithm) ===
    moe_expert_ids: list[int] | None = None  # Expert IDs to intervene
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

        # Validate generate position control parameters
        if (
            self.generate_first_k_tokens is not None
            and self.generate_after_k_tokens is not None
        ):
            raise ValueError(
                "Cannot specify both generate_first_k_tokens and "
                "generate_after_k_tokens. "
                "They are mutually exclusive."
            )

        if (
            self.generate_first_k_tokens is not None
            and self.generate_first_k_tokens < 0
        ):
            raise ValueError(
                "generate_first_k_tokens must be >= 0, got "
                f"{self.generate_first_k_tokens}"
            )

        if (
            self.generate_after_k_tokens is not None
            and self.generate_after_k_tokens < 0
        ):
            raise ValueError(
                "generate_after_k_tokens must be >= 0, got "
                f"{self.generate_after_k_tokens}"
            )

        if self.is_multi_vector:
            if self.steer_vector_local_path:
                raise ValueError(
                    "Cannot specify both steer_vector_local_path and vector_configs"
                )
            if not self.vector_configs:
                raise ValueError("vector_configs cannot be empty in multi-vector mode")
            for i, vc in enumerate(self.vector_configs):
                if not _has_triggers(vc):
                    raise ValueError(
                        f"vector_configs[{i}] has no trigger fields and would "
                        "steer no tokens. Set prefill_trigger_tokens=[-1] "
                        "and/or generate_trigger_tokens=[-1] to apply it to "
                        "all tokens of that phase."
                    )
        else:
            # Special case: moe_router algorithm can work without a file path
            # It uses moe_expert_ids and moe_mode from the request directly
            if self.algorithm != "moe_router" and not self.steer_vector_local_path:
                raise ValueError(
                    "Must specify steer_vector_local_path in single-vector mode "
                    "(except for moe_router algorithm)"
                )
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
            if not _has_triggers(self):
                raise ValueError(
                    "Steering request has no trigger fields and would steer no "
                    "tokens. Set prefill_trigger_tokens=[-1] and/or "
                    "generate_trigger_tokens=[-1] to apply the vector to all "
                    "tokens of that phase."
                )

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
    VectorConfigParam, VectorConfigParam.model_fields, STEER_APPLY_FIELDS
)
_assert_schema_complete(
    SteerVectorRequestParam,
    SteerVectorRequestParam.model_fields,
    STEER_APPLY_FIELDS + STEER_MOE_FIELDS,
)
_assert_schema_complete(
    VectorConfig, VectorConfig.__struct_fields__, STEER_APPLY_FIELDS
)
_assert_schema_complete(
    SteerVectorRequest,
    SteerVectorRequest.__struct_fields__,
    STEER_APPLY_FIELDS + STEER_MOE_FIELDS,
)
