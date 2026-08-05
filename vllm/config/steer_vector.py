# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for Steer Vectors."""

import hashlib
from typing import TYPE_CHECKING, Any, Literal
import torch
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.config.cache import CacheConfig
else:
    ModelConfig = Any
    CacheConfig = Any

logger = init_logger(__name__)

SteerVectorDType = Literal["auto", "float16", "bfloat16", "float32"]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SteerVectorConfig:
    """Configuration for Steer Vectors.

    Steer vectors allow runtime intervention in model behavior by adding
    control vectors to hidden states at specific layers.
    """

    max_steer_vectors: int = Field(default=8, ge=1)
    """Maximum number of steer vectors in a single batch."""

    max_cpu_steer_vectors: int | None = None
    """Maximum number of steer vectors to store in CPU memory. 
    Must be >= max_steer_vectors. If None, defaults to max_steer_vectors."""

    steer_vector_dtype: SteerVectorDType = "auto"
    """Data type for steer vectors. If 'auto', will default to base model dtype."""

    require_preload: bool = False
    """When True, per-request steering configs referencing vectors that
    were not explicitly preloaded are rejected at the frontend instead of
    lazily loading from disk during request admission. Recommended for
    latency-sensitive serving; leave False for dynamic-vector research
    workflows. (Not part of compute_hash: does not affect the graph.)"""

    algorithms: list[str] | str | None = None
    """The steering workload declaration: the algorithm names requests
    will use (e.g. ["direct", "lm_steer"]), or "all" to allow every
    algorithm. Required whenever steering is enabled without an
    engine-default steering_config (which declares its own workload).
    Requests using undeclared algorithms are rejected at admission on
    every engine, so the declaration is the serving contract; the
    engine also derives the graph execution tier from it (see
    graph_mode)."""

    multi_vector: bool = False
    """Declare that requests may carry multi-vector steering configs
    (SteeringSpec with more than one vector). Multi-vector composition
    is outside the in-graph kernel families, so declaring it resolves
    graph_mode=auto to "split". Implied by "all" and by a multi-vector
    engine-default steering_config."""

    graph_mode: str = "auto"
    """How steering integrates with compiled execution. "in_graph"
    keeps full CUDA graphs by capturing data-driven per-layer kernel
    families (additive, projection, low-rank, replace) that read
    persistent buffers filled host-side each step — single-vector
    configs of the graph-family algorithms are admitted, subject to
    per-payload conditions (see graph_max_rank). "split" adds the
    steering ops to the compilation splitting ops so the compiled
    graph is partitioned at every steered layer and steering runs
    eagerly between the segments — all algorithms supported, at
    roughly half the throughput of in-graph steering. "auto" (default)
    resolves from the declared workload during VllmConfig post-init:
    "in_graph" when compiled execution is on and every declared
    algorithm is unconditionally graph-safe (or the engine-default
    steering_config is concretely admissible), else "split".
    "in_graph" requires compiled execution: combining it with
    enforce_eager is rejected at engine construction. Explicit values
    are an expert override — the declaration still bounds what may
    run."""

    graph_max_rank: int = Field(default=32, ge=1)
    """Rank capacity of the in-graph low-rank steering buffers
    (loreft rotations, lm_steer projectors). Payloads above this rank
    are rejected under graph_mode=in_graph; raise it (more buffer
    memory, ~3 * layers * slots * hidden * rank values) or use
    split mode."""

    steering_config: str | None = None
    """Engine-default steering: a SteeringSpec as inline JSON or a path
    to a JSON file. Applied to every request; per-request steering is
    rejected while it is active. Normalized to canonical JSON text at
    validation time so every consumer (salt, worker install, endpoint)
    sees identical content."""

    @property
    def has_server_config(self) -> bool:
        """True when engine-default steering is configured."""
        return self.steering_config is not None

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.
        """
        factors: list[Any] = []
        factors.append(self.max_steer_vectors)
        factors.append(self.steer_vector_dtype)
        factors.append(self.graph_mode)
        factors.append(self.graph_max_rank)
        factors.append(self.steering_config)

        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @property
    def adapter_dtype(self) -> torch.dtype:
        """The resolved torch dtype vectors are loaded in.

        "auto" is resolved to the model dtype during VllmConfig
        post-init; seeing it here means that resolution never ran.
        """
        if isinstance(self.steer_vector_dtype, torch.dtype):
            return self.steer_vector_dtype
        if self.steer_vector_dtype == "auto":
            raise RuntimeError(
                "steer_vector_dtype='auto' was not resolved to the model "
                "dtype; this config did not go through VllmConfig "
                "initialization. Set an explicit dtype."
            )
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.steer_vector_dtype not in dtype_map:
            raise ValueError(f"Unknown steer_vector_dtype: {self.steer_vector_dtype}")
        return dtype_map[self.steer_vector_dtype]

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.max_cpu_steer_vectors is None:
            self.max_cpu_steer_vectors = self.max_steer_vectors
        if self.max_cpu_steer_vectors < self.max_steer_vectors:
            raise ValueError(
                f"max_cpu_steer_vectors ({self.max_cpu_steer_vectors}) "
                f"must be >= max_steer_vectors ({self.max_steer_vectors})"
            )
        if self.graph_mode in ("full", "piecewise"):
            renamed = {"full": "in_graph", "piecewise": "split"}
            raise ValueError(
                f"steer_graph_mode {self.graph_mode!r} was renamed to "
                f"{renamed[self.graph_mode]!r} (the steering tiers are "
                "distinct from vLLM's cudagraph modes)"
            )
        if self.graph_mode not in ("auto", "in_graph", "split"):
            raise ValueError(
                f"steer_graph_mode must be 'auto', 'in_graph' or "
                f"'split'; got {self.graph_mode!r}"
            )
        if self.algorithms is not None:
            from vllm.steer_vectors.algorithms.factory import (
                ALGORITHM_REGISTRY,
            )

            if isinstance(self.algorithms, str):
                names = (
                    ["all"]
                    if self.algorithms == "all"
                    else [
                        a.strip()
                        for a in self.algorithms.split(",")
                        if a.strip()
                    ]
                )
            else:
                names = list(self.algorithms)
            if not names:
                raise ValueError(
                    "steer_algorithms must not be empty; declare the "
                    "algorithm names requests will use, or 'all'"
                )
            if "all" in names:
                if len(names) > 1:
                    raise ValueError(
                        "steer_algorithms 'all' cannot be combined with "
                        "specific algorithm names"
                    )
                self.algorithms = "all"
            else:
                unknown = sorted(set(names) - set(ALGORITHM_REGISTRY))
                if unknown:
                    raise ValueError(
                        f"unknown steering algorithm(s) {unknown}; "
                        f"available: {sorted(ALGORITHM_REGISTRY)}"
                    )
                self.algorithms = sorted(set(names))
        if self.steering_config is not None:
            import os

            from vllm.steer_vectors.api import SteeringSpec

            text = self.steering_config
            if not text.lstrip().startswith("{"):
                if not os.path.exists(text):
                    raise ValueError(
                        f"steering_config file does not exist: {text!r} "
                        "(pass inline JSON or a path to a SteeringSpec "
                        "JSON file)"
                    )
                with open(text) as f:
                    text = f.read()
            self.steering_config = SteeringSpec.model_validate_json(
                text
            ).model_dump_json()
        return self
