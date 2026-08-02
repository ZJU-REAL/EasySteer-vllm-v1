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

    allow_cuda_graphs: bool = False
    """DEPRECATED, ignored. Steering now always runs eagerly between
    piecewise CUDA-graph segments under compiled execution
    (vllm::steer_apply splitting op); no configs are baked into graphs."""

    require_preload: bool = False
    """When True, per-request steering configs referencing vectors that
    were not explicitly preloaded are rejected at the frontend instead of
    lazily loading from disk during request admission. Recommended for
    latency-sensitive serving; leave False for dynamic-vector research
    workflows. (Not part of compute_hash: does not affect the graph.)"""

    graph_mode: str = "piecewise"
    """CUDA-graph execution mode for steering: "piecewise" (default)
    splits compiled graphs at every steered layer and runs steering
    eagerly between the segments — all algorithms supported. "full" keeps
    full CUDA graphs by capturing a data-driven per-layer kernel
    (hidden += mask * vector_table[token_row]) that reads persistent
    buffers filled host-side each step — only graph-safe configs are
    admitted (direct algorithm, no normalize, single-vector)."""

    steering_config: str | None = None
    """Engine-default steering (v2 API): a SteeringSpec as inline JSON or
    a path to a JSON file. Applied to every request; per-request steering
    is rejected while it is active. Normalized to canonical JSON text at
    validation time so every consumer (salt, worker install, endpoint)
    sees identical content."""

    # Deprecated v1 server-level steering flags; use steering_config.
    server_vector_path: str | None = None
    """Path to a steering vector file to load at startup."""

    server_scale: float = 1.0
    """Scaling factor for the server-level steering vector."""

    server_target_layers: list[int] | None = None
    """Which layers to steer. If None, the vector file determines layers."""

    server_algorithm: str = "direct"
    """Algorithm for server-level steering."""

    server_normalize: bool = True
    """Whether to normalize the server-level steering vector."""

    @property
    def has_server_config(self) -> bool:
        """True when engine-default (server-level) steering is configured."""
        return self.steering_config is not None or self.server_vector_path is not None

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.
        """
        factors: list[Any] = []
        factors.append(self.max_steer_vectors)
        factors.append(self.steer_vector_dtype)
        factors.append(self.allow_cuda_graphs)
        factors.append(self.graph_mode)
        factors.append(self.steering_config)
        factors.append(self.server_vector_path)
        factors.append(self.server_scale)
        factors.append(self.server_target_layers)
        factors.append(self.server_algorithm)
        factors.append(self.server_normalize)

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
        if self.steering_config is not None:
            if self.server_vector_path is not None:
                raise ValueError(
                    "steering_config and the deprecated --steer-vector-path "
                    "flags are mutually exclusive; use steering_config alone"
                )
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
