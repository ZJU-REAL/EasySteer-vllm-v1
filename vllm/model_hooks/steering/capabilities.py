# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static algorithm contracts used by validation and packaged clients.

Execution and graph support remain on the algorithm classes. This module
contains only authoring rules and can be read without loading model code.
"""

from dataclasses import dataclass

ROUTER_OVERRIDE_PARAMS = ("mode", "lambda", "topk")


@dataclass(frozen=True)
class AlgorithmCapabilities:
    payload_kind: str | None
    source: str
    normalize: bool = False
    params: tuple[str, ...] = ()
    target_component: str = "hidden_states"


ALGORITHM_CAPABILITIES = {
    "direct": AlgorithmCapabilities("direction", "gguf", True),
    "linear": AlgorithmCapabilities("linear", "none"),
    "loreft": AlgorithmCapabilities("reft", "none"),
    "lm_steer": AlgorithmCapabilities("lowrank", "none"),
    "erase": AlgorithmCapabilities("direction", "gguf", True),
    "replace": AlgorithmCapabilities("direction", "gguf", True),
    "concept_replace": AlgorithmCapabilities("concept_pair", "file", True),
    "moe_router": AlgorithmCapabilities(
        "router",
        "file",
        params=("expert_ids", "mode", "lambda", "topk"),
        target_component="router_logits",
    ),
}


def algorithm_target(name: str) -> str:
    """Return the component this algorithm transforms, without model imports."""
    try:
        return ALGORITHM_CAPABILITIES[name].target_component
    except KeyError:
        raise ValueError(f"Unknown steering algorithm: {name!r}") from None


def declared_components(algorithms: list[str] | str | None) -> frozenset[str]:
    """Component hook topology required by a normalized workload declaration."""
    if algorithms is None:
        raise ValueError(
            "steering algorithms must be resolved before deriving execution topology"
        )
    names = ALGORITHM_CAPABILITIES if algorithms == "all" else algorithms
    return frozenset(algorithm_target(name) for name in names)


def validate_normalize(algorithm: str, normalize: bool) -> None:
    capability = ALGORITHM_CAPABILITIES.get(algorithm)
    if normalize and (capability is None or not capability.normalize):
        raise ValueError(f"algorithm {algorithm!r} does not support normalize=True")
