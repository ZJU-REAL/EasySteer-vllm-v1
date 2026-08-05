# SPDX-License-Identifier: Apache-2.0
"""Registry and factory for steering algorithms."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseSteerVectorAlgorithm

ALGORITHM_REGISTRY: dict[str, type["BaseSteerVectorAlgorithm"]] = {}


def register_algorithm(name: str):
    """Class decorator registering an algorithm under a unique name.

    Args:
        name: Unique name of the algorithm (e.g., "direct", "loreft").
    """

    def decorator(cls: type["BaseSteerVectorAlgorithm"]):
        if name in ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm '{name}' is already registered.")
        ALGORITHM_REGISTRY[name] = cls
        return cls

    return decorator


def get_algorithm(name: str) -> type["BaseSteerVectorAlgorithm"]:
    """Look up a registered algorithm class by name.

    Raises:
        ValueError: If the algorithm name is not registered.
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: '{name}'. Available algorithms: "
            f"{list(ALGORITHM_REGISTRY.keys())}"
        )
    return ALGORITHM_REGISTRY[name]


def create_algorithm(name: str, *args, **kwargs) -> "BaseSteerVectorAlgorithm":
    """Create an algorithm instance by registered name."""
    return get_algorithm(name)(*args, **kwargs)


def graph_safe_algorithms() -> frozenset[str]:
    """Algorithms declaring an in-graph (Tier-1) kernel family."""
    return frozenset(
        name
        for name, cls in ALGORITHM_REGISTRY.items()
        if cls.graph_family is not None
    )


def graph_condition(name: str) -> str | None:
    """The per-payload condition an algorithm's in-graph support carries.

    None for unconditionally graph-safe algorithms (any payload runs
    in-graph) and for algorithms with no kernel family at all (never
    in-graph — see graph_safe_algorithms). Conditional algorithms are
    resolved pessimistically when only their name is declared: without
    a concrete payload at boot, auto assumes the general case and picks
    split mode.
    """
    from vllm.steer_vectors.graph_kernels import GRAPH_FAMILIES

    cls = get_algorithm(name)
    if cls.graph_family is None:
        return None
    if name == "moe_router":
        return "only inline activate/deactivate expert configs run in-graph"
    dims = GRAPH_FAMILIES.get(cls.graph_family, {})
    if any("r" in d for d in dims.values()):
        return "payload rank must be <= steer_graph_max_rank"
    return None


def unconditionally_graph_safe_algorithms() -> frozenset[str]:
    """Algorithms whose every payload runs inside full CUDA graphs."""
    return frozenset(
        name
        for name in graph_safe_algorithms()
        if graph_condition(name) is None
    )


def steering_execution_modes() -> dict[str, tuple[str, ...]]:
    """Central algorithm -> supported steering graph tiers table.

    Derived from each algorithm's declared graph_family (the single
    source of truth on the class), never hand-maintained: every
    algorithm runs under split mode; those with a kernel family also
    run in-graph (conditionally for some — see graph_condition).
    """
    return {
        name: (("split", "in_graph") if cls.graph_family else ("split",))
        for name, cls in sorted(ALGORITHM_REGISTRY.items())
    }
