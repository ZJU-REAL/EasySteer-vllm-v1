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
    """Algorithms declaring a full-graph (Tier-1) kernel family."""
    return frozenset(
        name
        for name, cls in ALGORITHM_REGISTRY.items()
        if cls.graph_family is not None
    )


def steering_execution_modes() -> dict[str, tuple[str, ...]]:
    """Central algorithm -> supported CUDA execution modes table.

    Derived from each algorithm's declared graph_family (the single
    source of truth on the class), never hand-maintained: every
    algorithm runs under piecewise mode; those with a kernel family
    also run inside full CUDA graphs.
    """
    return {
        name: (("piecewise", "full") if cls.graph_family else ("piecewise",))
        for name, cls in sorted(ALGORITHM_REGISTRY.items())
    }
