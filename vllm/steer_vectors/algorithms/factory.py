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


def create_algorithm(name: str, *args, **kwargs) -> "BaseSteerVectorAlgorithm":
    """Create an algorithm instance by registered name.

    Raises:
        ValueError: If the algorithm name is not registered.
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: '{name}'. Available algorithms: "
            f"{list(ALGORITHM_REGISTRY.keys())}"
        )
    return ALGORITHM_REGISTRY[name](*args, **kwargs)
