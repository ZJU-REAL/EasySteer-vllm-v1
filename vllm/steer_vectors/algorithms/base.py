# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any


class BaseSteerVectorAlgorithm(ABC):
    """
    Base interface for steer vector algorithms.

    This class defines the core interface that all algorithm implementations must
    follow.
    Parameter management is handled by InterventionController in parameter_control.py,
    allowing algorithm developers to focus purely on transformation logic.
    """

    def __init__(self, layer_id: int | None = None):
        """
        Initialize algorithm with layer ID.

        Args:
            layer_id: Layer index where this algorithm will be applied
        """
        self.layer_id = layer_id

    @classmethod
    @abstractmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict[str, Any]:
        """Load steer vector data from file path and return a dictionary containing "
        "parameters."""
        pass

    @abstractmethod
    def set_payload(self, payload: Any, scale_factor: float = 1.0) -> None:
        """Store this intervention's payload."""
        pass
