# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any


class BaseSteerVectorAlgorithm(ABC):
    """
    Base interface for steer vector algorithms.

    This class defines the core interface that all algorithm implementations must
    follow.
    Trigger management (where a vector applies) is handled by TriggerController in
    triggers.py, allowing algorithm developers to focus purely on transformation
    logic.
    """

    def __init__(self, layer_id: int | None = None):
        """
        Initialize algorithm with layer ID.

        Args:
            layer_id: Layer index where this algorithm will be applied
        """
        self.layer_id = layer_id

    @classmethod
    def load_from_path(
        cls,
        path: str,
        device: str,
        *,
        config,
        target_layers: list[int] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Load steer vector data from a source file.

        Only formats whose schema EasySteer itself defines are loaded
        engine-side (its GGUF export; the moe_router JSON config).
        Third-party checkpoint formats are interpreted client-side —
        load the file yourself (or use an ``easysteer.vectors``
        adapter) and pass ``VectorSpec(data=...)``.

        Args:
            path: Vector file or directory path.
            device: Device to load tensors on.
            config: The engine's SteerVectorConfig.
            target_layers: Optional layer restriction from the request.
            **kwargs: Algorithm-specific parameters (e.g. moe_mode).

        Returns:
            ``{"layer_payloads": {layer_idx: payload}}`` where each
            payload is whatever this algorithm's ``_transform`` consumes
            (tensor, dict, ...).
        """
        raise ValueError(
            f"algorithm {cls.__name__} loads no source files; pass its "
            "payload via VectorSpec(data=...) — see "
            "vllm.steer_vectors.payloads and easysteer.vectors"
        )

    @abstractmethod
    def set_payload(self, payload: Any, scale_factor: float = 1.0) -> None:
        """Store this intervention's payload."""
        pass
