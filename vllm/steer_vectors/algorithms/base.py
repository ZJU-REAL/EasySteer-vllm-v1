# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any


def wire_tensor_rank(wire, field: str) -> int | None:
    """Rank (second dim) of a named 2-D tensor in an inline wire payload."""
    if not isinstance(wire, dict):
        return None
    tensor = wire.get("tensors", {}).get(field)
    if isinstance(tensor, dict):
        shape = tensor.get("shape")
        if isinstance(shape, list) and len(shape) == 2:
            return int(shape[1])
    return None


class BaseSteerVectorAlgorithm(ABC):
    """
    Base interface for steer vector algorithms.

    This class defines the core interface that all algorithm implementations must
    follow.
    Trigger management (where a vector applies) is handled by TriggerController in
    triggers.py, allowing algorithm developers to focus purely on transformation
    logic.

    Full-graph (Tier-1) support is declared per algorithm: `graph_family`
    names a kernel family in vllm.steer_vectors.layers.GRAPH_FAMILIES
    (None = piecewise only, rejected at admission under graph_mode=full),
    and `graph_lower` maps this algorithm's (payload, scale) onto that
    family's slot tensors — colocated with the eager `_transform` whose
    math it must reproduce.
    """

    graph_family: str | None = None

    @staticmethod
    def graph_lower(payload: Any, scale: float) -> dict[str, Any]:
        """Lower one layer's (payload, scale) to family slot tensors.

        Returns {table_key: tensor-or-None}; None entries keep the
        table's zero default (e.g. an absent bias).
        """
        raise NotImplementedError

    @staticmethod
    def wire_rank(wire) -> int | None:
        """Rank of an inline wire payload (low-rank families only)."""
        return None

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
