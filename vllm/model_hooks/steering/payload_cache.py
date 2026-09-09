# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side LRU of materialized, content-addressed steering payloads."""

from collections import OrderedDict

from vllm.logger import init_logger
from vllm.model_hooks.steering.payloads import (
    effective_layers,
    is_broadcast_payload,
    materialize,
    validate_model_shape,
)

logger = init_logger(__name__)


class PayloadCache:
    """Cache tensor content independently of a request's layer mapping."""

    def __init__(self, device: str, steer_vector_config, *, hidden_size: int):
        self.device = device
        self.steer_vector_config = steer_vector_config
        self.hidden_size = hidden_size
        self.capacity = max(1, steer_vector_config.max_steer_vectors)
        self._entries: OrderedDict[str, dict] = OrderedDict()

    def get(
        self,
        wire: dict,
        *,
        target_layers: list[int] | None = None,
    ) -> dict:
        """Materialize admission-validated wire content once per resident entry."""
        broadcasts = is_broadcast_payload(wire)
        if broadcasts:
            target_layers = sorted(effective_layers(wire, target_layers))
        key = wire["sha256"]
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        else:
            validate_model_shape(wire, self.hidden_size)
            layer_payloads = materialize(
                wire,
                self.device,
                self.steer_vector_config.adapter_dtype,
                target_layers if broadcasts else None,
            )
            if not layer_payloads:
                raise ValueError(
                    f"{wire.get('kind')!r} payload produced no layer payloads; "
                    "the vector would steer nothing"
                )
            entry = (
                next(iter(layer_payloads.values())) if broadcasts else layer_payloads
            )
            self._entries[key] = entry
            logger.info("Materialized steer payload: %s (%s)", key[:12], wire["kind"])
            while len(self._entries) > self.capacity:
                evicted_key, _ = self._entries.popitem(last=False)
                logger.info("Evicted steer payload: %s", evicted_key[:12])
        if broadcasts:
            # Per-layer dictionaries are private; their immutable tensors are shared.
            return {layer: dict(entry) for layer in target_layers}
        return entry

    def clear(self) -> None:
        self._entries.clear()

    def preload(
        self,
        wire: dict,
        *,
        target_layers: list[int] | None = None,
    ) -> None:
        self.get(wire, target_layers=target_layers)
