# SPDX-License-Identifier: Apache-2.0
"""Capture request structures (legacy easysteer-client API).

The stream API (`start_capture`/`fetch_captured`) does not use these;
they are kept for clients that still pass request objects.
"""

import msgspec


class HiddenStatesCaptureRequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Request to enable hidden-states capture for specific requests."""

    request_id: str
    """Unique identifier for this capture request."""

    capture_layers: list[int] | None = None
    """0-based layer indices to capture; None captures all layers."""

    return_cpu: bool = True
    """Whether to return tensors on CPU."""

    def __post_init__(self):
        if self.capture_layers is not None:
            if not isinstance(self.capture_layers, list):
                raise ValueError("capture_layers must be a list of integers")
            if any(layer < 0 for layer in self.capture_layers):
                raise ValueError("Layer indices must be non-negative")

    def __eq__(self, value: object) -> bool:
        return isinstance(value, self.__class__) and self.request_id == value.request_id

    def __hash__(self) -> int:
        return hash(self.request_id)

    def should_capture_layer(self, layer_id: int) -> bool:
        """Whether the given 0-based layer index should be captured."""
        if self.capture_layers is None:
            return True
        return layer_id in self.capture_layers


class MoERouterLogitsCaptureRequest(HiddenStatesCaptureRequest):
    """Router-logits variant; identical fields and semantics."""
