# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed capture outputs and per-forward recording checks."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


class CaptureGraphState:
    def __init__(
        self, signature: tuple, expected_outputs: set[tuple[str, int]] | None = None
    ):
        self.signature = signature
        self.ready = False
        self.recording = False
        self.buffers: dict[tuple[str, int], tuple[torch.Tensor, str]] = {}
        self.allocation_bytes = 0
        self._expected = (
            {(stream, layer) for stream, layers in signature for layer in layers}
            if expected_outputs is None
            else expected_outputs
        )
        self._seen: set[tuple[str, int]] | None = None

    def close(self) -> None:
        self.buffers.clear()
        self.allocation_bytes = 0
        self.ready = False
        self.recording = False
        self._seen = None

    def begin_forward(self) -> None:
        self._seen = set()

    def end_forward(self) -> None:
        seen = self._seen
        self._seen = None
        if seen != self._expected:
            raise RuntimeError(
                "Capture graph forward did not write its configured outputs: "
                f"expected {sorted(self._expected)}, recorded {sorted(seen or ())}"
            )

    @property
    def buffer_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor, _ in self.buffers.values()
        )

    def record(self, stream: str, layer: int, tensor: torch.Tensor, name: str) -> None:
        key = (stream, layer)
        if self._seen is not None:
            if key in self._seen:
                raise RuntimeError(f"Capture graph output {key} was written twice")
            self._seen.add(key)
        if key not in self.buffers:
            # FULL buckets warm up in descending size, outside graph recording.
            self.buffers[key] = (torch.empty_like(tensor), name)
        target, _ = self.buffers[key]
        if tensor.shape[0] > target.shape[0] or tensor.shape[1:] != target.shape[1:]:
            raise RuntimeError("Capture graph output exceeds its fixed buffer")
        target[: tensor.shape[0]].copy_(tensor)

    @contextmanager
    def record_outputs(self, model: nn.Module) -> Iterator[None]:
        # Validate each warmup AND each actual recorded bucket independently.
        # Existing warmup buffers must never make a missing copy look complete.
        handles = [
            model.register_forward_pre_hook(lambda *_: self.begin_forward()),
            model.register_forward_hook(lambda *_: self.end_forward()),
        ]
        self.ready = False
        self.recording = True
        try:
            yield
            self.ready = True
        finally:
            self.recording = False
            self._seen = None
            for handle in handles:
                handle.remove()
