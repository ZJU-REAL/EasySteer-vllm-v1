# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step batch geometry and device views shared by model hooks."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np


@dataclass
class BatchView:
    """Complete per-request selection metadata on one device."""

    query_start_loc: torch.Tensor
    num_computed: torch.Tensor
    num_prompt: torch.Tensor
    num_output: torch.Tensor
    is_decode: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.is_decode = self.num_output > 0


@dataclass
class BatchGeometry:
    """One runner-produced snapshot shared by steering and capture each step.

    Request arrays follow query_start_loc segment order. Counts remain on the
    host for steering resolution; device views are created once on demand.
    """

    query_start_loc: torch.Tensor
    """Cumulative input-token counts per request on the model device."""
    query_start_loc_cpu: "np.ndarray"
    """Actual request boundaries on the host, including adaptive verification."""
    num_computed: torch.Tensor
    """Host counts of cached/computed tokens per request."""
    num_prompt: torch.Tensor
    """Host prompt lengths, stable across prefill chunks."""
    num_output: torch.Tensor
    """Host generated-token counts; zero while prefilling."""
    req_ids: list[str]
    token_ids: torch.Tensor
    """Flat input token ids, excluding batch padding."""

    _token_ids_cpu: "np.ndarray | None" = field(default=None, init=False, repr=False)
    _device_views: dict[torch.device, BatchView] = field(
        default_factory=dict, init=False, repr=False
    )

    def token_ids_cpu(self) -> "np.ndarray":
        """Read token ids on the host once per step, only when requested."""
        if self._token_ids_cpu is None:
            self._token_ids_cpu = self.token_ids.cpu().numpy()
        return self._token_ids_cpu

    def device_view(self, device: torch.device | None = None) -> BatchView:
        """Reuse one complete selection view per device for this step."""
        device = self.query_start_loc.device if device is None else device
        if device not in self._device_views:
            self._device_views[device] = BatchView(
                query_start_loc=self.query_start_loc.to(device),
                num_computed=self.num_computed.to(device, non_blocking=True),
                num_prompt=self.num_prompt.to(device, non_blocking=True),
                num_output=self.num_output.to(device, non_blocking=True),
            )
        return self._device_views[device]
