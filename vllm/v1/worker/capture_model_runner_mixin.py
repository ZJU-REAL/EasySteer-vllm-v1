# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner mixin exposing hook-based capture (hidden states, MoE
router logits) over collective_rpc.

The capture mechanism lives in vllm.model_hooks.capture.session.CaptureSession;
this mixin owns one session per worker, attaches its hooks at model
load, and exposes stream lifecycle RPCs. Eligible FULL batches use a
separate capture graph with fixed outputs; other batches needing rows
use the raw eager forward. Ordinary compiled artifacts omit capture hooks.
"""

from typing import Any

from torch import nn

from vllm.model_hooks.capture.session import CaptureSession
from vllm.v1.worker.gpu.model_hook_utils import release_capture_graph


class CaptureModelRunnerMixin:
    """Capture model lifecycle and RPCs for the V2 GPU runner."""

    def _capture_session(self) -> CaptureSession:
        if not hasattr(self, "capture_session"):
            self.capture_session = CaptureSession()
            self.capture_graph_manager = None
        return self.capture_session

    def _detach_capture_hooks(self) -> None:
        session = getattr(self, "capture_session", None)
        if session is not None:
            release_capture_graph(self)
            session.detach()
            del self.capture_session

    def _attach_capture_hooks(self, model: nn.Module, components) -> None:
        """Attach capture hooks at model load, releasing any previous model.

        On compiled engines the hook bodies
        trace to nothing (torch.compiler.is_compiling guard), so
        ordinary compiled artifacts carry no capture code. Capture
        graphs record a separate raw forward into fixed output buffers;
        other capture-active batches use raw eager execution.

        Must run after the steering hooks are registered so gate-hook
        ordering makes captured router logits post-steering.
        """
        self._detach_capture_hooks()
        self._capture_session().attach(model, components)

    # ------------------------------------------------------------------
    # Stream API
    # ------------------------------------------------------------------

    def start_capture(self, stream: str, **config_kwargs) -> bool:
        """Enable a capture stream ('hidden_states' or 'router_logits').

        Args:
            stream: Component name, such as 'hidden_states' or 'router_logits'.
            **config_kwargs: StreamConfig options: layers, dtype, select,
                reduce, and budget_rows. Select uses SelectSpec.to_wire().
        """
        self._capture_session().enable_stream(stream, **config_kwargs)
        return True

    def stop_capture(self, stream: str) -> bool:
        self._capture_session().disable_stream(stream)
        return True

    def fetch_captured(
        self,
        stream: str,
        clear: bool = True,
        layers: list[int] | None = None,
        req_ids: list[str] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Fetch (and by default clear) captured rows.

        layers restricts to a layer subset; req_ids (client-visible
        request ids) restricts to those requests' rows, with clear
        removing only the emitted rows — the per-request drain that
        bounds peak message size for large corpora.
        """
        return self._capture_session().fetch_stream(
            stream, clear=clear, layers=layers, req_ids=req_ids
        )

    def clear_captured(self, stream: str) -> bool:
        """Drop captured rows, keeping the stream enabled."""
        self._capture_session().clear_stream(stream)
        return True

    def capture_status(self, stream: str) -> dict[str, Any]:
        status = self._capture_session().stream_status(stream)
        status["graph_ready"] = self.capture_graph_manager is not None
        return status
