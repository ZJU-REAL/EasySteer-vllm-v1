# SPDX-License-Identifier: Apache-2.0
"""Model-runner mixin exposing hook-based capture (hidden states, MoE
router logits) over collective_rpc.

The capture mechanism lives in vllm.capture.session.CaptureSession;
this mixin owns one session per worker, attaches its hooks at model
load, and exposes stream lifecycle RPCs. Capture works on any engine:
while a stream is enabled the runner dispatches batches to the raw
eager forward (skip_compiled), where the hooks run natively; compiled
artifacts and CUDA graphs carry no capture code.
"""

from typing import Any

from torch import nn

from vllm.capture.session import CaptureSession
from vllm.logger import init_logger

logger = init_logger(__name__)


class CaptureModelRunnerMixin:
    """Capture support for the GPU model runner (V1 and V2)."""

    def _capture_session(self) -> CaptureSession:
        if not hasattr(self, "capture_session"):
            self.capture_session = CaptureSession()
        return self.capture_session

    def _check_capture_compatible(self) -> None:
        """Reject capture on engines where it would be silently incomplete.

        Prefix caching needs no engine-level restriction: cache-hit
        tokens are never recomputed, so capture requests carry a unique
        cache_salt (full recompute, no hits), and unsalted requests that
        do hit the cache while capture is enabled are flagged at
        admission and fail explicitly at fetch.
        """
        vllm_config = self.vllm_config
        if not vllm_config.use_v2_model_runner:
            raise RuntimeError(
                "Capture requires the V2 model runner (the V1 runner "
                "provides degraded batch geometry: no prompt lengths, no "
                "request identity on its graph path). Set "
                "VLLM_USE_V2_MODEL_RUNNER=1 — architectures outside the "
                "default-V2 list either work or fail explicitly at "
                "engine build."
            )

    def _attach_capture_hooks(self, model: nn.Module) -> nn.Module:
        """Attach capture hooks (once, at load). Model tree is untouched.

        Attached on every engine: on compiled engines the hook bodies
        trace to nothing (torch.compiler.is_compiling guard), so
        compiled artifacts and CUDA graphs carry no capture code;
        capture-active batches are dispatched to the raw eager forward
        (skip_compiled), where the hooks run natively.

        Must run after the steering hooks are registered so gate-hook
        ordering makes captured router logits post-steering.
        """
        self._capture_session().attach(model)
        return model

    # ------------------------------------------------------------------
    # Stream API
    # ------------------------------------------------------------------

    def start_capture(self, stream: str, **config_kwargs) -> bool:
        """Enable a capture stream ('hidden_states' or 'router_logits').

        config_kwargs: layers (list|None), dtype (e.g. 'float16'),
        select (SelectSpec wire dict — the shared where-clause language,
        see vllm.steer_vectors.SelectSpec.to_wire()), reduce
        ('all'|'last'|'mean'), budget_rows (int|None).
        """
        self._check_capture_compatible()
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
        return self._capture_session().stream_status(stream)
