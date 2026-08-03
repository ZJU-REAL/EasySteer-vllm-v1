# SPDX-License-Identifier: Apache-2.0
"""Model-runner mixin exposing hook-based capture (hidden states, MoE
router logits) over collective_rpc.

The capture mechanism lives in vllm.capture.session.CaptureSession;
this mixin owns one session per worker, attaches its hooks at model load
(eager engines only — hooks cannot run inside compiled graphs), and
exposes stream lifecycle RPCs. The legacy per-feature RPC names are kept
as thin shims over the stream API for existing easysteer clients.
"""

from typing import Any

from torch import nn

from vllm.capture.session import (
    HIDDEN_STATES,
    ROUTER_LOGITS,
    CaptureSession,
)
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

        With prefix caching, cache-hit tokens are never recomputed, so
        their hidden states / router logits cannot be captured (verified:
        a warm-cache request captures only the uncached suffix).
        """
        vllm_config = getattr(self, "vllm_config", None)
        if (
            vllm_config is not None
            and vllm_config.cache_config is not None
            and vllm_config.cache_config.enable_prefix_caching
        ):
            raise RuntimeError(
                "Capture requires enable_prefix_caching=False: prefix-cache "
                "hits skip recomputation, so the cached tokens' states "
                "would be silently missing from the capture."
            )

    def _attach_capture_hooks(self, model: nn.Module) -> nn.Module:
        """Attach capture hooks (once, at load). Model tree is untouched.

        Must run after the steering hooks are registered so gate-hook
        ordering makes captured router logits post-steering.
        """
        session = self._capture_session()
        enforce_eager = (
            getattr(self, "vllm_config", None) is not None
            and self.vllm_config.model_config is not None
            and self.vllm_config.model_config.enforce_eager
        )
        if not enforce_eager:
            logger.info(
                "Capture hooks not attached (compiled execution); "
                "launch with enforce_eager=True to use capture APIs."
            )
            return model
        session.attach(model)
        return model

    # Legacy attach entry points (V1 runner calls both; attach() is
    # idempotent and covers both streams).
    def _wrap_model_for_hidden_states(self, model: nn.Module) -> nn.Module:
        return self._attach_capture_hooks(model)

    def _wrap_model_for_moe_capture(self, model: nn.Module) -> nn.Module:
        return self._attach_capture_hooks(model)

    # ------------------------------------------------------------------
    # Stream API
    # ------------------------------------------------------------------

    def start_capture(self, stream: str, **config_kwargs) -> bool:
        """Enable a capture stream ('hidden_states' or 'router_logits').

        config_kwargs: layers (list|None), dtype (e.g. 'float16'),
        positions ('all'|'last'|'mean'|list of absolute positions,
        negatives from the prompt end), token_ids (list|None — capture
        only rows whose input token id matches; unions with a positions
        list), max_tokens (int|None).
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
    ) -> dict[int, dict[str, Any]]:
        return self._capture_session().fetch_stream(
            stream, clear=clear, layers=layers
        )

    def capture_status(self, stream: str) -> dict[str, Any]:
        return self._capture_session().stream_status(stream)

    # ------------------------------------------------------------------
    # Legacy RPC shims (existing easysteer clients)
    # ------------------------------------------------------------------

    def enable_hidden_states_capture(self, **config_kwargs):
        self._check_capture_compatible()
        self._capture_session().enable_stream(HIDDEN_STATES, **config_kwargs)

    def disable_hidden_states_capture(self):
        self._capture_session().disable_stream(HIDDEN_STATES)

    def get_captured_hidden_states(self) -> dict[int, dict[str, Any]]:
        return self._capture_session().fetch_stream(HIDDEN_STATES, clear=False)

    def clear_hidden_states(self):
        self._capture_session().clear_stream(HIDDEN_STATES)

    def get_hidden_states_debug_info(self) -> dict[str, Any]:
        return self._capture_session().stream_status(HIDDEN_STATES)

    def enable_moe_router_logits_capture(self, **config_kwargs):
        self._check_capture_compatible()
        self._capture_session().enable_stream(ROUTER_LOGITS, **config_kwargs)

    def disable_moe_router_logits_capture(self):
        self._capture_session().disable_stream(ROUTER_LOGITS)

    def get_moe_router_logits(self) -> dict[int, dict[str, Any]]:
        return self._capture_session().fetch_stream(ROUTER_LOGITS, clear=False)

    def clear_moe_router_logits(self):
        self._capture_session().clear_stream(ROUTER_LOGITS)

    def get_moe_debug_info(self) -> dict[str, Any]:
        return self._capture_session().stream_status(ROUTER_LOGITS)

    def get_all_capture_status(self) -> dict[str, Any]:
        session = self._capture_session()
        return {
            HIDDEN_STATES: session.stream_status(HIDDEN_STATES),
            ROUTER_LOGITS: session.stream_status(ROUTER_LOGITS),
        }

    def clear_all_captures(self):
        self.clear_hidden_states()
        self.clear_moe_router_logits()
