# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request admission policy for enabled capture streams."""

from copy import deepcopy
from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.model_hooks.capture.selection import needs_prompt_recompute


def normalize_capture_select(capture_select: dict | None) -> dict | None:
    """Validate and normalize per-request capture selections at admission.

    Shape: {stream_name: SelectSpec wire dict}. Validating here
    gives the client a clean error instead of a worker-side failure
    mid-forward.
    """
    if capture_select is None:
        return
    from vllm.model_hooks.selection.spec import SelectSpec

    if not isinstance(capture_select, dict):
        raise VLLMValidationError(
            "capture_select must be {stream_name: SelectSpec wire dict}"
        )
    normalized = {}
    for stream, wire in capture_select.items():
        if not isinstance(wire, dict):
            raise VLLMValidationError(
                f"capture_select[{stream!r}] must be a SelectSpec "
                "wire dict (SelectSpec.to_wire())"
            )
        try:
            normalized[stream] = SelectSpec.from_wire(wire).to_wire()
        except (ValueError, TypeError) as exc:
            raise VLLMValidationError(
                f"Invalid capture_select[{stream!r}]: {exc}"
            ) from exc
    return normalized


class CaptureRequestPolicy:
    """Mirror successful stream RPCs so capture can plan cache reads early."""

    def __init__(self) -> None:
        self.streams: dict[str, dict[str, Any]] = {}

    @staticmethod
    def snapshot_kwargs(method: Any, kwargs: dict | None) -> dict | None:
        """Use the same capture configuration during RPC and mirror updates."""
        return deepcopy(kwargs) if method == "start_capture" else kwargs

    def record_rpc(self, method: Any, args: tuple, kwargs: dict | None) -> None:
        """Update only after the worker successfully enabled/disabled a stream."""
        if method not in ("start_capture", "stop_capture"):
            return
        config = dict(kwargs or {})
        stream = args[0] if args else config.pop("stream")
        streams = self.streams.copy()
        if method == "start_capture":
            if config.get("select") is not None:
                from vllm.model_hooks.selection.spec import SelectSpec

                config["select"] = SelectSpec.from_wire(config["select"]).to_wire()
            streams[stream] = deepcopy(config)
        else:
            streams.pop(stream, None)
        # Admission also runs in the tokenizer thread pool. Its iterator must
        # retain a stable snapshot while a stream RPC completes on another thread.
        self.streams = streams

    def skip_prefix_read(
        self,
        prompt_token_ids: list[int] | None,
        request_selects: dict[str, dict] | None,
    ) -> bool:
        """Require recomputation only when a prefix hit could omit wanted rows."""
        overrides = request_selects or {}
        for stream, config in self.streams.items():
            reduce = config.get("reduce", "all")
            # Reductions do not accept per-request selection overrides.
            select = (
                overrides.get(stream, config.get("select"))
                if reduce == "all"
                else config.get("select")
            )
            if needs_prompt_recompute(select, reduce, prompt_token_ids):
                return True
        return False
