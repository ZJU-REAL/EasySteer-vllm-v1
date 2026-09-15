# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner mixin exposing hook-based capture (hidden states, MoE
router logits) over collective_rpc.

The capture mechanism lives in vllm.model_hooks.capture.session.CaptureSession;
this mixin owns one session per worker, registers components at model
load, and exposes stream lifecycle RPCs that install and remove hooks.
Eligible FULL batches use a separate capture graph with fixed outputs;
other batches needing rows
use the raw eager forward. Ordinary compiled artifacts omit capture hooks.
"""

from typing import TYPE_CHECKING, Any

from torch import nn

from vllm.model_hooks.capture.session import CaptureSession
from vllm.v1.worker.gpu.model_hook_utils import release_capture_graph

if TYPE_CHECKING:
    from vllm.model_hooks.components.registry import ModelComponents
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager


class CaptureModelRunnerMixin:
    """Capture model lifecycle and RPCs for the V2 GPU runner."""

    capture_session: CaptureSession
    capture_graph_manager: "ModelCudaGraphManager | None"

    def _capture_session(self) -> CaptureSession:
        if not hasattr(self, "capture_session"):
            topology = self._capture_topology()
            self.capture_session = CaptureSession(
                tp_rank=topology["tp_rank"], tp_size=topology["tp_size"]
            )
            self.capture_graph_manager = None
        return self.capture_session

    def _capture_topology(self) -> dict[str, Any]:
        parallel = getattr(self, "parallel_config", None)
        tp_size = getattr(parallel, "tensor_parallel_size", 1)
        compilation = getattr(self, "compilation_config", None)
        return {
            "tp_rank": getattr(parallel, "rank", 0) % tp_size,
            "tp_size": tp_size,
            "pp_size": getattr(parallel, "pipeline_parallel_size", 1),
            "dp_size": getattr(parallel, "data_parallel_size", 1),
            "pcp_size": getattr(parallel, "prefill_context_parallel_size", 1),
            "dcp_size": getattr(parallel, "decode_context_parallel_size", 1),
            "sequence_parallel": getattr(
                getattr(compilation, "pass_config", None), "enable_sp", False
            ),
            "sequence_parallel_moe": getattr(
                parallel, "use_sequence_parallel_moe", False
            ),
            "expert_parallel": getattr(parallel, "enable_expert_parallel", False),
        }

    def _detach_capture_hooks(self) -> None:
        session = getattr(self, "capture_session", None)
        if session is not None:
            release_capture_graph(self)
            session.detach()
            del self.capture_session

    def _attach_capture_hooks(
        self, model: nn.Module, components: "ModelComponents"
    ) -> None:
        """Register capture components at model load, releasing the previous model.

        On compiled engines the hook bodies
        trace to nothing (torch.compiler.is_compiling guard), so
        ordinary compiled artifacts carry no capture code. Capture
        graphs record a separate raw forward into fixed output buffers;
        other capture-active batches use raw eager execution.

        Stream enablement installs capture hooks after steering hooks, so
        captured router logits retain the post-steering order.
        """
        self._detach_capture_hooks()
        self._capture_session().attach(model, components)

    # ------------------------------------------------------------------
    # Stream API
    # ------------------------------------------------------------------

    def start_capture(self, stream: str, **config_kwargs) -> bool:
        """Enable a capture stream for a discovered model component.

        Args:
            stream: 'hidden_states', 'router_logits', or 'attention_heads'.
            **config_kwargs: StreamConfig options: layers, dtype, select,
                reduce, budget_rows, and budget_bytes. Select uses SelectSpec.to_wire().
        """
        topology = self._capture_topology()
        if any(
            topology[key] != 1 for key in ("pp_size", "dp_size", "pcp_size", "dcp_size")
        ) or any(
            topology[key]
            for key in ("sequence_parallel", "sequence_parallel_moe", "expert_parallel")
        ):
            raise ValueError(
                "Capture supports tensor parallelism with PP=DP=1 and no "
                "context, sequence or expert parallelism."
            )
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
        max_rows: int | None = None,
        row_offset: int = 0,
    ) -> dict[int, dict[str, Any]]:
        """Fetch (and by default clear) captured rows.

        layers restricts to a layer subset; req_ids (client-visible
        request ids) restricts to those requests' rows, with clear
        removing only the emitted rows — the per-request drain that
        bounds peak message size for large corpora.
        """
        return self._capture_session().fetch_stream(
            stream,
            clear=clear,
            layers=layers,
            req_ids=req_ids,
            max_rows=max_rows,
            row_offset=row_offset,
        )

    def release_capture_cache(self) -> bool:
        """Release cached capture graphs and their fixed GPU output buffers."""
        self._capture_session()
        release_capture_graph(self)
        return True

    def clear_captured(self, stream: str) -> bool:
        """Drop captured rows, keeping the stream enabled."""
        self._capture_session().clear_stream(stream)
        return True

    def capture_status(self, stream: str) -> dict[str, Any]:
        status = self._capture_session().stream_status(stream)
        status["graph_ready"] = self.capture_graph_manager is not None
        status["topology"] = self._capture_topology()
        return status
