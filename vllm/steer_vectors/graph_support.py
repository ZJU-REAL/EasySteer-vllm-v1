# SPDX-License-Identifier: Apache-2.0
"""Full-graph steering support: admissibility and per-slot table state.

`graph_request_problem` is the single admissibility check behind
graph_mode=in_graph — used by the frontend (reject before the engine core),
the worker (defense in depth) and the auto graph-mode resolution for
engine-default configs. `SteerGraphState` owns the Tier-1 buffer
lifecycle: row allocation per slot, table initialization across
controllers, per-step mask zeroing and payload distribution/release.
"""

import torch

from vllm.config import SteerVectorConfig
from vllm.logger import init_logger
from vllm.steer_vectors.request import SteerVectorRequest

logger = init_logger(__name__)


def graph_request_problem(
    request: SteerVectorRequest, max_rank: int
) -> str | None:
    """Why this request cannot run under full-graph steering, or None.

    Capability comes from each algorithm's declared graph_family; rank
    limits from the family schema.
    """
    from vllm.steer_vectors.algorithms import (
        get_algorithm,
        graph_safe_algorithms,
    )
    from vllm.steer_vectors.graph_kernels import GRAPH_FAMILIES

    if request.is_multi_vector:
        return "multi-vector configs"
    if request.algorithm not in graph_safe_algorithms():
        return f"algorithm '{request.algorithm}'"
    if request.algorithm == "moe_router":
        from vllm.steer_vectors.algorithms.moe_router import MoERouterAlgorithm

        if request.local_path:
            return (
                "a file-based moe_router config (only inline "
                "expert_ids/mode configs run under full graphs)"
            )
        mode = MoERouterAlgorithm.validate_mode(request.moe_mode or "activate")
        if mode not in ("activate", "deactivate"):
            return f"moe_router mode '{mode}'"
        return None
    algo_cls = get_algorithm(request.algorithm)
    rank_limited = any(
        "r" in dims
        for dims in GRAPH_FAMILIES.get(algo_cls.graph_family, {}).values()
    )
    if rank_limited:
        rank = algo_cls.wire_rank(request.inline_payload)
        if rank is None:
            return (
                f"a {request.algorithm} payload whose rank could not be "
                f"determined"
            )
        if rank > max_rank:
            return f"payload rank {rank} above steer_graph_max_rank {max_rank}"
    return None


def graph_reject_message(problem: str) -> str:
    """The user-facing rejection for a graph_request_problem result."""
    from vllm.steer_vectors.algorithms import graph_safe_algorithms

    return (
        f"steer graph_mode=in_graph supports single-vector configs of "
        f"{sorted(graph_safe_algorithms())}; got {problem}. Declare the "
        f"workload via steer_algorithms so auto picks the right tier, "
        f"or launch with steer_graph_mode='split' to run this config "
        f"under CUDA graphs."
    )


class SteerGraphState:
    """Tier-1 buffer state: slot -> table row, across all controllers."""

    def __init__(self, config: SteerVectorConfig, device: torch.device):
        self.config = config
        self.device = device
        self.params: tuple | None = None  # (hidden_size, dtype, max_tokens)
        self.controllers: list = []
        self.token_rows_buf: torch.Tensor | None = None
        self.slot_rows: dict[int, int] = {}
        self.slot_algos: dict[int, str] = {}
        self.slot_modules: dict[int, list] = {}
        self.free_rows: list[int] = list(range(config.max_steer_vectors, 0, -1))

    def enable(self, hidden_size: int, dtype: torch.dtype, max_num_tokens: int):
        """Record buffer geometry for full-graph steering (before wrap)."""
        self.params = (hidden_size, dtype, max_num_tokens)

    def init_tables(self, controller_manager) -> None:
        """Allocate Tier-1 buffers on every decoder and gate controller.

        Must run before compilation/graph capture so the captured
        kernels see the final buffer addresses.
        """
        assert self.params is not None, (
            "steer graph_mode=in_graph requires enable() before model wrap"
        )
        from vllm.steer_vectors.controllers import (
            DecoderSteerController,
            MoEGateSteerController,
        )

        hidden_size, dtype, max_num_tokens = self.params
        self.token_rows_buf = torch.zeros(
            max_num_tokens, dtype=torch.long, device=self.device
        )
        capacity = self.config.max_steer_vectors
        max_rank = self.config.graph_max_rank
        gates = 0
        for module in controller_manager.modules.values():
            if isinstance(module, DecoderSteerController):
                module.init_graph_table(
                    capacity,
                    hidden_size,
                    dtype,
                    self.device,
                    max_num_tokens,
                    self.token_rows_buf,
                    max_rank,
                )
                self.controllers.append(module)
            elif isinstance(module, MoEGateSteerController):
                if getattr(module.hook_target, "weight", None) is None:
                    # Fused/weightless gates cannot be hooked anyway
                    # (see moe_gate_is_fused); booting must not fail.
                    logger.warning(
                        "moe gate %s exposes no weight; full-graph moe "
                        "steering is unavailable on it.",
                        module.layer_id,
                    )
                    continue
                module.init_graph_table(
                    capacity, dtype, self.device, max_num_tokens,
                    self.token_rows_buf,
                )
                self.controllers.append(module)
                gates += 1
        logger.info(
            "Full-graph steering buffers allocated on %d controllers "
            "(%d gates; %d rows, hidden %d, rank %d, %d max tokens)",
            len(self.controllers),
            gates,
            capacity,
            hidden_size,
            max_rank,
            max_num_tokens,
        )

    def zero_step_masks(self) -> None:
        for module in self.controllers:
            module.zero_step_masks()

    def row_of(self, slot: int) -> int:
        return self.slot_rows.get(slot, 0)

    def _allocate_row(self) -> int:
        if not self.free_rows:
            raise RuntimeError(
                f"No free steering rows (capacity "
                f"{self.config.max_steer_vectors})."
            )
        return self.free_rows.pop()

    def distribute(
        self,
        slot: int,
        request: SteerVectorRequest,
        layer_payloads: dict,
        controller_manager,
    ) -> None:
        """Write one config's lowered payloads into a fresh table row."""
        from vllm.steer_vectors.algorithms.moe_router import MoERouterAlgorithm

        is_moe = request.algorithm == "moe_router"
        controller_type = "moe_layer" if is_moe else "decoder_layer"
        target_layers = request.target_layers
        row = self._allocate_row()
        modules: list = []
        for layer_idx, payload in (layer_payloads or {}).items():
            if not is_moe and target_layers and layer_idx not in target_layers:
                continue
            for module in controller_manager._get_modules_for_layer(
                layer_idx, controller_type
            ):
                if is_moe:
                    module.set_graph_toggles(
                        row, MoERouterAlgorithm.graph_lower(payload, request.scale)
                    )
                else:
                    module.set_graph_row(
                        row,
                        request.algorithm,
                        payload,
                        request.scale,
                        normalize=bool(request.normalize),
                    )
                modules.append(module)
        self.slot_rows[slot] = row
        self.slot_algos[slot] = request.algorithm
        self.slot_modules[slot] = modules

    def release(self, slot: int) -> None:
        row = self.slot_rows.pop(slot, None)
        if row is None:
            return
        self.slot_algos.pop(slot, None)
        for module in self.slot_modules.pop(slot, []):
            module.clear_graph_row(row)
        self.free_rows.append(row)
