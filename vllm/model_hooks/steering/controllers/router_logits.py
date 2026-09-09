# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interventions on MoE router logits before expert selection."""

import torch

from vllm.logger import init_logger
from vllm.model_hooks.components.registry import COMPONENTS, ROUTER_LOGITS
from vllm.model_hooks.steering.algorithms import get_algorithm
from vllm.model_hooks.steering.graph.kernels import apply_gate_intervention

from .base import SteeringController

logger = init_logger(__name__)


class RouterLogitsController(SteeringController):
    """Hook-based MoE router-logits steering controller.

    Registered as a forward hook on the MoE block's gate/router
    submodule: the logits are steered in place (via the opaque
    vllm::steer_moe_gate op, a splitting op under compiled execution)
    before top-k expert selection consumes them — architecture-agnostic,
    no per-model forward reimplementation.

    Slot-routed exactly like decoder-layer steering: router-logit rows
    share the token layout of hidden states, so each request's
    moe_router config applies only to its own tokens and distinct MoE
    configs batch together.
    """

    component_id = ROUTER_LOGITS

    def __init__(self) -> None:
        super().__init__()
        # Tier-1 full-graph mode: expert toggle tables read by the
        # captured gate kernel (see init_graph_buffers / the hook below).
        self._graph_mode: bool = False
        self.hook_target = None  # set by the controller manager at attachment
        self.graph_tables: dict[str, torch.Tensor] | None = None
        self.graph_mask: torch.Tensor | None = None
        self.graph_token_rows: torch.Tensor | None = None

    @property
    def output_width(self) -> int | None:
        """Expert-logit width, independent of quantized weight storage layout."""
        for attr in ("output_size", "out_features", "num_total_experts"):
            width = getattr(self.hook_target, attr, None)
            if width is not None:
                return width
        weight = getattr(self.hook_target, "weight", None)
        if weight is not None and weight.ndim == 2:
            return weight.shape[0]
        return None

    def graph_mask_names(self, families: frozenset[str]) -> tuple[str, ...] | None:
        if getattr(self.hook_target, "weight", None) is None:
            logger.warning(
                "moe gate %s exposes no weight; full-graph moe steering "
                "is unavailable on it.",
                self.layer_id,
            )
            return None
        return ("gate",)

    def init_graph_buffers(
        self,
        families,
        masks,
        *,
        capacity,
        hidden_size,
        max_rank,
        dtype,
        device,
        token_rows,
    ) -> None:
        """Allocate the gate kernel's expert toggle tables.

        The expert count comes from the hooked gate module's output width;
        row 0 stays zero so unrouted tokens are exact no-ops.
        """
        weight = getattr(self.hook_target, "weight", None)
        if weight is None:
            raise RuntimeError(
                "moe gate module exposes no weight; cannot size the "
                "full-graph expert toggle tables"
            )
        num_experts = self.output_width
        if num_experts is None:
            raise RuntimeError("moe gate module exposes no output width")
        rows = capacity + 1
        self.graph_tables = {
            "activate": torch.zeros(rows, num_experts, dtype=torch.bool, device=device),
            "deactivate": torch.zeros(
                rows, num_experts, dtype=torch.bool, device=device
            ),
            "epsilon": torch.zeros(rows, dtype=torch.float32, device=device),
            "strength": torch.zeros(rows, dtype=torch.float32, device=device),
            "topk": torch.zeros(rows, dtype=torch.int32, device=device),
            "mode": torch.zeros(rows, dtype=torch.int32, device=device),
        }
        self.graph_mask = masks["gate"]
        self.graph_token_rows = token_rows
        self._graph_mode = True

    def set_graph_row(
        self, row: int, algorithm: str, payload, scale: float, normalize: bool = False
    ) -> None:
        """Write a lowered router intervention, validating expert width once."""
        params = get_algorithm(algorithm).graph_lower(payload, scale)
        tables = self.graph_tables
        num_experts = tables["activate"].shape[1]
        mode = params["mode"]
        soft = mode in ("soft", "soft_topk")
        activate = params["experts"] if soft else params["activate"]
        deactivate = [] if soft else params["deactivate"]
        invalid = [e for e in activate + deactivate if not 0 <= e < num_experts]
        if invalid:
            logger.warning_once(
                "moe_router: expert ids %s are outside [0, %d) for this "
                "model and are ignored.",
                sorted(set(invalid)),
                num_experts,
            )
        if mode == "soft_topk" and params["topk"] > num_experts:
            raise ValueError(
                f"moe_router topk {params['topk']} exceeds expert count {num_experts}"
            )
        self.clear_graph_row(row)
        activate = [e for e in activate if 0 <= e < num_experts]
        deactivate = [e for e in deactivate if 0 <= e < num_experts]
        if not activate and not deactivate:
            return
        tables["activate"][row, activate] = True
        tables["deactivate"][row, deactivate] = True
        if soft:
            tables["strength"][row] = params["strength"]
            tables["topk"][row] = params["topk"]
            tables["mode"][row] = 3 if mode == "soft_topk" else 2
        else:
            tables["epsilon"][row] = params["epsilon"]
            tables["mode"][row] = 1

    def graph_row_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.graph_tables.values()) if self.graph_tables else ()

    def clear_graph_row(self, row: int) -> None:
        for table in self.graph_row_tensors():
            table[row].zero_()

    def process_output_hook(self, module, args, output):
        """Forward-hook entry point on the gate module.

        The logits tensor is mutated in place, so the original output
        structure flows on unchanged.
        """
        if self._op_key is None:
            return None
        adapter = COMPONENTS[ROUTER_LOGITS].adapter
        logits, residual, auxiliary, original_format = adapter.read_output(output)
        if logits is None:
            return None
        if self._graph_mode:
            apply_gate_intervention(
                self.graph_tables,
                self.graph_mask,
                self.graph_token_rows,
                logits,
            )
            return adapter.write_output(
                logits, residual, auxiliary, original_format, output
            )
        torch.ops.vllm.steer_moe_gate(logits, self._op_key)
        return adapter.write_output(
            logits, residual, auxiliary, original_format, output
        )
