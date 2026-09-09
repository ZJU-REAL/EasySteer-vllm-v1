# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker buffer allocation, payload distribution, and graph slot lifecycle."""

import torch

from vllm.config import SteerVectorConfig
from vllm.logger import init_logger
from vllm.model_hooks.steering.request import ResolvedVector

from .policy import declared_graph_families, declared_graph_gate

logger = init_logger(__name__)


class SteeringGraphState:
    """Tier-1 buffers use config slot + 1; row zero means no intervention."""

    def __init__(self, config: SteerVectorConfig, device: torch.device):
        self.config = config
        self.device = device
        self.params: tuple | None = None  # (hidden_size, dtype, max_tokens)
        self.controllers: list = []
        self.token_rows_buf: torch.Tensor | None = None
        self.step_masks: torch.Tensor | None = None
        self.slot_controllers: dict[int, list] = {}

    def enable(self, hidden_size: int, dtype: torch.dtype, max_num_tokens: int):
        """Record buffer geometry for full-graph steering (before wrap)."""
        self.params = (hidden_size, dtype, max_num_tokens)

    def close(self) -> None:
        """Release tensor and controller ownership before shutdown or reattachment."""
        self.controllers.clear()
        self.token_rows_buf = None
        self.step_masks = None
        self.slot_controllers.clear()

    def init_tables(self, controller_manager) -> None:
        """Allocate Tier-1 buffers on every decoder and gate controller.

        Must run before compilation/graph capture so the captured
        kernels see the final buffer addresses.
        """
        assert self.params is not None, (
            "steer graph_mode=in_graph requires enable() before model wrap"
        )
        hidden_size, dtype, max_num_tokens = self.params
        self.token_rows_buf = torch.zeros(
            max_num_tokens, dtype=torch.long, device=self.device
        )
        families = declared_graph_families(self.config.algorithms)
        if declared_graph_gate(self.config.algorithms):
            families = families | {"moe_gate"}
        plans = [
            (controller, controller.graph_mask_names(families))
            for controller in controller_manager.controllers.values()
        ]
        num_masks = sum(len(names) for _, names in plans if names is not None)
        self.step_masks = torch.zeros(
            (num_masks, max_num_tokens), dtype=dtype, device=self.device
        )
        mask_rows = iter(self.step_masks.unbind(0))
        for controller, names in plans:
            if names is None:
                continue
            controller.init_graph_buffers(
                families,
                {name: next(mask_rows) for name in names},
                capacity=self.config.max_steer_vectors,
                hidden_size=hidden_size,
                max_rank=self.config.graph_max_rank,
                dtype=dtype,
                device=self.device,
                token_rows=self.token_rows_buf,
            )
            if names:
                self.controllers.append(controller)
        logger.info(
            "Full-graph steering buffers allocated on %d controllers "
            "(families %s; %d rows, hidden %d, rank %d, %d max tokens)",
            len(self.controllers),
            sorted(families),
            self.config.max_steer_vectors,
            hidden_size,
            self.config.graph_max_rank,
            max_num_tokens,
        )

    def zero_step_masks(self, num_tokens: int | None = None) -> None:
        if self.step_masks is not None and self.step_masks.shape[0]:
            self.step_masks[:, :num_tokens].zero_()

    def row_of(self, slot: int) -> int:
        return slot + 1 if slot in self.slot_controllers else 0

    def distribute(
        self,
        slot: int,
        vector: ResolvedVector,
        layer_payloads: dict,
        controller_manager,
    ) -> None:
        """Write one config's lowered payloads into its reserved table row."""
        from vllm.model_hooks.components.registry import get_component
        from vllm.model_hooks.steering.capabilities import algorithm_target

        component = get_component(algorithm_target(vector.algorithm))
        target_layers = vector.target_layers
        targets: list = []
        for layer_idx, payload in (layer_payloads or {}).items():
            if target_layers and layer_idx not in target_layers:
                continue
            for controller in controller_manager.controllers_for_layer(
                layer_idx, component.id
            ):
                if not controller._graph_mode or not controller.graph_tables:
                    raise ValueError(
                        f"steering component {component.id!r} at layer "
                        f"{layer_idx} has no in-graph buffers"
                    )
                targets.append((controller, payload))
        if not targets:
            # Admission validated the global model inventory. This PP stage may
            # own none of the request's target layers and needs no graph row.
            return
        assert self.config.max_steer_vectors is not None
        assert 0 <= slot < self.config.max_steer_vectors
        row = slot + 1
        self.slot_controllers[slot] = [controller for controller, _ in targets]
        try:
            for controller, payload in targets:
                controller.set_graph_row(
                    row,
                    vector.algorithm,
                    payload,
                    vector.scale,
                    normalize=bool(vector.normalize),
                )
        except Exception:
            # Undo a partially written row; the original admission error propagates.
            self.release(slot)
            raise

    def release(self, slot: int) -> None:
        for controller in self.slot_controllers.pop(slot, []):
            controller.clear_graph_row(slot + 1)
