# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-level manager for steer vectors in vLLM V1.

Maps live requests to refcounted config slots; full-graph buffer state
and admissibility live in graph.state.SteeringGraphState."""

from dataclasses import dataclass

import torch

from vllm.config import SteerVectorConfig
from vllm.logger import init_logger
from vllm.model_hooks.steering.capabilities import algorithm_target, declared_components
from vllm.model_hooks.steering.controllers.manager import ControllerManager
from vllm.model_hooks.steering.graph.policy import (
    graph_reject_message,
    graph_request_problem,
)
from vllm.model_hooks.steering.graph.state import SteeringGraphState
from vllm.model_hooks.steering.payload_cache import PayloadCache
from vllm.model_hooks.steering.request import (
    ResolvedVector,
    SteeringRequest,
    config_fingerprint,
    steer_params_dict,
)

logger = init_logger(__name__)


@dataclass
class _ConfigSlot:
    slot: int
    refcount: int
    request: SteeringRequest


class WorkerSteeringState:
    """Worker-side owner of steering state.

    Maps live requests to refcounted config slots (payload loading and
    layer distribution happen at admission, never in the forward pass)
    and owns the payload cache and the full-graph buffers.
    """

    def __init__(
        self,
        device: torch.device,
        steer_vector_config: SteerVectorConfig,
        *,
        hidden_size: int,
    ):
        self._controller_manager: ControllerManager | None = None
        self.steer_vector_config = steer_vector_config
        self.device = device
        self.payload_cache = PayloadCache(
            str(device), steer_vector_config, hidden_size=hidden_size
        )
        self._config_slots: dict[str, _ConfigSlot] = {}
        self._slot_clauses: dict[int, list[dict | None]] = {}
        self._slot_groups: dict[int, tuple[tuple, ...]] = {}
        self._req_fingerprints: dict[str, str] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        # Tier-1 full-graph mode state (see graph.state).
        self._graph_full = steer_vector_config.graph_mode == "in_graph"
        self.graph_state = SteeringGraphState(steer_vector_config, device)

    def close(self) -> None:
        """Release model hooks, cached tensors, graph buffers and request routing."""
        if self._controller_manager is not None:
            self._controller_manager.remove_hooks()
            self._controller_manager = None
        self.graph_state.close()
        self.payload_cache.clear()
        self._config_slots.clear()
        self._slot_clauses.clear()
        self._slot_groups.clear()
        self._req_fingerprints.clear()
        self._free_slots.clear()
        self._next_slot = 0

    def enable_graph_mode(
        self, hidden_size: int, dtype: torch.dtype, max_num_tokens: int
    ) -> None:
        """Record full-graph buffer geometry before attaching hooks."""
        self.graph_state.enable(hidden_size, dtype, max_num_tokens)

    @property
    def token_rows_buf(self):
        return self.graph_state.token_rows_buf

    @property
    def graph_masks_buf(self):
        return self.graph_state.step_masks

    def zero_graph_masks(self, num_tokens: int | None = None) -> None:
        self.graph_state.zero_step_masks(num_tokens)

    def graph_batch_entries(self) -> dict[int, tuple]:
        """slot -> (row, request, controllers) for all live graph configs."""
        controllers = self.graph_state.slot_controllers
        return {
            entry.slot: (
                entry.slot + 1,
                entry.request,
                controllers[entry.slot],
            )
            for entry in self._config_slots.values()
            if entry.slot in controllers
        }

    def _assert_graph_safe(self, request: SteeringRequest) -> None:
        problem = graph_request_problem(
            request, self.steer_vector_config.graph_max_rank
        )
        if problem is not None:
            raise ValueError(graph_reject_message(problem))

    def preload_vectors(self, payloads: list[dict]) -> None:
        for payload in payloads:
            self.payload_cache.preload(payload)

    def preload_request(self, request: SteeringRequest) -> None:
        """Warm payload storage without allocating a configuration slot."""
        for vector in request.vectors:
            self.payload_cache.preload(
                vector.payload,
                target_layers=vector.target_layers,
            )

    def acquire_config(
        self,
        req_id: str,
        request: SteeringRequest,
    ) -> int:
        """Register a live request's steering config; returns its slot.

        All config kinds route per-request: single-vector, multi-vector
        and moe_router configs steer only their own requests' tokens.
        """
        if self._graph_full:
            self._assert_graph_safe(request)

        fp = config_fingerprint(request)
        entry = self._config_slots.get(fp)
        if entry is not None:
            entry.refcount += 1
            self._req_fingerprints[req_id] = fp
            return entry.slot

        capacity = self.steer_vector_config.max_steer_vectors
        assert capacity is not None
        if len(self._config_slots) >= capacity:
            # The scheduler defers requests when all slots are taken
            # (concurrent distinct configurations are a scheduling
            # constraint, like max_loras); reaching here means that
            # accounting drifted from the worker's slot keying.
            raise RuntimeError(
                f"steering slot capacity exceeded: {len(self._config_slots)} "
                f"distinct configurations live, max_steer_vectors="
                f"{capacity}. The scheduler should have deferred this "
                f"request; please report this as a bug."
            )

        # Commit the slot only after installation succeeds. Failed admission
        # must not consume capacity or leave partial eager interventions behind.
        slot = self._free_slots[-1] if self._free_slots else self._next_slot
        try:
            if self._graph_full:
                assert self._controller_manager is not None
                vector = request.vectors[0]
                payload = self._materialize_vector(vector)
                self.graph_state.distribute(
                    slot, vector, payload, self._controller_manager
                )
            else:
                self._distribute_config(slot, request)
        except Exception:
            if not self._graph_full and self._controller_manager is not None:
                for controller in self._controller_manager.controllers.values():
                    controller.clear_slot(slot)
            raise
        if self._free_slots:
            self._free_slots.pop()
        else:
            self._next_slot += 1
        self._config_slots[fp] = _ConfigSlot(slot, 1, request)
        self._slot_clauses[slot] = [vector.apply_spec for vector in request.vectors]
        from vllm.model_hooks.steering.validation import request_position_groups

        self._slot_groups[slot] = request_position_groups(request)
        self._req_fingerprints[req_id] = fp
        logger.debug("Configured steering slot %d for %s", slot, fp)
        return slot

    def release_config(self, req_id: str) -> None:
        fp = self._req_fingerprints.pop(req_id, None)
        if fp is None:
            return
        entry = self._config_slots.get(fp)
        if entry is None:
            return
        entry.refcount -= 1
        if entry.refcount > 0:
            return
        slot = entry.slot
        del self._config_slots[fp]
        del self._slot_clauses[slot]
        del self._slot_groups[slot]
        if slot in self.graph_state.slot_controllers:
            self.graph_state.release(slot)
        elif self._controller_manager is not None:
            for controller in self._controller_manager.controllers.values():
                controller.clear_slot(slot)
        self._free_slots.append(slot)

    def slot_for_request(self, req_id: str) -> int | None:
        fp = self._req_fingerprints.get(req_id)
        if fp is None:
            return None
        entry = self._config_slots.get(fp)
        return None if entry is None else entry.slot

    def slot_clauses(self) -> dict[int, list[dict | None]]:
        """slot -> ordered where-clauses of its live interventions.

        Maintained with config admission/release rather than rebuilding
        a dictionary from all live requests on every forward step.
        """
        return self._slot_clauses

    def slot_groups(self) -> dict[int, tuple[tuple, ...]]:
        return self._slot_groups

    def _distribute_config(self, slot: int, request: SteeringRequest) -> None:
        """Configure one slot with the request's ordered interventions."""
        specs = []
        for vector in request.vectors:
            payload = self._materialize_vector(vector)
            specs.append((steer_params_dict(vector), payload))
        self._configure_layer_slots(slot, specs, request.conflict_resolution)

    def _materialize_vector(self, vector: ResolvedVector) -> dict:
        return self.payload_cache.get(
            vector.payload, target_layers=vector.target_layers
        )

    def _configure_layer_slots(
        self,
        slot: int,
        specs: list,
        conflict_resolution: str,
    ) -> None:
        """Write an ordered intervention list into each targeted layer."""
        assert self._controller_manager is not None
        layer_ids: set[int] = set()
        for fields, payloads in specs:
            tl = fields.get("target_layers")
            layer_ids.update(
                layer_idx for layer_idx in payloads if not tl or layer_idx in tl
            )
        for layer_idx in sorted(layer_ids):
            component_specs: dict[str, list] = {}
            for fields, payloads in specs:
                tl = fields.get("target_layers")
                if tl and layer_idx not in tl:
                    continue
                payload = payloads.get(layer_idx)
                if payload is None:
                    continue
                target = algorithm_target(fields["algorithm"])
                component_specs.setdefault(target, []).append(
                    {**fields, "payload": payload}
                )
            for target, layer_specs in component_specs.items():
                for controller in self._controller_manager.controllers_for_layer(
                    layer_idx, target
                ):
                    controller.configure_slot(slot, layer_specs, conflict_resolution)

    def list_configs(self) -> set[int]:
        """Int ids of all live request steering configs."""
        return {
            entry.request.steer_vector_int_id for entry in self._config_slots.values()
        }

    def model_info(self) -> dict[str, dict[int, int | None]]:
        """Available targets and their widths for model-aware request admission."""
        from vllm.model_hooks.components.registry import HIDDEN_STATES

        if self._controller_manager is None:
            return {}
        info: dict[str, dict[int, int | None]] = {}
        for controller in self._controller_manager.controllers.values():
            if self._graph_full and (
                not controller._graph_mode or not controller.graph_tables
            ):
                continue
            if controller.component_id == HIDDEN_STATES:
                width = self.payload_cache.hidden_size
            else:
                width = controller.output_width
            assert controller.layer_id is not None
            info.setdefault(controller.component_id, {})[controller.layer_id] = width
        return info

    def attach_steering_hooks(self, components) -> None:
        """Attach only declared targets; capture retains the complete directory."""
        if self._controller_manager is not None:
            self.close()
        targets = declared_components(self.steer_vector_config.algorithms)
        try:
            self._controller_manager = ControllerManager(
                {kind: layers for kind, layers in components.items() if kind in targets}
            )
            if self._graph_full:
                self.graph_state.init_tables(self._controller_manager)
        except Exception:
            self.close()
            raise
