# SPDX-License-Identifier: Apache-2.0

"""Worker-level manager for steer vectors in vLLM V1.

Maps live requests to refcounted config slots; full-graph buffer state
and admissibility live in graph_support.SteerGraphState."""

from typing import Any

import torch

from vllm.config import SteerVectorConfig
from vllm.logger import init_logger
from vllm.steer_vectors.controller_manager import (
    LoadedSteerVector,
    SteerControllerManager,
    create_steer_controller_manager,
)
from vllm.steer_vectors.graph_support import (
    SteerGraphState,
    graph_reject_message,
    graph_request_problem,
)
from vllm.steer_vectors.request import (
    STEER_APPLY_FIELDS,
    STEER_MOE_FIELDS,
    SteerVectorRequest,
    steer_params_dict,
)

logger = init_logger(__name__)


def config_fingerprint(request: SteerVectorRequest) -> str:
    """Stable identity of a steering *configuration* (not just the vector).

    Requests with the same fingerprint share one layer slot; the vector
    payload itself is deduplicated separately by the VectorStore. Built
    from the canonical field registry so new parameters participate
    automatically.
    """

    def _canon_value(value):
        if isinstance(value, list):
            return tuple(_canon_value(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((k, _canon_value(v)) for k, v in value.items()))
        return value

    def _canon(obj, fields):
        return [_canon_value(getattr(obj, name)) for name in fields]

    from vllm.steer_vectors.store import file_version

    # The file version participates so a vector regenerated at the same
    # path gets a fresh slot (and thus a fresh load) even while requests
    # with the old version are still running.
    values = [request.local_path, request.payload_sha256]
    if request.local_path:
        values.append(file_version(request.local_path))
    values.extend(_canon(request, STEER_APPLY_FIELDS))
    if request.is_multi_vector:
        values.append(request.conflict_resolution)
        for vc in request.vector_configs:
            values.append(vc.path)
            values.append(vc.payload_sha256)
            if vc.path:
                values.append(file_version(vc.path))
            values.extend(_canon(vc, STEER_APPLY_FIELDS))
    if request.algorithm == "moe_router":
        values.extend(_canon(request, STEER_MOE_FIELDS))
    return repr(tuple(values))


class WorkerSteerVectorManager:
    """Worker-side owner of steering state.

    Maps live requests to refcounted config slots (payload loading and
    layer distribution happen at admission, never in the forward pass)
    and owns the vector store, the server-level default config, and the
    Tier-1 full-graph buffers.
    """

    _manager_cls: type[SteerControllerManager] = SteerControllerManager

    def __init__(
        self,
        device: torch.device,
        steer_vector_config: SteerVectorConfig,
        loaded_vector_cls: type[LoadedSteerVector] = LoadedSteerVector,
    ):
        self._controller_manager: SteerControllerManager | None = None
        self._loaded_vector_cls = loaded_vector_cls
        self.steer_vector_config = steer_vector_config
        self.device = device
        from vllm.steer_vectors.store import VectorStore

        self.vector_store = VectorStore(str(device), steer_vector_config)
        # Per-request config routing state: fingerprint -> [slot, refcount,
        # request]; req_id -> fingerprint.
        self._config_slots: dict[str, list] = {}
        self._req_fingerprints: dict[str, str] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        # Tier-1 full-graph mode state (see graph_support).
        self._graph_full = steer_vector_config.graph_mode == "in_graph"
        self.graph_state = SteerGraphState(steer_vector_config, device)

    def enable_graph_mode(
        self, hidden_size: int, dtype: torch.dtype, max_num_tokens: int
    ) -> None:
        """Record buffer geometry for full-graph steering (before wrap)."""
        self.graph_state.enable(hidden_size, dtype, max_num_tokens)

    def _init_graph_tables(self) -> None:
        assert self._controller_manager is not None
        self.graph_state.init_tables(self._controller_manager)

    @property
    def token_rows_buf(self):
        return self.graph_state.token_rows_buf

    def zero_graph_masks(self) -> None:
        self.graph_state.zero_step_masks()

    def graph_row_of(self, slot: int) -> int:
        return self.graph_state.row_of(slot)

    def graph_batch_entries(self) -> dict[int, tuple]:
        """slot -> (row, request, controllers) for all live graph configs."""
        rows = self.graph_state.slot_rows
        return {
            entry[0]: (
                rows[entry[0]],
                entry[2],
                self.graph_state.slot_modules[entry[0]],
            )
            for entry in self._config_slots.values()
            if entry[0] in rows
        }

    def _assert_graph_safe(self, request: SteerVectorRequest) -> None:
        problem = graph_request_problem(
            request, self.steer_vector_config.graph_max_rank
        )
        if problem is not None:
            raise ValueError(graph_reject_message(problem))

    def _distribute_graph_config(
        self, slot: int, model: LoadedSteerVector, request: SteerVectorRequest
    ) -> None:
        assert self._controller_manager is not None
        self.graph_state.distribute(
            slot, request, model.layer_payloads, self._controller_manager
        )

    def _distribute_graph_moe(self, slot: int, request: SteerVectorRequest) -> None:
        """MoE payloads are built here (file/params paths), then lowered
        per layer by MoERouterAlgorithm.graph_lower inside distribute."""
        assert self._controller_manager is not None
        model = self._build_moe_model(request)
        self.graph_state.distribute(
            slot, request, model.layer_payloads, self._controller_manager
        )


    # ------------------------------------------------------------------
    # Per-request config routing (Phase C)
    # ------------------------------------------------------------------

    def preload_vectors(self, paths: list[str], algorithm: str = "direct"):
        for path in paths:
            self.vector_store.preload(path, algorithm)

    def acquire_config(self, req_id: str, request: SteerVectorRequest) -> int:
        """Register a live request's steering config; returns its slot.

        All config kinds route per-request: single-vector, multi-vector
        and moe_router configs steer only their own requests' tokens.
        """
        if self._graph_full:
            self._assert_graph_safe(request)

        fp = config_fingerprint(request)
        entry = self._config_slots.get(fp)
        if entry is not None:
            entry[1] += 1
            self._req_fingerprints[req_id] = fp
            return entry[0]

        capacity = self.steer_vector_config.max_steer_vectors
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

        slot = self._free_slots.pop() if self._free_slots else self._next_slot
        if slot == self._next_slot:
            self._next_slot += 1

        if request.is_multi_vector:
            self._distribute_multi_config(slot, request)
        elif request.algorithm == "moe_router":
            if self._graph_full:
                self._distribute_graph_moe(slot, request)
            else:
                self._distribute_moe_slot(slot, request)
        else:
            model = self._load_entry(
                request.local_path,
                request.algorithm,
                request.target_layers,
                request.inline_payload,
            )
            if self._graph_full:
                self._distribute_graph_config(slot, model, request)
            else:
                self._distribute_config(slot, model, request)
        self._config_slots[fp] = [slot, 1, request]
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
        entry[1] -= 1
        if entry[1] > 0:
            return
        slot, _, request = entry
        del self._config_slots[fp]
        if slot in self.graph_state.slot_rows:
            self.graph_state.release(slot)
        elif self._controller_manager is not None:
            for module in self._controller_manager.modules.values():
                module.reset_steer_vector(slot)
        self._free_slots.append(slot)

    def slot_for_request(self, req_id: str) -> int | None:
        fp = self._req_fingerprints.get(req_id)
        if fp is None:
            return None
        entry = self._config_slots.get(fp)
        return None if entry is None else entry[0]

    def slot_clauses(self) -> dict[int, list[dict | None]]:
        """slot -> ordered where-clauses of its live interventions.

        Input of the per-step position resolver; derived from the live
        request registry so it can never drift from slot routing.
        """
        clauses: dict[int, list[dict | None]] = {}
        for slot, _, request in self._config_slots.values():
            if request.is_multi_vector:
                clauses[slot] = [vc.apply_spec for vc in request.vector_configs]
            else:
                clauses[slot] = [request.apply_spec]
        return clauses

    def _distribute_config(
        self, slot: int, model: LoadedSteerVector, request: SteerVectorRequest
    ) -> None:
        """Configure a single-vector request as a one-intervention slot."""
        fields = steer_params_dict(request)
        self._configure_layer_slots(
            slot, [(fields, model.layer_payloads or {})], "priority"
        )

    def _distribute_multi_config(self, slot: int, request: SteerVectorRequest) -> None:
        """Configure a multi-vector request's sub-vectors on one slot.

        Sub-vector payloads are deduplicated through the VectorStore; each
        layer receives the specs of the vectors that target it.
        """
        specs = []
        for vc in request.vector_configs:
            model = self._load_entry(
                vc.path, vc.algorithm, vc.target_layers, vc.inline_payload
            )
            fields = steer_params_dict(vc)
            specs.append((fields, model.layer_payloads or {}))
        self._configure_layer_slots(slot, specs, request.conflict_resolution)

    def _load_entry(
        self,
        path: str,
        algorithm: str,
        target_layers: list[int] | None,
        inline_payload: dict | None,
    ) -> LoadedSteerVector:
        """Resolve a vector source (file path or inline payload)."""
        if inline_payload is not None:
            return self.vector_store.get_inline(
                inline_payload, algorithm, target_layers=target_layers
            )
        return self.vector_store.get(
            path, algorithm, target_layers=target_layers, lazy=True
        )

    def _configure_layer_slots(
        self,
        slot: int,
        specs: list,
        conflict_resolution: str,
        controller_type: str = "decoder_layer",
    ) -> None:
        """Write an ordered intervention list into each targeted layer."""
        assert self._controller_manager is not None
        layer_ids: set[int] = set()
        for fields, payloads in specs:
            tl = fields.get("target_layers")
            layer_ids.update(
                layer_idx for layer_idx in payloads if not tl or layer_idx in tl
            )
        configured = 0
        for layer_idx in sorted(layer_ids):
            layer_specs = []
            for fields, payloads in specs:
                tl = fields.get("target_layers")
                if tl and layer_idx not in tl:
                    continue
                payload = payloads.get(layer_idx)
                if payload is None:
                    continue
                layer_specs.append({**fields, "payload": payload})
            if not layer_specs:
                continue
            for module in self._controller_manager._get_modules_for_layer(
                layer_idx, controller_type
            ):
                module.configure_slot(slot, layer_specs, conflict_resolution)
                configured += 1
        if configured == 0:
            payload_layers = sorted({l for _, p in specs for l in p})
            targets = [fields.get("target_layers") for fields, _ in specs]
            raise ValueError(
                "steering config targets no modules — the request would "
                f"steer nothing (vector payload layers {payload_layers}, "
                f"target_layers {targets}, controller {controller_type!r})"
            )

    def _build_moe_model(self, request: SteerVectorRequest) -> LoadedSteerVector:
        if not request.local_path:
            if request.moe_expert_ids is None:
                raise ValueError(
                    "moe_router algorithm requires moe_expert_ids when no "
                    "config file path is given"
                )
            layer_payloads = {}
            moe_mode = request.moe_mode or "activate"
            for layer_id in request.target_layers or []:
                payload = {
                    "expert_ids": request.moe_expert_ids,
                    "mode": moe_mode,
                }
                if moe_mode == "soft":
                    payload["lambda"] = request.moe_lambda
                layer_payloads[layer_id] = payload
            return self._loaded_vector_cls(
                steer_vector_id=request.steer_vector_id,
                layer_payloads=layer_payloads,
                scale_factor=1.0,
                algorithm="moe_router",
            )
        return self._loaded_vector_cls.from_local_checkpoint(
            steer_vector_model_path=request.local_path,
            steer_vector_id=request.steer_vector_id,
            config=self.steer_vector_config,
            device=str(self.device),
            scale_factor=request.scale,
            algorithm="moe_router",
            target_layers=request.target_layers,
            moe_mode=request.moe_mode,
            moe_lambda=request.moe_lambda,
            moe_topk=request.moe_topk,
        )

    def _distribute_moe_slot(self, slot: int, request: SteerVectorRequest) -> None:
        """Configure a moe_router request as a one-intervention slot on
        the MoE gate controllers (token-routed like any other config)."""
        model = self._build_moe_model(request)
        fields = steer_params_dict(request)
        self._configure_layer_slots(
            slot,
            [(fields, model.layer_payloads or {})],
            "priority",
            controller_type="moe_layer",
        )

    # ------------------------------------------------------------------
    # Server-level (default) steering config
    # ------------------------------------------------------------------

    _SERVER_REQ_ID = "__server__"

    def set_server_config(self, request: SteerVectorRequest) -> bool:
        """Install (or replace) the server-level default steering config.

        The config occupies a normal routing slot; requests without their
        own steer_vector_request are routed to it via the default slot.
        """
        self.clear_server_config()
        slot = self.acquire_config(self._SERVER_REQ_ID, request)
        logger.info("Server-level steering installed on slot %d", slot)
        return True

    def clear_server_config(self) -> bool:
        had = self._SERVER_REQ_ID in self._req_fingerprints
        self.release_config(self._SERVER_REQ_ID)
        return had

    @property
    def server_slot(self) -> int:
        slot = self.slot_for_request(self._SERVER_REQ_ID)
        return -1 if slot is None else slot

    def list_configs(self) -> set[int]:
        """Int ids of all live steering configs (including the server's)."""
        return {entry[2].steer_vector_int_id for entry in self._config_slots.values()}

    @property
    def is_enabled(self) -> bool:
        return True

    def create_steer_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        """Create and initialize the steer vector manager for the model."""
        steer_vector_manager = create_steer_controller_manager(
            model,
            steer_vector_config=self.steer_vector_config,
            steer_vector_manager_cls=self._manager_cls,
        )
        self._controller_manager = steer_vector_manager
        if self._graph_full:
            self._init_graph_tables()
        return steer_vector_manager.model
