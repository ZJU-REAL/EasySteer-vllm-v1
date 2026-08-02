# SPDX-License-Identifier: Apache-2.0

"""Worker-level manager for steer vectors in vLLM V1."""

from typing import Any

import torch

from vllm.config import SteerVectorConfig
from vllm.logger import init_logger
from vllm.steer_vectors.models import (
    LoadedSteerVector,
    SteerControllerManager,
    create_steer_controller_manager,
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
    values = [request.local_path, request.debug]
    if request.local_path:
        values.append(file_version(request.local_path))
    values.extend(_canon(request, STEER_APPLY_FIELDS))
    if request.is_multi_vector:
        values.append(request.conflict_resolution)
        for vc in request.vector_configs:
            values.append(vc.path)
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
        # Tier-1 full-graph mode state (enable_graph_mode + wrap).
        self._graph_full = steer_vector_config.graph_mode == "full"
        self._graph_params: tuple | None = None
        self._graph_controllers: list = []
        self.row_tok_buf: torch.Tensor | None = None
        self._slot_rows: dict[int, int] = {}
        self._slot_graph_modules: dict[int, list] = {}
        self._free_rows: list[int] = list(
            range(steer_vector_config.max_steer_vectors, 0, -1)
        )

    def enable_graph_mode(
        self, hidden_size: int, dtype: torch.dtype, max_num_tokens: int
    ) -> None:
        """Record buffer geometry for full-graph steering (before wrap)."""
        self._graph_params = (hidden_size, dtype, max_num_tokens)

    def _init_graph_tables(self) -> None:
        """Allocate Tier-1 buffers on every decoder controller."""
        assert self._controller_manager is not None
        assert self._graph_params is not None, (
            "steer graph_mode=full requires enable_graph_mode() before model wrap"
        )
        from vllm.steer_vectors.layers import DecoderLayerWithSteerVector

        hidden_size, dtype, max_num_tokens = self._graph_params
        self.row_tok_buf = torch.zeros(
            max_num_tokens, dtype=torch.long, device=self.device
        )
        capacity = self.steer_vector_config.max_steer_vectors
        for module in self._controller_manager.modules.values():
            if isinstance(module, DecoderLayerWithSteerVector):
                module.init_graph_table(
                    capacity,
                    hidden_size,
                    dtype,
                    self.device,
                    max_num_tokens,
                    self.row_tok_buf,
                )
                self._graph_controllers.append(module)
        logger.info(
            "Full-graph steering buffers allocated on %d layers "
            "(%d rows, hidden %d, %d max tokens)",
            len(self._graph_controllers),
            capacity,
            hidden_size,
            max_num_tokens,
        )

    def zero_graph_masks(self) -> None:
        for module in self._graph_controllers:
            module.graph_mask.zero_()

    def graph_row_of(self, slot: int) -> int:
        return self._slot_rows.get(slot, 0)

    def graph_batch_entries(self) -> dict[int, tuple]:
        """slot -> (row, request, controllers) for all live graph configs."""
        return {
            entry[0]: (
                self._slot_rows[entry[0]],
                entry[2],
                self._slot_graph_modules[entry[0]],
            )
            for entry in self._config_slots.values()
            if entry[0] in self._slot_rows
        }

    def _assert_graph_safe(self, request: SteerVectorRequest) -> None:
        problem = None
        if request.is_multi_vector:
            problem = "multi-vector configs"
        elif request.algorithm != "direct":
            problem = f"algorithm '{request.algorithm}'"
        elif request.normalize:
            problem = "normalize=True"
        if problem is not None:
            raise ValueError(
                f"steer graph_mode=full supports only graph-safe configs "
                f"(direct algorithm, no normalize, single vector); got "
                f"{problem}. Use --steer-graph-mode=piecewise."
            )

    def _distribute_graph_config(
        self, slot: int, model: LoadedSteerVector, request: SteerVectorRequest
    ) -> None:
        """Write a config's scaled payload into per-layer vector tables."""
        assert self._controller_manager is not None
        if not self._free_rows:
            raise RuntimeError(
                f"No free steering rows (capacity "
                f"{self.steer_vector_config.max_steer_vectors})."
            )
        row = self._free_rows.pop()
        target_layers = request.target_layers
        modules: list = []
        for layer_idx, payload in (model.layer_payloads or {}).items():
            if target_layers and layer_idx not in target_layers:
                continue
            for module in self._controller_manager._get_modules_for_layer(
                layer_idx, "decoder_layer"
            ):
                module.set_graph_row(row, payload * request.scale)
                modules.append(module)
        self._slot_rows[slot] = row
        self._slot_graph_modules[slot] = modules

    def _release_graph_config(self, slot: int) -> None:
        row = self._slot_rows.pop(slot, None)
        if row is None:
            return
        for module in self._slot_graph_modules.pop(slot, []):
            module.clear_graph_row(row)
        self._free_rows.append(row)

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

        slot = self._free_slots.pop() if self._free_slots else self._next_slot
        if slot == self._next_slot:
            self._next_slot += 1

        if request.is_multi_vector:
            self._distribute_multi_config(slot, request)
        elif request.algorithm == "moe_router":
            self._distribute_moe_slot(slot, request)
        elif self._graph_full:
            model = self.vector_store.get(
                request.local_path, request.algorithm, lazy=True
            )
            self._distribute_graph_config(slot, model, request)
        else:
            model = self.vector_store.get(
                request.local_path, request.algorithm, lazy=True
            )
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
        if slot in self._slot_rows:
            self._release_graph_config(slot)
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

    def _distribute_config(
        self, slot: int, model: LoadedSteerVector, request: SteerVectorRequest
    ) -> None:
        """Configure a single-vector request as a one-intervention slot."""
        fields = {**steer_params_dict(request), "debug": request.debug}
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
            model = self.vector_store.get(vc.path, vc.algorithm, lazy=True)
            fields = {**steer_params_dict(vc), "debug": request.debug}
            specs.append((fields, model.layer_payloads or {}))
        self._configure_layer_slots(slot, specs, request.conflict_resolution)

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
        fields = {**steer_params_dict(request), "debug": request.debug}
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
