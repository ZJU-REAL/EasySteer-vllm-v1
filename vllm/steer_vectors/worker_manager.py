# SPDX-License-Identifier: Apache-2.0

"""Worker-level manager for steer vectors in vLLM V1."""

import logging
from typing import Any, Dict, List, Set

import torch

from vllm.config import SteerVectorConfig
from vllm.steer_vectors.models import (
    SteerVectorModel,
    SteerVectorModelManager,
    LRUCacheSteerVectorModelManager,
    create_sv_manager
)
from vllm.steer_vectors.request import (
    STEER_APPLY_FIELDS,
    SteerVectorRequest,
    layer_apply_kwargs,
    steer_params_dict,
)

logger = logging.getLogger(__name__)


# Config slots (per-request routing) are allocated from a separate id range
# so they never collide with legacy adapter slots (0..capacity).
from vllm.steer_vectors.layers import CONFIG_SLOT_BASE as _CONFIG_SLOT_BASE


def config_fingerprint(request: SteerVectorRequest) -> str:
    """Stable identity of a steering *configuration* (not just the vector).

    Requests with the same fingerprint share one layer slot; the vector
    payload itself is deduplicated separately by the VectorStore. Built
    from the canonical field registry so new parameters participate
    automatically.
    """
    values = [request.local_path, request.debug]
    for name in STEER_APPLY_FIELDS:
        value = getattr(request, name)
        values.append(tuple(value) if isinstance(value, list) else value)
    return repr(tuple(values))


class WorkerSteerVectorManager:
    """WorkerSteerVectorManager that manages steer vector models on the worker side.

    Every request, the requested steer vectors will be loaded (unless they are already loaded),
    and every other steer vector will be unloaded.
    """

    _manager_cls: type[SteerVectorModelManager] = SteerVectorModelManager

    def __init__(
        self,
        device: torch.device,
        steer_vector_config: SteerVectorConfig,
        steer_vector_model_cls: type[SteerVectorModel] = SteerVectorModel
    ):
        self._adapter_manager: SteerVectorModelManager | None = None
        self._steer_vector_model_cls = steer_vector_model_cls
        self.steer_vector_config = steer_vector_config
        self.device = device
        from vllm.steer_vectors.store import VectorStore
        self.vector_store = VectorStore(str(device), steer_vector_config)
        # Per-request config routing state: fingerprint -> [slot, refcount,
        # request]; req_id -> fingerprint.
        self._config_slots: dict[str, list] = {}
        self._req_fingerprints: dict[str, str] = {}
        self._free_slots: list[int] = []
        self._next_slot = _CONFIG_SLOT_BASE

    # ------------------------------------------------------------------
    # Per-request config routing (Phase C)
    # ------------------------------------------------------------------

    def preload_vectors(self, paths: list[str], algorithm: str = "direct"):
        for path in paths:
            self.vector_store.preload(path, algorithm)

    def acquire_config(
        self, req_id: str, request: SteerVectorRequest
    ) -> int | None:
        """Register a live request's steering config; returns its slot.

        Returns None for configs that cannot be routed per-request yet
        (multi-vector and moe_router fall back to the legacy global path).
        """
        if request.is_multi_vector or request.algorithm == "moe_router":
            logger.warning(
                "Per-request routing does not yet support %s configs; "
                "falling back to globally-activated steering.",
                "multi-vector" if request.is_multi_vector else "moe_router",
            )
            return None

        fp = config_fingerprint(request)
        entry = self._config_slots.get(fp)
        if entry is not None:
            entry[1] += 1
            self._req_fingerprints[req_id] = fp
            return entry[0]

        slot = self._free_slots.pop() if self._free_slots else self._next_slot
        if slot == self._next_slot:
            self._next_slot += 1

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
        slot = entry[0]
        del self._config_slots[fp]
        if self._adapter_manager is not None:
            for module in self._adapter_manager.modules.values():
                module.reset_steer_vector(slot)
                slot_algos = getattr(module, "slot_algorithms", None)
                if slot_algos is not None:
                    slot_algos.pop(slot, None)
        self._free_slots.append(slot)

    def slot_for_request(self, req_id: str) -> int | None:
        fp = self._req_fingerprints.get(req_id)
        if fp is None:
            return None
        entry = self._config_slots.get(fp)
        return None if entry is None else entry[0]

    def _distribute_config(
        self, slot: int, model: SteerVectorModel, request: SteerVectorRequest
    ) -> None:
        """Write a config's scaled payload + triggers into layer slot state."""
        assert self._adapter_manager is not None
        params = layer_apply_kwargs(request)
        target_layers = params.pop("target_layers")
        for layer_idx, payload in (model.layer_payloads or {}).items():
            if target_layers and layer_idx not in target_layers:
                continue
            for module in self._adapter_manager._get_modules_for_layer(
                layer_idx, "decoder_layer"
            ):
                module.set_steer_vector(slot, payload=payload, **params)

    @property
    def is_enabled(self) -> bool:
        return True

    def create_steer_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        """Create and initialize the steer vector manager for the model."""
        steer_vector_manager = create_sv_manager(
            model,
            steer_vector_config=self.steer_vector_config,
            steer_vector_manager_cls=self._manager_cls,
        )
        self._adapter_manager = steer_vector_manager
        return steer_vector_manager.model

    def _load_adapter(
        self,
        steer_vector_request: SteerVectorRequest
    ) -> SteerVectorModel:
        """Load a steer vector from a request.
        
        This method acts as the decoupling layer between SteerVectorRequest
        and SteerVectorModel, extracting parameters from the request and
        calling the appropriate factory method.
        
        Similar to LoRA's WorkerLoRAManager._load_adapter() pattern.
        """
        try:
            if not steer_vector_request.is_multi_vector:
                # Check if this is MoE router algorithm WITHOUT a config file path
                if (steer_vector_request.algorithm == "moe_router" and 
                    not steer_vector_request.local_path):
                    # MoE router: create model directly from request parameters
                    # No need to load from file - parameters come from request
                    if steer_vector_request.moe_expert_ids is None:
                        raise ValueError("moe_router algorithm requires moe_expert_ids parameter")
                    
                    # Build layer payloads from request MoE parameters
                    layer_payloads = {}
                    target_layers = steer_vector_request.target_layers or []
                    
                    for layer_id in target_layers:
                        payload = {
                            'expert_ids': steer_vector_request.moe_expert_ids,
                            'mode': steer_vector_request.moe_mode,  # 'boost', 'suppress', or 'soft'
                        }
                        # Add lambda parameter for 'soft' mode
                        if steer_vector_request.moe_mode == 'soft':
                            payload['lambda'] = steer_vector_request.moe_lambda
                        layer_payloads[layer_id] = payload
                    
                    steer_vector = self._steer_vector_model_cls(
                        steer_vector_id=steer_vector_request.steer_vector_id,
                        layer_payloads=layer_payloads,
                        scale_factor=1.0,
                        algorithm="moe_router",
                    )
                else:
                    # Single-vector mode: extract parameters and call from_local_checkpoint
                    # This includes moe_router with a config file path
                    
                    # Build kwargs for algorithm-specific parameters
                    load_kwargs = {}
                    if steer_vector_request.algorithm == "moe_router":
                        # Pass moe_mode, moe_lambda and moe_topk for moe_router algorithm
                        load_kwargs['moe_mode'] = steer_vector_request.moe_mode
                        load_kwargs['moe_lambda'] = steer_vector_request.moe_lambda
                        load_kwargs['moe_topk'] = steer_vector_request.moe_topk
                    
                    steer_vector = self._steer_vector_model_cls.from_local_checkpoint(
                        steer_vector_model_path=steer_vector_request.local_path,
                        steer_vector_id=steer_vector_request.steer_vector_id,
                        config=self.steer_vector_config,
                        device=str(self.device),
                        scale_factor=steer_vector_request.scale,
                        algorithm=steer_vector_request.algorithm,
                        target_layers=steer_vector_request.target_layers,
                        **load_kwargs,
                    )
            else:
                # Multi-vector mode: load each vector individually and assemble
                multi_vector_data = []
                
                for i, vector_config in enumerate(steer_vector_request.vector_configs):
                    try:
                        # Load individual vector using from_local_checkpoint
                        single_model = self._steer_vector_model_cls.from_local_checkpoint(
                            steer_vector_model_path=vector_config.path,
                            steer_vector_id=f"{steer_vector_request.steer_vector_id}_vec_{i}",
                            config=self.steer_vector_config,
                            device=str(self.device),
                            scale_factor=vector_config.scale,
                            algorithm=vector_config.algorithm,
                            target_layers=vector_config.target_layers,
                        )
                        
                        # Store vector data with its configuration
                        # (canonical fields + payloads/path extras)
                        vector_data = {
                            **steer_params_dict(vector_config),
                            'payloads': single_model.layer_payloads,
                            'path': vector_config.path,
                        }
                        multi_vector_data.append(vector_data)
                        
                        logger.debug(
                            f"Loaded vector {i}: {vector_config.path} "
                            f"(algorithm: {vector_config.algorithm}, scale: {vector_config.scale})"
                        )
                        
                    except Exception as e:
                        logger.error(f"Failed to load vector {i} from {vector_config.path}: {e}")
                        raise RuntimeError(
                            f"Failed to load vector {i} from {vector_config.path}"
                        ) from e
                
                logger.debug(
                    f"Successfully loaded {len(multi_vector_data)} vectors for "
                    f"multi-vector request '{steer_vector_request.steer_vector_name}'"
                )
                
                # Create multi-vector model (note: no from_steer_vector_request needed!)
                steer_vector = self._steer_vector_model_cls(
                    steer_vector_id=steer_vector_request.steer_vector_id,
                    layer_payloads=None,
                    scale_factor=1.0,
                    algorithm="multi_vector",
                    multi_vector_data=multi_vector_data
                )
                
        except Exception as e:
            request_info = (
                steer_vector_request.local_path 
                if not steer_vector_request.is_multi_vector 
                else f"multi-vector request with {len(steer_vector_request.vector_configs)} vectors"
            )
            # Import traceback to get full error details
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Failed to load steer vector {request_info}:\n{error_details}")
            raise RuntimeError(
                f"Loading steer vector {request_info} failed: {str(e)}"
            ) from e
        
        return steer_vector

    def add_adapter(self, adapter_request: SteerVectorRequest) -> bool:
        """Add a steer vector adapter."""
        if self._adapter_manager is None:
            logger.warning("Steer vector manager not initialized")
            return False
        
        # Support replacing adapters with the same ID by removing old one first
        if adapter_request.steer_vector_id in self.list_adapters():
            logger.debug(
                f"Replacing existing steer vector with ID {adapter_request.steer_vector_id}"
            )
            self.remove_adapter(adapter_request.steer_vector_id)
        
        # Load the adapter
        adapter = self._load_adapter(adapter_request)
        
        # Add to manager
        if not self._adapter_manager.add_adapter(adapter):
            return False
        
        # Activate based on request type
        if adapter_request.is_multi_vector:
            # Multi-vector mode: activation is handled internally
            self._adapter_manager.activate_adapter(
                adapter_request.steer_vector_id,
                debug=adapter_request.debug,
                conflict_resolution=adapter_request.conflict_resolution
            )
        else:
            # Single-vector mode: use request-level parameters
            self._adapter_manager.activate_adapter(
                adapter_request.steer_vector_id,
                target_layers=adapter_request.target_layers,
                prefill_trigger_tokens=adapter_request.prefill_trigger_tokens,
                prefill_trigger_positions=adapter_request.prefill_trigger_positions,
                prefill_exclude_tokens=adapter_request.prefill_exclude_tokens,
                prefill_exclude_positions=adapter_request.prefill_exclude_positions,
                generate_trigger_tokens=adapter_request.generate_trigger_tokens,
                generate_first_k_tokens=adapter_request.generate_first_k_tokens,
                generate_after_k_tokens=adapter_request.generate_after_k_tokens,
                debug=adapter_request.debug,
                normalize=adapter_request.normalize
            )
        
        return True

    def remove_adapter(self, adapter_id: int) -> bool:
        """Remove a steer vector adapter."""
        if self._adapter_manager is None:
            return False
        return self._adapter_manager.remove_adapter(adapter_id)

    def remove_all_adapters(self):
        """Remove all steer vector adapters."""
        if self._adapter_manager is not None:
            self._adapter_manager.remove_all_adapters()

    def list_adapters(self) -> Set[int]:
        """List all registered adapter IDs."""
        if self._adapter_manager is None:
            return set()
        return set(self._adapter_manager.list_adapters().keys())


class LRUCacheWorkerSteerVectorManager(WorkerSteerVectorManager):
    """WorkerSteerVectorManager that manages steer vector models with LRU cache.

    Uses an LRU Cache. Every request, the requested steer vectors will be loaded 
    (unless they are already loaded) and least recently used steer vectors will
    be unloaded if the cache is above capacity.
    """

    _steer_vector_manager_cls: type[
        LRUCacheSteerVectorModelManager
    ] = LRUCacheSteerVectorModelManager

    def create_steer_vector_manager(
        self,
        model: torch.nn.Module,
    ) -> Any:
        """Create LRU cache steer vector manager."""
        steer_vector_manager = create_sv_manager(
            model,
            steer_vector_config=self.steer_vector_config,
            steer_vector_manager_cls=self._steer_vector_manager_cls
        )
        self._adapter_manager: LRUCacheSteerVectorModelManager = (
            steer_vector_manager
        )
        return steer_vector_manager.model

    def add_adapter(
        self,
        steer_vector_request: SteerVectorRequest
    ) -> bool:
        """Add adapter with LRU cache management."""
        if self._adapter_manager is None:
            return False
        
        if steer_vector_request.steer_vector_id not in self.list_adapters():
            # Remove before we load the new steer vector to save memory
            if (len(self._adapter_manager._registered_adapters) + 1 
                > self._adapter_manager.capacity):
                self._adapter_manager.remove_oldest_adapter()
            
            steer_vector = self._load_adapter(steer_vector_request)
            loaded = self._adapter_manager.add_adapter(steer_vector)
        else:
            # Support replacing adapters with the same ID
            logger.debug(
                f"Replacing existing steer vector with ID "
                f"{steer_vector_request.steer_vector_id}"
            )
            self._adapter_manager.remove_adapter(steer_vector_request.steer_vector_id)
            steer_vector = self._load_adapter(steer_vector_request)
            loaded = self._adapter_manager.add_adapter(steer_vector)
        
        if not loaded:
            return False
        
        # Activate based on mode
        if steer_vector_request.is_multi_vector:
            self._adapter_manager.activate_adapter(
                steer_vector_request.steer_vector_id,
                debug=steer_vector_request.debug,
                conflict_resolution=steer_vector_request.conflict_resolution
            )
        else:
            self._adapter_manager.activate_adapter(
                steer_vector_request.steer_vector_id,
                target_layers=steer_vector_request.target_layers,
                prefill_trigger_tokens=steer_vector_request.prefill_trigger_tokens,
                prefill_trigger_positions=steer_vector_request.prefill_trigger_positions,
                prefill_exclude_tokens=steer_vector_request.prefill_exclude_tokens,
                prefill_exclude_positions=steer_vector_request.prefill_exclude_positions,
                generate_trigger_tokens=steer_vector_request.generate_trigger_tokens,
                generate_first_k_tokens=steer_vector_request.generate_first_k_tokens,
                generate_after_k_tokens=steer_vector_request.generate_after_k_tokens,
                debug=steer_vector_request.debug,
                normalize=steer_vector_request.normalize
            )
        
        return loaded
