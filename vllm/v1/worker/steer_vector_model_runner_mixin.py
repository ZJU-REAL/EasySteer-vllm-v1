# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixin for SteerVector support in V1 GPUModelRunner."""

from typing import TYPE_CHECKING

import torch.nn as nn

from vllm.logger import init_logger
from vllm.steer_vectors.request import SteerVectorRequest

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class SteerVectorModelRunnerMixin:
    """Mixin to add SteerVector support to V1 GPUModelRunner.
    
    This mixin should be used together with GPUModelRunner to enable
    runtime intervention via steer vectors.
    """

    def _init_steer_vector_manager(self, vllm_config: "VllmConfig"):
        """Initialize the steer vector worker manager with LRU cache."""
        from vllm.steer_vectors.worker_manager import LRUCacheWorkerSteerVectorManager
        
        self.steer_vector_manager = LRUCacheWorkerSteerVectorManager(
            device=self.device,  # type: ignore
            steer_vector_config=vllm_config.steer_vector_config,  # type: ignore
        )
        logger.info("Initialized SteerVector worker manager")

    def _wrap_model_with_steer_vectors(self, model: nn.Module) -> nn.Module:
        """Wrap the model to support steer vectors.

        This should be called in load_model() after the model is loaded.
        If server-level steering is configured, also loads and activates
        the steering vector so it is baked in before CUDA graph capture.
        """
        # Lazy initialization: check if steer_vector_config is enabled
        if not hasattr(self, 'steer_vector_manager'):
            self.steer_vector_manager = None
            if hasattr(self, 'vllm_config') and self.vllm_config.steer_vector_config:  # type: ignore
                self._init_steer_vector_manager(self.vllm_config)  # type: ignore

        if self.steer_vector_manager is not None:
            logger.info("Wrapping model with steer vector support")
            model = self.steer_vector_manager.create_steer_vector_manager(model)
            # If server-level steering is configured, load the vector now
            # (before CUDA graph capture) so graphs record the steered path.
            self._maybe_load_server_steer_vector()
        return model

    def _maybe_load_server_steer_vector(self) -> None:
        """Load and activate the server-level steering vector if configured.

        This must run after create_steer_vector_manager() wraps the model
        but before CUDA graph capture, so graphs record the steered forward
        path rather than the identity path.
        """
        if self.steer_vector_manager is None:
            return
        steer_config = getattr(self, 'vllm_config', None)
        if steer_config is None:
            return
        steer_config = steer_config.steer_vector_config  # type: ignore
        if steer_config is None or not steer_config.has_server_config:
            return

        logger.info(
            "Loading server-level steering vector: path=%s, scale=%s, "
            "layers=%s, algorithm=%s, normalize=%s",
            steer_config.server_vector_path,
            steer_config.server_scale,
            steer_config.server_target_layers,
            steer_config.server_algorithm,
            steer_config.server_normalize,
        )

        # Build a SteerVectorRequest from the server config
        server_request = SteerVectorRequest(
            steer_vector_name="__server__",
            steer_vector_int_id=1,
            steer_vector_local_path=steer_config.server_vector_path,
            scale=steer_config.server_scale,
            target_layers=steer_config.server_target_layers,
            algorithm=steer_config.server_algorithm,
            normalize=steer_config.server_normalize,
            # Global triggers — required for CUDA graph safety
            prefill_trigger_tokens=[-1],
            generate_trigger_tokens=[-1],
        )

        success = self.steer_vector_manager.add_adapter(server_request)
        if not success:
            raise RuntimeError(
                f"Failed to load server-level steering vector: "
                f"{steer_config.server_vector_path}"
            )
        logger.info("Server-level steering vector loaded and active")

    def add_steer_vector(self, steer_vector_request: SteerVectorRequest) -> bool:
        """Add a steer vector to the model.
        
        Args:
            steer_vector_request: SteerVectorRequest object
            
        Returns:
            True if successful, False otherwise
        """
        if self.steer_vector_manager is None:
            logger.warning("SteerVector not enabled, cannot add steer vector")
            return False
        
        # Handle msgspec deserialization - convert list back to SteerVectorRequest
        if isinstance(steer_vector_request, (list, tuple)):
            import msgspec

            from vllm.steer_vectors.request import SteerVectorRequest as SVR
            steer_vector_request = msgspec.convert(steer_vector_request, type=SVR)
        
        return self.steer_vector_manager.add_adapter(steer_vector_request)

    def remove_steer_vector(self, steer_vector_id: int) -> bool:
        """Remove a steer vector from the model.
        
        Args:
            steer_vector_id: ID of the steer vector to remove
            
        Returns:
            True if successful, False otherwise
        """
        if self.steer_vector_manager is None:
            return False
        
        return self.steer_vector_manager.remove_adapter(steer_vector_id)

    def set_active_steer_vectors(
        self, steer_vector_requests: set[SteerVectorRequest]
    ) -> None:
        """Set active steer vectors for the current batch.

        This method is called during execute_model to activate the steer vectors
        needed for the current batch of requests (lazy loading).

        When server-level steering is active, the server vector is already loaded
        and stays active. No per-request vectors should arrive (rejected upstream).

        Args:
            steer_vector_requests: Set of SteerVectorRequest objects
                for the current batch.
        """
        mgr = getattr(self, "steer_vector_manager", None)
        if mgr is None:
            if steer_vector_requests:
                raise RuntimeError(
                    "SteerVector is not enabled. Use"
                    " --enable-steer-vector to enable it."
                )
            return

        # For each steer vector request, add and activate if not already loaded
        for steer_vector_request in steer_vector_requests:
            steer_vector_id = steer_vector_request.steer_vector_int_id

            # Check if already loaded
            if steer_vector_id not in self.steer_vector_manager.list_adapters():
                # Load the steer vector (LRU cache will handle eviction if needed)
                self.steer_vector_manager.add_adapter(steer_vector_request)

    def list_steer_vectors(self) -> set[int]:
        """List all active steer vector IDs.
        
        Returns:
            Set of steer vector IDs
        """
        if self.steer_vector_manager is None:
            return set()
        
        return self.steer_vector_manager.list_adapters()

    def remove_all_steer_vectors(self):
        """Remove all steer vectors from the model."""
        if self.steer_vector_manager is not None:
            self.steer_vector_manager.remove_all_adapters()

