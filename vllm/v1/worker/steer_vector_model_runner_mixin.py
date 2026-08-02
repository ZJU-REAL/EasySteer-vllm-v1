# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixin for SteerVector support in the GPU model runner."""

from typing import TYPE_CHECKING

import torch.nn as nn

from vllm.logger import init_logger
from vllm.steer_vectors.request import SteerVectorRequest

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class SteerVectorModelRunnerMixin:
    """Adds steer-vector wrapping and config plumbing to the model runner.

    All steering is per-request slot-routed; the server-level config (if
    any) occupies the default slot that requests without their own
    steer_vector_request are routed to.
    """

    def _init_steer_vector_manager(self, vllm_config: "VllmConfig"):
        from vllm.steer_vectors.worker_manager import WorkerSteerVectorManager

        self.steer_vector_manager = WorkerSteerVectorManager(
            device=self.device,  # type: ignore
            steer_vector_config=vllm_config.steer_vector_config,  # type: ignore
        )
        logger.info("Initialized SteerVector worker manager")

    def _wrap_model_with_steer_vectors(self, model: nn.Module) -> nn.Module:
        """Wrap the model to support steer vectors.

        This should be called in load_model() after the model is loaded.
        If server-level steering is configured, it is installed on the
        default routing slot before compilation/graph capture.
        """
        if not hasattr(self, "steer_vector_manager"):
            self.steer_vector_manager = None
            if hasattr(self, "vllm_config") and self.vllm_config.steer_vector_config:  # type: ignore
                self._init_steer_vector_manager(self.vllm_config)  # type: ignore

        if self.steer_vector_manager is not None:
            logger.info("Wrapping model with steer vector support")
            vllm_config = self.vllm_config  # type: ignore
            if vllm_config.steer_vector_config.graph_mode == "full":
                # Tier-1 buffers must exist before compile/graph capture.
                self.steer_vector_manager.enable_graph_mode(
                    vllm_config.model_config.get_hidden_size(),
                    vllm_config.model_config.dtype,
                    vllm_config.scheduler_config.max_num_batched_tokens,
                )
            model = self.steer_vector_manager.create_steer_vector_manager(model)
            self._maybe_load_server_steer_vector()
        return model

    def _maybe_load_server_steer_vector(self) -> None:
        """Install the server-level steering config on the default slot."""
        if self.steer_vector_manager is None:
            return
        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            return
        steer_config = vllm_config.steer_vector_config  # type: ignore
        if steer_config is None or not steer_config.has_server_config:
            return

        logger.info(
            "Loading engine-default steering spec: %s",
            steer_config.steering_config,
        )
        from vllm.steer_vectors.request import build_server_request

        self.steer_vector_manager.set_server_config(build_server_request(steer_config))
        logger.info("Server-level steering vector loaded and active")

    def preload_steer_vectors(
        self, paths: list[str], algorithm: str = "direct"
    ) -> bool:
        """Load steering vectors into the worker's vector store ahead of
        use, so request admission never blocks on disk I/O."""
        mgr = getattr(self, "steer_vector_manager", None)
        if mgr is None:
            logger.warning("SteerVector not enabled, cannot preload vectors")
            return False
        mgr.preload_vectors(list(paths), algorithm)
        return True

    def add_steer_vector(self, steer_vector_request: SteerVectorRequest) -> bool:
        """Install (or replace) the server-level default steering config."""
        if self.steer_vector_manager is None:
            logger.warning("SteerVector not enabled, cannot add steer vector")
            return False
        vllm_config = getattr(self, "vllm_config", None)
        if (
            vllm_config is not None
            and vllm_config.cache_config is not None
            and vllm_config.cache_config.enable_prefix_caching
            and not (
                vllm_config.steer_vector_config is not None
                and vllm_config.steer_vector_config.has_server_config
            )
        ):
            # Replacing a startup server config (scale updates) is fine:
            # hashes are salted and the update path resets the prefix
            # cache. A fresh install is not — pre-install blocks were
            # hashed without any salt.
            raise RuntimeError(
                "Cannot install server-level steering at runtime on a "
                "prefix-caching engine without a startup server config "
                "(--steering-config): existing cache blocks were hashed "
                "without the server salt."
            )

        # Handle msgspec deserialization - convert list back to SteerVectorRequest
        if isinstance(steer_vector_request, (list, tuple)):
            import msgspec

            from vllm.steer_vectors.request import SteerVectorRequest as SVR

            steer_vector_request = msgspec.convert(steer_vector_request, type=SVR)

        return self.steer_vector_manager.set_server_config(steer_vector_request)

    def remove_steer_vector(self, steer_vector_id: int) -> bool:
        """Clear the server-level default steering config."""
        if self.steer_vector_manager is None:
            return False
        return self.steer_vector_manager.clear_server_config()

    def list_steer_vectors(self) -> set[int]:
        """Int ids of all live steering configs."""
        if self.steer_vector_manager is None:
            return set()
        return self.steer_vector_manager.list_configs()
