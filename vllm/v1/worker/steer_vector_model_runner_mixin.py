# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixin for SteerVector support in the GPU model runner."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.model_hooks.components.registry import ModelComponents
    from vllm.model_hooks.steering.worker_manager import WorkerSteeringState

logger = init_logger(__name__)


class SteerVectorModelRunnerMixin:
    """Attach steering hooks and preload payloads in the model runner.

    Every request carries its effective steering configuration.
    """

    device: "torch.device"
    vllm_config: "VllmConfig"
    steer_vector_manager: "WorkerSteeringState | None"

    def _init_steer_vector_manager(self, vllm_config: "VllmConfig") -> None:
        from vllm.model_hooks.steering.worker_manager import WorkerSteeringState

        config = vllm_config.steer_vector_config
        assert config is not None
        self.steer_vector_manager = WorkerSteeringState(
            device=self.device,
            steer_vector_config=config,
            hidden_size=vllm_config.model_config.get_hidden_size(),
        )
        default = config._default_request
        if default is not None:
            self.steer_vector_manager.preload_request(default)
        logger.info("Initialized SteerVector worker manager")

    def _attach_steering_hooks(self, components: "ModelComponents") -> None:
        """Attach steering to the discovered components after loading the model."""
        from vllm.model_hooks.components.registry import ROUTER_LOGITS
        from vllm.model_hooks.steering.capabilities import declared_components

        self._close_steering()
        vllm_config = self.vllm_config
        if vllm_config.steer_vector_config is not None:
            parallel = getattr(vllm_config, "parallel_config", None)
            if getattr(parallel, "use_sequence_parallel_moe", False) and (
                ROUTER_LOGITS
                in declared_components(vllm_config.steer_vector_config.algorithms)
            ):
                raise ValueError(
                    "moe_router steering requires replicated token rows and does "
                    "not support sequence-parallel MoE; omit moe_router from "
                    "steer_algorithms or disable sequence-parallel MoE."
                )
            try:
                self._init_steer_vector_manager(vllm_config)
                manager = self.steer_vector_manager
                assert manager is not None
                logger.info("Attaching steering hooks")
                if vllm_config.steer_vector_config.graph_mode == "in_graph":
                    # Tier-1 buffers must exist before compile/graph capture.
                    manager.enable_graph_mode(
                        vllm_config.model_config.get_hidden_size(),
                        vllm_config.model_config.dtype,
                        vllm_config.scheduler_config.max_num_batched_tokens,
                    )
                manager.attach_steering_hooks(components)
            except Exception:
                self._close_steering()
                raise

    def _close_steering(self) -> None:
        manager = getattr(self, "steer_vector_manager", None)
        if manager is not None:
            manager.close()
        self.steer_vector_manager = None

    def preload_steer_vectors(self, payloads: list[dict]) -> bool:
        """Materialize admitted payload snapshots in the worker cache."""
        if self.steer_vector_manager is None:
            raise ValueError("SteerVector is not enabled")
        self.steer_vector_manager.preload_vectors(payloads)
        return True

    def list_steer_vectors(self) -> set[int]:
        """Int ids of all live steering configs."""
        if self.steer_vector_manager is None:
            return set()
        return self.steer_vector_manager.list_configs()
