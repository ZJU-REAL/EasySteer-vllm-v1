# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixin for SteerVector support in the GPU model runner."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class SteerVectorModelRunnerMixin:
    """Attach steering hooks and preload payloads in the model runner.

    Every request carries its effective steering configuration.
    """

    def _init_steer_vector_manager(self, vllm_config: "VllmConfig"):
        from vllm.model_hooks.steering.worker_manager import WorkerSteeringState

        self.steer_vector_manager = WorkerSteeringState(
            device=self.device,  # type: ignore
            steer_vector_config=vllm_config.steer_vector_config,  # type: ignore
            hidden_size=vllm_config.model_config.get_hidden_size(),
        )
        default = vllm_config.steer_vector_config._default_request
        if default is not None:
            self.steer_vector_manager.preload_request(default)
        logger.info("Initialized SteerVector worker manager")

    def _attach_steering_hooks(self, components) -> None:
        """Attach steering to the discovered components after loading the model."""
        self._close_steering()
        vllm_config = self.vllm_config  # type: ignore
        if vllm_config.steer_vector_config is not None:
            try:
                self._init_steer_vector_manager(vllm_config)
                logger.info("Attaching steering hooks")
                if vllm_config.steer_vector_config.graph_mode == "in_graph":
                    # Tier-1 buffers must exist before compile/graph capture.
                    self.steer_vector_manager.enable_graph_mode(
                        vllm_config.model_config.get_hidden_size(),
                        vllm_config.model_config.dtype,
                        vllm_config.scheduler_config.max_num_batched_tokens,
                    )
                self.steer_vector_manager.attach_steering_hooks(components)
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
