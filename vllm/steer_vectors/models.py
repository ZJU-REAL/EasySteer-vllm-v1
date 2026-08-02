# SPDX-License-Identifier: Apache-2.0

"""Steer vector loading and controller attachment.

`LoadedSteerVector` holds per-layer payloads loaded from disk;
`SteerControllerManager` discovers steerable modules on a model (see
steer_vectors.discovery) and registers the steering hooks.
"""

import os
from typing import TypeVar

from huggingface_hub import hf_hub_download
from torch import nn

from vllm.config import SteerVectorConfig
from vllm.logger import init_logger
from vllm.steer_vectors import ops as steer_ops
from vllm.steer_vectors.algorithms.factory import ALGORITHM_REGISTRY
from vllm.steer_vectors.discovery import (
    SUPPORTED_DECODER_LAYERS,
    SUPPORTED_MOE_LAYERS,
    extract_layer_id_from_module_name,
    find_decoder_layers,
    find_moe_blocks,
    find_moe_gate,
    moe_gate_is_fused,
)
from vllm.steer_vectors.layers import (
    DecoderLayerWithSteerVector,
    MoEGateSteerController,
)

logger = init_logger(__name__)

T = TypeVar("T")

_CONTROLLER_CLASSES = {
    "decoder_layer": DecoderLayerWithSteerVector,
    "moe_layer": MoEGateSteerController,
}

_STRUCTURAL_FINDERS = {
    "decoder_layer": find_decoder_layers,
    "moe_layer": find_moe_blocks,
}

_FALLBACK_CLASS_NAMES = {
    "decoder_layer": SUPPORTED_DECODER_LAYERS,
    "moe_layer": SUPPORTED_MOE_LAYERS,
}


class LoadedSteerVector:
    """A steer vector loaded from disk, ready to configure into slots.

    Encapsulates the data and metadata for loaded steer vectors,
    supporting both single-vector and multi-vector configurations.

    Attributes:
        id: Unique identifier for this steer vector
        layer_payloads: Dict mapping layer indices to their vector payloads
        scale_factor: Global scaling factor for single-vector mode
        algorithm: Algorithm type ('direct', 'linear', etc.)
        multi_vector_data: Configuration for multi-vector mode (if applicable)
    """

    def __init__(
        self,
        steer_vector_id=None,
        layer_payloads=None,
        scale_factor=1.0,
        algorithm="direct",
        multi_vector_data=None,
    ) -> None:
        self.id = steer_vector_id
        self.layer_payloads = layer_payloads
        self.scale_factor = scale_factor
        self.algorithm = algorithm
        self.multi_vector_data = multi_vector_data

    @property
    def is_multi_vector(self) -> bool:
        return self.multi_vector_data is not None

    @classmethod
    def from_local_checkpoint(
        cls,
        steer_vector_model_path: str,
        steer_vector_id: int,
        config: SteerVectorConfig,
        device: str = "cuda",
        scale_factor: float = 1.0,
        algorithm: str = "direct",
        target_layers: list[int] | None = None,
        **kwargs,  # Additional algorithm-specific parameters (e.g., moe_lambda).
    ) -> "LoadedSteerVector":
        """Load a steer vector from a local checkpoint or HuggingFace Hub.

        Args:
            steer_vector_model_path: Path to the vector file (local or HF format)
            steer_vector_id: Unique ID for this vector
            config: Steer vector configuration
            device: Device to load tensors on
            scale_factor: Global scaling factor
            algorithm: Algorithm type (can be embedded in path with "|")
            target_layers: Optional list of target layer indices

        Returns:
            Loaded LoadedSteerVector instance
        """
        # An algorithm can be embedded in the path ("path/to/vector|linear").
        if "|" in steer_vector_model_path:
            steer_vector_model_path, algorithm = steer_vector_model_path.split("|", 1)

        if os.path.exists(steer_vector_model_path):
            file_path = os.path.abspath(steer_vector_model_path)
        else:
            parts = steer_vector_model_path.split("/")
            repo_id = "/".join(parts[:2])
            file_name = "/".join(parts[2:])
            file_path = hf_hub_download(
                repo_id=repo_id, filename=file_name, revision="main"
            )

        if algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(f"Unsupported algorithm for loading: '{algorithm}'")

        algo_class = ALGORITHM_REGISTRY[algorithm]
        loaded_params = algo_class.load_from_path(
            file_path, device, config=config, target_layers=target_layers, **kwargs
        )

        return cls(
            steer_vector_id=steer_vector_id,
            layer_payloads=loaded_params.get("layer_payloads"),
            scale_factor=scale_factor,
            algorithm=algorithm,
        )


class SteerControllerManager:
    """Discovers steerable modules and owns their steering controllers.

    On construction it finds decoder layers and MoE gates on the model,
    registers a forward hook per module, and registers each controller
    with the custom-op registry. The original modules stay in the tree
    untouched (module names, classes and state-dict keys are preserved,
    keeping FSDP/checkpointing consumers such as VERL working);
    controllers live outside the model.
    """

    def __init__(self, model: nn.Module, steer_vector_config: SteerVectorConfig):
        self.model = model
        self.steer_vector_config = steer_vector_config
        self.model.steer_vector_manager = self
        self.modules: dict[str, nn.Module] = {}
        self._hook_handles: list = []
        self._hooked_modules: set[str] = set()
        for controller_type in _CONTROLLER_CLASSES:
            self._attach_controllers(controller_type)

    def _attach_controllers(self, controller_type: str) -> None:
        """Discover and hook all modules of one controller type.

        Discovery is structural first (anchor-based, architecture
        agnostic); the per-family class-name lists in discovery.py are
        the fallback for layouts structural discovery cannot identify.
        """
        controller_class = _CONTROLLER_CLASSES[controller_type]
        matches = _STRUCTURAL_FINDERS[controller_type](self.model)
        if not matches:
            class_names = _FALLBACK_CLASS_NAMES[controller_type]
            matches = {
                module_name: module
                for module_name, module in self.model.named_modules()
                if any(
                    class_name in module.__class__.__name__
                    for class_name in class_names
                )
            }
            if matches:
                logger.info(
                    "Structural discovery found no %s modules; using the "
                    "class-name list fallback (%d modules, e.g. %s).",
                    controller_type,
                    len(matches),
                    next(iter(matches)),
                )

        hooked_count = 0
        for module_name, module in matches.items():
            if controller_type == "decoder_layer":
                if module_name in self._hooked_modules:
                    continue
                controller = controller_class()
                layer_id = extract_layer_id_from_module_name(module_name)
                if layer_id is not None:
                    controller.set_layer_id(layer_id)
                controller._op_key = module_name
                steer_ops.register_controller(module_name, controller)
                handle = module.register_forward_hook(controller.process_output_hook)
                self._hook_handles.append(handle)
                self._hooked_modules.add(module_name)
                self.register_module(module_name, controller)
                hooked_count += 1
                logger.debug("Hooked %s: %s", controller_type, module_name)
            elif controller_type == "moe_layer":
                # Steer router logits by hooking the block's gate/router
                # submodule (architecture-agnostic: the logits are
                # modified in place before top-k expert selection).
                gate = find_moe_gate(module)
                if gate is None:
                    logger.warning(
                        "MoE block %s has no gate/router submodule; "
                        "cannot steer its router logits.",
                        module_name,
                    )
                    continue
                if moe_gate_is_fused(module):
                    logger.warning(
                        "MoE block %s fuses gate weights into the MoE "
                        "runner (the gate module forward is bypassed), so "
                        "gate hooks would never fire; router-logit "
                        "steering is unavailable for this block.",
                        module_name,
                    )
                    continue
                op_key = f"{module_name}::gate"
                if op_key in self._hooked_modules:
                    continue
                controller = controller_class()
                layer_id = extract_layer_id_from_module_name(module_name)
                if layer_id is not None:
                    controller.set_layer_id(layer_id)
                controller._op_key = op_key
                steer_ops.register_controller(op_key, controller)
                handle = gate.register_forward_hook(controller.process_gate_output_hook)
                self._hook_handles.append(handle)
                self._hooked_modules.add(op_key)
                self.register_module(module_name, controller)
                hooked_count += 1
                logger.debug("Hooked %s gate: %s", controller_type, module_name)

        if hooked_count > 0:
            logger.debug(
                "Using %s-level steering (%d modules hooked)",
                controller_type,
                hooked_count,
            )
        else:
            logger.warning("No %s modules found for steering", controller_type)

    def _resolve_controller_type(self, algorithm_name: str) -> str:
        """Determine which controller type should receive a given algorithm."""
        if algorithm_name == "moe_router":
            return "moe_layer"
        return "decoder_layer"

    def _get_modules_for_layer(
        self, layer_idx: int, controller_type: str | None = None
    ) -> list[nn.Module]:
        """Get all controllers for the specified layer."""
        modules = []
        target_class = None
        if controller_type:
            target_class = _CONTROLLER_CLASSES.get(controller_type)
        for module_name, module in self.modules.items():
            if extract_layer_id_from_module_name(module_name) == layer_idx:
                if target_class is not None and not isinstance(module, target_class):
                    continue
                modules.append(module)
        return modules

    def register_module(self, module_name: str, module: nn.Module):
        self.modules[module_name] = module

    def remove_hooks(self):
        """Detach all forward hooks registered on the model."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for module_name in self._hooked_modules:
            steer_ops.unregister_controller(module_name)
        self._hooked_modules.clear()


def create_steer_controller_manager(
    model: nn.Module,
    steer_vector_config: SteerVectorConfig,
    steer_vector_manager_cls: type[SteerControllerManager] = SteerControllerManager,
) -> SteerControllerManager:
    """Create and attach a steer controller manager for a model."""
    return steer_vector_manager_cls(
        model=model, steer_vector_config=steer_vector_config
    )
