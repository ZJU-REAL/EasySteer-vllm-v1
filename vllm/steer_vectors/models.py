# SPDX-License-Identifier: Apache-2.0

"""Steer vector loading and controller attachment.

`LoadedSteerVector` holds per-layer payloads loaded from disk;
`SteerControllerManager` discovers steerable modules on a model (see
steer_vectors.discovery) and registers the steering hooks.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

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
    find_layers_with_fallback,
    find_moe_blocks,
    find_moe_gate,
    moe_gate_is_fused,
)
from vllm.steer_vectors.layers import (
    DecoderLayerWithSteerVector,
    MoEGateSteerController,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class _ControllerSpec:
    """Everything type-specific about attaching one controller family."""

    controller_class: type
    structural_finder: Callable[[nn.Module], dict[str, nn.Module]]
    fallback_class_names: list[str]
    # (module_name, module) -> submodule to hook, or None to skip (after
    # warning about why the module cannot be steered).
    hook_target: Callable[[str, nn.Module], nn.Module | None]
    hook_method: str
    op_key_suffix: str
    display_name: str
    debug_label: str


def _decoder_hook_target(module_name: str, module: nn.Module) -> nn.Module | None:
    return module


def _moe_gate_hook_target(module_name: str, module: nn.Module) -> nn.Module | None:
    # Steer router logits by hooking the block's gate/router submodule
    # (architecture-agnostic: the logits are modified in place before
    # top-k expert selection).
    gate = find_moe_gate(module)
    if gate is None:
        logger.warning(
            "MoE block %s has no gate/router submodule; "
            "cannot steer its router logits.",
            module_name,
        )
        return None
    if moe_gate_is_fused(module):
        logger.warning(
            "MoE block %s fuses gate weights into the MoE "
            "runner (the gate module forward is bypassed), so "
            "gate hooks would never fire; router-logit "
            "steering is unavailable for this block.",
            module_name,
        )
        return None
    return gate


_CONTROLLER_REGISTRY: dict[str, _ControllerSpec] = {
    "decoder_layer": _ControllerSpec(
        controller_class=DecoderLayerWithSteerVector,
        structural_finder=find_decoder_layers,
        fallback_class_names=SUPPORTED_DECODER_LAYERS,
        hook_target=_decoder_hook_target,
        hook_method="process_output_hook",
        op_key_suffix="",
        display_name="Decoder layer",
        debug_label="decoder_layer",
    ),
    "moe_layer": _ControllerSpec(
        controller_class=MoEGateSteerController,
        structural_finder=find_moe_blocks,
        fallback_class_names=SUPPORTED_MOE_LAYERS,
        hook_target=_moe_gate_hook_target,
        hook_method="process_gate_output_hook",
        op_key_suffix="::gate",
        display_name="MoE block",
        debug_label="moe_layer gate",
    ),
}


class LoadedSteerVector:
    """A steer vector loaded from disk, ready to configure into slots.

    Attributes:
        id: Unique identifier for this steer vector
        layer_payloads: Dict mapping layer indices to their vector payloads
        scale_factor: Global scaling factor for single-vector mode
        algorithm: Algorithm type ('direct', 'linear', etc.)
    """

    def __init__(
        self,
        steer_vector_id=None,
        layer_payloads=None,
        scale_factor=1.0,
        algorithm="direct",
    ) -> None:
        self.id = steer_vector_id
        self.layer_payloads = layer_payloads
        self.scale_factor = scale_factor
        self.algorithm = algorithm

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

        layer_payloads = loaded_params.get("layer_payloads")
        if not layer_payloads:
            raise ValueError(
                f"Algorithm '{algorithm}' loaded no layer payloads from "
                f"{file_path}; the vector would steer nothing"
            )

        return cls(
            steer_vector_id=steer_vector_id,
            layer_payloads=layer_payloads,
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
        for controller_type in _CONTROLLER_REGISTRY:
            self._attach_controllers(controller_type)

    def _attach_controllers(self, controller_type: str) -> None:
        """Discover and hook all modules of one controller type.

        Discovery is structural first (anchor-based, architecture
        agnostic); the per-family class-name lists in discovery.py are
        the fallback for layouts structural discovery cannot identify.
        """
        spec = _CONTROLLER_REGISTRY[controller_type]
        matches = find_layers_with_fallback(
            self.model,
            spec.structural_finder,
            spec.fallback_class_names,
            controller_type,
        )

        hooked_count = 0
        for module_name, module in matches.items():
            target = spec.hook_target(module_name, module)
            if target is None:
                continue
            op_key = module_name + spec.op_key_suffix
            if op_key in self._hooked_modules:
                continue
            controller = spec.controller_class()
            layer_id = extract_layer_id_from_module_name(module_name)
            if layer_id is not None:
                controller.set_layer_id(layer_id)
            else:
                logger.warning(
                    "%s %s has no numeric layer id in its module name; "
                    "it can never be targeted by layer-indexed steering "
                    "configs.",
                    spec.display_name,
                    module_name,
                )
            controller._op_key = op_key
            # Keep a reference to the hooked module without registering
            # it as a submodule (that would cycle the module tree); the
            # MoE gate graph tables read their expert count from it.
            object.__setattr__(controller, "hook_target", target)
            steer_ops.register_controller(op_key, controller)
            hook = getattr(controller, spec.hook_method)
            self._hook_handles.append(target.register_forward_hook(hook))
            self._hooked_modules.add(op_key)
            self.register_module(module_name, controller)
            hooked_count += 1
            logger.debug("Hooked %s: %s", spec.debug_label, module_name)

        if hooked_count > 0:
            logger.debug(
                "Using %s-level steering (%d modules hooked)",
                controller_type,
                hooked_count,
            )
        else:
            logger.warning("No %s modules found for steering", controller_type)

    def _get_modules_for_layer(
        self, layer_idx: int, controller_type: str | None = None
    ) -> list[nn.Module]:
        """Get all controllers for the specified layer."""
        modules = []
        target_class = None
        if controller_type:
            spec = _CONTROLLER_REGISTRY.get(controller_type)
            target_class = spec.controller_class if spec is not None else None
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
