# SPDX-License-Identifier: Apache-2.0

"""Steer vector model loading and hook management.

`SteerVectorModel` holds per-layer payloads loaded from disk;
`SteerVectorModelManager` discovers steerable modules on a model and
registers the steering hooks (module discovery is structural first,
with the class-name lists in config.py as fallback).
"""

import os
from typing import TypeVar

from huggingface_hub import hf_hub_download
from torch import nn

from vllm.config import SteerVectorConfig
from vllm.logger import init_logger
from vllm.steer_vectors import ops as steer_ops
from vllm.steer_vectors.algorithms.factory import ALGORITHM_REGISTRY
from vllm.steer_vectors.config import (
    SUPPORTED_DECODER_LAYERS,
    SUPPORTED_MOE_LAYERS,
)
from vllm.steer_vectors.layers import (
    DecoderLayerWithSteerVector,
    MoEGateSteerController,
    extract_layer_id_from_module_name,
)

logger = init_logger(__name__)

T = TypeVar("T")


def find_decoder_layers_structurally(model: nn.Module) -> dict[str, nn.Module]:
    """Find decoder-layer modules without relying on class-name lists.

    Matches the direct children of any ModuleList that contain a
    KV-cache-backed attention layer (AttentionLayerBase). Vision towers and
    other encoder stacks use plain attention modules, so this anchors on the
    decoder stack only. Used as a fallback for model families that are not
    in the per-family class-name lists.
    """
    from vllm.model_executor.layers.attention_layer_base import (
        AttentionLayerBase,
    )

    matches: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        for child_name, child in module.named_children():
            if any(isinstance(m, AttentionLayerBase) for m in child.modules()):
                matches[f"{name}.{child_name}" if name else child_name] = child
    # Drop outer matches that merely contain other matches (nested stacks).
    inner_only = {
        name: module
        for name, module in matches.items()
        if not any(other != name and other.startswith(f"{name}.") for other in matches)
    }
    return inner_only


def find_moe_blocks_structurally(model: nn.Module) -> dict[str, nn.Module]:
    """Find sparse-MoE block modules by their fused-MoE child.

    Fallback for model families not covered by the class-name lists: a MoE
    block is a module with a direct fused-MoE child (the gate + experts
    runner). Since vLLM 0.22 `FusedMoE(...)` is a factory returning a
    MoERunner, so we anchor on MoERunnerInterface, with a class-name check
    as a safety net for older/newer layouts.
    """
    try:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (  # noqa: E501
            MoERunnerInterface,
        )
    except ImportError:
        MoERunnerInterface = None

    def is_moe_child(child: nn.Module) -> bool:
        if MoERunnerInterface is not None and isinstance(child, MoERunnerInterface):
            return True
        return type(child).__name__ in ("FusedMoE", "MoERunner", "SharedFusedMoE")

    matches: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if any(is_moe_child(c) for c in module.children()):
            matches[name] = module
    return matches


def find_moe_gate(moe_block: nn.Module) -> nn.Module | None:
    """Return the routing (gate) submodule of a sparse-MoE block.

    Architecture-agnostic: virtually every MoE block exposes the router
    as a `gate` or `router` child whose output logits feed top-k expert
    selection.
    """
    for attr in ("gate", "router"):
        gate = getattr(moe_block, attr, None)
        if isinstance(gate, nn.Module):
            return gate
    return None


def moe_gate_is_fused(moe_block: nn.Module) -> bool:
    """Whether the block's MoE runner bypasses the gate module forward.

    When a model provides both a gate and a shared-expert gate, the MoE
    runner fuses their weights and computes routing with a raw ``F.linear``
    (``MoERunner._fse_fuse_gate``) — the gate module is never called, so
    forward hooks on it would silently never fire.
    """
    return any(
        getattr(child, "_fse_fuse_gate", False) for child in moe_block.children()
    )


_all_sv_classes = {
    "decoder_layer": DecoderLayerWithSteerVector,
    "moe_layer": MoEGateSteerController,
}

_STRUCTURAL_FINDERS = {
    "decoder_layer": find_decoder_layers_structurally,
    "moe_layer": find_moe_blocks_structurally,
}

_FALLBACK_CLASS_NAMES = {
    "decoder_layer": SUPPORTED_DECODER_LAYERS,
    "moe_layer": SUPPORTED_MOE_LAYERS,
}


class SteerVectorModel:
    """Represents a steer vector model that can be applied to layers.

    This class encapsulates the data and metadata for loaded steer vectors,
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
        """Check if this is a multi-vector model."""
        return self.multi_vector_data is not None

    # ------------------------------------------------------------------------
    # Factory Methods - Loading from Different Sources
    # ------------------------------------------------------------------------

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
        **kwargs,  # Accept additional algorithm-specific parameters (e.g., moe_lambda)
    ) -> "SteerVectorModel":
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
            Loaded SteerVectorModel instance
        """
        try:
            # Handle algorithm parameter in path (e.g., "path/to/vector|linear")
            if "|" in steer_vector_model_path:
                steer_vector_model_path, path_algorithm = steer_vector_model_path.split(
                    "|", 1
                )
                algorithm = path_algorithm

            # Resolve path (local file or HuggingFace Hub)
            if os.path.exists(steer_vector_model_path):
                file_path = os.path.abspath(steer_vector_model_path)
            else:
                # Download from HuggingFace Hub
                parts = steer_vector_model_path.split("/")
                repo_id = "/".join(parts[:2])
                file_name = "/".join(parts[2:])
                file_path = hf_hub_download(
                    repo_id=repo_id, filename=file_name, revision="main"
                )

            # Dynamically get the algorithm class from the registry
            if algorithm not in ALGORITHM_REGISTRY:
                raise ValueError(f"Unsupported algorithm for loading: '{algorithm}'")

            algo_class = ALGORITHM_REGISTRY[algorithm]

            # Delegate loading to the algorithm's class method
            # Pass kwargs to support algorithm-specific parameters (e.g., moe_lambda)
            loaded_params = algo_class.load_from_path(
                file_path, device, config=config, target_layers=target_layers, **kwargs
            )

            # Create SteerVectorModel instance from loaded parameters
            return cls(
                steer_vector_id=steer_vector_id,
                layer_payloads=loaded_params.get("layer_payloads"),
                scale_factor=scale_factor,
                algorithm=algorithm,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load steer vector from {steer_vector_model_path} "
                f"with algorithm '{algorithm}'"
            ) from e


class SteerVectorModelManager:
    """Manages steer vector models for a given model.

    This is the core orchestrator that:
    1. Wraps model layers with appropriate intervention wrappers
    2. Manages the lifecycle of steer vector adapters
    3. Activates/deactivates vectors at runtime
    4. Coordinates between configuration, wrappers, and algorithms

    The manager is designed to be completely agnostic to specific wrapper
    types - it uses configuration-driven dynamic dispatch for extensibility.
    """

    def __init__(self, model: nn.Module, steer_vector_config: SteerVectorConfig):
        self.model = model
        self.steer_vector_config = steer_vector_config
        self.model.steer_vector_manager = self
        self.modules: dict[str, nn.Module] = {}
        self._hook_handles: list = []
        self._hooked_modules: set[str] = set()
        self._create_sv_modules()

    # ------------------------------------------------------------------------
    # Wrapper Management (Configuration-Driven)
    # ------------------------------------------------------------------------

    def _create_sv_modules(self):
        """Hook every supported wrapper type on the model."""
        for wrapper_type in _all_sv_classes:
            self._wrap_modules_by_type(wrapper_type)

    def _wrap_modules_by_type(self, wrapper_type: str) -> None:
        """Discover and hook all modules of one wrapper type.

        Discovery is structural first (anchor-based, architecture
        agnostic); the per-family class-name lists in config.py are the
        fallback for layouts structural discovery cannot identify.
        """
        wrapper_class = _all_sv_classes[wrapper_type]
        matches = _STRUCTURAL_FINDERS[wrapper_type](self.model)
        if not matches:
            class_names = _FALLBACK_CLASS_NAMES[wrapper_type]
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
                    wrapper_type,
                    len(matches),
                    next(iter(matches)),
                )

        # Hook matching modules: the original modules stay in the tree
        # (module names, classes and state-dict keys are untouched,
        # keeping FSDP/checkpointing consumers such as VERL working);
        # controllers live outside the model.
        wrapped_count = 0
        for module_name, module in matches.items():
            if wrapper_type == "decoder_layer":
                if module_name in self._hooked_modules:
                    continue
                controller = wrapper_class()
                layer_id = extract_layer_id_from_module_name(module_name)
                if layer_id is not None:
                    controller.set_layer_id(layer_id)
                controller._op_key = module_name
                steer_ops.register_controller(module_name, controller)
                handle = module.register_forward_hook(controller.process_output_hook)
                self._hook_handles.append(handle)
                self._hooked_modules.add(module_name)
                self.register_module(module_name, controller)
                wrapped_count += 1
                logger.debug("Hooked %s: %s", wrapper_type, module_name)
            elif wrapper_type == "moe_layer":
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
                controller = wrapper_class()
                layer_id = extract_layer_id_from_module_name(module_name)
                if layer_id is not None:
                    controller.set_layer_id(layer_id)
                controller._op_key = op_key
                steer_ops.register_controller(op_key, controller)
                handle = gate.register_forward_hook(controller.process_gate_output_hook)
                self._hook_handles.append(handle)
                self._hooked_modules.add(op_key)
                self.register_module(module_name, controller)
                wrapped_count += 1
                logger.debug("Hooked %s gate: %s", wrapper_type, module_name)

        # Log summary
        if wrapped_count > 0:
            logger.debug(
                "Using %s-level steer vector intervention (%d modules wrapped)",
                wrapper_type,
                wrapped_count,
            )
        else:
            logger.warning(
                "No %s modules found for steer vector intervention",
                wrapper_type,
            )

    # ------------------------------------------------------------------------
    # Internal Helper Methods
    # ------------------------------------------------------------------------

    def _resolve_wrapper_type(self, algorithm_name: str) -> str:
        """Determine which wrapper type should receive a given algorithm."""
        if algorithm_name == "moe_router":
            return "moe_layer"
        return "decoder_layer"

    def _get_modules_for_layer(
        self, layer_idx: int, wrapper_type: str | None = None
    ) -> list[nn.Module]:
        """Get all modules for the specified layer."""
        modules = []
        target_class = None
        if wrapper_type:
            target_class = _all_sv_classes.get(wrapper_type)
        for module_name, module in self.modules.items():
            if extract_layer_id_from_module_name(module_name) == layer_idx:
                if target_class is not None and not isinstance(module, target_class):
                    continue
                modules.append(module)
        return modules

    def register_module(self, module_name: str, module: nn.Module):
        """Register a wrapped module."""
        self.modules[module_name] = module

    def remove_hooks(self):
        """Detach all forward hooks registered on the model."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for module_name in self._hooked_modules:
            steer_ops.unregister_controller(module_name)
        self._hooked_modules.clear()


def create_sv_manager(
    model: nn.Module,
    steer_vector_config: SteerVectorConfig,
    steer_vector_manager_cls: type[SteerVectorModelManager] = SteerVectorModelManager,
) -> SteerVectorModelManager:
    """Factory function to create a steer vector manager.

    Args:
        model: The neural network model to manage
        steer_vector_config: Configuration for steer vectors
        steer_vector_manager_cls: Manager class to instantiate

    Returns:
        Initialized steer vector manager
    """
    steer_vector_manager = steer_vector_manager_cls(
        model=model, steer_vector_config=steer_vector_config
    )
    return steer_vector_manager
