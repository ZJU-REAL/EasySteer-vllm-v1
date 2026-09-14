# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Steering controller attachment.

`ControllerManager` attaches steering hooks to the component directory
shared with capture.
"""

from vllm.logger import init_logger
from vllm.model_hooks.components.registry import (
    ATTENTION_HEADS,
    ModelComponents,
    get_component,
)
from vllm.model_hooks.steering import ops as steer_ops

from .base import SteeringController
from .hidden_states import HiddenStatesController
from .router_logits import RouterLogitsController

logger = init_logger(__name__)


_CONTROLLER_REGISTRY: dict[str, type[SteeringController]] = {
    controller.component_id: controller
    for controller in (HiddenStatesController, RouterLogitsController)
}
_CONTROLLER_REGISTRY[ATTENTION_HEADS] = HiddenStatesController


class ControllerManager:
    """Owns steering controllers indexed by component and global layer id.

    On construction it registers a forward hook per resolved target and
    adds each controller to the custom-op registry. Original modules stay in the tree
    untouched (module names, classes and state-dict keys are preserved,
    keeping FSDP/checkpointing consumers such as VERL working);
    controllers live outside the model.
    """

    def __init__(self, components: ModelComponents):
        self.controllers: dict[str, SteeringController] = {}
        self._layer_controllers: dict[tuple[str, int], list[SteeringController]] = {}
        self._hook_handles: list = []
        try:
            for component_id, targets in components.items():
                self._attach_controllers(component_id, targets)
        except Exception:
            self.remove_hooks()
            raise

    def _attach_controllers(self, component_id: str, targets) -> None:
        """Attach controllers to the shared component directory."""
        controller_class = _CONTROLLER_REGISTRY[component_id]
        component = get_component(component_id)

        hooked_count = 0
        for layer in targets:
            module_name = layer.name
            target = layer.module
            op_key = module_name + component.op_key_suffix
            if op_key in self.controllers:
                continue
            controller = controller_class()
            controller.component_id = component_id
            controller.layer_id = layer.layer_id
            controller._output_width = layer.width
            controller._global_output_width = layer.global_width
            controller._feature_start = layer.feature_start
            controller._op_key = op_key
            # Keep a reference to the hooked module without registering
            # it as a submodule (that would cycle the module tree); the
            # MoE gate graph tables read their expert count from it.
            object.__setattr__(controller, "hook_target", target)
            self.controllers[op_key] = controller
            steer_ops.register_controller(op_key, controller)
            self._hook_handles.append(
                target.register_forward_hook(controller.process_output_hook)
            )
            self._layer_controllers.setdefault(
                (component_id, layer.layer_id), []
            ).append(controller)
            hooked_count += 1
            logger.debug("Hooked %s: %s", component.id, module_name)

        if hooked_count > 0:
            logger.debug(
                "Using %s-level steering (%d modules hooked)",
                component.id,
                hooked_count,
            )
        else:
            logger.warning("No %s modules found for steering", component.id)

    def controllers_for_layer(
        self, layer_id: int, component_id: str
    ) -> list[SteeringController]:
        """Look up controllers by their public component id and layer index."""
        get_component(component_id)
        return self._layer_controllers.get((component_id, layer_id), [])

    def remove_hooks(self) -> None:
        """Detach hooks and release this manager's model and payload references."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for module_name, controller in self.controllers.items():
            steer_ops.unregister_controller(module_name, controller)
            object.__setattr__(controller, "hook_target", None)
            controller.slot_interventions.clear()
            controller.slot_position_groups.clear()
        self.controllers.clear()
        self._layer_controllers.clear()
