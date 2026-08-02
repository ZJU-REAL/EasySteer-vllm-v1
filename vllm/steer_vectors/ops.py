# SPDX-License-Identifier: Apache-2.0
"""Custom-op entry point for steer-vector application.

``vllm::steer_apply`` is the single opaque op through which decoder-layer
steering runs. In eager mode it dispatches straight to the layer's
controller. Under compiled execution the hook body is traced by dynamo and
the op lands in the FX graph as an opaque node; it is then registered as a
piecewise splitting op (see ``VllmConfig.__post_init__``) so it executes
eagerly between CUDA-graph segments while all steering configuration stays
out of the captured graphs.
"""

from typing import TYPE_CHECKING

import torch

from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.steer_vectors.layers import DecoderLayerWithSteerVector

# Name used in CompilationConfig.splitting_ops.
STEER_APPLY_OP = "vllm::steer_apply"

# layer key (module name) -> controller for the currently loaded model.
# One model per worker process; re-registration on model reload overwrites.
_CONTROLLERS: dict[str, "DecoderLayerWithSteerVector"] = {}


def register_controller(key: str,
                        controller: "DecoderLayerWithSteerVector") -> None:
    _CONTROLLERS[key] = controller


def unregister_controller(key: str) -> None:
    _CONTROLLERS.pop(key, None)


def steer_apply(hidden: torch.Tensor, layer_name: str) -> None:
    """Apply steering for `layer_name` to `hidden` (complete hidden states,
    i.e. hidden + residual) in place."""
    controller = _CONTROLLERS.get(layer_name)
    if controller is None:
        return
    result = controller.apply_steering(hidden)
    if result is not None and result is not hidden:
        hidden.copy_(result)


def steer_apply_fake(hidden: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="steer_apply",
    op_func=steer_apply,
    mutates_args=["hidden"],
    fake_impl=steer_apply_fake,
)

# Name used in CompilationConfig.splitting_ops.
STEER_MOE_GATE_OP = "vllm::steer_moe_gate"


def steer_moe_gate(logits: torch.Tensor, layer_name: str) -> None:
    """Apply MoE router-logits steering for `layer_name` in place."""
    controller = _CONTROLLERS.get(layer_name)
    if controller is None:
        return
    result = controller.apply_gate_steering(logits)
    if result is not None and result is not logits:
        logits.copy_(result)


def steer_moe_gate_fake(logits: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="steer_moe_gate",
    op_func=steer_moe_gate,
    mutates_args=["logits"],
    fake_impl=steer_moe_gate_fake,
)
