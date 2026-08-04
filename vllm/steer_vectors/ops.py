# SPDX-License-Identifier: Apache-2.0
"""Custom-op entry point for steer-vector application.

``vllm::steer_apply`` is the single opaque op through which decoder-layer
steering runs. In eager mode it dispatches straight to the layer's
controller. Under compiled execution the hook body is traced by dynamo and
the op lands in the FX graph as an opaque node; it is then registered as a
piecewise splitting op (see ``VllmConfig.__post_init__``) so it executes
eagerly between CUDA-graph segments while all steering configuration stays
out of the captured graphs.

Both ops mutate only their first tensor argument, in place, on the
selected token rows; ``steer_apply`` reads ``residual`` so transforms see
the complete hidden state without collapsing the residual stream.
"""

from typing import TYPE_CHECKING

import torch

from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.steer_vectors.controllers import DecoderSteerController

# Names used in CompilationConfig.splitting_ops.
STEER_APPLY_OP = "vllm::steer_apply"
STEER_MOE_GATE_OP = "vllm::steer_moe_gate"

# layer key (module name) -> controller for the currently loaded model.
# One model per worker process; re-registration on model reload overwrites.
_CONTROLLERS: dict[str, "DecoderSteerController"] = {}


def register_controller(key: str, controller: "DecoderSteerController") -> None:
    _CONTROLLERS[key] = controller


def unregister_controller(key: str) -> None:
    _CONTROLLERS.pop(key, None)


def steer_apply(
    hidden: torch.Tensor, residual: torch.Tensor | None, layer_name: str
) -> None:
    controller = _CONTROLLERS.get(layer_name)
    if controller is None:
        return
    controller.apply_steering(hidden, residual)


def steer_apply_fake(
    hidden: torch.Tensor, residual: torch.Tensor | None, layer_name: str
) -> None:
    return


def steer_moe_gate(logits: torch.Tensor, layer_name: str) -> None:
    controller = _CONTROLLERS.get(layer_name)
    if controller is None:
        return
    controller.apply_gate_steering(logits)


def steer_moe_gate_fake(logits: torch.Tensor, layer_name: str) -> None:
    return


direct_register_custom_op(
    op_name="steer_apply",
    op_func=steer_apply,
    mutates_args=["hidden"],
    fake_impl=steer_apply_fake,
)
direct_register_custom_op(
    op_name="steer_moe_gate",
    op_func=steer_moe_gate,
    mutates_args=["logits"],
    fake_impl=steer_moe_gate_fake,
)
