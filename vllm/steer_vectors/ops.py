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

# Names used in CompilationConfig.splitting_ops.
STEER_APPLY_OP = "vllm::steer_apply"
STEER_MOE_GATE_OP = "vllm::steer_moe_gate"

# layer key (module name) -> controller for the currently loaded model.
# One model per worker process; re-registration on model reload overwrites.
_CONTROLLERS: dict[str, "DecoderLayerWithSteerVector"] = {}


def register_controller(key: str, controller: "DecoderLayerWithSteerVector") -> None:
    _CONTROLLERS[key] = controller


def unregister_controller(key: str) -> None:
    _CONTROLLERS.pop(key, None)


def _steer_dispatch(tensor: torch.Tensor, layer_name: str, method_name: str) -> None:
    """Apply the controller's steering method to `tensor` in place."""
    controller = _CONTROLLERS.get(layer_name)
    if controller is None:
        return
    result = getattr(controller, method_name)(tensor)
    if result is not None and result is not tensor:
        tensor.copy_(result)


def _register_steer_op(op_name: str, arg_name: str, method_name: str) -> None:
    """Register `vllm::{op_name}({arg_name}, layer_name)` and its no-op fake.

    The op mutates its tensor argument in place via the layer's
    registered controller. The tensor parameter name is load-bearing:
    direct_register_custom_op infers the op schema from the impl's real
    signature, and `mutates_args` must name that parameter — so the op
    and fake functions are generated with the exact signature.
    """
    src = (
        f"def {op_name}({arg_name}: torch.Tensor, layer_name: str) -> None:\n"
        f"    _steer_dispatch({arg_name}, layer_name, {method_name!r})\n"
        f"def {op_name}_fake({arg_name}: torch.Tensor, layer_name: str) -> None:\n"
        f"    return\n"
    )
    ns = {"torch": torch, "_steer_dispatch": _steer_dispatch}
    exec(src, ns)
    direct_register_custom_op(
        op_name=op_name,
        op_func=ns[op_name],
        mutates_args=[arg_name],
        fake_impl=ns[f"{op_name}_fake"],
    )


# steer_apply: decoder-layer steering on `hidden` (complete hidden
# states, i.e. hidden + residual), in place.
_register_steer_op("steer_apply", "hidden", "apply_steering")
# steer_moe_gate: MoE router-logits steering on `logits`, in place.
_register_steer_op("steer_moe_gate", "logits", "apply_gate_steering")
