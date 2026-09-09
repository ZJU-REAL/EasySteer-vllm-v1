# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model component contracts shared by steering and capture."""

from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter

from torch import nn

from vllm.model_hooks.components.discovery import (
    DiscoveredLayer,
    ModelDiscovery,
    resolve_moe_gate,
)
from vllm.model_hooks.components.outputs import (
    extract_gate_logits,
    reconstruct_decoder_output,
    split_decoder_output,
)

HIDDEN_STATES = "hidden_states"
ROUTER_LOGITS = "router_logits"


class DecoderOutputAdapter:
    read_output = staticmethod(split_decoder_output)
    write_output = staticmethod(reconstruct_decoder_output)

    @staticmethod
    def capture_rows(output):
        hidden, residual, _, _ = split_decoder_output(output)
        return (hidden + residual, True) if residual is not None else (hidden, False)


class GateOutputAdapter:
    @staticmethod
    def read_output(output):
        # A linear gate's second output is bias, never a residual stream.
        return extract_gate_logits(output), None, None, "gate"

    @staticmethod
    def write_output(values, residual, auxiliary, original_format, original_output):
        if values is extract_gate_logits(original_output):
            return original_output
        if isinstance(original_output, tuple):
            return (values,) + original_output[1:]
        return values

    @staticmethod
    def capture_rows(output):
        return extract_gate_logits(output), False


def _decoder_target(name: str, module: nn.Module) -> nn.Module:
    return module


@dataclass(frozen=True)
class ComponentDescriptor:
    id: str
    discover: Callable[[ModelDiscovery], list[DiscoveredLayer]]
    # None means the discovered component has no usable forward hook;
    # the resolver reports the reason (e.g. a fused MoE gate is bypassed).
    resolve_target: Callable[[str, nn.Module], nn.Module | None]
    adapter: type[DecoderOutputAdapter] | type[GateOutputAdapter]
    steering_op: str
    op_key_suffix: str = ""


COMPONENTS = {
    HIDDEN_STATES: ComponentDescriptor(
        HIDDEN_STATES,
        attrgetter("decoder_layers"),
        _decoder_target,
        DecoderOutputAdapter,
        steering_op="vllm::steer_apply",
    ),
    ROUTER_LOGITS: ComponentDescriptor(
        ROUTER_LOGITS,
        attrgetter("moe_blocks"),
        resolve_moe_gate,
        GateOutputAdapter,
        steering_op="vllm::steer_moe_gate",
        op_key_suffix="::gate",
    ),
}


@dataclass(frozen=True)
class ComponentTarget:
    """One usable hook target with its model name and global layer index."""

    name: str
    layer_id: int
    module: nn.Module


ModelComponents = dict[str, tuple[ComponentTarget, ...]]


def discover_components(model: nn.Module) -> ModelComponents:
    """Resolve each component once for both steering and capture hooks."""
    discovery = ModelDiscovery(model)
    result = {}
    for component in COMPONENTS.values():
        targets = []
        for layer in component.discover(discovery):
            target = component.resolve_target(layer.name, layer.module)
            if target is not None:
                targets.append(ComponentTarget(layer.name, layer.layer_id, target))
        result[component.id] = tuple(targets)
    return result


def get_component(name: str) -> ComponentDescriptor:
    try:
        return COMPONENTS[name]
    except KeyError:
        raise ValueError(f"Unknown steering/capture component: {name!r}") from None
