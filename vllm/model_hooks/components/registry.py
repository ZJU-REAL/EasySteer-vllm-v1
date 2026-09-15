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
ATTENTION_HEADS = "attention_heads"


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


def _module_target(name: str, module: nn.Module) -> nn.Module:
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
        _module_target,
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
    ATTENTION_HEADS: ComponentDescriptor(
        ATTENTION_HEADS,
        attrgetter("attention_heads"),
        _module_target,
        DecoderOutputAdapter,
        steering_op="vllm::steer_apply",
        op_key_suffix="::attention_heads",
    ),
}


@dataclass(frozen=True)
class ComponentTarget:
    """One hook target with a global layer index and local output geometry."""

    name: str
    layer_id: int
    module: nn.Module
    width: int | None = None
    num_heads: int | None = None
    # Per-head output width, which can differ from the query/key head size.
    head_size: int | None = None
    # Feature partition owned by this module, independent of engine topology.
    tp_rank: int = 0
    tp_size: int = 1

    @property
    def global_width(self) -> int | None:
        return None if self.width is None else self.width * self.tp_size

    @property
    def global_num_heads(self) -> int | None:
        return None if self.num_heads is None else self.num_heads * self.tp_size

    @property
    def feature_start(self) -> int:
        return (self.width or 0) * self.tp_rank


ModelComponents = dict[str, tuple[ComponentTarget, ...]]


def _decoder_width(model: nn.Module, layer: DiscoveredLayer) -> int | None:
    from vllm.model_executor.layers.layernorm import RMSNorm

    widths = set()
    width = getattr(layer.module, "hidden_size", None)
    if type(width) is int and width > 0:
        widths.add(width)
    for child in layer.module.children():
        if isinstance(child, RMSNorm):
            width = child.hidden_size
        elif isinstance(child, (nn.LayerNorm, nn.RMSNorm)):
            shape = child.normalized_shape
            width = shape[0] if len(shape) == 1 else None
        else:
            continue
        if type(width) is int and width > 0:
            widths.add(width)
    if widths:
        return next(iter(widths)) if len(widths) == 1 else None

    name = layer.name
    while True:
        owner = model.get_submodule(name) if name else model
        config = getattr(owner, "config", None)
        width = getattr(config, "hidden_size", None)
        if type(width) is int and width > 0:
            return width
        if not name:
            return None
        name = name.rpartition(".")[0]


def _router_width(target: nn.Module) -> int | None:
    from vllm.model_executor.layers.linear import ReplicatedLinear

    target = getattr(target, "base_layer", target)
    if isinstance(target, ReplicatedLinear):
        # Quantized weights may be packed; the declared output size remains
        # valid and is replicated even when the layer knows its engine TP group.
        width = target.output_size
    elif isinstance(target, nn.Linear):
        width = target.out_features
    else:
        return None
    return width if type(width) is int and width > 0 else None


def discover_components(model: nn.Module) -> ModelComponents:
    """Resolve each component once for both steering and capture hooks."""
    discovery = ModelDiscovery(model)
    result = {}
    for component in COMPONENTS.values():
        targets = []
        for layer in component.discover(discovery):
            target = component.resolve_target(layer.name, layer.module)
            if target is not None:
                layout = {}
                if component.id == ATTENTION_HEADS:
                    layout = {
                        "width": target.num_heads * target.head_size_v,
                        "num_heads": target.num_heads,
                        "head_size": target.head_size_v,
                        "tp_rank": layer.tp_rank,
                        "tp_size": layer.tp_size,
                    }
                elif component.id == HIDDEN_STATES:
                    layout = {"width": _decoder_width(model, layer)}
                elif component.id == ROUTER_LOGITS:
                    layout = {"width": _router_width(target)}
                targets.append(
                    ComponentTarget(layer.name, layer.layer_id, target, **layout)
                )
        result[component.id] = tuple(targets)
    return result


def get_component(name: str) -> ComponentDescriptor:
    try:
        return COMPONENTS[name]
    except KeyError:
        raise ValueError(f"Unknown steering/capture component: {name!r}") from None
