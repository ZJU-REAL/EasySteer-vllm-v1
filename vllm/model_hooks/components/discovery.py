# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Module discovery shared by steering and capture.

Locates decoder layers and sparse-MoE blocks on an arbitrary model.
Discovery uses vLLM component interfaces and decoder-stack contracts,
without model-family names. Output adapters live in outputs.py; token
selection is shared through model_hooks.selection.
"""

from dataclasses import dataclass
from functools import cached_property
from inspect import signature

from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class DiscoveredLayer:
    """One model module and its global decoder-stack index."""

    name: str
    module: nn.Module
    layer_id: int


class ModelDiscovery:
    """One model's structural discovery, shared by steering and capture."""

    def __init__(self, model: nn.Module):
        self.model = model

    @cached_property
    def stack_indices(self) -> dict[str, int]:
        return _stack_indices(self.model)

    @cached_property
    def decoder_layers(self) -> list[DiscoveredLayer]:
        return _find_decoder_layers(self.model, self.stack_indices)

    @cached_property
    def moe_blocks(self) -> list[DiscoveredLayer]:
        return _find_moe_blocks(self.model, self.stack_indices, self.decoder_layers)


def _stack_indices(model: nn.Module) -> dict[str, int]:
    from vllm.model_executor.models.utils import PPMissingLayer

    indices = {}
    for name, stack in model.named_modules():
        if not isinstance(stack, nn.ModuleList):
            continue
        parent_name = name.rpartition(".")[0]
        parent = model.get_submodule(parent_name) if parent_name else model
        start = getattr(parent, "start_layer", 0)
        end = getattr(parent, "end_layer", None)
        # make_layers retains PPMissingLayer placeholders and global indices.
        # A compact local stack instead declares its global [start, end).
        offset = (
            start
            if isinstance(start, int)
            and isinstance(end, int)
            and len(stack) == end - start
            else 0
        )
        for index, child in enumerate(stack):
            if not isinstance(child, PPMissingLayer):
                indices[f"{name}.{index}" if name else str(index)] = offset + index
    return indices


def _layer_record(name, module, stack_indices, decoder_layers=()):
    owners = [
        layer
        for layer in decoder_layers
        if name == layer.name or name.startswith(layer.name + ".")
    ]
    if owners:
        layer_id = max(owners, key=lambda layer: len(layer.name)).layer_id
    elif name in stack_indices:
        layer_id = stack_indices[name]
    else:
        explicit = [getattr(module, attr, None) for attr in ("layer_idx", "layer_id")]
        explicit = [value for value in explicit if type(value) is int and value >= 0]
        if explicit:
            if len(set(explicit)) != 1:
                raise ValueError(f"Conflicting layer indices on {name!r}: {explicit}")
            layer_id = explicit[0]
        else:
            numbers = [int(part) for part in name.split(".") if part.isdigit()]
            if len(numbers) != 1:
                raise ValueError(
                    f"Cannot determine a global layer index for {name!r}; "
                    "use an indexed decoder stack or an explicit layer_idx"
                )
            layer_id = numbers[0]
    return DiscoveredLayer(name, module, layer_id)


def _find_decoder_layers(
    model: nn.Module, indices: dict[str, int]
) -> list[DiscoveredLayer]:
    """Find decoder blocks by vLLM attention/Mamba and MoE interfaces.

    Residual-stream blocks supplement anchored or explicitly bounded stacks,
    including hybrid models' pure MLP layers. Other siblings are skipped.
    """
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
        MoERunnerInterface,
    )

    modules = dict(model.named_modules(remove_duplicate=False))
    anchors = (AttentionLayerBase, MoERunnerInterface)
    matches = {
        name: modules[name]
        for name in indices
        if not isinstance(modules[name], anchors)
        and any(isinstance(child, anchors) for child in modules[name].modules())
    }
    # A nested stack owns its blocks; its enclosing ModuleList entry does not.
    matches = {
        name: module
        for name, module in matches.items()
        if not any(other.startswith(name + ".") for other in matches)
    }
    stacks = {name.rpartition(".")[0] for name in matches}
    for name in indices:
        stack_name = name.rpartition(".")[0]
        parent = modules[stack_name.rpartition(".")[0]]
        start = getattr(parent, "start_layer", None)
        end = getattr(parent, "end_layer", None)
        if type(start) is int and type(end) is int and 0 <= start < end:
            stacks.add(stack_name)
    for name in indices:
        module = modules[name]
        if name in matches or name.rpartition(".")[0] not in stacks:
            continue
        if not _has_residual_stream(module):
            continue
        if any(
            other.startswith(name + ".") or name.startswith(other + ".")
            for other in matches
        ):
            continue
        matches[name] = module
    if not matches:
        # A non-ModuleList decoder can declare its index directly.
        matches = {
            name: module
            for name, module in modules.items()
            if not isinstance(module, anchors)
            and _has_explicit_layer_index(module)
            and _has_residual_stream(module)
        }
    for name in indices:
        if name.rpartition(".")[0] in stacks and name not in matches:
            logger.warning_once(
                "Decoder stack contains an unrecognized block %s (%s); "
                "steering and capture will skip it.",
                name,
                type(modules[name]).__name__,
            )
    records = [_layer_record(name, module, indices) for name, module in matches.items()]
    seen = {}
    seen_modules = {}
    for layer in records:
        if layer.layer_id in seen:
            raise ValueError(
                f"Ambiguous decoder layer index {layer.layer_id}: "
                f"{seen[layer.layer_id]!r} and {layer.name!r}"
            )
        seen[layer.layer_id] = layer.name
        if id(layer.module) in seen_modules:
            raise ValueError(
                f"Decoder module is shared by {seen_modules[id(layer.module)]!r} "
                f"and {layer.name!r}; hooks cannot distinguish their layer indices"
            )
        seen_modules[id(layer.module)] = layer.name
    return sorted(records, key=lambda layer: layer.layer_id)


def _has_residual_stream(module: nn.Module) -> bool:
    parameters = signature(module.forward).parameters
    return "hidden_states" in parameters and "residual" in parameters


def _has_explicit_layer_index(module: nn.Module) -> bool:
    return any(
        type(index := getattr(module, attr, None)) is int and index >= 0
        for attr in ("layer_idx", "layer_id")
    )


def _gate_module(module: nn.Module) -> nn.Module | None:
    for attr in ("gate", "router"):
        gate = getattr(module, attr, None)
        if isinstance(gate, nn.Module):
            return gate
    return None


def _has_routed_expert_layout(module: nn.Module) -> bool:
    """Recognize blocks that invoke expert kernels without a vLLM runner."""
    top_k = getattr(module, "top_k", None)
    if type(top_k) is not int or top_k <= 0 or _gate_module(module) is None:
        return False
    return any(
        type(count := getattr(module, attr, None)) is int and count >= top_k
        for attr in ("num_experts", "num_total_experts")
    )


def _find_moe_blocks(
    model: nn.Module, indices: dict[str, int], decoders: list[DiscoveredLayer]
) -> list[DiscoveredLayer]:
    """Find MoE blocks and use their owning decoder's global index."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
        MoERunnerInterface,
    )

    matches = {
        name: module
        for name, module in model.named_modules()
        if not isinstance(module, MoERunnerInterface)
        and (
            any(isinstance(child, MoERunnerInterface) for child in module.children())
            or _has_routed_expert_layout(module)
        )
    }
    matches = {
        name: module
        for name, module in matches.items()
        if not any(other.startswith(name + ".") for other in matches)
    }
    return [
        _layer_record(name, module, indices, decoders)
        for name, module in matches.items()
    ]


def resolve_moe_gate(module_name: str, moe_block: nn.Module) -> nn.Module | None:
    """Resolve a gate whose forward is usable by both steering and capture."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
        MoERunnerInterface,
    )

    runners = [
        child for child in moe_block.children() if isinstance(child, MoERunnerInterface)
    ]
    if any(getattr(runner, "_fse_fuse_gate", False) for runner in runners):
        logger.warning_once(
            "MoE block %s fuses gate weights into the runner and bypasses gate "
            "forward; router-logit steering and capture are unavailable.",
            module_name,
        )
        return None
    for owner in (moe_block, *runners):
        gate = _gate_module(owner)
        if gate is not None:
            return gate
    logger.warning_once(
        "MoE block %s has no gate/router module; router-logit steering and "
        "capture are unavailable.",
        module_name,
    )
    return None
