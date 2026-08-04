# SPDX-License-Identifier: Apache-2.0
"""Module discovery shared by steering and capture.

Locates decoder layers and sparse-MoE blocks on an arbitrary model.
Structural discovery (anchor-based, architecture agnostic) is primary;
the per-family class-name lists are the fallback for layouts it cannot
identify. Module I/O and batch-geometry helpers live in geometry.py.
"""

from collections.abc import Callable

import torch
from torch import nn

from vllm.logger import init_logger

try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None

logger = init_logger(__name__)

SUPPORTED_DECODER_LAYERS: list[str] = [
    # A
    "ApertusDecoderLayer",
    "ArceeDecoderLayer",
    "ArcticDecoderLayer",
    "AriaTextDecoderLayer",
    # B
    "BaiChuanDecoderLayer",
    "BailingMoeBlock",
    "BambaAttentionDecoderLayer",
    "BambaMixerDecoderLayer",
    "BertLayer",
    "BloomBlock",
    # C
    "ChameleonDecoderLayer",
    "ChameleonSwinDecoderLayer",
    "CohereDecoderLayer",
    # D
    "DbrxBlock",
    "DeciLMDecoderLayer",
    "DecoderLayer",
    "DeepseekDecoderLayer",
    "DeepseekV2DecoderLayer",
    "Dots1DecoderLayer",
    # E
    "Ernie4_5_MoeDecoderLayer",
    "Ernie4_5_VLMoeDecoderLayer",
    "Exaone4DecoderLayer",
    "ExaoneDecoderLayer",
    # F
    "FalconDecoderLayer",
    "FalconH1AttentionDecoderLayer",
    "FalconH1SSMDecoderLayer",
    "FlashDecoderLayer",
    "FlexOlmoDecoderLayer",
    # G
    "Gemma2DecoderLayer",
    "Gemma3DecoderLayer",
    "Gemma3nDecoderLayer",
    "GemmaDecoderLayer",
    "Glm4DecoderLayer",
    "Glm4MoeDecoderLayer",
    "GLMBlock",
    "GPT2Block",
    "GPTBigCodeBlock",
    "GPTJBlock",
    "GPTNeoXLayer",
    "GraniteDecoderLayer",
    "GraniteMoeDecoderLayer",
    "GraniteMoeHybridAttentionDecoderLayer",
    "GraniteMoeHybridMambaDecoderLayer",
    "GraniteMoeSharedDecoderLayer",
    "Grok1DecoderLayer",
    # H
    "HunYuanDecoderLayer",
    # I
    "InternLM2VEDecoderLayer",
    "InternLMDecoderLayer",
    # J
    "JAISBlock",
    "JambaAttentionDecoderLayer",
    "JambaMambaDecoderLayer",
    # L
    "Lfm2AttentionDecoderLayer",
    "Lfm2MoeAttentionDecoderLayer",
    "Lfm2MoeShortConvDecoderLayer",
    "Lfm2ShortConvDecoderLayer",
    "Llama4DecoderLayer",
    "LlamaDecoderLayer",
    # M
    "Mamba2DecoderLayer",
    "MambaDecoderLayer",
    "MiniCPM3DecoderLayer",
    "MiniCPMDecoderLayer",
    "MiniMaxText01DecoderLayer",
    "MixtralDecoderLayer",
    "MolmoDecoderLayer",
    "MolmoDecoderNormAfterLayer",
    "MPTBlock",
    # N
    "NemotronDecoderLayer",
    "NemotronHAttentionDecoderLayer",
    "NemotronHMambaDecoderLayer",
    "NemotronHMLPDecoderLayer",
    "NemotronHMoEDecoderLayer",
    # O
    "Olmo2DecoderLayer",
    "OlmoDecoderLayer",
    "OlmoeDecoderLayer",
    "OPTDecoderLayer",
    "OrionDecoderLayer",
    # P
    "PersimmonDecoderLayer",
    "PhiLayer",
    "PhiMoEDecoderLayer",
    "Plamo2DecoderLayer",
    # Q
    "Qwen2DecoderLayer",
    "Qwen2MoeDecoderLayer",
    "Qwen3DecoderLayer",
    "Qwen3MoeDecoderLayer",
    "Qwen3NextDecoderLayer",
    "QWenBlock",
    # S
    "SeedOssDecoderLayer",
    "SolarDecoderLayer",
    "StablelmDecoderLayer",
    "Starcoder2DecoderLayer",
    "Step3TextDecoderLayer",
    # T
    "TransformerBlock",
    # W
    "WhisperDecoderLayer",
    # Z
    "Zamba2AttentionDecoderLayer",
    "Zamba2HybridLayer",
    "Zamba2MambaDecoderLayer",
]


SUPPORTED_MOE_LAYERS: list[str] = [
    # Qwen family
    "Qwen2MoeSparseMoeBlock",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3NextSparseMoeBlock",
    "QwenMoE",
    "Qwen2MoE",
    # Mixtral / Llama family
    "MixtralMoE",
    "Llama4MoE",
    "PhiMoE",
    # DeepSeek family
    "DeepseekMoE",
    "DeepseekV2MoE",
    # Kimi
    "KimiMoE",
    # GLM
    "Glm4MoE",
    "GLMMoE",
    # Ernie
    "Ernie4MoE",
    "Ernie4_5_MoeMoE",
    "Ernie4_5_VLMoeMoE",
    # Others
    "DbrxExperts",
    "DbrxMoE",
    "ArcticMoE",
    "JambaMoE",
    "Grok1MoE",
    "GraniteMoeMoE",
    "MiniMaxText01MoE",
    "MiniMaxM2MoE",
    "MiniCPMMoE",
    "OlmoeMoE",
    "FlexOlmoMoE",
    "NemotronHMoE",
    "BailingMoE",
    "Dots1MoE",
    "NomicMoE",
]


def find_decoder_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Find decoder-layer modules without relying on class-name lists.

    Matches the direct children of any ModuleList that contain a
    KV-cache-backed attention layer (AttentionLayerBase). Vision towers
    and other encoder stacks use plain attention modules, so this
    anchors on the decoder stack only.
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


def find_moe_blocks(model: nn.Module) -> dict[str, nn.Module]:
    """Find sparse-MoE block modules by their fused-MoE child.

    A MoE block is a module with a direct fused-MoE child (the gate +
    experts runner). Since vLLM 0.22 `FusedMoE(...)` is a factory
    returning a MoERunner, so we anchor on MoERunnerInterface, with a
    class-name check as a safety net for older/newer layouts.
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


def find_layers_with_fallback(
    model: nn.Module,
    structural_finder: Callable[[nn.Module], dict[str, nn.Module]],
    class_names: list[str],
    kind: str,
) -> dict[str, nn.Module]:
    """Structural discovery with the class-name-list fallback.

    Structural discovery (anchor-based, architecture agnostic) is
    primary; the per-family class-name lists are the fallback for
    layouts it cannot identify.
    """
    matches = structural_finder(model)
    if matches:
        return matches
    matches = {
        module_name: module
        for module_name, module in model.named_modules()
        if any(
            class_name in module.__class__.__name__ for class_name in class_names
        )
    }
    if matches:
        logger.info(
            "Structural discovery found no %s modules; using the "
            "class-name list fallback (%d modules, e.g. %s).",
            kind,
            len(matches),
            next(iter(matches)),
        )
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


def extract_layer_id_from_module_name(module_name: str) -> int | None:
    """Extract the layer index from a module name.

    Examples:
        'model.layers.0' -> 0
        'transformer.h.12' -> 12
        'model.embed_tokens' -> None
    """
    for part in module_name.split("."):
        if part.isdigit():
            return int(part)
    return None
