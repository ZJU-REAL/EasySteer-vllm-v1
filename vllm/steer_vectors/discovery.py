# SPDX-License-Identifier: Apache-2.0
"""Model introspection shared by steering and capture.

Three groups of helpers:

- Module discovery: locate decoder layers and sparse-MoE blocks on an
  arbitrary model. Structural discovery (anchor-based, architecture
  agnostic) is primary; the per-family class-name lists are the fallback
  for layouts it cannot identify.
- Module I/O: split and rebuild decoder-layer outputs across the tuple /
  tensor / object formats different model families return, and pull
  router logits out of a gate module's output.
- Forward-pass metadata: per-sample boundaries and phases for the
  current batch, read from the forward context.
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


def split_decoder_output(output):
    """Split a decoder-layer output into its parts.

    Handles the output formats different model families return:
    (hidden_states, residual) tuples (Qwen2 and similar), bare tensors
    (Phi and similar), longer tuples, and objects with a hidden_states
    attribute.

    Returns:
        (hidden_states, residual, other_outputs, original_format), where
        original_format is the tag `reconstruct_decoder_output` needs to
        rebuild the output.
    """
    if isinstance(output, tuple):
        if len(output) == 2:
            hidden_states, residual = output
            if (
                isinstance(hidden_states, torch.Tensor)
                and isinstance(residual, torch.Tensor)
                and hidden_states.shape == residual.shape
            ):
                return hidden_states, residual, None, "tuple_2"
            # Shapes differ, so this is not a (hidden_states, residual) pair.
            return output[0], None, output[1:], "tuple_other"
        elif len(output) > 2:
            return output[0], None, output[1:], "tuple_multi"
        else:
            return output[0], None, None, "tuple_1"
    elif isinstance(output, torch.Tensor):
        return output, None, None, "tensor"
    if hasattr(output, "hidden_states"):
        return output.hidden_states, getattr(output, "residual", None), output, "object"
    return output, None, None, "unknown"


def reconstruct_decoder_output(
    modified_hidden_states, residual, other_outputs, original_format, original_output
):
    """Rebuild a decoder-layer output split by `split_decoder_output`."""
    if original_format == "tuple_2":
        return (modified_hidden_states, residual)
    elif original_format in ("tuple_other", "tuple_multi"):
        return (modified_hidden_states,) + other_outputs
    elif original_format == "tuple_1":
        return (modified_hidden_states,)
    elif original_format == "tensor":
        return modified_hidden_states
    elif original_format == "object":
        if hasattr(original_output, "hidden_states"):
            original_output.hidden_states = modified_hidden_states
        return original_output
    return modified_hidden_states


def extract_gate_logits(output):
    """Pull the router-logits tensor out of a gate module's output.

    vLLM linear layers typically return (logits, bias); some models
    return a bare tensor.
    """
    if isinstance(output, tuple):
        output = output[0]
    return output if isinstance(output, torch.Tensor) else None


def extract_samples_info(attn_metadata) -> dict[str, torch.Tensor] | None:
    """Extract sample boundaries and phases from the forward context.

    Args:
        attn_metadata: Attention metadata from forward context

    Returns:
        Dict with GPU tensors:
            'query_start_loc': [num_samples+1] tensor of sample boundaries
            'num_computed': [num_samples] tensor of cached token counts (or None)
            'is_decode_mask': [num_samples] boolean tensor (True for decode samples)
            'num_output_tokens': [num_samples] tensor of generated counts (or None)
        or None if query_start_loc is unavailable
    """
    # Primary path: the runner's BatchGeometry on the forward context —
    # scheduler ground truth, backend-agnostic, one producer.
    ctx = forward_context_or_none()
    geo = getattr(ctx, "batch_geometry", None) if ctx is not None else None
    if geo is not None:
        return geo.samples_info()

    # Fallback: backend attention metadata carries only query_start_loc
    # (no real backend metadata class provides the per-request counts),
    # so phases degrade to the single-token-chunk heuristic.
    query_start_loc = _attn_metadata_field(attn_metadata, "query_start_loc")
    if query_start_loc is None or len(query_start_loc) <= 1:
        return None
    starts = query_start_loc[:-1]
    ends = query_start_loc[1:]
    return {
        "query_start_loc": query_start_loc,
        "num_computed": None,
        "is_decode_mask": (ends - starts) == 1,
        "num_output_tokens": None,
        "num_prompt_tokens": None,
    }


def resolve_batch_positions(
    samples_info: dict[str, torch.Tensor],
    total_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Map the step's flat token indices to per-sample coordinates.

    Returns ``(sample_ids, abs_positions, num_computed)`` for the first
    `total_tokens` rows of the batch: each token's sample index, its
    absolute position in that sample (chunk-relative position plus the
    sample's cached-token count when available), and `num_computed` as a
    device tensor (coerced from a list if needed) or None.
    """
    query_start_loc = samples_info["query_start_loc"].to(device)
    num_computed = samples_info["num_computed"]
    if num_computed is not None:
        if not isinstance(num_computed, torch.Tensor):
            num_computed = torch.tensor(num_computed, device=device, dtype=torch.long)
        else:
            num_computed = num_computed.to(device)
    all_positions = torch.arange(total_tokens, device=device)
    sample_ids = torch.searchsorted(query_start_loc, all_positions, right=True) - 1
    relative_positions = all_positions - query_start_loc[:-1][sample_ids]
    if num_computed is not None:
        abs_positions = relative_positions + num_computed[sample_ids]
    else:
        abs_positions = relative_positions
    return sample_ids, abs_positions, num_computed


def forward_context_or_none():
    """The current ForwardContext, or None outside a forward pass."""
    if get_forward_context is None:
        return None
    try:
        return get_forward_context()
    except AssertionError:
        # get_forward_context asserts when no context is set.
        return None


# Backwards-compat alias for the pre-promotion private name.
_forward_context_or_none = forward_context_or_none


def _attn_metadata_field(attn_metadata, field: str):
    """Read a field from per-layer attention metadata (dict or object)."""
    if isinstance(attn_metadata, dict):
        if not attn_metadata:
            return None
        attn_metadata = next(iter(attn_metadata.values()))
    return getattr(attn_metadata, field, None)

