# SPDX-License-Identifier: Apache-2.0
"""Module I/O and batch geometry shared by steering and capture.

- Module I/O: split and rebuild decoder-layer outputs across the tuple /
  tensor / object formats different model families return, and pull
  router logits out of a gate module's output.
- Forward-pass metadata: per-sample boundaries and phases for the
  current batch, read from the forward context.
"""

import torch

try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


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


def _attn_metadata_field(attn_metadata, field: str):
    """Read a field from per-layer attention metadata (dict or object)."""
    if isinstance(attn_metadata, dict):
        if not attn_metadata:
            return None
        attn_metadata = next(iter(attn_metadata.values()))
    return getattr(attn_metadata, field, None)

