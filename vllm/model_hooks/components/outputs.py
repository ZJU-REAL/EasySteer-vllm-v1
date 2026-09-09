# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read and rebuild model component outputs across model families."""

import torch


def split_decoder_output(output):
    """Split a decoder-layer output into its parts.

    Handles the output formats different model families return:
    (hidden_states, residual) tuples (Qwen2 and similar), bare tensors
    (Phi and similar), and objects with a hidden_states attribute. Longer
    tuples follow the existing (hidden_states, *auxiliary_outputs) contract;
    a residual in another position requires a model-specific adapter.

    Returns:
        (hidden_states, residual, other_outputs, original_format), where
        original_format is the tag `reconstruct_decoder_output` needs to
        rebuild the output.
    """
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], torch.Tensor):
            raise TypeError("Decoder output tuple must start with hidden-state tensor")
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
        hidden, residual = output.hidden_states, getattr(output, "residual", None)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("Decoder output.hidden_states must be a tensor")
        if residual is not None and (
            not isinstance(residual, torch.Tensor) or residual.shape != hidden.shape
        ):
            raise ValueError("Decoder residual must have the hidden-state shape")
        return hidden, residual, output, "object"
    raise TypeError(f"Unsupported decoder output type: {type(output).__name__}")


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
    raise ValueError(f"Unsupported decoder output format: {original_format}")


def extract_gate_logits(output):
    """Pull the router-logits tensor out of a gate module's output.

    vLLM linear layers typically return (logits, bias); some models
    return a bare tensor.
    """
    if isinstance(output, tuple):
        output = output[0] if output else None
    return output if isinstance(output, torch.Tensor) else None
