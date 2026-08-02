# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
from torch import nn

from .algorithms import BaseSteerVectorAlgorithm, create_algorithm

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


def extract_layer_id_from_module_name(module_name: str) -> Optional[int]:
    """
    Extract layer ID from module name.
    
    Args:
        module_name: Module name like 'model.layers.0' or 'transformer.h.12'
    
    Returns:
        Layer ID as integer, or None if not found
    
    Examples:
        'model.layers.0' -> 0
        'transformer.h.12' -> 12
        'model.embed_tokens' -> None
    """
    parts = module_name.split('.')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


class BaseLayerWithSteerVector(nn.Module):
    pass


# Per-request config slots live above this id; legacy adapter slots below.
CONFIG_SLOT_BASE = 1000


def _extract_hidden_states_and_residual(output):
    """
    Extract hidden_states and residual from DecoderLayer output.

    Args:
        output: DecoderLayer output, possible formats:
               - (hidden_states, residual)  # Qwen2 and similar models
               - hidden_states              # Phi and similar models
               - tuple with more elements   # Other possible formats

    Returns:
        (hidden_states, residual, other_outputs, original_format)
    """
    if isinstance(output, tuple):
        if len(output) == 2:
            # Assume (hidden_states, residual) format
            hidden_states, residual = output
            if (isinstance(hidden_states, torch.Tensor) and
                    isinstance(residual, torch.Tensor) and
                    hidden_states.shape == residual.shape):
                return hidden_states, residual, None, "tuple_2"
            else:
                # If shapes don't match, may not be (hidden_states, residual) format
                return output[0], None, output[1:], "tuple_other"
        elif len(output) > 2:
            # More complex tuple, assume first element is hidden_states
            return output[0], None, output[1:], "tuple_multi"
        else:
            # Single-element tuple
            return output[0], None, None, "tuple_1"
    elif isinstance(output, torch.Tensor):
        # Direct tensor output, e.g., Phi models
        return output, None, None, "tensor"
    else:
        # Other formats, try to extract from attributes
        if hasattr(output, 'hidden_states'):
            hidden_states = output.hidden_states
            residual = getattr(output, 'residual', None)
            return hidden_states, residual, output, "object"
        else:
            # Unrecognized format, return original output
            return output, None, None, "unknown"


def _reconstruct_output(modified_hidden_states, residual, other_outputs, original_format, original_output):
    """
    Reconstruct output based on original format.

    Args:
        modified_hidden_states: Modified hidden_states
        residual: Residual (if any)
        other_outputs: Other output elements
        original_format: Original format identifier
        original_output: Original output (for reconstructing complex objects)

    Returns:
        Reconstructed output
    """
    if original_format == "tuple_2":
        return (modified_hidden_states, residual)
    elif original_format == "tuple_other":
        return (modified_hidden_states,) + other_outputs
    elif original_format == "tuple_multi":
        return (modified_hidden_states,) + other_outputs
    elif original_format == "tuple_1":
        return (modified_hidden_states,)
    elif original_format == "tensor":
        return modified_hidden_states
    elif original_format == "object":
        # For object format, modify the corresponding attribute
        if hasattr(original_output, 'hidden_states'):
            original_output.hidden_states = modified_hidden_states
        return original_output
    else:
        # Unknown format, return modified hidden_states
        return modified_hidden_states


class DecoderLayerWithSteerVector(BaseLayerWithSteerVector):
    """
    Generic DecoderLayer intervention controller for full hidden states.
    Uses lazy loading mechanism to create algorithm instances only when needed, saving memory.

    Preferred usage is hook-based: the controller stays outside the model
    tree and `process_output_hook` is registered as a forward hook on the
    original decoder layer, so module names, classes and state-dict keys
    are untouched (safe for FSDP/checkpointing, e.g. VERL). Wrapping a
    layer as a submodule (`base_layer`) is kept for backward compatibility.
    """

    def __init__(self, base_layer=None) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.algorithms: Dict[str, BaseSteerVectorAlgorithm] = {}
        self.layer_id: Optional[int] = None
        # Per-request routing: config slot -> algorithm name for the slots
        # configured on this layer (slots >= CONFIG_SLOT_BASE).
        self.slot_algorithms: Dict[int, str] = {}
        # Key under which this controller is reachable from the
        # vllm::steer_apply custom op (set when the hook is registered).
        self._op_key: Optional[str] = None
        # Tier-1 full-graph mode: persistent buffers read by the captured
        # kernel `hidden += mask * vectors[row_tok]` (see init_graph_table).
        self._graph_mode: bool = False
        self.graph_vectors: Optional[torch.Tensor] = None
        self.graph_mask: Optional[torch.Tensor] = None
        self.graph_row_tok: Optional[torch.Tensor] = None

    def init_graph_table(
        self,
        num_rows: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        max_num_tokens: int,
        row_tok: torch.Tensor,
    ) -> None:
        """Allocate Tier-1 persistent buffers (full-graph steering mode).

        Must run before compilation/graph capture so the captured kernel
        sees the final buffer addresses. Row 0 of the vector table stays
        zero (the no-steer row).
        """
        self.graph_vectors = torch.zeros(
            num_rows + 1, hidden_size, dtype=dtype, device=device
        )
        self.graph_mask = torch.zeros(
            max_num_tokens, dtype=dtype, device=device
        )
        self.graph_row_tok = row_tok
        self._graph_mode = True

    def set_graph_row(self, row: int, payload: torch.Tensor) -> None:
        self.graph_vectors[row].copy_(payload.to(self.graph_vectors.dtype))

    def clear_graph_row(self, row: int) -> None:
        if self.graph_vectors is not None:
            self.graph_vectors[row].zero_()

    def _get_or_create_algorithm(self, name: str, **kwargs) -> BaseSteerVectorAlgorithm:
        """Lazy load or get algorithm instance by name."""
        if name not in self.algorithms:
            # Pass kwargs (e.g., normalize) from external calls to the constructor
            self.algorithms[name] = create_algorithm(name, layer_id=self.layer_id, **kwargs)
        return self.algorithms[name]

    def set_layer_id(self, layer_id: int) -> None:
        """Set layer ID for all created algorithms."""
        self.layer_id = layer_id
        for algo in self.algorithms.values():
            algo.layer_id = layer_id

    def set_steer_vector(self, index: int, **kwargs):
        """Configure a per-request routing slot on this layer.

        Distributes the payload and trigger parameters of one steering
        config to the algorithm instance handling it; `index` is a config
        slot (>= CONFIG_SLOT_BASE).
        """
        algorithm_name = kwargs.pop("algorithm_name", "direct")

        # Extract constructor parameters (e.g., normalize)
        init_kwargs = {}
        if "normalize" in kwargs:
            init_kwargs["normalize"] = kwargs.get("normalize")

        algo = self._get_or_create_algorithm(algorithm_name, **init_kwargs)

        # Always update normalize on the algorithm instance, even if it already existed.
        # _get_or_create_algorithm only passes init_kwargs on first creation;
        # subsequent calls with a different normalize value would be silently ignored.
        if "normalize" in kwargs:
            algo.normalize = kwargs["normalize"]

        # Set payload and per-slot trigger parameters
        algo.set_steer_vector(index, **kwargs)
        self.slot_algorithms[index] = algorithm_name
        algo.slot_params(index).configure_from_dict(kwargs)

    def reset_steer_vector(self, index: int):
        """Reset the vector at specified index in all algorithms."""
        for algo in self.algorithms.values():
            algo.reset_steer_vector(index)

    def forward(self, *args, **kwargs):
        """Wrap the forward method of DecoderLayer (legacy wrapper mode)."""
        output = self.base_layer(*args, **kwargs)
        return self.process_output(output)

    def process_output_hook(self, module, args, output):
        """torch forward-hook entry point: intervene on the layer output.

        Kept dynamo-traceable: only tensor-format handling runs inline;
        all steering logic executes inside the opaque vllm::steer_apply
        custom op, which is a piecewise splitting op under compiled
        execution (it runs eagerly between CUDA-graph segments).
        """
        if self._op_key is None:
            return self.process_output(output)

        hidden_states, residual, other_outputs, original_format = \
            _extract_hidden_states_and_residual(output)
        if residual is not None:
            complete_hidden_states = hidden_states + residual
        else:
            complete_hidden_states = hidden_states

        if self._graph_mode:
            # Tier-1 full-graph kernel: pure tensor math over persistent
            # buffers, safe to capture into CUDA graphs / compile. Rows
            # and masks are filled host-side before each step; row 0 is
            # the zero vector, so unsteered/padding tokens are no-ops.
            n = complete_hidden_states.shape[0]
            complete_hidden_states = complete_hidden_states + (
                self.graph_mask[:n].unsqueeze(1)
                * self.graph_vectors[self.graph_row_tok[:n]]
            )
        else:
            torch.ops.vllm.steer_apply(complete_hidden_states, self._op_key)

        if residual is not None:
            zero_residual = torch.zeros_like(residual)
            return _reconstruct_output(complete_hidden_states, zero_residual,
                                       other_outputs, original_format, output)
        return _reconstruct_output(complete_hidden_states, None,
                                   other_outputs, original_format, output)

    def apply_steering(self, complete_hidden_states):
        """Dispatch steering on complete hidden states (hidden + residual).

        Applies each batch-active config to its own requests' tokens via
        the forward context's sample->slot map; no map means no steering
        this step.
        """
        ctx = get_forward_context() if get_forward_context is not None else None
        token_slots = getattr(ctx, "steer_token_slots", None)
        active_slots = getattr(ctx, "steer_active_slots", None)
        if token_slots is None or not active_slots:
            return complete_hidden_states

        modified_complete_hidden_states = complete_hidden_states
        for slot in active_slots:
            algo_name = self.slot_algorithms.get(slot)
            if algo_name is None:
                # This layer is not targeted by this config.
                continue
            algo = self._get_or_create_algorithm(algo_name)
            modified_complete_hidden_states = algo.apply_intervention(
                modified_complete_hidden_states,
                slot=slot,
                token_slots=token_slots,
            )
        return modified_complete_hidden_states

    def process_output(self, output):
        """Apply the active steering algorithm to a decoder layer output
        (legacy wrapper-mode path; the hook path goes through
        vllm::steer_apply)."""
        # Extract hidden_states and residual from decoder layer output
        hidden_states, residual, other_outputs, original_format = _extract_hidden_states_and_residual(output)

        # Construct complete hidden state
        if residual is not None:
            complete_hidden_states = hidden_states + residual
        else:
            complete_hidden_states = hidden_states

        modified_complete_hidden_states = self.apply_steering(
            complete_hidden_states)

        # Reconstruct output format
        if residual is not None:
            zero_residual = torch.zeros_like(residual)
            return _reconstruct_output(modified_complete_hidden_states, zero_residual, other_outputs,
                                       original_format, output)
        else:
            return _reconstruct_output(modified_complete_hidden_states, None, other_outputs, original_format,
                                       output)


