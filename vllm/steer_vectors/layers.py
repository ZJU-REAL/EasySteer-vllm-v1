# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
from torch import nn

from .algorithms import BaseSteerVectorAlgorithm, create_algorithm
from vllm.steer_vectors import trace

# Import forward context to get current token information
try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None

# Sentinel for lazily-fetched forward-context info (None is a valid
# "unavailable" fetch result).
_CTX_UNSET = object()


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


def _resolve_conflicts(collected, mode: str):
    """Resolve position conflicts between a slot's interventions.

    `collected` is [(idx, algo, positions)] in priority order. 'priority'
    gives earlier interventions exclusive claim to their positions,
    'sequential' applies all in order, 'error' raises on overlap.
    """
    if mode == "sequential" or len(collected) <= 1:
        return collected
    if mode == "error":
        for i in range(len(collected)):
            for j in range(i + 1, len(collected)):
                overlap = torch.isin(collected[i][2], collected[j][2])
                if overlap.any():
                    raise ValueError(
                        f"Steering vectors conflict at positions "
                        f"{collected[i][2][overlap].tolist()} between "
                        f"vectors {collected[i][0]} and {collected[j][0]}. "
                        f"Set conflict_resolution='priority' or "
                        f"'sequential'."
                    )
        return collected
    if mode != "priority":
        raise ValueError(f"Unknown conflict resolution strategy: {mode}")
    claimed = None
    filtered = []
    for idx, algo, positions in collected:
        if claimed is not None and claimed.numel() > 0:
            positions = positions[~torch.isin(positions, claimed)]
            if positions.numel() == 0:
                continue
        claimed = (positions if claimed is None
                   else torch.cat([claimed, positions]))
        filtered.append((idx, algo, positions))
    return filtered


class DecoderLayerWithSteerVector(BaseLayerWithSteerVector):
    """
    Generic DecoderLayer intervention controller for full hidden states.

    Every routing slot holds an ordered list of interventions (a
    single-vector config is a list of one), each backed by a private
    algorithm instance with its own payload and trigger controller;
    conflict resolution between a slot's interventions is a slot
    property.

    Preferred usage is hook-based: the controller stays outside the model
    tree and `process_output_hook` is registered as a forward hook on the
    original decoder layer, so module names, classes and state-dict keys
    are untouched (safe for FSDP/checkpointing, e.g. VERL). Wrapping a
    layer as a submodule (`base_layer`) is kept for backward compatibility.
    """

    def __init__(self, base_layer=None) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.layer_id: Optional[int] = None
        # Per-request routing: config slot -> ordered intervention list.
        self.slot_interventions: Dict[int, list] = {}
        self.slot_conflict: Dict[int, str] = {}
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

    def set_layer_id(self, layer_id: int) -> None:
        """Set layer ID for existing and future interventions."""
        self.layer_id = layer_id
        for entries in self.slot_interventions.values():
            for algo in entries:
                algo.layer_id = layer_id

    def configure_slot(
        self,
        slot: int,
        vector_specs: list,
        conflict_resolution: str = "priority",
    ) -> None:
        """Configure a routing slot from an ordered list of vector specs.

        Each spec carries `algorithm`, `payload`, and the canonical
        steering fields (scale, triggers, normalize, debug) of one
        intervention.
        """
        entries = []
        for spec in vector_specs:
            init_kwargs = {}
            if spec.get("normalize") is not None:
                init_kwargs["normalize"] = spec["normalize"]
            algo = create_algorithm(
                spec.get("algorithm") or "direct",
                layer_id=self.layer_id,
                **init_kwargs,
            )
            scale = spec.get("scale")
            algo.set_steer_vector(
                0,
                payload=spec["payload"],
                scale_factor=1.0 if scale is None else scale,
            )
            algo.set_active_tensor(0)
            algo.params.configure_from_dict(spec)
            entries.append(algo)
        self.slot_interventions[slot] = entries
        self.slot_conflict[slot] = conflict_resolution

    def reset_steer_vector(self, index: int):
        """Drop the intervention list of a routing slot."""
        self.slot_interventions.pop(index, None)
        self.slot_conflict.pop(index, None)

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

        Applies each batch-active slot's interventions to its own
        requests' tokens via the forward context's sample->slot map; no
        map means no steering this step.
        """
        ctx = get_forward_context() if get_forward_context is not None else None
        token_slots = getattr(ctx, "steer_token_slots", None)
        active_slots = getattr(ctx, "steer_active_slots", None)
        if token_slots is None or not active_slots:
            return complete_hidden_states

        hidden = complete_hidden_states
        ctx_info = _CTX_UNSET
        for slot in active_slots:
            entries = self.slot_interventions.get(slot)
            if not entries:
                # This layer is not targeted by this config.
                continue
            collected = []
            for idx, algo in enumerate(entries):
                params = algo._get_params()
                if not algo._is_valid(params):
                    continue
                if not algo.params.has_any_triggers():
                    continue
                if algo.params.is_global_only_config():
                    # All tokens of this slot's requests.
                    positions = (token_slots == slot).nonzero(
                        as_tuple=False).squeeze(-1)
                else:
                    if ctx_info is _CTX_UNSET:
                        ctx_info = algo._get_forward_context_and_samples(
                            hidden)
                    if ctx_info is None:
                        continue
                    _, samples_info, current_tokens = ctx_info
                    positions = algo.params.collect_intervention_positions(
                        hidden_states=hidden,
                        current_tokens=current_tokens,
                        samples_info=samples_info,
                    )
                    if positions is None or positions.numel() == 0:
                        continue
                    positions = positions[token_slots[positions] == slot]
                if positions.numel() == 0:
                    continue
                collected.append((idx, algo, positions))

            collected = _resolve_conflicts(
                collected, self.slot_conflict.get(slot, "priority"))
            for idx, algo, positions in collected:
                hidden = algo._batch_transform_tensor(
                    hidden, positions, algo._get_params())
                if trace.enabled():
                    label = (algo.__class__.__name__ if len(entries) == 1
                             else f"multi:{idx}:{algo.__class__.__name__}")
                    trace.record_apply(
                        self.layer_id, slot, label, positions.tolist())
        return hidden

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


