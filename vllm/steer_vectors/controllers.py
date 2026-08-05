# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from vllm.steer_vectors import trace
from vllm.steer_vectors.geometry import (
    extract_gate_logits,
    reconstruct_decoder_output,
    split_decoder_output,
)

from .algorithms import create_algorithm, get_algorithm
from .graph_kernels import (
    GRAPH_FAMILIES,
    apply_decoder_families,
    apply_gate_toggles,
)

try:
    from vllm.forward_context import get_forward_context
except ImportError:
    get_forward_context = None


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
        claimed = positions if claimed is None else torch.cat([claimed, positions])
        filtered.append((idx, algo, positions))
    return filtered


class SlotRoutedSteerController(nn.Module):
    """Shared slot-routing engine for steering controllers.

    Every routing slot holds an ordered list of interventions (a
    single-vector config is a list of one), each backed by a private
    algorithm instance with its own payload and trigger controller;
    conflict resolution between a slot's interventions is a slot
    property.

    `apply_steering` routes each batch-active slot's interventions to
    its own requests' token rows via the forward context's token->slot
    map. It operates on any per-token row tensor — decoder hidden
    states and MoE router logits share the same first-dimension token
    layout — so subclasses only choose the hook point and the tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer_id: int | None = None
        # Per-request routing: config slot -> ordered intervention list.
        self.slot_interventions: dict[int, list] = {}
        self.slot_conflict: dict[int, str] = {}
        # Key under which this controller is reachable from its custom
        # op (set when the hook is registered).
        self._op_key: str | None = None

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
            algo.set_payload(spec["payload"], 1.0 if scale is None else scale)
            algo.clause.configure_from_dict(spec)
            entries.append(algo)
        self.slot_interventions[slot] = entries
        self.slot_conflict[slot] = conflict_resolution

    def reset_steer_vector(self, index: int):
        """Drop the intervention list of a routing slot."""
        self.slot_interventions.pop(index, None)
        self.slot_conflict.pop(index, None)

    def apply_steering(self, token_tensor, residual=None):
        """Dispatch slot-routed steering on a per-token row tensor.

        Applies each batch-active slot's interventions to its own
        requests' token rows; positions are pre-resolved once per step
        by the runner (forward context's steer_slot_positions, keyed by
        clause) — no per-layer trigger work or device syncs. With
        `residual`, transforms see the complete hidden state of the
        selected rows but write back only `token_tensor` (delta form) —
        the residual stream is never collapsed or zeroed.
        """
        ctx = get_forward_context() if get_forward_context is not None else None
        token_slots = getattr(ctx, "steer_token_slots", None)
        active_slots = getattr(ctx, "steer_active_slots", None)
        if token_slots is None or not active_slots:
            return token_tensor
        slot_positions = ctx.steer_slot_positions

        tensor = token_tensor
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
                key = algo.clause.cache_key
                if key is None:
                    continue
                if slot_positions is None or (slot, key) not in slot_positions:
                    raise RuntimeError(
                        f"steering positions for slot {slot} were not "
                        "resolved this step — the runner's clause registry "
                        "is out of sync with the layer controllers"
                    )
                positions = slot_positions[(slot, key)]
                if positions is None:
                    continue
                collected.append((idx, algo, positions))

            collected = _resolve_conflicts(
                collected, self.slot_conflict.get(slot, "priority")
            )
            for idx, algo, positions in collected:
                tensor = algo._batch_transform_tensor(
                    tensor, positions, algo._get_params(), residual=residual
                )
                if trace.enabled():
                    label = (
                        algo.__class__.__name__
                        if len(entries) == 1
                        else f"multi:{idx}:{algo.__class__.__name__}"
                    )
                    trace.record_apply(self.layer_id, slot, label, positions.tolist())
        return tensor


class DecoderSteerController(SlotRoutedSteerController):
    """DecoderLayer intervention controller for full hidden states.

    Hook-based: the controller stays outside the model tree and
    `process_output_hook` is registered as a forward hook on the
    original decoder layer, so module names, classes and state-dict
    keys are untouched (safe for FSDP/checkpointing, e.g. VERL).
    """

    def __init__(self) -> None:
        super().__init__()
        # Tier-1 full-graph mode: persistent buffers read by the captured
        # kernel families (see GRAPH_FAMILIES, init_graph_table and
        # process_output_hook).
        self._graph_mode: bool = False
        self.graph_tables: dict[str, dict[str, torch.Tensor]] | None = None
        self.graph_mask: torch.Tensor | None = None
        self.replace_mask: torch.Tensor | None = None
        self.graph_token_rows: torch.Tensor | None = None
        # Per-row normalize flag: steered rows of flagged configs are
        # rescaled back to the original complete-state norm.
        self.normalize_flag: torch.Tensor | None = None

    def init_graph_table(
        self,
        num_rows: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        max_num_tokens: int,
        row_tok: torch.Tensor,
        max_rank: int,
    ) -> None:
        """Allocate Tier-1 persistent buffers (full-graph steering mode).

        Must run before compilation/graph capture so the captured kernel
        sees the final buffer addresses. Row 0 of every table stays zero
        (the no-steer row): unused families contribute an exact zero
        delta, so they cost only cached zero-row reads when idle.
        """
        dim_of = {"h": hidden_size, "r": max_rank}
        rows = num_rows + 1
        self.graph_tables = {
            family: {
                key: torch.zeros(
                    rows, *(dim_of[d] for d in dims), dtype=dtype, device=device
                )
                for key, dims in schema.items()
            }
            for family, schema in GRAPH_FAMILIES.items()
        }
        self.graph_mask = torch.zeros(max_num_tokens, dtype=dtype, device=device)
        self.replace_mask = torch.zeros(max_num_tokens, dtype=dtype, device=device)
        self.normalize_flag = torch.zeros(rows, dtype=dtype, device=device)
        self.graph_token_rows = row_tok
        self._graph_mode = True

    def set_graph_row(
        self,
        row: int,
        algorithm: str,
        payload,
        scale: float,
        normalize: bool = False,
    ) -> None:
        """Write one config's lowered payload into its family's tables.

        The algorithm class owns the lowering (graph_lower, colocated
        with its eager math); this writer is generic: smaller tensors are
        zero-padded into the table slot (exact no-op padding), larger
        ones rejected.
        """
        algo_cls = get_algorithm(algorithm)
        self.normalize_flag[row] = 1.0 if normalize else 0.0
        family = algo_cls.graph_family
        if family is None:
            raise ValueError(
                f"algorithm {algorithm!r} has no full-graph kernel family"
            )
        tables = self.graph_tables[family]
        for key, value in algo_cls.graph_lower(payload, scale).items():
            if value is None:
                continue
            dest = tables[key][row]
            if value.dim() != dest.dim() or any(
                v > d for v, d in zip(value.shape, dest.shape)
            ):
                raise ValueError(
                    f"{algorithm} payload {key} shape {tuple(value.shape)} "
                    f"exceeds the full-graph buffer {tuple(dest.shape)}; "
                    f"raise steer_graph_max_rank or launch with "
                    f"steer_graph_mode='split'"
                )
            dest[tuple(slice(0, s) for s in value.shape)].copy_(
                value.to(dest.dtype)
            )

    def clear_graph_row(self, row: int) -> None:
        if self.graph_tables is None:
            return
        self.normalize_flag[row] = 0.0
        for tables in self.graph_tables.values():
            for table in tables.values():
                table[row].zero_()

    def zero_step_masks(self) -> None:
        self.graph_mask.zero_()
        self.replace_mask.zero_()

    def process_output_hook(self, module, args, output):
        """torch forward-hook entry point: intervene on the layer output.

        Kept dynamo-traceable: only tensor-format handling runs inline;
        all steering logic executes inside the opaque vllm::steer_apply
        custom op, which is a piecewise splitting op under compiled
        execution (it runs eagerly between CUDA-graph segments).

        Steering is row-local on `hidden_states`: adding a delta to
        `hidden` is equivalent to adding it to `hidden + residual`, so
        the residual stream flows on untouched and the next layer's
        fused add-RMSNorm is preserved. Unsteered steps cost only the
        op dispatch.
        """
        if self._op_key is None:
            return output

        hidden_states, residual, other_outputs, original_format = split_decoder_output(
            output
        )

        if self._graph_mode:
            steered = apply_decoder_families(
                self.graph_tables,
                self.graph_mask,
                self.replace_mask,
                self.normalize_flag,
                self.graph_token_rows,
                hidden_states,
                residual,
            )
            return reconstruct_decoder_output(
                steered, residual, other_outputs, original_format, output
            )

        torch.ops.vllm.steer_apply(hidden_states, residual, self._op_key)
        return output


class MoEGateSteerController(SlotRoutedSteerController):
    """Hook-based MoE router-logits steering controller.

    Registered as a forward hook on the MoE block's gate/router
    submodule: the logits are steered in place (via the opaque
    vllm::steer_moe_gate op, a splitting op under compiled execution)
    before top-k expert selection consumes them — architecture-agnostic,
    no per-model forward reimplementation.

    Slot-routed exactly like decoder-layer steering: router-logit rows
    share the token layout of hidden states, so each request's
    moe_router config applies only to its own tokens and distinct MoE
    configs batch together.
    """

    def __init__(self) -> None:
        super().__init__()
        # Tier-1 full-graph mode: expert toggle tables read by the
        # captured gate kernel (see init_graph_table / the hook below).
        self._graph_mode: bool = False
        self.hook_target = None  # set at attach (models.py)
        self.moe_act: torch.Tensor | None = None
        self.moe_deact: torch.Tensor | None = None
        self.moe_eps: torch.Tensor | None = None
        self.graph_mask: torch.Tensor | None = None
        self.graph_token_rows: torch.Tensor | None = None

    def init_graph_table(
        self,
        num_rows: int,
        dtype: torch.dtype,
        device: torch.device,
        max_num_tokens: int,
        row_tok: torch.Tensor,
    ) -> None:
        """Allocate the gate kernel's expert toggle tables.

        The expert count comes from the hooked gate module's weight;
        row 0 stays zero so unrouted tokens are exact no-ops.
        """
        weight = getattr(self.hook_target, "weight", None)
        if weight is None:
            raise RuntimeError(
                "moe gate module exposes no weight; cannot size the "
                "full-graph expert toggle tables"
            )
        num_experts = weight.shape[0]
        rows = num_rows + 1
        self.moe_act = torch.zeros(
            rows, num_experts, dtype=dtype, device=device
        )
        self.moe_deact = torch.zeros(
            rows, num_experts, dtype=dtype, device=device
        )
        self.moe_eps = torch.zeros(rows, dtype=dtype, device=device)
        self.graph_mask = torch.zeros(max_num_tokens, dtype=dtype, device=device)
        self.graph_token_rows = row_tok
        self._graph_mode = True

    def set_graph_toggles(self, row: int, toggles: dict) -> None:
        """Write one config's lowered expert toggles into the tables.

        `toggles` comes from MoERouterAlgorithm.graph_lower (the
        algorithm owns mode routing and overlap semantics); this writer
        owns the width-aware part: out-of-range ids warn and drop.
        """
        from vllm.logger import init_logger

        num_experts = self.moe_act.shape[1]
        activate = toggles["activate"]
        deactivate = toggles["deactivate"]
        invalid = [
            e for e in activate + deactivate if not 0 <= e < num_experts
        ]
        if invalid:
            init_logger(__name__).warning_once(
                "moe_router: expert ids %s are outside [0, %d) for this "
                "model and are ignored.",
                sorted(set(invalid)),
                num_experts,
            )
        self.moe_act[row].zero_()
        self.moe_deact[row].zero_()
        for e in activate:
            if 0 <= e < num_experts:
                self.moe_act[row, e] = 1.0
        for e in deactivate:
            if 0 <= e < num_experts:
                self.moe_deact[row, e] = 1.0
        self.moe_eps[row] = toggles["epsilon"]

    def clear_graph_row(self, row: int) -> None:
        if self.moe_act is None:
            return
        self.moe_act[row].zero_()
        self.moe_deact[row].zero_()
        self.moe_eps[row] = 0.0

    def zero_step_masks(self) -> None:
        self.graph_mask.zero_()

    def process_gate_output_hook(self, module, args, output):
        """Forward-hook entry point on the gate module.

        The logits tensor is mutated in place, so the original output
        structure flows on unchanged (hook returns None).
        """
        if self._op_key is None:
            return None
        logits = extract_gate_logits(output)
        if logits is None:
            return None
        if self._graph_mode:
            apply_gate_toggles(
                self.moe_act,
                self.moe_deact,
                self.moe_eps,
                self.graph_mask,
                self.graph_token_rows,
                logits,
            )
            return None
        torch.ops.vllm.steer_moe_gate(logits, self._op_key)
        return None

    def apply_gate_steering(self, logits):
        """Op implementation: slot-routed router-logit steering."""
        return self.apply_steering(logits)
