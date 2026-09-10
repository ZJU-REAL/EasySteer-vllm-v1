# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared slot routing for component steering controllers."""

from torch import nn

from vllm.forward_context import get_forward_context
from vllm.model_hooks.selection.runtime import clause_cache_key
from vllm.model_hooks.steering import trace
from vllm.model_hooks.steering.algorithms import create_algorithm


class SteeringController(nn.Module):
    """Shared slot-routing engine for steering controllers.

    Every routing slot holds an ordered list of interventions (a
    single-vector config is a list of one), each backed by a private
    algorithm instance with its own payload;
    conflict resolution between a slot's interventions is a slot
    property.

    `apply_steering` routes each batch-active slot's interventions to
    its own requests' token rows via the forward context's resolved
    clause positions. It operates on any per-token row tensor — decoder hidden
    states and MoE router logits share the same first-dimension token
    layout — so subclasses only choose the hook point and the tensor.
    """

    component_id: str

    def __init__(self) -> None:
        super().__init__()
        self.layer_id: int | None = None
        # Per-request routing: config slot -> ordered intervention list.
        self.slot_interventions: dict[int, list] = {}
        self.slot_position_groups: dict[int, tuple] = {}
        # Key under which this controller is reachable from its custom
        # op (set when the hook is registered).
        self._op_key: str | None = None
        self._output_width: int | None = None

    @property
    def output_width(self) -> int | None:
        return self._output_width

    def configure_slot(
        self,
        slot: int,
        vector_specs: list,
        conflict_resolution: str = "priority",
    ) -> None:
        """Configure a routing slot from an ordered list of vector specs.

        Each spec carries `algorithm`, `payload`, and the canonical
        steering fields (scale, triggers, normalize) of one
        intervention.
        """
        entries = []
        for spec in vector_specs:
            algo = create_algorithm(
                spec["algorithm"], normalize=bool(spec.get("normalize", False))
            )
            scale = spec.get("scale")
            algo.set_payload(spec["payload"], 1.0 if scale is None else scale)
            entries.append(algo)
        self.slot_interventions[slot] = entries
        self.slot_position_groups[slot] = (
            conflict_resolution,
            tuple(clause_cache_key(spec.get("apply_spec")) for spec in vector_specs),
        )

    def clear_slot(self, slot: int) -> None:
        """Drop the intervention list of a routing slot."""
        self.slot_interventions.pop(slot, None)
        self.slot_position_groups.pop(slot, None)

    def apply_steering(self, token_tensor, residual=None):
        """Dispatch slot-routed steering on a per-token row tensor.

        Applies each batch-active slot's interventions to its own
        requests' token rows; positions are pre-resolved once per step
        by the runner (forward context's steer_slot_positions, keyed by
        intervention group and index) — no per-layer trigger work or syncs. With
        `residual`, transforms see the complete hidden state of the
        selected rows but write back only `token_tensor` (delta form) —
        the residual stream is never collapsed or zeroed.
        """
        ctx = get_forward_context()
        active_slots = ctx.steer_active_slots
        if not active_slots:
            return token_tensor
        slot_positions = ctx.steer_slot_positions

        tensor = token_tensor
        for slot in active_slots:
            entries = self.slot_interventions.get(slot)
            if not entries:
                # This layer is not targeted by this config.
                continue
            group = self.slot_position_groups[slot]
            for idx, algo in enumerate(entries):
                params = algo.payload
                if not algo._is_valid(params):
                    continue
                key = (slot, group, idx)
                if slot_positions is None or key not in slot_positions:
                    raise RuntimeError(
                        f"steering positions for slot {slot} were not "
                        "resolved this step — the runner's clause registry "
                        "is out of sync with the layer controllers"
                    )
                positions = slot_positions[key]
                if positions is None:
                    continue
                tensor = algo._batch_transform_tensor(
                    tensor, positions, algo.payload, residual=residual
                )
                if trace.enabled():
                    label = (
                        algo.__class__.__name__
                        if len(entries) == 1
                        else f"multi:{idx}:{algo.__class__.__name__}"
                    )
                    trace.record_apply(self.layer_id, slot, label, positions.tolist())
        return tensor
