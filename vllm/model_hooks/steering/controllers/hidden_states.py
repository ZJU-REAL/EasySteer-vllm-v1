# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interventions on complete decoder hidden states."""

import torch

from vllm.model_hooks.components.registry import COMPONENTS, HIDDEN_STATES
from vllm.model_hooks.steering.algorithms import get_algorithm
from vllm.model_hooks.steering.graph.kernels import (
    GRAPH_FAMILIES,
    apply_decoder_families,
)

from .base import SteeringController


class HiddenStatesController(SteeringController):
    """DecoderLayer intervention controller for full hidden states.

    Hook-based: the controller stays outside the model tree and
    `process_output_hook` is registered as a forward hook on the
    original decoder layer, so module names, classes and state-dict
    keys are untouched (safe for FSDP/checkpointing, e.g. VERL).
    """

    component_id = HIDDEN_STATES

    def __init__(self) -> None:
        super().__init__()
        # Tier-1 full-graph mode: persistent buffers read by the captured
        # kernel families (see GRAPH_FAMILIES, init_graph_buffers and
        # process_output_hook).
        self._graph_mode: bool = False
        self.graph_tables: dict[str, dict[str, torch.Tensor]] | None = None
        self.graph_mask: torch.Tensor | None = None
        self.replace_mask: torch.Tensor | None = None
        self.graph_token_rows: torch.Tensor | None = None
        # Per-row normalize flag: steered rows of flagged configs are
        # rescaled back to the original complete-state norm.
        self.normalize_flag: torch.Tensor | None = None

    def graph_mask_names(self, families: frozenset[str]) -> tuple[str, ...]:
        """Masks needed by this component's declared kernel families."""
        families = families.intersection(GRAPH_FAMILIES)
        return (
            *(("transform",) if families - {"replace"} else ()),
            *(("replace",) if "replace" in families else ()),
        )

    def init_graph_buffers(
        self,
        families,
        masks,
        *,
        capacity,
        hidden_size,
        max_rank,
        dtype,
        device,
        token_rows,
    ) -> None:
        """Allocate Tier-1 persistent buffers (full-graph steering mode).

        Must run before compilation/graph capture so the captured kernel
        sees the final buffer addresses. Only `families` (the declared
        workload's kernel families) are allocated and compiled into the
        kernel — a family outside the declaration can
        never receive a payload, so it contributes no compute. Row 0 of
        every allocated table stays zero (the no-steer row): idle rows
        contribute an exact zero delta.
        """
        dim_of = {"h": hidden_size, "r": max_rank}
        families = families.intersection(GRAPH_FAMILIES)
        rows = capacity + 1
        self.graph_tables = {
            family: {
                key: torch.zeros(
                    rows, *(dim_of[d] for d in dims), dtype=dtype, device=device
                )
                for key, dims in schema.items()
            }
            for family, schema in GRAPH_FAMILIES.items()
            if family in families
        }
        self.graph_mask = masks.get("transform")
        self.replace_mask = masks.get("replace")
        if families:
            self.normalize_flag = torch.zeros(rows, dtype=dtype, device=device)
        self.graph_token_rows = token_rows
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
        with its eager math). Hidden dimensions match the model exactly;
        only rank dimensions may be zero-padded into the table slot.
        """
        algo_cls = get_algorithm(algorithm)
        assert self.normalize_flag is not None
        assert self.graph_tables is not None
        self.normalize_flag[row] = 1.0 if normalize else 0.0
        family = algo_cls.graph_family
        if family is None:
            raise ValueError(f"algorithm {algorithm!r} has no full-graph kernel family")
        tables = self.graph_tables.get(family)
        if tables is None:
            # Admission rejects undeclared algorithms; reaching here
            # means the declaration and the compiled kernel families
            # have drifted apart.
            raise RuntimeError(
                f"algorithm {algorithm!r} needs kernel family {family!r}, "
                f"which was not compiled into this engine (families: "
                f"{sorted(self.graph_tables)}); declare the algorithm via "
                f"steer_algorithms at launch"
            )
        for key, value in algo_cls.graph_lower(payload, scale).items():
            if value is None:
                continue
            dest = tables[key][row]
            axes = GRAPH_FAMILIES[family][key]
            if value.dim() != dest.dim() or any(
                v > d if axis == "r" else v != d
                for v, d, axis in zip(value.shape, dest.shape, axes)
            ):
                raise ValueError(
                    f"{algorithm} payload {key} shape {tuple(value.shape)} "
                    f"does not fit the full-graph buffer {tuple(dest.shape)}; "
                    "hidden dimensions must match exactly. For larger ranks, "
                    "raise steer_graph_max_rank or use steer_graph_mode='split'"
                )
            dest[tuple(slice(0, s) for s in value.shape)].copy_(value.to(dest.dtype))

    def graph_row_tensors(self) -> tuple[torch.Tensor, ...]:
        """Persistent slot tables, including flags, for clear and rollback."""
        if not self.graph_tables:
            return ()
        assert self.normalize_flag is not None
        return (
            self.normalize_flag,
            *(
                table
                for family in self.graph_tables.values()
                for table in family.values()
            ),
        )

    def clear_graph_row(self, row: int) -> None:
        for table in self.graph_row_tensors():
            table[row].zero_()

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
        adapter = COMPONENTS[HIDDEN_STATES].adapter
        hidden_states, residual, other_outputs, original_format = adapter.read_output(
            output
        )

        if self._graph_mode:
            assert self.graph_tables is not None
            steered = apply_decoder_families(
                self.graph_tables,
                self.graph_mask,
                self.replace_mask,
                self.normalize_flag,
                self.graph_token_rows,
                hidden_states,
                residual,
            )
            return adapter.write_output(
                steered, residual, other_outputs, original_format, output
            )

        torch.ops.vllm.steer_apply(hidden_states, residual, self._op_key)
        return output
