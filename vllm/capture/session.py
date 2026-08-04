# SPDX-License-Identifier: Apache-2.0
"""Hook-based capture of intermediate model state.

One CaptureSession per worker owns named capture *streams*:

- ``hidden_states``: complete post-layer hidden states (hidden + residual)
  from every decoder layer, discovered structurally with the same
  class-name-list fallback as steering (architecture-agnostic).
- ``router_logits``: MoE router logits, captured by a forward hook on
  each MoE block's gate/router submodule (works for any architecture
  whose experts run on vLLM's fused-MoE stack — no per-model class list
  required; the list is only a hint). When router-logits steering is
  active, the captured logits are the post-steering ones.

Hooks never mutate the model tree (no wrappers, no module renames) and
are inert until a stream is enabled. Each stream is configured at enable
time with a layer subset, storage dtype, a row-selection clause
(`select`, the shared SelectSpec where-clause language also used by
steering), a per-sample reduction (`reduce`: "all" | "last" | "mean")
and a row budget, bounding capture memory — selection and reductions
turn an O(tokens) capture into O(matches)/O(samples), which covers the
common probing/diffmean workflows.

Storage (vllm.capture.store) appends one CPU chunk per forward step and
concatenates at fetch time; there is no batch-boundary heuristic. Every
stored row is labelled with (request id, absolute position, token id)
so clients never re-derive sample alignment. Row selection and
labelling live in vllm.capture.selection.

Relation to upstream: vLLM's ``extract_hidden_states`` speculative
method (with a hidden-states KV connector) also extracts per-layer
hidden states, computed identically (hidden + residual, see
``SupportsEagle3._maybe_add_hidden_state``) — so values from the two
mechanisms are directly comparable. Prefer that path for bulk offline
extraction over a fixed layer set; this session is for interactive use:
runtime enable/disable, arbitrary layer subsets with no graph change,
reductions/budgets, and router logits (which upstream does not capture).
"""

from typing import Any

import torch
from torch import nn

from vllm.capture.selection import prepare_rows
from vllm.capture.store import StreamConfig, StreamStore
from vllm.logger import init_logger

logger = init_logger(__name__)

HIDDEN_STATES = "hidden_states"
ROUTER_LOGITS = "router_logits"


class CaptureSession:
    """Owns capture hooks and streams for one worker's model."""

    def __init__(self):
        self._streams: dict[str, StreamStore | None] = {
            HIDDEN_STATES: None,
            ROUTER_LOGITS: None,
        }
        # Engine-internal request id -> {stream: SelectSpec wire dict}.
        self._request_selects: dict[str, dict[str, dict]] = {}
        self._hook_handles: list = []
        self._hidden_layers = 0
        self._gate_layers = 0
        self._attached = False

    # ------------------------------------------------------------------
    # Per-request selection overrides (runner add/remove lifecycle)
    # ------------------------------------------------------------------

    def add_request(self, req_id: str, capture_select: dict[str, dict]) -> None:
        """Register a request's selection override ({stream: wire}).

        Clause structure is validated at admission (input processor); a
        raise here would take down the engine core, so worker-side
        conflicts warn and drop the override instead.
        """
        accepted = {}
        for stream, wire in capture_select.items():
            store = self._streams.get(stream)
            if stream not in self._streams or store is None:
                logger.warning_once(
                    "Request carries a capture select for stream %r, "
                    "which is not enabled; the override is ignored.",
                    stream,
                )
                continue
            if store.config.reduce != "all":
                logger.warning_once(
                    "Per-request capture selects cannot combine with "
                    "the %r reduction; the override is ignored.",
                    store.config.reduce,
                )
                continue
            accepted[stream] = wire
        if accepted:
            self._request_selects[req_id] = accepted

    def remove_request(self, req_id: str) -> None:
        self._request_selects.pop(req_id, None)

    # ------------------------------------------------------------------
    # Hook attachment (once, at model load; inert until a stream enables)
    # ------------------------------------------------------------------

    def attach(self, model: nn.Module) -> None:
        if self._attached:
            return
        self._hidden_layers = self._attach_hidden_hooks(model)
        self._gate_layers = self._attach_gate_hooks(model)

        def flush_hook(mod, args, output):
            for store in self._streams.values():
                if store is not None:
                    store.flush()

        # One post-forward flush per step: per-layer hooks only stage
        # GPU rows; the D2H copies coalesce here (pinned, non-blocking,
        # single sync).
        self._hook_handles.append(model.register_forward_hook(flush_hook))
        self._attached = True

    def _attach_stream_hooks(
        self,
        stream: str,
        matches: dict[str, nn.Module],
        resolve_target,
        extract_rows,
        log_fmt: str,
    ) -> int:
        """Hook one stream's modules; returns the number of hooks placed.

        `resolve_target(name, module)` picks the submodule to hook (None
        skips the module, after warning); `extract_rows(output)` pulls
        the [tokens, dim] tensor out of the hook output (None skips the
        step). Modules without a numeric layer id in their name get
        sequential fallback ids.
        """
        from vllm.steer_vectors.discovery import extract_layer_id_from_module_name

        count = 0
        fallback_id = 0
        for name, module in matches.items():
            target = resolve_target(name, module)
            if target is None:
                continue
            layer_id = extract_layer_id_from_module_name(name)
            if layer_id is None:
                layer_id = fallback_id
            fallback_id = layer_id + 1

            def hook(mod, args, output, _lid=layer_id, _name=name):
                store = self._streams[stream]
                if store is None or not store.wants_layer(_lid):
                    return
                tensor = extract_rows(output)
                if tensor is None:
                    return
                rows, meta = prepare_rows(
                    tensor, store, stream, self._request_selects
                )
                if rows is not None:
                    store.append(_lid, rows, meta, _name)

            self._hook_handles.append(target.register_forward_hook(hook))
            count += 1
        if count:
            logger.info(log_fmt, count)
        return count

    def _attach_hidden_hooks(self, model: nn.Module) -> int:
        from vllm.steer_vectors.discovery import (
            SUPPORTED_DECODER_LAYERS,
            find_decoder_layers,
            find_layers_with_fallback,
        )
        from vllm.steer_vectors.geometry import split_decoder_output

        def extract_rows(output):
            hidden, residual, _, _ = split_decoder_output(output)
            if not isinstance(hidden, torch.Tensor):
                return None
            return hidden + residual if residual is not None else hidden

        return self._attach_stream_hooks(
            HIDDEN_STATES,
            find_layers_with_fallback(
                model, find_decoder_layers, SUPPORTED_DECODER_LAYERS, "decoder_layer"
            ),
            lambda name, module: module,
            extract_rows,
            "[Capture] hooked %d decoder layers for hidden states",
        )

    def _attach_gate_hooks(self, model: nn.Module) -> int:
        from vllm.steer_vectors.discovery import (
            find_moe_blocks,
            find_moe_gate,
            moe_gate_is_fused,
        )
        from vllm.steer_vectors.geometry import extract_gate_logits

        def resolve_target(name, block):
            gate = find_moe_gate(block)
            if gate is None:
                logger.warning(
                    "[Capture] MoE block %s has no gate/router submodule; "
                    "its router logits cannot be captured.",
                    name,
                )
                return None
            if moe_gate_is_fused(block):
                logger.warning(
                    "[Capture] MoE block %s fuses gate weights into the "
                    "MoE runner (gate forward bypassed); its router "
                    "logits cannot be captured.",
                    name,
                )
                return None
            return gate

        # NOTE: gate hooks are registered after any steering gate hook
        # (steering attaches first at load), so captured logits are
        # post-steering.
        return self._attach_stream_hooks(
            ROUTER_LOGITS,
            find_moe_blocks(model),
            resolve_target,
            extract_gate_logits,
            "[Capture] hooked %d MoE gates for router logits",
        )

    def detach(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._attached = False

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def any_enabled(self) -> bool:
        return any(store is not None for store in self._streams.values())

    def enable_stream(self, stream: str, **config_kwargs) -> None:
        if stream not in self._streams:
            raise ValueError(f"Unknown capture stream: {stream}")
        if not self._attached:
            raise RuntimeError(
                "Capture hooks are not attached. Capture requires "
                "enforce_eager=True (hooks cannot run inside compiled "
                "graphs)."
            )
        if stream == ROUTER_LOGITS and self._gate_layers == 0:
            logger.warning(
                "Enabling router_logits capture but no MoE gates were "
                "found in this model."
            )
        self._streams[stream] = StreamStore(StreamConfig(**config_kwargs))

    def disable_stream(self, stream: str) -> None:
        if stream in self._streams:
            self._streams[stream] = None

    def fetch_stream(
        self,
        stream: str,
        clear: bool = True,
        layers: list[int] | None = None,
        req_ids: list[str] | None = None,
    ) -> dict[int, dict[str, Any]]:
        store = self._streams.get(stream)
        if store is None:
            return {}
        store.flush()  # capture any rows staged since the last step
        if req_ids is not None:
            return store.serialize(
                layers=layers, req_ids=req_ids, clear_selected=clear
            )
        result = store.serialize(layers=layers)
        if clear:
            if layers is None:
                # Keep capturing into a fresh store with the same config.
                self._streams[stream] = StreamStore(store.config)
            else:
                store.drop_layers(list(result))
        return result

    def clear_stream(self, stream: str) -> None:
        store = self._streams.get(stream)
        if store is not None:
            self._streams[stream] = StreamStore(store.config)

    def stream_status(self, stream: str) -> dict[str, Any]:
        store = self._streams.get(stream)
        hooked = self._hidden_layers if stream == HIDDEN_STATES else self._gate_layers
        if store is None:
            return {"enabled": False, "hooked_layers": hooked}
        return {
            "enabled": True,
            "hooked_layers": hooked,
            "layers_captured": len(store.chunks),
            "tokens_stored": store.tokens_stored,
            "tokens_dropped": store.tokens_dropped,
            "reduce": store.config.reduce,
            "select": store.config.select,
            "meta_complete": store.meta_complete,
        }
