# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hook-based capture of intermediate model state.

One CaptureSession per worker owns named capture *streams*:

- ``hidden_states``: complete post-layer hidden states (hidden + residual)
  from every decoder layer, discovered structurally with steering.
- ``router_logits``: MoE router logits, captured by a forward hook on
  each MoE block's gate/router submodule (works for any architecture
  exposing a separate gate/router module). When router-logits steering is
  active, the captured logits are the post-steering ones.
- ``attention_heads``: concatenated query-head outputs of ordinary decoder
  attention, before the output projection and after any head-output steering.

Hooks never mutate the model tree (no wrappers, no module renames) and
are inert until a stream is enabled. On the V2 runner, hook bodies trace
to nothing under torch.compile (ordinary compiled artifacts and CUDA
graphs carry no capture code). Selected rows use a separate
FULL graph with fixed capture buffers when supported, or raw eager
forward otherwise; proven-empty batches keep normal graph dispatch.
Prefix caching needs no engine-level restriction either — requests can
skip cache reads when their selected rows require full recomputation.
Cache hits are checked against each request's effective selection at
admission or its first scheduled batch after capture starts; fetching raises
only if selected prompt rows were skipped.
Each stream is configured at enable
time with a layer subset, storage dtype, a row-selection clause
(`select`, the shared SelectSpec where-clause language also used by
steering), a per-sample reduction (`reduce`: "all" | "last" | "mean")
and a row budget, bounding capture memory — selection and reductions
turn an O(tokens) capture into O(matches)/O(samples), which covers the
common probing/diffmean workflows.

Storage (vllm.model_hooks.capture.store) indexes CPU chunks by request and
concatenates only fetched rows; there is no batch-boundary heuristic. Every
stored row is labelled with (request id, absolute position, token id)
so clients never re-derive sample alignment. Row selection and
labelling live in vllm.model_hooks.capture.selection.

Relation to upstream: vLLM's ``extract_hidden_states`` speculative
method (with a hidden-states KV connector) also extracts per-layer
hidden states, computed identically (hidden + residual, see
``SupportsEagle3._maybe_add_hidden_state``) — so values from the two
mechanisms are directly comparable. Prefer that path for bulk offline
extraction over a fixed layer set; this session is for interactive use:
runtime enable/disable, arbitrary layer subsets with no model reload,
reductions/budgets, and router logits (which upstream does not capture).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_hooks.capture.graph import CaptureGraphState
from vllm.model_hooks.capture.selection import (
    count_selected_rows,
    may_select_step_rows,
    prepare_rows,
    selects_skipped_prompt_rows,
)
from vllm.model_hooks.capture.store import StreamConfig, StreamStore
from vllm.model_hooks.components.registry import (
    ATTENTION_HEADS,
    COMPONENTS,
    ComponentDescriptor,
    ComponentTarget,
    ModelComponents,
)

if TYPE_CHECKING:
    from vllm.model_hooks.selection.batch import BatchGeometry

logger = init_logger(__name__)


class CaptureSession:
    """Owns capture hooks and streams for one worker's model."""

    def __init__(self):
        self._streams: dict[str, StreamStore | None] = dict.fromkeys(COMPONENTS)
        # Engine-internal request id -> {stream: SelectSpec wire dict}.
        self._request_selects: dict[str, dict[str, dict]] = {}
        # Retain cache-hit metadata even with no stream enabled. A stream
        # started later must check rows that were already skipped.
        self._cache_hits: dict[str, tuple[list[int], int]] = {}
        self._unsupported_requests: set[str] = set()
        self._hook_handles: list = []
        self._hooked_layers: dict[str, set[int]] = {name: set() for name in COMPONENTS}
        self._layouts: dict[str, dict[int, dict[str, int]]] = {}
        self._attached = False
        self.graph_state: CaptureGraphState | None = None
        self._graph_replays = 0
        self._eager_capture_forwards = 0

    # ------------------------------------------------------------------
    # Per-request selection overrides (runner admission/completion lifecycle)
    # ------------------------------------------------------------------

    def add_request(
        self,
        req_id: str,
        capture_select: dict[str, dict],
        *,
        capture_supported: bool = True,
    ) -> None:
        """Register a request's selection override ({stream: wire}).

        Clause structure is validated at admission (input processor); a
        raise here would take down the engine core, so unknown streams warn
        and drop the override. Known overrides survive disabled streams and
        preemption; reductions ignore them while enabled.
        """
        if capture_supported:
            self._unsupported_requests.discard(req_id)
        else:
            self._unsupported_requests.add(req_id)
        accepted = {}
        for stream, wire in capture_select.items():
            store = self._streams.get(stream)
            if stream not in self._streams:
                logger.warning_once(
                    "Request carries a capture select for stream %r, "
                    "which is unknown; the override is ignored.",
                    stream,
                )
                continue
            if store is not None and store.config.reduce != "all":
                logger.warning_once(
                    "Per-request capture selects cannot combine with "
                    "the %r reduction; the override is ignored.",
                    store.config.reduce,
                )
            accepted[stream] = wire
        if accepted:
            self._request_selects[req_id] = accepted
        else:
            self._request_selects.pop(req_id, None)

    def finish_requests(self, req_ids: set[str]) -> None:
        """Forget completed request history while retaining preempted requests."""
        for req_id in req_ids:
            self._cache_hits.pop(req_id, None)
            self._request_selects.pop(req_id, None)
        self._unsupported_requests.difference_update(req_ids)
        for store in self._streams.values():
            if store is not None:
                store.finish_requests(req_ids)

    def fail_requests(self, errors: dict[str, str]) -> None:
        """Reject fetches of rows from requests whose steering failed."""
        if not errors:
            return
        for store in self._streams.values():
            if store is not None:
                store.fail_requests(errors)

    def mark_cache_elided(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        num_computed_tokens: int,
    ) -> None:
        """Flag selected prompt rows skipped by a new request's cache hit.

        Request overrides have already been registered by the runner. A hit is
        harmless when every wanted row is in the uncached tail or generation.
        """
        self._cache_hits[req_id] = (prompt_token_ids, num_computed_tokens)
        for stream, store in self._streams.items():
            if store is None:
                continue
            self._check_skipped_rows(
                stream, store, req_id, prompt_token_ids, num_computed_tokens
            )

    def _check_skipped_rows(
        self,
        stream: str,
        store: StreamStore,
        req_id: str,
        prompt_token_ids: list[int],
        num_computed_tokens: int,
    ) -> None:
        select = store.config.select
        if store.config.reduce == "all":
            select = self._request_selects.get(req_id, {}).get(stream, select)
        if selects_skipped_prompt_rows(
            select,
            prompt_token_ids,
            num_computed_tokens,
            reduce=store.config.reduce,
        ):
            store.mark_elided(req_id)

    # ------------------------------------------------------------------
    # Hook attachment (once, at model load; inert until a stream enables)
    # ------------------------------------------------------------------

    def attach(self, model: nn.Module, components: ModelComponents) -> None:
        if self._attached:
            return
        for component in COMPONENTS.values():
            self._layouts[component.id] = {
                target.layer_id: {
                    key: value
                    for key in ("width", "num_heads", "head_size")
                    if (value := getattr(target, key)) is not None
                }
                for target in components[component.id]
                if target.width is not None
            }
            self._hooked_layers[component.id] = self._attach_stream_hooks(
                components[component.id], component
            )

        def flush_hook(mod, args, output):
            if self.graph_state is not None and self.graph_state.recording:
                return
            if any(
                store is not None and store._pending for store in self._streams.values()
            ):
                self._eager_capture_forwards += 1
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
        targets: tuple[ComponentTarget, ...],
        component: ComponentDescriptor,
    ) -> set[int]:
        """Attach capture with the same discovery and output contract as steering."""
        stream = component.id
        hooked = set()
        for layer in targets:
            name, layer_id = layer.name, layer.layer_id
            target = layer.module

            def hook(mod, args, output, _lid=layer_id, _name=name, _width=layer.width):
                if torch.compiler.is_compiling():
                    # Ordinary compiled artifacts stay free of capture.
                    # A capture graph records raw forward with skip_compiled;
                    # this hook then only copies to its fixed GPU buffer.
                    return
                store = self._streams[stream]
                if store is None or not store.wants_layer(_lid):
                    return
                graph_state = self.graph_state
                recording = graph_state is not None and graph_state.recording
                if not recording and store.remaining_rows(_lid) == 0:
                    return
                tensor, tensor_owned = component.adapter.capture_rows(output)
                if tensor is None:
                    return
                if _width is not None and (
                    tensor.ndim != 2 or tensor.shape[-1] != _width
                ):
                    raise ValueError(
                        f"{stream} on {_name!r} expects (tokens, {_width}), "
                        f"got {tuple(tensor.shape)}"
                    )
                if recording:
                    assert graph_state is not None
                    graph_state.record(stream, _lid, tensor, _name)
                    return
                rows, meta = prepare_rows(
                    tensor, store, _lid, stream, self._request_selects, tensor_owned
                )
                if rows is not None:
                    store.append(_lid, rows, meta, _name)

            self._hook_handles.append(target.register_forward_hook(hook))
            hooked.add(layer_id)
        if hooked:
            logger.info("[Capture] hooked %d layers for %s", len(hooked), stream)
        return hooked

    def detach(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._attached = False
        self.release_graph()
        self._streams = dict.fromkeys(COMPONENTS)
        self._request_selects.clear()
        self._cache_hits.clear()
        self._unsupported_requests.clear()
        self._layouts.clear()
        for layers in self._hooked_layers.values():
            layers.clear()

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def any_enabled(self) -> bool:
        return any(store is not None for store in self._streams.values())

    def graph_signature(self) -> tuple:
        return tuple(
            (
                stream,
                tuple(
                    sorted(
                        lid
                        for lid in self._hooked_layers[stream]
                        if store.wants_layer(lid)
                    )
                ),
            )
            for stream, store in self._streams.items()
            if store is not None
        )

    def release_graph(self) -> None:
        """Release fixed outputs after the runner releases their graph."""
        if self.graph_state is not None:
            self.graph_state.close()
            self.graph_state = None

    def collect_graph_outputs(self, num_tokens: int) -> None:
        assert self.graph_state is not None
        self._graph_replays += 1
        for (stream, layer), (tensor, name) in self.graph_state.buffers.items():
            store = self._streams[stream]
            assert store is not None
            if store.remaining_rows(layer) == 0:
                continue
            rows, meta = prepare_rows(
                tensor[:num_tokens], store, layer, stream, self._request_selects
            )
            if rows is not None:
                store.append(layer, rows, meta, name)
        for store in self._streams.values():
            if store is not None:
                store.flush()

    def prepare_batch(self, geometry: "BatchGeometry") -> None:
        """Account for full layers even when this batch uses ordinary graphs."""
        unsupported = (
            {
                req_id: "Capture does not support requests with prompt embeddings"
                for req_id in geometry.req_ids
                if req_id in self._unsupported_requests
            }
            if self._unsupported_requests
            else {}
        )
        for stream, store in self._streams.items():
            if store is None:
                continue
            store.fail_requests(unsupported)
            # Completion cleanup can arrive one scheduler batch later. Check
            # only requests actually scheduled after this stream started.
            for req_id in geometry.req_ids:
                if req_id in unsupported or req_id in store._captured_requests:
                    continue
                hit = self._cache_hits.get(req_id)
                if hit is not None:
                    self._check_skipped_rows(stream, store, req_id, *hit)
            store.mark_processed(geometry.req_ids)
            if store.config.budget_rows is None:
                continue
            full_layers = sum(
                store.wants_layer(layer) and store.remaining_rows(layer) == 0
                for layer in self._hooked_layers[stream]
            )
            if full_layers:
                store.drop_rows(
                    full_layers
                    * count_selected_rows(
                        geometry, store, stream, self._request_selects
                    )
                )

    def needs_capture_for_batch(
        self,
        req_ids: list[str],
        num_computed: Sequence[int],
        num_scheduled: Sequence[int],
        prompt_lengths: Sequence[int],
        is_prefilling: Sequence[bool],
    ) -> bool:
        """Keep normal graph dispatch when every effective selection is empty."""
        for stream, store in self._streams.items():
            if store is None or not any(
                store.wants_layer(layer) and store.remaining_rows(layer) != 0
                for layer in self._hooked_layers[stream]
            ):
                continue
            for i, req_id in enumerate(req_ids):
                if req_id in self._unsupported_requests:
                    continue
                select = store.config.select
                if store.config.reduce == "all":
                    select = self._request_selects.get(req_id, {}).get(stream, select)
                if may_select_step_rows(
                    select,
                    int(num_computed[i]),
                    int(num_scheduled[i]),
                    int(prompt_lengths[i]),
                    bool(is_prefilling[i]),
                    reduce=store.config.reduce,
                ):
                    return True
        return False

    def enable_stream(self, stream: str, **config_kwargs) -> None:
        if stream not in self._streams:
            raise ValueError(f"Unknown capture stream: {stream}")
        if not self._attached:
            raise RuntimeError(
                "Capture hooks are not attached to this model; "
                "attachment happens once at model load."
            )
        config = StreamConfig(**config_kwargs)
        if stream == ATTENTION_HEADS:
            available = self._hooked_layers[stream]
            if not available:
                raise ValueError("No supported decoder attention head outputs found")
            if config.layers is not None and config.layers - available:
                raise ValueError(
                    "Unsupported attention head capture layers: "
                    f"{sorted(config.layers - available)}"
                )
        if not self._hooked_layers[stream]:
            logger.warning(
                "Enabling %s capture but no usable component hooks were found.",
                stream,
            )
        self._streams[stream] = StreamStore(config)

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
            result = store.serialize(
                layers=layers, req_ids=req_ids, clear_selected=clear
            )
        else:
            result = store.serialize(layers=layers)
            if clear:
                if layers is None:
                    store.clear()
                else:
                    store.drop_layers(list(result))
        for layer, info in result.items():
            if layout := self._layouts.get(stream, {}).get(layer):
                info["layout"] = layout.copy()
        return result

    def clear_stream(self, stream: str) -> None:
        store = self._streams.get(stream)
        if store is not None:
            store.clear()

    def stream_status(self, stream: str) -> dict[str, Any]:
        store = self._streams.get(stream)
        hooked = len(self._hooked_layers.get(stream, ()))
        execution = {
            "layouts": self._layouts.get(stream, {}),
            "graph_ready": self.graph_state is not None and self.graph_state.ready,
            "graph_buffer_bytes": 0
            if self.graph_state is None
            else self.graph_state.buffer_bytes,
            "graph_allocation_bytes": 0
            if self.graph_state is None
            else self.graph_state.allocation_bytes,
            # Session-lifetime counters survive stop; a single inactive graph
            # variant stays cached for repeated capture() calls.
            "graph_replays": self._graph_replays,
            "eager_capture_forwards": self._eager_capture_forwards,
        }
        if store is None:
            return {"enabled": False, "hooked_layers": hooked, **execution}
        return {
            "enabled": True,
            "hooked_layers": hooked,
            "layers_captured": len(store.chunks),
            "tokens_stored": store.tokens_stored,
            "tokens_dropped": store.tokens_dropped,
            "storage_bytes": store.storage_bytes,
            "budget_bytes": store.config.budget_bytes,
            "reduce": store.config.reduce,
            "select": store.config.select,
            "meta_complete": True,
            "cache_elided_reqs": len(store.elided_reqs),
            **execution,
        }
