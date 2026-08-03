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
steering), a per-sample reduction ("all" | "last" | "mean") and a token
budget, bounding capture memory — selection and reductions turn an
O(tokens) capture into O(matches)/O(samples), which covers the common
probing/diffmean workflows.

Storage appends one CPU chunk per forward step and concatenates at fetch
time; there is no batch-boundary heuristic. Every stored row is labelled
with (request id, absolute position, token id) so clients never
re-derive sample alignment.

Relation to upstream: vLLM's ``extract_hidden_states`` speculative
method (with a hidden-states KV connector) also extracts per-layer
hidden states, computed identically (hidden + residual, see
``SupportsEagle3._maybe_add_hidden_state``) — so values from the two
mechanisms are directly comparable. Prefer that path for bulk offline
extraction over a fixed layer set; this session is for interactive use:
runtime enable/disable, arbitrary layer subsets with no graph change,
reductions/budgets, and router logits (which upstream does not capture).
"""

import threading
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

HIDDEN_STATES = "hidden_states"
ROUTER_LOGITS = "router_logits"


class StreamConfig:
    """Per-stream capture configuration.

    Row selection uses the shared `SelectSpec` where-clause language
    (`select`, wire form of ``vllm.steer_vectors.SelectSpec``) — the
    same clause semantics steering resolves, so "which rows to capture"
    and "which tokens to steer" mean the same thing. The legacy
    `positions=[...]`/`token_ids=[...]` kwargs translate into an
    equivalent clause. `positions` as a string keeps the within-sample
    reductions ("all" | "last" | "mean").
    """

    def __init__(
        self,
        layers: list[int] | None = None,
        dtype: str | None = None,
        positions: "str | list[int]" = "all",
        token_ids: list[int] | None = None,
        max_tokens: int | None = None,
        select: dict | None = None,
    ):
        from vllm.steer_vectors.api import SelectSpec

        legacy_positions: list[int] | None = None
        if isinstance(positions, (list, tuple)):
            if not positions:
                raise ValueError(
                    "positions list must be non-empty ('all' captures "
                    "every position)"
                )
            legacy_positions = list(positions)
            positions = "all"
        elif positions not in ("all", "last", "mean"):
            raise ValueError(
                "positions must be 'all', 'last', 'mean' or a list of "
                f"absolute positions, got {positions}"
            )
        if token_ids is not None and not token_ids:
            raise ValueError("token_ids must be None or non-empty")
        if select is not None and (
            legacy_positions is not None or token_ids is not None
        ):
            raise ValueError(
                "select cannot combine with the legacy positions-list/"
                "token_ids kwargs; put the filters inside the select "
                "clause"
            )
        if positions in ("last", "mean") and (
            select is not None or token_ids is not None
        ):
            raise ValueError(
                "row selection cannot combine with the 'last'/'mean' "
                "reductions; use positions='all' with a select clause"
            )
        if select is None and (
            legacy_positions is not None or token_ids is not None
        ):
            select = SelectSpec(
                phases=["prompt", "generation"],
                tokens=list(token_ids) if token_ids is not None else None,
                positions=legacy_positions,
            ).to_wire()
        elif select is not None:
            # Validate at enable time instead of failing mid-forward.
            select = SelectSpec.from_wire(select).to_wire()

        self.layers = set(layers) if layers is not None else None
        self.dtype = getattr(torch, dtype) if dtype else None
        self.positions = positions
        self.select = select
        self.max_tokens = max_tokens
        if positions == "all" and select is None and max_tokens is None:
            logger.warning_once(
                "Capture enabled with positions='all', no select clause "
                "and no max_tokens budget: every position of every "
                "request accumulates in CPU memory. Prefer a select "
                "clause, a reduction, or an explicit budget for long "
                "corpora."
            )

    @property
    def selects_rows(self) -> bool:
        """Whether a source-side row selection (not a reduction) is set."""
        return self.select is not None


class StreamStore:
    """Bounded, chunk-appending CPU store for one capture stream.

    Every stored row carries labels — ``(req_idx, position, token_id)``
    int32 columns, where ``req_idx`` indexes ``req_table`` (request id
    strings) — so clients never re-derive sample alignment. Rows
    captured while batch geometry was unavailable poison the labels for
    the whole store (``meta`` serializes as None with a warning) rather
    than shipping silently misaligned labels.
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self.chunks: dict[int, list[torch.Tensor]] = {}
        self.meta_chunks: dict[int, list[torch.Tensor]] = {}
        self.layer_names: dict[int, str] = {}
        self.req_table: list[str] = []
        self._req_index: dict[str, int] = {}
        self.meta_complete = True
        # Rows stored per layer; the budget caps each layer at
        # max_tokens rows (i.e. sequence tokens, per layer).
        self._layer_rows: dict[int, int] = {}
        self.tokens_dropped = 0
        self._warned_budget = False
        self._pending: list[
            tuple[int, torch.Tensor, torch.Tensor | None, str]
        ] = []
        self.lock = threading.Lock()

    @property
    def tokens_stored(self) -> int:
        return max(self._layer_rows.values(), default=0)

    def wants_layer(self, layer_id: int) -> bool:
        return self.config.layers is None or layer_id in self.config.layers

    def req_index(self, req_id: str) -> int:
        """Stable index of a request id in this store's req_table."""
        idx = self._req_index.get(req_id)
        if idx is None:
            idx = len(self.req_table)
            self.req_table.append(req_id)
            self._req_index[req_id] = idx
        return idx

    def append(
        self,
        layer_id: int,
        tensor: torch.Tensor,
        meta: torch.Tensor | None,
        layer_name: str,
    ):
        """Stage one layer's selected rows for this step (GPU side).

        ``meta`` is an int32 ``[rows, 3]`` tensor of
        (req_idx, position, token_id) labels, or None when geometry was
        unavailable. The device-to-host copy happens once per step in
        ``flush()``: per-layer synchronous ``.cpu()`` calls stall the
        GPU stream once per hooked layer, which serializes
        capture-heavy runs.
        """
        with self.lock:
            if meta is None:
                self.meta_complete = False
            elif meta.shape[0] != tensor.shape[0]:
                raise RuntimeError(
                    f"capture meta rows ({meta.shape[0]}) != data rows "
                    f"({tensor.shape[0]}) for layer {layer_id}"
                )
            stored = self._layer_rows.get(layer_id, 0)
            pending = sum(
                t.shape[0] for lid, t, _, _ in self._pending if lid == layer_id
            )
            stored += pending
            if self.config.max_tokens is not None and stored >= self.config.max_tokens:
                self.tokens_dropped += tensor.shape[0]
                if not self._warned_budget:
                    self._warned_budget = True
                    logger.warning(
                        "Capture token budget (%d rows per layer) "
                        "reached; further tokens are dropped.",
                        self.config.max_tokens,
                    )
                return
            if self.config.max_tokens is not None:
                keep = self.config.max_tokens - stored
                if tensor.shape[0] > keep:
                    self.tokens_dropped += tensor.shape[0] - keep
                    tensor = tensor[:keep]
                    if meta is not None:
                        meta = meta[:keep]
            if self.config.dtype is not None:
                tensor = tensor.to(self.config.dtype)
            # `tensor` is owned by the hook (freshly materialized), so an
            # async copy in flush() cannot race buffer reuse.
            self._pending.append(
                (
                    layer_id,
                    tensor.detach(),
                    meta.detach() if meta is not None else None,
                    layer_name,
                )
            )

    def flush(self):
        """Move this step's staged rows to CPU in one coalesced pass.

        Copies go through pinned staging buffers with non_blocking=True
        (one stream sync at the end), amortizing D2H latency across all
        hooked layers instead of paying it per layer.
        """
        with self.lock:
            if not self._pending:
                return
            pinned = []
            any_cuda = False
            for layer_id, tensor, meta, layer_name in self._pending:
                host_pair = []
                for t in (tensor, meta):
                    if t is not None and t.is_cuda:
                        host = torch.empty_like(t, device="cpu", pin_memory=True)
                        host.copy_(t, non_blocking=True)
                        any_cuda = True
                    else:
                        host = t
                    host_pair.append(host)
                pinned.append((layer_id, host_pair[0], host_pair[1], layer_name))
            if any_cuda:
                torch.cuda.current_stream().synchronize()
            self._pending.clear()
            for layer_id, host, meta_host, layer_name in pinned:
                self.chunks.setdefault(layer_id, []).append(host)
                if meta_host is not None:
                    self.meta_chunks.setdefault(layer_id, []).append(meta_host)
                self.layer_names[layer_id] = layer_name
                self._layer_rows[layer_id] = (
                    self._layer_rows.get(layer_id, 0) + host.shape[0]
                )

    def serialize(
        self, layers: list[int] | None = None
    ) -> dict[int, dict[str, Any]]:
        """Concatenate chunks and pack for RPC transmission.

        Tensors ship as raw bytes of their stored dtype (bf16 rides as
        int16 bytes and is reinterpreted client-side) — no float32
        upcast, so the wire volume equals the stored volume. Passing
        ``layers`` serializes a subset, letting clients fetch layer by
        layer instead of one monolithic message.

        Each layer entry carries a ``meta`` sub-dict labelling its rows
        (``req_table`` request-id strings + int32 ``req_idx`` /
        ``positions`` / ``token_ids`` columns), or ``meta: None`` when
        any row was captured without batch geometry.
        """
        with self.lock:
            if not self.meta_complete:
                logger.warning_once(
                    "Capture rows were stored without batch geometry; "
                    "row labels (meta) are unavailable for this store."
                )
            result: dict[int, dict[str, Any]] = {}
            wanted = sorted(self.chunks) if layers is None else [
                lid for lid in layers if lid in self.chunks
            ]
            for layer_id in wanted:
                tensor = torch.cat(self.chunks[layer_id], dim=0).contiguous()
                orig_dtype = tensor.dtype
                if tensor.dtype == torch.bfloat16:
                    wire = tensor.view(torch.int16)
                else:
                    wire = tensor
                meta_wire = None
                if self.meta_complete and layer_id in self.meta_chunks:
                    meta = torch.cat(
                        self.meta_chunks[layer_id], dim=0
                    ).contiguous()
                    meta_wire = {
                        "req_table": list(self.req_table),
                        "req_idx": meta[:, 0].numpy().tobytes(),
                        "positions": meta[:, 1].numpy().tobytes(),
                        "token_ids": meta[:, 2].numpy().tobytes(),
                    }
                result[layer_id] = {
                    "data": wire.numpy().tobytes(),
                    "shape": list(tensor.shape),
                    "dtype": str(orig_dtype),
                    "encoding": "raw",
                    "layer_name": self.layer_names.get(layer_id, ""),
                    "meta": meta_wire,
                }
            return result

    def drop_layers(self, layers: list[int]) -> None:
        with self.lock:
            for layer_id in layers:
                self.chunks.pop(layer_id, None)
                self.meta_chunks.pop(layer_id, None)


def _prepare_rows(
    tensor: torch.Tensor, store: StreamStore
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Select/reduce a [tokens, dim] step tensor and label the rows.

    Returns ``(rows, meta)``: the rows to store and an int32
    ``[rows, 3]`` label tensor of (req_idx, position, token_id), where
    req_idx indexes the store's req_table. ``rows`` is None when
    nothing matches this step. ``meta`` is None when batch geometry or
    request identity is unavailable (fallback: all rows, unlabeled).

    Selection resolves the stream's `select` clause with the same
    trigger collector steering uses, so the clause semantics are
    identical. Reductions:
    - 'last': one row per sample per step; under chunked prefill only
      samples whose chunk is final contribute (one row per logical
      step, not one per chunk). Labelled with the row's own position
      and token.
    - 'mean': one synthesized row per sample per step, labelled with
      position -1 and token_id -1. Continuation chunks produce
      per-chunk means (warned once) — use 'all' and reduce client-side
      when chunked prompts need exact means.
    """
    from vllm.forward_context import get_forward_context
    from vllm.steer_vectors.algorithms.triggers import (
        collect_positions_apply_spec,
    )
    from vllm.steer_vectors.discovery import extract_samples_info

    config = store.config
    ctx = get_forward_context()
    samples_info = (
        extract_samples_info(ctx.attn_metadata)
        if ctx is not None and ctx.attn_metadata is not None
        else None
    )
    current_tokens = getattr(ctx, "current_tokens", None) if ctx else None
    if samples_info is None or (
        config.selects_rows and current_tokens is None
    ):
        if config.selects_rows or config.positions != "all":
            logger.warning_once(
                "Capture selection/reduction requested but batch "
                "geometry is unavailable; keeping all rows unlabelled "
                "for this step."
            )
        return tensor, None

    device = tensor.device
    qsl = samples_info["query_start_loc"].to(device)
    total = int(qsl[-1].item())
    total = min(total, tensor.shape[0])
    num_computed = samples_info.get("num_computed")
    num_prompt = samples_info.get("num_prompt_tokens")

    all_positions = torch.arange(total, device=device)
    sample_ids = torch.searchsorted(qsl, all_positions, right=True) - 1
    relative = all_positions - qsl[:-1][sample_ids]
    if num_computed is not None:
        abs_positions = relative + num_computed.to(device)[sample_ids]
    else:
        abs_positions = relative

    if config.selects_rows:
        indices = collect_positions_apply_spec(
            current_tokens=current_tokens[:total].to(device),
            samples_info=samples_info,
            spec=config.select,
        )
        if indices is None:
            return None, None
    elif config.positions == "last":
        starts = qsl[:-1].clamp(max=total)
        ends = qsl[1:].clamp(max=total)
        indices = (ends - 1).clamp(min=0)
        if num_computed is not None and num_prompt is not None:
            # Keep decode samples and final prefill chunks only.
            final = (num_computed.to(ends.device) + (ends - starts)) >= (
                num_prompt.to(ends.device)
            )
            indices = indices[final]
    elif config.positions == "mean":
        return _mean_reduce(tensor, store, ctx, qsl, total, num_computed, num_prompt)
    else:  # "all"
        indices = all_positions

    rows = tensor[:total][indices] if indices is not all_positions else tensor[:total]
    meta = _row_labels(
        store, ctx, indices, sample_ids, abs_positions, current_tokens, total
    )
    return rows, meta


def _row_labels(
    store: StreamStore,
    ctx,
    indices: torch.Tensor,
    sample_ids: torch.Tensor,
    abs_positions: torch.Tensor,
    current_tokens: torch.Tensor | None,
    total: int,
) -> torch.Tensor | None:
    """Build int32 [rows, 3] (req_idx, position, token_id) labels."""
    req_ids = getattr(ctx, "req_ids", None) if ctx else None
    if req_ids is None or current_tokens is None:
        logger.warning_once(
            "Capture cannot label rows: the runner did not provide "
            "req_ids/current_tokens for this step."
        )
        return None
    device = sample_ids.device
    step_req = torch.tensor(
        [store.req_index(r) for r in req_ids],
        dtype=torch.int32,
        device=device,
    )
    sel_samples = sample_ids[indices]
    return torch.stack(
        [
            step_req[sel_samples],
            abs_positions[indices].to(torch.int32),
            current_tokens[:total].to(device)[indices].to(torch.int32),
        ],
        dim=1,
    )


def _mean_reduce(
    tensor: torch.Tensor,
    store: StreamStore,
    ctx,
    qsl: torch.Tensor,
    total: int,
    num_computed,
    num_prompt,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One mean row per sample; labels use position/token sentinel -1."""
    starts = qsl[:-1].clamp(max=total)
    ends = qsl[1:].clamp(max=total)
    if (
        num_computed is not None
        and num_prompt is not None
        and bool(((num_computed > 0) & (num_computed < num_prompt)).any())
    ):
        logger.warning_once(
            "Capture 'mean' reduction under chunked prefill produces one "
            "mean per chunk, not per prompt; use positions='all' and "
            "reduce client-side for chunked prompts."
        )
    rows = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        rows.append(tensor[s:e].mean(dim=0) if e > s else torch.zeros_like(tensor[0]))
    stacked = torch.stack(rows)
    req_ids = getattr(ctx, "req_ids", None) if ctx else None
    if req_ids is None:
        logger.warning_once(
            "Capture cannot label rows: the runner did not provide "
            "req_ids for this step."
        )
        return stacked, None
    step_req = torch.tensor(
        [store.req_index(r) for r in req_ids],
        dtype=torch.int32,
        device=stacked.device,
    )
    sentinel = torch.full_like(step_req, -1)
    return stacked, torch.stack([step_req, sentinel, sentinel], dim=1)


class CaptureSession:
    """Owns capture hooks and streams for one worker's model."""

    def __init__(self):
        self._streams: dict[str, StreamStore | None] = {
            HIDDEN_STATES: None,
            ROUTER_LOGITS: None,
        }
        self._hook_handles: list = []
        self._hidden_layers = 0
        self._gate_layers = 0
        self._attached = False

    # ------------------------------------------------------------------
    # Hook attachment (once, at model load; inert until a stream enables)
    # ------------------------------------------------------------------

    def attach(self, model: nn.Module) -> None:
        if self._attached:
            return
        self._attach_hidden_hooks(model)
        self._attach_gate_hooks(model)

        def flush_hook(mod, args, output):
            for store in self._streams.values():
                if store is not None:
                    store.flush()

        # One post-forward flush per step: per-layer hooks only stage
        # GPU rows; the D2H copies coalesce here (pinned, non-blocking,
        # single sync).
        self._hook_handles.append(model.register_forward_hook(flush_hook))
        self._attached = True

    def _attach_hidden_hooks(self, model: nn.Module) -> None:
        from vllm.steer_vectors.discovery import (
            SUPPORTED_DECODER_LAYERS,
            extract_layer_id_from_module_name,
            find_decoder_layers,
            split_decoder_output,
        )

        matches = find_decoder_layers(model)
        if not matches:
            matches = {
                name: module
                for name, module in model.named_modules()
                if any(
                    cls in module.__class__.__name__ for cls in SUPPORTED_DECODER_LAYERS
                )
            }

        fallback_id = 0
        for name, module in matches.items():
            layer_id = extract_layer_id_from_module_name(name)
            if layer_id is None:
                layer_id = fallback_id
            fallback_id = layer_id + 1

            def hook(mod, args, output, _lid=layer_id, _name=name):
                store = self._streams[HIDDEN_STATES]
                if store is None or not store.wants_layer(_lid):
                    return
                hidden, residual, _, _ = split_decoder_output(output)
                if not isinstance(hidden, torch.Tensor):
                    return
                complete = hidden + residual if residual is not None else hidden
                rows, meta = _prepare_rows(complete, store)
                if rows is not None:
                    store.append(_lid, rows, meta, _name)

            self._hook_handles.append(module.register_forward_hook(hook))
        self._hidden_layers = len(matches)
        if matches:
            logger.info(
                "[Capture] hooked %d decoder layers for hidden states",
                len(matches),
            )

    def _attach_gate_hooks(self, model: nn.Module) -> None:
        from vllm.steer_vectors.discovery import (
            extract_gate_logits,
            extract_layer_id_from_module_name,
            find_moe_blocks,
            find_moe_gate,
            moe_gate_is_fused,
        )

        blocks = find_moe_blocks(model)
        fallback_id = 0
        for name, block in blocks.items():
            gate = find_moe_gate(block)
            if gate is None:
                logger.warning(
                    "[Capture] MoE block %s has no gate/router submodule; "
                    "its router logits cannot be captured.",
                    name,
                )
                continue
            if moe_gate_is_fused(block):
                logger.warning(
                    "[Capture] MoE block %s fuses gate weights into the "
                    "MoE runner (gate forward bypassed); its router "
                    "logits cannot be captured.",
                    name,
                )
                continue
            layer_id = extract_layer_id_from_module_name(name)
            if layer_id is None:
                layer_id = fallback_id
            fallback_id = layer_id + 1

            def hook(mod, args, output, _lid=layer_id, _name=name):
                store = self._streams[ROUTER_LOGITS]
                if store is None or not store.wants_layer(_lid):
                    return
                logits = extract_gate_logits(output)
                if logits is None:
                    return
                rows, meta = _prepare_rows(logits, store)
                if rows is not None:
                    store.append(_lid, rows, meta, _name)

            # NOTE: registered after any steering gate hook (steering wrap
            # runs first at load), so captured logits are post-steering.
            self._hook_handles.append(gate.register_forward_hook(hook))
            self._gate_layers += 1
        if self._gate_layers:
            logger.info(
                "[Capture] hooked %d MoE gates for router logits",
                self._gate_layers,
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
    ) -> dict[int, dict[str, Any]]:
        store = self._streams.get(stream)
        if store is None:
            return {}
        store.flush()  # capture any rows staged since the last step
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
            "positions": store.config.positions,
            "select": store.config.select,
            "meta_complete": store.meta_complete,
        }
