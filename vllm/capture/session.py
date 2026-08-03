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
time with a layer subset, storage dtype, per-sample position reduction
("all" | "last" | "mean") and a token budget, bounding capture memory —
per-sample reductions turn an O(tokens) capture into O(samples), which
covers the common probing/diffmean workflows.

Storage appends one CPU chunk per forward step and concatenates at fetch
time; there is no batch-boundary heuristic.

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
    def __init__(
        self,
        layers: list[int] | None = None,
        dtype: str | None = None,
        positions: "str | list[int]" = "all",
        token_ids: list[int] | None = None,
        max_tokens: int | None = None,
    ):
        if isinstance(positions, (list, tuple)):
            if not positions:
                raise ValueError(
                    "positions list must be non-empty ('all' captures "
                    "every position)"
                )
            positions = list(positions)
        elif positions not in ("all", "last", "mean"):
            raise ValueError(
                "positions must be 'all', 'last', 'mean' or a list of "
                f"absolute positions, got {positions}"
            )
        if token_ids is not None and not token_ids:
            raise ValueError("token_ids must be None or non-empty")
        if token_ids is not None and isinstance(positions, str) and (
            positions in ("last", "mean")
        ):
            raise ValueError(
                "token_ids cannot combine with the 'last'/'mean' "
                "reductions; use positions='all' or a position list"
            )
        self.layers = set(layers) if layers is not None else None
        self.dtype = getattr(torch, dtype) if dtype else None
        self.positions = positions
        self.token_ids = list(token_ids) if token_ids is not None else None
        self.max_tokens = max_tokens
        if positions == "all" and token_ids is None and max_tokens is None:
            logger.warning_once(
                "Capture enabled with positions='all' and no max_tokens "
                "budget: every position of every request accumulates in "
                "CPU memory. Prefer a position list, token_ids, a "
                "reduction, or an explicit budget for long corpora."
            )

    @property
    def selects_rows(self) -> bool:
        """Whether a source-side row selection (not a reduction) is set."""
        return isinstance(self.positions, list) or self.token_ids is not None


class StreamStore:
    """Bounded, chunk-appending CPU store for one capture stream."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.chunks: dict[int, list[torch.Tensor]] = {}
        self.layer_names: dict[int, str] = {}
        # Rows stored per layer; the budget caps each layer at
        # max_tokens rows (i.e. sequence tokens, per layer).
        self._layer_rows: dict[int, int] = {}
        self.tokens_dropped = 0
        self._warned_budget = False
        self._pending: list[tuple[int, torch.Tensor, str]] = []
        self.lock = threading.Lock()

    @property
    def tokens_stored(self) -> int:
        return max(self._layer_rows.values(), default=0)

    def wants_layer(self, layer_id: int) -> bool:
        return self.config.layers is None or layer_id in self.config.layers

    def append(self, layer_id: int, tensor: torch.Tensor, layer_name: str):
        """Stage one layer's selected rows for this step (GPU side).

        The device-to-host copy happens once per step in ``flush()``:
        per-layer synchronous ``.cpu()`` calls stall the GPU stream once
        per hooked layer, which serializes capture-heavy runs.
        """
        with self.lock:
            stored = self._layer_rows.get(layer_id, 0)
            pending = sum(
                t.shape[0] for lid, t, _ in self._pending if lid == layer_id
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
            if self.config.dtype is not None:
                tensor = tensor.to(self.config.dtype)
            # `tensor` is owned by the hook (freshly materialized), so an
            # async copy in flush() cannot race buffer reuse.
            self._pending.append((layer_id, tensor.detach(), layer_name))

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
            for layer_id, tensor, layer_name in self._pending:
                if tensor.is_cuda:
                    host = torch.empty_like(tensor, device="cpu", pin_memory=True)
                    host.copy_(tensor, non_blocking=True)
                else:
                    host = tensor
                pinned.append((layer_id, host, layer_name))
            if any(t.is_cuda for _, t, _ in self._pending):
                torch.cuda.current_stream().synchronize()
            self._pending.clear()
            for layer_id, host, layer_name in pinned:
                self.chunks.setdefault(layer_id, []).append(host)
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
        """
        with self.lock:
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
                result[layer_id] = {
                    "data": wire.numpy().tobytes(),
                    "shape": list(tensor.shape),
                    "dtype": str(orig_dtype),
                    "encoding": "raw",
                    "layer_name": self.layer_names.get(layer_id, ""),
                }
            return result

    def drop_layers(self, layers: list[int]) -> None:
        with self.lock:
            for layer_id in layers:
                self.chunks.pop(layer_id, None)


def _select_rows(tensor: torch.Tensor, config: StreamConfig) -> torch.Tensor:
    """Select the configured rows of a [tokens, dim] step tensor.

    Source-side selection: only rows at the configured absolute
    positions (negatives resolve from each sample's prompt end, stable
    across prefill chunks) and/or rows whose input token id is in
    ``token_ids`` are kept — everything else never leaves the GPU. The
    two filters union when both are set. Falls back to keeping all rows
    when batch geometry is unavailable.
    """
    from vllm.forward_context import get_forward_context
    from vllm.steer_vectors.algorithms.triggers import (
        _isin_token_set,
        _match_positions,
    )
    from vllm.steer_vectors.discovery import extract_samples_info

    ctx = get_forward_context()
    samples_info = (
        extract_samples_info(ctx.attn_metadata)
        if ctx is not None and ctx.attn_metadata is not None
        else None
    )
    if samples_info is None:
        logger.warning_once(
            "Capture row selection requested but batch geometry is "
            "unavailable; keeping all rows for this step."
        )
        return tensor
    device = tensor.device
    qsl = samples_info["query_start_loc"].to(device)
    total = tensor.shape[0]
    all_positions = torch.arange(total, device=device)
    sample_ids = torch.searchsorted(qsl, all_positions, right=True) - 1
    relative = all_positions - qsl[:-1][sample_ids]
    num_computed = samples_info.get("num_computed")
    if num_computed is not None:
        abs_positions = relative + num_computed.to(device)[sample_ids]
    else:
        abs_positions = relative
    num_prompt = samples_info.get("num_prompt_tokens")
    neg_base = (
        num_prompt.to(device)
        if num_prompt is not None
        else qsl[1:] - qsl[:-1]
    )

    mask = torch.zeros(total, dtype=torch.bool, device=device)
    if isinstance(config.positions, list):
        mask |= _match_positions(
            abs_positions, config.positions, neg_base, sample_ids
        )
    if config.token_ids is not None:
        current_tokens = getattr(ctx, "current_tokens", None)
        if current_tokens is None:
            logger.warning_once(
                "Capture token_ids selection requested but the runner "
                "did not provide current_tokens; keeping all rows."
            )
            return tensor
        mask |= _isin_token_set(
            current_tokens[:total].to(device), config.token_ids
        )
    return tensor[mask]


def _sample_reduce(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce a [tokens, dim] step tensor to one row per sample.

    Under chunked prefill, 'last' keeps only samples whose chunk is
    final (decode samples, or prefill chunks reaching the prompt end),
    so each sample contributes one row per logical step, not one per
    chunk. 'mean' is inherently per-step: continuation chunks produce
    per-chunk means (warned once) — use 'all' and reduce client-side
    when chunked prompts need exact means.
    """
    if mode == "all":
        return tensor
    from vllm.forward_context import get_forward_context
    from vllm.steer_vectors.discovery import extract_samples_info

    ctx = get_forward_context()
    samples_info = (
        extract_samples_info(ctx.attn_metadata)
        if ctx is not None and ctx.attn_metadata is not None
        else None
    )
    if samples_info is None:
        # No sample boundaries available; keep all tokens.
        return tensor
    qsl = samples_info["query_start_loc"].to(tensor.device)
    starts = qsl[:-1].clamp(max=tensor.shape[0])
    ends = qsl[1:].clamp(max=tensor.shape[0])
    num_computed = samples_info.get("num_computed")
    num_prompt = samples_info.get("num_prompt_tokens")
    if mode == "last":
        idx = (ends - 1).clamp(min=0)
        if num_computed is not None and num_prompt is not None:
            # Keep decode samples and final prefill chunks only.
            final = (num_computed + (ends - starts)) >= num_prompt.to(ends.device)
            idx = idx[final]
        return tensor[idx]
    # mean
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
    return torch.stack(rows)


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
                if store.config.selects_rows:
                    selected = _select_rows(complete, store.config)
                else:
                    selected = _sample_reduce(complete, store.config.positions)
                store.append(_lid, selected, _name)

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
                if store.config.selects_rows:
                    selected = _select_rows(logits, store.config)
                else:
                    selected = _sample_reduce(logits, store.config.positions)
                store.append(_lid, selected, _name)

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
        }
