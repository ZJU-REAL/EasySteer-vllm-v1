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
        positions: str = "all",
        max_tokens: int | None = None,
    ):
        if positions not in ("all", "last", "mean"):
            raise ValueError(
                f"positions must be 'all', 'last' or 'mean', got {positions}"
            )
        self.layers = set(layers) if layers is not None else None
        self.dtype = getattr(torch, dtype) if dtype else None
        self.positions = positions
        self.max_tokens = max_tokens


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
        self.lock = threading.Lock()

    @property
    def tokens_stored(self) -> int:
        return max(self._layer_rows.values(), default=0)

    def wants_layer(self, layer_id: int) -> bool:
        return self.config.layers is None or layer_id in self.config.layers

    def append(self, layer_id: int, tensor: torch.Tensor, layer_name: str):
        with self.lock:
            stored = self._layer_rows.get(layer_id, 0)
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
            chunk = tensor.detach().cpu()
            self.chunks.setdefault(layer_id, []).append(chunk)
            self.layer_names[layer_id] = layer_name
            self._layer_rows[layer_id] = stored + chunk.shape[0]

    def serialize(self) -> dict[int, dict[str, Any]]:
        """Concatenate chunks and pack for RPC transmission."""
        with self.lock:
            result: dict[int, dict[str, Any]] = {}
            for layer_id in sorted(self.chunks):
                tensor = torch.cat(self.chunks[layer_id], dim=0)
                orig_dtype = tensor.dtype
                if tensor.dtype in (torch.bfloat16, torch.float16):
                    tensor = tensor.to(torch.float32)
                result[layer_id] = {
                    "data": tensor.numpy().tobytes(),
                    "shape": list(tensor.shape),
                    "dtype": str(orig_dtype),
                    "layer_name": self.layer_names.get(layer_id, ""),
                }
            return result


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
                store.append(
                    _lid,
                    _sample_reduce(complete, store.config.positions),
                    _name,
                )

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
                store.append(
                    _lid,
                    _sample_reduce(logits, store.config.positions),
                    _name,
                )

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
        self, stream: str, clear: bool = True
    ) -> dict[int, dict[str, Any]]:
        store = self._streams.get(stream)
        if store is None:
            return {}
        result = store.serialize()
        if clear:
            # Keep capturing into a fresh store with the same config.
            self._streams[stream] = StreamStore(store.config)
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
