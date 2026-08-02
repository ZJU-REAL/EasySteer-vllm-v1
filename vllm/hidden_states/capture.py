# SPDX-License-Identifier: Apache-2.0
"""Hook-based capture of intermediate model state.

One CaptureSession per worker owns named capture *streams*:

- ``hidden_states``: complete post-layer hidden states (hidden + residual)
  from every decoder layer, discovered by the same class-name lists +
  structural fallback as steering (architecture-agnostic).
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
"""

import logging
import threading
from typing import Any, Dict, List, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

HIDDEN_STATES = "hidden_states"
ROUTER_LOGITS = "router_logits"


class StreamConfig:
    def __init__(
        self,
        layers: Optional[List[int]] = None,
        dtype: Optional[str] = None,
        positions: str = "all",
        max_tokens: Optional[int] = None,
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
        self.chunks: Dict[int, List[torch.Tensor]] = {}
        self.layer_names: Dict[int, str] = {}
        # Rows stored per layer; the budget caps each layer at
        # max_tokens rows (i.e. sequence tokens, per layer).
        self._layer_rows: Dict[int, int] = {}
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
            if (
                self.config.max_tokens is not None
                and stored >= self.config.max_tokens
            ):
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

    def serialize(self) -> Dict[int, Dict[str, Any]]:
        """Concatenate chunks and pack for RPC transmission."""
        with self.lock:
            result: Dict[int, Dict[str, Any]] = {}
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
    """Reduce a [tokens, dim] step tensor to one row per sample."""
    if mode == "all":
        return tensor
    from vllm.forward_context import get_forward_context
    from vllm.steer_vectors.algorithms.utils import extract_samples_info

    ctx = get_forward_context()
    samples_info = (
        extract_samples_info(ctx.attn_metadata)
        if ctx is not None and ctx.attn_metadata is not None else None
    )
    if samples_info is None:
        # No sample boundaries available; keep all tokens.
        return tensor
    qsl = samples_info["query_start_loc"].to(tensor.device)
    ends = qsl[1:].clamp(max=tensor.shape[0])
    if mode == "last":
        return tensor[(ends - 1).clamp(min=0)]
    # mean
    starts = qsl[:-1].clamp(max=tensor.shape[0])
    rows = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        rows.append(
            tensor[s:e].mean(dim=0) if e > s
            else torch.zeros_like(tensor[0])
        )
    return torch.stack(rows)


class CaptureSession:
    """Owns capture hooks and streams for one worker's model."""

    def __init__(self):
        self._streams: Dict[str, Optional[StreamStore]] = {
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
        from vllm.steer_vectors.config import get_target_modules
        from vllm.steer_vectors.layers import (
            _extract_hidden_states_and_residual,
            extract_layer_id_from_module_name,
        )
        from vllm.steer_vectors.models import find_decoder_layers_structurally

        target_classes = get_target_modules("decoder_layer")
        matches = {
            name: module
            for name, module in model.named_modules()
            if any(cls in module.__class__.__name__ for cls in target_classes)
        }
        if not matches:
            matches = find_decoder_layers_structurally(model)

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
                hidden, residual, _, _ = \
                    _extract_hidden_states_and_residual(output)
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
        from vllm.steer_vectors.layers import (
            extract_gate_logits,
            extract_layer_id_from_module_name,
        )
        from vllm.steer_vectors.models import (
            find_moe_blocks_structurally,
            find_moe_gate,
        )

        blocks = find_moe_blocks_structurally(model)
        fallback_id = 0
        for name, block in blocks.items():
            gate = find_moe_gate(block)
            if gate is None:
                logger.warning(
                    "[Capture] MoE block %s has no gate/router submodule; "
                    "its router logits cannot be captured.", name,
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
    ) -> Dict[int, Dict[str, Any]]:
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

    def stream_status(self, stream: str) -> Dict[str, Any]:
        store = self._streams.get(stream)
        hooked = (
            self._hidden_layers if stream == HIDDEN_STATES
            else self._gate_layers
        )
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
