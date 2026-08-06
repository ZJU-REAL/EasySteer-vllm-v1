# SPDX-License-Identifier: Apache-2.0
"""Per-stream capture configuration and bounded CPU row storage."""

import threading
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class StreamConfig:
    """Per-stream capture configuration.

    Row selection uses the shared `SelectSpec` where-clause language
    (`select`, wire form of ``vllm.steer_vectors.SelectSpec``) — the
    same clause semantics steering resolves, so "which rows to capture"
    and "which tokens to steer" mean the same thing. `reduce` applies a
    within-sample reduction ("all" | "last" | "mean"); `budget_rows`
    caps stored rows per layer.
    """

    def __init__(
        self,
        layers: list[int] | None = None,
        dtype: str | None = None,
        reduce: str = "all",
        select: dict | None = None,
        budget_rows: int | None = None,
    ):
        from vllm.steer_vectors.api import SelectSpec

        if reduce not in ("all", "last", "mean"):
            raise ValueError(
                f"reduce must be 'all', 'last' or 'mean', got {reduce!r}"
            )
        if reduce != "all" and select is not None:
            raise ValueError(
                "row selection cannot combine with the 'last'/'mean' "
                "reductions; use reduce='all' with a select clause"
            )
        if select is not None:
            # Validate at enable time instead of failing mid-forward.
            select = SelectSpec.from_wire(select).to_wire()
        if dtype is not None:
            torch_dtype = getattr(torch, dtype, None)
            if not isinstance(torch_dtype, torch.dtype):
                raise ValueError(f"unknown storage dtype {dtype!r}")
        else:
            torch_dtype = None

        self.layers = set(layers) if layers is not None else None
        self.dtype = torch_dtype
        self.reduce = reduce
        self.select = select
        self.budget_rows = budget_rows
        if reduce == "all" and select is None and budget_rows is None:
            logger.warning_once(
                "Capture enabled with reduce='all', no select clause and "
                "no budget_rows: every position of every request "
                "accumulates in CPU memory. Prefer a select clause, a "
                "reduction, or an explicit budget for long corpora."
            )

    @property
    def selects_rows(self) -> bool:
        """Whether a source-side row selection (not a reduction) is set."""
        return self.select is not None

    @property
    def wants_prompt_rows(self) -> bool:
        """Whether this stream can select prompt rows other than the
        final prompt token.

        Determines exposure to prefix-cache elision: a cache hit skips
        recomputation of the prompt head but always leaves at least the
        final prompt token to local compute, so the 'last' reduction and
        generation-only selections are unaffected.
        """
        if self.reduce == "last":
            return False
        if self.select is not None:
            covers_prompt = self.select.get("prompt") == "all" or any(
                self.select.get(key) is not None
                for key in ("prompt_tokens", "prompt_positions", "prompt_window")
            )
            if not covers_prompt:
                return False
        return True


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
        # Requests whose prompt head was skipped by a prefix-cache hit
        # while this stream was enabled: their early rows were never
        # computed, so the store is incomplete for them (fetch raises).
        self.elided_reqs: set[str] = set()
        # Rows stored per layer; the budget caps each layer at
        # budget_rows rows (i.e. sequence tokens, per layer).
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

    def mark_elided(self, req_id: str) -> None:
        """Flag a request admitted with a prefix-cache hit as incomplete.

        A request already present in the req_table captured rows before
        (e.g. rescheduled after preemption onto its own earlier blocks);
        nothing is missing for it, so it is not flagged.
        """
        with self.lock:
            if req_id not in self._req_index:
                self.elided_reqs.add(req_id)

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
            if self.config.budget_rows is not None and stored >= self.config.budget_rows:
                self.tokens_dropped += tensor.shape[0]
                if not self._warned_budget:
                    self._warned_budget = True
                    logger.warning(
                        "Capture token budget (%d rows per layer) "
                        "reached; further tokens are dropped.",
                        self.config.budget_rows,
                    )
                return
            if self.config.budget_rows is not None:
                keep = self.config.budget_rows - stored
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
        self,
        layers: list[int] | None = None,
        req_ids: list[str] | None = None,
        clear_selected: bool = False,
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

        ``req_ids`` (client-visible ids) restricts the payload to rows
        of those requests; with ``clear_selected`` the emitted rows are
        also removed from the store, so clients can drain request by
        request with bounded peak message size. Request filtering
        requires labelled rows and raises otherwise.
        """
        from vllm.capture.serde import match_capture_request_id

        with self.lock:
            elided = self.elided_reqs
            if req_ids is not None and elided:
                elided = {
                    rid
                    for rid in elided
                    if any(
                        match_capture_request_id(rid, ext) for ext in req_ids
                    )
                }
            if elided:
                raise RuntimeError(
                    "prefix-cache hits skipped recomputation of the prompt "
                    f"head for request(s) {sorted(elided)[:5]}; their early "
                    "rows were never computed, so this capture is "
                    "incomplete. Submit capture requests with a unique "
                    "cache_salt (easysteer's capture() does this "
                    "automatically), fetch with req_ids excluding them, or "
                    "clear the stream."
                )
            if not self.meta_complete:
                if req_ids is not None:
                    raise RuntimeError(
                        "Per-request fetch requires labelled rows, but "
                        "this store holds rows captured without batch "
                        "geometry."
                    )
                logger.warning_once(
                    "Capture rows were stored without batch geometry; "
                    "row labels (meta) are unavailable for this store."
                )
            table_match = None
            if req_ids is not None:
                table_match = torch.tensor(
                    [
                        i
                        for i, rid in enumerate(self.req_table)
                        if any(
                            match_capture_request_id(rid, ext)
                            for ext in req_ids
                        )
                    ],
                    dtype=torch.int32,
                )
            result: dict[int, dict[str, Any]] = {}
            wanted = sorted(self.chunks) if layers is None else [
                lid for lid in layers if lid in self.chunks
            ]
            for layer_id in wanted:
                tensor = torch.cat(self.chunks[layer_id], dim=0).contiguous()
                meta = None
                if self.meta_complete and layer_id in self.meta_chunks:
                    meta = torch.cat(
                        self.meta_chunks[layer_id], dim=0
                    ).contiguous()
                if table_match is not None:
                    assert meta is not None
                    row_mask = torch.isin(meta[:, 0], table_match)
                    if clear_selected:
                        keep = ~row_mask
                        self.chunks[layer_id] = [tensor[keep]]
                        self.meta_chunks[layer_id] = [meta[keep]]
                        self._layer_rows[layer_id] = int(keep.sum())
                    tensor = tensor[row_mask].contiguous()
                    meta = meta[row_mask].contiguous()
                    if tensor.shape[0] == 0:
                        continue
                orig_dtype = tensor.dtype
                if tensor.dtype == torch.bfloat16:
                    wire = tensor.view(torch.int16)
                else:
                    wire = tensor
                meta_wire = None
                if meta is not None:
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
