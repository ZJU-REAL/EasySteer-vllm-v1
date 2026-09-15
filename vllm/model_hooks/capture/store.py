# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-stream capture configuration and bounded CPU row storage."""

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_hooks.capture.serialization import (
    match_capture_request_id,
    resolve_storage_dtype,
    serialize_capture_layer,
)
from vllm.model_hooks.selection.spec import SelectSpec

if TYPE_CHECKING:
    from vllm.model_hooks.selection.batch import BatchGeometry

logger = init_logger(__name__)


class StreamConfig:
    """Per-stream capture configuration.

    Row selection uses the shared `SelectSpec` where-clause language
    (`select`, wire form of ``vllm.model_hooks.selection.spec.SelectSpec``) — the
    same clause semantics steering resolves, so "which rows to capture"
    and "which tokens to steer" mean the same thing. `reduce` applies a
    within-sample reduction ("all" | "last" | "mean"); `budget_rows`
    caps stored rows per layer. `budget_bytes` limits stored values/labels and
    pending transfers across layers, reporting overflow when capture is fetched.
    `staging_bytes` bounds reusable pinned transfer storage; retained chunks use
    pageable memory. `device_budget_bytes` covers pending GPU rows and capture
    outputs reserved by the session.
    """

    def __init__(
        self,
        layers: list[int] | None = None,
        dtype: str | None = None,
        reduce: str = "all",
        select: dict | None = None,
        budget_rows: int | None = None,
        budget_bytes: int | None = None,
        staging_bytes: int = 16 * 1024 * 1024,
        device_budget_bytes: int | None = None,
    ):
        if reduce not in ("all", "last", "mean"):
            raise ValueError(f"reduce must be 'all', 'last' or 'mean', got {reduce!r}")
        if reduce != "all" and select is not None:
            raise ValueError(
                "row selection cannot combine with the 'last'/'mean' "
                "reductions; use reduce='all' with a select clause"
            )
        if select is not None:
            # Validate at enable time instead of failing mid-forward.
            select = SelectSpec.from_wire(select).to_wire()
        self.layers = set(layers) if layers is not None else None
        self.dtype = resolve_storage_dtype(dtype) if dtype is not None else None
        self.reduce = reduce
        self.select = select
        if budget_rows is not None and (
            type(budget_rows) is not int or budget_rows < 0
        ):
            raise ValueError("budget_rows must be a nonnegative integer")
        self.budget_rows = budget_rows
        if budget_bytes is not None and (
            type(budget_bytes) is not int or budget_bytes < 0
        ):
            raise ValueError("budget_bytes must be a nonnegative integer")
        self.budget_bytes = budget_bytes
        if type(staging_bytes) is not int or staging_bytes < 1:
            raise ValueError("staging_bytes must be a positive integer")
        self.staging_bytes = staging_bytes
        if device_budget_bytes is not None and (
            type(device_budget_bytes) is not int or device_budget_bytes < 0
        ):
            raise ValueError("device_budget_bytes must be a nonnegative integer")
        self.device_budget_bytes = device_budget_bytes
        if (
            reduce == "all"
            and select is None
            and budget_rows is None
            and budget_bytes is None
        ):
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


@dataclass
class _StoredChunk:
    tensor: torch.Tensor
    meta: torch.Tensor
    request_index: int


class StreamStore:
    """Bounded, chunk-appending CPU store for one capture stream.

    Every stored row carries labels — ``(req_idx, position, token_id)``
    int32 columns, where ``req_idx`` indexes ``req_table`` (request id
    strings) — so clients never re-derive sample alignment. Missing or
    malformed labels are rejected before any rows enter the store.
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self.chunks: dict[int, dict[int, _StoredChunk]] = {}
        self._next_chunk = 0
        self._request_chunks: dict[int, dict[int, set[int]]] = {}
        self.layer_names: dict[int, str] = {}
        self.req_table: dict[int, str] = {}
        self._req_index: dict[str, int] = {}
        self._request_aliases: dict[str, set[int]] = {}
        self._free_request_indices: list[int] = []
        self._next_request_index = 0
        self._unused_request_indices: set[int] = set()
        # Active requests retain capture history across draining and preemption.
        self._captured_requests: set[str] = set()
        # Cache hits that omitted selected prompt rows, including requests
        # already running when this stream started (fetch raises).
        self.elided_reqs: set[str] = set()
        self._failed_requests: dict[str, str] = {}
        # Rows stored per layer; the budget caps each layer at
        # budget_rows rows (i.e. sequence tokens, per layer).
        self._layer_rows: dict[int, int] = {}
        self.tokens_dropped = 0
        self._warned_budget = False
        self._pending: list[tuple[int, torch.Tensor, torch.Tensor, str]] = []
        self._staging: torch.Tensor | None = None
        self.device_reserved_bytes = 0
        self._stored_bytes = 0
        self._budget_error: str | None = None
        # Geometry and request selections are fixed during one forward pass.
        # Layers share its row plans; a fresh geometry invalidates every plan.
        self._row_plan_geometry: BatchGeometry | None = None
        self._row_plan_request_selects: dict[str, dict[str, dict]] | None = None
        self._row_plans: dict = {}
        self.lock = threading.Lock()

    @property
    def tokens_stored(self) -> int:
        return max(self._layer_rows.values(), default=0)

    def wants_layer(self, layer_id: int) -> bool:
        return self.config.layers is None or layer_id in self.config.layers

    @property
    def storage_bytes(self) -> int:
        """Raw stored values/labels plus their pending transfer volume.

        Graph buffers, model temporaries and serialization copies are separate.
        """
        return self._stored_bytes + self.pending_bytes

    @property
    def pending_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() + meta.numel() * meta.element_size()
            for _, tensor, meta, _ in self._pending
        )

    @property
    def pending_device_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for _, tensor, meta, _ in self._pending
            for value in (tensor, meta)
            if value.is_cuda
        )

    @property
    def staging_allocation_bytes(self) -> int:
        return 0 if self._staging is None else self._staging.numel()

    def _within_device_budget(self, additional_bytes: int) -> bool:
        budget = self.config.device_budget_bytes
        if budget is None:
            return True
        required = (
            self.device_reserved_bytes + self.pending_device_bytes + additional_bytes
        )
        if required > budget:
            self._budget_error = (
                f"Capture device staging budget ({budget} bytes) exceeded: "
                f"the next rows require {required} bytes. Reduce the capture "
                "batch, layers or selected rows, or increase device_budget_bytes. "
                "No complete capture result is available."
            )
            return False
        return True

    def row_bytes(self, tensor: torch.Tensor) -> int:
        dtype = self.config.dtype
        element_size = (
            tensor.element_size()
            if dtype is None
            else torch.empty((), dtype=dtype, device="cpu").element_size()
        )
        return (tensor.numel() // tensor.shape[0]) * element_size + 3 * 4

    def _within_byte_budget(self, rows: int, row_bytes: int) -> bool:
        if self._budget_error is not None:
            return False
        budget = self.config.budget_bytes
        if budget is None:
            return True
        required = self.storage_bytes + rows * row_bytes
        if required > budget:
            # Report at fetch, keeping the model forward and engine healthy.
            self._budget_error = (
                f"Capture CPU storage budget ({budget} bytes) exceeded: "
                f"the next rows require {required} bytes. Reduce the capture "
                "batch, layers or selected rows, or increase budget_bytes. "
                "No complete capture result is available."
            )
            return False
        return True

    def mark_elided(self, req_id: str) -> None:
        """Flag a request admitted with a prefix-cache hit as incomplete.

        Draining rows does not erase an active request's capture history:
        rescheduling onto its earlier blocks must not flag a new cache hit.
        """
        with self.lock:
            if req_id not in self._captured_requests:
                self.elided_reqs.add(req_id)

    def req_index(self, req_id: str) -> int:
        """Stable index of a request id in this store's req_table."""
        self._captured_requests.add(req_id)
        idx = self._req_index.get(req_id)
        if idx is None:
            if self._free_request_indices:
                idx = self._free_request_indices.pop()
            else:
                idx = self._next_request_index
                self._next_request_index += 1
            self.req_table[idx] = req_id
            self._req_index[req_id] = idx
            for alias in self._aliases(req_id):
                self._request_aliases.setdefault(alias, set()).add(idx)
            self._unused_request_indices.add(idx)
        return idx

    @staticmethod
    def _aliases(req_id: str) -> set[str]:
        external, _, _ = req_id.rpartition("-")
        return (
            {req_id, external}
            if match_capture_request_id(req_id, external)
            else {req_id}
        )

    def mark_processed(self, req_ids: list[str]) -> None:
        with self.lock:
            self._captured_requests.update(req_ids)

    def finish_requests(self, req_ids: set[str]) -> None:
        with self.lock:
            self._captured_requests.difference_update(req_ids)

    def fail_requests(self, errors: dict[str, str]) -> None:
        """Retain request failures so later fetches cannot return invalid capture."""
        if errors:
            with self.lock:
                self._failed_requests.update(errors)

    def remaining_rows(self, layer_id: int) -> int | None:
        if self._budget_error is not None:
            return 0
        if self.config.budget_rows is None:
            return None
        pending = sum(t.shape[0] for lid, t, _, _ in self._pending if lid == layer_id)
        return max(
            0, self.config.budget_rows - self._layer_rows.get(layer_id, 0) - pending
        )

    def _drop_rows(self, count: int) -> None:
        if count <= 0:
            return
        self.tokens_dropped += count
        if not self._warned_budget:
            self._warned_budget = True
            logger.warning(
                "Capture token budget (%d rows per layer) "
                "reached; further tokens are dropped.",
                self.config.budget_rows,
            )

    def drop_rows(self, count: int) -> None:
        with self.lock:
            self._drop_rows(count)

    def _limit_rows(self, layer_id: int, count: int) -> int:
        if self._budget_error is not None:
            return 0
        remaining = self.remaining_rows(layer_id)
        keep = count if remaining is None else min(count, remaining)
        self._drop_rows(count - keep)
        return keep

    def limit_rows(
        self,
        layer_id: int,
        count: int,
        row_bytes: int | None = None,
        *,
        device_bytes: int | None = None,
    ) -> int:
        """Apply the budget before gathering or cloning activation values."""
        budget = self.config.device_budget_bytes
        if device_bytes is not None and budget is not None:
            with self.lock:
                flush = (
                    self.device_reserved_bytes + device_bytes <= budget
                    and self.device_reserved_bytes
                    + self.pending_device_bytes
                    + device_bytes
                    > budget
                )
            if flush:
                # The caller's row plan may already contain request indices
                # that its next append has not claimed yet.
                self.flush(reclaim_requests=False)
        with self.lock:
            keep = self._limit_rows(layer_id, count)
            if row_bytes is not None and not self._within_byte_budget(keep, row_bytes):
                return 0
            if device_bytes is not None and not self._within_device_budget(
                device_bytes
            ):
                return 0
            return keep

    def append(
        self,
        layer_id: int,
        tensor: torch.Tensor,
        meta: torch.Tensor,
        layer_name: str,
    ):
        """Stage one layer's selected rows for this step (GPU side).

        ``meta`` is an int32 ``[rows, 3]`` tensor of
        (req_idx, position, token_id) labels. ``flush()`` coalesces device-to-host
        transfers across layers. A device budget can trigger an earlier flush
        before the next layer materializes its selected rows.
        """
        with self.lock:
            if (
                not isinstance(meta, torch.Tensor)
                or meta.ndim != 2
                or meta.shape[1] != 3
                or meta.dtype != torch.int32
            ):
                raise RuntimeError(
                    f"capture layer {layer_id} requires int32 [rows, 3] row labels"
                )
            if meta.shape[0] != tensor.shape[0]:
                raise RuntimeError(
                    f"capture meta rows ({meta.shape[0]}) != data rows "
                    f"({tensor.shape[0]}) for layer {layer_id}"
                )
            keep = self._limit_rows(layer_id, tensor.shape[0])
            if keep == 0:
                return
            if self.config.budget_bytes is not None and not self._within_byte_budget(
                keep, self.row_bytes(tensor)
            ):
                return
            tensor = tensor[:keep]
            meta = meta[:keep]
            if tensor.is_cuda:
                device_bytes = tensor.numel() * tensor.element_size()
                if meta.is_cuda:
                    device_bytes += meta.numel() * meta.element_size()
                if self.config.dtype is not None and tensor.dtype != self.config.dtype:
                    device_bytes += keep * (self.row_bytes(tensor) - 3 * 4)
                if not self._within_device_budget(device_bytes):
                    return
            if self.config.dtype is not None:
                tensor = tensor.to(self.config.dtype)
            # `tensor` is owned by the hook (freshly materialized), so an
            # async copy in flush() cannot race buffer reuse.
            self._pending.append(
                (
                    layer_id,
                    tensor.detach(),
                    meta.detach(),
                    layer_name,
                )
            )

    def flush(self, *, reclaim_requests: bool = True):
        """Copy through one reusable pinned page into owned pageable chunks.

        Layers share a transfer and synchronization when they fit in the page.
        Larger steps reuse that page instead of pinning their entire payload.
        """
        with self.lock:
            if not self._pending:
                return
            jobs: list[tuple[int, torch.Tensor, torch.Tensor, str]] = []
            cursor = 0

            def finish_page():
                nonlocal cursor
                if jobs:
                    torch.cuda.current_stream().synchronize()
                    for layer, values, labels, name in jobs:
                        self._store_page(layer, values, labels, name)
                    jobs.clear()
                    cursor = 0

            for layer_id, tensor, meta, layer_name in self._pending:
                value_bytes = tensor[0].numel() * tensor.element_size()
                row_bytes = value_bytes + 3 * 4
                page_rows = max(1, self.config.staging_bytes // row_bytes)
                if not tensor.is_cuda and not meta.is_cuda:
                    finish_page()
                    for start in range(0, tensor.shape[0], page_rows):
                        self._store_page(
                            layer_id,
                            tensor[start : start + page_rows],
                            meta[start : start + page_rows],
                            layer_name,
                        )
                    continue
                # A single wide row can use a synchronous pageable copy without
                # exceeding the pinned staging limit.
                if row_bytes + 8 > self.config.staging_bytes:
                    finish_page()
                    for start in range(tensor.shape[0]):
                        self._store_page(
                            layer_id,
                            tensor[start : start + 1].cpu(),
                            meta[start : start + 1].cpu(),
                            layer_name,
                        )
                    continue
                if self._staging is None:
                    self._staging = torch.empty(
                        self.config.staging_bytes,
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=True,
                    )
                start = 0
                while start < tensor.shape[0]:
                    offset = (cursor + 7) // 8 * 8
                    available = (self.config.staging_bytes - offset - 4) // row_bytes
                    if available < 1:
                        finish_page()
                        continue
                    count = min(tensor.shape[0] - start, available)
                    end_values = offset + count * value_bytes
                    meta_offset = (end_values + 3) // 4 * 4
                    end_meta = meta_offset + count * 3 * 4
                    values = (
                        self._staging[offset:end_values]
                        .view(tensor.dtype)
                        .view(count, *tensor.shape[1:])
                    )
                    labels = (
                        self._staging[meta_offset:end_meta]
                        .view(torch.int32)
                        .view(count, 3)
                    )
                    values.copy_(tensor[start : start + count], non_blocking=True)
                    labels.copy_(meta[start : start + count], non_blocking=True)
                    jobs.append((layer_id, values, labels, layer_name))
                    cursor = end_meta
                    start += count
            finish_page()
            self._pending.clear()
            if reclaim_requests:
                self._reclaim_requests()

    @staticmethod
    def _pageable_copy(tensor: torch.Tensor) -> torch.Tensor:
        result = torch.empty(
            tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=False
        )
        result.copy_(tensor)
        return result

    def _store_page(self, layer_id, values, labels, layer_name):
        requests, counts = torch.unique_consecutive(labels[:, 0], return_counts=True)
        start = 0
        for request_index, count in zip(requests.tolist(), counts.tolist()):
            tensor = self._pageable_copy(values[start : start + count])
            meta = self._pageable_copy(labels[start : start + count])
            chunk_id = self._next_chunk
            self._next_chunk += 1
            self.chunks.setdefault(layer_id, {})[chunk_id] = _StoredChunk(
                tensor, meta, request_index
            )
            self._request_chunks.setdefault(request_index, {}).setdefault(
                layer_id, set()
            ).add(chunk_id)
            self._unused_request_indices.discard(request_index)
            start += count
        self.layer_names[layer_id] = layer_name
        self._layer_rows[layer_id] = self._layer_rows.get(layer_id, 0) + values.shape[0]
        self._stored_bytes += (
            values.numel() * values.element_size()
            + labels.numel() * labels.element_size()
        )

    def serialize(
        self,
        layers: list[int] | None = None,
        req_ids: list[str] | None = None,
        clear_selected: bool = False,
        max_rows: int | None = None,
        row_offset: int = 0,
    ) -> dict[int, dict[str, Any]]:
        """Pack a bounded page per layer for RPC transmission.

        Tensors ship as raw bytes of their stored dtype (bf16 rides as
        int16 bytes and is reinterpreted client-side) — no float32
        upcast, so the wire volume equals the stored volume. Passing
        ``layers`` serializes a subset, letting clients fetch layer by
        layer instead of one monolithic message.

        Each layer entry carries a ``meta`` sub-dict labelling its rows
        (``req_table`` request-id strings + int32 ``req_idx`` /
        ``positions`` / ``token_ids`` columns).

        ``req_ids`` (client-visible ids) restricts the payload to rows
        of those requests; with ``clear_selected`` the emitted rows are
        also removed from the store, so clients can drain request by
        request with bounded peak message size.

        ``max_rows`` caps each layer after request filtering. ``row_offset``
        skips rows in that filtered order and requires a non-clearing fetch.
        """
        if max_rows is not None and (type(max_rows) is not int or max_rows < 1):
            raise ValueError("max_rows must be a positive integer")
        if type(row_offset) is not int or row_offset < 0:
            raise ValueError("row_offset must be a nonnegative integer")
        if clear_selected and row_offset:
            raise ValueError("row_offset requires clear=False")
        with self.lock:
            if self._budget_error is not None:
                raise RuntimeError(self._budget_error)
            failed = {
                rid: reason
                for rid, reason in self._failed_requests.items()
                if req_ids is None
                or any(match_capture_request_id(rid, ext) for ext in req_ids)
            }
            if failed:
                raise RuntimeError(
                    f"capture contains failed request(s) {list(failed.items())[:5]}; "
                    "their rows do not represent valid capture. Fetch with "
                    "req_ids excluding them, or clear the stream."
                )
            elided = self.elided_reqs
            if req_ids is not None and elided:
                elided = {
                    rid
                    for rid in elided
                    if any(match_capture_request_id(rid, ext) for ext in req_ids)
                }
            if elided:
                raise RuntimeError(
                    "prefix-cache hits skipped recomputation of the prompt "
                    f"head for request(s) {sorted(elided)[:5]}; their early "
                    "rows were never computed, so this capture is "
                    "incomplete. Submit capture requests with "
                    "SamplingParams(skip_reading_prefix_cache=True), "
                    "fetch with req_ids excluding them, or "
                    "clear the stream."
                )
            table_match: set[int] | None = None
            if req_ids is not None:
                table_match = {
                    i
                    for req_id in req_ids
                    for i in self._request_aliases.get(req_id, ())
                }
            result: dict[int, dict[str, Any]] = {}
            wanted = (
                sorted(self.chunks)
                if layers is None
                else [lid for lid in layers if lid in self.chunks]
            )
            for layer_id in wanted:
                layer_name = self.layer_names[layer_id]
                chunks = self.chunks[layer_id]
                if table_match is not None:
                    chunk_ids = sorted(
                        chunk
                        for index in table_match
                        for chunk in self._request_chunks.get(index, {}).get(
                            layer_id, ()
                        )
                    )
                else:
                    chunk_ids = list(chunks)
                if not chunk_ids:
                    continue
                selected = []
                remaining = max_rows
                skip = row_offset
                for index in chunk_ids:
                    chunk = chunks[index]
                    count = chunk.tensor.shape[0]
                    if skip >= count:
                        skip -= count
                        continue
                    end = count if remaining is None else min(count, skip + remaining)
                    selected.append((index, skip, end))
                    if remaining is not None:
                        remaining -= end - skip
                        if remaining == 0:
                            break
                    skip = 0
                if not selected:
                    continue
                if len(selected) == 1:
                    index, start, end = selected[0]
                    tensor = chunks[index].tensor[start:end]
                    meta = chunks[index].meta[start:end]
                else:
                    tensor = torch.cat(
                        [
                            chunks[index].tensor[start:end]
                            for index, start, end in selected
                        ]
                    )
                    meta = torch.cat(
                        [
                            chunks[index].meta[start:end]
                            for index, start, end in selected
                        ]
                    )
                result[layer_id] = serialize_capture_layer(
                    tensor, meta, self.req_table, layer_name
                )
                if clear_selected:
                    for index, _, end in selected:
                        chunk = chunks[index]
                        if end == chunk.tensor.shape[0]:
                            self._remove_chunk(layer_id, index)
                        else:
                            self._stored_bytes -= end * self.row_bytes(chunk.tensor)
                            self._layer_rows[layer_id] -= end
                            chunk.tensor = self._pageable_copy(chunk.tensor[end:])
                            chunk.meta = self._pageable_copy(chunk.meta[end:])
            if clear_selected:
                self._reclaim_requests()
            return result

    def _remove_chunk(self, layer_id: int, chunk_id: int) -> None:
        chunk = self.chunks[layer_id].pop(chunk_id)
        self._stored_bytes -= (
            chunk.tensor.numel() * chunk.tensor.element_size()
            + chunk.meta.numel() * chunk.meta.element_size()
        )
        self._layer_rows[layer_id] -= chunk.tensor.shape[0]
        if not self.chunks[layer_id]:
            self.chunks.pop(layer_id)
            self._layer_rows.pop(layer_id)
            self.layer_names.pop(layer_id)
        layers = self._request_chunks[chunk.request_index]
        layers[layer_id].remove(chunk_id)
        if not layers[layer_id]:
            layers.pop(layer_id)
        if not layers:
            self._request_chunks.pop(chunk.request_index)
            self._unused_request_indices.add(chunk.request_index)

    def _reclaim_requests(self) -> None:
        """Recycle unused label slots without rewriting unrelated row labels."""
        if self._pending or not self._unused_request_indices:
            return
        for index in self._unused_request_indices:
            req_id = self.req_table.pop(index)
            self._req_index.pop(req_id)
            for alias in self._aliases(req_id):
                indices = self._request_aliases[alias]
                indices.remove(index)
                if not indices:
                    self._request_aliases.pop(alias)
            self._free_request_indices.append(index)
        self._unused_request_indices.clear()
        self._row_plans.clear()
        self._row_plan_geometry = None
        self._row_plan_request_selects = None

    def clear(self) -> None:
        """Reset captured rows and counters, retaining active request history."""
        with self.lock:
            self.chunks.clear()
            self._request_chunks.clear()
            self._next_chunk = 0
            self.layer_names.clear()
            self.req_table.clear()
            self._req_index.clear()
            self._request_aliases.clear()
            self._free_request_indices.clear()
            self._next_request_index = 0
            self._unused_request_indices.clear()
            self.elided_reqs.clear()
            self._failed_requests.clear()
            self._layer_rows.clear()
            self._stored_bytes = 0
            self._budget_error = None
            self.tokens_dropped = 0
            self._warned_budget = False
            self._pending.clear()
            self._row_plans.clear()
            self._row_plan_geometry = None
            self._row_plan_request_selects = None

    def drop_layers(self, layers: list[int]) -> None:
        with self.lock:
            for layer_id in layers:
                for chunk_id in list(self.chunks.get(layer_id, ())):
                    self._remove_chunk(layer_id, chunk_id)
            dropped = set(layers)
            self._pending = [
                entry for entry in self._pending if entry[0] not in dropped
            ]
            self._reclaim_requests()
