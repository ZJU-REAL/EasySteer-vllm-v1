# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw tensor and mandatory row-label encoding for capture RPCs."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

_RAW_DTYPES = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.bfloat16: np.int16,  # raw bf16 bytes ride as int16
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
}
_WIRE_DTYPES = {str(dtype): dtype for dtype in _RAW_DTYPES}


def resolve_storage_dtype(name: str) -> torch.dtype:
    """Validate a requested storage dtype against the RPC wire format."""
    dtype = getattr(torch, name, None) if isinstance(name, str) else None
    if not isinstance(dtype, torch.dtype) or dtype not in _RAW_DTYPES:
        supported = ", ".join(
            str(dtype).removeprefix("torch.") for dtype in _RAW_DTYPES
        )
        raise ValueError(f"unsupported capture storage dtype {name!r}; use {supported}")
    return dtype


def serialize_capture_layer(
    tensor: torch.Tensor,
    meta: torch.Tensor,
    req_table: Mapping[int, str],
    layer_name: str,
) -> dict[str, Any]:
    """Encode CPU rows and their int32 request/position/token labels."""
    if tensor.dtype not in _RAW_DTYPES:
        raise ValueError(f"unsupported capture wire dtype {str(tensor.dtype)!r}")
    wire = tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor
    request_indices, row_indices = torch.unique(
        meta[:, 0], sorted=True, return_inverse=True
    )
    return {
        "data": wire.numpy().tobytes(),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "encoding": "raw",
        "layer_name": layer_name,
        "meta": {
            "req_table": [req_table[index] for index in request_indices.tolist()],
            "req_idx": row_indices.to(torch.int32).numpy().tobytes(),
            "positions": meta[:, 1].numpy().tobytes(),
            "token_ids": meta[:, 2].numpy().tobytes(),
        },
    }


def match_capture_request_id(label_req_id: str, request_id: str) -> bool:
    """Whether a captured row's label belongs to a client request.

    Rows are labelled with the engine-internal request id, which is the
    client-visible id plus an 8-hex-char uniqueness suffix
    (``{request_id}-{8 hex}``, see InputProcessor) — or identical to it
    when suffixing is disabled.
    """
    if label_req_id == request_id:
        return True
    return (
        label_req_id.startswith(request_id + "-")
        and len(label_req_id) == len(request_id) + 9
        and all(c in "0123456789abcdef" for c in label_req_id[-8:])
    )


class CaptureMeta:
    """Row labels for one captured layer.

    Attributes:
        req_ids: engine-internal request id string per row (the
            client-visible id plus an 8-hex uniqueness suffix; match
            with :func:`match_capture_request_id`).
        positions: absolute sequence position per row (int32; -1 for
            synthesized rows such as 'mean' reductions).
        token_ids: input token id per row (int32; -1 for synthesized
            rows).
    """

    def __init__(
        self,
        req_ids: list[str],
        positions: torch.Tensor,
        token_ids: torch.Tensor,
    ):
        self.req_ids = req_ids
        self.positions = positions
        self.token_ids = token_ids

    def __len__(self) -> int:
        return len(self.req_ids)


def deserialize_captured(
    serialized_data: dict[int, dict[str, Any]],
) -> tuple[dict[int, torch.Tensor], dict[int, CaptureMeta]]:
    """Rebuild per-layer tensors AND their row labels.

    Tensors arrive as raw bytes of their stored dtype (bf16 rides as
    int16 bytes and is reinterpreted here) — see StreamStore.serialize.

    Returns ``(tensors, meta)`` where ``meta[layer]`` labels each row of
    ``tensors[layer]`` with (request id, absolute position, token id).
    Every layer must carry one label per row. Incomplete or older unlabeled
    payloads raise instead of silently losing sample alignment.
    """
    tensors: dict[int, torch.Tensor] = {}
    meta: dict[int, CaptureMeta] = {}
    for layer_id, info in serialized_data.items():
        encoding = info.get("encoding")
        if encoding != "raw":
            raise ValueError(
                f"unknown capture wire encoding {encoding!r} for layer "
                f"{layer_id}; engine and client versions do not match"
            )
        wire_dtype = info["dtype"]
        if wire_dtype not in _WIRE_DTYPES:
            raise ValueError(
                f"unknown capture wire dtype {wire_dtype!r} for layer {layer_id}"
            )
        shape = tuple(info["shape"])
        array = np.frombuffer(info["data"], dtype=_RAW_DTYPES[_WIRE_DTYPES[wire_dtype]])
        tensor = torch.from_numpy(array.reshape(shape).copy())
        if wire_dtype == "torch.bfloat16":
            tensor = tensor.view(torch.bfloat16)
        tensors[layer_id] = tensor

        m = info.get("meta")
        if m is None:
            raise ValueError(
                f"capture layer {layer_id} is missing mandatory row labels"
            )
        req_idx = np.frombuffer(m["req_idx"], dtype=np.int32)
        table = m["req_table"]
        positions = np.frombuffer(m["positions"], dtype=np.int32)
        token_ids = np.frombuffer(m["token_ids"], dtype=np.int32)
        rows = tensor.shape[0]
        if len(req_idx) != rows or len(positions) != rows or len(token_ids) != rows:
            raise ValueError(f"capture layer {layer_id} has inconsistent row labels")
        if (req_idx < 0).any() or (req_idx >= len(table)).any():
            raise ValueError(f"capture layer {layer_id} has invalid request indices")
        meta[layer_id] = CaptureMeta(
            req_ids=[table[i] for i in req_idx],
            positions=torch.from_numpy(positions.copy()),
            token_ids=torch.from_numpy(token_ids.copy()),
        )
    return tensors, meta


def validate_capture_topology(statuses: list[dict[str, Any]]) -> int:
    """Require one complete ordinary TP group before changing its streams."""
    if not statuses:
        raise ValueError("Capture requires at least one worker")
    if len(statuses) == 1 and "topology" not in statuses[0]:
        # Preserve compatibility with the original single-worker RPC format.
        return 1
    ranks = set()
    for status in statuses:
        topology = status.get("topology")
        if not isinstance(topology, dict):
            raise ValueError("Multiworker capture requires explicit worker topology")
        rank, size = topology.get("tp_rank"), topology.get("tp_size")
        if (
            type(rank) is not int
            or type(size) is not int
            or size != len(statuses)
            or not 0 <= rank < size
            or rank in ranks
        ):
            raise ValueError(
                "Capture worker topology has missing or duplicate TP ranks"
            )
        ranks.add(rank)
        for name in ("pp_size", "dp_size", "pcp_size", "dcp_size"):
            if topology.get(name) != 1:
                raise ValueError(f"Capture requires {name}=1")
        for name in ("sequence_parallel", "sequence_parallel_moe", "expert_parallel"):
            if topology.get(name) is not False:
                raise ValueError(f"Capture does not support {name}")
    return len(statuses)


def assemble_captured(
    worker_results: list[dict[int, dict[str, Any]]],
    *,
    tp_size: int,
) -> tuple[dict[int, torch.Tensor], dict[int, CaptureMeta], dict[int, dict[str, int]]]:
    """Decode owner exports and join labelled attention feature shards.

    Worker reply order does not determine shard order. Every distributed layer
    declares its TP rank, feature offset, global width and representation kind.
    Replicated values must come from exactly one owner; feature shards must cover
    the full TP group and agree on row identities before concatenation.
    """
    if tp_size < 1 or len(worker_results) != tp_size:
        raise ValueError("Capture fetch returned an incomplete worker group")
    if tp_size == 1:
        raw = worker_results[0]
        tensors, meta = deserialize_captured(raw)
        layouts = {lid: info["layout"] for lid, info in raw.items() if "layout" in info}
        return tensors, meta, layouts

    by_layer: dict[int, list[tuple]] = {}
    worker_ranks = set()
    for raw in worker_results:
        tensors, meta = deserialize_captured(raw)
        reply_rank = None
        for layer, info in raw.items():
            shard = info.get("shard")
            if not isinstance(shard, dict):
                raise ValueError(f"Capture layer {layer} is missing shard metadata")
            rank = shard.get("tp_rank")
            if (
                type(rank) is not int
                or not 0 <= rank < tp_size
                or shard.get("tp_size") != tp_size
                or (reply_rank is not None and rank != reply_rank)
            ):
                raise ValueError(f"Capture layer {layer} has inconsistent TP topology")
            reply_rank = rank
            by_layer.setdefault(layer, []).append((info, tensors[layer], meta[layer]))
        if reply_rank is not None:
            if reply_rank in worker_ranks:
                raise ValueError("Capture fetch returned duplicate TP ranks")
            worker_ranks.add(reply_rank)

    tensors, metadata, layouts = {}, {}, {}
    for layer, parts in by_layer.items():
        first_info, first_tensor, first_meta = parts[0]
        first_shard = first_info["shard"]
        kind, width = first_shard.get("kind"), first_shard.get("global_width")
        if kind not in ("replicated", "feature_shard") or type(width) is not int:
            raise ValueError(f"Capture layer {layer} has an unsupported shard layout")
        if width <= 0:
            raise ValueError(f"Capture layer {layer} has an invalid global width")
        ranks = {info["shard"]["tp_rank"] for info, _, _ in parts}
        expected_ranks = {0} if kind == "replicated" else set(range(tp_size))
        if ranks != expected_ranks or len(parts) != len(expected_ranks):
            raise ValueError(f"Capture layer {layer} has missing or duplicate shards")

        head_size = None
        for info, tensor, labels in parts:
            shard = info["shard"]
            start = shard.get("feature_start")
            if (
                shard.get("kind") != kind
                or shard.get("global_width") != width
                or type(start) is not int
                or start < 0
                or tensor.ndim != 2
                or tensor.dtype != first_tensor.dtype
                or info.get("layer_name") != first_info.get("layer_name")
            ):
                raise ValueError(f"Capture layer {layer} has inconsistent shard layout")
            if (
                labels.req_ids != first_meta.req_ids
                or not torch.equal(labels.positions, first_meta.positions)
                or not torch.equal(labels.token_ids, first_meta.token_ids)
            ):
                raise ValueError(
                    f"Capture layer {layer} shards have different row labels"
                )
            layout = info.get("layout")
            if layout is not None and (
                not isinstance(layout, dict) or layout.get("width") != tensor.shape[1]
            ):
                raise ValueError(
                    f"Capture layer {layer} has inconsistent output layout"
                )
            if kind == "feature_shard" or (
                layout is not None and ("head_size" in layout or "num_heads" in layout)
            ):
                if not isinstance(layout, dict):
                    raise ValueError(
                        f"Capture layer {layer} is missing attention layout"
                    )
                size, heads = layout.get("head_size"), layout.get("num_heads")
                if (
                    type(size) is not int
                    or size <= 0
                    or type(heads) is not int
                    or heads <= 0
                    or layout.get("width") != tensor.shape[1]
                    or heads * size != tensor.shape[1]
                    or start % size != 0
                    or width % size != 0
                    or (head_size is not None and size != head_size)
                ):
                    raise ValueError(
                        f"Capture layer {layer} has inconsistent head layout"
                    )
                head_size = size

        ordered = sorted(parts, key=lambda part: part[0]["shard"]["feature_start"])
        offset = 0
        for info, tensor, _ in ordered:
            if info["shard"]["feature_start"] != offset:
                raise ValueError(
                    f"Capture layer {layer} has overlapping or missing features"
                )
            offset += tensor.shape[1]
        if offset != width:
            raise ValueError(
                f"Capture layer {layer} shards do not cover its global width"
            )
        tensors[layer] = (
            first_tensor
            if kind == "replicated"
            else torch.cat([tensor for _, tensor, _ in ordered], dim=-1)
        )
        metadata[layer] = first_meta
        if first_info.get("layout") is not None:
            layouts[layer] = {"width": width}
        if head_size is not None:
            layouts[layer] = {
                "width": width,
                "num_heads": width // head_size,
                "head_size": head_size,
            }
    return tensors, metadata, layouts
