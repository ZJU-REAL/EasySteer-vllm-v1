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
