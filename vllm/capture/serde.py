# SPDX-License-Identifier: Apache-2.0
"""Deserialization helpers for captured state fetched over RPC."""

from typing import Any

import numpy as np
import torch

_NUMPY_RAW = {
    "torch.float32": np.float32,
    "torch.float16": np.float16,
    "torch.bfloat16": np.int16,  # raw bf16 bytes ride as int16
    "torch.float64": np.float64,
    "torch.int32": np.int32,
    "torch.int64": np.int64,
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
) -> tuple[dict[int, torch.Tensor], dict[int, CaptureMeta] | None]:
    """Rebuild per-layer tensors AND their row labels.

    Tensors arrive as raw bytes of their stored dtype (bf16 rides as
    int16 bytes and is reinterpreted here) — see StreamStore.serialize.

    Returns ``(tensors, meta)`` where ``meta[layer]`` labels each row of
    ``tensors[layer]`` with (request id, absolute position, token id).
    ``meta`` is None when the engine captured rows without batch
    geometry (it warns engine-side).
    """
    tensors: dict[int, torch.Tensor] = {}
    meta: dict[int, CaptureMeta] = {}
    labelled = True
    for layer_id, info in serialized_data.items():
        encoding = info.get("encoding")
        if encoding != "raw":
            raise ValueError(
                f"unknown capture wire encoding {encoding!r} for layer "
                f"{layer_id}; engine and client versions do not match"
            )
        wire_dtype = info["dtype"]
        if wire_dtype not in _NUMPY_RAW:
            raise ValueError(
                f"unknown capture wire dtype {wire_dtype!r} for layer "
                f"{layer_id}"
            )
        shape = tuple(info["shape"])
        array = np.frombuffer(info["data"], dtype=_NUMPY_RAW[wire_dtype])
        tensor = torch.from_numpy(array.reshape(shape).copy())
        if wire_dtype == "torch.bfloat16":
            tensor = tensor.view(torch.bfloat16)
        tensors[layer_id] = tensor

        m = info.get("meta")
        if m is None:
            labelled = False
            continue
        req_idx = np.frombuffer(m["req_idx"], dtype=np.int32)
        table = m["req_table"]
        meta[layer_id] = CaptureMeta(
            req_ids=[table[i] for i in req_idx],
            positions=torch.from_numpy(
                np.frombuffer(m["positions"], dtype=np.int32).copy()
            ),
            token_ids=torch.from_numpy(
                np.frombuffer(m["token_ids"], dtype=np.int32).copy()
            ),
        )
    if not labelled or not meta:
        return tensors, None
    return tensors, meta
