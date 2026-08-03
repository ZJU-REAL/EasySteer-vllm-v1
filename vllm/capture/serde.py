# SPDX-License-Identifier: Apache-2.0
"""Deserialization helpers for captured state fetched over RPC."""

from typing import Any

import numpy as np
import torch

_DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float64": torch.float64,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


_NUMPY_RAW = {
    "torch.float32": np.float32,
    "torch.float16": np.float16,
    "torch.bfloat16": np.int16,  # raw bf16 bytes ride as int16
    "torch.float64": np.float64,
    "torch.int32": np.int32,
    "torch.int64": np.int64,
}


def deserialize_hidden_states(
    serialized_data: dict[int, dict[str, Any]],
) -> dict[int, torch.Tensor]:
    """Rebuild per-layer tensors from the capture RPC wire format.

    Handles both encodings: 'raw' ships the stored dtype's bytes
    unchanged (bf16 reinterpreted from int16); the legacy format ships
    float32 bytes and is converted back to the original dtype.

    Returns:
        layer_id -> tensor restored to its original dtype.
    """
    tensors = {}
    for layer_id, info in serialized_data.items():
        shape = tuple(info["shape"])
        dtype = _DTYPE_MAP.get(info["dtype"], torch.float32)
        if info.get("encoding") == "raw":
            np_dtype = _NUMPY_RAW[info["dtype"]]
            array = np.frombuffer(info["data"], dtype=np_dtype).reshape(shape)
            tensor = torch.from_numpy(array.copy())
            if info["dtype"] == "torch.bfloat16":
                tensor = tensor.view(torch.bfloat16)
        else:
            array = np.frombuffer(info["data"], dtype=np.float32).reshape(shape)
            tensor = torch.from_numpy(array.copy())
            if dtype != tensor.dtype:
                tensor = tensor.to(dtype)
        tensors[layer_id] = tensor
    return tensors


# Router-logit captures use the same wire format.
deserialize_moe_router_logits = deserialize_hidden_states


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

    Returns ``(tensors, meta)`` where ``meta[layer]`` labels each row of
    ``tensors[layer]`` with (request id, absolute position, token id).
    ``meta`` is None when the engine captured rows without batch
    geometry (it warns engine-side) or when talking to an engine
    predating labelled rows.
    """
    tensors = deserialize_hidden_states(serialized_data)
    meta: dict[int, CaptureMeta] = {}
    for layer_id, info in serialized_data.items():
        m = info.get("meta")
        if m is None:
            return tensors, None
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
    return tensors, (meta if meta else None)


def print_hidden_states_summary(
    hidden_states: dict[int, torch.Tensor],
) -> None:
    """Print a per-layer summary of captured tensors."""
    print(f"Captured {len(hidden_states)} layers:")
    for layer_id in sorted(hidden_states):
        tensor = hidden_states[layer_id]
        print(
            f"  Layer {layer_id:2d}: shape {tuple(tensor.shape)}, "
            f"dtype {tensor.dtype}, device {tensor.device}"
        )


print_moe_router_logits_summary = print_hidden_states_summary
