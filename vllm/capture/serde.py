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
