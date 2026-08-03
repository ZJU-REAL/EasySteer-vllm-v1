# SPDX-License-Identifier: Apache-2.0
"""Shared file-format helpers for algorithm loaders."""

import os

import numpy as np
import torch


def read_gguf_directions(
    path: str, device: str, dtype: torch.dtype
) -> dict[int, torch.Tensor]:
    """Read per-layer direction vectors from a steer-vector GGUF file.

    Returns a mapping of layer index -> direction tensor, from tensors
    named ``direction.<layer>``.
    """
    import gguf

    reader = gguf.GGUFReader(path)
    directions: dict[int, torch.Tensor] = {}
    for tensor in reader.tensors:
        if not tensor.name.startswith("direction."):
            continue
        try:
            layer = int(tensor.name.split(".")[1])
        except (ValueError, IndexError) as e:
            raise ValueError(
                f".gguf file has invalid direction field name: {tensor.name}"
            ) from e
        np_copy = np.array(tensor.data, copy=True)
        directions[layer] = torch.from_numpy(np_copy).to(device).to(dtype)
    return directions


def require_extension(path: str, ext: str, algo_name: str) -> None:
    """Uniform format guard for algorithms with one accepted extension."""
    actual = os.path.splitext(path)[1].lower()
    if actual != ext:
        raise ValueError(
            f"{algo_name} only loads {ext} files (its own export "
            f"format), got: {actual or path!r}. For other formats, load "
            "the file yourself and pass VectorSpec(data=...)."
        )
