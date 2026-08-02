# SPDX-License-Identifier: Apache-2.0
"""Shared file-format helpers for algorithm loaders."""

import glob
import json
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


def find_reft_checkpoint(
    path: str, target_layers: list[int] | None = None
) -> tuple[str, int]:
    """Locate the weight file and layer index of a ReFT checkpoint directory.

    Expects exactly one ``*.bin`` file and exactly one config file
    (``reft_config.json`` or ``config.json``). The layer index comes from
    the config's ``representations`` entry, with the
    ``intkey_layer_<n>_`` filename convention as fallback.

    Returns:
        (bin_file_path, layer_idx)

    Raises:
        ValueError: If the directory layout is ambiguous, the layer index
            cannot be determined, or it is not in ``target_layers``.
    """
    if not os.path.isdir(path):
        raise ValueError(f"ReFT checkpoint path must be a directory. Got: {path}")

    bin_files = glob.glob(os.path.join(path, "*.bin"))
    if not bin_files:
        raise ValueError(f"No .bin files found in directory: {path}")
    if len(bin_files) > 1:
        raise ValueError(
            f"Multiple .bin files found in directory {path}. Please ensure only "
            "one exists."
        )
    bin_file_path = bin_files[0]

    config_files = [
        os.path.join(path, f)
        for f in ["reft_config.json", "config.json"]
        if os.path.exists(os.path.join(path, f))
    ]
    if not config_files:
        raise ValueError(
            "No config file (reft_config.json or config.json) found in "
            f"directory: {path}"
        )
    if len(config_files) > 1:
        raise ValueError(
            f"Multiple config files found in directory {path}. Please ensure only "
            "one exists."
        )
    config_file_path = config_files[0]

    with open(config_file_path) as f:
        config_data = json.load(f)

    layer_idx = None
    representations = config_data.get("representations") or []
    if representations:
        first_repr = representations[0]
        if isinstance(first_repr, dict):
            layer_idx = first_repr.get("layer")
        # Older list-based representation format.
        elif isinstance(first_repr, list) and len(first_repr) > 0:
            layer_idx = first_repr[0]

    if layer_idx is None:
        bin_filename = os.path.basename(bin_file_path)
        if "intkey_layer_" in bin_filename:
            layer_str = bin_filename.split("intkey_layer_")[1].split("_")[0]
            if layer_str.isdigit():
                layer_idx = int(layer_str)

    if layer_idx is None:
        raise ValueError(
            f"Could not extract layer info from config {config_file_path} or "
            f"filename {os.path.basename(bin_file_path)}"
        )

    if target_layers and layer_idx not in target_layers:
        raise ValueError(
            f"Layer mismatch: config specifies layer {layer_idx}, but "
            f"target_layers is {target_layers}."
        )

    return bin_file_path, layer_idx
