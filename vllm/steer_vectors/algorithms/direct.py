# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch

from .factory import register_algorithm
from .loading import find_reft_checkpoint, read_gguf_directions
from .template import AlgorithmTemplate


@register_algorithm("direct")
class DirectAlgorithm(AlgorithmTemplate):
    """Direct addition: h' = h + vector.

    Payload: a single direction tensor per layer. Loads from GGUF and
    .pt files or a ReFT bias-intervention directory.
    """

    def _transform(
        self, hidden_state: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        transformed = hidden_state + params
        if self.normalize:
            return self._renormalize(hidden_state, transformed)
        return transformed

    @classmethod
    def load_from_path(
        cls,
        path: str,
        device: str,
        *,
        config,
        target_layers: list[int] | None = None,
        **kwargs,
    ) -> dict:
        if os.path.isdir(path):
            return cls._load_from_reft_dir(path, device, config, target_layers)
        if os.path.splitext(path)[1].lower() == ".pt":
            return cls._load_from_pt(path, device, config, target_layers)
        return {
            "layer_payloads": read_gguf_directions(path, device, config.adapter_dtype)
        }

    @classmethod
    def _load_from_pt(
        cls, path: str, device: str, config, target_layers: list[int] | None
    ) -> dict:
        """Load a single-layer direction vector from a .pt file.

        The file holds one tensor; it is assigned to the first (and
        only) entry of target_layers.
        """
        if not target_layers:
            raise ValueError("Loading .pt files requires non-empty target_layers")
        target_layer = target_layers[0]

        # weights_only=False: vectors may be saved as numpy arrays.
        vector = torch.load(path, map_location=device, weights_only=False)
        if isinstance(vector, np.ndarray):
            vector = torch.tensor(vector, device=device)
        elif not isinstance(vector, torch.Tensor):
            raise ValueError(
                f"PT file does not contain a tensor or numpy array: {type(vector)}"
            )
        vector = vector.to(device).to(config.adapter_dtype)
        return {"layer_payloads": {target_layer: vector}}

    @classmethod
    def _load_from_reft_dir(
        cls, path: str, device: str, config, target_layers: list[int] | None
    ) -> dict:
        """Load a direction vector from a ReFT directory (BiasIntervention)."""
        bin_file_path, layer_idx = find_reft_checkpoint(path, target_layers)
        state_dict = torch.load(bin_file_path, map_location=device)

        if len(state_dict) == 1:
            vector = next(iter(state_dict.values()))
        elif "source_representation" in state_dict:
            vector = state_dict["source_representation"]
        elif "bias" in state_dict:
            vector = state_dict["bias"]
        elif "weight" in state_dict:
            vector = state_dict["weight"]
        else:
            raise ValueError(
                "Could not determine the correct tensor from .bin file with multiple "
                f"tensors. Keys found: {list(state_dict.keys())}"
            )

        if not isinstance(vector, torch.Tensor):
            raise ValueError(f"Loaded payload is not a tensor. Type: {type(vector)}")

        vector = vector.to(device).to(config.adapter_dtype)
        return {"layer_payloads": {layer_idx: vector}}
