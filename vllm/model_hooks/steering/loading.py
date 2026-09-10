# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve supported sources into content-addressed CPU payload snapshots.

File IO belongs to admission, before requests cross process boundaries.
Workers only materialize the resulting wire content. Third-party checkpoint
schemas remain explicit client-side adapters in ``easysteer.vectors``.
"""

import copy
import glob
import json
import os
from collections import OrderedDict
from threading import RLock
from typing import Any

import numpy as np

from vllm.model_hooks.steering.capabilities import (
    ALGORITHM_CAPABILITIES,
    ROUTER_OVERRIDE_PARAMS,
)
from vllm.model_hooks.steering.input_validation import (
    validate_params,
    validate_vector_input,
)
from vllm.model_hooks.steering.payloads import (
    ConceptPair,
    DirectionVector,
    Payload,
    RouterConfig,
    from_wire,
)

_SOURCE_FORMATS = {
    "direction": "gguf",
    "concept_pair": "concept_pair",
    "router": "moe_router",
}
_FILE_CACHE: OrderedDict[tuple, tuple[tuple, dict]] = OrderedDict()
_CACHE_LOCK = RLock()
_CACHE_CAPACITY = 64


def resolve_source_path(path: str) -> str:
    """Resolve EasySteer's local path or ``org/repo/file`` source syntax."""
    if "|" in path:
        raise ValueError("Use a plain path and the separate algorithm argument")
    if os.path.exists(path):
        return os.path.abspath(path)
    parts = path.split("/")
    if os.path.isabs(path) or path.startswith(".") or len(parts) < 3:
        raise FileNotFoundError(f"Steering source not found: {path}")
    from vllm.transformers_utils.repo_utils import hf_api

    return hf_api().hf_hub_download(
        repo_id="/".join(parts[:2]), filename="/".join(parts[2:]), revision="main"
    )


def _source_files(path: str, format: str) -> tuple[str, ...]:
    if format in ("gguf", "moe_router"):
        extension = ".gguf" if format == "gguf" else ".json"
        if os.path.splitext(path)[1].lower() != extension:
            raise ValueError(f"{format} only loads {extension} files, got: {path!r}")
        return (path,)
    if format != "concept_pair":
        raise ValueError(f"Unsupported engine payload format: {format!r}")
    if not os.path.isdir(path):
        raise ValueError(f"concept_pair requires a directory path, got: {path}")
    candidates = sorted(glob.glob(os.path.join(path, "*.gguf")))
    roles = []
    for role in ("h1", "h2"):
        matches = [
            f
            for f in candidates
            if os.path.basename(f).lower() == f"{role}.gguf"
            or f"_{role}" in os.path.basename(f).lower()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"concept_pair requires exactly one named {role} file; "
                f"found {len(matches)} in {path}. Use h1.gguf/h2.gguf or "
                "*_h1*/*_h2*; h1 is the component replaced with h2."
            )
        roles.append(matches[0])
    if roles[0] == roles[1]:
        raise ValueError("concept_pair h1 and h2 must be distinct files")
    return tuple(roles)


def _file_versions(paths: tuple[str, ...]) -> tuple:
    versions = []
    for path in paths:
        st = os.stat(path)
        versions.append(
            (path, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        )
    return tuple(versions)


def _read_gguf(path: str) -> DirectionVector:
    import gguf

    directions = {}
    for tensor in gguf.GGUFReader(path).tensors:
        if not tensor.name.startswith("direction."):
            continue
        try:
            layer = int(tensor.name.removeprefix("direction."))
        except ValueError:
            raise ValueError(f"invalid GGUF direction field: {tensor.name}") from None
        if layer < 0 or layer in directions:
            raise ValueError(
                f"invalid or duplicate GGUF direction field: {tensor.name}"
            )
        directions[layer] = np.array(tensor.data, copy=True)
    return DirectionVector(directions)


def _router_payload(layers: dict, params: dict) -> RouterConfig:
    """Apply overrides to canonical configs, identically for files and data."""
    payload = RouterConfig(layers)
    if not params:
        return payload
    return RouterConfig(
        {layer: {**config, **params} for layer, config in payload.layers.items()}
    )


def _read_payload(files: tuple[str, ...], format: str, params: dict) -> Payload:
    if format == "gguf":
        return _read_gguf(files[0])
    if format == "concept_pair":
        return ConceptPair(_read_gguf(files[0]), _read_gguf(files[1]))
    with open(files[0]) as stream:
        config = json.load(stream)
    return _router_payload(config.get("layer_configs", {}), params)


def _load_file_wire(source: str, format: str, params: dict) -> dict:
    path = resolve_source_path(source)
    key = (path, format, json.dumps(params, sort_keys=True, separators=(",", ":")))
    for _ in range(3):
        files = _source_files(path, format)
        version = _file_versions(files)
        with _CACHE_LOCK:
            cached = _FILE_CACHE.get(key)
            if cached is not None and cached[0] == version:
                _FILE_CACHE.move_to_end(key)
                return copy.deepcopy(cached[1])
        wire = _read_payload(files, format, params).to_wire()
        if files != _source_files(path, format) or version != _file_versions(files):
            continue
        with _CACHE_LOCK:
            _FILE_CACHE[key] = (version, wire)
            _FILE_CACHE.move_to_end(key)
            while len(_FILE_CACHE) > _CACHE_CAPACITY:
                _FILE_CACHE.popitem(last=False)
        return copy.deepcopy(wire)
    raise RuntimeError(f"Steering source changed repeatedly while being read: {path}")


def load_file_payload(path: str, *, format: str, **params) -> Payload:
    """Read one explicitly selected EasySteer format into its canonical payload."""
    allowed = ROUTER_OVERRIDE_PARAMS if format == "moe_router" else ()
    params = validate_params(params, allowed, f"format {format!r}")
    return from_wire(_load_file_wire(path, format, params))


def resolve_vector_payload(
    source: str | None,
    data: Payload | dict | None,
    algorithm: str,
    layers: list[int] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze a vector's file, memory or inline router input at admission.

    The returned dictionary owns all mutable metadata; tensor data is immutable
    bytes. Unchanged files reuse a validated content snapshot without rehashing.
    """
    params = validate_vector_input(source, data, algorithm, layers, params)
    if data is not None:
        payload = from_wire(data) if isinstance(data, dict) else data
        if isinstance(payload, RouterConfig) and params:
            payload = _router_payload(payload.layers, params)
        return payload.to_wire()
    if source is not None:
        kind = ALGORITHM_CAPABILITIES[algorithm].payload_kind
        assert kind is not None
        format = _SOURCE_FORMATS[kind]
        return _load_file_wire(source, format, params)
    assert layers is not None  # Validated pure router-parameter input.
    return _router_payload({layer: params for layer in layers}, {}).to_wire()


def prepare_preload(
    paths: list[str],
    algorithm: str,
    params: dict | None,
    *,
    hidden_size: int,
    model_info: dict | None,
) -> list[dict]:
    """Resolve and validate the complete preload batch before worker materialization."""
    from vllm.model_hooks.steering.validation import validate_payload_model

    payloads = [
        resolve_vector_payload(path, None, algorithm, params=params) for path in paths
    ]
    for payload in payloads:
        validate_payload_model(
            payload, algorithm, hidden_size=hidden_size, model_info=model_info
        )
    return payloads
