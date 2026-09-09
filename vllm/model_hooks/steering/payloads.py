# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical in-memory steering payloads.

Each steering algorithm declares the minimal data structure it needs;
users construct these directly (from their own files, training runs, or
`easysteer.vectors` adapters) and pass them as ``VectorSpec(data=...)``.
The engine never guesses at third-party file schemas: it accepts these
structures, or its own deterministic GGUF export format.

Wire form: `to_wire()` produces a msgspec-friendly dict of raw tensor
bytes with a content sha256 (the payload's identity for slot dedup and
prefix-cache keying); `from_wire()` rebuilds it, accepting base64
strings in place of bytes so payloads can also travel as JSON (HTTP).
`materialize()` converts a wire dict into the per-layer payload dict an
algorithm's `_transform` consumes, on the target device and dtype.
"""

import base64
import copy
import hashlib
import json
import math
from typing import Any

import numpy as np

_WIRE_VERSION = 1


def _as_array(name: str, value: Any, ndim: int) -> np.ndarray:
    """Coerce a tensor-like value to a contiguous float32 numpy array."""
    if value is None:
        raise ValueError(f"{name} must not be None")
    if hasattr(value, "detach"):  # torch.Tensor, without importing torch
        value = value.detach().cpu().float().numpy()
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _check_dims(tensors: dict[str, np.ndarray], axis_of: dict[str, int]) -> None:
    """Require the hidden dimension to agree across named tensors."""
    dims = {name: t.shape[axis_of[name]] for name, t in tensors.items()}
    if len(set(dims.values())) > 1:
        raise ValueError(f"hidden dimensions disagree across tensors: {dims}")


class Payload:
    """Base class: named float32 tensors plus payload-specific fields."""

    kind: str  # set by subclasses

    def _tensors(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _extra(self) -> dict[str, Any]:
        return {}

    def to_wire(self) -> dict[str, Any]:
        tensors = self._tensors()
        wire_tensors = {}
        for name in sorted(tensors):
            arr = tensors[name]
            wire_tensors[name] = {
                "shape": list(arr.shape),
                "data": arr.tobytes(),
            }
        wire = {
            "version": _WIRE_VERSION,
            "kind": self.kind,
            "tensors": wire_tensors,
            "extra": copy.deepcopy(self._extra()),
        }
        wire["sha256"] = _wire_digest(wire)
        return wire


class DirectionVector(Payload):
    """Per-layer direction vectors: ``h' = h + scale * v``.

    Accepted by the direct, erase and replace algorithms. Layers are
    keyed by true layer id; each value is a 1-D vector of the model's
    hidden size.
    """

    kind = "direction"

    def __init__(self, layers: dict[int, Any]):
        if not layers:
            raise ValueError("DirectionVector requires at least one layer")
        self.layers = {
            int(layer): _as_array(f"layers[{layer}]", vec, ndim=1)
            for layer, vec in layers.items()
        }
        dims = {v.shape[0] for v in self.layers.values()}
        if len(dims) > 1:
            raise ValueError(f"layer vectors have differing sizes: {sorted(dims)}")

    def _tensors(self) -> dict[str, np.ndarray]:
        return {f"layer.{layer}": vec for layer, vec in self.layers.items()}


class LinearMap(Payload):
    """An affine map ``h' = h @ W.T + b`` applied to each target layer.

    Accepted by the linear algorithm. Requires ``VectorSpec.layers``.
    """

    kind = "linear"

    def __init__(self, weight: Any, bias: Any = None):
        self.weight = _as_array("weight", weight, ndim=2)
        if self.weight.shape[0] != self.weight.shape[1]:
            raise ValueError(f"weight must be square, got shape {self.weight.shape}")
        self.bias = None if bias is None else _as_array("bias", bias, ndim=1)
        if self.bias is not None and self.bias.shape[0] != self.weight.shape[0]:
            raise ValueError(
                f"bias size {self.bias.shape[0]} != weight rows {self.weight.shape[0]}"
            )

    def _tensors(self) -> dict[str, np.ndarray]:
        out = {"weight": self.weight}
        if self.bias is not None:
            out["bias"] = self.bias
        return out


class LowRankProjector(Payload):
    """A low-rank update ``h' = h + scale * (h @ P1) @ P2.T``.

    Accepted by the lm_steer algorithm. Requires ``VectorSpec.layers``.
    """

    kind = "lowrank"

    def __init__(self, projector1: Any, projector2: Any):
        self.projector1 = _as_array("projector1", projector1, ndim=2)
        self.projector2 = _as_array("projector2", projector2, ndim=2)
        _check_dims(
            {"projector1": self.projector1, "projector2": self.projector2},
            {"projector1": 0, "projector2": 0},
        )
        if self.projector1.shape[1] != self.projector2.shape[1]:
            raise ValueError(
                f"projector ranks disagree: {self.projector1.shape[1]} vs "
                f"{self.projector2.shape[1]}"
            )

    def _tensors(self) -> dict[str, np.ndarray]:
        return {"projector1": self.projector1, "projector2": self.projector2}


class ReftIntervention(Payload):
    """A LoReFT intervention: rotation basis + learned source map.

    Accepted by the loreft algorithm. ``layer`` fixes the target layer
    (as recorded in a checkpoint); when None, ``VectorSpec.layers`` is
    required and the intervention is applied to each listed layer.
    """

    kind = "reft"

    def __init__(
        self,
        rotate_layer: Any,
        learned_source_weight: Any,
        learned_source_bias: Any = None,
        layer: int | None = None,
    ):
        self.rotate_layer = _as_array("rotate_layer", rotate_layer, ndim=2)
        self.learned_source_weight = _as_array(
            "learned_source_weight", learned_source_weight, ndim=2
        )
        self.learned_source_bias = (
            None
            if learned_source_bias is None
            else _as_array("learned_source_bias", learned_source_bias, ndim=1)
        )
        hidden_size, rank = self.rotate_layer.shape
        if self.learned_source_weight.shape != (rank, hidden_size):
            raise ValueError(
                f"learned_source_weight shape {self.learned_source_weight.shape} "
                f"must be {(rank, hidden_size)} for rotate_layer shape "
                f"{self.rotate_layer.shape}"
            )
        if self.learned_source_bias is not None and self.learned_source_bias.shape != (
            rank,
        ):
            raise ValueError(
                f"learned_source_bias size {self.learned_source_bias.shape[0]} "
                f"!= rotation rank {rank}"
            )
        self.layer = None if layer is None else int(layer)

    def _tensors(self) -> dict[str, np.ndarray]:
        out = {
            "rotate_layer": self.rotate_layer,
            "learned_source_weight": self.learned_source_weight,
        }
        if self.learned_source_bias is not None:
            out["learned_source_bias"] = self.learned_source_bias
        return out

    def _extra(self) -> dict[str, Any]:
        return {"layer": self.layer}


class ConceptPair(Payload):
    """Concept substitution directions: replace the h1 component with h2.

    Accepted by the concept_replace algorithm. Both sides carry the same
    layer set; the roles are explicit — there is no filename or ordering
    heuristic.
    """

    kind = "concept_pair"

    def __init__(self, h1: "DirectionVector | dict", h2: "DirectionVector | dict"):
        self.h1 = h1 if isinstance(h1, DirectionVector) else DirectionVector(h1)
        self.h2 = h2 if isinstance(h2, DirectionVector) else DirectionVector(h2)
        if set(self.h1.layers) != set(self.h2.layers):
            raise ValueError(
                f"h1 layers {sorted(self.h1.layers)} != h2 layers "
                f"{sorted(self.h2.layers)}"
            )
        for layer in self.h1.layers:
            _check_dims(
                {"h1": self.h1.layers[layer], "h2": self.h2.layers[layer]},
                {"h1": 0, "h2": 0},
            )

    def _tensors(self) -> dict[str, np.ndarray]:
        out = {}
        for layer, vec in self.h1.layers.items():
            out[f"h1.layer.{layer}"] = vec
        for layer, vec in self.h2.layers.items():
            out[f"h2.layer.{layer}"] = vec
        return out


ROUTER_MODES = ("activate", "deactivate", "soft", "soft_topk", "soft_random")


def validate_router_mode(mode: str) -> str:
    if mode not in ROUTER_MODES:
        raise ValueError(
            f"unknown moe_router mode {mode!r}; expected one of {sorted(ROUTER_MODES)}"
        )
    return mode


def _router_expert_ids(config: dict, name: str, layer: int) -> list[int]:
    values = config.get(name) or []
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in values
    ):
        raise ValueError(f"Layer {layer}: {name} must contain expert ids >= 0")
    return list(values)


class RouterConfig(Payload):
    """Per-layer expert interventions, shared by JSON and in-memory inputs."""

    kind = "router"

    def __init__(self, layers: dict[int, dict[str, Any]]):
        if not layers:
            raise ValueError("RouterConfig requires at least one layer")
        self.layers = {}
        for layer, config in layers.items():
            layer = int(layer)
            if layer < 0:
                raise ValueError("router layer ids must be non-negative")
            mode = config.get("mode", "activate")
            try:
                validate_router_mode(mode)
            except ValueError as exc:
                raise ValueError(f"Layer {layer}: {exc}") from None

            normalized = {
                "mode": mode,
                "expert_ids": _router_expert_ids(config, "expert_ids", layer),
            }
            if mode in ("activate", "deactivate"):
                normalized.update(
                    activate_ids=_router_expert_ids(config, "activate_ids", layer),
                    deactivate_ids=_router_expert_ids(config, "deactivate_ids", layer),
                    epsilon=float(config.get("epsilon", 0.01)),
                )
                if not any(
                    normalized[k]
                    for k in ("expert_ids", "activate_ids", "deactivate_ids")
                ):
                    raise ValueError(f"Layer {layer}: {mode} config has no expert ids")
                if not math.isfinite(normalized["epsilon"]):
                    raise ValueError(f"Layer {layer}: epsilon must be finite")
            else:
                if not normalized["expert_ids"]:
                    raise ValueError(f"Layer {layer}: {mode} config has no expert ids")
                normalized["lambda"] = float(config.get("lambda", 0.5))
                if not math.isfinite(normalized["lambda"]):
                    raise ValueError(f"Layer {layer}: lambda must be finite")
                if mode == "soft_topk":
                    topk = config.get("topk", 8)
                    if not isinstance(topk, int) or isinstance(topk, bool) or topk < 1:
                        raise ValueError(
                            f"Layer {layer}: topk must be a positive integer"
                        )
                    normalized["topk"] = topk
            if layer in self.layers:
                raise ValueError(f"duplicate router layer id: {layer}")
            self.layers[layer] = normalized

    def _tensors(self) -> dict[str, np.ndarray]:
        return {}

    def _extra(self) -> dict[str, Any]:
        return {"layers": {str(k): v for k, v in sorted(self.layers.items())}}


PAYLOAD_KINDS: dict[str, type[Payload]] = {
    cls.kind: cls
    for cls in (
        DirectionVector,
        LinearMap,
        LowRankProjector,
        ReftIntervention,
        ConceptPair,
        RouterConfig,
    )
}

# Kinds whose single payload is broadcast to VectorSpec.layers (as
# opposed to carrying their own layer keys).
_BROADCAST_KINDS = ("linear", "lowrank")


def is_broadcast_kind(kind: str) -> bool:
    """Whether this payload kind requires VectorSpec.layers."""
    return kind in _BROADCAST_KINDS


def is_broadcast_payload(wire: dict[str, Any]) -> bool:
    """Whether this payload carries no layer keys of its own."""
    return is_broadcast_kind(wire["kind"]) or (
        wire["kind"] == "reft" and wire["extra"].get("layer") is None
    )


def effective_layers(
    wire: dict[str, Any], target_layers: list[int] | None = None
) -> set[int]:
    """Resolve a payload's layer keys and the optional request layer filter."""
    kind = wire["kind"]
    if is_broadcast_payload(wire):
        if not target_layers:
            raise ValueError(
                f"a {kind!r} payload has no layer keys; VectorSpec.layers is required"
            )
        return set(target_layers)
    if kind in ("direction", "concept_pair"):
        prefix = "layer." if kind == "direction" else "h1.layer."
        layers = {
            int(name[len(prefix) :])
            for name in wire["tensors"]
            if name.startswith(prefix)
        }
    elif kind == "router":
        layers = {int(layer) for layer in wire["extra"]["layers"]}
    else:  # ReFT with a recorded checkpoint layer.
        layers = {int(wire["extra"]["layer"])}
    return layers if target_layers is None else layers.intersection(target_layers)


def _wire_bytes(entry: dict[str, Any], name: str) -> np.ndarray:
    data = entry["data"]
    if isinstance(data, str):  # base64 (JSON transport)
        data = base64.b64decode(data, validate=True)
    arr = np.frombuffer(data, dtype=np.float32).reshape(entry["shape"])
    return arr


def _wire_digest(wire: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{wire['kind']}:{_WIRE_VERSION}".encode())
    for name, entry in sorted(wire["tensors"].items()):
        digest.update(name.encode())
        digest.update(str(tuple(entry["shape"])).encode())
        data = entry["data"]
        digest.update(
            base64.b64decode(data, validate=True) if isinstance(data, str) else data
        )
    for key, value in sorted(wire["extra"].items()):
        # Preserve existing scalar metadata hashes; nested router configs
        # need a canonical representation independent of JSON object order.
        if isinstance(value, dict):
            value_repr = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            value_repr = repr(value)
        digest.update(f"{key}={value_repr}".encode())
    return digest.hexdigest()


def validate_wire(wire: dict[str, Any]) -> str:
    """Structurally validate a payload wire dict; returns its kind."""
    if not isinstance(wire, dict):
        raise ValueError("payload wire must be a dictionary")
    for key in ("version", "kind", "tensors", "extra", "sha256"):
        if key not in wire:
            raise ValueError(f"payload wire dict is missing {key!r}")
    if wire["version"] != _WIRE_VERSION:
        raise ValueError(
            f"unsupported payload wire version {wire['version']!r} "
            f"(expected {_WIRE_VERSION}); engine and client versions "
            "do not match"
        )
    kind = wire["kind"]
    if not isinstance(kind, str) or kind not in PAYLOAD_KINDS:
        raise ValueError(
            f"unknown payload kind {kind!r}; expected one of {sorted(PAYLOAD_KINDS)}"
        )
    if not isinstance(wire["tensors"], dict) or not isinstance(wire["extra"], dict):
        raise ValueError("payload tensors and extra must be dictionaries")
    if not isinstance(wire["sha256"], str):
        raise ValueError("payload sha256 must be a string")
    for name, entry in wire["tensors"].items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("payload tensors require names and tensor dictionaries")
        shape, data = entry.get("shape"), entry.get("data")
        if not isinstance(shape, (list, tuple)) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
            for dim in shape
        ):
            raise ValueError(f"payload tensor {name!r} has an invalid shape")
        if not isinstance(data, (bytes, bytearray, str)):
            raise ValueError(f"payload tensor {name!r} requires bytes or base64 data")
    required = {
        "linear": {"weight"},
        "lowrank": {"projector1", "projector2"},
        "reft": {"rotate_layer", "learned_source_weight"},
    }.get(kind, set())
    if missing := required - wire["tensors"].keys():
        raise ValueError(f"payload is missing tensor fields: {sorted(missing)}")
    if kind == "router" and not isinstance(wire["extra"].get("layers"), dict):
        raise ValueError("router payload requires per-layer configurations")
    return kind


def from_wire(wire: dict[str, Any]) -> Payload:
    """Validate content identity and rebuild the canonical payload on CPU."""
    kind = validate_wire(wire)
    if _wire_digest(wire) != wire["sha256"]:
        raise ValueError("payload sha256 does not match its content")
    tensors = {
        name: _wire_bytes(entry, name) for name, entry in wire["tensors"].items()
    }

    def layers(prefix):
        selected = {}
        for name, value in tensors.items():
            if name.startswith(prefix):
                layer = int(name[len(prefix) :])
                if layer < 0 or layer in selected:
                    raise ValueError(f"invalid or duplicate payload layer: {name}")
                selected[layer] = value
        return selected

    if kind == "direction":
        payload = DirectionVector(layers("layer."))
    elif kind == "concept_pair":
        payload = ConceptPair(layers("h1.layer."), layers("h2.layer."))
    elif kind == "linear":
        payload = LinearMap(tensors["weight"], tensors.get("bias"))
    elif kind == "lowrank":
        payload = LowRankProjector(tensors["projector1"], tensors["projector2"])
    elif kind == "reft":
        payload = ReftIntervention(
            tensors["rotate_layer"],
            tensors["learned_source_weight"],
            tensors.get("learned_source_bias"),
            layer=wire["extra"].get("layer"),
        )
    else:
        payload = RouterConfig(wire["extra"]["layers"])
    if set(payload._tensors()) != set(tensors):
        raise ValueError(f"unexpected tensor fields in {kind!r} payload")
    if set(payload._extra()) != set(wire["extra"]):
        raise ValueError(f"unexpected metadata fields in {kind!r} payload")
    return payload


def validate_model_shape(wire: dict[str, Any], hidden_size: int) -> None:
    """Check admitted tensor shapes against the model before device allocation.

    Hidden axes must match exactly in every execution mode; rank axes are
    independent of the model width and may use padded full-graph storage.
    """
    kind = wire["kind"]
    named_axes = {
        "linear": {"weight": (0, 1), "bias": (0,)},
        "lowrank": {"projector1": (0,), "projector2": (0,)},
        "reft": {
            "rotate_layer": (0,),
            "learned_source_weight": (1,),
            "learned_source_bias": (),
        },
        "router": {},
    }
    for name, entry in wire["tensors"].items():
        axes = (0,) if kind in ("direction", "concept_pair") else named_axes[kind][name]
        shape = entry["shape"]
        if any(shape[axis] != hidden_size for axis in axes):
            raise ValueError(
                f"{kind} payload {name!r} shape {tuple(shape)} does not match "
                f"model hidden size {hidden_size}; hidden dimensions must match "
                "exactly"
            )


def materialize(
    wire: dict[str, Any],
    device: str,
    dtype,
    target_layers: list[int] | None,
) -> dict[int, Any]:
    """Build the per-layer payload dict an algorithm consumes.

    Returns ``{layer_id: payload}`` with tensors on ``device`` in ``dtype``.
    """
    import torch

    kind = validate_wire(wire)
    layers = effective_layers(wire, target_layers)
    if kind == "router":
        return {
            layer: copy.deepcopy(wire["extra"]["layers"][str(layer)])
            for layer in layers
        }
    tensors = {
        name: torch.from_numpy(_wire_bytes(entry, name).copy()).to(
            device=device, dtype=dtype
        )
        for name, entry in wire["tensors"].items()
    }

    if kind == "direction":
        return {layer: tensors[f"layer.{layer}"] for layer in layers}
    if kind == "concept_pair":
        return {
            layer: {
                "h1": tensors[f"h1.layer.{layer}"],
                "h2": tensors[f"h2.layer.{layer}"],
            }
            for layer in layers
        }
    if kind == "linear":
        payload = {"weight": tensors["weight"], "bias": tensors.get("bias")}
    elif kind == "lowrank":
        payload = {
            "projector1": tensors["projector1"],
            "projector2": tensors["projector2"],
        }
    elif kind == "reft":
        payload = {
            "rotate_layer": tensors["rotate_layer"],
            "learned_source_weight": tensors["learned_source_weight"],
            "learned_source_bias": tensors.get("learned_source_bias"),
        }
    else:
        raise AssertionError(f"unhandled payload kind {kind!r}")
    return {layer: dict(payload) for layer in layers}
