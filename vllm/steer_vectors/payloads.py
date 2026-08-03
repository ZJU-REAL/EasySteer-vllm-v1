# SPDX-License-Identifier: Apache-2.0
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
import hashlib
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
        digest = hashlib.sha256()
        digest.update(f"{self.kind}:{_WIRE_VERSION}".encode())
        wire_tensors = {}
        for name in sorted(tensors):
            arr = tensors[name]
            digest.update(name.encode())
            digest.update(str(arr.shape).encode())
            digest.update(arr.tobytes())
            wire_tensors[name] = {
                "shape": list(arr.shape),
                "data": arr.tobytes(),
            }
        extra = self._extra()
        for key in sorted(extra):
            digest.update(f"{key}={extra[key]!r}".encode())
        return {
            "version": _WIRE_VERSION,
            "kind": self.kind,
            "tensors": wire_tensors,
            "extra": extra,
            "sha256": digest.hexdigest(),
        }


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
        self.bias = None if bias is None else _as_array("bias", bias, ndim=1)
        if self.bias is not None and self.bias.shape[0] != self.weight.shape[0]:
            raise ValueError(
                f"bias size {self.bias.shape[0]} != weight rows "
                f"{self.weight.shape[0]}"
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
    """Concept substitution directions: steer toward h1, away from h2.

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

    def _tensors(self) -> dict[str, np.ndarray]:
        out = {}
        for layer, vec in self.h1.layers.items():
            out[f"h1.layer.{layer}"] = vec
        for layer, vec in self.h2.layers.items():
            out[f"h2.layer.{layer}"] = vec
        return out


PAYLOAD_KINDS: dict[str, type[Payload]] = {
    cls.kind: cls
    for cls in (
        DirectionVector,
        LinearMap,
        LowRankProjector,
        ReftIntervention,
        ConceptPair,
    )
}

# Which payload kind each tensor-carrying algorithm accepts via data=.
ALGORITHM_PAYLOADS: dict[str, str] = {
    "direct": "direction",
    "erase": "direction",
    "replace": "direction",
    "linear": "linear",
    "lm_steer": "lowrank",
    "loreft": "reft",
    "concept_replace": "concept_pair",
}

# Kinds whose single payload is broadcast to VectorSpec.layers (as
# opposed to carrying their own layer keys).
_BROADCAST_KINDS = ("linear", "lowrank")


def is_broadcast_kind(kind: str) -> bool:
    """Whether this payload kind requires VectorSpec.layers."""
    return kind in _BROADCAST_KINDS


def _wire_bytes(entry: dict[str, Any], name: str) -> np.ndarray:
    data = entry["data"]
    if isinstance(data, str):  # base64 (JSON transport)
        data = base64.b64decode(data)
    arr = np.frombuffer(data, dtype=np.float32).reshape(entry["shape"])
    return arr


def validate_wire(wire: dict[str, Any]) -> str:
    """Structurally validate a payload wire dict; returns its kind."""
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
    if kind not in PAYLOAD_KINDS:
        raise ValueError(
            f"unknown payload kind {kind!r}; expected one of "
            f"{sorted(PAYLOAD_KINDS)}"
        )
    return kind


def materialize(
    wire: dict[str, Any],
    device: str,
    dtype,
    target_layers: list[int] | None,
) -> dict[int, Any]:
    """Build the per-layer payload dict an algorithm consumes.

    Returns ``{layer_id: payload}`` with tensors on ``device`` in
    ``dtype`` — the same structure the file loaders produce, so the
    downstream slot/configure machinery is unchanged.
    """
    import torch

    kind = validate_wire(wire)
    tensors = {
        name: torch.from_numpy(_wire_bytes(entry, name).copy()).to(
            device=device, dtype=dtype
        )
        for name, entry in wire["tensors"].items()
    }

    def _layer_keyed(prefix: str) -> dict[int, Any]:
        out = {}
        for name, tensor in tensors.items():
            if name.startswith(prefix):
                out[int(name[len(prefix):])] = tensor
        return out

    if kind == "direction":
        return _layer_keyed("layer.")
    if kind == "concept_pair":
        h1 = _layer_keyed("h1.layer.")
        h2 = _layer_keyed("h2.layer.")
        return {layer: {"h1": h1[layer], "h2": h2[layer]} for layer in h1}
    if kind in _BROADCAST_KINDS:
        if not target_layers:
            raise ValueError(
                f"a {kind!r} payload holds one map applied to each "
                "target layer; VectorSpec.layers is required"
            )
        if kind == "linear":
            payload = {"weight": tensors["weight"], "bias": tensors.get("bias")}
        else:  # lowrank
            payload = {
                "projector1": tensors["projector1"],
                "projector2": tensors["projector2"],
            }
        return {layer: dict(payload) for layer in target_layers}
    if kind == "reft":
        payload = {
            "rotate_layer": tensors["rotate_layer"],
            "learned_source_weight": tensors["learned_source_weight"],
            "learned_source_bias": tensors.get("learned_source_bias"),
        }
        layer = wire["extra"].get("layer")
        if layer is not None:
            return {int(layer): payload}
        if not target_layers:
            raise ValueError(
                "this ReftIntervention records no layer; VectorSpec.layers "
                "is required"
            )
        return {int(la): dict(payload) for la in target_layers}
    raise AssertionError(f"unhandled payload kind {kind!r}")
