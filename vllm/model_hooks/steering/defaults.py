# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request admission and immutable snapshots for default steering."""

import copy
import json
from dataclasses import dataclass
from typing import Literal

from vllm.model_hooks.steering.api import SteeringSpec, to_engine_request
from vllm.model_hooks.steering.request import SteeringRequest

SteeringChoice = SteeringSpec | Literal[False] | None
SteeringRequestChoice = SteeringRequest | Literal[False] | None


def build_default_request(steer_config) -> SteeringRequest:
    """The canonical engine-default steering request for a config.

    Resolve startup defaults once before workers are constructed. The
    immutable payload bytes are reused when requests inherit this config.
    """
    if steer_config.steering_config is None:
        raise ValueError(
            "build_default_request called without an engine-default "
            "steering config (steering_config is None)"
        )
    if (
        steer_config._default_request is not None
        and steer_config._default_request_spec == steer_config.steering_config
    ):
        return steer_config._default_request
    spec = SteeringSpec.model_validate_json(steer_config.steering_config)
    request = to_engine_request(spec, name="default-steering", int_id=1)
    steer_config.cache_default_request(request)
    return request


def resolve_steering_choice(steering: SteeringChoice) -> SteeringRequestChoice:
    """Convert an authoring choice, preserving inheritance and explicit off."""
    if steering is None or steering is False:
        return steering
    if not isinstance(steering, SteeringSpec):
        raise TypeError("steering must be a SteeringSpec, None, or False")
    return to_engine_request(steering)


@dataclass(frozen=True)
class DefaultSteeringState:
    """One atomic default version; only detached request copies leave this object."""

    _status_json: str = '{"active": false}'
    _request: SteeringRequest | None = None

    @classmethod
    def snapshot(cls, spec: SteeringSpec | str, request: SteeringRequest):
        resolved_spec = (
            SteeringSpec.model_validate_json(spec) if isinstance(spec, str) else spec
        )
        status = {
            "active": True,
            "spec": resolved_spec.model_dump(
                mode="json", exclude={"vectors": {"__all__": {"data"}}}
            ),
        }
        payloads = []
        for index, (vector, resolved) in enumerate(
            zip(resolved_spec.vectors, request.vectors)
        ):
            if vector.data is not None:
                payloads.append(
                    {
                        "vector": index,
                        "kind": resolved.payload["kind"],
                        "sha256": resolved.payload_sha256,
                    }
                )
        if payloads:
            status.update(data_omitted=True, payloads=payloads)
        # deepcopy shares immutable payload bytes and copies mutable selection/configs.
        return cls(json.dumps(status), copy.deepcopy(request))

    def resolve(self, request: SteeringRequestChoice) -> SteeringRequest | None:
        if request is False:
            return None
        if request is None:
            request = self._request
        elif not isinstance(request, SteeringRequest):
            raise TypeError("steer_vector_request must be a request, None, or False")
        return copy.deepcopy(request)

    def status(self) -> dict:
        return json.loads(self._status_json)


def require_unsteered_beam(request: SteeringRequestChoice) -> None:
    """Beam iterations reinterpret prior generated tokens as prompt tokens."""
    if request is not None and request is not False:
        from vllm.exceptions import VLLMValidationError

        raise VLLMValidationError(
            "Steering is not supported with beam search. "
            "Pass steering=False or use ordinary sampling."
        )
