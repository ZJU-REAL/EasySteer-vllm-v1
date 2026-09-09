# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manage defaults and preloaded steering payloads for request admission."""

import asyncio
import json

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from vllm.exceptions import VLLMClientError
from vllm.model_hooks.steering.api import SteeringSpec

router = APIRouter()


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": message})


@router.get("/v1/steering/vectors")
async def list_steering_vectors(raw_request: Request) -> JSONResponse:
    """List source paths whose payloads were preloaded on the workers."""
    if raw_request.app.state.vllm_config.steer_vector_config is None:
        return _bad_request("SteerVector is not enabled.")
    engine_client = raw_request.app.state.engine_client
    return JSONResponse(
        content={"preloaded": engine_client.list_preloaded_steer_vectors()}
    )


@router.post("/v1/steering/vectors")
async def preload_steering_vectors(raw_request: Request) -> JSONResponse:
    """Preload paths with the same algorithm and params used by VectorSpec."""
    if raw_request.app.state.vllm_config.steer_vector_config is None:
        return _bad_request("SteerVector is not enabled.")
    try:
        body = await raw_request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return _bad_request(f"Invalid JSON: {error}")
    if not isinstance(body, dict):
        return _bad_request("Body must be a JSON object.")
    paths = body.get("paths")
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or not path.strip() for path in paths)
    ):
        return _bad_request("'paths' must be a non-empty list of non-empty strings.")
    algorithm = body.get("algorithm", "direct")
    if not isinstance(algorithm, str) or not algorithm.strip():
        return _bad_request("'algorithm' must be a non-empty string.")
    params = body.get("params")
    if params is not None and not isinstance(params, dict):
        return _bad_request("'params' must be an object or null.")
    engine_client = raw_request.app.state.engine_client
    try:
        await engine_client.preload_steer_vectors(paths, algorithm, params)
    except VLLMClientError as error:
        return _bad_request(f"Preload failed: {error}")
    return JSONResponse(
        content={"preloaded": engine_client.list_preloaded_steer_vectors()}
    )


@router.get("/v1/steering")
async def get_steering_config(raw_request: Request) -> JSONResponse:
    """Return the default used by newly admitted requests."""
    return JSONResponse(
        content=raw_request.app.state.engine_client.get_default_steering()
    )


@router.post("/v1/steering")
async def update_steering_config(raw_request: Request) -> JSONResponse:
    """Replace the default with {"spec": <SteeringSpec>} or clear with null."""
    try:
        body = await raw_request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return _bad_request(f"Invalid JSON: {error}")
    if not isinstance(body, dict) or "spec" not in body:
        return _bad_request('Body must be {"spec": <SteeringSpec or null>}.')
    engine_client = raw_request.app.state.engine_client
    async with raw_request.app.state.steering_update_lock:
        try:
            spec = (
                None
                if body["spec"] is None
                else SteeringSpec.model_validate(body["spec"])
            )
            await engine_client.set_default_steering(spec)
        except (
            ValidationError,
            VLLMClientError,
            ValueError,
            TypeError,
            OSError,
        ) as error:
            return _bad_request(f"Default steering rejected: {error}")
    return JSONResponse(content=engine_client.get_default_steering())


def attach_router(app: FastAPI) -> None:
    app.state.steering_update_lock = asyncio.Lock()
    app.include_router(router)
