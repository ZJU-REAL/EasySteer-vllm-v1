# SPDX-License-Identifier: Apache-2.0

"""Steer Vectors for vLLM V1.

This module provides runtime intervention capabilities for LLMs through
steer vectors, allowing dynamic control over model behavior.
"""

from vllm.steer_vectors.layers import DecoderLayerWithSteerVector
from vllm.steer_vectors.models import (
    LoadedSteerVector,
    SteerControllerManager,
    create_steer_controller_manager,
)
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig
from vllm.steer_vectors.worker_manager import WorkerSteerVectorManager

__all__ = [
    # Layers
    "DecoderLayerWithSteerVector",
    # Models
    "LoadedSteerVector",
    "SteerControllerManager",
    "create_steer_controller_manager",
    # Request
    "SteerVectorRequest",
    "VectorConfig",
    # Worker Manager
    "WorkerSteerVectorManager",
]
