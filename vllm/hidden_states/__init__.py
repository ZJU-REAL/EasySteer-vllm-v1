# SPDX-License-Identifier: Apache-2.0
"""
Hidden States Capture for vLLM

Provides functionality to capture intermediate hidden states from transformer models
during inference. Integrated into vLLM's V1 worker architecture.
"""

from vllm.hidden_states.storage import HiddenStatesStore
from vllm.hidden_states.wrapper import VLLMTransformerLayerWrapper
from vllm.hidden_states.request import HiddenStatesCaptureRequest
from vllm.hidden_states.utils import (
    deserialize_hidden_states,
    print_hidden_states_summary,
)

# MoE router logits capture
from vllm.hidden_states.moe_storage import MoERouterLogitsStore
from vllm.hidden_states.moe_wrapper import (
    VLLMMoELayerWrapper,
    is_moe_layer,
    extract_moe_layer_id_from_name,
    MOE_LAYER_CLASSES,
)
from vllm.hidden_states.moe_request import MoERouterLogitsCaptureRequest
from vllm.hidden_states.moe_utils import (
    deserialize_moe_router_logits,
    print_moe_router_logits_summary,
)

__all__ = [
    # Hidden states
    "HiddenStatesStore",
    "VLLMTransformerLayerWrapper",
    "HiddenStatesCaptureRequest",
    "deserialize_hidden_states",
    "print_hidden_states_summary",
    # MoE router logits
    "MoERouterLogitsStore",
    "VLLMMoELayerWrapper",
    "MoERouterLogitsCaptureRequest",
    "is_moe_layer",
    "extract_moe_layer_id_from_name",
    "MOE_LAYER_CLASSES",
    "deserialize_moe_router_logits",
    "print_moe_router_logits_summary",
]

__version__ = "1.0.0"

