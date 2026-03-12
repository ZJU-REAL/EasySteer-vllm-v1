# SPDX-License-Identifier: Apache-2.0

# Base classes and template
from .base import BaseSteerVectorAlgorithm
from .template import AlgorithmTemplate

# Factory and registration
from .factory import create_algorithm, register_algorithm

# Algorithm implementations (import to register them)
from .direct import DirectAlgorithm
from .loreft import LoReFTAlgorithm
from .multi_vector import MultiVectorAlgorithm
from .linear import LinearTransformAlgorithm
from .lm_steer import LMSteerAlgorithm
from .moe_router import MoERouterAlgorithm
from .replace import ReplaceAlgorithm
from .erase import EraseAlgorithm
from .concept_replace import ConceptReplaceAlgorithm