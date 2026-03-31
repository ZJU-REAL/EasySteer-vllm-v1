# SPDX-License-Identifier: Apache-2.0
"""Unit tests for server-level steering configuration and buffer reuse.

These tests do NOT require a GPU — they exercise config, CLI argument
parsing, and the in-place buffer strategy in AlgorithmTemplate.

Run with:
    pytest tests/basic_correctness/test_server_level_steering.py -v
"""
from __future__ import annotations

import torch

# ── SteerVectorConfig tests ──────────────────────────────────────────


class TestSteerVectorConfig:
    """Tests for the server-level fields on SteerVectorConfig."""

    def test_default_has_no_server_config(self) -> None:
        from vllm.config.steer_vector import SteerVectorConfig

        cfg = SteerVectorConfig()
        assert not cfg.has_server_config

    def test_server_vector_path_enables_has_server_config(self) -> None:
        from vllm.config.steer_vector import SteerVectorConfig

        cfg = SteerVectorConfig(
            server_vector_path="/tmp/test.gguf",
            server_scale=4.0,
            server_target_layers=[10, 11, 12],
            server_algorithm="direct",
            server_normalize=True,
        )
        assert cfg.has_server_config
        assert cfg.server_vector_path == "/tmp/test.gguf"
        assert cfg.server_scale == 4.0
        assert cfg.server_target_layers == [10, 11, 12]

    def test_hash_includes_server_fields(self) -> None:
        from vllm.config.steer_vector import SteerVectorConfig

        cfg_default = SteerVectorConfig()
        cfg_server = SteerVectorConfig(server_vector_path="/tmp/test.gguf")
        assert cfg_default.compute_hash() != cfg_server.compute_hash()

    def test_hash_changes_with_server_scale(self) -> None:
        from vllm.config.steer_vector import SteerVectorConfig

        cfg_a = SteerVectorConfig(
            server_vector_path="/tmp/test.gguf", server_scale=1.0
        )
        cfg_b = SteerVectorConfig(
            server_vector_path="/tmp/test.gguf", server_scale=2.0
        )
        assert cfg_a.compute_hash() != cfg_b.compute_hash()


# ── EngineArgs tests ─────────────────────────────────────────────────


class TestEngineArgsServerSteering:
    """Tests for CLI argument handling of server-level steering."""

    def test_steer_vector_path_field_defaults_none(self) -> None:
        from vllm.engine.arg_utils import EngineArgs

        args = EngineArgs(model="test")
        assert args.steer_vector_path is None
        assert args.steer_scale == 1.0
        assert args.steer_target_layers is None
        assert args.steer_algorithm == "direct"
        assert args.steer_normalize is True

    def test_steer_vector_path_can_be_set(self) -> None:
        from vllm.engine.arg_utils import EngineArgs

        args = EngineArgs(
            model="test",
            steer_vector_path="/tmp/v.gguf",
            steer_scale=3.0,
            steer_target_layers=[1, 2, 3],
            steer_algorithm="loreft",
            steer_normalize=False,
        )
        assert args.steer_vector_path == "/tmp/v.gguf"
        assert args.steer_scale == 3.0
        assert args.steer_target_layers == [1, 2, 3]
        assert args.steer_algorithm == "loreft"
        assert args.steer_normalize is False


# ── AlgorithmTemplate buffer reuse tests ─────────────────────────────


class _DummyAlgorithm:
    """Minimal concrete subclass for testing AlgorithmTemplate."""

    def __new__(cls):
        from typing import Any

        from vllm.steer_vectors.algorithms.template import AlgorithmTemplate

        class _Impl(AlgorithmTemplate):
            def _transform(
                self, hidden_state: torch.Tensor, params: Any
            ) -> torch.Tensor:
                return hidden_state + params

            @classmethod
            def load_from_path(
                cls, path: str, device: str, **kwargs: Any
            ) -> dict[str, Any]:
                return {"payload": torch.zeros(10)}

        return _Impl(layer_id=0)


class TestTemplateBufferReuse:
    """Tests for in-place buffer strategy in set_steer_vector."""

    def test_first_load_creates_buffer(self) -> None:
        algo = _DummyAlgorithm()
        vec = torch.randn(10)
        algo.set_steer_vector(0, payload=vec, scale_factor=2.0)
        buf = algo._payloads[0]
        assert torch.allclose(buf, vec * 2.0)

    def test_reload_same_shape_reuses_address(self) -> None:
        algo = _DummyAlgorithm()
        vec1 = torch.randn(10)
        algo.set_steer_vector(0, payload=vec1, scale_factor=2.0)
        addr1 = algo._payloads[0].data_ptr()

        vec2 = torch.randn(10)
        algo.set_steer_vector(0, payload=vec2, scale_factor=3.0)
        addr2 = algo._payloads[0].data_ptr()

        assert addr1 == addr2, "Buffer address must be reused for CUDA graph safety"
        assert torch.allclose(algo._payloads[0], vec2 * 3.0)

    def test_different_shape_allocates_new_buffer(self) -> None:
        algo = _DummyAlgorithm()
        algo.set_steer_vector(0, payload=torch.randn(10), scale_factor=1.0)
        addr1 = algo._payloads[0].data_ptr()

        algo.set_steer_vector(0, payload=torch.randn(20), scale_factor=1.0)
        addr2 = algo._payloads[0].data_ptr()

        assert addr1 != addr2

    def test_set_active_tensor_reuses_address(self) -> None:
        algo = _DummyAlgorithm()
        vec = torch.randn(10)
        algo.set_steer_vector(0, payload=vec, scale_factor=2.0)

        # Pre-set active payload with compatible shape
        algo._active_payload = torch.zeros(10)
        active_ptr = algo._active_payload.data_ptr()

        algo.set_active_tensor(0)

        assert algo._active_payload.data_ptr() == active_ptr, (
            "set_active_tensor must copy_ into existing buffer"
        )
        assert torch.allclose(algo._active_payload, vec * 2.0)

    def test_set_active_tensor_incompatible_replaces(self) -> None:
        algo = _DummyAlgorithm()
        algo.set_steer_vector(0, payload=torch.randn(10), scale_factor=1.0)

        # Active payload has different shape
        algo._active_payload = torch.zeros(5)
        algo.set_active_tensor(0)

        assert algo._active_payload.shape == (10,)

    def test_dict_payload_preserves_scale_factor(self) -> None:
        algo = _DummyAlgorithm()
        algo.set_steer_vector(
            0, payload={"matrix": torch.eye(3)}, scale_factor=0.5
        )
        p = algo._payloads[0]
        assert isinstance(p, dict)
        assert p["scale_factor"] == 0.5
