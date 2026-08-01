# SPDX-License-Identifier: Apache-2.0
"""Verify that steered generation produces equivalent logprobs in eager
mode and CUDA-graph mode using **server-level steering**.

Server-level steering loads the vector at startup (before CUDA graph
capture), so graphs record the steered forward path.  This test
exercises both the legacy per-request path (eager only) and the new
server-level path (eager + CUDA graphs) and asserts logprob equivalence.

Run with:
    pytest tests/basic_correctness/test_steer_vector_cuda_graphs.py -v -s

Requires GPU and a steering vector at the path below.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from vllm import LLM, SamplingParams

# ── Configuration ──────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Resolve vector relative to the repo root (EasySteer/)
_REPO_ROOT = Path(__file__).resolve().parents[2]  # vllm-steer/
_EASYSTEER_ROOT = _REPO_ROOT.parent  # EasySteer/
VECTOR_PATH = str(_EASYSTEER_ROOT / "vectors" / "happy_diffmean.gguf")
TARGET_LAYERS = list(range(10, 26))
SCALE = 4.0
PROMPT = "Describe a rainy Monday morning."
MAX_TOKENS = 20

# Maximum allowed absolute logprob difference per token.
# Tiny float-precision diffs are OK; large diffs indicate missing steering.
LOGPROB_ATOL = 1e-4


def _skip_if_no_vector() -> None:
    if not Path(VECTOR_PATH).exists():
        pytest.skip(f"Steering vector not found: {VECTOR_PATH}")


def _generate_with_logprobs_server_level(
    *,
    enable_cuda_graphs: bool,
) -> tuple[list[str], list[float]]:
    """Run generation with server-level steering config."""
    llm = LLM(
        model=MODEL,
        enable_steer_vector=True,
        enforce_eager=not enable_cuda_graphs,
        steer_allow_cuda_graphs=enable_cuda_graphs,
        steer_vector_path=VECTOR_PATH,
        steer_scale=SCALE,
        steer_target_layers=TARGET_LAYERS,
        steer_algorithm="direct",
        steer_normalize=True,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.5,
        max_model_len=512,
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=0,  # return logprobs for chosen token
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # No per-request steer_vector_request needed — server-level handles it
    outputs = llm.generate(prompt_text, sampling_params=sampling)

    out = outputs[0].outputs[0]

    tokens: list[str] = []
    logprobs: list[float] = []
    for i, lp_dict in enumerate(out.logprobs):
        token_id = out.token_ids[i]
        lp_entry = lp_dict[token_id]
        tokens.append(lp_entry.decoded_token or f"<id:{token_id}>")
        logprobs.append(lp_entry.logprob)

    # Clean up GPU memory
    del llm
    torch.cuda.empty_cache()

    return tokens, logprobs


def _generate_with_logprobs_per_request(
    *,
    enable_cuda_graphs: bool,
) -> tuple[list[str], list[float]]:
    """Run steered generation with per-request steering (legacy)."""
    from vllm.steer_vectors.request import SteerVectorRequest

    llm = LLM(
        model=MODEL,
        enable_steer_vector=True,
        enforce_eager=not enable_cuda_graphs,
        steer_allow_cuda_graphs=enable_cuda_graphs,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.5,
        max_model_len=512,
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=0,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )

    steer_req = SteerVectorRequest(
        steer_vector_local_path=VECTOR_PATH,
        scale=SCALE,
        target_layers=TARGET_LAYERS,
        algorithm="direct",
        normalize=True,
        prefill_trigger_tokens=[-1],
        generate_trigger_tokens=[-1],
    )

    outputs = llm.generate(
        prompt_text,
        sampling_params=sampling,
        steer_vector_request=steer_req,
    )

    out = outputs[0].outputs[0]

    tokens: list[str] = []
    logprobs: list[float] = []
    for i, lp_dict in enumerate(out.logprobs):
        token_id = out.token_ids[i]
        lp_entry = lp_dict[token_id]
        tokens.append(lp_entry.decoded_token or f"<id:{token_id}>")
        logprobs.append(lp_entry.logprob)

    del llm
    torch.cuda.empty_cache()

    return tokens, logprobs


def _print_comparison(
    eager_tokens: list[str],
    eager_lps: list[float],
    cg_tokens: list[str],
    cg_lps: list[float],
    label: str,
) -> float:
    """Print logprob comparison table and return max diff."""
    print(f"\nLogprob comparison ({label}):")
    print(f"{'pos':>4} {'eager_tok':>14} {'eager_lp':>10} "
          f"{'cg_tok':>14} {'cg_lp':>10} {'diff':>8}")
    print("-" * 70)

    max_diff = 0.0
    for i in range(min(len(eager_lps), len(cg_lps))):
        diff = abs(eager_lps[i] - cg_lps[i])
        max_diff = max(max_diff, diff)
        marker = " ***" if diff > LOGPROB_ATOL else ""
        print(
            f"{i:4d} {eager_tokens[i]!r:>14} {eager_lps[i]:10.4f} "
            f"{cg_tokens[i]!r:>14} {cg_lps[i]:10.4f} {diff:8.4f}{marker}"
        )

    print(f"\nMax logprob difference: {max_diff:.4f} (tolerance: {LOGPROB_ATOL})")
    return max_diff


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_server_level_steering_cuda_graphs_logprob_equivalence() -> None:
    """Server-level steering: logprobs must match between eager and CUDA graphs.

    This is the primary correctness test.  Server-level config loads the
    vector before graph capture, so graphs include the steering operations.
    """
    _skip_if_no_vector()

    eager_tokens, eager_lps = _generate_with_logprobs_server_level(
        enable_cuda_graphs=False
    )
    cg_tokens, cg_lps = _generate_with_logprobs_server_level(
        enable_cuda_graphs=True
    )

    _print_comparison(eager_tokens, eager_lps, cg_tokens, cg_lps,
                      "server-level: eager vs CUDA graphs")

    assert eager_tokens == cg_tokens, (
        f"Token sequences differ!\n"
        f"  Eager: {eager_tokens}\n"
        f"  CUDA graphs: {cg_tokens}"
    )

    for i, (e_lp, c_lp) in enumerate(zip(eager_lps, cg_lps)):
        assert math.isclose(e_lp, c_lp, abs_tol=LOGPROB_ATOL), (
            f"Token {i} ({eager_tokens[i]!r}): logprob diff "
            f"{abs(e_lp - c_lp):.4f} > {LOGPROB_ATOL} "
            f"(eager={e_lp:.4f}, cg={c_lp:.4f})"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_server_level_matches_per_request_eager() -> None:
    """Server-level eager output must match per-request eager output.

    This ensures the server-level loading path produces the same
    intervention as the existing per-request path.
    """
    _skip_if_no_vector()

    server_tokens, server_lps = _generate_with_logprobs_server_level(
        enable_cuda_graphs=False
    )
    per_req_tokens, per_req_lps = _generate_with_logprobs_per_request(
        enable_cuda_graphs=False
    )

    _print_comparison(server_tokens, server_lps, per_req_tokens, per_req_lps,
                      "server-level eager vs per-request eager")

    assert server_tokens == per_req_tokens, (
        f"Token sequences differ!\n"
        f"  Server-level: {server_tokens}\n"
        f"  Per-request: {per_req_tokens}"
    )

    for i, (s_lp, p_lp) in enumerate(zip(server_lps, per_req_lps)):
        assert math.isclose(s_lp, p_lp, abs_tol=LOGPROB_ATOL), (
            f"Token {i} ({server_tokens[i]!r}): logprob diff "
            f"{abs(s_lp - p_lp):.4f} > {LOGPROB_ATOL} "
            f"(server={s_lp:.4f}, per_req={p_lp:.4f})"
        )
