# SPDX-License-Identifier: Apache-2.0
"""Verify that steered generation produces equivalent logprobs in eager
mode and CUDA-graph mode (--steer-allow-cuda-graphs).

The test loads a small model with a steering vector in both modes and
compares per-token logprobs.  If CUDA graphs silently skip the steering
intervention during decode, logprobs will diverge significantly.

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
from vllm.steer_vectors.request import SteerVectorRequest

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
LOGPROB_ATOL = 0.05


def _skip_if_no_vector() -> None:
    if not Path(VECTOR_PATH).exists():
        pytest.skip(f"Steering vector not found: {VECTOR_PATH}")


def _generate_with_logprobs(
    *,
    enable_cuda_graphs: bool,
) -> tuple[list[str], list[float]]:
    """Run steered generation and return (tokens, logprobs)."""
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
        logprobs=0,  # return logprobs for chosen token
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

    # Extract tokens and logprobs from SampleLogprobs.
    # Each position is a dict[token_id, Logprob]; the chosen token is the
    # one actually generated (out.token_ids[i]).
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)
def test_steer_vector_cuda_graphs_logprob_equivalence() -> None:
    """Logprobs must be close between eager and CUDA-graph steered generation.

    This is the critical correctness test for --steer-allow-cuda-graphs.
    If CUDA graphs capture the model forward during a dummy run (before any
    steering vector is loaded), the graph may freeze the "no steering" code
    path.  On replay, steering would be silently skipped during decode,
    producing divergent logprobs.
    """
    _skip_if_no_vector()

    eager_tokens, eager_lps = _generate_with_logprobs(enable_cuda_graphs=False)
    cg_tokens, cg_lps = _generate_with_logprobs(enable_cuda_graphs=True)

    print("\nLogprob comparison (eager vs CUDA graphs):")
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

    # Primary assertion: tokens must be identical
    assert eager_tokens == cg_tokens, (
        f"Token sequences differ!\n"
        f"  Eager: {eager_tokens}\n"
        f"  CUDA graphs: {cg_tokens}"
    )

    # Secondary assertion: logprobs must be close
    for i, (e_lp, c_lp) in enumerate(zip(eager_lps, cg_lps)):
        assert math.isclose(e_lp, c_lp, abs_tol=LOGPROB_ATOL), (
            f"Token {i} ({eager_tokens[i]!r}): logprob diff "
            f"{abs(e_lp - c_lp):.4f} > {LOGPROB_ATOL} "
            f"(eager={e_lp:.4f}, cg={c_lp:.4f})"
        )
