"""Test: scale=0 steered vLLM vs plain vLLM (no steering at all).

If steering at scale=0 changes logits compared to a plain vLLM instance,
then either the steering infrastructure or the chunked_prefill=False
setting is introducing numerical differences.

Usage:
    cd EasySteer/vllm-steer
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/basic_correctness/demo_scale0_vs_plain.py
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EASYSTEER_ROOT = Path(__file__).resolve().parents[2].parent
VECTOR_PATH = str(EASYSTEER_ROOT / "vectors" / "happy_diffmean.gguf")
TARGET_LAYERS = list(range(10, 26))
MAX_TOKENS = 40
LOGPROB_ATOL = 1e-4

PROMPTS = [
    "Describe a rainy Monday morning.",
    "What happens when you find a lost cat?",
    "Tell me about riding a bicycle through the park.",
    "What's the best way to make coffee?",
    "Describe the view from a mountaintop at sunrise.",
]


def build_prompts() -> list[str]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]


def extract_logprobs(outputs) -> list[tuple[list[str], list[float]]]:
    results = []
    for out in outputs:
        comp = out.outputs[0]
        tokens, lps = [], []
        for i, lp_dict in enumerate(comp.logprobs):
            tid = comp.token_ids[i]
            entry = lp_dict[tid]
            tokens.append(entry.decoded_token or f"<id:{tid}>")
            lps.append(entry.logprob)
        results.append((tokens, lps))
    return results


def run_plain(prompts: list[str]) -> list[tuple[list[str], list[float]]]:
    """Plain vLLM — no steering at all."""
    print("\n" + "=" * 60)
    print("PLAIN vLLM (no steering, chunked_prefill=False)")
    print("=" * 60)

    llm = LLM(
        model=MODEL,
        # No steering enabled at all
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.4,
        max_model_len=512,
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=0)
    outputs = llm.generate(prompts, sampling_params=sampling)
    results = extract_logprobs(outputs)

    for i, (tokens, lps) in enumerate(results):
        print(f"  Prompt {i}: {''.join(tokens)[:80]}...")

    del llm
    torch.cuda.empty_cache()
    return results


def run_plain_chunked(prompts: list[str]) -> list[tuple[list[str], list[float]]]:
    """Plain vLLM — no steering, chunked_prefill=True (the default)."""
    print("\n" + "=" * 60)
    print("PLAIN vLLM (no steering, chunked_prefill=True)")
    print("=" * 60)

    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.4,
        max_model_len=512,
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=0)
    outputs = llm.generate(prompts, sampling_params=sampling)
    results = extract_logprobs(outputs)

    for i, (tokens, lps) in enumerate(results):
        print(f"  Prompt {i}: {''.join(tokens)[:80]}...")

    del llm
    torch.cuda.empty_cache()
    return results


def run_steered_scale0(prompts: list[str]) -> list[tuple[list[str], list[float]]]:
    """Server-level steering at scale=0 (should be a no-op)."""
    print("\n" + "=" * 60)
    print("STEERED vLLM (scale=0, chunked_prefill=False)")
    print("=" * 60)

    llm = LLM(
        model=MODEL,
        steer_vector_path=VECTOR_PATH,
        steer_scale=0.0,
        steer_target_layers=TARGET_LAYERS,
        steer_algorithm="direct",
        steer_normalize=True,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.4,
        max_model_len=512,
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=0)
    outputs = llm.generate(prompts, sampling_params=sampling)
    results = extract_logprobs(outputs)

    for i, (tokens, lps) in enumerate(results):
        print(f"  Prompt {i}: {''.join(tokens)[:80]}...")

    del llm
    torch.cuda.empty_cache()
    return results


def compare(
    name_a: str,
    results_a: list[tuple[list[str], list[float]]],
    name_b: str,
    results_b: list[tuple[list[str], list[float]]],
) -> bool:
    print(f"\n--- {name_a} vs {name_b} ---")
    print(f"{'prompt':>6} {'tokens':>7} {'match':>6} {'max_diff':>12}")
    print("-" * 35)

    all_pass = True
    global_max = 0.0
    for i in range(len(results_a)):
        toks_a, lps_a = results_a[i]
        toks_b, lps_b = results_b[i]
        match = toks_a == toks_b
        max_diff = max(
            (abs(a - b) for a, b in zip(lps_a, lps_b)),
            default=0.0,
        )
        global_max = max(global_max, max_diff)
        ok = match and max_diff <= LOGPROB_ATOL
        if not ok:
            all_pass = False
        print(f"{i:6d} {len(toks_a):7d} {'yes' if match else 'NO':>6} {max_diff:12.6f}{'  ***' if not ok else ''}")

    print(f"Global max diff: {global_max:.6f} (tol={LOGPROB_ATOL})")
    return all_pass


def main() -> None:
    assert Path(VECTOR_PATH).exists(), f"Vector not found: {VECTOR_PATH}"
    prompts = build_prompts()

    plain = run_plain(prompts)
    plain_chunked = run_plain_chunked(prompts)
    steered_s0 = run_steered_scale0(prompts)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    p1 = compare("plain (no chunk)", plain, "steered scale=0 (no chunk)", steered_s0)
    p2 = compare("plain (no chunk)", plain, "plain (chunked)", plain_chunked)

    print()
    if p1:
        print("[PASS] scale=0 steering is identical to plain vLLM")
    else:
        print("[FAIL] scale=0 steering differs from plain vLLM!")

    if p2:
        print("[PASS] chunked vs non-chunked plain vLLM are identical")
    else:
        print("[FAIL] chunked vs non-chunked plain vLLM differ!")

    if not (p1 and p2):
        exit(1)


if __name__ == "__main__":
    main()
