"""Demo: old per-request steering vs new server-level steering.

Old way (slow, no CUDA graphs):
  - Server started with --enable-steer-vector
  - Each request includes steer_vector_request
  - enforce_eager=True → no CUDA graphs

New way (fast, CUDA graphs):
  - Server started with --steer-vector-path (server-level config)
  - Vector loaded at startup, before CUDA graph capture
  - Requests need no steer_vector_request
  - enforce_eager=False, CUDAGraphMode.FULL_DECODE_ONLY

Both should produce allclose logits.

Usage:
    cd EasySteer/vllm-steer
    .venv/bin/python tests/basic_correctness/demo_server_vs_perrequest.py
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# ── Configuration ──────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_SCRIPT_DIR = Path(__file__).resolve().parent
_EASYSTEER_ROOT = _SCRIPT_DIR.parents[1].parent  # EasySteer/
VECTOR_PATH = str(_EASYSTEER_ROOT / "vectors" / "happy_diffmean.gguf")
TARGET_LAYERS = list(range(10, 26))
SCALE = 4.0
MAX_TOKENS = 60
LOGPROB_ATOL = 1e-4

PROMPTS = [
    "Describe a rainy Monday morning.",
    "What happens when you find a lost cat?",
    "Tell me about riding a bicycle through the park.",
    "What's the best way to make coffee?",
    "Describe the view from a mountaintop at sunrise.",
    "Write a story about a friendly robot.",
    "Explain why the ocean is so fascinating.",
    "What would you do with a million dollars?",
    "Describe a perfect summer afternoon.",
    "Tell me about an adventure in a magical forest.",
    "What makes a good friend?",
    "Describe the feeling of finishing a marathon.",
    "Tell me about learning to cook for the first time.",
    "What happens when it snows in a big city?",
    "Describe a day at the beach with your family.",
    "Write about discovering a hidden garden.",
    "What's it like to fly in a hot air balloon?",
    "Describe the sounds of a busy market.",
    "Tell me about a journey across the desert.",
    "What makes music so powerful?",
]


def build_prompt(text: str) -> str:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_logprobs(outputs) -> list[tuple[list[str], list[float]]]:
    """Extract (tokens, logprobs) for each output."""
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


def run_old_way(prompts: list[str]) -> tuple[list, float]:
    """Old way: per-request steering, enforce_eager=True (no CUDA graphs)."""
    print("\n" + "=" * 70)
    print("OLD WAY: per-request steering, enforce_eager=True (no CUDA graphs)")
    print("=" * 70)

    llm = LLM(
        model=MODEL,
        enable_steer_vector=True,
        enforce_eager=True,           # ← no CUDA graphs
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.4,
        max_model_len=512,
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=0,
    )

    steer_req = SteerVectorRequest(
        steer_vector_name="happy_demo_server_vs_perrequest_test",
        steer_vector_int_id=1,
        steer_vector_local_path=VECTOR_PATH,
        scale=SCALE,
        target_layers=TARGET_LAYERS,
        algorithm="direct",
        normalize=True,
        prefill_trigger_tokens=[-1],
        generate_trigger_tokens=[-1],
    )

    t0 = time.perf_counter()
    outputs = llm.generate(
        prompts, sampling_params=sampling,
        steer_vector_request=steer_req,
    )
    elapsed = time.perf_counter() - t0

    results = extract_logprobs(outputs)

    for i, (tokens, lps) in enumerate(results):
        print(f"\nPrompt {i}: {PROMPTS[i]!r}")
        print(f"  Output: {''.join(tokens)}")
        print(f"  Logprobs (first 10): {[f'{lp:.4f}' for lp in lps[:10]]}")

    print(f"\nTime: {elapsed:.2f}s")

    del llm
    torch.cuda.empty_cache()
    return results, elapsed


def run_new_way(prompts: list[str]) -> tuple[list, float]:
    """New way: server-level steering, CUDA graphs enabled."""
    print("\n" + "=" * 70)
    print("NEW WAY: server-level steering, CUDA graphs (FULL_DECODE_ONLY)")
    print("=" * 70)

    llm = LLM(
        model=MODEL,
        # Server-level steering — vector loaded at startup, before graph capture
        steer_vector_path=VECTOR_PATH,
        steer_scale=SCALE,
        steer_target_layers=TARGET_LAYERS,
        steer_algorithm="direct",
        steer_normalize=True,
        # No enforce_eager → CUDA graphs enabled
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.4,
        max_model_len=512,
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=0,
    )

    # No steer_vector_request needed — server handles it
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling)
    elapsed = time.perf_counter() - t0

    results = extract_logprobs(outputs)

    for i, (tokens, lps) in enumerate(results):
        print(f"\nPrompt {i}: {PROMPTS[i]!r}")
        print(f"  Output: {''.join(tokens)}")
        print(f"  Logprobs (first 10): {[f'{lp:.4f}' for lp in lps[:10]]}")

    print(f"\nTime: {elapsed:.2f}s")

    del llm
    torch.cuda.empty_cache()
    return results, elapsed


def compare(
    old_results: list[tuple[list[str], list[float]]],
    new_results: list[tuple[list[str], list[float]]],
    old_time: float,
    new_time: float,
) -> None:
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    all_pass = True
    total_tokens = 0
    global_max_diff = 0.0

    # Print per-prompt summary
    print(f"\n{'prompt':>5} {'tokens':>7} {'match':>6} {'max_diff':>10} {'status':>8}")
    print("-" * 42)

    for i in range(len(old_results)):
        old_tokens, old_lps = old_results[i]
        new_tokens, new_lps = new_results[i]

        max_diff = 0.0
        for j in range(min(len(old_lps), len(new_lps))):
            diff = abs(old_lps[j] - new_lps[j])
            max_diff = max(max_diff, diff)

        global_max_diff = max(global_max_diff, max_diff)
        tokens_match = old_tokens == new_tokens
        n_tokens = len(old_tokens)
        total_tokens += n_tokens
        status = "PASS" if tokens_match and max_diff <= LOGPROB_ATOL else "FAIL"

        if not tokens_match or max_diff > LOGPROB_ATOL:
            all_pass = False

        match_s = "yes" if tokens_match else "NO"
        print(
            f"{i:5d} {n_tokens:7d} {match_s:>6}"
            f" {max_diff:10.6f} {status:>8}"
        )

    # Show first prompt detail as a spot check
    print(f"\nSpot check — Prompt 0: {PROMPTS[0]!r}")
    old_tokens, old_lps = old_results[0]
    new_tokens, new_lps = new_results[0]
    print(
        f"  {'pos':>4} {'old_tok':>16} {'old_lp':>10}"
        f" {'new_tok':>16} {'new_lp':>10} {'diff':>8}"
    )
    print(f"  {'-'*66}")
    for j in range(min(10, len(old_lps))):
        diff = abs(old_lps[j] - new_lps[j])
        print(
            f"  {j:4d} {old_tokens[j]!r:>16} {old_lps[j]:10.4f} "
            f"{new_tokens[j]!r:>16} {new_lps[j]:10.4f} {diff:8.4f}"
        )
    if len(old_lps) > 10:
        print(f"  ... ({len(old_lps) - 10} more tokens)")

    # Summary
    speedup = old_time / new_time if new_time > 0 else float("inf")
    old_tps = total_tokens / old_time
    new_tps = total_tokens / new_time

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Prompts:          {len(old_results)}")
    print(f"  Tokens/prompt:    {MAX_TOKENS}")
    print(f"  Total tokens:     {total_tokens}")
    print(f"  Global max diff:  {global_max_diff:.6f} (tol={LOGPROB_ATOL})")
    print(f"  Old (eager):      {old_time:.2f}s  ({old_tps:.1f} tok/s)")
    print(f"  New (CUDA graph): {new_time:.2f}s  ({new_tps:.1f} tok/s)")
    print(f"  Speedup:          {speedup:.2f}x")

    if all_pass:
        n = len(old_results)
        print(f"\n*** ALL {n} PROMPTS PASSED -- logprobs allclose ***")
    else:
        print("\n*** SOME CHECKS FAILED — see table above ***")


def main() -> None:
    assert Path(VECTOR_PATH).exists(), f"Vector not found: {VECTOR_PATH}"

    prompts = [build_prompt(p) for p in PROMPTS]

    old_results, old_time = run_old_way(prompts)
    new_results, new_time = run_new_way(prompts)
    compare(old_results, new_results, old_time, new_time)


if __name__ == "__main__":
    main()
