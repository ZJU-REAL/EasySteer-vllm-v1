#!/usr/bin/env python3
"""
Test EasySteer steer vector isolation correctness.

Verifies:
  1. Persistence:   steered request does NOT leak into later non-steered requests
  2. Scale=0 clear:  scale=0 explicitly clears any active vector
  3. Concurrent mix: requests with different vectors don't interfere
  4. User scenario:  no-extra_body after steered (exact reported bug)

Usage:
  # Test unpatched server (port 8017)
  python scripts/test_steer_isolation.py \
      --base-url http://localhost:8017/v1 \
      --vector-path /netdisk/zhaochengfeng/memory/EasySteer/vectors/happy_diffmean.gguf

  # Test patched server (port 8018)
  python scripts/test_steer_isolation.py \
      --base-url http://localhost:8018/v1 \
      --vector-path /netdisk/zhaochengfeng/memory/EasySteer/vectors/happy_diffmean.gguf

  # Stress test with high concurrency
  python scripts/test_steer_isolation.py \
      --base-url http://localhost:8018/v1 \
      --vector-path ... \
      --stress-concurrency 20
"""

import argparse
import asyncio
import os
import sys
import time
from collections import Counter
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Prompts — designed to show clear steering effects with happy_diffmean.gguf
# ---------------------------------------------------------------------------

# Prompt 1: emotional response — happy vector makes it inappropriately cheerful
PROMPT_COMFORT = (
    "Alice's beloved dog of 15 years has just passed away. "
    "She is heartbroken. Write a short, warm message to comfort her."
)

# Prompt 2: neutral description — happy vector injects unwarranted positivity
PROMPT_DESCRIBE = (
    "What's your feeling now? Describe your current emotional state in a few sentences."
)

SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_PROMPT = PROMPT_COMFORT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_steer_request(vector_path: str, scale: float, int_id: int,
                        target_layers: list[int]) -> dict:
    return {
        "steer_vector_local_path": os.path.abspath(vector_path),
        "scale": scale,
        "target_layers": target_layers,
        "prefill_trigger_tokens": [-1],
        "generate_trigger_tokens": [-1],
        "normalize": True,
        "steer_vector_int_id": int_id,
        "steer_vector_name": f"test_{int_id}",
        "algorithm": "direct",
    }


async def make_request(client: AsyncOpenAI, model: str,
                       extra_body: dict | None = None,
                       max_tokens: int = 128,
                       prompt: str = DEFAULT_PROMPT) -> str:
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if extra_body is not None:
        params["extra_body"] = extra_body
    response = await client.chat.completions.create(**params)
    return response.choices[0].message.content


def jaccard(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def verdict(close: float, far: float) -> tuple[bool, str]:
    """close should be HIGH (similar), far should be LOW (different).
    Returns (passed, label)."""
    margin = close - far
    if margin > 0.15:
        return (True, "PASS")
    elif margin > 0.0:
        return (True, f"WEAK (margin={margin:+.3f})")
    else:
        return (False, "FAIL")


def print_section(title: str):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

async def test_persistence(client, model, vector_path, target_layers) -> bool:
    """steered(2.0) → non-steered(no extra_body) → steered(2.0)"""
    print_section("Test 1: Persistence (steered → non-steered → steered)")
    int_id = 10001
    sreq = build_steer_request(vector_path, 2.0, int_id, target_layers)

    s1 = await make_request(client, model, {"steer_vector_request": sreq})
    print(f"  steered_1:   {s1[:120]}...")

    ns = await make_request(client, model, extra_body=None)
    print(f"  non_steered: {ns[:120]}...")

    s2 = await make_request(client, model, {"steer_vector_request": sreq})
    print(f"  steered_2:   {s2[:120]}...")

    sim_close = jaccard(s1, s2)   # steered vs steered — should be similar
    sim_far   = jaccard(ns, s1)   # non-steered vs steered — should differ
    ok, label = verdict(sim_close, sim_far)
    print(f"  steered~steered={sim_close:.3f}  non_steered~steered={sim_far:.3f}  → {label}")

    # Extra check: if all texts are word-for-word identical, the vector
    # had no visible effect — this is ambiguous (could be bug OR weak vector).
    if s1 == ns == s2:
        print("  WARNING: all 3 outputs are identical — vector may have no visible effect on this prompt")

    return ok


async def test_scale_zero(client, model, vector_path, target_layers) -> bool:
    """baseline → steered(2.0) → scale=0 → non-steered"""
    print_section("Test 2: Scale=0 deactivation")
    int_id = 10002

    baseline = await make_request(client, model, extra_body=None)
    print(f"  baseline:  {baseline[:120]}...")

    sreq = build_steer_request(vector_path, 2.0, int_id, target_layers)
    steered = await make_request(client, model, {"steer_vector_request": sreq})
    print(f"  steered:   {steered[:120]}...")

    zreq = build_steer_request(vector_path, 0.0, int_id, target_layers)
    _ = await make_request(client, model, {"steer_vector_request": zreq})

    az = await make_request(client, model, extra_body=None)
    print(f"  after_0:   {az[:120]}...")

    sim_close = jaccard(az, baseline)   # both non-steered — should match
    sim_far   = jaccard(az, steered)    # non-steered vs steered — should differ
    ok, label = verdict(sim_close, sim_far)
    print(f"  after0~baseline={sim_close:.3f}  after0~steered={sim_far:.3f}  → {label}")
    return ok


async def test_concurrent_3(client, model, vector_path, target_layers) -> bool:
    """3 concurrent: A(2.0), B(0.0), C(2.0). A≈C, A≠B."""
    print_section("Test 3: Concurrent isolation (3 req, A=2.0 B=0.0 C=2.0)")
    id_a, id_b = 20001, 20002

    ra = build_steer_request(vector_path, 2.0, id_a, target_layers)
    rb = build_steer_request(vector_path, 0.0, id_b, target_layers)
    rc = build_steer_request(vector_path, 2.0, id_a, target_layers)

    t0 = time.time()
    ta, tb, tc = await asyncio.gather(
        make_request(client, model, {"steer_vector_request": ra}),
        make_request(client, model, {"steer_vector_request": rb}),
        make_request(client, model, {"steer_vector_request": rc}),
    )
    dt = time.time() - t0

    print(f"  A (steered): {ta[:120]}...")
    print(f"  B (zero):    {tb[:120]}...")
    print(f"  C (steered): {tc[:120]}...")
    print(f"  time: {dt:.1f}s")

    sim_close = jaccard(ta, tc)   # A≈C (both steered)
    sim_far   = jaccard(ta, tb)   # A≠B (steered vs zero)
    ok, label = verdict(sim_close, sim_far)
    print(f"  A~C={sim_close:.3f}  A~B={sim_far:.3f}  → {label}")

    # Also check B≠C (B shouldn't be polluted by steered)
    sim_bc = jaccard(tb, tc)
    if sim_bc > sim_close:
        print(f"  WARNING: B~C={sim_bc:.3f} > A~C={sim_close:.3f} — B may be polluted!")

    return ok


async def test_stress_concurrent(client, model, vector_path, target_layers,
                                 n_reqs: int = 20) -> bool:
    """Stress test: N concurrent requests, half steered half zero, interleaved."""
    print_section(f"Test 4: Stress — {n_reqs} concurrent, steered & zero interleaved")

    id_steered = 30001
    id_zero = 30002

    sreq = build_steer_request(vector_path, 2.0, id_steered, target_layers)
    zreq = build_steer_request(vector_path, 0.0, id_zero, target_layers)

    # Build request list: alternating steered / zero
    tasks = []
    for i in range(n_reqs):
        if i % 2 == 0:
            tasks.append(("S", make_request(client, model, {"steer_vector_request": sreq})))
        else:
            tasks.append(("Z", make_request(client, model, {"steer_vector_request": zreq})))

    t0 = time.time()
    results = await asyncio.gather(*[t[1] for t in tasks])
    dt = time.time() - t0

    # Group results
    steered_texts = [results[i] for i in range(0, n_reqs, 2)]
    zero_texts    = [results[i] for i in range(1, n_reqs, 2)]

    # Check: all steered texts should be the same (temperature=0)
    sc = Counter(steered_texts)
    zc = Counter(zero_texts)
    s_unique = len(sc)
    z_unique = len(zc)

    print(f"  steered group: {s_unique} unique / {len(steered_texts)} requests")
    print(f"  zero group:    {z_unique} unique / {len(zero_texts)} requests")
    print(f"  time: {dt:.1f}s ({dt/n_reqs:.2f}s/req)")

    # Show top variants
    for label, counter, group_texts in [("steered", sc, steered_texts), ("zero", zc, zero_texts)]:
        if len(counter) <= 2:
            for text, count in counter.most_common(3):
                print(f"  {label}[{count}x]: {text[:100]}...")
        else:
            print(f"  {label}: {len(counter)} distinct outputs (expected 1 at T=0)")

    # Check cross-contamination: no zero text should equal a steered text
    steered_set = set(steered_texts)
    zero_set = set(zero_texts)
    overlap = steered_set & zero_set
    if overlap:
        print(f"  FAIL: {len(overlap)} texts appear in BOTH groups — cross-contamination!")
        return False

    # Check internal consistency within each group
    all_ok = True
    if s_unique > 1:
        print(f"  WARNING: steered group has {s_unique} variants — batching inconsistency?")
        all_ok = False
    if z_unique > 1:
        print(f"  WARNING: zero group has {z_unique} variants — batching inconsistency?")
        all_ok = False

    if all_ok:
        print(f"  PASS: groups are internally consistent & isolated from each other")
    return all_ok


async def test_user_scenario(client, model, vector_path, target_layers) -> bool:
    """Exact user-reported bug: steered → wait → non-steered(no extra_body)."""
    print_section("Test 5: User scenario (steered → no-extra_body)")
    int_id = 40001

    baseline = await make_request(client, model, extra_body=None)
    print(f"  baseline:    {baseline[:120]}...")

    sreq = build_steer_request(vector_path, 2.0, int_id, target_layers)
    steered = await make_request(client, model, {"steer_vector_request": sreq})
    print(f"  steered:     {steered[:120]}...")

    await asyncio.sleep(0.3)
    ns = await make_request(client, model, extra_body=None)
    print(f"  non_steered: {ns[:120]}...")

    sim_close = jaccard(ns, baseline)   # should match
    sim_far   = jaccard(ns, steered)    # should differ
    ok, label = verdict(sim_close, sim_far)
    print(f"  ns~baseline={sim_close:.3f}  ns~steered={sim_far:.3f}  → {label}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Test EasySteer steer vector isolation")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="/netcache/huggingface/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--vector-path", required=True)
    parser.add_argument("--target-layers", type=int, nargs="+",
                        default=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    parser.add_argument("--stress-concurrency", type=int, default=20,
                        help="Number of concurrent requests in stress test")
    parser.add_argument("--skip-stress", action="store_true",
                        help="Skip the stress test (faster)")
    args = parser.parse_args()

    if not os.path.exists(args.vector_path):
        print(f"ERROR: vector file not found: {args.vector_path}")
        sys.exit(1)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=120.0)

    print(f"Server: {args.base_url}")
    print(f"Model:  {args.model}")
    print(f"Vector: {os.path.abspath(args.vector_path)}")
    print(f"Layers: {args.target_layers}")

    try:
        models = await client.models.list()
        print(f"Models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"ERROR: Cannot connect: {e}")
        sys.exit(1)

    results = {}

    # Run sequential tests first (they set up state)
    results["1-Persistence"]     = await test_persistence(client, args.model, args.vector_path, args.target_layers)
    results["2-Scale=0"]         = await test_scale_zero(client, args.model, args.vector_path, args.target_layers)
    results["3-Concurrent-3"]    = await test_concurrent_3(client, args.model, args.vector_path, args.target_layers)

    if not args.skip_stress:
        results["4-Stress"]      = await test_stress_concurrent(client, args.model, args.vector_path,
                                                                args.target_layers, args.stress_concurrency)
    results["5-UserScenario"]    = await test_user_scenario(client, args.model, args.vector_path, args.target_layers)

    # Summary
    print_section("RESULTS")
    n_pass = 0
    n_fail = 0
    for name, ok in results.items():
        label = "PASS" if ok else "FAIL"
        print(f"  [{label}]  {name}")
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n  {n_pass} passed, {n_fail} failed out of {len(results)} tests")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
