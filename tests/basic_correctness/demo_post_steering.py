"""Integration test: server-level steering with POST /v1/steering.

A single self-contained script that:
1. Starts a vLLM server with --steer-vector-path (server-level steering)
2. Verifies GET /v1/steering returns the active config
3. Generates at scale=4.0 and records logprobs
4. POSTs scale=0.0 and verifies output changes (unsteered)
5. POSTs scale=4.0 back and verifies logprobs match step 3 exactly
6. Verifies per-request steer_vector_request is rejected
7. Shuts down the server

Usage:
    cd EasySteer/vllm-steer
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        tests/basic_correctness/demo_post_steering.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

# ── Configuration ──────────────────────────────────────────────────────
EASYSTEER_ROOT = Path(__file__).resolve().parents[2].parent  # EasySteer/
VECTOR_PATH = str(EASYSTEER_ROOT / "vectors" / "happy_diffmean.gguf")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LAYERS = list(range(10, 26))
PORT = 8019
BASE_URL = f"http://localhost:{PORT}"
MAX_TOKENS = 20
PROMPT = "Describe a rainy Monday morning."
LOGPROB_ATOL = 1e-4


# ── Helpers ────────────────────────────────────────────────────────────

def wait_for_server(timeout: float = 180.0) -> None:
    """Poll /v1/models until the server is ready."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{BASE_URL}/v1/models", timeout=2.0)
            if r.status_code == 200:
                elapsed = time.time() - t0
                print(f"  Server ready after {elapsed:.1f}s")
                return
        except httpx.ConnectError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"Server not ready after {timeout}s")


def generate(prompt: str) -> tuple[str, list[float]]:
    """Chat completion with logprobs. Returns (text, logprobs)."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 1,
    }
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=body, timeout=60.0)
    if r.status_code != 200:
        raise RuntimeError(f"Generate failed ({r.status_code}): {r.text[:500]}")
    data = r.json()
    choice = data["choices"][0]
    text = choice["message"]["content"]
    lps: list[float] = []
    if choice.get("logprobs") and choice["logprobs"].get("content"):
        for entry in choice["logprobs"]["content"]:
            lps.append(entry["logprob"])
    return text, lps


def generate_with_per_request_steering() -> httpx.Response:
    """Send a request with steer_vector_request — should be rejected."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "steer_vector_request": {
            "steer_vector_local_path": VECTOR_PATH,
            "scale": 1.0,
            "target_layers": TARGET_LAYERS[:3],
            "prefill_trigger_tokens": [-1],
            "generate_trigger_tokens": [-1],
        },
    }
    return httpx.post(f"{BASE_URL}/v1/chat/completions", json=body, timeout=30.0)


def get_steering() -> dict:
    r = httpx.get(f"{BASE_URL}/v1/steering", timeout=5.0)
    r.raise_for_status()
    return r.json()


def post_steering(scale: float) -> dict:
    r = httpx.post(f"{BASE_URL}/v1/steering", json={"scale": scale}, timeout=30.0)
    if r.status_code != 200:
        raise RuntimeError(
            f"POST /v1/steering failed ({r.status_code}):"
            f" {r.text[:500]}"
        )
    return r.json()


def start_server() -> subprocess.Popen:
    """Start vLLM with server-level steering and return the Popen handle."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--steer-vector-path", VECTOR_PATH,
        "--steer-scale", "4.0",
        "--steer-target-layers", *[str(layer) for layer in TARGET_LAYERS],
        "--steer-normalize",
        "--port", str(PORT),
        "--gpu-memory-utilization", "0.4",
        "--max-model-len", "512",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
    ]
    log_path = "/tmp/vllm_post_steering_test.log"
    log_file = open(log_path, "w")  # noqa: SIM115
    print(f"  Starting server (log: {log_path})")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # Pass through CUDA_VISIBLE_DEVICES from parent
        env={**os.environ},
    )
    # Store the log file handle so we can close it later
    proc._log_file = log_file  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    if hasattr(proc, "_log_file"):
        proc._log_file.close()  # type: ignore[attr-defined]


# ── Main test ──────────────────────────────────────────────────────────

def main() -> None:
    assert Path(VECTOR_PATH).exists(), f"Vector not found: {VECTOR_PATH}"
    results: list[tuple[str, bool]] = []

    print("=" * 60)
    print("Integration test: POST /v1/steering")
    print("=" * 60)

    proc = start_server()
    try:
        wait_for_server()

        # ── Step 1: Check initial config ──────────────────────────
        print("\n[Step 1] GET /v1/steering — check initial config")
        config = get_steering()
        print(f"  Config: {json.dumps(config)}")
        ok = config["active"] is True and config["scale"] == 4.0
        results.append(("Initial config correct", ok))
        print(f"  -> {'PASS' if ok else 'FAIL'}")

        # ── Step 2: Generate at scale=4.0 ─────────────────────────
        print("\n[Step 2] Generate at scale=4.0")
        text_s4, lps_s4 = generate(PROMPT)
        print(f"  Output: {text_s4[:100]}...")
        print(f"  Logprobs[:5]: {[f'{lp:.4f}' for lp in lps_s4[:5]]}")
        ok = len(lps_s4) > 0
        results.append(("Generate at scale=4.0 succeeds", ok))
        print(f"  -> {'PASS' if ok else 'FAIL'}")

        # ── Step 3: POST scale=0.0, verify output changes ────────
        print("\n[Step 3] POST scale=0.0, then generate")
        resp = post_steering(0.0)
        print(f"  POST response: {json.dumps(resp)}")
        text_s0, lps_s0 = generate(PROMPT)
        print(f"  Output: {text_s0[:100]}...")
        print(f"  Logprobs[:5]: {[f'{lp:.4f}' for lp in lps_s0[:5]]}")
        output_changed = text_s4 != text_s0
        results.append(("scale=0.0 produces different output", output_changed))
        status = "PASS" if output_changed else "FAIL"
        print(f"  Texts differ: {output_changed} -> {status}")

        # ── Step 4: POST scale=4.0, verify logprobs restored ─────
        print("\n[Step 4] POST scale=4.0 (restore), then generate")
        resp = post_steering(4.0)
        print(f"  POST response: {json.dumps(resp)}")
        text_s4b, lps_s4b = generate(PROMPT)
        print(f"  Output: {text_s4b[:100]}...")
        print(f"  Logprobs[:5]: {[f'{lp:.4f}' for lp in lps_s4b[:5]]}")

        tokens_match = text_s4 == text_s4b
        results.append(("Restored text matches original", tokens_match))
        tm_status = "PASS" if tokens_match else "FAIL"
        print(f"  Texts match: {tokens_match} -> {tm_status}")

        if lps_s4 and lps_s4b:
            max_diff = max(
                abs(a - b) for a, b in zip(lps_s4, lps_s4b)
            )
            lp_ok = max_diff <= LOGPROB_ATOL
            results.append((f"Restored logprob max_diff={max_diff:.6f}", lp_ok))
            lp_status = "PASS" if lp_ok else "FAIL"
            print(f"  Logprob max diff: {max_diff:.6f} -> {lp_status}")

        # ── Step 5: Per-request steering is rejected ──────────────
        print("\n[Step 5] Per-request steer_vector_request should be rejected")
        r = generate_with_per_request_steering()
        rejected = r.status_code != 200
        results.append(("Per-request steering rejected", rejected))
        print(f"  Status: {r.status_code} -> {'PASS' if rejected else 'FAIL'}")
        if rejected:
            print(f"  Error: {r.text[:200]}")

        # ── Step 6: Structural change is rejected ─────────────────
        print("\n[Step 6] Structural change (vector_path) should be rejected")
        r = httpx.post(
            f"{BASE_URL}/v1/steering",
            json={"vector_path": "/tmp/other.gguf"},
            timeout=10.0,
        )
        struct_rejected = r.status_code == 400
        results.append(("Structural change rejected", struct_rejected))
        print(f"  Status: {r.status_code} -> {'PASS' if struct_rejected else 'FAIL'}")
        if struct_rejected:
            print(f"  Error: {r.json().get('error', '')[:200]}")

    finally:
        print("\nStopping server...")
        stop_server(proc)
        print("Server stopped.")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(f"*** ALL {len(results)} CHECKS PASSED ***")
    else:
        n_fail = sum(1 for _, ok in results if not ok)
        print(f"*** {n_fail}/{len(results)} CHECKS FAILED ***")
        sys.exit(1)


if __name__ == "__main__":
    main()
