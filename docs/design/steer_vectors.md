# Steer Vectors — Design

How activation steering integrates with compiled execution, CUDA graphs, continuous batching, and prefix caching. The user-facing guide is [Steer Vectors](../features/steer_vectors.md).

## Three kinds of variability, three mechanisms

Steering makes execution *vary* — per request, per token, per deployment. The design assigns each kind of variability to the cheapest mechanism that can express it:

| Variability | Mechanism | Decided |
|---|---|---|
| Which config steers which token | **data** — per-token row indices into persistent tables | every step, host-side |
| Which algorithms can exist in the engine | **compilation** — one of two graph tiers | engine construction |
| Whether activations are observed (capture) | **dispatch** — per-batch eager escape | per batch, automatically |

The consequences: batches freely mix steering configurations and unsteered traffic without recapturing anything; the only boot-time decision is the graph tier, and it is derived from the user's declaration; and capture costs exactly the batches that observe.

## The two graph tiers

### `in_graph` (Tier 1)

The steering kernel is captured *inside* vLLM's CUDA graphs. It is pure tensor math over persistent buffers organized as a closed set of **kernel families**, each with bounded per-slot parameters:

| Family | Delta | Tables (per slot) |
|---|---|---|
| `additive` | `V[row]` | `V(h)` |
| `projection` | `(x·B[row])·C[row]` | `B(h)`, `C(h)` |
| `lowrank` | `(x·A[row]+b[row])·Rout[row]ᵀ` | `A(h,r)`, `Rout(h,r)`, `b(r)` with `r ≤ steer_graph_max_rank` |
| `replace` | `mask·(V[row]−x)` | `V(h)` |

Before each step the host fills a per-token `token_rows` index buffer and the step masks; the captured kernel applies every family's delta to every token through its row. **Row 0 of every table is zero**, so unsteered and padding tokens receive an exact-zero delta — an unsteered request co-batched with steered ones is bit-identical to a vanilla engine's output. Buffer addresses are baked into the captured graphs, so tables are allocated before model wrap and sized at boot (`max_steer_vectors` slots, `steer_graph_max_rank` rank capacity).

The closed schema is the point: the kernel is compiled with exactly the families the declared workload can use (`steer_algorithms=["direct"]` compiles only `additive`; an undeclared family can never receive a payload, so it contributes no compute or table memory), and admission can decide *exactly* what fits. Within a compiled family, idle rows cost only zero-row reads. A general `h' = W·h + b` (the `linear` algorithm) would need an `(h × h)` table per slot and a per-token batched matmul even when idle — that is not "one more family", it breaks the tier's premise, which is why `linear` is split-only. Bounded matrix interventions belong in `lowrank` (that is exactly what LoReFT is).

### `split` (Tier 2)

The steering ops (`vllm::steer_apply`, `vllm::steer_moe_gate`) are registered as compilation **splitting ops** — the same mechanism vLLM uses for attention. The compiled artifact is partitioned at every steered layer; between the graph segments the op runs as ordinary eager Python, so any algorithm, any rank, and multi-vector composition all work. The cost is compile-time and structural: full-cudagraph replay is impossible with splits, so `cudagraph_mode` is downgraded to piecewise graphs.

Both tiers apply steering identically from the user's perspective; they differ in coverage and throughput. The tier is fixed at construction because it determines *what gets compiled and captured* — after boot, the other tier's machinery does not exist in the engine.

## Declaration and resolution

Enabling steering requires a workload declaration (`steer_algorithms`, plus `steer_multi_vector` for multi-vector configs). The declaration serves two roles:

1. **Serving contract.** At admission — on every engine, in front of the engine core — a request using an undeclared algorithm or undeclared multi-vector composition is rejected. Behavior therefore never depends on the resolved tier, and the failure mode is a self-explanatory error, not a graph-internals lesson.
2. **Resolution evidence.** `steer_graph_mode="auto"` picks the tier by a ladder ordered by quality of evidence:
    - *Explicit mode* (expert override) wins; boot cross-checks reject overrides that can never serve the declaration and warn on conditional ones.
    - *Concrete evidence*: an engine-default `steering_config` is judged exactly — the same `graph_request_problem` check used at admission runs over the actual payloads.
    - *Names*: resolved pessimistically. Only unconditionally graph-safe algorithms (no rank cap, no config-form condition) keep the in-graph tier; anything conditional resolves to split, because without a payload the engine must assume the general case. A names-only declaration therefore never produces an engine that rejects its declared workload.
    - *No compiled execution* (e.g. `enforce_eager=True`): split, which degenerates to plain eager steering ops — the in-graph tier's restrictions would buy nothing without graphs.

There is exactly **one admissibility implementation** — `graph_request_problem` — shared by the frontend, the worker (defense in depth), and boot-time resolution, so boot and admission cannot disagree. The capability table (`steering_execution_modes`, `graph_condition`) is derived from each algorithm class's declared `graph_family`, never hand-maintained.

`in_graph` requires compiled execution; the combination with eager engines is rejected at construction. (`VLLM_STEER_EAGER_IN_GRAPH=1` is a test-only escape: eager execution is the only cross-boot byte-deterministic mode, so the in-graph kernel path is byte-golden validated there.)

## Routing and triggers

Per-request configurations are installed into **slots** (`max_steer_vectors` concurrent distinct configs), refcounted by config fingerprint so identical configs share a slot. Each scheduler step, trigger resolution runs host-side: for every token in the batch, its request's apply clauses (phases, positions, generation windows, token filters, exclusions) are evaluated against the request's positional state, producing the row index (in-graph) or the apply plan (split) for that token. Resolution is one numpy pass grouped by slot — clauses match host-known geometry, so each slot's clauses are evaluated only over that slot's own token rows and the matched positions ship to the device in a single copy per step. Per-step resolution cost therefore scales with the batch's tokens, not with the number of distinct live configurations (in-graph mask writes are likewise batched per steered layer, not per config). Positions are exact across chunked prefill and one-token prompts; `conflict` decides how multiple matching vectors compose (`priority` first-match or `sequential` stacking).

## Prefix caching

KV cache blocks are keyed by the **steering configuration fingerprint** in addition to content: requests only reuse blocks computed under an identical config; steered and unsteered traffic never share blocks; length-sensitive clauses (negative positions, generation windows) fold the prompt length into the key so the same spec on a different-length prompt re-keys. This makes warm and cold runs of an identical steered request byte-identical *by construction* — steered rows inside a cache hit were computed with the same steering and need no recomputation. Replacing the engine-default config at runtime (`POST /v1/steering`) resets the prefix cache so KV steered under the old config is never reused.

## Determinism boundary

The steering machinery itself is deterministic and adds no numeric noise: row 0 deltas are exactly zero, and a zero-scale steered batch is byte-identical to an unsteered one wherever the engine itself is deterministic. What steering *inherits* is vLLM's own behavior: compiled engines are not batch-deterministic at larger batch shapes (identical unsteered runs can differ from themselves), and outputs are not stable across recompilation or batch-geometry changes. Byte-level oracles for steering are therefore valid on eager engines and small batches; at scale, correctness is verified mechanically (trace attribution, co-batched same-geometry comparisons) rather than by byte equality. This was verified explicitly: at batch 32 on a compiled engine, unsteered runs self-diverge with the same magnitude and request population as zero-scale steered runs.

## Capture coexistence

Hidden-state capture hooks attach on every engine. Their bodies fold out of compiled artifacts (`torch.compiler.is_compiling()` guard), and when a capture stream is enabled the batch dispatches through the raw eager forward (`skip_compiled`), where hooks fire natively — steering included, since the eager forward contains the same steering ops. Idle batches keep their graphs. Capture requests carry a `cache_salt` so prefix-cache hits never skip the recomputation capture needs; unsalted hits fail explicitly at fetch.
