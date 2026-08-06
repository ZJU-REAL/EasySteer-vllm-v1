# Steer Vectors

This document shows how to use activation steering with vLLM: adding, replacing, or transforming hidden states at chosen layers and token positions, per request, without touching model weights.

Steering is **declaration-first**: an engine states which steering algorithms it will serve, and everything else — CUDA-graph integration, admission checks, error messages — follows from that declaration. See [Steer Vectors design](../design/steer_vectors.md) for the architecture.

## Enabling steering

Pass `enable_steer_vector=True` together with the workload declaration:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enable_steer_vector=True,
    steer_algorithms=["direct"],
)
```

`steer_algorithms` is required (unless an engine-default `steering_config` is given, which declares its own workload). It is the serving contract:

- Requests using algorithms outside the declared set are rejected at admission, on every engine, with an error naming the missing declaration.
- The engine derives its CUDA-graph integration from the declaration — declared workloads are always servable, so a request never fails because of an internal graph decision.
- `steer_algorithms="all"` allows every algorithm (useful for exploration and demo servers); the engine then runs steering in split graph mode.
- Requests carrying more than one vector must be declared with `steer_multi_vector=True`.

An unsteered request on a steering-enabled engine behaves exactly like stock vLLM — enabling steering does not change unsteered outputs.

## Describing steering: the spec

A steering configuration is three nested objects:

```python
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

spec = SteeringSpec(vectors=[VectorSpec(
    source="vectors/happy_diffmean.gguf",  # or data=<payload> for in-memory vectors
    algorithm="direct",
    scale=2.0,
    layers=list(range(10, 26)),
    apply=ApplySpec(phases=["prompt", "generation"]),
)])

outputs = llm.generate(prompts, sampling_params, steering=spec)
```

- **`SteeringSpec`** — the whole configuration: a list of vectors and a `conflict` policy (`"priority"`: first matching vector wins per position; `"sequential"`: all matching vectors apply in order).
- **`VectorSpec`** — one intervention: the payload (`source` file or in-memory `data`), the `algorithm`, a `scale`, the `layers` it applies to, optional `normalize=True` (rescale the steered hidden state back to its original norm), and its `apply` clause.
- **`ApplySpec`** — *where* the vector fires. `phases` (`"prompt"`: prefill tokens, correct across chunked prefill; `"generation"`: decode steps; or both) is the outer gate; six include selectors — three per phase, each named for it — and their six symmetric exclude twins operate within it:

    | include | exclude twin | matches |
    |---|---|---|
    | `prompt_tokens` | `exclude_prompt_tokens` | the given token ids, prompt occurrences only |
    | `prompt_positions` | `exclude_prompt_positions` | prompt positions: `0`, `1`, … or negative (`-1` = last prompt token); positive values past the prompt end clamp to the last prompt token (warned at admission) |
    | `prompt_window` | `exclude_prompt_window` | half-open `(start, stop)` over prompt positions; negative bounds and `stop=None` resolve from the prompt end (`(-5, None)` = last five prompt tokens) |
    | `generation_tokens` | `exclude_generation_tokens` | the given token ids, generated occurrences only |
    | `generation_positions` | `exclude_generation_positions` | exact 0-based decode steps |
    | `generation_window` | `exclude_generation_window` | half-open `(start, stop)` over 0-based decode steps |

    The include selectors select the **union** of their matches (with none set, the whole gated phases); the exclude selectors union and always subtract — where include and exclude overlap, the exclusion wins. One clause can therefore steer the prompt tail *and* the first decode steps: `ApplySpec(phases=["prompt", "generation"], prompt_window=(-4, None), generation_window=(0, 4))`.

`steering=` accepts one spec for the whole batch or a list with one entry (or `None`) per prompt — different requests in one batch can carry entirely different configurations, and each is applied only to its own request.

## Algorithms

| Algorithm | Intervention | Payload | Graph tiers |
|---|---|---|---|
| `direct` | add a vector: `h' = h + s·v` | `.gguf` or data | split, in-graph |
| `erase` | remove a direction (projection) | `.gguf` or data | split, in-graph |
| `replace` | replace the hidden state | `.gguf` or data | split, in-graph |
| `concept_replace` | swap one direction for another | data | split, in-graph |
| `loreft` | learned low-rank edit (ReFT) | data (`from_pyreft`) | split; in-graph when rank ≤ `steer_graph_max_rank` |
| `lm_steer` | low-rank projector pair before the LM head | data (`from_lm_steer`) | split; in-graph when rank ≤ `steer_graph_max_rank` |
| `linear` | full affine map `h' = W·h + b` | data (`from_linear_transport`) | split only |
| `moe_router` | (de)activate MoE experts at the gate | inline config or file | split; in-graph for inline activate/deactivate configs |

"In-graph" and "split" are the two steering graph tiers — see [graph tiers](#graph-tiers). Algorithms marked "split only" or with a condition resolve `auto` to split when declared by name; the table's conditions come from the same source of truth the engine enforces (`vllm.steer_vectors.algorithms.steering_execution_modes` and `graph_condition`).

## Graph tiers

Steering integrates with compiled execution in one of two ways, fixed at engine construction:

- **`in_graph`** — the steering math lives *inside* the captured CUDA graphs as a data-driven kernel; vLLM keeps its full cudagraphs and unsteered batches run at near-native speed. The kernel is specialized to the declaration: only the declared algorithms' kernel families are compiled in, so a `steer_algorithms=["direct"]` engine carries none of the other families' compute. Only single-vector configs of graph-family algorithms are admissible, with per-payload conditions (e.g. the rank cap).
- **`split`** — the steering ops become compilation splitting ops: the compiled graph is partitioned at every steered layer and steering runs eagerly between the segments. Every algorithm and multi-vector composition is supported, at roughly half the decode throughput of in-graph steering.

You normally never choose: `steer_graph_mode="auto"` (the default) resolves from the declaration — in-graph when every declared algorithm is unconditionally graph-safe, split otherwise — and logs its reasoning at boot:

```text
Steering graph mode resolved to 'split': declared algorithm(s) ['lm_steer'] are not
unconditionally graph-safe ('lm_steer': payload rank must be <= steer_graph_max_rank); ...
```

Experts can override with `steer_graph_mode="in_graph"` or `"split"`. The declaration still bounds what may run: an override that can never serve the declared workload is a boot error, and an in-graph override with conditionally-safe declarations boots with a warning and re-checks each payload at request time. `in_graph` requires compiled execution (`enforce_eager=False`).

## Engine configuration reference

| Argument | Default | Meaning |
|---|---|---|
| `enable_steer_vector` | `False` | Enable the steering subsystem |
| `steer_algorithms` | — | **Required.** Declared workload: algorithm names or `"all"` |
| `steer_multi_vector` | `False` | Declare multi-vector requests |
| `max_steer_vectors` | `min(256, max_num_seqs)` | Concurrent distinct configurations (slot capacity). A scheduling constraint like `max_loras`: additional differently-configured requests wait in the queue until a slot frees; identically-configured requests share a slot. An engine-default `steering_config` occupies one slot. The default never throttles below realistic concurrency; lower it to bound per-step config variety, raise it only past 256 concurrent distinct configs. |
| `steer_graph_mode` | `"auto"` | Graph tier: `auto` / `in_graph` / `split` (expert) |
| `steer_graph_max_rank` | `32` | Rank capacity of in-graph low-rank buffers |
| `steer_vector_dtype` | `"auto"` | Vector dtype (defaults to model dtype) |
| `steer_require_preload` | `False` | Reject configs referencing vectors that were not preloaded |
| `steering_config` | `None` | Engine-default steering (JSON spec or path); disables per-request steering |

## Online serving

All flags are available on `vllm serve`:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --enable-steer-vector --steer-algorithms direct
```

Per-request steering is a `steering` field on `/v1/completions` and `/v1/chat/completions` (for OpenAI client libraries, pass it via `extra_body`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Tell me about your day."}],
    extra_body={"steering": {
        "vectors": [{
            "source": "vectors/happy_diffmean.gguf",
            "algorithm": "direct",
            "scale": 2.0,
            "layers": list(range(10, 26)),
            "apply": {"phases": ["prompt", "generation"]},
        }]
    }},
)
```

Management endpoints:

- `GET /v1/steering` — the active engine-default steering config, if any.
- `POST /v1/steering` — replace the engine-default config at runtime (`{"spec": <SteeringSpec JSON>}`); the prefix cache is reset so KV steered under the old config is never reused. Only available when the server was started with `--steering-config`.
- `GET /v1/steering/vectors` — list preloaded vector paths.
- `POST /v1/steering/vectors` — preload vectors (`{"paths": [...], "algorithm": "direct"}`). With `--steer-require-preload`, only preloaded vectors are accepted in per-request configs — combined with the declaration this makes every admissible request known at launch time.

For production serving, declare the exact workload and preload its vectors: undeclared algorithms, undeclared multi-vector configs, and unknown vector paths are all rejected at the frontend before reaching the engine core.

## Interaction with other features

- **Prefix caching** works with steering: cached KV blocks are keyed by the steering configuration fingerprint, so requests only reuse blocks computed under an identical config, steered and unsteered requests never share blocks, and position-sensitive specs re-key on prompt length. Warm and cold runs of the same steered request are byte-identical.
- **CUDA graphs** are kept (see [graph tiers](#graph-tiers)); per-request config differences ride through captured graphs as data, so batches freely mix different configs and unsteered traffic.
- **Chunked prefill** is supported; `"prompt"`-phase clauses cover every prompt token across chunks.
- **Hidden-state capture** coexists with steering on the same engine; capture-active batches dispatch eagerly per batch and everything else keeps its graphs.
