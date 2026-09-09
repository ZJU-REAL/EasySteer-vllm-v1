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
- With the default `steer_graph_mode="auto"`, the engine selects a CUDA-graph tier that supports the declared algorithms and multi-vector setting. Explicit graph overrides also impose the payload conditions described below.
- `steer_algorithms="all"` allows every algorithm (useful for exploration and demo servers); the engine then runs steering in split graph mode.
- Requests carrying more than one vector must be declared with `steer_multi_vector=True`.

An unsteered request receives no steering transformation; it follows the ordinary model forward path in the selected engine configuration.

## Describing steering: the spec

A steering configuration is three nested objects:

```python
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

spec = SteeringSpec(vectors=[VectorSpec(
    source="vectors/happy_diffmean.gguf",  # or data=<payload> for in-memory vectors
    algorithm="direct",
    scale=2.0,
    layers=list(range(10, 26)),
    apply=ApplySpec(prompt="all", generation="all"),
)])

outputs = llm.generate(prompts, sampling_params, steering=spec)
```

- **`SteeringSpec`** — the whole configuration: a list of vectors and a `conflict` policy (`"priority"`: first matching vector wins per position; `"sequential"`: all matching vectors apply in order).
- **`VectorSpec`** — one intervention: the payload (`source` file or in-memory `data`), the `algorithm`, a `scale`, the `layers` it applies to, and its `apply` clause. `normalize=True` rescales the steered hidden state back to its original norm for `direct`, `erase`, `replace`, and `concept_replace`; leave it `False` for the other algorithms.
- **`ApplySpec`** — *where* the vector fires. Each phase is selected independently and only by what you name: `prompt="all"` selects every prefill token (correct across chunked prefill), `generation="all"` every decode step, and six narrower include selectors — three per phase, each named for it — plus their six symmetric exclude twins refine the selection. A phase you don't mention is untouched:

    | include | exclude twin | matches |
    | --- | --- | --- |
    | `prompt_tokens` | `exclude_prompt_tokens` | the given token ids, prompt occurrences only |
    | `prompt_positions` | `exclude_prompt_positions` | prompt positions: `0`, `1`, … or negative (`-1` = last prompt token); positive values past the prompt end clamp to the last prompt token (warned at admission) |
    | `prompt_window` | `exclude_prompt_window` | half-open `(start, stop)` over prompt positions; negative bounds and `stop=None` resolve from the prompt end (`(-5, None)` = last five prompt tokens) |
    | `generation_tokens` | `exclude_generation_tokens` | the given token ids, generated occurrences only |
    | `generation_positions` | `exclude_generation_positions` | exact 0-based decode steps |
    | `generation_window` | `exclude_generation_window` | half-open `(start, stop)` over 0-based decode steps |

    The include selectors (with `prompt="all"` / `generation="all"` as the widest of each phase) select the **union** of their matches; the exclude selectors union and always subtract — where include and exclude overlap, the exclusion wins. Mixed granularities compose in one clause: `ApplySpec(prompt_positions=[-1], generation="all")` steers the last prompt token plus every generated token, and `ApplySpec(prompt_window=(-4, None), generation_window=(0, 4))` the prompt tail plus the first decode steps. A clause that selects nothing is rejected, as is the removed `phases` key.

`steering=` accepts one spec for the whole batch or a list with one entry per
prompt. An omitted value or `None` inherits the current default; `False` disables
steering for that request; a spec completely overrides the default. Each request
retains the configuration and weights resolved when it is admitted.

Set an initial default with `LLM(steering_config=...)`. Use
`llm.set_default_steering(spec)` to replace it, `llm.set_default_steering(None)` to
clear it, and `llm.get_default_steering()` to inspect it. These operations affect
new requests without resetting the prefix cache. A default uses the same request
execution path and slot cache as explicit steering.

## Algorithms

| Algorithm | Intervention | Payload | Graph tiers |
| --- | --- | --- | --- |
| `direct` | add a vector: `h' = h + s·v` | `.gguf` or data | split, in-graph |
| `erase` | remove a direction (projection) | `.gguf` or data | split, in-graph |
| `replace` | replace the hidden state | `.gguf` or data | split, in-graph |
| `concept_replace` | swap one direction for another | data | split, in-graph |
| `loreft` | learned low-rank edit (ReFT) | data (`from_pyreft`) | split; in-graph when rank ≤ `steer_graph_max_rank` |
| `lm_steer` | low-rank projector pair at the selected decoder layers | data (`from_lm_steer`) | split; in-graph when rank ≤ `steer_graph_max_rank` |
| `linear` | full affine map `h' = W·h + b` | data (`from_linear_transport`) | split only |
| `moe_router` | adjust MoE expert routing at the gate | `RouterConfig`, inline config, or file | split; in-graph for `activate`, `deactivate`, `soft`, and `soft_topk` |

Algorithms select their target component: `moe_router` edits `router_logits` at
an accessible gate; the other algorithms edit decoder `hidden_states`. Steering
and capture share component discovery and availability checks.

Native source files and in-memory data are normalized to canonical payloads
before execution. Algorithms consume their payload kind rather than implement
separate file readers. Use `easysteer.vectors.load(path, format=...)` for an
explicit client-side format adapter, then pass the result as `VectorSpec.data`.
Worker processes receive the same content snapshot used for admission and cache
identity.

For router files and `RouterConfig` data, explicit `params.mode`, `lambda`, and
`topk` override payload values. `params.expert_ids` is only valid for inline
configuration without `source` or `data`; otherwise expert IDs belong in the
payload's per-layer configuration.

"In-graph" and "split" are the two steering graph tiers — see [graph tiers](#graph-tiers). Algorithms marked "split only" or with a condition resolve `auto` to split when declared by name; the table's conditions come from the same source of truth the engine enforces (`vllm.model_hooks.steering.algorithms.steering_execution_modes` and `graph_problem`). For `moe_router`, `soft_random` requires split execution. File-backed and inline router configs use the same parsed per-layer payloads for admission and execution.

## Graph tiers

Steering integrates with compiled execution in one of two ways, fixed at engine construction:

- **`in_graph`** — the steering math lives *inside* the captured CUDA graphs as a data-driven kernel; vLLM keeps its full cudagraphs and unsteered batches run at near-native speed. The kernel is specialized to the declaration: only the declared algorithms' kernel families are compiled in, so a `steer_algorithms=["direct"]` engine carries none of the other families' compute. Only single-vector configs of graph-family algorithms are admissible, with per-payload conditions (e.g. the rank cap).
- **`split`** — the steering ops become compilation splitting ops: the compiled graph is partitioned at every steered layer and steering runs eagerly between the segments. Every algorithm and multi-vector composition is supported. Throughput depends on the model and workload; benchmark the two tiers for the configuration being deployed.

`steer_graph_mode="auto"` (the default) logs the selected tier and reason at boot.
An engine-default `steering_config` is evaluated using its actual payloads.
With a names-only declaration, auto selects in-graph when every declared
algorithm has no payload restrictions; rank-limited algorithms, `moe_router`,
and multi-vector workloads select split. Engines without compiled execution
also select split.

Experts can override with `steer_graph_mode="in_graph"` or `"split"`. The declaration still bounds what may run: an override that can never serve the declared workload is a boot error, and an in-graph override with conditionally-safe declarations boots with a warning and re-checks each payload at request time. `in_graph` requires compiled execution (`enforce_eager=False`).

## Engine configuration reference

| Argument | Default | Meaning |
| --- | --- | --- |
| `enable_steer_vector` | `False` | Enable the steering subsystem |
| `steer_algorithms` | — | Declared workload: algorithm names or `"all"`; required unless inferred from `steering_config` |
| `steer_multi_vector` | `False` | Declare multi-vector requests; implied by `"all"` or a multi-vector engine default |
| `max_steer_vectors` | `min(256, max_num_seqs)` | Concurrent distinct configurations (slot capacity). A scheduling constraint like `max_loras`: additional differently-configured requests wait in the queue until a slot frees; identically-configured requests share a slot. Defaults do not reserve a separate slot. |
| `steer_graph_mode` | `"auto"` | Graph tier: `auto` / `in_graph` / `split` (expert) |
| `steer_graph_max_rank` | `32` | Rank capacity of in-graph low-rank buffers |
| `steer_vector_dtype` | `"auto"` | Vector dtype (defaults to model dtype) |
| `steer_require_preload` | `False` | Reject configs referencing vectors that were not preloaded |
| `steering_config` | `None` | Initial default steering (JSON spec or path), used when a request omits a spec |

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
            "apply": {"prompt": "all", "generation": "all"},
        }]
    }},
)
```

Management endpoints:

- `GET /v1/steering` — the active engine-default steering config, if any.
- `POST /v1/steering` — set or replace the default (`{"spec": <SteeringSpec JSON>}`), or clear it (`{"spec": null}`). Updates affect new requests and preserve prefix-cache entries. Steering must be enabled and the algorithms declared; a startup default is optional.
- `GET /v1/steering/vectors` — list preloaded vector paths.
- `POST /v1/steering/vectors` — preload vectors (`{"paths": [...], "algorithm": "direct", "params": {...}}`, with optional `params`). With `--steer-require-preload`, the request's algorithm and resolved payload must have been preloaded. Router parameters that modify file contents must match; changing scale, layer subset or selection reuses the same preload. The path refers to a file on the server.

For production serving, declare the exact workload and preload its vectors: undeclared algorithms, undeclared multi-vector configs, and unknown vector paths are all rejected at the frontend before reaching the engine core.

Generation requests use `"steering": false` to bypass the default, `null` or
omission to inherit it, and a spec to override it. These are the same choices as
the Python API.
Runtime default updates require `--api-server-count=1`; with multiple API
frontends use startup defaults or explicit request specs. Status responses omit
in-memory tensor bytes and report their payload kinds and content hashes.
For Python tensor payloads sent over HTTP, convert `payload.to_wire()` tensor bytes
to base64 strings before placing the dictionary in a vector's `data` field.

## Interaction with other features

- **Prefix caching** works with steering: cached KV blocks are keyed by the steering configuration fingerprint, so requests only reuse blocks computed under an identical config, steered and unsteered requests never share blocks, and position-sensitive specs re-key on prompt length.
- **CUDA graphs** are kept (see [graph tiers](#graph-tiers)); per-request config differences ride through captured graphs as data, so batches freely mix different configs and unsteered traffic.
- **Chunked prefill** is supported; `"prompt"`-phase clauses cover every prompt token across chunks.
- **Beam search** rejects effective steering because its changing prompt/generation boundary does not preserve selection semantics. Use `steering=False` (HTTP `false`) to bypass a default when using beam search.
- **Hidden-state capture** coexists with steering and shares its component discovery. Steps without selected rows keep ordinary graph dispatch; eligible steps use a separate FULL capture graph, with eager execution for other capture steps (see [capture coexistence](../design/steer_vectors.md#capture-coexistence)). Request admission bypasses prefix-cache reads when the effective capture selection needs otherwise skipped prompt rows; cache writes remain enabled.
