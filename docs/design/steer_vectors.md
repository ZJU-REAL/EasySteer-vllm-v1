# Steer Vectors — Design

How activation steering integrates with compiled execution, CUDA graphs, continuous batching, and prefix caching. The user-facing guide is [Steer Vectors](../features/steer_vectors.md).

## Three kinds of variability, three mechanisms

Steering makes execution *vary* — per request, per token, per deployment. The design assigns each kind of variability to the cheapest mechanism that can express it:

| Variability | Mechanism | Decided |
| --- | --- | --- |
| Which config steers which token | **data** — per-token row indices into persistent tables | every step, host-side |
| Which algorithms can exist in the engine | **compilation** — one of two graph tiers | engine construction |
| Whether activations are observed (capture) | **dispatch** — ordinary graph, separate capture graph, or eager forward | per batch, automatically |

Batches can mix steering configurations and unsteered traffic without recapturing graphs. The workload declaration determines the graph tier and kernel families at construction; capture changes dispatch only while a stream is active.

## Component contracts

Steering and capture are independent consumers of the shared infrastructure in
`vllm/model_hooks/`. `selection/` owns selection specs, token matching and batch
geometry; `components/` owns model discovery, hook targets and output adapters.
The public entry points remain `vllm.steer_vectors` and `vllm.capture`.

The component descriptors live in `model_hooks/components/registry.py`.
Each descriptor supplies layer discovery, hook-target resolution, output adapters,
and availability checks. `hidden_states` targets decoder outputs;
`attention_heads` targets standard decoder attention outputs before their output
projection; `router_logits` targets an accessible MoE gate. A fused router that
bypasses the gate module is unavailable to both consumers.

Discovery uses attention/Mamba and MoE interfaces plus indexed decoder-stack
contracts, without model-class lists. Residual-stream blocks can supplement an
identified stack, including pure MLP blocks in hybrid models. Direct expert-kernel
implementations expose their expert count, top-k and gate/router module. Gates
still use the `gate`/`router` attribute convention because vLLM has no shared gate
accessor; new layouts outside these contracts require an explicit adaptation.

`discover_components` resolves one `ModelComponents` directory for the model.
Steering and capture consume its `ComponentTarget` records, which contain the
global layer index, module name and usable hook target. Attention targets also
record the query head count and value-output head size to determine the actual
component width. Controllers are indexed directly by component and layer.
Each controller declares its graph masks and
initializes its component tables; graph state uses this shared interface without
testing concrete controller classes.

Steering controllers live under `model_hooks/steering/controllers/`, with one
module each for hidden states and router logits. A component's eager and graph
execution stay together. Attention head outputs reuse the hidden-state
controller and additive kernel with their own component width. The shared
controller lifecycle and controller index live in `base.py` and `manager.py`.
Graph eligibility, persistent state and tensor
kernels live in `model_hooks/steering/graph/`.

Algorithm capabilities declare the target component, so controller installation
and payload distribution use the same lookup. The adapters preserve component
semantics: decoder capture includes the residual stream when present, while the
second output of a gate projection is bias, not a residual. Algorithm math and
`graph_lower` remain on the algorithm classes; component discovery does not
determine an algorithm's graph support.

Only components named by the declared algorithms receive steering controllers.
Capture continues to use the complete component directory. The same declaration
selects the split-tier custom ops and participates in the compilation cache key.

## Payload loading

`model_hooks/steering/loading.py` resolves native GGUF, concept-pair and router files into the same
canonical payloads accepted by `VectorSpec.data`. An algorithm declares its
payload kind and consumes materialized layer values; it does not own a file
reader. Third-party checkpoints are converted explicitly by
`easysteer.vectors.load(path, format=...)` or its convenience adapters.

`input_validation.py` shares the source/data/params rules between `VectorSpec`
and the loader. It performs authoring checks without reading files. Conversion
produces a `SteeringRequest` containing an ordered list of `ResolvedVector`
entries, each with one canonical `payload` and its application fields. A single
vector uses the same structure as a composition.

Admission fixes content identity and owns a metadata snapshot before sending the
request to workers. The native file cache reuses unchanged snapshots and reloads
changed files. Workers materialize the validated wire payload without reopening
its original path. Layer selection, shape validation and resident deduplication
therefore use the same rules for file-backed and in-memory inputs.

The worker's `PayloadCache` holds materialized per-layer payload dictionaries,
keyed by content, in `model_hooks/steering/payload_cache.py`.
Broadcast payloads share their tensors across target-layer combinations; each
request gets a lightweight layer mapping. Algorithms using the same content
share materialized values. Application scale, selectors
and normalization belong to configuration slots rather than the payload cache.

The frontend validates component widths, each vector's effective target layers
and router top-k against an inventory returned by the loaded workers. Pipeline
stages contribute their global layer indices to that inventory. Invalid requests
are rejected before worker admission, including a composition with one invalid
vector. Preloads also validate component widths before the worker RPC. The cache
retains a defensive shape check before materialization. Canonical payloads
validate matrix and bias relationships when constructed. Graph tables may pad
low-rank axes to their configured capacity; component widths must match exactly.

## The two graph tiers

### `in_graph` (Tier 1)

The steering kernel is captured *inside* vLLM's CUDA graphs. It is pure tensor math over persistent buffers organized as a closed set of **kernel families**, each with bounded per-slot parameters:

| Family | Delta | Tables (per slot) |
| --- | --- | --- |
| `additive` | `V[row]` | `V(h)` |
| `projection` | `(x·B[row])·C[row]` | `B(h)`, `C(h)` |
| `lowrank` | `(x·A[row]+b[row])·Rout[row]ᵀ` | `A(h,r)`, `Rout(h,r)`, `b(r)` with `r ≤ steer_graph_max_rank` |
| `replace` | `mask·(V[row]−x)` | `V(h)` |
| `moe_gate` | expert activation/deactivation or soft logit adjustment | expert masks, mode, epsilon, strength, top-k |

Before each step the host fills a per-token `token_rows` index buffer and the step masks; the captured kernel applies every family's delta to every token through its row. **Row 0 of every table is zero**, so unsteered and padding tokens receive an exact-zero steering delta. Buffer addresses are baked into the captured graphs, so tables are allocated before model wrap and sized at boot (`max_steer_vectors` slots, `steer_graph_max_rank` rank capacity).

Only declared families allocate tables and participate in the compiled kernel. For example, `steer_algorithms=["direct"]` uses the additive kernel, which skips inactive rows and computes norms only for rows requesting normalization. Other families may still perform work on idle rows. The `linear` algorithm requires a full hidden-by-hidden matrix and uses the split tier; bounded matrix interventions such as LoReFT use `lowrank`.

For `moe_router`, file-backed and inline configurations are parsed into the same
per-layer snapshot before admission. That snapshot determines the fingerprint,
worker payload and graph eligibility. `activate`, `deactivate`, `soft`, and
`soft_topk` have graph kernels; `soft_random` uses split execution for per-token
random expert selection.

Masks share one allocation and are cleared together over the current padded token span. Unused family masks are omitted. Table and mask addresses remain stable across requests. The compilation cache key describes this layout and its dimensions; vector contents, scales, selectors and layer targets update table values without changing that key. The separate KV-cache fingerprint retains the complete steering configuration.

### `split` (Tier 2)

The steering ops (`vllm::steer_apply`, `vllm::steer_moe_gate`) required by the declared components are registered as compilation **splitting ops** — the same mechanism vLLM uses for attention. The compiled artifact is partitioned at every steered layer; between the graph segments the op runs as ordinary eager Python, so any algorithm, any rank, and multi-vector composition all work. The cost is compile-time and structural: full-cudagraph replay is impossible with splits, so `cudagraph_mode` is downgraded to piecewise graphs.

Both tiers apply steering identically from the user's perspective; they differ in coverage and throughput. The tier is fixed at construction because it determines *what gets compiled and captured* — after boot, the other tier's machinery does not exist in the engine.

## Declaration and resolution

Enabling steering requires a workload declaration (`steer_algorithms`, plus `steer_multi_vector` for multi-vector configs). The declaration serves two roles:

1. **Serving contract.** At admission — on every engine, in front of the engine core — a request using an undeclared algorithm or undeclared multi-vector composition is rejected. Behavior therefore never depends on the resolved tier, and the failure mode is a self-explanatory error, not a graph-internals lesson.
2. **Resolution evidence.** `steer_graph_mode="auto"` picks the tier by a ladder ordered by quality of evidence:
    - *Explicit mode* (expert override) wins; boot cross-checks reject overrides that can never serve the declaration and warn on conditional ones.
    - *No compiled execution* (e.g. `enforce_eager=True`): auto selects split, which degenerates to plain eager steering ops.
    - *Concrete evidence*: an engine-default `steering_config` is judged exactly — the same `graph_request_problem` check used at admission runs over the actual payloads.
    - *Names*: only algorithms with no payload restrictions keep the in-graph tier. Rank-limited algorithms, `moe_router` (which includes `soft_random`), and multi-vector declarations select split so every declared variant can be served.

`resolve_graph_mode` supplies the resolved tier and its reason at boot.
`graph_request_problem` is shared by frontend admission, worker admission and
engine-default resolution; it delegates algorithm conditions to `graph_problem`.
The capability table (`steering_execution_modes`) and these conditions derive
from each algorithm's `graph_family`, `graph_payload_problem`, and family buffer
schema. Name-only resolution and concrete-payload admission therefore use the
same rules.

`in_graph` requires compiled execution; the combination with eager engines is rejected at construction. `VLLM_STEER_EAGER_IN_GRAPH=1` is a test-only escape that exercises the in-graph kernel path without graph capture.

## Routing and triggers

The request frontend resolves defaults before admission. Omitted or `None`
steering inherits the current default, `False` selects no intervention, and a
spec supplies a complete override. The resolved request carries an immutable
weight snapshot; default updates affect only requests admitted afterward.
`defaults.py` owns this resolution and startup snapshot construction. The
management HTTP router lives in `entrypoints/serve/steering/api_router.py`.
Workers and schedulers route effective requests without a server-specific slot
or forward-context fallback. Retained payloads and inactive configurations use
the ordinary cache and capacity rules.

Per-request configurations are installed into **slots** (`max_steer_vectors` concurrent distinct configs), refcounted by config fingerprint so identical configs share a slot. The capacity is a scheduler constraint (mirroring `max_loras`, keyed by the same fingerprint the worker allocates by): a waiting request whose configuration cannot get a slot is deferred until a running configuration releases one. Each scheduler step, trigger resolution runs host-side: for every token in the batch, its request's apply clauses (unioned phase-scoped include selectors minus their exclude twins, with no separate phase gate) are evaluated against the request's positional state, producing the row index (in-graph) or the apply plan (split) for that token. Resolution is a numpy pass grouped by slot: each slot's clauses are evaluated over its own token rows, and matched positions are copied to the device. Positions retain their prompt/decode meaning across ordinary chunked prefill and one-token prompts; `conflict` decides how multiple matching vectors compose (`priority` first-match, `sequential` stacking, or `error` on overlap).

The runner builds `BatchGeometry` from its input batch through
`gpu/model_hook_utils.py`. The shared types live in
`model_hooks/selection/batch.py`; each geometry caches a complete `BatchView`
per device for capture selection. Split hooks receive active slots and resolved
positions through `ForwardContext`. Capture receives the same geometry directly,
without constructing steering routing fields. Token-id filters take a cached
host snapshot when needed, while positional filters use existing host metadata.
Adaptive verification uses actual device-produced request boundaries rather
than scheduler estimates.

For compositions, conflict resolution runs on the host once per distinct ordered
intervention group per step; hooks on layers sharing that group reuse its resolved
positions. Groups on different target layers do not compete with each other.
Known prompt conflicts are rejected at admission. Generation-dependent conflicts
finish only the affected requests with an error; other batch members continue.

Runner shutdown and reattachment close steering state idempotently, unregister
controllers, remove hooks and release configuration, payload and graph ownership.

## Prefix caching

Multi-vector `conflict="error"` requests do not publish new prefix-cache blocks.
Their runtime validation can fail after vLLM would ordinarily publish scheduled
blocks, and asynchronous mutable model states cannot safely be committed later
without a separate snapshot protocol. Cache reads remain available; single-vector,
`priority` and `sequential` behavior is unchanged. Multi-vector `error` mode with
KV transfer is rejected at admission because exported rows cannot be retracted.

KV cache blocks are keyed by the **effective request's steering fingerprint** in
addition to content: requests only reuse blocks computed under an identical
config; steered and unsteered traffic never share blocks; length-sensitive clauses
(negative positions, prompt-end-relative prompt windows, generation-step
selectors) fold prompt length into the key. Steered rows inside a cache hit were
computed with the same steering and need no recomputation. Default and explicit
requests with identical configurations can share cache entries. Updating or
clearing the default preserves entries for previous configurations and requires
no global salt or cache reset.

## Validation

Use row labels and steering traces to check request attribution, layer targets and
selected token positions exactly. For generated-text comparisons, temperature zero
reduces sampling differences, though wording can still vary slightly; compare the
same workload and batch geometry. The
[EasySteer test guide](https://github.com/ZJU-REAL/EasySteer/blob/main/tests/README.md)
describes the available structural checks, numerical comparisons and benchmarks.

## Capture coexistence

Capture lives in `model_hooks/capture/` and can run without steering algorithms,
configuration slots or payload loading. Its hooks use the shared component
descriptors and token-selection rules. The
EasySteer capture helper currently requires a single worker and does not merge
tensor-parallel shards.

Steps whose effective selections are empty keep ordinary graph dispatch. When
selected rows may exist, a single-worker engine can use a separate FULL capture
graph if the batch is eligible for FULL replay, speculative decoding and LoRA
are disabled, and steering is either disabled or `in_graph`. Other capture steps
use the raw eager forward. Ordinary compiled graphs contain no capture hooks.

The separate graph records the raw model forward, including steering, and copies
component outputs into fixed GPU buffers. Selection, reduction, dtype conversion
and storage run after replay through the same path as eager capture. One variant
is retained for the active streams and discovered layer sets; changing selectors,
reduction or output dtype does not require recording it again. Stopping capture
disables its dispatch while retaining that variant for reuse. A different stream
or layer signature replaces it when another FULL capture is needed.

Successful capture RPCs are mirrored at request admission. `InputProcessor`
evaluates each request's effective selection, including per-request overrides,
and sets `skip_reading_prefix_cache=True` only when a cache hit could omit selected
prompt rows. Cache writes remain enabled under the ordinary key; the helper
preserves caller salts and steering fingerprints. If capture starts after a
request has already reused selected prompt rows, fetch reports the incomplete
capture explicitly. Workers retain cache-hit metadata and per-request selections
through preemption until request completion, including while streams are disabled.
Each request's first scheduled batch after stream activation checks that history
against its effective selection and reduction. Completed requests awaiting worker
cleanup do not participate in the new stream.

Within a forward pass, each capture stream reuses its row-selection plan and labels across layers. Only the layer values are gathered or reduced again. Plans are invalidated for a new batch geometry or store, and distinguish device and effective token length. Captured values are staged until the step-end transfer, with ownership preserved for model buffers that may be reused.

Capture serialization and deserialization use one dtype contract in
`capture/serialization.py`. Fetch results include each captured layer's component
layout; `attention_heads` records `width`, query `num_heads`, and value-output
`head_size`. Stored values remain two-dimensional `(rows, width)` tensors.
Storage budgets apply to currently retained rows;
fetching and clearing a layer releases its row budget for subsequent captures.
Budget limits are applied before gathering activation values. When all requested
layers are full, normal graph dispatch resumes while selected dropped rows are
still counted. Draining retained rows makes that capacity available again.

Each serialized layer includes only the request IDs referenced by its labels.
CPU chunks own rows for one request and are indexed by request and layer. A
per-request drain concatenates only the requested chunks and releases them without
copying the remaining activations. Stored label indices are recycled when their
last rows are released; unrelated labels do not need rewriting. Active request
history is retained separately until completion so preemption keeps its
prefix-cache semantics.
If steering fails for a request, fetching its captured rows raises with the
request's error; other requests can still be fetched by ID. Clearing the stream
clears the retained failure alongside its rows.
Runner shutdown releases capture hooks, streams, graph objects and output buffers.
