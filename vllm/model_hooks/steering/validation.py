# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workload and model admission checks, using metadata without loading tensors."""

from vllm.model_hooks.steering.capabilities import algorithm_target
from vllm.model_hooks.steering.payloads import effective_layers, validate_model_shape


def validate_prompt_conflicts(request, prompt_token_ids: list[int]) -> None:
    """Reject known overlaps before scheduling; generation is checked each step."""
    if request.conflict_resolution != "error" or len(request.vectors) < 2:
        return
    import numpy as np

    from vllm.model_hooks.selection.host import clause_mask
    from vllm.model_hooks.selection.runtime import clause_cache_key

    tokens = np.asarray(prompt_token_ids)
    size = len(tokens)
    masks = {}
    for vector in request.vectors:
        key = clause_cache_key(vector.apply_spec)
        if key not in masks:
            masks[key] = clause_mask(
                vector.apply_spec,
                np.zeros(size, dtype=bool),
                np.arange(size),
                np.full(size, size),
                np.full(size, -1),
                lambda: tokens,
            )
    for _, keys in request_position_groups(request):
        claimed = np.zeros(size, dtype=bool)
        for key in keys:
            if np.any(claimed & masks[key]):
                raise ValueError(
                    "Steering vectors conflict at prompt positions "
                    f"{np.flatnonzero(claimed & masks[key]).tolist()}; "
                    "use conflict_resolution='priority' or 'sequential'."
                )
            claimed |= masks[key]


def merge_model_info(workers: list[dict]) -> dict[str, dict[int, int | None]]:
    """Union pipeline stages; tensor-parallel replicas must agree on widths."""
    result: dict[str, dict[int, int | None]] = {}
    for worker in workers:
        for component, layers in worker.items():
            merged = result.setdefault(component, {})
            for layer, width in layers.items():
                layer = int(layer)
                if layer in merged and merged[layer] != width:
                    raise ValueError(f"Inconsistent {component} width at layer {layer}")
                merged[layer] = width
    return result


def request_position_groups(request) -> tuple[tuple, ...]:
    """Share conflict resolution only for identical ordered layer interventions."""
    from vllm.model_hooks.selection.runtime import clause_cache_key

    layers: dict[tuple[str, int], list[tuple | None]] = {}
    for vector in request.vectors:
        for layer in sorted(effective_layers(vector.payload, vector.target_layers)):
            target = (algorithm_target(vector.algorithm), layer)
            layers.setdefault(target, []).append(clause_cache_key(vector.apply_spec))
    return tuple(
        dict.fromkeys(
            (request.conflict_resolution, tuple(keys)) for keys in layers.values()
        )
    )


def validate_request_model(request, hidden_size: int, model_info: dict | None) -> None:
    for vector in request.vectors:
        validate_payload_model(
            vector.payload,
            vector.algorithm,
            hidden_size=hidden_size,
            model_info=model_info,
            target_layers=vector.target_layers,
        )


def validate_payload_model(
    payload: dict,
    algorithm: str,
    *,
    hidden_size: int,
    model_info: dict | None,
    target_layers: list[int] | None = None,
) -> None:
    """Validate requests and preloads against the same discovered component."""
    component = algorithm_target(algorithm)
    if component == "hidden_states":
        validate_model_shape(payload, hidden_size)
    if model_info is None:
        # Startup defaults are checked again after workers load the model.
        return
    available = model_info.get(component, {})
    layers = effective_layers(payload, target_layers)
    if not layers or not layers.intersection(available):
        raise ValueError(
            f"Steering vector targets no modules: {component} "
            f"layers {sorted(layers)}; available layers {sorted(available)}"
        )
    missing = layers.difference(available)
    if missing:
        raise ValueError(
            f"Steering vector targets unavailable {component} layers {sorted(missing)}"
        )
    if component == "attention_heads":
        validate_model_shape(payload, {layer: available[layer] for layer in layers})
    if payload["kind"] == "router":
        configs = payload["extra"]["layers"]
        for layer in layers:
            config = configs[str(layer)]
            if config["mode"] == "soft_topk":
                width = available[layer]
                if width is None:
                    raise ValueError(
                        f"Cannot validate moe_router topk: gate at layer "
                        f"{layer} does not expose its output width"
                    )
                if config["topk"] > width:
                    raise ValueError(
                        f"moe_router topk {config['topk']} exceeds expert "
                        f"count {width} at layer {layer}"
                    )


def validate_request_admission(
    steer_vector_request,
    config,
    *,
    hidden_size: int,
    model_info: dict | None,
    preloaded_payloads: set[tuple[str, str]],
    has_kv_transfer: bool,
) -> None:
    """Validate an enabled engine's workload and exact payload before scheduling."""
    if config.require_preload:
        missing = []
        for vector in steer_vector_request.vectors:
            path = vector.source
            if path and (vector.algorithm, vector.payload_sha256) not in (
                preloaded_payloads
            ):
                missing.append(path)
        if missing:
            raise ValueError(
                f"steer_require_preload is set and these vectors are "
                f"not preloaded with their current content: {missing}. "
                f"Preload them via LLM.preload_steer_vectors([...]) or "
                f"POST /v1/steering/vectors."
            )

    # The declared workload bounds requests in every graph mode.
    if config.algorithms != "all":
        used = sorted({vector.algorithm for vector in steer_vector_request.vectors})
        undeclared = sorted(set(used) - set(config.algorithms or []))
        if undeclared:
            raise ValueError(
                f"steering algorithm(s) {undeclared} were not "
                f"declared at launch (declared: {config.algorithms}). "
                f"Add them to steer_algorithms and restart the "
                f"engine."
            )
        if len(steer_vector_request.vectors) > 1 and not config.multi_vector:
            raise ValueError(
                "multi-vector steering was not declared at launch; "
                "start the engine with steer_multi_vector=True."
            )

    if config.graph_mode == "in_graph":
        from vllm.model_hooks.steering.graph.policy import (
            graph_reject_message,
            graph_request_problem,
        )

        problem = graph_request_problem(
            steer_vector_request,
            config.graph_max_rank,
        )
        if problem is not None:
            raise ValueError(graph_reject_message(problem))

    if (
        steer_vector_request.conflict_resolution == "error"
        and len(steer_vector_request.vectors) > 1
        and has_kv_transfer
    ):
        raise ValueError(
            "Multi-vector conflict_resolution='error' cannot be used with "
            "KV transfer: a dynamic conflict cannot retract exported KV rows. "
            "Use 'priority' or 'sequential', or disable KV transfer."
        )
    validate_request_model(steer_vector_request, hidden_size, model_info)
