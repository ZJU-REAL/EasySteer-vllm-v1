# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph capabilities, workload mode selection, and request admission."""

from vllm.model_hooks.steering.algorithms import get_algorithm
from vllm.model_hooks.steering.algorithms.registry import ALGORITHM_REGISTRY
from vllm.model_hooks.steering.capabilities import algorithm_target
from vllm.model_hooks.steering.request import SteeringRequest

from .kernels import GRAPH_FAMILIES


def graph_safe_algorithms() -> frozenset[str]:
    """Algorithms declaring an in-graph (Tier-1) kernel family."""
    return frozenset(
        name for name, cls in ALGORITHM_REGISTRY.items() if cls.graph_family is not None
    )


def graph_problem(name: str, request=None, max_rank: int = 32) -> str | None:
    """Judge a concrete payload, or the guarantee available from its name.

    A name alone cannot promise a bounded rank or restricted algorithm
    variant. These conditions drive conservative auto selection; the same
    check admits concrete requests when they satisfy the conditions.
    """

    cls = get_algorithm(name)
    if cls.graph_family is None:
        return f"algorithm '{name}' has no in-graph kernel family"
    problem = cls.graph_payload_problem(request)
    if problem is not None:
        return problem
    dims = GRAPH_FAMILIES.get(cls.graph_family, {})
    if any("r" in d for d in dims.values()):
        if request is None:
            return "payload rank must be <= steer_graph_max_rank"
        rank = cls.wire_rank(request.payload)
        if rank is None:
            return f"a {name} payload whose rank could not be determined"
        if rank > max_rank:
            return f"payload rank {rank} above steer_graph_max_rank {max_rank}"
    return None


def steering_execution_modes() -> dict[str, tuple[str, ...]]:
    """Central algorithm -> supported steering graph tiers table.

    Derived from each algorithm's declared graph_family (the single
    source of truth on the class), never hand-maintained: every
    algorithm runs under split mode; those with a kernel family also
    run in-graph (conditionally for some — see graph_problem).
    """
    return {
        name: (("split", "in_graph") if cls.graph_family else ("split",))
        for name, cls in sorted(ALGORITHM_REGISTRY.items())
    }


def graph_request_problem(request: SteeringRequest, max_rank: int) -> str | None:
    """Why this request cannot run under full-graph steering, or None.

    Capability comes from each algorithm's declared graph_family; rank
    limits from the family schema.
    """
    if len(request.vectors) != 1:
        return "multi-vector configs"
    vector = request.vectors[0]
    return graph_problem(vector.algorithm, vector, max_rank)


def resolve_graph_mode(
    config, *, compiled: bool, default_request=None, allow_eager: bool = False
) -> tuple[str, str | None]:
    """Resolve the declared workload; explicit in_graph never falls back.

    Concrete engine defaults are judged like requests. Names-only declarations
    must cover every possible payload, so bounded-rank or variant conditions
    select split automatically. Explicit in_graph accepts those declarations
    and enforces their conditions when each request arrives.
    """
    if config.graph_mode == "split":
        return "split", None
    names = (
        sorted(ALGORITHM_REGISTRY) if config.algorithms == "all" else config.algorithms
    )
    conditions = {
        name: problem for name in names if (problem := graph_problem(name)) is not None
    }
    payload_problem = (
        graph_request_problem(default_request, config.graph_max_rank)
        if default_request is not None
        else None
    )
    if config.graph_mode == "auto":
        if not compiled:
            return "split", "no compiled execution"
        if default_request is not None:
            return (
                ("split", f"engine-default steering config: {payload_problem}")
                if payload_problem
                else (
                    "in_graph",
                    "engine-default steering config fits the graph buffers",
                )
            )
        if config.multi_vector:
            return "split", "multi-vector configs have no in-graph lowering"
        if conditions:
            detail = "; ".join(f"'{name}': {why}" for name, why in conditions.items())
            return "split", f"names alone do not guarantee graph support ({detail})"
        return "in_graph", f"declared algorithms {names} have no payload restrictions"

    if not compiled and not allow_eager:
        raise ValueError(
            "steer_graph_mode='in_graph' requires compiled execution; "
            "with enforce_eager or compilation disabled use 'split' or 'auto'."
        )
    if config.algorithms == "all":
        raise ValueError(
            "steer_graph_mode='in_graph' cannot serve steer_algorithms='all'; "
            "declare specific algorithms or use 'split'."
        )
    if config.multi_vector:
        raise ValueError(
            "steer_graph_mode='in_graph' cannot serve multi-vector steering; "
            "use 'split' or drop the steer_multi_vector declaration."
        )
    unsupported = [
        conditions[name] for name in names if get_algorithm(name).graph_family is None
    ]
    if unsupported or payload_problem:
        problem = "; ".join(unsupported) if unsupported else payload_problem
        raise ValueError(
            f"steer_graph_mode='in_graph' cannot serve this workload: "
            f"{problem}. Use 'split' or 'auto'."
        )
    if conditions and default_request is None:
        detail = "; ".join(f"'{name}': {why}" for name, why in conditions.items())
        return "in_graph", (
            f"requests outside these graph conditions are rejected: {detail}"
        )
    return "in_graph", None


def declared_graph_families(
    algorithms: list[str] | str | None,
    *,
    component_id: str | None = None,
) -> frozenset[str]:
    """Decoder kernel families the declared workload can ever use.

    Takes the normalized `SteerVectorConfig.algorithms` value. The
    in-graph kernel is compiled with exactly these families: admission
    rejects undeclared algorithms on every engine, so a family no
    declared algorithm maps to can never receive a payload — compiling
    it in would only cost every unsteered token. A missing or 'all'
    declaration (defensive; auto never resolves those to in_graph)
    keeps every family.
    """

    names = (
        ALGORITHM_REGISTRY if algorithms is None or algorithms == "all" else algorithms
    )
    families = set()
    for name in names:
        if component_id is not None and algorithm_target(name) != component_id:
            continue
        family = get_algorithm(name).graph_family
        if family in GRAPH_FAMILIES:  # decoder families only (not moe_gate)
            families.add(family)
    return frozenset(families)


def declared_graph_gate(algorithms: list[str] | str | None) -> bool:
    """Whether the declared workload needs the MoE gate kernel."""

    return (
        algorithms is None
        or algorithms == "all"
        or any(get_algorithm(name).graph_family == "moe_gate" for name in algorithms)
    )


def graph_reject_message(problem: str) -> str:
    """The user-facing rejection for a graph_request_problem result."""
    return (
        f"steer graph_mode=in_graph supports single-vector configs of "
        f"{sorted(graph_safe_algorithms())}; got {problem}. Declare the "
        f"workload via steer_algorithms so auto picks the right tier, "
        f"or launch with steer_graph_mode='split' to run this config "
        f"under CUDA graphs."
    )
