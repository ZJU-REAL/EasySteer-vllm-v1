# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tier-1 full-graph steering kernels and their buffer schema.

The kernels here are pure tensor math over persistent buffers, safe to
capture into CUDA graphs / compiled code. Rows and masks are filled
host-side before each step; row 0 of every table is zero, so unsteered
or padding tokens and idle families contribute exact zero deltas.

GRAPH_FAMILIES is the closed schema the kernels are written against —
adding a family means extending both the schema and the kernel in the
same change. Algorithms opt in by declaring `graph_family` +
`graph_lower` on their class (see algorithms/base.py).
"""

import torch
from torch.library import wrap_triton

from vllm.triton_utils import tl, triton

# Family -> {table: dims}, dims over "h" (hidden size) and "r"
# (steer_graph_max_rank).
GRAPH_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    # delta = V[row]
    "additive": {"V": ("h",)},
    # delta = (x . B[row]) * C[row]
    "projection": {"B": ("h",), "C": ("h",)},
    # delta = (x @ A[row] + b[row]) @ Rout[row]^T
    "lowrank": {"A": ("h", "r"), "Rout": ("h", "r"), "b": ("r",)},
    # complete state becomes V[row]: delta = mask * (V[row] - x)
    "replace": {"V": ("h",)},
}

# Families whose delta is not neutralized by a zero table row carry
# their own per-token mask; all others share "graph_mask".
GRAPH_FAMILY_MASKS: dict[str | None, str] = {"replace": "replace_mask"}


def graph_family_mask_attr(family: str | None) -> str:
    return GRAPH_FAMILY_MASKS.get(family, "graph_mask")


@triton.jit
def _additive_kernel(
    Hidden,
    Residual,
    Vectors,
    Mask,
    Normalize,
    Rows,
    Out,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_STRIDE_0: tl.constexpr,
    HIDDEN_STRIDE_1: tl.constexpr,
    RESIDUAL_STRIDE_0: tl.constexpr,
    RESIDUAL_STRIDE_1: tl.constexpr,
    VECTOR_STRIDE_0: tl.constexpr,
    VECTOR_STRIDE_1: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < HIDDEN_SIZE
    dtype = Hidden.dtype.element_ty
    hidden = tl.load(
        Hidden + token * HIDDEN_STRIDE_0 + columns * HIDDEN_STRIDE_1,
        valid,
        other=0,
    ).to(tl.float32)
    out = hidden
    mask = tl.load(Mask + token).to(tl.float32)
    row = tl.load(Rows + token)
    if (mask != 0) & (row != 0):
        vector = tl.load(
            Vectors + row * VECTOR_STRIDE_0 + columns * VECTOR_STRIDE_1,
            valid,
            other=0,
        ).to(tl.float32)
        delta = (mask * vector).to(dtype).to(tl.float32)
        out = (hidden + delta).to(dtype).to(tl.float32)
        normalize = tl.load(Normalize + row).to(tl.float32)
        if normalize != 0:
            if HAS_RESIDUAL:
                residual = tl.load(
                    Residual + token * RESIDUAL_STRIDE_0 + columns * RESIDUAL_STRIDE_1,
                    valid,
                    other=0,
                ).to(tl.float32)
                x = (hidden + residual).to(dtype).to(tl.float32)
            else:
                x = hidden
            y = (x + delta).to(dtype).to(tl.float32)
            norm_x = tl.sqrt(tl.sum(x * x, axis=0))
            norm_y = tl.sqrt(tl.sum(y * y, axis=0))
            renormed = (y * norm_x / (norm_y + 1.0e-8)).to(dtype).to(tl.float32)
            difference = (renormed - y).to(dtype).to(tl.float32)
            flag = (normalize * mask).to(dtype).to(tl.float32)
            correction = (flag * difference).to(dtype).to(tl.float32)
            out += correction
    tl.store(Out + token * HIDDEN_SIZE + columns, out, valid)


def _apply_additive(
    vectors: torch.Tensor,
    graph_mask: torch.Tensor,
    normalize_flag: torch.Tensor,
    token_rows: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """One captured launch; inactive and unnormalized rows skip norm work."""
    n, hidden_size = hidden_states.shape
    out = hidden_states.new_empty((n, hidden_size))
    wrap_triton(_additive_kernel)[(n,)](
        hidden_states,
        residual,
        vectors,
        graph_mask,
        normalize_flag,
        token_rows,
        out,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_STRIDE_0=hidden_states.stride(0),
        HIDDEN_STRIDE_1=hidden_states.stride(1),
        RESIDUAL_STRIDE_0=0 if residual is None else residual.stride(0),
        RESIDUAL_STRIDE_1=0 if residual is None else residual.stride(1),
        VECTOR_STRIDE_0=vectors.stride(0),
        VECTOR_STRIDE_1=vectors.stride(1),
        HAS_RESIDUAL=residual is not None,
        BLOCK_SIZE=triton.next_power_of_2(hidden_size),
        num_warps=4 if hidden_size <= 2048 else 8,
        enable_fp_fusion=False,
    )
    return out


def apply_decoder_families(
    tables: dict[str, dict[str, torch.Tensor]],
    graph_mask: torch.Tensor | None,
    replace_mask: torch.Tensor | None,
    normalize_flag: torch.Tensor | None,
    token_rows: torch.Tensor | None,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """The captured decoder-layer steering kernel.

    Applies each family's delta to the selected rows of
    `hidden_states` (delta form over the complete state x = hidden +
    residual; the residual stream is never touched). `tables` holds
    only the families compiled into this engine — the declared
    workload's families (see declared_graph_families) — so families no
    declared algorithm can use contribute no compute at all; within
    one family, idle rows still cost only zero-row reads.

    The low-rank family picks its formulation from the slot capacity
    (a static buffer dimension, so the choice compiles in): dense
    all-slot coefficients selected per token while capacity is small,
    per-token weight gathers beyond that — the dense path's [slots,
    tokens, hidden] product grows linearly with capacity and OOMs
    Inductor autotuning at a few hundred slots.
    """
    if not tables:
        return hidden_states
    assert normalize_flag is not None
    assert token_rows is not None
    if len(tables) == 1 and "additive" in tables and hidden_states.is_cuda:
        assert graph_mask is not None
        return _apply_additive(
            tables["additive"]["V"],
            graph_mask,
            normalize_flag,
            token_rows,
            hidden_states,
            residual,
        )
    n = hidden_states.shape[0]
    rt = token_rows[:n]
    mask = None if graph_mask is None else graph_mask[:n].unsqueeze(1)
    x = hidden_states + residual if residual is not None else hidden_states

    delta = None
    if "additive" in tables:
        delta = tables["additive"]["V"][rt]
    if "projection" in tables:
        # delta += (x . B[row]) * C[row]
        proj = tables["projection"]
        coef = (x @ proj["B"].T).gather(1, rt.unsqueeze(1))
        term = coef * proj["C"][rt]
        delta = term if delta is None else delta + term
    if "lowrank" in tables:
        # delta += (x @ A[row] + b[row]) @ Rout[row]^T
        lowrank = tables["lowrank"]
        num_rows, _, rank = lowrank["A"].shape
        if num_rows <= 2 * rank:
            low = torch.einsum("nh,shr->snr", x, lowrank["A"])
            low = low + lowrank["b"].unsqueeze(1)
            out = torch.einsum("snr,shr->snh", low, lowrank["Rout"])
            term = out[rt, torch.arange(n, device=rt.device)]
        else:
            a_t = lowrank["A"][rt]
            low = torch.einsum("nh,nhr->nr", x, a_t) + lowrank["b"][rt]
            term = torch.einsum("nr,nhr->nh", low, lowrank["Rout"][rt])
        delta = term if delta is None else delta + term

    if delta is not None:
        assert mask is not None
    total = None if delta is None else mask * delta
    nf_mask = None if delta is None else mask
    if "replace" in tables:
        assert replace_mask is not None
        repl = replace_mask[:n].unsqueeze(1)
        term = repl * (tables["replace"]["V"][rt] - x)
        total = term if total is None else total + term
        nf_mask = repl if nf_mask is None else nf_mask + repl

    # normalize: rescale flagged steered rows to the original
    # complete-state norm (float32, mirroring _renormalize). Gated by
    # the token masks too — an eps-renorm on an unsteered row would
    # break bit-exactness.
    y = x + total
    norm_x = torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True)
    norm_y = torch.linalg.vector_norm(y.float(), dim=-1, keepdim=True)
    renormed = (y.float() * norm_x / (norm_y + 1e-8)).to(y.dtype)
    nf = normalize_flag[rt].unsqueeze(1) * nf_mask
    return hidden_states + total + nf * (renormed - y)


@triton.jit
def _gate_kernel(
    Logits,
    Activate,
    Deactivate,
    Epsilon,
    Strength,
    TopK,
    Mode,
    Mask,
    Rows,
    EXPERTS: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    row = tl.load(Rows + token)
    mask = tl.load(Mask + token)
    mode = tl.load(Mode + row)
    if (mask != 0) & (row != 0) & (mode != 0):
        columns = tl.arange(0, BLOCK_SIZE)
        valid = columns < EXPERTS
        dtype = Logits.dtype.element_ty
        ptrs = Logits + token * STRIDE_0 + columns * STRIDE_1
        x = tl.load(ptrs, valid, other=0).to(tl.float32)
        act = tl.load(Activate + row * EXPERTS + columns, valid, other=0).to(tl.int1)
        if mode == 1:
            deact = tl.load(Deactivate + row * EXPERTS + columns, valid, other=0).to(
                tl.int1
            )
            masked = tl.where(valid, x, float("-inf"))
            maximum = tl.max(masked, axis=0)
            logsum = tl.log(tl.sum(tl.exp(masked - maximum), axis=0))
            scores = (x - maximum - logsum).to(dtype).to(tl.float32)
            hi = tl.max(tl.where(valid, scores, float("-inf")), axis=0)
            lo = tl.min(tl.where(valid, scores, float("inf")), axis=0)
            eps = tl.load(Epsilon + row)
            out = tl.where(act, hi + eps, scores)
            out = tl.where(deact, lo - eps, out)
        else:
            mean = tl.sum(x, axis=0) / EXPERTS
            centered = tl.where(valid, x - mean, 0.0)
            std = tl.sqrt(tl.sum(centered * centered, axis=0) / (EXPERTS - 1))
            std = std.to(dtype).to(tl.float32)
            strength = tl.load(Strength + row)
            delta = (std * strength).to(dtype).to(tl.float32)
            selected = act
            if mode == 3:
                k = tl.load(TopK + row)
                ordered = tl.sort(tl.where(valid, x, float("-inf")), descending=True)
                threshold = tl.sum(tl.where(columns == k - 1, ordered, 0.0), axis=0)
                greater = tl.sum((valid & (x > threshold)).to(tl.int32), axis=0)
                tied = valid & (x == threshold)
                tie_rank = tl.cumsum(tied.to(tl.int32), axis=0)
                in_topk = (x > threshold) | (tied & (tie_rank <= k - greater))
                selected = selected & ~in_topk
            out = tl.where(selected, x + delta, x)
        tl.store(ptrs, out, valid)


def apply_gate_intervention(
    tables: dict[str, torch.Tensor],
    graph_mask: torch.Tensor,
    token_rows: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    """Captured router transform; only soft_topk rows perform a sort.

    The fixed expert width supports a different k in each slot. Boundary
    ties use expert-id order; torch.topk's tied index order is unspecified.
    Unselected rows and empty expert sets are exact no-ops.
    """
    if logits.is_cuda:
        n, experts = logits.shape
        wrap_triton(_gate_kernel)[(n,)](
            logits,
            tables["activate"],
            tables["deactivate"],
            tables["epsilon"],
            tables["strength"],
            tables["topk"],
            tables["mode"],
            graph_mask,
            token_rows,
            EXPERTS=experts,
            STRIDE_0=logits.stride(0),
            STRIDE_1=logits.stride(1),
            BLOCK_SIZE=triton.next_power_of_2(experts),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return
    n, experts = logits.shape
    rt = token_rows[:n]
    mode = tables["mode"][rt, None]
    active = (graph_mask[:n, None] != 0) & (rt[:, None] != 0) & (mode != 0)
    act = tables["activate"][rt]
    deact = tables["deactivate"][rt]
    scores = torch.nn.functional.log_softmax(logits, dim=-1)
    hi = scores.max(dim=-1, keepdim=True).values.float()
    lo = scores.min(dim=-1, keepdim=True).values.float()
    eps = tables["epsilon"][rt, None]
    toggled = torch.where(act, hi + eps, scores.float())
    toggled = torch.where(deact, lo - eps, toggled).to(logits.dtype)
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    in_topk = torch.zeros_like(act).scatter(
        1, order, torch.arange(experts)[None, :] < tables["topk"][rt, None]
    )
    selected = act & ((mode != 3) | ~in_topk)
    std = logits.std(dim=-1, keepdim=True)
    delta = (std.float() * tables["strength"][rt, None]).to(logits.dtype)
    softened = torch.where(selected, logits + delta, logits)
    out = torch.where(mode == 1, toggled, softened)
    logits.copy_(torch.where(active, out, logits))
