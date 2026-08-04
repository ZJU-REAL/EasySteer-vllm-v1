# SPDX-License-Identifier: Apache-2.0
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
GRAPH_FAMILY_MASKS: dict[str, str] = {"replace": "replace_mask"}


def graph_family_mask_attr(family: str | None) -> str:
    return GRAPH_FAMILY_MASKS.get(family, "graph_mask")


def apply_decoder_families(
    tables: dict[str, dict[str, torch.Tensor]],
    graph_mask: torch.Tensor,
    replace_mask: torch.Tensor,
    normalize_flag: torch.Tensor,
    token_rows: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> torch.Tensor:
    """The captured decoder-layer steering kernel.

    Applies every family's delta to the selected rows of
    `hidden_states` (delta form over the complete state x = hidden +
    residual; the residual stream is never touched). Low-rank and
    projection coefficients are computed for all slots at once (slot
    count is small) and then selected per token — this avoids
    materializing per-token [n, hidden, rank] weight gathers.
    """
    n = hidden_states.shape[0]
    rt = token_rows[:n]
    mask = graph_mask[:n].unsqueeze(1)
    if residual is not None:
        x = hidden_states + residual
    else:
        x = hidden_states

    delta = tables["additive"]["V"][rt]
    # Projection family: delta += (x . B[row]) * C[row]
    proj = tables["projection"]
    coef = (x @ proj["B"].T).gather(1, rt.unsqueeze(1))
    delta = delta + coef * proj["C"][rt]
    # Low-rank family: delta += (x @ A[row] + b[row]) @ Rout[row]^T
    lowrank = tables["lowrank"]
    low = torch.einsum("nh,shr->snr", x, lowrank["A"])
    low = low + lowrank["b"].unsqueeze(1)
    out = torch.einsum("snr,shr->snh", low, lowrank["Rout"])
    delta = delta + out[rt, torch.arange(n, device=rt.device)]

    repl = replace_mask[:n].unsqueeze(1)
    total = mask * delta + repl * (tables["replace"]["V"][rt] - x)
    # normalize: rescale flagged steered rows to the original
    # complete-state norm (float32, mirroring _renormalize). Gated by
    # the token masks too — an eps-renorm on an unsteered row would
    # break bit-exactness.
    y = x + total
    norm_x = torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True)
    norm_y = torch.linalg.vector_norm(y.float(), dim=-1, keepdim=True)
    renormed = (y.float() * norm_x / (norm_y + 1e-8)).to(y.dtype)
    nf = normalize_flag[rt].unsqueeze(1) * (mask + repl)
    return hidden_states + total + nf * (renormed - y)


def apply_gate_toggles(
    moe_act: torch.Tensor,
    moe_deact: torch.Tensor,
    moe_eps: torch.Tensor,
    graph_mask: torch.Tensor,
    token_rows: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    """The captured MoE gate steering kernel (mutates logits in place).

    Mirrors MoERouterAlgorithm._transform_toggle: steered rows become
    log-softmax scores with activated experts at max+eps and
    deactivated at min-eps (deactivation last); unsteered rows keep the
    raw logits bit-exactly.
    """
    n = logits.shape[0]
    rt = token_rows[:n]
    mask = graph_mask[:n].unsqueeze(1)
    scores = torch.nn.functional.log_softmax(logits, dim=-1)
    mx = scores.max(dim=-1, keepdim=True).values
    mn = scores.min(dim=-1, keepdim=True).values
    act = moe_act[rt]
    deact = moe_deact[rt]
    eps = moe_eps[rt].unsqueeze(1)
    out = scores * (1 - act) + act * (mx + eps)
    out = out * (1 - deact) + deact * (mn - eps)
    logits.copy_(mask * out + (1 - mask) * logits)
