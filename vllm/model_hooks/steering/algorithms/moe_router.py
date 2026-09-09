# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MoE Router Logits Intervention Algorithm

This algorithm allows steering MoE model behavior by modifying router logits,
which control expert selection probabilities.
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_hooks.steering.payloads import ROUTER_MODES, validate_router_mode

from .base import BaseSteerVectorAlgorithm
from .registry import register_algorithm

logger = init_logger(__name__)


@register_algorithm("moe_router")
class MoERouterAlgorithm(BaseSteerVectorAlgorithm):
    """
    MoE Router Logits intervention algorithm.

    Modifies router logits before top-k expert selection.

    Canonical modes:
    - 'activate': force expert_ids INTO the top-k. Log-softmax the
      logits, set the experts to per-token max + epsilon (mechanism
      from SteerMoE, arXiv:2509.09660: guaranteed selection, untouched
      experts keep their relative weights).
    - 'deactivate': force expert_ids OUT of the top-k (per-token
      min - epsilon, guaranteed exclusion).
      Both honor optional 'activate_ids'/'deactivate_ids' keys for
      layers that need both directions at once.
    - 'soft': z'_k = z_k + lambda * std(z), soft intervention scaled
      by the logit spread
    - 'soft_topk': soft intervention only when the expert is NOT
      already in top-k
    - 'soft_random': soft intervention on random experts (expert_ids
      determines the count)

    Payload format (dict):
    {
        'mode': 'deactivate',       # see above
        # experts for the mode's direction (for soft_random: count only)
        'expert_ids': [1, 5, 10],
        'activate_ids': [3, 7],    # (optional) also force into top-k
        'deactivate_ids': [1, 5],  # (optional) also force out of top-k
        'epsilon': 0.01,           # (optional) tie-breaking margin
        'lambda': 0.5,             # (optional) 'soft*' strength
    }
    """

    graph_family = "moe_gate"

    @classmethod
    def graph_payload_problem(cls, request=None) -> str | None:
        if request is None:
            return "soft_random requires split mode for per-token random sampling"
        for layer, payload in request.payload["extra"]["layers"].items():
            if request.target_layers and int(layer) not in request.target_layers:
                continue
            if payload["mode"] == "soft_random":
                return f"moe_router mode 'soft_random' at layer {layer}"
        return None

    @classmethod
    def graph_lower(cls, payload, scale):
        """Resolve one layer's parameters for the gate kernel tables.

        Mirrors _transform_toggle's routing: the canonical mode decides
        which direction expert_ids maps to, explicit activate_ids /
        deactivate_ids are honored in either mode, and deactivation
        wins overlaps (applied last in-kernel). Range validation is
        width-aware and lives in the table writer.
        """
        expert_ids = payload.get("expert_ids") or []
        activate = list(payload.get("activate_ids") or [])
        deactivate = list(payload.get("deactivate_ids") or [])
        mode = validate_router_mode(payload.get("mode", "activate"))
        if mode in ("soft", "soft_topk"):
            return {
                "mode": mode,
                "experts": expert_ids,
                "strength": payload.get("lambda", 0.5),
                "topk": payload.get("topk", 8),
            }
        if mode == "soft_random":
            raise ValueError("moe_router soft_random has no in-graph lowering")
        if mode == "activate":
            activate += expert_ids
        else:
            deactivate += expert_ids
        return {
            "mode": mode,
            "activate": activate,
            "deactivate": deactivate,
            "epsilon": payload.get("epsilon", 0.01),
        }

    def __init__(self, *, normalize: bool = False):
        if normalize:
            raise ValueError("moe_router does not support normalize")
        super().__init__()

    def _transform(self, router_logits: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Apply intervention to router logits.

        Args:
            router_logits: (num_tokens, n_experts) - raw logits from gate
            params: Intervention parameters dict (see class docstring)

        Returns:
            Modified router_logits with same shape
        """
        expert_ids = params.get("expert_ids", [])
        mode = validate_router_mode(params.get("mode", "activate"))
        lambda_param = params.get("lambda", 0.5)
        topk_param = params.get("topk", 8)

        if mode in ("activate", "deactivate"):
            return self._transform_toggle(router_logits, params, mode)

        if not expert_ids:
            return router_logits

        n_experts = router_logits.shape[-1]
        expert_ids = [eid for eid in expert_ids if 0 <= eid < n_experts]

        if not expert_ids:
            logger.warning("No valid expert IDs found in range [0, %s)", n_experts)
            return router_logits

        modified_logits = router_logits.clone()

        if mode == "soft":
            # Scale the intervention by each token's logit spread.
            logits_std = modified_logits.std(dim=-1, keepdim=True)
            modified_logits[:, expert_ids] += lambda_param * logits_std

        elif mode == "soft_topk":
            # Adjust only target experts outside each token's current top-k.
            topk_indices = modified_logits.topk(topk_param, dim=-1)[1]

            expert_ids_tensor = torch.tensor(
                expert_ids, device=modified_logits.device, dtype=torch.long
            )

            in_topk_mask = (
                topk_indices.unsqueeze(-1) == expert_ids_tensor.view(1, 1, -1)
            ).any(dim=1)

            logits_std = modified_logits.std(dim=-1, keepdim=True)
            delta = lambda_param * logits_std
            modified_logits[:, expert_ids] += delta * (~in_topk_mask).to(delta.dtype)

        elif mode == "soft_random":
            # Sample len(expert_ids) experts per token without replacement.
            num_experts_to_select = len(expert_ids)
            n_experts = modified_logits.shape[-1]
            num_tokens = modified_logits.shape[0]

            random_expert_ids = torch.stack(
                [
                    torch.randperm(n_experts, device=modified_logits.device)[
                        :num_experts_to_select
                    ]
                    for _ in range(num_tokens)
                ]
            )

            logits_std = modified_logits.std(dim=-1, keepdim=True)
            delta = lambda_param * logits_std

            batch_indices = (
                torch.arange(num_tokens, device=modified_logits.device)
                .unsqueeze(1)
                .expand(-1, num_experts_to_select)
            )

            batch_flat = batch_indices.flatten()
            expert_flat = random_expert_ids.flatten()
            delta_flat = delta.expand(-1, num_experts_to_select).flatten()

            modified_logits[batch_flat, expert_flat] += delta_flat

        return modified_logits

    def _transform_toggle(
        self, router_logits: torch.Tensor, params: dict, mode: str
    ) -> torch.Tensor:
        """Hard expert (de)activation (mechanism from arXiv:2509.09660).

        Logits are log-softmax normalized, then activated experts are set
        to the per-token max score + epsilon (guaranteeing top-k
        selection) and deactivated experts to the per-token min score -
        epsilon (guaranteeing exclusion). Downstream top-k softmax is
        monotone, so the untouched experts keep their relative weights.

        `mode` decides which direction `expert_ids` maps to; the
        explicit `activate_ids`/`deactivate_ids` keys are honored in
        either mode for layers steering both directions at once. An
        expert listed in both directions ends up deactivated
        (deactivation is applied last).
        """
        n_experts = router_logits.shape[-1]
        expert_ids = params.get("expert_ids") or []
        activate_ids = list(params.get("activate_ids") or [])
        deactivate_ids = list(params.get("deactivate_ids") or [])
        if mode == "activate":
            activate_ids += expert_ids
        else:
            deactivate_ids += expert_ids
        invalid = [e for e in activate_ids + deactivate_ids if not 0 <= e < n_experts]
        if invalid:
            # Raising here would kill the engine mid-forward; warn loudly
            # instead — a silently ignored expert id reads as "steering
            # has no effect".
            logger.warning_once(
                "moe_router: expert ids %s are outside [0, %d) for this "
                "model and are ignored.",
                str(sorted(set(invalid))),
                n_experts,
            )
        activate_ids = [e for e in activate_ids if 0 <= e < n_experts]
        deactivate_ids = [e for e in deactivate_ids if 0 <= e < n_experts]
        if not activate_ids and not deactivate_ids:
            logger.warning_once(
                "moe_router: no in-range expert ids remain; expert steering is a no-op."
            )
            return router_logits
        epsilon = params.get("epsilon", 0.01)

        scores = torch.nn.functional.log_softmax(router_logits, dim=-1)
        max_per_tok = scores.max(dim=-1, keepdim=True)[0]
        min_per_tok = scores.min(dim=-1, keepdim=True)[0]
        if activate_ids:
            scores[:, activate_ids] = max_per_tok + epsilon
        if deactivate_ids:
            scores[:, deactivate_ids] = min_per_tok - epsilon
        return scores

    def _is_valid(self, params: Any) -> bool:
        """Check if intervention parameters are valid."""
        if params is None:
            return False

        if not isinstance(params, dict):
            return False

        mode = params.get("mode", "activate")
        if mode not in ROUTER_MODES:
            return False
        if mode in ("activate", "deactivate"):
            return bool(
                params.get("expert_ids")
                or params.get("activate_ids")
                or params.get("deactivate_ids")
            )

        # Soft modes must have expert_ids
        expert_ids = params.get("expert_ids", [])
        return bool(expert_ids) and isinstance(expert_ids, list)
