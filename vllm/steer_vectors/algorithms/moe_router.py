# SPDX-License-Identifier: Apache-2.0
"""
MoE Router Logits Intervention Algorithm

This algorithm allows steering MoE model behavior by modifying router logits,
which control expert selection probabilities.
"""

import json
from typing import Any

import torch

from vllm.logger import init_logger

from .factory import register_algorithm
from .template import AlgorithmTemplate

logger = init_logger(__name__)


@register_algorithm("moe_router")
class MoERouterAlgorithm(AlgorithmTemplate):
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

    Deprecated aliases (kept working): 'boost' and 'soft_hard' ->
    'activate'; 'suppress' -> 'deactivate'; 'steermoe' -> the
    activate_ids/deactivate_ids form. The alias implementations set raw
    logits to the exact max/min, which made top-k tie-break ambiguous
    ('soft_hard' worked around it with random noise); the canonical
    epsilon mechanism replaces them deterministically.

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

    CANONICAL_MODES = ("activate", "deactivate", "soft", "soft_topk",
                       "soft_random")
    MODE_ALIASES = {
        "boost": "activate",
        "soft_hard": "activate",
        "suppress": "deactivate",
        "steermoe": "activate",
    }

    @classmethod
    def validate_mode(cls, mode: str) -> str:
        """Reject unknown modes; returns the canonical spelling.

        The single mode validator — admission paths (api.py, request.py)
        and the file loader all call this.
        """
        canonical = cls.MODE_ALIASES.get(mode, mode)
        if canonical not in cls.CANONICAL_MODES:
            known = sorted(set(cls.CANONICAL_MODES) | set(cls.MODE_ALIASES))
            raise ValueError(
                f"unknown moe_router mode {mode!r}; expected one of {known}"
            )
        return canonical

    # Gate-hook kernel family (expert toggle tables on the gate
    # controller); activate/deactivate only — soft modes and file-based
    # configs stay piecewise (see graph_request_problem).
    graph_family = "moe_gate"

    def __init__(self, layer_id: int | None = None, **kwargs):
        # MoE router doesn't use normalize parameter - remove it from kwargs if present
        kwargs.pop("normalize", None)
        super().__init__(layer_id=layer_id, normalize=False, **kwargs)

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
        mode = params.get("mode", "activate")
        mode = self.MODE_ALIASES.get(mode, mode)
        lambda_param = params.get("lambda", 0.5)  # Default lambda for soft modes
        topk_param = params.get("topk", 8)  # Default top-k for soft_topk mode

        if mode in ("activate", "deactivate"):
            return self._transform_toggle(router_logits, params, mode)

        if not expert_ids:
            # No experts specified, return original
            return router_logits

        # Validate expert_ids
        n_experts = router_logits.shape[-1]
        expert_ids = [eid for eid in expert_ids if 0 <= eid < n_experts]

        if not expert_ids:
            logger.warning("No valid expert IDs found in range [0, %s)", n_experts)
            return router_logits

        # Clone to avoid modifying original
        modified_logits = router_logits.clone()

        if mode == "soft":
            # Soft intervention: z'_k = z_k + lambda * std(z)
            # Calculate standard deviation of logits for each token
            # std(z) shape: (num_tokens, 1)
            logits_std = modified_logits.std(dim=-1, keepdim=True)

            # Apply intervention: add (or subtract if lambda < 0) lambda * std(z)
            # This adjusts logits proportionally to their distribution spread
            # Batch operation: use advanced indexing to modify all expert_ids at once
            # Broadcasting: logits_std (num_tokens, 1) -> (num_tokens, len(expert_ids))
            modified_logits[:, expert_ids] += lambda_param * logits_std

        elif mode == "soft_topk":
            # Conditional soft intervention: only intervene if expert is NOT in top-k
            # This boosts experts that are not already selected, encouraging diversity

            # Get top-k expert indices for each token
            # topk_indices shape: (num_tokens, topk_param)
            topk_indices = modified_logits.topk(topk_param, dim=-1)[1]

            # Create tensor of target expert IDs on same device
            expert_ids_tensor = torch.tensor(
                expert_ids, device=modified_logits.device, dtype=torch.long
            )

            # Check which experts are in top-k for each token
            # Broadcast comparison: (num_tokens, topk_param, 1) == (1, 1,
            # len(expert_ids))
            # Result: (num_tokens, topk_param, len(expert_ids))
            # .any(dim=1): collapse topk dimension -> (num_tokens, len(expert_ids))
            in_topk_mask = (
                topk_indices.unsqueeze(-1) == expert_ids_tensor.view(1, 1, -1)
            ).any(dim=1)

            # Calculate soft intervention delta
            logits_std = modified_logits.std(dim=-1, keepdim=True)
            delta = lambda_param * logits_std  # (num_tokens, 1)

            # Apply intervention only where expert is NOT in top-k
            # Multiply delta by inverted mask (broadcast to match expert_ids dimension)
            # ~in_topk_mask: (num_tokens, len(expert_ids)) - True where expert is NOT in
            # top-k
            # delta: (num_tokens, 1) -> broadcasts to (num_tokens, len(expert_ids))
            # Convert mask to same dtype as delta to avoid type mismatch
            modified_logits[:, expert_ids] += delta * (~in_topk_mask).to(delta.dtype)

        elif mode == "soft_random":
            # Random soft intervention: randomly select same number of experts and apply
            # soft intervention
            # This mode uses expert_ids only to determine the COUNT of experts to
            # randomly select

            num_experts_to_select = len(expert_ids)
            n_experts = modified_logits.shape[-1]

            # Randomly select experts for each token
            # For each token, we randomly pick num_experts_to_select experts without
            # replacement
            num_tokens = modified_logits.shape[0]

            # Generate random expert indices for each token
            # random_expert_ids shape: (num_tokens, num_experts_to_select)
            random_expert_ids = torch.stack(
                [
                    torch.randperm(n_experts, device=modified_logits.device)[
                        :num_experts_to_select
                    ]
                    for _ in range(num_tokens)
                ]
            )

            # Calculate soft intervention delta
            logits_std = modified_logits.std(dim=-1, keepdim=True)  # (num_tokens, 1)
            delta = lambda_param * logits_std

            # Apply intervention to randomly selected experts
            # Use advanced indexing to modify different experts for each token
            # batch_indices: [0, 1, 2, ..., num_tokens-1] repeated for each expert
            batch_indices = (
                torch.arange(num_tokens, device=modified_logits.device)
                .unsqueeze(1)
                .expand(-1, num_experts_to_select)
            )

            # Flatten indices for scatter operation
            batch_flat = batch_indices.flatten()
            expert_flat = random_expert_ids.flatten()
            delta_flat = delta.expand(-1, num_experts_to_select).flatten()

            # Apply delta to randomly selected experts
            modified_logits[batch_flat, expert_flat] += delta_flat

        else:
            logger.warning(
                "Unknown intervention mode: %s, must be 'activate', "
                "'deactivate', 'soft', 'soft_topk', or 'soft_random' (or "
                "a deprecated alias: 'boost', 'suppress', 'soft_hard', "
                "'steermoe')",
                mode,
            )
            return router_logits

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
        invalid = [
            e for e in activate_ids + deactivate_ids
            if not 0 <= e < n_experts
        ]
        if invalid:
            # Raising here would kill the engine mid-forward; warn loudly
            # instead — a silently ignored expert id reads as "steering
            # has no effect".
            logger.warning_once(
                "moe_router: expert ids %s are outside [0, %d) for this "
                "model and are ignored.",
                sorted(set(invalid)),
                n_experts,
            )
        activate_ids = [e for e in activate_ids if 0 <= e < n_experts]
        deactivate_ids = [e for e in deactivate_ids if 0 <= e < n_experts]
        if not activate_ids and not deactivate_ids:
            logger.warning_once(
                "moe_router: no in-range expert ids remain; expert "
                "steering is a no-op."
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

    @classmethod
    def load_from_path(
        cls,
        path: str,
        device: str,
        *,
        config=None,
        target_layers: list[int] | None = None,
        **kwargs,
    ) -> dict:
        """
        Load MoE router intervention config from JSON file.

        File format:
        {
            "layer_configs": {
                "15": {
                    "expert_ids": [1, 5, 10],
                    "mode": "activate"  # Optional, defaults to "activate"
                },
                "20": {
                    "expert_ids": [0, 2],
                    "mode": "deactivate"
                },
                "22": {
                    "mode": "activate",       # both directions at one layer
                    "activate_ids": [3],
                    "deactivate_ids": [0, 2],
                    "epsilon": 0.01           # Optional tie-breaking margin
                },
                "25": {
                    "expert_ids": [3, 7, 12],
                    "mode": "soft",
                    "lambda": 0.5  # Optional, for 'soft' mode, defaults to 0.5
                },
                "30": {
                    "expert_ids": [1, 3, 5, 7],
                    "mode": "soft_topk",
                    "lambda": 0.8,  # Optional, defaults to request.moe_lambda or 0.5
                    # Optional, only intervene if expert is NOT in top-8, defaults to 8
                    "topk": 8
                },
                "35": {
                    "expert_ids": [1, 5, 10],  # Only count matters, not specific IDs
                    "mode": "soft_random",
                    "lambda": 0.5  # Optional, defaults to request.moe_lambda or 0.5
                }
            }
        }

        Deprecated mode aliases 'boost', 'suppress', 'soft_hard' and
        'steermoe' are still accepted (see class docstring).

        Args:
            path: Path to JSON config file
            device: Target device (not used for config loading)
            **kwargs: Additional arguments (target_layers, moe_lambda, moe_topk, etc.)

        Returns:
            Dict with 'layer_payloads' key mapping layer_id to intervention params
        """
        import os

        # Defaults from the SteerVectorRequest; a request-level moe_mode
        # overrides the JSON per-layer modes.
        default_mode = kwargs.get("moe_mode")
        default_lambda = kwargs.get("moe_lambda", 0.5)
        default_topk = kwargs.get("moe_topk", 8)

        if not os.path.exists(path):
            raise FileNotFoundError(f"MoE config file not found: {path}")

        with open(path) as f:
            try:
                moe_config = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse MoE config {path}: {e}") from e

        layer_configs = moe_config.get("layer_configs", {})
        layer_payloads = {}
        for layer_str, params in layer_configs.items():
            try:
                layer_id = int(layer_str)
            except ValueError:
                raise ValueError(
                    f"Invalid layer id {layer_str!r} in MoE config {path}"
                ) from None

            # Mode priority: request moe_mode > JSON > 'activate'.
            mode = (
                default_mode
                if default_mode is not None
                else params.get("mode", "activate")
            )
            try:
                canonical = cls.validate_mode(mode)
            except ValueError as e:
                raise ValueError(f"Layer {layer_id}: {e}") from None

            if canonical in ("activate", "deactivate"):
                if not (
                    params.get("expert_ids")
                    or params.get("activate_ids")
                    or params.get("deactivate_ids")
                ):
                    raise ValueError(
                        f"Layer {layer_id} {mode} config has no expert ids "
                        "('expert_ids', 'activate_ids' or 'deactivate_ids')"
                    )
            elif "expert_ids" not in params:
                raise ValueError(f"Layer {layer_id} is missing 'expert_ids'")

            intervention_params = {
                "expert_ids": params.get("expert_ids", []),
                "mode": mode,
            }

            if canonical in ("activate", "deactivate"):
                intervention_params["activate_ids"] = params.get("activate_ids", [])
                intervention_params["deactivate_ids"] = params.get("deactivate_ids", [])
                if "epsilon" in params:
                    intervention_params["epsilon"] = params["epsilon"]

            # lambda/topk: JSON value wins, else the request-level default.
            if "lambda" in params:
                intervention_params["lambda"] = params["lambda"]
            elif canonical in ("soft", "soft_topk", "soft_random"):
                intervention_params["lambda"] = default_lambda
                logger.debug(
                    "Layer %d: using default lambda=%s for %s mode",
                    layer_id,
                    default_lambda,
                    mode,
                )

            if "topk" in params:
                intervention_params["topk"] = params["topk"]
            elif canonical == "soft_topk":
                intervention_params["topk"] = default_topk
                logger.debug(
                    "Layer %d: using default topk=%s for soft_topk mode",
                    layer_id,
                    default_topk,
                )

            layer_payloads[layer_id] = intervention_params

        if not layer_payloads:
            raise ValueError(f"No layer configurations found in {path}")

        return {"layer_payloads": layer_payloads}

    def _is_valid(self, params: Any) -> bool:
        """Check if intervention parameters are valid."""
        if params is None:
            return False

        if not isinstance(params, dict):
            return False

        mode = params.get("mode", "activate")
        if self.MODE_ALIASES.get(mode, mode) in ("activate", "deactivate"):
            return bool(
                params.get("expert_ids")
                or params.get("activate_ids")
                or params.get("deactivate_ids")
            )

        # Soft modes must have expert_ids
        expert_ids = params.get("expert_ids", [])
        if not expert_ids or not isinstance(expert_ids, list):
            return False

        return True
