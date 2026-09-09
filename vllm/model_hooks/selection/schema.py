# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical selection fields shared by API and execution paths."""

# Include selectors and their exclude twins, symmetric by construction.
INCLUDE_SELECTOR_KEYS: tuple[str, ...] = tuple(
    f"{phase}_{kind}"
    for phase in ("prompt", "generation")
    for kind in ("tokens", "positions", "window")
)
EXCLUDE_SELECTOR_KEYS: tuple[str, ...] = tuple(
    f"exclude_{key}" for key in INCLUDE_SELECTOR_KEYS
)
APPLY_SPEC_KEYS: tuple[str, ...] = (
    "prompt",
    "generation",
    *INCLUDE_SELECTOR_KEYS,
    *EXCLUDE_SELECTOR_KEYS,
)
