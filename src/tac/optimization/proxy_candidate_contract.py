# SPDX-License-Identifier: MIT
"""Canonical false-authority contract for proxy candidate rows.

Optuna, CMA-ES, Kaggle, macOS CPU, and other proxy surfaces are valuable for
search and curve-fitting, but they must all cross the same evidence boundary:
they can rank planning rows and warm-start exact work, never promote scores.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

PROXY_FALSE_AUTHORITY_FIELDS: dict[str, bool] = {
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "score_claim": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "field_selection_ready_for_exact_eval_dispatch": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
    "score_affecting_payload_changed": False,
    "charged_bits_changed": False,
}

PROXY_TARGET_MODES = ["contest_exact_eval_planning"]
PROXY_DEPLOYMENT_TARGET = "desktop_research"

PROXY_DISPATCH_BLOCKERS: tuple[str, ...] = (
    "optimizer_candidate_queue_is_planning_only",
    "requires_exact_eval_readiness_gate",
    "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
    "requires_non_proxy_score_evidence_before_promotion",
)


def ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def proxy_authority_fields() -> dict[str, bool | list[str] | str]:
    """Return canonical non-authority metadata for a proxy/planning row."""

    return {
        **PROXY_FALSE_AUTHORITY_FIELDS,
        "target_modes": list(PROXY_TARGET_MODES),
        "deployment_target": PROXY_DEPLOYMENT_TARGET,
    }


def apply_proxy_evidence_boundary(
    row: Mapping[str, Any],
    *,
    dispatch_blockers: Iterable[str] = (),
) -> dict[str, Any]:
    """Return ``row`` forced through the proxy/planning evidence boundary."""

    existing_blockers = row.get("dispatch_blockers")
    if isinstance(existing_blockers, str):
        incoming_blockers = [existing_blockers]
    elif isinstance(existing_blockers, Iterable):
        incoming_blockers = [str(item) for item in existing_blockers if str(item)]
    else:
        incoming_blockers = []
    out = dict(row)
    out.update(proxy_authority_fields())
    out["dispatch_blockers"] = ordered_unique(
        [
            *PROXY_DISPATCH_BLOCKERS,
            *incoming_blockers,
            *[str(item) for item in dispatch_blockers if str(item)],
        ]
    )
    return out


def validate_proxy_candidate(row: Mapping[str, Any]) -> list[str]:
    """Return violations for rows that leak proxy evidence as score authority."""

    violations: list[str] = []
    for key, expected in PROXY_FALSE_AUTHORITY_FIELDS.items():
        if row.get(key) is not expected:
            violations.append(f"{key}_must_be_{str(expected).lower()}")
    if row.get("target_modes") != PROXY_TARGET_MODES:
        violations.append("target_modes_must_be_contest_exact_eval_planning")
    blockers = row.get("dispatch_blockers")
    blocker_set = set(blockers if isinstance(blockers, list) else [])
    missing = [b for b in PROXY_DISPATCH_BLOCKERS if b not in blocker_set]
    if missing:
        violations.append("missing_proxy_dispatch_blockers:" + ",".join(missing))
    return violations


__all__ = [
    "PROXY_DEPLOYMENT_TARGET",
    "PROXY_DISPATCH_BLOCKERS",
    "PROXY_FALSE_AUTHORITY_FIELDS",
    "PROXY_TARGET_MODES",
    "apply_proxy_evidence_boundary",
    "ordered_unique",
    "proxy_authority_fields",
    "validate_proxy_candidate",
]
