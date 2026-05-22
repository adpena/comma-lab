# SPDX-License-Identifier: MIT
"""Canonical false-authority contract for proxy candidate rows.

Optuna, CMA-ES, Kaggle, macOS CPU, and other proxy surfaces are valuable for
search and curve-fitting, but they must all cross the same evidence boundary:
they can rank planning rows and warm-start exact work, never promote scores.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

PROXY_FALSE_AUTHORITY_FIELDS: dict[str, bool] = {
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "score_claim": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "field_selection_ready_for_exact_eval_dispatch": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
    "score_affecting_payload_changed": False,
    "charged_bits_changed": False,
}

CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS: tuple[str, ...] = (
    "score_claim",
    "score_claim_valid",
    "score_claim_eligible",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "field_selection_ready_for_exact_eval_dispatch",
    "dispatch_attempted",
    "gpu_launched",
    "dispatch_packet_ready",
    "exact_cuda_auth_eval",
    "contest_cuda_auth_eval",
    "promotable",
)

CONTEST_AUTH_SCORE_AXES: frozenset[str] = frozenset(
    {
        "contest-cpu",
        "contest-cpu-gha",
        "contest-cpu-gha-linux-x86-64",
        "contest-cpu-linux-x86-64",
        "contest-cuda",
    }
)
CONTEST_AUTH_SCORE_AXIS_PREFIXES: tuple[str, ...] = (
    "contest-cpu-gha-",
    "contest-cpu-linux-x86-64",
    "contest-cuda-",
)

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


def _normalized_axis(value: Any) -> str:
    text = str(value or "").strip().strip("[]").lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def auth_bridge_score_rankable(bridge: Mapping[str, Any]) -> bool:
    """Return true only for comparable contest-axis auth bridge scores."""

    if bridge.get("score_comparable") is not True:
        return False
    axis = _normalized_axis(bridge.get("score_axis"))
    return axis in CONTEST_AUTH_SCORE_AXES or axis.startswith(CONTEST_AUTH_SCORE_AXIS_PREFIXES)


def _authority_truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def truthy_authority_field_violations(
    payload: Mapping[str, Any],
    *,
    fields: Iterable[str] = CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    prefix: str = "",
) -> list[str]:
    """Return authority-bearing fields whose value is explicitly true.

    This is intentionally narrower than ``validate_proxy_candidate``: consumer
    payloads are optional passthrough metadata, so they may omit false fields.
    They must never smuggle a truthy score, promotion, rank/kill, dispatch, or
    exact-eval authority flag into downstream queues.
    """

    field_set = set(fields)
    root = prefix.rstrip(".")
    violations: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, inner in value.items():
                key_text = str(key)
                next_path = f"{path}.{key_text}" if path else key_text
                if key_text in field_set and _authority_truthy(inner):
                    violations.append(f"{next_path}=truthy")
                visit(inner, next_path)
        elif isinstance(value, list | tuple):
            for index, inner in enumerate(value):
                visit(inner, f"{path}[{index}]")

    visit(payload, root)
    return violations


def require_no_truthy_authority_fields(
    payload: Mapping[str, Any],
    *,
    context: str,
    fields: Iterable[str] = CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS,
) -> None:
    """Raise ``ValueError`` when a payload carries authority as ``true``."""

    violations = truthy_authority_field_violations(payload, fields=fields)
    if violations:
        raise ValueError(f"{context}: forbidden truthy authority fields: {', '.join(violations)}")


__all__ = [
    "CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS",
    "CONTEST_AUTH_SCORE_AXES",
    "CONTEST_AUTH_SCORE_AXIS_PREFIXES",
    "PROXY_DEPLOYMENT_TARGET",
    "PROXY_DISPATCH_BLOCKERS",
    "PROXY_FALSE_AUTHORITY_FIELDS",
    "PROXY_TARGET_MODES",
    "apply_proxy_evidence_boundary",
    "auth_bridge_score_rankable",
    "ordered_unique",
    "proxy_authority_fields",
    "require_no_truthy_authority_fields",
    "truthy_authority_field_violations",
    "validate_proxy_candidate",
]
