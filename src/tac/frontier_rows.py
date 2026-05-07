"""Shared deterministic row schema for cross-paradigm frontier inventories."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

FRONTIER_ROW_SCHEMA = "cross_paradigm_frontier_row_v1"
FRONTIER_ROW_FIELDS: tuple[str, ...] = (
    "schema",
    "source_tool",
    "source_path",
    "key",
    "candidate_id",
    "title",
    "family",
    "family_group",
    "pareto_scope",
    "paradigms",
    "role",
    "status",
    "evidence_grade",
    "action_class",
    "priority_tier",
    "score_claim",
    "dispatch_attempted",
    "candidate_static_preflight_ready",
    "ready_for_exact_eval_dispatch",
    "pareto_eligible",
    "pareto_frontier",
    "score_evidence_rankable",
    "planning_priority_rankable",
    "expected_total_score_delta",
    "byte_delta",
    "expected_seg_dist_delta",
    "expected_pose_dist_delta",
    "expected_information_gain_nats",
    "blockers",
    "next_required_proof",
    "next_patch",
    "code_paths",
    "evidence_paths",
)


def _string_list(values: Iterable[Any] | Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values else []
    if isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        return [str(value) for value in values if str(value)]
    return [str(values)] if str(values) else []


def _ordered_unique(values: Iterable[Any] | Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in _string_list(values):
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def build_frontier_row(
    *,
    source_tool: str,
    source_path: str = "",
    key: str,
    candidate_id: str | None = None,
    title: str = "",
    family: str = "",
    family_group: str = "",
    pareto_scope: str = "",
    paradigms: Iterable[Any] | Any = (),
    role: str = "",
    status: str = "",
    evidence_grade: str = "",
    action_class: str = "",
    priority_tier: Any = None,
    score_claim: bool = False,
    dispatch_attempted: bool = False,
    candidate_static_preflight_ready: bool = False,
    ready_for_exact_eval_dispatch: bool = False,
    pareto_eligible: bool = False,
    pareto_frontier: bool = False,
    score_evidence_rankable: bool = False,
    planning_priority_rankable: bool = False,
    expected_total_score_delta: Any = None,
    byte_delta: Any = None,
    expected_seg_dist_delta: Any = None,
    expected_pose_dist_delta: Any = None,
    expected_information_gain_nats: Any = None,
    blockers: Iterable[Any] | Any = (),
    next_required_proof: Iterable[Any] | Any = (),
    next_patch: str = "",
    code_paths: Iterable[Any] | Any = (),
    evidence_paths: Iterable[Any] | Any = (),
) -> dict[str, Any]:
    """Return one stable, JSON-ready frontier row across planning tools."""

    resolved_candidate_id = candidate_id or key
    resolved_family = family or family_group or key
    resolved_family_group = family_group or resolved_family
    resolved_pareto_scope = pareto_scope or resolved_family_group
    return {
        "schema": FRONTIER_ROW_SCHEMA,
        "source_tool": source_tool,
        "source_path": source_path,
        "key": key,
        "candidate_id": resolved_candidate_id,
        "title": title,
        "family": resolved_family,
        "family_group": resolved_family_group,
        "pareto_scope": resolved_pareto_scope,
        "paradigms": _ordered_unique(paradigms),
        "role": role,
        "status": status,
        "evidence_grade": evidence_grade,
        "action_class": action_class,
        "priority_tier": _optional_int(priority_tier),
        "score_claim": bool(score_claim),
        "dispatch_attempted": bool(dispatch_attempted),
        "candidate_static_preflight_ready": bool(candidate_static_preflight_ready),
        "ready_for_exact_eval_dispatch": bool(ready_for_exact_eval_dispatch),
        "pareto_eligible": bool(pareto_eligible),
        "pareto_frontier": bool(pareto_frontier),
        "score_evidence_rankable": bool(score_evidence_rankable),
        "planning_priority_rankable": bool(planning_priority_rankable),
        "expected_total_score_delta": _optional_float(expected_total_score_delta),
        "byte_delta": _optional_float(byte_delta),
        "expected_seg_dist_delta": _optional_float(expected_seg_dist_delta),
        "expected_pose_dist_delta": _optional_float(expected_pose_dist_delta),
        "expected_information_gain_nats": _optional_float(expected_information_gain_nats),
        "blockers": _ordered_unique(blockers),
        "next_required_proof": _ordered_unique(next_required_proof),
        "next_patch": next_patch,
        "code_paths": _ordered_unique(code_paths),
        "evidence_paths": _ordered_unique(evidence_paths),
    }
