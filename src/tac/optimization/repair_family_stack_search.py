# SPDX-License-Identifier: MIT
"""Cross-family stack search over repair byte-transform execution reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_posterior import (
    load_repair_campaign_stackability_posterior_rows,
)
from tac.optimization.repair_family_byte_transform_executor import (
    REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA,
)
from tac.repo_io import sha256_file

REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA = "repair_family_stack_search_plan.v1"
REPAIR_FAMILY_STACK_SEARCH_ROW_SCHEMA = "repair_family_stack_search_row.v1"

_LEVEL_ORDER: tuple[str, ...] = (
    "bit",
    "byte",
    "pixel",
    "boundary",
    "region",
    "frame",
    "pair",
    "batch",
    "full_video",
)


class RepairFamilyStackSearchError(ValueError):
    """Raised when repair-family stack search cannot be planned."""


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _report_file_record(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise RepairFamilyStackSearchError(f"execution report missing: {path}")
    return {
        "path": _repo_rel(resolved, repo_root),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _posterior_demotions(
    posterior_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_family: dict[str, dict[str, Any]] = {}
    for row in posterior_rows:
        if not isinstance(row, Mapping):
            continue
        family = str(row.get("family_id") or "unclassified_repair_family")
        policy_delta = _mapping(row.get("acquisition_policy_delta"))
        policy = str(policy_delta.get("recommended_acquisition_policy") or "")
        direction = str(policy_delta.get("family_priority_direction") or "")
        blockers = " ".join(_string_list(row.get("blockers")))
        negative = (
            direction == "decrease"
            or policy.startswith("decrease_")
            or "non_improving" in blockers
            or "negative_result" in blockers
        )
        state = by_family.setdefault(
            family,
            {
                "family_id": family,
                "observation_count": 0,
                "negative_result_count": 0,
                "policies": [],
            },
        )
        state["observation_count"] += 1
        state["policies"].append(policy)
        if negative:
            state["negative_result_count"] += 1
    for state in by_family.values():
        observations = max(1, int(state["observation_count"]))
        negatives = int(state["negative_result_count"])
        state["demotion_multiplier"] = max(0.20, 1.0 - min(0.80, negatives / observations))
        state["demoted"] = negatives > 0
        state["policies"] = ordered_unique(state["policies"])
        state.update(FALSE_AUTHORITY)
    return by_family


def _level_penalty(levels: Sequence[str]) -> float:
    ordered = [level for level in _LEVEL_ORDER if level in set(levels)]
    if len(ordered) <= 1:
        return 0.0
    return min(0.18, 0.02 * (len(ordered) - 1))


def _scope_levels(report: Mapping[str, Any]) -> list[str]:
    scope = _mapping(report.get("fractal_optimization_scope"))
    return _string_list(scope.get("active_levels")) or _string_list(
        scope.get("declared_levels")
    )


def _stage_order(report: Mapping[str, Any]) -> int:
    stage = _mapping(report.get("active_entropy_stage"))
    return _safe_int(stage.get("order"), default=999)


def _byte_credit_feasible(report: Mapping[str, Any], remaining_budget: int | None) -> bool:
    delta = _mapping(report.get("byte_transform_delta"))
    requested = _safe_int(report.get("allocated_repair_bytes") or delta.get("bytes"))
    if remaining_budget is None:
        return True
    return requested <= max(0, remaining_budget)


def _stack_row(
    *,
    report: Mapping[str, Any],
    report_path: str | Path,
    repo_root: str | Path,
    posterior_demotions: Mapping[str, Mapping[str, Any]],
    remaining_budget: int | None,
    order_hint: int,
) -> dict[str, Any]:
    require_no_truthy_authority_fields(
        report,
        context=f"repair_family_stack_search_report:{report_path}",
    )
    if report.get("schema") != REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA:
        raise RepairFamilyStackSearchError(
            "stack search requires repair_family_byte_transform_execution_report.v1"
        )
    family = str(report.get("family_id") or "unclassified_repair_family")
    delta = _mapping(report.get("mlx_local_probe_delta"))
    combined_delta = _safe_float(delta.get("combined_delta_score_units"))
    local_improvement = max(0.0, -combined_delta)
    levels = _scope_levels(report)
    entropy_order = _stage_order(report)
    byte_delta = _mapping(report.get("byte_transform_delta"))
    delta_bytes = _safe_int(byte_delta.get("bytes"))
    demotion = dict(posterior_demotions.get(family) or {})
    demotion_multiplier = _safe_float(demotion.get("demotion_multiplier")) or 1.0
    scope_penalty = _level_penalty(levels)
    stack_penalty = _safe_float(report.get("interaction_penalty")) + scope_penalty
    feasible = _byte_credit_feasible(report, remaining_budget)
    negative_demoted = demotion.get("demoted") is True
    score = (
        (local_improvement / max(1, delta_bytes))
        * demotion_multiplier
        * (1.0 - min(0.95, stack_penalty))
        if feasible
        else 0.0
    )
    blockers = ordered_unique(
        [
            *_string_list(report.get("blockers")),
            *([] if feasible else ["byte_credit_exhausted_for_stack_row"]),
            *(
                ["automatic_negative_result_demotion_active"]
                if negative_demoted
                else []
            ),
            "exact_axis_required_before_score_or_budget",
        ]
    )
    row = {
        "schema": REPAIR_FAMILY_STACK_SEARCH_ROW_SCHEMA,
        "source_execution_report": _report_file_record(
            report_path,
            repo_root=repo_root,
        ),
        "family_id": family,
        "typed_response_id": report.get("typed_response_id"),
        "candidate_chain_id": report.get("candidate_chain_id"),
        "entropy_position_label": report.get("entropy_position_label"),
        "entropy_stage_order": entropy_order,
        "fractal_scope_levels": levels,
        "scope_order_indexes": [
            _LEVEL_ORDER.index(level) for level in levels if level in _LEVEL_ORDER
        ],
        "delta_payload_bytes": delta_bytes,
        "local_mlx_combined_delta_score_units": combined_delta,
        "local_mlx_expected_improvement_score_units": local_improvement,
        "stackability_penalty": stack_penalty,
        "posterior_demotion": demotion,
        "automatic_negative_result_demoted": negative_demoted,
        "byte_credit_feasible": feasible,
        "interaction_aware_stack_score": score,
        "order_hint": order_hint,
        "selected_for_materialization_handoff": False,
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_family_cross_stack_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        row,
        context=f"repair_family_stack_search_row:{family}",
    )
    return row


def plan_repair_family_stack_search(
    *,
    execution_reports: Sequence[Mapping[str, Any]],
    execution_report_paths: Sequence[str | Path],
    repo_root: str | Path,
    posterior_path: str | Path | None = None,
    byte_credit_budget: int | None = None,
) -> dict[str, Any]:
    """Plan a fail-closed cross-family stack over concrete byte-transform rows."""

    if len(execution_reports) != len(execution_report_paths):
        raise RepairFamilyStackSearchError(
            "execution_reports and execution_report_paths length mismatch"
        )
    posterior_rows = (
        load_repair_campaign_stackability_posterior_rows(posterior_path)
        if posterior_path is not None
        else []
    )
    demotions = _posterior_demotions(posterior_rows)
    remaining_budget = byte_credit_budget
    rows: list[dict[str, Any]] = []
    for order_hint, (report, path) in enumerate(
        zip(execution_reports, execution_report_paths, strict=True),
        start=1,
    ):
        row = _stack_row(
            report=report,
            report_path=path,
            repo_root=repo_root,
            posterior_demotions=demotions,
            remaining_budget=remaining_budget,
            order_hint=order_hint,
        )
        if remaining_budget is not None and row["byte_credit_feasible"]:
            remaining_budget = max(0, remaining_budget - int(row["delta_payload_bytes"]))
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["entropy_stage_order"],
            row["automatic_negative_result_demoted"],
            -float(row["interaction_aware_stack_score"]),
            row["order_hint"],
            str(row.get("typed_response_id") or ""),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["planned_stack_order"] = index
    plan = {
        "schema": REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA,
        "execution_report_count": len(rows),
        "posterior_path": None if posterior_path is None else str(posterior_path),
        "posterior_row_count": len(posterior_rows),
        "byte_credit_budget": byte_credit_budget,
        "automatic_negative_result_demotion_enabled": True,
        "entropy_ordering_rule": (
            "before_entropy_distribution_then_scorer_repair_then_selector_codec_"
            "then_entropy_boundary_then_post_container"
        ),
        "fractal_scope_order": list(_LEVEL_ORDER),
        "stack_rows": rows,
        "planned_family_order": ordered_unique(row["family_id"] for row in rows),
        "candidate_improvement_observed": any(
            float(row.get("local_mlx_expected_improvement_score_units") or 0.0) > 0.0
            for row in rows
        ),
        "exact_eval_handoff_gate": {
            "schema": "repair_family_stack_search_exact_handoff_gate.v1",
            "eligible_for_exact_eval_handoff": False,
            "blockers": [
                "byte_closed_archive_runtime_receiver_proof_required_per_stack",
                "contest_cpu_or_cuda_exact_axis_payload_required",
                "lane_dispatch_claim_required_before_exact_eval",
            ],
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_family_cross_stack_local_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        plan,
        context="repair_family_stack_search_plan",
    )
    return plan


__all__ = [
    "REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA",
    "REPAIR_FAMILY_STACK_SEARCH_ROW_SCHEMA",
    "RepairFamilyStackSearchError",
    "plan_repair_family_stack_search",
]
