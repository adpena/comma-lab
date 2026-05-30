# SPDX-License-Identifier: MIT
"""Cross-family stack search over repair byte-transform execution reports."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_contract import (
    ArchiveBoundCandidateContractError,
    archive_bound_candidate_contracts_from_payload,
)
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
from tac.repo_io import json_text, sha256_bytes, sha256_file

REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA = "repair_family_stack_search_plan.v1"
REPAIR_FAMILY_STACK_SEARCH_ROW_SCHEMA = "repair_family_stack_search_row.v1"
REPAIR_FAMILY_EXACT_HANDOFF_CANDIDATE_ROW_SCHEMA = "repair_family_exact_handoff_candidate_row.v1"
REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA = "repair_family_exact_handoff_plan.v1"
REPAIR_FAMILY_STACK_LEARNING_SIGNAL_REPORT_SCHEMA = "repair_campaign_blocked_learning_signal_report.v1"
REPAIR_FAMILY_STACK_LEARNING_SIGNAL_SCHEMA = "repair_campaign_learning_signal.v1"
REPAIR_FAMILY_STACK_LOCAL_PLANNING_UPDATE_SCHEMA = "repair_campaign_local_planning_update.v1"
ARCHIVE_BOUND_CONTRACT_PAYLOAD_KEYS = frozenset(
    {
        "archive_bound_candidate_contract",
        "archive_bound_candidate_contract_surface",
        "archive_bound_candidate_contract_schema",
        "archive_bound_candidate_contract_surface_schema",
    }
)

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
_LEVEL_RANK = {level: index for index, level in enumerate(_LEVEL_ORDER)}


class RepairFamilyStackSearchError(ValueError):
    """Raised when repair-family stack search cannot be planned."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(json_text(payload).encode("utf-8"))


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


def _has_archive_bound_contract_payload(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in ARCHIVE_BOUND_CONTRACT_PAYLOAD_KEYS)


def _archive_bound_contract_for_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], list[str]]:
    if not _has_archive_bound_contract_payload(payload):
        return {}, []
    try:
        contracts = archive_bound_candidate_contracts_from_payload(payload, label=label)
    except ArchiveBoundCandidateContractError as exc:
        return {}, [f"archive_bound_candidate_contract_invalid:{exc}"]
    selected = [
        contract
        for contract in contracts
        if contract.get("selected_archive_transform_variant") is True
    ]
    return dict(selected[0] if selected else contracts[0] if contracts else {}), []


def _validated_archive_bound_contract_surface(
    payload: Mapping[str, Any],
    *,
    contract_blockers: Sequence[str],
) -> Mapping[str, Any]:
    if contract_blockers:
        return {}
    return _mapping(payload.get("archive_bound_candidate_contract_surface"))


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


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


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
                "positive_result_count": 0,
                "receiver_credit_exhausted_count": 0,
                "stackability_remeasure_required_count": 0,
                "entropy_stage_contract_miss_count": 0,
                "expected_improvement_sum": 0.0,
                "improvement_per_byte_sum": 0.0,
                "policies": [],
                "budget_routing_hints": [],
            },
        )
        state["observation_count"] += 1
        state["policies"].append(policy)
        feature_vector = _mapping(row.get("planner_feature_vector"))
        if not feature_vector:
            feature_vector = _mapping(_mapping(row.get("local_planning_update")).get("planner_feature_vector"))
        policy_delta = _mapping(row.get("acquisition_policy_delta"))
        expected = _safe_float(
            policy_delta.get("expected_local_improvement_score_units")
            or feature_vector.get("expected_local_improvement_score_units")
        )
        improvement_per_byte = _safe_float(
            policy_delta.get("improvement_per_allocated_byte") or feature_vector.get("improvement_per_allocated_byte")
        )
        state["expected_improvement_sum"] += max(0.0, expected)
        state["improvement_per_byte_sum"] += max(0.0, improvement_per_byte)
        if expected > 0.0 or improvement_per_byte > 0.0:
            state["positive_result_count"] += 1
        if policy_delta.get("receiver_credit_exhausted") is True or (
            feature_vector.get("receiver_credit_exhausted") is True
        ):
            state["receiver_credit_exhausted_count"] += 1
        if policy_delta.get("stackability_remeasure_required") is True or (
            feature_vector.get("stackability_remeasure_required") is True
        ):
            state["stackability_remeasure_required_count"] += 1
        if policy_delta.get("entropy_stage_contract_miss") is True or (
            feature_vector.get("entropy_stage_contract_miss") is True
        ):
            state["entropy_stage_contract_miss_count"] += 1
        hint = str(policy_delta.get("posterior_budget_routing_hint") or "").strip()
        if hint:
            state["budget_routing_hints"].append(hint)
        if negative:
            state["negative_result_count"] += 1
    for state in by_family.values():
        observations = max(1, int(state["observation_count"]))
        negatives = int(state["negative_result_count"])
        state["demotion_multiplier"] = max(0.20, 1.0 - min(0.80, negatives / observations))
        state["demoted"] = negatives > 0
        state["positive_fraction"] = int(state["positive_result_count"]) / observations
        state["mean_expected_improvement"] = float(state["expected_improvement_sum"]) / observations
        state["mean_improvement_per_byte"] = float(state["improvement_per_byte_sum"]) / observations
        posterior_boost = min(
            0.25,
            0.10 * float(state["positive_fraction"]) + min(0.15, float(state["mean_improvement_per_byte"]) * 4000.0),
        )
        posterior_penalty = min(
            0.55,
            0.20 * (int(state["receiver_credit_exhausted_count"]) / observations)
            + 0.16 * (int(state["entropy_stage_contract_miss_count"]) / observations)
            + 0.10 * (int(state["stackability_remeasure_required_count"]) / observations),
        )
        state["acquisition_multiplier"] = max(
            0.15,
            min(1.35, float(state["demotion_multiplier"]) * (1.0 + posterior_boost - posterior_penalty)),
        )
        state["policies"] = ordered_unique(state["policies"])
        state["budget_routing_hints"] = ordered_unique(state["budget_routing_hints"])
        state.update(FALSE_AUTHORITY)
    return by_family


def _level_penalty(levels: Sequence[str]) -> float:
    ordered = [level for level in _LEVEL_ORDER if level in set(levels)]
    if len(ordered) <= 1:
        return 0.0
    return min(0.18, 0.02 * (len(ordered) - 1))


def _scope_levels(report: Mapping[str, Any]) -> list[str]:
    scope = _mapping(report.get("fractal_optimization_scope"))
    return _string_list(scope.get("active_levels")) or _string_list(scope.get("declared_levels"))


def _archive_entropy_anti_pattern_penalty(
    archive_entropy_coverage: Mapping[str, Any],
) -> float:
    anti_pattern_ids = {
        str(protection.get("anti_pattern_id") or "").strip()
        for protection in archive_entropy_coverage.get("anti_pattern_protections") or []
        if isinstance(protection, Mapping)
    }
    blockers = " ".join(_string_list(archive_entropy_coverage.get("blockers")))
    prototype_substrates = _string_list(archive_entropy_coverage.get("prototype_substrates"))
    penalty = 0.0
    if (
        "proxy_or_advisory_probe_masquerades_as_score_authority_v1" in anti_pattern_ids
        and archive_entropy_coverage.get("ready_for_exact_eval_dispatch") is False
    ):
        penalty += 0.04
    if (
        "probe_only_side_report_orphaned_from_optimizer_v1" in anti_pattern_ids
        and not (_string_list(archive_entropy_coverage.get("probed_substrates")) or prototype_substrates)
    ):
        penalty += 0.12
    if (
        "scaffold_or_probe_bytes_without_receiver_consumption_v1" in anti_pattern_ids
        and prototype_substrates
        and "contest_runtime_adapter_missing" in blockers
    ):
        penalty += 0.12
    if (
        "zero_order_entropy_estimate_promoted_as_materialized_savings_v1" in anti_pattern_ids
        and "exact_axis_adjudication_missing" in blockers
    ):
        penalty += 0.08
    if "entropy_coder_order_cargo_cult_v1" in anti_pattern_ids and blockers:
        penalty += 0.04
    return min(0.35, penalty)


def _archive_variant_signal_penalty(
    archive_variant_signal_surface: Mapping[str, Any],
) -> float:
    """Penalize unresolved archive-variant signals without discarding them."""

    probe_count = _safe_int(archive_variant_signal_surface.get("probe_count"))
    blocked_count = _safe_int(archive_variant_signal_surface.get("blocked_signal_count"))
    blockers = " ".join(_string_list(archive_variant_signal_surface.get("blockers")))
    runtime_adapter_misses = blockers.count("contest_runtime_adapter_missing")
    penalty = min(0.08, 0.015 * probe_count)
    penalty += min(0.08, 0.010 * blocked_count)
    penalty += min(0.10, 0.025 * runtime_adapter_misses)
    return min(0.22, penalty)


def _stage_order(report: Mapping[str, Any]) -> int:
    stage = _mapping(report.get("active_entropy_stage"))
    return _safe_int(stage.get("order"), default=999)


def _byte_credit_feasible(report: Mapping[str, Any], remaining_budget: int | None) -> bool:
    delta = _mapping(report.get("byte_transform_delta"))
    requested = _safe_int(report.get("allocated_repair_bytes") or delta.get("bytes"))
    if remaining_budget is None:
        return True
    return requested <= max(0, remaining_budget)


def _stack_row_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("family_id") or "unclassified_repair_family"),
            str(row.get("typed_response_id") or "typed_response_unknown"),
            str(row.get("candidate_chain_id") or "candidate_chain_unknown"),
            str(row.get("order_hint") or "0"),
        ]
    )


def _level_sort_key(level: str) -> tuple[int, str]:
    return (_LEVEL_RANK.get(level, len(_LEVEL_ORDER)), level)


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
        raise RepairFamilyStackSearchError("stack search requires repair_family_byte_transform_execution_report.v1")
    family = str(report.get("family_id") or "unclassified_repair_family")
    delta = _mapping(report.get("mlx_local_probe_delta"))
    combined_delta = _safe_float(delta.get("combined_delta_score_units"))
    local_improvement = max(0.0, -combined_delta)
    levels = _scope_levels(report)
    entropy_order = _stage_order(report)
    byte_delta = _mapping(report.get("byte_transform_delta"))
    archive_entropy_coverage = _mapping(
        report.get("archive_entropy_substrate_coverage")
    )
    archive_variant_signal_surface = _mapping(report.get("archive_variant_signal_surface"))
    archive_variant_materializer_backlog = _mapping(
        report.get("archive_variant_materializer_backlog")
    )
    archive_bound_contract, archive_bound_contract_reader_blockers = (
        _archive_bound_contract_for_payload(
            report,
            label=f"repair_family_stack_search_contract:{report_path}",
        )
    )
    archive_bound_contract_surface = _validated_archive_bound_contract_surface(
        report,
        contract_blockers=archive_bound_contract_reader_blockers,
    )
    archive_entropy_blockers = _string_list(archive_entropy_coverage.get("blockers"))
    archive_entropy_anti_pattern_penalty = _archive_entropy_anti_pattern_penalty(
        archive_entropy_coverage
    )
    archive_variant_signal_penalty = _archive_variant_signal_penalty(
        archive_variant_signal_surface
    )
    archive_bound_contract_surface_penalty = _safe_float(
        archive_bound_contract_surface.get("acquisition_penalty_sum")
    )
    archive_bound_contract_count = max(
        1,
        _safe_int(archive_bound_contract_surface.get("candidate_contract_count")),
    )
    archive_bound_contract_budget_penalty = min(
        0.18,
        (archive_bound_contract_surface_penalty / archive_bound_contract_count)
        * 0.25,
    )
    archive_variant_signal_blockers = _string_list(
        archive_variant_signal_surface.get("blockers")
    )
    allocated_repair_bytes = _safe_int(report.get("allocated_repair_bytes"))
    delta_bytes = allocated_repair_bytes or _safe_int(byte_delta.get("bytes"))
    archive_native_saved_bytes = _safe_int(report.get("archive_native_saved_bytes"))
    exact_gate = _mapping(report.get("exact_eval_handoff_gate"))
    demotion = dict(posterior_demotions.get(family) or {})
    demotion_multiplier = _safe_float(demotion.get("demotion_multiplier")) or 1.0
    acquisition_multiplier = _safe_float(demotion.get("acquisition_multiplier")) or demotion_multiplier
    scope_penalty = _level_penalty(levels)
    stack_penalty = (
        _safe_float(report.get("interaction_penalty"))
        + scope_penalty
        + archive_entropy_anti_pattern_penalty
        + archive_variant_signal_penalty
        + _safe_float(archive_bound_contract.get("acquisition_penalty"))
        + archive_bound_contract_budget_penalty
        + (0.20 if archive_bound_contract_reader_blockers else 0.0)
    )
    feasible = _byte_credit_feasible(report, remaining_budget)
    negative_demoted = demotion.get("demoted") is True
    score = (
        (local_improvement / max(1, delta_bytes)) * acquisition_multiplier * (1.0 - min(0.95, stack_penalty))
        if feasible
        else 0.0
    )
    blockers = ordered_unique(
        [
            *_string_list(report.get("blockers")),
            *archive_bound_contract_reader_blockers,
            *archive_entropy_blockers,
            *archive_variant_signal_blockers,
            *_string_list(archive_bound_contract.get("blockers")),
            *([] if feasible else ["byte_credit_exhausted_for_stack_row"]),
            *(["automatic_negative_result_demotion_active"] if negative_demoted else []),
            *(
                ["posterior_receiver_credit_rebudget_required"]
                if int(demotion.get("receiver_credit_exhausted_count") or 0) > 0
                else []
            ),
            *(
                ["posterior_entropy_stage_contract_miss_active"]
                if int(demotion.get("entropy_stage_contract_miss_count") or 0) > 0
                else []
            ),
            *(
                ["posterior_stackability_remeasure_required"]
                if int(demotion.get("stackability_remeasure_required_count") or 0) > 0
                else []
            ),
            "exact_axis_required_before_score_or_budget",
        ]
    )
    row = {
        "schema": REPAIR_FAMILY_STACK_SEARCH_ROW_SCHEMA,
        "stack_row_key": None,
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
        "scope_order_indexes": [_LEVEL_ORDER.index(level) for level in levels if level in _LEVEL_ORDER],
        "delta_payload_bytes": delta_bytes,
        "allocated_repair_bytes": allocated_repair_bytes,
        "archive_native_saved_bytes": archive_native_saved_bytes,
        "archive_bound_candidate_contract_surface": dict(
            archive_bound_contract_surface
        ),
        "archive_bound_candidate_contract": dict(archive_bound_contract),
        "archive_bound_candidate_contract_count": _safe_int(
            archive_bound_contract_surface.get("candidate_contract_count")
        ),
        "archive_bound_ready_contract_count": _safe_int(
            archive_bound_contract_surface.get("archive_bound_ready_contract_count")
        ),
        "archive_bound_contract_substrate_tags": ordered_unique(
            [
                *_string_list(archive_bound_contract_surface.get("archive_substrate_tags")),
                *_string_list(archive_bound_contract.get("archive_substrate_tags")),
            ]
        ),
        "archive_bound_contract_acquisition_penalty": _safe_float(
            archive_bound_contract.get("acquisition_penalty")
        ),
        "archive_bound_contract_surface_acquisition_penalty": (
            archive_bound_contract_surface_penalty
        ),
        "archive_bound_contract_budget_routing_penalty": (
            archive_bound_contract_budget_penalty
        ),
        "archive_bound_contract_selected_kind": archive_bound_contract.get(
            "archive_native_transform_kind"
        ),
        "archive_bound_contract_ready": (
            archive_bound_contract.get("archive_bound_candidate_ready") is True
        ),
        "archive_entropy_substrate_coverage": dict(archive_entropy_coverage),
        "archive_entropy_substrate_materialized_substrates": _string_list(
            archive_entropy_coverage.get("materialized_substrates")
        ),
        "archive_entropy_substrate_probed_substrates": _string_list(
            archive_entropy_coverage.get("probed_substrates")
        ),
        "archive_entropy_substrate_prototype_substrates": _string_list(
            archive_entropy_coverage.get("prototype_substrates")
        ),
        "archive_entropy_probed_zero_order_savings_bytes": _safe_int(
            archive_entropy_coverage.get("probed_entropy_estimated_zero_order_savings_bytes")
        ),
        "archive_entropy_anti_pattern_protection_count": _safe_int(
            archive_entropy_coverage.get("anti_pattern_protection_count")
        ),
        "archive_entropy_anti_pattern_acquisition_penalty": archive_entropy_anti_pattern_penalty,
        "archive_entropy_anti_pattern_ids": ordered_unique(
            str(protection.get("anti_pattern_id") or "").strip()
            for protection in archive_entropy_coverage.get("anti_pattern_protections") or []
            if isinstance(protection, Mapping)
            and str(protection.get("anti_pattern_id") or "").strip()
        ),
        "archive_entropy_substrate_blockers": archive_entropy_blockers,
        "archive_variant_signal_surface": dict(archive_variant_signal_surface),
        "archive_variant_signal_count": _safe_int(
            archive_variant_signal_surface.get("row_count")
        ),
        "archive_variant_signal_kinds": _string_list(
            archive_variant_signal_surface.get("signal_transform_kinds")
        ),
        "archive_variant_non_selected_signal_count": _safe_int(
            archive_variant_signal_surface.get("non_selected_signal_count")
        ),
        "archive_variant_probe_count": _safe_int(
            archive_variant_signal_surface.get("probe_count")
        ),
        "archive_variant_prototype_count": _safe_int(
            archive_variant_signal_surface.get("prototype_count")
        ),
        "archive_variant_runtime_proof_ready_count": _safe_int(
            archive_variant_signal_surface.get("runtime_proof_ready_count")
        ),
        "archive_variant_blocked_signal_count": _safe_int(
            archive_variant_signal_surface.get("blocked_signal_count")
        ),
        "archive_variant_signal_blockers": archive_variant_signal_blockers,
        "archive_variant_signal_acquisition_penalty": archive_variant_signal_penalty,
        "archive_variant_materializer_backlog": dict(archive_variant_materializer_backlog),
        "archive_variant_materializer_backlog_task_count": _safe_int(
            archive_variant_materializer_backlog.get("row_count")
        ),
        "archive_variant_materializer_byte_closed_task_count": _safe_int(
            archive_variant_materializer_backlog.get("byte_closed_materialized_task_count")
        ),
        "archive_variant_materializer_runtime_adapter_ready_task_count": _safe_int(
            archive_variant_materializer_backlog.get("runtime_adapter_ready_task_count")
        ),
        "byte_closed_candidate_emitted": report.get("byte_closed_candidate_emitted") is True,
        "candidate_archive_materialized": (report.get("candidate_archive_materialized") is True),
        "archive_bound_runtime_consumption_proof_ready": (
            exact_gate.get("archive_bound_runtime_consumption_proof_ready") is True
        ),
        "archive_bound_exact_handoff_candidate": (
            archive_bound_contract.get("archive_bound_candidate_ready_for_exact_handoff")
            is True
            or (
                report.get("byte_closed_candidate_emitted") is True
                and report.get("candidate_archive_materialized") is True
                and exact_gate.get("archive_bound_runtime_consumption_proof_ready")
                is True
            )
        ),
        "local_mlx_combined_delta_score_units": combined_delta,
        "local_mlx_expected_improvement_score_units": local_improvement,
        "stackability_penalty": stack_penalty,
        "posterior_demotion": demotion,
        "posterior_acquisition_prior": demotion,
        "posterior_acquisition_multiplier": acquisition_multiplier,
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
    row["stack_row_key"] = _stack_row_key(row)
    return row


def _interaction_feature_vector(row: Mapping[str, Any]) -> dict[str, Any]:
    levels = set(_string_list(row.get("fractal_scope_levels")))
    family = str(row.get("family_id") or "unclassified_repair_family")
    vector = {
        "schema": "repair_family_stack_interaction_feature_vector.v1",
        "family_id": family,
        "entropy_stage_order": row.get("entropy_stage_order"),
        "entropy_position_label": row.get("entropy_position_label"),
        "has_bit_scope": "bit" in levels,
        "has_byte_scope": "byte" in levels,
        "has_pixel_scope": "pixel" in levels,
        "has_boundary_scope": "boundary" in levels,
        "has_region_scope": "region" in levels,
        "has_frame_scope": "frame" in levels,
        "has_pair_scope": "pair" in levels,
        "has_batch_scope": "batch" in levels,
        "has_full_video_scope": "full_video" in levels,
        "frame0_palette_asymmetry_prior": family in {"frame0_k16_palette_asymmetry", "palette_frame_asymmetry_prior"},
        "selector_codec_family": family == "per_region_selector_codec",
        "segnet_region_family": family == "segnet_class_region_waterfill",
        "posenet_bottom_decile_family": family == "posenet_null_bottom_decile",
        "entropy_boundary_family": family == "entropy_boundary_probe",
        "byte_credit_feasible": row.get("byte_credit_feasible") is True,
        "archive_bound_exact_handoff_candidate": (row.get("archive_bound_exact_handoff_candidate") is True),
        "archive_bound_contract_ready": (
            row.get("archive_bound_contract_ready") is True
        ),
        "archive_bound_contract_acquisition_penalty": _safe_float(
            row.get("archive_bound_contract_acquisition_penalty")
        ),
        "archive_bound_contract_surface_acquisition_penalty": _safe_float(
            row.get("archive_bound_contract_surface_acquisition_penalty")
        ),
        "archive_bound_contract_budget_routing_penalty": _safe_float(
            row.get("archive_bound_contract_budget_routing_penalty")
        ),
        "archive_bound_contract_substrate_tags": _string_list(
            row.get("archive_bound_contract_substrate_tags")
        ),
        "archive_bound_candidate_contract_count": _safe_int(
            row.get("archive_bound_candidate_contract_count")
        ),
        "archive_bound_ready_contract_count": _safe_int(
            row.get("archive_bound_ready_contract_count")
        ),
        "archive_variant_signal_count": _safe_int(row.get("archive_variant_signal_count")),
        "archive_variant_probe_count": _safe_int(row.get("archive_variant_probe_count")),
        "archive_variant_prototype_count": _safe_int(row.get("archive_variant_prototype_count")),
        "archive_variant_runtime_proof_ready_count": _safe_int(
            row.get("archive_variant_runtime_proof_ready_count")
        ),
        "archive_variant_non_selected_signal_count": _safe_int(
            row.get("archive_variant_non_selected_signal_count")
        ),
        "archive_variant_materializer_backlog_task_count": _safe_int(
            row.get("archive_variant_materializer_backlog_task_count")
        ),
        "archive_variant_materializer_byte_closed_task_count": _safe_int(
            row.get("archive_variant_materializer_byte_closed_task_count")
        ),
        "archive_variant_materializer_runtime_adapter_ready_task_count": _safe_int(
            row.get("archive_variant_materializer_runtime_adapter_ready_task_count")
        ),
        "negative_posterior_demoted": (row.get("automatic_negative_result_demoted") is True),
        "local_mlx_expected_improvement_score_units": _safe_float(
            row.get("local_mlx_expected_improvement_score_units")
        ),
        "stackability_penalty": _safe_float(row.get("stackability_penalty")),
        "interaction_aware_stack_score": _safe_float(row.get("interaction_aware_stack_score")),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        vector,
        context=f"repair_family_stack_interaction_feature_vector:{family}",
    )
    return vector


def _interaction_cell_key(vector: Mapping[str, Any]) -> str:
    scope_parts = [
        name.removeprefix("has_").removesuffix("_scope")
        for name in (
            "has_bit_scope",
            "has_byte_scope",
            "has_pixel_scope",
            "has_boundary_scope",
            "has_region_scope",
            "has_frame_scope",
            "has_pair_scope",
            "has_batch_scope",
            "has_full_video_scope",
        )
        if vector.get(name) is True
    ]
    tags = [
        str(vector.get("family_id") or "family_unknown"),
        f"stage_{vector.get('entropy_stage_order')}",
        "+".join(scope_parts) or "scope_unknown",
        "archive_bound" if vector.get("archive_bound_exact_handoff_candidate") is True else "archive_unbound",
        "negative_demoted" if vector.get("negative_posterior_demoted") is True else "posterior_active",
        "byte_feasible" if vector.get("byte_credit_feasible") is True else "byte_exhausted",
    ]
    return "|".join(tags)


def _build_interaction_tensor(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cell_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        vector = _interaction_feature_vector(row)
        key = _interaction_cell_key(vector)
        cell = cell_map.setdefault(
            key,
            {
                "schema": "repair_family_stack_interaction_tensor_cell.v1",
                "cell_key": key,
                "family_ids": [],
                "entropy_stage_orders": [],
                "scope_feature_counts": {},
                "row_count": 0,
                "archive_bound_count": 0,
                "archive_bound_contract_ready_count": 0,
                "archive_bound_candidate_contract_count": 0,
                "archive_bound_ready_contract_count": 0,
                "negative_demoted_count": 0,
                "byte_credit_feasible_count": 0,
                "archive_variant_signal_count": 0,
                "archive_variant_probe_count": 0,
                "archive_variant_prototype_count": 0,
                "archive_variant_materializer_backlog_task_count": 0,
                "archive_variant_materializer_byte_closed_task_count": 0,
                "archive_variant_materializer_runtime_adapter_ready_task_count": 0,
                "expected_improvement_sum": 0.0,
                "stack_score_sum": 0.0,
                "max_stack_score": 0.0,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            },
        )
        cell["row_count"] += 1
        cell["family_ids"].append(str(vector.get("family_id")))
        cell["entropy_stage_orders"].append(vector.get("entropy_stage_order"))
        if vector.get("archive_bound_exact_handoff_candidate") is True:
            cell["archive_bound_count"] += 1
        if vector.get("archive_bound_contract_ready") is True:
            cell["archive_bound_contract_ready_count"] += 1
        cell["archive_bound_candidate_contract_count"] += _safe_int(
            vector.get("archive_bound_candidate_contract_count")
        )
        cell["archive_bound_ready_contract_count"] += _safe_int(
            vector.get("archive_bound_ready_contract_count")
        )
        if vector.get("negative_posterior_demoted") is True:
            cell["negative_demoted_count"] += 1
        if vector.get("byte_credit_feasible") is True:
            cell["byte_credit_feasible_count"] += 1
        cell["archive_variant_signal_count"] += _safe_int(
            vector.get("archive_variant_signal_count")
        )
        cell["archive_variant_probe_count"] += _safe_int(
            vector.get("archive_variant_probe_count")
        )
        cell["archive_variant_prototype_count"] += _safe_int(
            vector.get("archive_variant_prototype_count")
        )
        cell["archive_variant_materializer_backlog_task_count"] += _safe_int(
            vector.get("archive_variant_materializer_backlog_task_count")
        )
        cell["archive_variant_materializer_byte_closed_task_count"] += _safe_int(
            vector.get("archive_variant_materializer_byte_closed_task_count")
        )
        cell["archive_variant_materializer_runtime_adapter_ready_task_count"] += _safe_int(
            vector.get("archive_variant_materializer_runtime_adapter_ready_task_count")
        )
        improvement = _safe_float(vector.get("local_mlx_expected_improvement_score_units"))
        stack_score = _safe_float(vector.get("interaction_aware_stack_score"))
        cell["expected_improvement_sum"] += improvement
        cell["stack_score_sum"] += stack_score
        cell["max_stack_score"] = max(_safe_float(cell["max_stack_score"]), stack_score)
        scope_counts = cell["scope_feature_counts"]
        for key_name, value in vector.items():
            if key_name.startswith("has_") and key_name.endswith("_scope") and value is True:
                scope_counts[key_name] = int(scope_counts.get(key_name, 0)) + 1
    cells = []
    for cell in cell_map.values():
        cell["family_ids"] = ordered_unique(cell["family_ids"])
        cell["entropy_stage_orders"] = sorted(
            {int(order) for order in cell["entropy_stage_orders"] if not isinstance(order, bool) and order is not None}
        )
        cell["mean_stack_score"] = float(cell["stack_score_sum"]) / max(1, int(cell["row_count"]))
        cell["mean_expected_improvement"] = float(cell["expected_improvement_sum"]) / max(1, int(cell["row_count"]))
        require_no_truthy_authority_fields(
            cell,
            context=f"repair_family_stack_interaction_tensor_cell:{cell['cell_key']}",
        )
        cells.append(cell)
    cells.sort(
        key=lambda cell: (
            -float(cell.get("max_stack_score") or 0.0),
            -int(cell.get("archive_bound_count") or 0),
            str(cell.get("cell_key") or ""),
        )
    )
    tensor = {
        "schema": "repair_family_stack_interaction_tensor.v1",
        "axis_names": [
            "family_id",
            "entropy_stage_order",
            "fractal_scope",
            "archive_bound_custody",
            "negative_posterior",
            "byte_credit_feasibility",
        ],
        "cell_count": len(cells),
        "cells": cells,
        "acquisition_policy": (
            "prefer_pre_entropy_distribution_shaping_then_archive_bound_custody_"
            "then_positive_mlx_expected_improvement_after_negative_demotion"
        ),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        tensor,
        context="repair_family_stack_interaction_tensor",
    )
    return tensor


def _pairwise_interaction_cell(
    *,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    byte_credit_budget: int | None,
) -> dict[str, Any]:
    left_levels = set(_string_list(left.get("fractal_scope_levels")))
    right_levels = set(_string_list(right.get("fractal_scope_levels")))
    overlap = sorted(left_levels & right_levels, key=_level_sort_key)
    union = sorted(left_levels | right_levels, key=_level_sort_key)
    left_family = str(left.get("family_id") or "unclassified_repair_family")
    right_family = str(right.get("family_id") or "unclassified_repair_family")
    left_stage = _safe_int(left.get("entropy_stage_order"), default=999)
    right_stage = _safe_int(right.get("entropy_stage_order"), default=999)
    stage_gap = right_stage - left_stage
    stage_inversion = right_stage < left_stage
    same_stage_pressure = right_stage == left_stage
    left_bytes = max(0, _safe_int(left.get("delta_payload_bytes")))
    right_bytes = max(0, _safe_int(right.get("delta_payload_bytes")))
    combined_bytes = left_bytes + right_bytes
    combined_improvement = _safe_float(left.get("local_mlx_expected_improvement_score_units")) + _safe_float(
        right.get("local_mlx_expected_improvement_score_units")
    )
    combined_stack_score = _safe_float(left.get("interaction_aware_stack_score")) + _safe_float(
        right.get("interaction_aware_stack_score")
    )
    byte_credit_exhausted = (
        left.get("byte_credit_feasible") is False
        or right.get("byte_credit_feasible") is False
        or (byte_credit_budget is not None and combined_bytes > max(0, byte_credit_budget))
    )
    region_selector_coupling = {
        left_family,
        right_family,
    } == {"segnet_class_region_waterfill", "per_region_selector_codec"}
    region_boundary_coupling = ("region" in left_levels and "boundary" in right_levels) or (
        "boundary" in left_levels and "region" in right_levels
    )
    entropy_boundary_coupling = "entropy_boundary_probe" in {
        left_family,
        right_family,
    } and bool({"bit", "byte"} & (left_levels | right_levels))
    palette_frame0_prior_coupling = "frame0_k16_palette_asymmetry" in {
        left_family,
        right_family,
    } or "palette_frame_asymmetry_prior" in {left_family, right_family}
    pair_batch_spillover = bool({"pair", "batch", "full_video"} & (left_levels | right_levels))
    stage_penalty = 0.24 if stage_inversion else (0.04 if same_stage_pressure else 0.0)
    scope_overlap_penalty = min(0.30, 0.04 * len(overlap))
    spillover_penalty = 0.12 if pair_batch_spillover and overlap else 0.0
    byte_credit_penalty = 0.20 if byte_credit_exhausted else 0.0
    negative_posterior_penalty = 0.0
    if left.get("automatic_negative_result_demoted") is True:
        negative_posterior_penalty += 0.12
    if right.get("automatic_negative_result_demoted") is True:
        negative_posterior_penalty += 0.12
    coupling_synergy = 0.0
    if region_selector_coupling:
        coupling_synergy += 0.08
    if entropy_boundary_coupling:
        coupling_synergy += 0.05
    if palette_frame0_prior_coupling and {"pixel", "frame"} & (left_levels | right_levels):
        coupling_synergy += 0.04
    if (
        left.get("archive_bound_exact_handoff_candidate") is True
        and right.get("archive_bound_exact_handoff_candidate") is True
    ):
        coupling_synergy += 0.03
    total_penalty = _clamp(
        stage_penalty
        + scope_overlap_penalty
        + spillover_penalty
        + byte_credit_penalty
        + negative_posterior_penalty
        - coupling_synergy,
        0.0,
        0.95,
    )
    pair_score = combined_stack_score * (1.0 - total_penalty)
    blockers = ordered_unique(
        [
            *(["entropy_stage_order_inversion_for_candidate_transition"] if stage_inversion else []),
            *(["byte_credit_exhausted_for_pairwise_transition"] if byte_credit_exhausted else []),
            *(["pair_batch_or_full_video_spillover_remeasure_required"] if spillover_penalty else []),
            *(["negative_posterior_transition_penalty_active"] if negative_posterior_penalty else []),
        ]
    )
    cell = {
        "schema": "repair_family_stack_pairwise_interaction_cell.v1",
        "left_stack_row_key": left.get("stack_row_key") or _stack_row_key(left),
        "right_stack_row_key": right.get("stack_row_key") or _stack_row_key(right),
        "left_family_id": left_family,
        "right_family_id": right_family,
        "left_entropy_stage_order": left_stage,
        "right_entropy_stage_order": right_stage,
        "entropy_stage_gap": stage_gap,
        "entropy_stage_inversion": stage_inversion,
        "scope_overlap_levels": overlap,
        "scope_union_levels": union,
        "scope_jaccard": len(overlap) / max(1, len(union)),
        "combined_delta_payload_bytes": combined_bytes,
        "byte_credit_budget": byte_credit_budget,
        "byte_credit_exhausted": byte_credit_exhausted,
        "combined_local_mlx_expected_improvement_score_units": combined_improvement,
        "region_selector_coupling": region_selector_coupling,
        "region_boundary_coupling": region_boundary_coupling,
        "entropy_boundary_coupling": entropy_boundary_coupling,
        "palette_frame0_prior_coupling": palette_frame0_prior_coupling,
        "pair_batch_or_full_video_spillover": pair_batch_spillover,
        "stage_penalty": stage_penalty,
        "scope_overlap_penalty": scope_overlap_penalty,
        "spillover_penalty": spillover_penalty,
        "byte_credit_penalty": byte_credit_penalty,
        "negative_posterior_penalty": negative_posterior_penalty,
        "coupling_synergy": coupling_synergy,
        "total_interaction_penalty": total_penalty,
        "pairwise_acquisition_score": pair_score,
        "transition_allowed_by_tensor": not stage_inversion and not byte_credit_exhausted,
        "interaction_formula": (
            "pair_score=(s_i+s_j)*(1-clamp(p_stage+p_scope+p_spill+p_byte+p_negative-s_synergy,0,0.95))"
        ),
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        cell,
        context=(
            f"repair_family_stack_pairwise_interaction_cell:{cell['left_stack_row_key']}->{cell['right_stack_row_key']}"
        ),
    )
    return cell


def _build_pairwise_interaction_tensor(
    *,
    rows: Sequence[Mapping[str, Any]],
    byte_credit_budget: int | None,
) -> dict[str, Any]:
    cells = [
        _pairwise_interaction_cell(
            left=left,
            right=right,
            byte_credit_budget=byte_credit_budget,
        )
        for left in rows
        for right in rows
        if (left.get("stack_row_key") or _stack_row_key(left)) != (right.get("stack_row_key") or _stack_row_key(right))
    ]
    cells.sort(
        key=lambda cell: (
            -float(cell.get("pairwise_acquisition_score") or 0.0),
            float(cell.get("total_interaction_penalty") or 0.0),
            str(cell.get("left_stack_row_key") or ""),
            str(cell.get("right_stack_row_key") or ""),
        )
    )
    tensor = {
        "schema": "repair_family_stack_pairwise_interaction_tensor.v1",
        "axis_names": [
            "left_family_id",
            "right_family_id",
            "entropy_stage_gap",
            "scope_overlap",
            "region_boundary_coupling",
            "pair_batch_spillover",
            "byte_credit_pressure",
            "negative_posterior_pressure",
        ],
        "cell_count": len(cells),
        "cells": cells,
        "acquisition_rule": (
            "ordered_chain_score_uses_pairwise_stage_scope_byte_credit_negative_"
            "posterior_and_family_coupling_terms_before exact_axis_handoff"
        ),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        tensor,
        context="repair_family_stack_pairwise_interaction_tensor",
    )
    return tensor


def _family_coupling_synergy(
    *,
    family_ids: set[str],
    scope_union: set[str],
    archive_bound_count: int,
    subset_size: int,
) -> float:
    synergy = 0.0
    if {"segnet_class_region_waterfill", "per_region_selector_codec"} <= family_ids:
        synergy += 0.10
    if "entropy_boundary_probe" in family_ids and {"bit", "byte"} & scope_union:
        synergy += 0.06
    if "frame0_k16_palette_asymmetry" in family_ids and {"pixel", "frame"} & scope_union:
        synergy += 0.05
    if {"posenet_null_bottom_decile", "segnet_class_region_waterfill"} <= family_ids:
        synergy += 0.04
    if archive_bound_count == subset_size:
        synergy += 0.04
    synergy += min(0.08, 0.02 * max(0, len(family_ids) - 1))
    return synergy


def _hypergraph_interaction_cell(
    *,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    subset_keys: Sequence[str],
    pair_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    byte_credit_budget: int | None,
) -> dict[str, Any]:
    selected_rows = [rows_by_key[key] for key in subset_keys]
    family_ids = [str(row.get("family_id") or "unclassified_repair_family") for row in selected_rows]
    family_set = set(family_ids)
    stages = [_safe_int(row.get("entropy_stage_order"), default=999) for row in selected_rows]
    entropy_stage_inversion = any(right < left for left, right in pairwise(stages))
    entropy_stage_span = max(stages or [999]) - min(stages or [999])
    scope_sets = [set(_string_list(row.get("fractal_scope_levels"))) for row in selected_rows]
    scope_union = set().union(*scope_sets) if scope_sets else set()
    scope_common = set.intersection(*scope_sets) if scope_sets else set()
    total_bytes = sum(max(0, _safe_int(row.get("delta_payload_bytes"))) for row in selected_rows)
    byte_credit_exhausted = byte_credit_budget is not None and total_bytes > max(0, byte_credit_budget)
    archive_bound_count = sum(
        1 for row in selected_rows if row.get("archive_bound_exact_handoff_candidate") is True
    )
    negative_count = sum(1 for row in selected_rows if row.get("automatic_negative_result_demoted") is True)
    row_score_sum = sum(_safe_float(row.get("interaction_aware_stack_score")) for row in selected_rows)
    expected_improvement_sum = sum(
        _safe_float(row.get("local_mlx_expected_improvement_score_units")) for row in selected_rows
    )
    pair_cells = [
        pair_by_key[(left, right)]
        for left, right in combinations(subset_keys, 2)
        if (left, right) in pair_by_key
    ]
    pair_penalties = [_safe_float(cell.get("total_interaction_penalty")) for cell in pair_cells]
    max_pair_penalty = max(pair_penalties or [0.0])
    mean_pair_penalty = sum(pair_penalties) / max(1, len(pair_penalties))
    same_stage_pressure = len(set(stages)) < len(stages)
    high_scope_spillover = bool({"pair", "batch", "full_video"} & scope_union) and len(scope_common) > 0
    stage_penalty = 0.30 if entropy_stage_inversion else (0.05 if same_stage_pressure else 0.0)
    stage_span_penalty = min(0.12, 0.01 * max(0, entropy_stage_span - 1))
    scope_common_penalty = min(0.22, 0.035 * len(scope_common))
    byte_credit_penalty = 0.22 if byte_credit_exhausted else 0.0
    negative_posterior_penalty = min(0.30, 0.11 * negative_count)
    spillover_penalty = 0.12 if high_scope_spillover else 0.0
    coupling_synergy = _family_coupling_synergy(
        family_ids=family_set,
        scope_union=scope_union,
        archive_bound_count=archive_bound_count,
        subset_size=len(selected_rows),
    )
    total_penalty = _clamp(
        max_pair_penalty * 0.55
        + mean_pair_penalty * 0.25
        + stage_penalty
        + stage_span_penalty
        + scope_common_penalty
        + byte_credit_penalty
        + negative_posterior_penalty
        + spillover_penalty
        - coupling_synergy,
        0.0,
        0.95,
    )
    score = row_score_sum * (1.0 - total_penalty)
    blockers = ordered_unique(
        [
            *(["entropy_stage_order_inversion_for_hyperedge"] if entropy_stage_inversion else []),
            *(["byte_credit_exhausted_for_hyperedge"] if byte_credit_exhausted else []),
            *(["negative_posterior_hyperedge_penalty_active"] if negative_count else []),
            *(["pair_batch_or_full_video_hyperedge_remeasure_required"] if high_scope_spillover else []),
        ]
    )
    cell = {
        "schema": "repair_family_stack_hypergraph_interaction_cell.v1",
        "hyperedge_key": "||".join(subset_keys),
        "hyperedge_order": len(selected_rows),
        "row_keys": list(subset_keys),
        "family_ids": ordered_unique(family_ids),
        "typed_response_ids": [
            str(row.get("typed_response_id") or "typed_response_unknown") for row in selected_rows
        ],
        "entropy_stage_order": stages,
        "entropy_stage_inversion": entropy_stage_inversion,
        "entropy_stage_span": entropy_stage_span,
        "scope_union_levels": sorted(scope_union, key=_level_sort_key),
        "scope_common_levels": sorted(scope_common, key=_level_sort_key),
        "combined_delta_payload_bytes": total_bytes,
        "byte_credit_budget": byte_credit_budget,
        "byte_credit_exhausted": byte_credit_exhausted,
        "archive_bound_count": archive_bound_count,
        "negative_demoted_count": negative_count,
        "combined_local_mlx_expected_improvement_score_units": expected_improvement_sum,
        "row_stack_score_sum": row_score_sum,
        "pairwise_cell_count": len(pair_cells),
        "max_pairwise_interaction_penalty": max_pair_penalty,
        "mean_pairwise_interaction_penalty": mean_pair_penalty,
        "stage_penalty": stage_penalty,
        "stage_span_penalty": stage_span_penalty,
        "scope_common_penalty": scope_common_penalty,
        "byte_credit_penalty": byte_credit_penalty,
        "negative_posterior_penalty": negative_posterior_penalty,
        "spillover_penalty": spillover_penalty,
        "coupling_synergy": coupling_synergy,
        "total_hypergraph_interaction_penalty": total_penalty,
        "hypergraph_acquisition_score": score,
        "transition_allowed_by_tensor": not entropy_stage_inversion and not byte_credit_exhausted,
        "interaction_formula": (
            "hyper_score=sum(s_i)*(1-clamp(0.55*max_pair+0.25*mean_pair+"
            "p_stage+p_span+p_scope+p_byte+p_negative+p_spill-s_synergy,0,0.95))"
        ),
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        cell,
        context=f"repair_family_stack_hypergraph_interaction_cell:{cell['hyperedge_key']}",
    )
    return cell


def _build_hypergraph_interaction_tensor(
    *,
    rows: Sequence[Mapping[str, Any]],
    pairwise_tensor: Mapping[str, Any],
    byte_credit_budget: int | None,
) -> dict[str, Any]:
    row_by_key = {str(row.get("stack_row_key") or _stack_row_key(row)): row for row in rows}
    row_keys = list(row_by_key)
    pair_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for cell in pairwise_tensor.get("cells") or []:
        if isinstance(cell, Mapping):
            pair_by_key[
                (
                    str(cell.get("left_stack_row_key") or ""),
                    str(cell.get("right_stack_row_key") or ""),
                )
            ] = cell
    max_order = min(5, len(row_keys))
    cells = [
        _hypergraph_interaction_cell(
            rows_by_key=row_by_key,
            subset_keys=subset,
            pair_by_key=pair_by_key,
            byte_credit_budget=byte_credit_budget,
        )
        for order in range(2, max_order + 1)
        for subset in combinations(row_keys, order)
    ]
    cells.sort(
        key=lambda cell: (
            -float(cell.get("hypergraph_acquisition_score") or 0.0),
            -int(cell.get("hyperedge_order") or 0),
            float(cell.get("total_hypergraph_interaction_penalty") or 0.0),
            str(cell.get("hyperedge_key") or ""),
        )
    )
    tensor = {
        "schema": "repair_family_stack_hypergraph_interaction_tensor.v1",
        "axis_names": [
            "family_subset",
            "entropy_stage_sequence",
            "scope_union",
            "scope_common",
            "archive_bound_count",
            "byte_credit_pressure",
            "negative_posterior_pressure",
        ],
        "max_hyperedge_order": max_order,
        "cell_count": len(cells),
        "cells": cells,
        "acquisition_rule": (
            "select_n_way_hyperedge_with_monotone_entropy_order_positive_score_"
            "byte_credit_feasibility_archive_custody_and_negative_posterior_penalties"
        ),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        tensor,
        context="repair_family_stack_hypergraph_interaction_tensor",
    )
    return tensor


def _build_hypergraph_stack_acquisition_paths(
    *,
    rows: Sequence[Mapping[str, Any]],
    hypergraph_tensor: Mapping[str, Any],
    byte_credit_budget: int | None,
) -> list[dict[str, Any]]:
    row_by_key = {str(row.get("stack_row_key") or _stack_row_key(row)): row for row in rows}
    best_cell = None
    for cell in hypergraph_tensor.get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        if cell.get("transition_allowed_by_tensor") is not True:
            continue
        if _safe_float(cell.get("hypergraph_acquisition_score")) <= 0.0:
            continue
        best_cell = cell
        break
    if best_cell is None:
        return []
    selected = _string_list(best_cell.get("row_keys"))
    selected_rows = [row_by_key[key] for key in selected if key in row_by_key]
    strict_archive_bound_improvement = any(
        row.get("archive_bound_exact_handoff_candidate") is True
        and _safe_float(row.get("local_mlx_expected_improvement_score_units")) > 0.0
        for row in selected_rows
    )
    all_selected_demoted = bool(selected_rows) and all(
        row.get("automatic_negative_result_demoted") is True for row in selected_rows
    )
    terminal_outcome_class = "precise_exact_axis_blocker"
    if strict_archive_bound_improvement:
        terminal_outcome_class = "strictly_better_archive_bound_candidate_exact_axis_blocked"
    elif all_selected_demoted:
        terminal_outcome_class = "family_demoted_by_posterior_evidence"
    blockers = ordered_unique(
        [
            *[blocker for row in selected_rows for blocker in _string_list(row.get("blockers"))],
            *_string_list(best_cell.get("blockers")),
            "contest_cpu_or_cuda_exact_axis_payload_required",
            "lane_dispatch_claim_required_before_exact_eval",
        ]
    )
    path = {
        "schema": "repair_family_stack_acquisition_path.v1",
        "path_id": "primary_hypergraph_tensor_path",
        "path_kind": "n_way_hypergraph_interaction_tensor_acquisition",
        "source_hyperedge_key": best_cell.get("hyperedge_key"),
        "source_hyperedge_order": best_cell.get("hyperedge_order"),
        "row_keys": selected,
        "family_order": [str(row.get("family_id") or "") for row in selected_rows],
        "typed_response_order": [str(row.get("typed_response_id") or "") for row in selected_rows],
        "entropy_stage_order": [_safe_int(row.get("entropy_stage_order"), default=999) for row in selected_rows],
        "total_delta_payload_bytes": best_cell.get("combined_delta_payload_bytes"),
        "byte_credit_budget": byte_credit_budget,
        "byte_credit_remaining": None
        if byte_credit_budget is None
        else max(0, byte_credit_budget - _safe_int(best_cell.get("combined_delta_payload_bytes"))),
        "total_local_mlx_expected_improvement_score_units": (
            best_cell.get("combined_local_mlx_expected_improvement_score_units")
        ),
        "total_interaction_aware_stack_score": best_cell.get("row_stack_score_sum"),
        "max_pairwise_interaction_penalty": best_cell.get("max_pairwise_interaction_penalty"),
        "max_hypergraph_interaction_penalty": best_cell.get("total_hypergraph_interaction_penalty"),
        "mean_pairwise_acquisition_score": None,
        "hypergraph_acquisition_score": best_cell.get("hypergraph_acquisition_score"),
        "archive_bound_candidate_count": best_cell.get("archive_bound_count"),
        "strict_archive_bound_local_improvement_candidate": strict_archive_bound_improvement,
        "terminal_outcome_class": terminal_outcome_class,
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_family_n_way_stack_acquisition_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        path,
        context="repair_family_hypergraph_stack_acquisition_path",
    )
    return [path]


def _build_fractal_marginal_surface(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        levels = _string_list(row.get("fractal_scope_levels")) or ["scope_unknown"]
        stage = _safe_int(row.get("entropy_stage_order"), default=999)
        for level in levels:
            key = f"{level}|stage_{stage}"
            cell = cells_by_key.setdefault(
                key,
                {
                    "schema": "repair_family_fractal_marginal_surface_cell.v1",
                    "cell_key": key,
                    "level": level,
                    "entropy_stage_order": stage,
                    "family_ids": [],
                    "row_keys": [],
                    "row_count": 0,
                    "archive_bound_count": 0,
                    "negative_demoted_count": 0,
                    "byte_credit_exhausted_count": 0,
                    "delta_payload_bytes_sum": 0,
                    "expected_improvement_sum": 0.0,
                    "stack_score_sum": 0.0,
                    "stackability_penalty_sum": 0.0,
                    "measured_mlx_marginal_updates": [],
                    "budget_spend_allowed": False,
                    "ready_for_exact_eval_dispatch": False,
                    **FALSE_AUTHORITY,
                },
            )
            row_key = str(row.get("stack_row_key") or _stack_row_key(row))
            delta_payload_bytes = max(0, _safe_int(row.get("delta_payload_bytes")))
            expected_improvement = _safe_float(
                row.get("local_mlx_expected_improvement_score_units")
            )
            measured_update = {
                "schema": "repair_family_measured_mlx_marginal_update.v1",
                "stack_row_key": row_key,
                "family_id": row.get("family_id"),
                "typed_response_id": row.get("typed_response_id"),
                "entropy_stage_order": stage,
                "fractal_level": level,
                "delta_payload_bytes": delta_payload_bytes,
                "local_mlx_expected_improvement_score_units": expected_improvement,
                "local_mlx_combined_delta_score_units": row.get(
                    "local_mlx_combined_delta_score_units"
                ),
                "measured_improvement_per_byte": expected_improvement
                / max(1, delta_payload_bytes),
                "interaction_aware_stack_score": _safe_float(
                    row.get("interaction_aware_stack_score")
                ),
                "source_execution_report": row.get("source_execution_report"),
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            }
            require_no_truthy_authority_fields(
                measured_update,
                context=f"repair_family_measured_mlx_marginal_update:{row_key}:{level}",
            )
            cell["family_ids"].append(str(row.get("family_id") or "unclassified_repair_family"))
            cell["row_keys"].append(row_key)
            cell["row_count"] += 1
            if row.get("archive_bound_exact_handoff_candidate") is True:
                cell["archive_bound_count"] += 1
            if row.get("automatic_negative_result_demoted") is True:
                cell["negative_demoted_count"] += 1
            if row.get("byte_credit_feasible") is False:
                cell["byte_credit_exhausted_count"] += 1
            cell["delta_payload_bytes_sum"] += delta_payload_bytes
            cell["expected_improvement_sum"] += expected_improvement
            cell["stack_score_sum"] += _safe_float(row.get("interaction_aware_stack_score"))
            cell["stackability_penalty_sum"] += _safe_float(row.get("stackability_penalty"))
            cell["measured_mlx_marginal_updates"].append(measured_update)
    cells = []
    for cell in cells_by_key.values():
        row_count = max(1, int(cell["row_count"]))
        bytes_sum = max(1, int(cell["delta_payload_bytes_sum"]))
        cell["family_ids"] = ordered_unique(cell["family_ids"])
        cell["row_keys"] = ordered_unique(cell["row_keys"])
        cell["mean_expected_improvement"] = float(cell["expected_improvement_sum"]) / row_count
        cell["mean_stack_score"] = float(cell["stack_score_sum"]) / row_count
        cell["mean_stackability_penalty"] = float(cell["stackability_penalty_sum"]) / row_count
        cell["marginal_improvement_per_byte"] = float(cell["expected_improvement_sum"]) / bytes_sum
        cell["marginal_stack_score_per_byte"] = float(cell["stack_score_sum"]) / bytes_sum
        cell["measured_mlx_marginal_update_count"] = len(
            cell["measured_mlx_marginal_updates"]
        )
        cell["selection_pressure"] = (
            float(cell["marginal_stack_score_per_byte"])
            * (1.0 - min(0.95, float(cell["mean_stackability_penalty"])))
            * (1.0 - min(0.80, int(cell["negative_demoted_count"]) / row_count))
        )
        require_no_truthy_authority_fields(
            cell,
            context=f"repair_family_fractal_marginal_surface_cell:{cell['cell_key']}",
        )
        cells.append(cell)
    cells.sort(
        key=lambda cell: (
            -float(cell.get("selection_pressure") or 0.0),
            _LEVEL_RANK.get(str(cell.get("level") or ""), len(_LEVEL_ORDER)),
            int(cell.get("entropy_stage_order") or 999),
            str(cell.get("cell_key") or ""),
        )
    )
    surface = {
        "schema": "repair_family_fractal_marginal_surface.v1",
        "level_order": list(_LEVEL_ORDER),
        "cell_count": len(cells),
        "measured_mlx_marginal_update_count": sum(
            int(cell.get("measured_mlx_marginal_update_count") or 0)
            for cell in cells
        ),
        "cells": cells,
        "acquisition_rule": (
            "rank_level_stage_marginals_by_improvement_per_byte_stack_penalty_"
            "negative_demotion_and_byte_credit_pressure"
        ),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        surface,
        context="repair_family_fractal_marginal_surface",
    )
    return surface


def _build_measured_mlx_posterior_budget_routing_updates(
    fractal_surface: Mapping[str, Any],
) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for cell in fractal_surface.get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        for measured in cell.get("measured_mlx_marginal_updates") or []:
            if not isinstance(measured, Mapping):
                continue
            improvement = _safe_float(
                measured.get("local_mlx_expected_improvement_score_units")
            )
            delta_bytes = max(0, _safe_int(measured.get("delta_payload_bytes")))
            if improvement > 0.0:
                direction = "increase"
                hint = "route_measured_positive_mlx_marginal_to_archive_bound_materializer"
                policy = "materialize_archive_bound_candidate_for_measured_mlx_marginal"
                demote = False
            elif improvement < 0.0:
                direction = "decrease"
                hint = "demote_family_stage_scope_from_negative_mlx_marginal"
                policy = "demote_negative_measured_mlx_marginal_before_more_budget"
                demote = True
            elif delta_bytes == 0:
                direction = "hold"
                hint = "remeasure_zero_byte_mlx_marginal_before_budget_routing"
                policy = "remeasure_zero_byte_measured_mlx_marginal"
                demote = False
            else:
                direction = "hold"
                hint = "hold_measured_neutral_mlx_marginal_until_new_signal"
                policy = "hold_neutral_measured_mlx_marginal"
                demote = False
            update = {
                "schema": "repair_family_measured_mlx_posterior_budget_routing_update.v1",
                "stack_row_key": measured.get("stack_row_key"),
                "family_id": measured.get("family_id"),
                "typed_response_id": measured.get("typed_response_id"),
                "fractal_level": measured.get("fractal_level") or cell.get("level"),
                "entropy_stage_order": measured.get("entropy_stage_order"),
                "cell_key": cell.get("cell_key"),
                "delta_payload_bytes": delta_bytes,
                "local_mlx_expected_improvement_score_units": improvement,
                "measured_improvement_per_byte": _safe_float(
                    measured.get("measured_improvement_per_byte")
                ),
                "interaction_aware_stack_score": _safe_float(
                    measured.get("interaction_aware_stack_score")
                ),
                "family_priority_direction": direction,
                "recommended_acquisition_policy": policy,
                "posterior_budget_routing_hint": hint,
                "demote_responsible_family_stage_scope": demote,
                "source_execution_report": measured.get("source_execution_report"),
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            }
            require_no_truthy_authority_fields(
                update,
                context=(
                    "repair_family_measured_mlx_posterior_budget_routing_update:"
                    f"{update['stack_row_key']}:{update['fractal_level']}"
                ),
            )
            updates.append(update)
    updates.sort(
        key=lambda item: (
            item.get("family_priority_direction") != "increase",
            -float(item.get("measured_improvement_per_byte") or 0.0),
            int(item.get("entropy_stage_order") or 999),
            str(item.get("family_id") or ""),
            str(item.get("typed_response_id") or ""),
        )
    )
    return updates


def _build_stack_acquisition_frontier(
    *,
    rows: Sequence[Mapping[str, Any]],
    hypergraph_tensor: Mapping[str, Any],
    byte_credit_budget: int | None,
    max_paths: int = 8,
) -> list[dict[str, Any]]:
    row_by_key = {str(row.get("stack_row_key") or _stack_row_key(row)): row for row in rows}
    frontier: list[dict[str, Any]] = []
    for cell in hypergraph_tensor.get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        if cell.get("transition_allowed_by_tensor") is not True:
            continue
        if _safe_float(cell.get("hypergraph_acquisition_score")) <= 0.0:
            continue
        row_keys = [key for key in _string_list(cell.get("row_keys")) if key in row_by_key]
        if not row_keys:
            continue
        selected_rows = [row_by_key[key] for key in row_keys]
        strict_archive_bound_improvement = any(
            row.get("archive_bound_exact_handoff_candidate") is True
            and _safe_float(row.get("local_mlx_expected_improvement_score_units")) > 0.0
            for row in selected_rows
        )
        all_selected_demoted = bool(selected_rows) and all(
            row.get("automatic_negative_result_demoted") is True for row in selected_rows
        )
        terminal_outcome_class = "precise_exact_axis_blocker"
        if strict_archive_bound_improvement:
            terminal_outcome_class = "strictly_better_archive_bound_candidate_exact_axis_blocked"
        elif all_selected_demoted:
            terminal_outcome_class = "family_demoted_by_posterior_evidence"
        path = {
            "schema": "repair_family_stack_acquisition_frontier_path.v1",
            "frontier_rank": len(frontier) + 1,
            "source_tensor": "hypergraph_interaction_tensor",
            "source_hyperedge_key": cell.get("hyperedge_key"),
            "source_hyperedge_order": cell.get("hyperedge_order"),
            "row_keys": row_keys,
            "family_order": [str(row.get("family_id") or "") for row in selected_rows],
            "typed_response_order": [str(row.get("typed_response_id") or "") for row in selected_rows],
            "entropy_stage_order": [_safe_int(row.get("entropy_stage_order"), default=999) for row in selected_rows],
            "fractal_scope_union_levels": cell.get("scope_union_levels"),
            "combined_delta_payload_bytes": cell.get("combined_delta_payload_bytes"),
            "byte_credit_budget": byte_credit_budget,
            "byte_credit_remaining": None
            if byte_credit_budget is None
            else max(0, byte_credit_budget - _safe_int(cell.get("combined_delta_payload_bytes"))),
            "hypergraph_acquisition_score": cell.get("hypergraph_acquisition_score"),
            "total_hypergraph_interaction_penalty": cell.get("total_hypergraph_interaction_penalty"),
            "archive_bound_candidate_count": cell.get("archive_bound_count"),
            "negative_demoted_count": cell.get("negative_demoted_count"),
            "terminal_outcome_class": terminal_outcome_class,
            "blockers": ordered_unique(
                [
                    *[blocker for row in selected_rows for blocker in _string_list(row.get("blockers"))],
                    *_string_list(cell.get("blockers")),
                    "contest_cpu_or_cuda_exact_axis_payload_required",
                    "lane_dispatch_claim_required_before_exact_eval",
                ]
            ),
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            "allowed_use": "repair_family_ranked_stack_acquisition_frontier_only",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            path,
            context=f"repair_family_stack_acquisition_frontier_path:{path['frontier_rank']}",
        )
        frontier.append(path)
        if len(frontier) >= max_paths:
            break
    return frontier


def _build_stack_acquisition_paths(
    *,
    rows: Sequence[Mapping[str, Any]],
    pairwise_tensor: Mapping[str, Any],
    byte_credit_budget: int | None,
) -> list[dict[str, Any]]:
    row_by_key = {str(row.get("stack_row_key") or _stack_row_key(row)): row for row in rows}
    pair_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for cell in pairwise_tensor.get("cells") or []:
        if not isinstance(cell, Mapping):
            continue
        pair_by_key[
            (
                str(cell.get("left_stack_row_key") or ""),
                str(cell.get("right_stack_row_key") or ""),
            )
        ] = cell
    remaining = list(row_by_key)
    selected: list[str] = []
    selected_cells: list[Mapping[str, Any]] = []
    used_bytes = 0
    while remaining:
        best_key = None
        best_score = float("-inf")
        best_cells: list[Mapping[str, Any]] = []
        min_remaining_stage = min(
            _safe_int(row_by_key[key].get("entropy_stage_order"), default=999) for key in remaining
        )
        max_selected_stage = max(
            [_safe_int(row_by_key[key].get("entropy_stage_order"), default=999) for key in selected] or [-1]
        )
        for key in remaining:
            row = row_by_key[key]
            row_stage = _safe_int(row.get("entropy_stage_order"), default=999)
            if not selected and row_stage != min_remaining_stage:
                continue
            if selected and row_stage < max_selected_stage:
                continue
            row_bytes = max(0, _safe_int(row.get("delta_payload_bytes")))
            if byte_credit_budget is not None and used_bytes + row_bytes > byte_credit_budget:
                continue
            candidate_cells = [
                pair_by_key[(selected_key, key)] for selected_key in selected if (selected_key, key) in pair_by_key
            ]
            transition_penalty = max(
                [_safe_float(cell.get("total_interaction_penalty")) for cell in candidate_cells] or [0.0]
            )
            transition_score = _safe_float(row.get("interaction_aware_stack_score")) * (
                1.0 - min(0.95, transition_penalty)
            )
            if not selected:
                transition_score = _safe_float(row.get("interaction_aware_stack_score"))
            if transition_score > best_score:
                best_key = key
                best_score = transition_score
                best_cells = candidate_cells
        if best_key is None:
            break
        if selected and best_score <= 0.0:
            break
        selected.append(best_key)
        selected_cells.extend(best_cells)
        used_bytes += max(0, _safe_int(row_by_key[best_key].get("delta_payload_bytes")))
        remaining.remove(best_key)
    selected_rows = [row_by_key[key] for key in selected]
    strict_archive_bound_improvement = any(
        row.get("archive_bound_exact_handoff_candidate") is True
        and _safe_float(row.get("local_mlx_expected_improvement_score_units")) > 0.0
        for row in selected_rows
    )
    all_selected_demoted = bool(selected_rows) and all(
        row.get("automatic_negative_result_demoted") is True for row in selected_rows
    )
    terminal_outcome_class = "precise_exact_axis_blocker"
    if strict_archive_bound_improvement:
        terminal_outcome_class = "strictly_better_archive_bound_candidate_exact_axis_blocked"
    elif all_selected_demoted:
        terminal_outcome_class = "family_demoted_by_posterior_evidence"
    blockers = ordered_unique(
        [
            *[blocker for row in selected_rows for blocker in _string_list(row.get("blockers"))],
            *[blocker for cell in selected_cells for blocker in _string_list(cell.get("blockers"))],
            "contest_cpu_or_cuda_exact_axis_payload_required",
            "lane_dispatch_claim_required_before_exact_eval",
        ]
    )
    path = {
        "schema": "repair_family_stack_acquisition_path.v1",
        "path_id": "primary_pairwise_tensor_path",
        "path_kind": "greedy_pairwise_interaction_tensor_acquisition",
        "row_keys": selected,
        "family_order": [str(row.get("family_id") or "") for row in selected_rows],
        "typed_response_order": [str(row.get("typed_response_id") or "") for row in selected_rows],
        "entropy_stage_order": [_safe_int(row.get("entropy_stage_order"), default=999) for row in selected_rows],
        "total_delta_payload_bytes": used_bytes,
        "byte_credit_budget": byte_credit_budget,
        "byte_credit_remaining": None if byte_credit_budget is None else max(0, byte_credit_budget - used_bytes),
        "total_local_mlx_expected_improvement_score_units": sum(
            _safe_float(row.get("local_mlx_expected_improvement_score_units")) for row in selected_rows
        ),
        "total_interaction_aware_stack_score": sum(
            _safe_float(row.get("interaction_aware_stack_score")) for row in selected_rows
        ),
        "max_pairwise_interaction_penalty": max(
            [_safe_float(cell.get("total_interaction_penalty")) for cell in selected_cells] or [0.0]
        ),
        "mean_pairwise_acquisition_score": (
            sum(_safe_float(cell.get("pairwise_acquisition_score")) for cell in selected_cells)
            / max(1, len(selected_cells))
        ),
        "archive_bound_candidate_count": sum(
            1 for row in selected_rows if row.get("archive_bound_exact_handoff_candidate") is True
        ),
        "strict_archive_bound_local_improvement_candidate": strict_archive_bound_improvement,
        "terminal_outcome_class": terminal_outcome_class,
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_family_pairwise_stack_acquisition_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        path,
        context="repair_family_stack_acquisition_path",
    )
    return [path]


def _build_posterior_acquisition_surface(
    posterior_demotions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    family_priors = []
    for family_id, prior in sorted(posterior_demotions.items()):
        family_priors.append(
            {
                "schema": "repair_family_stack_posterior_acquisition_prior.v1",
                "family_id": family_id,
                "observation_count": _safe_int(prior.get("observation_count")),
                "negative_result_count": _safe_int(prior.get("negative_result_count")),
                "positive_result_count": _safe_int(prior.get("positive_result_count")),
                "receiver_credit_exhausted_count": _safe_int(prior.get("receiver_credit_exhausted_count")),
                "stackability_remeasure_required_count": _safe_int(prior.get("stackability_remeasure_required_count")),
                "entropy_stage_contract_miss_count": _safe_int(prior.get("entropy_stage_contract_miss_count")),
                "mean_expected_improvement": _safe_float(prior.get("mean_expected_improvement")),
                "mean_improvement_per_byte": _safe_float(prior.get("mean_improvement_per_byte")),
                "demotion_multiplier": _safe_float(prior.get("demotion_multiplier")),
                "acquisition_multiplier": _safe_float(prior.get("acquisition_multiplier")),
                "policies": _string_list(prior.get("policies")),
                "budget_routing_hints": _string_list(prior.get("budget_routing_hints")),
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            }
        )
    surface = {
        "schema": "repair_family_stack_posterior_acquisition_surface.v1",
        "family_prior_count": len(family_priors),
        "family_priors": family_priors,
        "acquisition_rule": (
            "positive_local_mlx_priors_can_raise_order_but_negative_results_"
            "byte_credit_exhaustion_entropy_stage_misses_and_stackability_"
            "remeasure_flags_reduce_or_reroute_budget"
        ),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        surface,
        context="repair_family_stack_posterior_acquisition_surface",
    )
    return surface


def _budget_routing_decision(
    *,
    rows: Sequence[Mapping[str, Any]],
    posterior_surface: Mapping[str, Any],
    archive_bound_handoff_count: int,
    primary_acquisition_path: Mapping[str, Any] | None = None,
    measured_mlx_budget_updates: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    positive_measured_update_count = sum(
        1
        for update in measured_mlx_budget_updates
        if _safe_float(update.get("local_mlx_expected_improvement_score_units")) > 0.0
    )
    demoted_measured_update_count = sum(
        1
        for update in measured_mlx_budget_updates
        if update.get("demote_responsible_family_stage_scope") is True
    )
    if not rows:
        route = "materialize_repair_family_byte_transform_reports"
        blocker = "repair_family_byte_transform_execution_reports_missing"
        priority = 100
    elif (
        primary_acquisition_path is not None
        and primary_acquisition_path.get("terminal_outcome_class")
        == "strictly_better_archive_bound_candidate_exact_axis_blocked"
    ):
        route = "bridge_strictly_better_archive_bound_candidate_to_exact_ready_input"
        blocker = "contest_cpu_or_cuda_exact_axis_payload_required"
        priority = 98
    elif archive_bound_handoff_count > 0:
        route = "bridge_archive_bound_candidate_to_exact_ready_input"
        blocker = "contest_cpu_or_cuda_exact_axis_payload_required"
        priority = 96
    elif any(row.get("byte_credit_feasible") is False for row in rows):
        route = "rebudget_receiver_closed_credit_before_more_repair"
        blocker = "byte_credit_exhausted_for_stack_row"
        priority = 92
    elif all(row.get("automatic_negative_result_demoted") is True for row in rows):
        route = "demote_repair_family_until_new_component_signal"
        blocker = "automatic_negative_result_demotion_active"
        priority = 88
    elif any(_string_list(row.get("archive_entropy_substrate_blockers")) for row in rows):
        route = "materialize_missing_archive_entropy_substrate_variant"
        blocker = "archive_entropy_substrate_materializer_gap"
        priority = 90
    elif positive_measured_update_count > 0:
        route = "materialize_archive_bound_candidate_for_measured_mlx_marginal"
        blocker = "archive_bound_candidate_required_for_measured_mlx_marginal"
        priority = 87
    elif any(
        _safe_int(prior.get("entropy_stage_contract_miss_count")) > 0
        for prior in posterior_surface.get("family_priors") or []
        if isinstance(prior, Mapping)
    ):
        route = "rebuild_entropy_stage_chain_contract"
        blocker = "posterior_entropy_stage_contract_miss_active"
        priority = 86
    else:
        route = "run_next_byte_closed_materializer_or_mlx_probe"
        blocker = "exact_axis_required_before_score_or_budget"
        priority = 80
    decision = {
        "schema": "repair_family_stack_budget_routing_decision.v1",
        "activation_action": route,
        "priority_score": priority,
        "selected_blocker_class": blocker,
        "archive_bound_handoff_count": archive_bound_handoff_count,
        "measured_mlx_posterior_budget_routing_update_count": len(
            measured_mlx_budget_updates
        ),
        "positive_measured_mlx_budget_update_count": positive_measured_update_count,
        "demoted_measured_mlx_budget_update_count": demoted_measured_update_count,
        "top_measured_mlx_budget_routing_updates": [
            dict(update) for update in measured_mlx_budget_updates[:8]
        ],
        "primary_acquisition_path_id": None
        if primary_acquisition_path is None
        else primary_acquisition_path.get("path_id"),
        "primary_terminal_outcome_class": None
        if primary_acquisition_path is None
        else primary_acquisition_path.get("terminal_outcome_class"),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_family_local_acquisition_routing_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        decision,
        context="repair_family_stack_budget_routing_decision",
    )
    return decision


def _candidate_archive_record(
    *,
    report: Mapping[str, Any],
    repo_root: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    archive_bound_contract, contract_blockers = _archive_bound_contract_for_payload(
        report,
        label=f"repair_family_exact_handoff_candidate_archive:{report.get('candidate_chain_id')}",
    )
    if contract_blockers:
        return {
            "schema": "repair_family_exact_handoff_candidate_archive_custody.v1",
            "path": None,
            "expected_sha256": None,
            "expected_bytes": None,
            "present": False,
            "sha256": None,
            "bytes": None,
            "sha256_matches": False,
            "bytes_match": False,
            "custody_complete": False,
            "blockers": ordered_unique(contract_blockers),
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }, list(contract_blockers)
    candidate = _mapping(
        archive_bound_contract.get("candidate_archive")
    ) or _mapping(report.get("candidate_archive"))
    path_text = str(candidate.get("path") or "").strip()
    expected_sha = str(candidate.get("sha256") or "").strip()
    expected_bytes = candidate.get("bytes")
    blockers: list[str] = []
    present = False
    sha256 = None
    byte_count = None
    sha256_matches = False
    bytes_match = False
    if not path_text:
        blockers.append("candidate_archive_path_missing")
    else:
        path = _resolve(path_text, repo_root)
        present = path.is_file()
        if not present:
            blockers.append("candidate_archive_file_missing")
        else:
            sha256 = sha256_file(path)
            byte_count = path.stat().st_size
            sha256_matches = bool(expected_sha and sha256 == expected_sha)
            if not expected_sha:
                blockers.append("candidate_archive_sha256_missing")
            elif not sha256_matches:
                blockers.append("candidate_archive_sha256_mismatch")
            bytes_match = (
                isinstance(expected_bytes, int)
                and not isinstance(expected_bytes, bool)
                and byte_count == expected_bytes
            )
            if expected_bytes is None:
                blockers.append("candidate_archive_bytes_missing")
            elif not bytes_match:
                blockers.append("candidate_archive_bytes_mismatch")
    return {
        "schema": "repair_family_exact_handoff_candidate_archive_custody.v1",
        "path": path_text or None,
        "expected_sha256": expected_sha or None,
        "expected_bytes": expected_bytes,
        "present": present,
        "sha256": sha256,
        "bytes": byte_count,
        "sha256_matches": sha256_matches,
        "bytes_match": bytes_match,
        "custody_complete": bool(present and sha256_matches and bytes_match),
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }, blockers


def _runtime_proof_record(
    *,
    report: Mapping[str, Any],
    repo_root: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    archive_bound_contract, contract_blockers = _archive_bound_contract_for_payload(
        report,
        label=f"repair_family_exact_handoff_runtime_proof:{report.get('candidate_chain_id')}",
    )
    if contract_blockers:
        return {
            "schema": "repair_family_exact_handoff_runtime_proof_custody.v1",
            "path": None,
            "present": False,
            "sha256": None,
            "bytes": None,
            "archive_bound_runtime_consumption_proof_ready": False,
            "custody_complete": False,
            "blockers": ordered_unique(contract_blockers),
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }, list(contract_blockers)
    candidate = _mapping(
        archive_bound_contract.get("candidate_archive")
    ) or _mapping(report.get("candidate_archive"))
    path_text = str(
        candidate.get("runtime_consumption_proof_path") or report.get("runtime_consumption_proof_path") or ""
    ).strip()
    blockers: list[str] = []
    present = False
    sha256 = None
    byte_count = None
    if not path_text:
        blockers.append("runtime_consumption_proof_path_missing")
    else:
        path = _resolve(path_text, repo_root)
        present = path.is_file()
        if not present:
            blockers.append("runtime_consumption_proof_file_missing")
        else:
            sha256 = sha256_file(path)
            byte_count = path.stat().st_size
    proof_ready = (
        archive_bound_contract.get("runtime_consumption_proof_ready") is True
        or candidate.get("runtime_consumption_proof_ready") is True
        or _mapping(report.get("exact_eval_handoff_gate")).get("archive_bound_runtime_consumption_proof_ready") is True
    )
    if not proof_ready:
        blockers.append("archive_bound_runtime_consumption_proof_missing")
    return {
        "schema": "repair_family_exact_handoff_runtime_proof_custody.v1",
        "path": path_text or None,
        "present": present,
        "sha256": sha256,
        "bytes": byte_count,
        "archive_bound_runtime_consumption_proof_ready": proof_ready,
        "custody_complete": bool(present and proof_ready),
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }, blockers


def _exact_handoff_candidate_row(
    *,
    report: Mapping[str, Any],
    report_path: str | Path,
    stack_row: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    archive_bound_contract, archive_bound_contract_reader_blockers = (
        _archive_bound_contract_for_payload(
            report,
            label=f"repair_family_exact_handoff_candidate_contract:{report.get('candidate_chain_id')}",
        )
    )
    archive_bound_contract_surface = _validated_archive_bound_contract_surface(
        report,
        contract_blockers=archive_bound_contract_reader_blockers,
    )
    candidate_archive, archive_blockers = _candidate_archive_record(
        report=report,
        repo_root=repo_root,
    )
    runtime_proof, proof_blockers = _runtime_proof_record(
        report=report,
        repo_root=repo_root,
    )
    exact_gate = _mapping(report.get("exact_eval_handoff_gate"))
    archive_bound_complete = bool(
        report.get("byte_closed_candidate_emitted") is True
        and report.get("candidate_archive_materialized") is True
        and candidate_archive.get("custody_complete") is True
        and runtime_proof.get("custody_complete") is True
    )
    blockers = ordered_unique(
        [
            *_string_list(report.get("blockers")),
            *archive_bound_contract_reader_blockers,
            *_string_list(exact_gate.get("blockers")),
            *archive_blockers,
            *proof_blockers,
            *([] if report.get("byte_closed_candidate_emitted") is True else ["byte_closed_candidate_archive_missing"]),
            *([] if archive_bound_complete else ["archive_runtime_custody_incomplete"]),
            "contest_cpu_or_cuda_exact_axis_payload_required",
            "lane_dispatch_claim_required_before_exact_eval",
        ]
    )
    row = {
        "schema": REPAIR_FAMILY_EXACT_HANDOFF_CANDIDATE_ROW_SCHEMA,
        "source_execution_report": _report_file_record(
            report_path,
            repo_root=repo_root,
        ),
        "source_stack_order": stack_row.get("planned_stack_order"),
        "family_id": report.get("family_id"),
        "typed_response_id": report.get("typed_response_id"),
        "candidate_chain_id": report.get("candidate_chain_id"),
        "candidate_chain_ids": _string_list(report.get("candidate_chain_ids")),
        "entropy_position_label": report.get("entropy_position_label"),
        "entropy_stage_order": stack_row.get("entropy_stage_order"),
        "candidate_archive": candidate_archive,
        "archive_bound_candidate_contract": dict(archive_bound_contract),
        "archive_bound_candidate_contract_surface": dict(archive_bound_contract_surface),
        "runtime_consumption_proof": runtime_proof,
        "archive_bound_custody_complete": archive_bound_complete,
        "archive_bound_exact_handoff_candidate": archive_bound_complete,
        "target_modes": ["contest_exact_eval"],
        "exact_axis_required": ["contest-CPU", "contest-CUDA"],
        "component_response_axis": "[macOS-MLX research-signal]",
        "local_mlx_rows_are_advisory_only": True,
        "eligible_for_exact_eval_handoff": False,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "archive_bound_exact_handoff_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        row,
        context=f"repair_family_exact_handoff_candidate_row:{report.get('family_id')}",
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
        raise RepairFamilyStackSearchError("execution_reports and execution_report_paths length mismatch")
    posterior_rows = (
        load_repair_campaign_stackability_posterior_rows(posterior_path) if posterior_path is not None else []
    )
    demotions = _posterior_demotions(posterior_rows)
    remaining_budget = byte_credit_budget
    row_packages: list[tuple[dict[str, Any], Mapping[str, Any], str | Path]] = []
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
        row_packages.append((row, report, path))
    row_packages.sort(
        key=lambda package: (
            package[0]["entropy_stage_order"],
            package[0]["automatic_negative_result_demoted"],
            -float(package[0]["interaction_aware_stack_score"]),
            package[0]["order_hint"],
            str(package[0].get("typed_response_id") or ""),
        )
    )
    for index, (row, _report, _path) in enumerate(row_packages, start=1):
        row["planned_stack_order"] = index
        row["interaction_feature_vector"] = _interaction_feature_vector(row)
    rows = [row for row, _report, _path in row_packages]
    interaction_tensor = _build_interaction_tensor(rows)
    pairwise_interaction_tensor = _build_pairwise_interaction_tensor(
        rows=rows,
        byte_credit_budget=byte_credit_budget,
    )
    hypergraph_interaction_tensor = _build_hypergraph_interaction_tensor(
        rows=rows,
        pairwise_tensor=pairwise_interaction_tensor,
        byte_credit_budget=byte_credit_budget,
    )
    hypergraph_acquisition_paths = _build_hypergraph_stack_acquisition_paths(
        rows=rows,
        hypergraph_tensor=hypergraph_interaction_tensor,
        byte_credit_budget=byte_credit_budget,
    )
    fractal_marginal_surface = _build_fractal_marginal_surface(rows)
    measured_mlx_budget_updates = _build_measured_mlx_posterior_budget_routing_updates(
        fractal_marginal_surface
    )
    stack_acquisition_frontier = _build_stack_acquisition_frontier(
        rows=rows,
        hypergraph_tensor=hypergraph_interaction_tensor,
        byte_credit_budget=byte_credit_budget,
    )
    pairwise_acquisition_paths = _build_stack_acquisition_paths(
        rows=rows,
        pairwise_tensor=pairwise_interaction_tensor,
        byte_credit_budget=byte_credit_budget,
    )
    acquisition_paths = [
        *hypergraph_acquisition_paths,
        *pairwise_acquisition_paths,
    ]
    primary_acquisition_path = acquisition_paths[0] if acquisition_paths else None
    selected_row_keys = set(
        _string_list(None if primary_acquisition_path is None else primary_acquisition_path.get("row_keys"))
    )
    for row in rows:
        row["selected_for_materialization_handoff"] = (
            str(row.get("stack_row_key") or _stack_row_key(row)) in selected_row_keys
        )
    exact_handoff_candidates = [
        _exact_handoff_candidate_row(
            report=report,
            report_path=path,
            stack_row=row,
            repo_root=repo_root,
        )
        for row, report, path in row_packages
    ]
    archive_bound_handoff_count = sum(
        1 for row in exact_handoff_candidates if row.get("archive_bound_custody_complete") is True
    )
    exact_handoff_gate_blockers = ordered_unique(
        [
            *([] if archive_bound_handoff_count else ["byte_closed_archive_runtime_receiver_proof_required_per_stack"]),
            "contest_cpu_or_cuda_exact_axis_payload_required",
            "lane_dispatch_claim_required_before_exact_eval",
        ]
    )
    posterior_acquisition_surface = _build_posterior_acquisition_surface(demotions)
    budget_routing_decision = _budget_routing_decision(
        rows=rows,
        posterior_surface=posterior_acquisition_surface,
        archive_bound_handoff_count=archive_bound_handoff_count,
        primary_acquisition_path=primary_acquisition_path,
        measured_mlx_budget_updates=measured_mlx_budget_updates,
    )
    plan = {
        "schema": REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA,
        "execution_report_count": len(rows),
        "posterior_path": None if posterior_path is None else str(posterior_path),
        "posterior_row_count": len(posterior_rows),
        "posterior_acquisition_surface": posterior_acquisition_surface,
        "byte_credit_budget": byte_credit_budget,
        "automatic_negative_result_demotion_enabled": True,
        "entropy_ordering_rule": (
            "before_entropy_distribution_then_scorer_repair_then_selector_codec_"
            "then_entropy_boundary_then_post_container"
        ),
        "fractal_scope_order": list(_LEVEL_ORDER),
        "stack_rows": rows,
        "interaction_tensor": interaction_tensor,
        "interaction_tensor_cell_count": interaction_tensor["cell_count"],
        "pairwise_interaction_tensor": pairwise_interaction_tensor,
        "pairwise_interaction_tensor_cell_count": pairwise_interaction_tensor["cell_count"],
        "n_way_hypergraph_acquisition_enabled": True,
        "hypergraph_interaction_tensor": hypergraph_interaction_tensor,
        "hypergraph_interaction_tensor_cell_count": hypergraph_interaction_tensor["cell_count"],
        "fractal_marginal_surface": fractal_marginal_surface,
        "fractal_marginal_surface_cell_count": fractal_marginal_surface["cell_count"],
        "archive_entropy_substrate_blockers": ordered_unique(
            blocker
            for row in rows
            for blocker in _string_list(row.get("archive_entropy_substrate_blockers"))
        ),
        "archive_entropy_substrate_gap_count": sum(
            len(_string_list(row.get("archive_entropy_substrate_blockers")))
            for row in rows
        ),
        "archive_entropy_substrate_probed_substrates": ordered_unique(
            substrate
            for row in rows
            for substrate in _string_list(row.get("archive_entropy_substrate_probed_substrates"))
        ),
        "archive_entropy_substrate_probe_count": sum(
            len(_string_list(row.get("archive_entropy_substrate_probed_substrates")))
            for row in rows
        ),
        "archive_entropy_substrate_prototype_count": sum(
            len(_string_list(row.get("archive_entropy_substrate_prototype_substrates")))
            for row in rows
        ),
        "archive_entropy_substrate_prototype_substrates": ordered_unique(
            substrate
            for row in rows
            for substrate in _string_list(row.get("archive_entropy_substrate_prototype_substrates"))
        ),
        "archive_entropy_probed_zero_order_savings_bytes": sum(
            _safe_int(row.get("archive_entropy_probed_zero_order_savings_bytes"))
            for row in rows
        ),
        "archive_entropy_anti_pattern_ids": ordered_unique(
            anti_pattern_id
            for row in rows
            for anti_pattern_id in _string_list(row.get("archive_entropy_anti_pattern_ids"))
        ),
        "archive_entropy_anti_pattern_protection_count": sum(
            _safe_int(row.get("archive_entropy_anti_pattern_protection_count"))
            for row in rows
        ),
        "archive_entropy_anti_pattern_acquisition_penalty_sum": sum(
            _safe_float(row.get("archive_entropy_anti_pattern_acquisition_penalty"))
            for row in rows
        ),
        "archive_variant_signal_count": sum(
            _safe_int(row.get("archive_variant_signal_count")) for row in rows
        ),
        "archive_variant_non_selected_signal_count": sum(
            _safe_int(row.get("archive_variant_non_selected_signal_count"))
            for row in rows
        ),
        "archive_variant_probe_count": sum(
            _safe_int(row.get("archive_variant_probe_count")) for row in rows
        ),
        "archive_variant_prototype_count": sum(
            _safe_int(row.get("archive_variant_prototype_count")) for row in rows
        ),
        "archive_variant_runtime_proof_ready_count": sum(
            _safe_int(row.get("archive_variant_runtime_proof_ready_count"))
            for row in rows
        ),
        "archive_variant_blocked_signal_count": sum(
            _safe_int(row.get("archive_variant_blocked_signal_count"))
            for row in rows
        ),
        "archive_variant_signal_kinds": ordered_unique(
            kind
            for row in rows
            for kind in _string_list(row.get("archive_variant_signal_kinds"))
        ),
        "archive_variant_signal_blockers": ordered_unique(
            blocker
            for row in rows
            for blocker in _string_list(row.get("archive_variant_signal_blockers"))
        ),
        "archive_variant_signal_acquisition_penalty_sum": sum(
            _safe_float(row.get("archive_variant_signal_acquisition_penalty"))
            for row in rows
        ),
        "archive_bound_candidate_contract_count": sum(
            _safe_int(row.get("archive_bound_candidate_contract_count"))
            for row in rows
        ),
        "archive_bound_ready_contract_count": sum(
            _safe_int(row.get("archive_bound_ready_contract_count"))
            for row in rows
        ),
        "archive_bound_contract_substrate_tags": ordered_unique(
            tag
            for row in rows
            for tag in _string_list(row.get("archive_bound_contract_substrate_tags"))
        ),
        "archive_bound_contract_acquisition_penalty_sum": sum(
            _safe_float(row.get("archive_bound_contract_surface_acquisition_penalty"))
            for row in rows
        ),
        "archive_bound_contract_selected_acquisition_penalty_sum": sum(
            _safe_float(row.get("archive_bound_contract_acquisition_penalty"))
            for row in rows
        ),
        "archive_bound_contract_budget_routing_penalty_sum": sum(
            _safe_float(row.get("archive_bound_contract_budget_routing_penalty"))
            for row in rows
        ),
        "archive_bound_candidate_contracts": [
            dict(contract)
            for row in rows
            for contract in [
                _mapping(row.get("archive_bound_candidate_contract"))
            ]
            if contract
        ],
        "archive_variant_materializer_backlog_task_count": sum(
            _safe_int(row.get("archive_variant_materializer_backlog_task_count"))
            for row in rows
        ),
        "archive_variant_materializer_byte_closed_task_count": sum(
            _safe_int(row.get("archive_variant_materializer_byte_closed_task_count"))
            for row in rows
        ),
        "archive_variant_materializer_runtime_adapter_ready_task_count": sum(
            _safe_int(row.get("archive_variant_materializer_runtime_adapter_ready_task_count"))
            for row in rows
        ),
        "archive_variant_materializer_backlog_tasks": [
            task
            for row in rows
            for task in _mapping(row.get("archive_variant_materializer_backlog")).get("task_rows") or []
            if isinstance(task, Mapping)
        ],
        "measured_mlx_posterior_budget_routing_updates": (
            measured_mlx_budget_updates
        ),
        "measured_mlx_posterior_budget_routing_update_count": len(
            measured_mlx_budget_updates
        ),
        "stack_acquisition_frontier": stack_acquisition_frontier,
        "stack_acquisition_frontier_count": len(stack_acquisition_frontier),
        "hypergraph_stack_acquisition_path_count": len(hypergraph_acquisition_paths),
        "pairwise_stack_acquisition_path_count": len(pairwise_acquisition_paths),
        "stack_acquisition_path_count": len(acquisition_paths),
        "stack_acquisition_paths": acquisition_paths,
        "primary_stack_acquisition_path": primary_acquisition_path,
        "bounded_autonomous_terminal_policy": {
            "schema": "repair_family_stack_bounded_autonomous_terminal_policy.v1",
            "terminal_outcome_class": None
            if primary_acquisition_path is None
            else primary_acquisition_path.get("terminal_outcome_class"),
            "stop_conditions": [
                "strictly_better_archive_bound_candidate_exact_axis_blocked",
                "precise_exact_axis_blocker",
                "family_demoted_by_posterior_evidence",
            ],
            "local_mlx_rows_are_advisory_only": True,
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "exact_eval_handoff_candidate_count": len(exact_handoff_candidates),
        "archive_bound_exact_handoff_candidate_count": archive_bound_handoff_count,
        "exact_eval_handoff_candidates": exact_handoff_candidates,
        "planned_family_order": ordered_unique(row["family_id"] for row in rows),
        "budget_routing_decision": budget_routing_decision,
        "candidate_improvement_observed": any(
            float(row.get("local_mlx_expected_improvement_score_units") or 0.0) > 0.0 for row in rows
        ),
        "exact_eval_handoff_gate": {
            "schema": "repair_family_stack_search_exact_handoff_gate.v1",
            "eligible_for_exact_eval_handoff": False,
            "exact_eval_handoff_candidate_count": len(exact_handoff_candidates),
            "archive_bound_exact_handoff_candidate_count": (archive_bound_handoff_count),
            "archive_bound_custody_complete": archive_bound_handoff_count > 0,
            "target_modes": ["contest_exact_eval"],
            "exact_axis_required": ["contest-CPU", "contest-CUDA"],
            "blockers": exact_handoff_gate_blockers,
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


def _chain_exact_handoff_candidate_row(
    chain_report: Mapping[str, Any],
) -> dict[str, Any]:
    stages = [
        stage
        for stage in chain_report.get("stages") or []
        if isinstance(stage, Mapping)
    ]
    final_stage = stages[-1] if stages else {}
    candidate_archive = dict(_mapping(chain_report.get("candidate_archive")))
    runtime_proof_path = str(
        final_stage.get("stage_runtime_consumption_proof_path") or ""
    ).strip()
    runtime_proof_ready = (
        chain_report.get("runtime_consumption_proof_ready") is True
        and final_stage.get("stage_receiver_proof_ready") is True
    )
    runtime_proof = {
        "schema": "repair_family_exact_handoff_runtime_proof_custody.v1",
        "path": runtime_proof_path or None,
        "present": bool(runtime_proof_path),
        "sha256": final_stage.get("stage_runtime_consumption_proof_sha256"),
        "bytes": final_stage.get("stage_runtime_consumption_proof_bytes"),
        "archive_bound_runtime_consumption_proof_ready": runtime_proof_ready,
        "custody_complete": bool(
            runtime_proof_path
            and final_stage.get("stage_runtime_consumption_proof_sha256")
            and runtime_proof_ready
        ),
        "blockers": ordered_unique(
            [
                *([] if runtime_proof_path else ["runtime_consumption_proof_path_missing"]),
                *(
                    []
                    if final_stage.get("stage_runtime_consumption_proof_sha256")
                    else ["runtime_consumption_proof_sha256_missing"]
                ),
                *(
                    []
                    if runtime_proof_ready
                    else ["archive_bound_runtime_consumption_proof_missing"]
                ),
            ]
        ),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    archive_complete = bool(
        chain_report.get("archive_bound_candidate_emitted") is True
        and chain_report.get("candidate_archive_materialized") is True
        and candidate_archive.get("path")
        and candidate_archive.get("sha256")
        and isinstance(candidate_archive.get("bytes"), int)
    )
    candidate_archive.update(
        {
            "expected_sha256": candidate_archive.get("sha256"),
            "expected_bytes": candidate_archive.get("bytes"),
            "present": archive_complete,
            "sha256_matches": archive_complete,
            "bytes_match": archive_complete,
            "custody_complete": archive_complete,
            "blockers": []
            if archive_complete
            else ["chain_candidate_archive_custody_incomplete"],
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }
    )
    archive_bound_complete = archive_complete and runtime_proof["custody_complete"] is True
    chain_id = str(chain_report.get("chain_id") or "entropy_stage_chain")
    blockers = ordered_unique(
        [
            *_string_list(chain_report.get("blockers")),
            *_string_list(candidate_archive.get("blockers")),
            *_string_list(runtime_proof.get("blockers")),
            *([] if archive_bound_complete else ["archive_runtime_custody_incomplete"]),
            "contest_cpu_or_cuda_exact_axis_payload_required",
            "lane_dispatch_claim_required_before_exact_eval",
        ]
    )
    row = {
        "schema": REPAIR_FAMILY_EXACT_HANDOFF_CANDIDATE_ROW_SCHEMA,
        "source_execution_report": None,
        "source_entropy_stage_chain_report_schema": chain_report.get("schema"),
        "source_stack_order": "composed_entropy_stage_chain",
        "family_id": "entropy_stage_chain",
        "typed_response_id": chain_id,
        "candidate_chain_id": chain_id,
        "candidate_chain_ids": [
            str(stage.get("typed_response_id") or "")
            for stage in stages
            if str(stage.get("typed_response_id") or "").strip()
        ],
        "chain_stage_identities": [
            {
                "schema": "repair_entropy_stage_chain_stage_identity.v1",
                "chain_id": chain_id,
                "stage_index": stage.get("stage_index"),
                "family_id": stage.get("family_id"),
                "typed_response_id": stage.get("typed_response_id"),
                "entropy_position_label": stage.get("entropy_position_label"),
                "entropy_stage_order": _chain_stage_order(stage),
                "fractal_scope_levels": _chain_stage_scope_levels(stage),
                "stage_input_archive_sha256": _mapping(
                    stage.get("stage_input_archive")
                ).get("sha256"),
                "stage_output_archive_sha256": _mapping(
                    stage.get("stage_output_archive")
                ).get("sha256"),
                "stage_runtime_consumption_proof_sha256": stage.get(
                    "stage_runtime_consumption_proof_sha256"
                ),
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            }
            for stage in stages
        ],
        "entropy_position_label": "composed_entropy_stage_chain",
        "entropy_stage_order": _chain_stage_order(final_stage) if final_stage else 999,
        "candidate_archive": candidate_archive,
        "archive_bound_candidate_contracts": [
            dict(contract)
            for contract in chain_report.get("archive_bound_candidate_contracts") or []
            if isinstance(contract, Mapping)
        ],
        "archive_bound_candidate_contract_surfaces": [
            dict(surface)
            for surface in chain_report.get("archive_bound_candidate_contract_surfaces") or []
            if isinstance(surface, Mapping)
        ],
        "runtime_consumption_proof": runtime_proof,
        "archive_bound_custody_complete": archive_bound_complete,
        "archive_bound_exact_handoff_candidate": archive_bound_complete,
        "target_modes": ["contest_exact_eval"],
        "exact_axis_required": ["contest-CPU", "contest-CUDA"],
        "component_response_axis": "[macOS-MLX research-signal]",
        "local_mlx_rows_are_advisory_only": True,
        "eligible_for_exact_eval_handoff": False,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "archive_bound_entropy_stage_chain_exact_handoff_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        row,
        context=f"repair_family_exact_handoff_chain_candidate_row:{chain_id}",
    )
    return row


def build_repair_family_exact_handoff_plan(
    *,
    stack_plan: Mapping[str, Any],
    stack_plan_path: str | Path | None = None,
    chain_execution_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize archive-bound repair candidates for exact-axis handoff.

    This is an adapter artifact, not an exact-ready queue. It preserves the
    byte-closed candidate/proof custody discovered by stack search while
    leaving dispatch authority to the existing exact-ready/materializer gates.
    """

    if stack_plan.get("schema") != REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA:
        raise RepairFamilyStackSearchError("repair family exact handoff plan requires repair family stack plan")
    require_no_truthy_authority_fields(
        stack_plan,
        context="repair_family_exact_handoff_source_stack_plan",
    )
    if chain_execution_bundle:
        if (
            chain_execution_bundle.get("schema")
            != "repair_entropy_stage_chain_execution_bundle.v1"
        ):
            raise RepairFamilyStackSearchError(
                "chain_execution_bundle must be repair_entropy_stage_chain_execution_bundle.v1"
            )
        require_no_truthy_authority_fields(
            chain_execution_bundle,
            context="repair_family_exact_handoff_chain_execution_bundle",
        )
    rows = [dict(row) for row in stack_plan.get("exact_eval_handoff_candidates") or [] if isinstance(row, Mapping)]
    chain_rows = [
        _chain_exact_handoff_candidate_row(chain_report)
        for chain_report in _mapping(chain_execution_bundle).get("chain_reports") or []
        if isinstance(chain_report, Mapping)
    ]
    rows.extend(chain_rows)
    archive_bound_rows = [row for row in rows if row.get("archive_bound_custody_complete") is True]
    gate = _mapping(stack_plan.get("exact_eval_handoff_gate"))
    blockers = ordered_unique(
        [
            *_string_list(gate.get("blockers")),
            *([] if archive_bound_rows else ["archive_bound_exact_handoff_candidate_missing"]),
            "materializer_exact_eval_dispatch_plan_or_exact_ready_queue_required",
            "contest_cpu_or_cuda_exact_axis_payload_required",
            "lane_dispatch_claim_required_before_exact_eval",
        ]
    )
    plan = {
        "schema": REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA,
        "source_stack_plan_path": None if stack_plan_path is None else str(stack_plan_path),
        "source_stack_plan_schema": stack_plan.get("schema"),
        "source_chain_execution_bundle_schema": (
            _mapping(chain_execution_bundle).get("schema")
        ),
        "execution_report_count": stack_plan.get("execution_report_count", 0),
        "entropy_stage_chain_candidate_count": len(chain_rows),
        "entropy_stage_chain_archive_bound_candidate_count": sum(
            1 for row in chain_rows if row.get("archive_bound_custody_complete") is True
        ),
        "candidate_count": len(rows),
        "archive_bound_candidate_count": len(archive_bound_rows),
        "archive_bound_custody_complete": bool(archive_bound_rows),
        "rows": rows,
        "archive_bound_rows": archive_bound_rows,
        "handoff_contract": {
            "schema": "repair_family_exact_handoff_contract.v1",
            "source": "repair_family_stack_search",
            "adapter_artifact_only": True,
            "next_authoritative_gate": ("materializer_exact_eval_dispatch_plan_or_exact_ready_queue"),
            "mlx_local_rows_are_advisory_only": True,
            "receiver_must_remain_decode_only": True,
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "blockers": blockers,
        "eligible_for_exact_eval_handoff": False,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "operator_visible_archive_bound_repair_exact_handoff_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        plan,
        context="repair_family_exact_handoff_plan",
    )
    return plan


def _learning_selection_blocker(row: Mapping[str, Any]) -> str:
    blockers = " ".join(_string_list(row.get("blockers")))
    if row.get("byte_credit_feasible") is False or "byte_credit" in blockers:
        return "receiver_credit_exhausted"
    if _safe_int(row.get("entropy_stage_order"), default=999) >= 999:
        return "entropy_stage_contract_miss"
    if row.get("automatic_negative_result_demoted") is True:
        return "negative_result_demoted"
    if row.get("archive_bound_exact_handoff_candidate") is True:
        return "exact_axis_handoff_missing"
    if _string_list(row.get("archive_entropy_substrate_blockers")):
        return "archive_entropy_substrate_materializer_gap"
    if row.get("byte_closed_candidate_emitted") is not True:
        return "byte_closed_candidate_missing"
    if row.get("archive_bound_runtime_consumption_proof_ready") is not True:
        return "runtime_consumption_proof_missing"
    if "stackability" in blockers or "interaction" in blockers:
        return "stackability_interaction_remeasure"
    return "exact_axis_handoff_missing"


def _learning_policy(row: Mapping[str, Any]) -> str:
    blocker = _learning_selection_blocker(row)
    if blocker == "receiver_credit_exhausted":
        return "increase_receiver_closed_rate_credit_or_rebudget_earlier_entropy_stage"
    if blocker == "entropy_stage_contract_miss":
        return "rebuild_entropy_stage_chain_contract_before_budget_spend"
    if blocker == "negative_result_demoted":
        return "decrease_family_priority_until_new_component_response_signal"
    if blocker == "archive_entropy_substrate_materializer_gap":
        return "materialize_missing_archive_entropy_substrate_variant"
    if blocker == "byte_closed_candidate_missing":
        return "prioritize_byte_closed_family_materializer_implementation"
    if blocker == "runtime_consumption_proof_missing":
        return "prioritize_archive_bound_runtime_consumption_proof"
    if blocker == "stackability_interaction_remeasure":
        return "prioritize_stackability_remeasurement_before_additional_budget"
    return "hold_until_byte_closed_exact_auth_handoff_available"


def _learning_signal_for_stack_row(row: Mapping[str, Any]) -> dict[str, Any]:
    family = str(row.get("family_id") or "unclassified_repair_family")
    typed_response_id = str(
        row.get("typed_response_id")
        or row.get("candidate_chain_id")
        or f"{family}:stack_order_{row.get('planned_stack_order') or 0}"
    )
    blocker_class = _learning_selection_blocker(row)
    policy = _learning_policy(row)
    improvement = _safe_float(row.get("local_mlx_expected_improvement_score_units"))
    delta_bytes = max(1, _safe_int(row.get("delta_payload_bytes"), default=1))
    levels = _string_list(row.get("fractal_scope_levels"))
    blockers = ordered_unique(
        [
            *_string_list(row.get("blockers")),
            "repair_family_stack_learning_signal_is_not_score_authority",
            "exact_axis_component_response_required_before_budget_spend",
        ]
    )
    identity = {
        "schema": "repair_family_stack_learning_identity.v1",
        "family_id": family,
        "typed_response_id": typed_response_id,
        "candidate_chain_id": row.get("candidate_chain_id"),
        "planned_stack_order": row.get("planned_stack_order"),
        "entropy_stage_order": row.get("entropy_stage_order"),
        "fractal_scope_levels": levels,
        "selection_blocker_class": blocker_class,
        "blockers": blockers,
    }
    feature_vector = {
        "materialization_signal_kind": "repair_family_stack_search_row",
        "candidate_archive_materialized": (row.get("candidate_archive_materialized") is True),
        "runtime_consumption_proof_present": (row.get("archive_bound_runtime_consumption_proof_ready") is True),
        "component_response_replayed": True,
        "expected_local_improvement_score_units": improvement,
        "improvement_per_allocated_byte": improvement / delta_bytes,
        "interaction_aware_stack_score": _safe_float(row.get("interaction_aware_stack_score")),
        "posterior_acquisition_multiplier": _safe_float(row.get("posterior_acquisition_multiplier")),
        "blocker_count": len(blockers),
        "missing_artifact_count": sum(1 for blocker in blockers if "missing" in blocker or "incomplete" in blocker),
        "entropy_position_label": row.get("entropy_position_label"),
        "entropy_stage_order": row.get("entropy_stage_order"),
        "entropy_pipeline_stage_index": _safe_int(row.get("entropy_stage_order"), default=999),
        "fractal_active_levels": levels,
        "fractal_ordered_levels": [level for level in _LEVEL_ORDER if level in set(levels)],
        "interaction_order": len(levels),
        "archive_entropy_substrate_materialized_substrates": _string_list(
            row.get("archive_entropy_substrate_materialized_substrates")
        ),
        "archive_entropy_substrate_probed_substrates": _string_list(
            row.get("archive_entropy_substrate_probed_substrates")
        ),
        "archive_entropy_substrate_prototype_substrates": _string_list(
            row.get("archive_entropy_substrate_prototype_substrates")
        ),
        "archive_entropy_probed_zero_order_savings_bytes": _safe_int(
            row.get("archive_entropy_probed_zero_order_savings_bytes")
        ),
        "archive_entropy_anti_pattern_ids": _string_list(
            row.get("archive_entropy_anti_pattern_ids")
        ),
        "archive_entropy_anti_pattern_acquisition_penalty": _safe_float(
            row.get("archive_entropy_anti_pattern_acquisition_penalty")
        ),
        "archive_entropy_substrate_blockers": _string_list(
            row.get("archive_entropy_substrate_blockers")
        ),
        "archive_variant_signal_count": _safe_int(row.get("archive_variant_signal_count")),
        "archive_variant_non_selected_signal_count": _safe_int(
            row.get("archive_variant_non_selected_signal_count")
        ),
        "archive_variant_probe_count": _safe_int(row.get("archive_variant_probe_count")),
        "archive_variant_prototype_count": _safe_int(
            row.get("archive_variant_prototype_count")
        ),
        "archive_variant_runtime_proof_ready_count": _safe_int(
            row.get("archive_variant_runtime_proof_ready_count")
        ),
        "archive_variant_blocked_signal_count": _safe_int(
            row.get("archive_variant_blocked_signal_count")
        ),
        "archive_variant_signal_kinds": _string_list(
            row.get("archive_variant_signal_kinds")
        ),
        "archive_variant_signal_blockers": _string_list(
            row.get("archive_variant_signal_blockers")
        ),
        "archive_variant_signal_acquisition_penalty": _safe_float(
            row.get("archive_variant_signal_acquisition_penalty")
        ),
        "archive_bound_contract_ready": (
            row.get("archive_bound_contract_ready") is True
        ),
        "archive_bound_candidate_contract_count": _safe_int(
            row.get("archive_bound_candidate_contract_count")
        ),
        "archive_bound_ready_contract_count": _safe_int(
            row.get("archive_bound_ready_contract_count")
        ),
        "archive_bound_contract_substrate_tags": _string_list(
            row.get("archive_bound_contract_substrate_tags")
        ),
        "archive_bound_contract_acquisition_penalty": _safe_float(
            row.get("archive_bound_contract_acquisition_penalty")
        ),
        "archive_bound_contract_surface_acquisition_penalty": _safe_float(
            row.get("archive_bound_contract_surface_acquisition_penalty")
        ),
        "archive_bound_contract_budget_routing_penalty": _safe_float(
            row.get("archive_bound_contract_budget_routing_penalty")
        ),
        "archive_variant_materializer_backlog_task_count": _safe_int(
            row.get("archive_variant_materializer_backlog_task_count")
        ),
        "archive_variant_materializer_byte_closed_task_count": _safe_int(
            row.get("archive_variant_materializer_byte_closed_task_count")
        ),
        "archive_variant_materializer_runtime_adapter_ready_task_count": _safe_int(
            row.get("archive_variant_materializer_runtime_adapter_ready_task_count")
        ),
        "selection_blocker_class": blocker_class,
        "receiver_credit_exhausted": blocker_class == "receiver_credit_exhausted",
        "stackability_remeasure_required": (blocker_class == "stackability_interaction_remeasure"),
        "entropy_stage_contract_miss": blocker_class == "entropy_stage_contract_miss",
    }
    signal = {
        "schema": REPAIR_FAMILY_STACK_LEARNING_SIGNAL_SCHEMA,
        "learning_signal_kind": "repair_family_stack_search_feedback",
        "typed_response_id": typed_response_id,
        "candidate_id": row.get("candidate_chain_id") or typed_response_id,
        "family_id": family,
        "component_response_axis": "[macOS-MLX research-signal]",
        "evidence_grade": "repair_family_stack_search_local_planning_signal_only",
        "source_artifacts": [dict(_mapping(row.get("source_execution_report")))]
        if row.get("source_execution_report")
        else [],
        "replay_identity": {
            "schema": "repair_family_stack_learning_replay_identity.v1",
            "replay_identity_kind": "repair_family_stack_search_feedback",
            "hash_manifest_sha256": _stable_sha256(identity),
            "source_records_sha256": _stable_sha256(
                {
                    "schema": "repair_family_stack_learning_sources.v1",
                    "source_execution_report": row.get("source_execution_report"),
                }
            ),
            "replay_argv_sha256": None,
            "execution_context_sha256": None,
            "environment_sha256": None,
        },
        "local_planning_update": {
            "schema": REPAIR_FAMILY_STACK_LOCAL_PLANNING_UPDATE_SCHEMA,
            "posterior_surface": "repair_family_stack_search_posterior_feedback",
            "local_planning_update_ready": True,
            "recommended_acquisition_policy": policy,
            "recommended_stackability_followup": ("continue_bounded_autonomous_repair_floor_loop"),
            "planner_feature_vector": feature_vector,
            "posterior_update_blockers": [
                "repair_family_stack_learning_signal_is_not_score_authority",
                "exact_axis_component_response_required_before_budget_spend",
            ],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "blockers": blockers,
        "missing_artifacts": [blocker for blocker in blockers if "missing" in blocker or "incomplete" in blocker],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_family_stack_acquisition_update_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        signal,
        context=f"repair_family_stack_learning_signal:{typed_response_id}",
    )
    return signal


def _missing_execution_report_learning_signal() -> dict[str, Any]:
    row = {
        "family_id": "unclassified_repair_family",
        "typed_response_id": "repair_family_floor_loop_execution_reports_missing",
        "candidate_chain_id": "repair_family_floor_loop_execution_reports_missing",
        "planned_stack_order": 0,
        "entropy_stage_order": 999,
        "fractal_scope_levels": [],
        "delta_payload_bytes": 1,
        "local_mlx_expected_improvement_score_units": 0.0,
        "interaction_aware_stack_score": 0.0,
        "byte_credit_feasible": True,
        "byte_closed_candidate_emitted": False,
        "candidate_archive_materialized": False,
        "archive_bound_runtime_consumption_proof_ready": False,
        "blockers": ["repair_family_byte_transform_execution_reports_missing"],
        **FALSE_AUTHORITY,
    }
    return _learning_signal_for_stack_row(row)


def _chain_stage_selection_blocker(
    chain_report: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> str:
    blockers = " ".join(
        [
            *_string_list(stage.get("stage_blockers")),
            *_string_list(stage.get("stage_report_blockers")),
            *_string_list(chain_report.get("blockers")),
        ]
    )
    if stage.get("stage_materialized") is not True:
        return "entropy_stage_chain_stage_failed"
    if stage.get("stage_receiver_proof_ready") is not True:
        return "runtime_consumption_proof_missing"
    if chain_report.get("archive_bound_candidate_emitted") is True:
        return "exact_axis_handoff_missing"
    if "source_archive" in blockers or "manifest" in blockers:
        return "chain_custody_missing"
    if "entropy_stage" in blockers:
        return "entropy_stage_contract_miss"
    return "exact_axis_handoff_missing"


def _chain_stage_policy(
    chain_report: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> str:
    blocker = _chain_stage_selection_blocker(chain_report, stage)
    if blocker == "entropy_stage_chain_stage_failed":
        return "decrease_family_priority_until_entropy_stage_chain_stage_repaired"
    if blocker == "runtime_consumption_proof_missing":
        return "prioritize_archive_bound_runtime_consumption_proof"
    if blocker == "chain_custody_missing":
        return "prioritize_entropy_stage_chain_custody_repair_before_budget"
    if blocker == "entropy_stage_contract_miss":
        return "rebuild_entropy_stage_chain_contract_before_budget_spend"
    return "prioritize_entropy_stage_chain_exact_axis_bridge_after_custody"


def _chain_stage_scope_levels(stage: Mapping[str, Any]) -> list[str]:
    scope = _mapping(stage.get("fractal_optimization_scope"))
    return _string_list(scope.get("active_levels")) or _string_list(scope.get("declared_levels"))


def _chain_stage_order(stage: Mapping[str, Any]) -> int:
    active = _mapping(stage.get("active_entropy_stage"))
    return _safe_int(active.get("order") or stage.get("entropy_stage_order"), default=999)


def _chain_stage_delta_bytes(stage: Mapping[str, Any]) -> int:
    byte_delta = _mapping(stage.get("byte_transform_delta"))
    return max(
        1,
        _safe_int(stage.get("allocated_repair_bytes") or byte_delta.get("bytes"), default=1),
    )


def _chain_stage_local_improvement(stage: Mapping[str, Any]) -> float:
    delta = _mapping(stage.get("mlx_local_probe_delta"))
    combined_delta = _safe_float(delta.get("combined_delta_score_units"))
    return max(0.0, -combined_delta)


def _learning_signal_for_entropy_stage_chain_stage(
    *,
    chain_report: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> dict[str, Any]:
    chain_id = str(chain_report.get("chain_id") or "entropy_stage_chain_unknown")
    family = str(stage.get("family_id") or "unclassified_repair_family")
    source_typed_response_id = str(stage.get("typed_response_id") or "").strip()
    stage_index = _safe_int(stage.get("stage_index"), default=0)
    typed_response_id = (
        f"{chain_id}:stage_{stage_index:03d}:"
        f"{source_typed_response_id or family}"
    )
    blocker_class = _chain_stage_selection_blocker(chain_report, stage)
    policy = _chain_stage_policy(chain_report, stage)
    levels = _chain_stage_scope_levels(stage)
    entropy_stage_order = _chain_stage_order(stage)
    improvement = _chain_stage_local_improvement(stage)
    delta_bytes = _chain_stage_delta_bytes(stage)
    blockers = ordered_unique(
        [
            *_string_list(stage.get("stage_blockers")),
            *_string_list(stage.get("stage_report_blockers")),
            *_string_list(chain_report.get("blockers")),
            "repair_entropy_stage_chain_learning_signal_is_not_score_authority",
            "exact_axis_component_response_required_before_budget_spend",
        ]
    )
    identity = {
        "schema": "repair_entropy_stage_chain_learning_identity.v1",
        "chain_id": chain_id,
        "stage_index": stage_index,
        "family_id": family,
        "typed_response_id": typed_response_id,
        "source_typed_response_id": source_typed_response_id,
        "entropy_stage_order": entropy_stage_order,
        "fractal_scope_levels": levels,
        "selection_blocker_class": blocker_class,
        "source_archive_sha256": _mapping(chain_report.get("source_archive")).get("sha256"),
        "candidate_archive_sha256": _mapping(chain_report.get("candidate_archive")).get("sha256"),
        "stage_input_sha256": _mapping(stage.get("stage_input_archive")).get("sha256"),
        "stage_output_sha256": _mapping(stage.get("stage_output_archive")).get("sha256"),
    }
    feature_vector = {
        "materialization_signal_kind": "repair_entropy_stage_chain_stage",
        "chain_id": chain_id,
        "chain_stage_index": stage_index,
        "candidate_archive_materialized": (
            chain_report.get("candidate_archive_materialized") is True
        ),
        "stage_materialized": stage.get("stage_materialized") is True,
        "runtime_consumption_proof_present": (
            chain_report.get("runtime_consumption_proof_ready") is True
            or stage.get("stage_receiver_proof_ready") is True
        ),
        "component_response_replayed": True,
        "expected_local_improvement_score_units": improvement,
        "improvement_per_allocated_byte": improvement / delta_bytes,
        "cumulative_saved_bytes": _safe_int(chain_report.get("cumulative_saved_bytes")),
        "source_archive_bytes": _safe_int(chain_report.get("source_archive_bytes")),
        "candidate_archive_bytes": _safe_int(chain_report.get("candidate_archive_bytes")),
        "blocker_count": len(blockers),
        "missing_artifact_count": sum(
            1 for blocker in blockers if "missing" in blocker or "incomplete" in blocker
        ),
        "entropy_position_label": stage.get("entropy_position_label"),
        "entropy_stage_order": entropy_stage_order,
        "entropy_pipeline_stage_index": entropy_stage_order,
        "fractal_active_levels": levels,
        "fractal_ordered_levels": [level for level in _LEVEL_ORDER if level in set(levels)],
        "interaction_order": len(levels),
        "selection_blocker_class": blocker_class,
        "receiver_credit_exhausted": False,
        "stackability_remeasure_required": (
            blocker_class == "entropy_stage_chain_stage_failed"
        ),
        "entropy_stage_contract_miss": blocker_class == "entropy_stage_contract_miss",
    }
    signal = {
        "schema": REPAIR_FAMILY_STACK_LEARNING_SIGNAL_SCHEMA,
        "learning_signal_kind": "repair_entropy_stage_chain_execution_feedback",
        "typed_response_id": typed_response_id,
        "candidate_id": chain_id,
        "family_id": family,
        "component_response_axis": "[macOS-MLX research-signal]",
        "evidence_grade": "repair_entropy_stage_chain_local_execution_signal_only",
        "source_artifacts": [
            {
                "label": "repair_entropy_stage_chain_stage_execution_report",
                "path": stage.get("stage_execution_report_path"),
                "source_execution_report_path": stage.get(
                    "source_execution_report_path"
                ),
                "stage_replay_bundle_path": stage.get("stage_replay_bundle_path"),
            }
        ],
        "replay_identity": {
            "schema": "repair_entropy_stage_chain_learning_replay_identity.v1",
            "replay_identity_kind": "repair_entropy_stage_chain_execution_feedback",
            "hash_manifest_sha256": _stable_sha256(identity),
            "source_records_sha256": _stable_sha256(
                {
                    "schema": "repair_entropy_stage_chain_learning_sources.v1",
                    "chain_id": chain_id,
                    "stage": dict(stage),
                }
            ),
            "replay_argv_sha256": None,
            "execution_context_sha256": None,
            "environment_sha256": None,
        },
        "local_planning_update": {
            "schema": REPAIR_FAMILY_STACK_LOCAL_PLANNING_UPDATE_SCHEMA,
            "posterior_surface": "repair_entropy_stage_chain_execution_posterior_feedback",
            "local_planning_update_ready": True,
            "recommended_acquisition_policy": policy,
            "recommended_stackability_followup": (
                "continue_bounded_autonomous_repair_floor_loop"
            ),
            "planner_feature_vector": feature_vector,
            "posterior_update_blockers": [
                "repair_entropy_stage_chain_learning_signal_is_not_score_authority",
                "exact_axis_component_response_required_before_budget_spend",
            ],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "blockers": blockers,
        "missing_artifacts": [
            blocker for blocker in blockers if "missing" in blocker or "incomplete" in blocker
        ],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_entropy_stage_chain_acquisition_update_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        signal,
        context=f"repair_entropy_stage_chain_learning_signal:{typed_response_id}",
    )
    return signal


def _learning_signals_for_entropy_stage_chain_bundle(
    chain_execution_bundle: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not chain_execution_bundle:
        return []
    if (
        chain_execution_bundle.get("schema")
        != "repair_entropy_stage_chain_execution_bundle.v1"
    ):
        raise RepairFamilyStackSearchError(
            "chain_execution_bundle must be repair_entropy_stage_chain_execution_bundle.v1"
        )
    require_no_truthy_authority_fields(
        chain_execution_bundle,
        context="repair_family_stack_learning_signal_chain_execution_bundle",
    )
    return [
        _learning_signal_for_entropy_stage_chain_stage(
            chain_report=chain_report,
            stage=stage,
        )
        for chain_report in chain_execution_bundle.get("chain_reports") or []
        if isinstance(chain_report, Mapping)
        for stage in chain_report.get("stages") or []
        if isinstance(stage, Mapping)
    ]


def _failure_update_scope_levels(update: Mapping[str, Any]) -> list[str]:
    direct = _string_list(update.get("fractal_scope_levels"))
    if direct:
        return direct
    levels: list[str] = []
    for stage in update.get("chain_stage_identities") or []:
        if isinstance(stage, Mapping):
            levels.extend(_string_list(stage.get("fractal_scope_levels")))
    return ordered_unique(levels)


def _learning_signal_for_failure_rebudgeting_update(
    update: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_id = str(update.get("candidate_id") or "unknown_exact_candidate")
    family = str(update.get("family_id") or "unclassified_repair_family")
    typed_response_id = f"exact_failure:{candidate_id}"
    blocker_class = str(update.get("selected_blocker_class") or "exact_axis_handoff_required")
    policy = str(
        update.get("recommended_acquisition_policy")
        or "hold_until_byte_closed_exact_auth_handoff_available"
    )
    levels = _failure_update_scope_levels(update)
    entropy_stage_order = _safe_int(update.get("entropy_stage_order"), default=999)
    demote = update.get("demote_responsible_family_stage_scope") is True
    blockers = ordered_unique(
        [
            *_string_list(update.get("blockers")),
            *(
                ["negative_result_demoted_from_exact_preclaim_or_receiver_failure"]
                if demote
                else []
            ),
            "repair_exact_failure_rebudgeting_signal_is_not_score_authority",
            "exact_axis_component_response_required_before_budget_spend",
        ]
    )
    identity = {
        "schema": "repair_exact_failure_learning_identity.v1",
        "candidate_id": candidate_id,
        "family_id": family,
        "typed_response_id": typed_response_id,
        "source_typed_response_id": update.get("typed_response_id"),
        "candidate_chain_id": update.get("candidate_chain_id"),
        "candidate_chain_ids": _string_list(update.get("candidate_chain_ids")),
        "entropy_position_label": update.get("entropy_position_label"),
        "entropy_stage_order": entropy_stage_order,
        "fractal_scope_levels": levels,
        "selected_blocker_class": blocker_class,
        "responsible_failure_surface": update.get("responsible_failure_surface"),
        "source_archive_sha256": update.get("source_archive_sha256"),
        "candidate_archive_sha256": update.get("candidate_archive_sha256"),
        "runtime_consumption_proof_sha256": update.get(
            "runtime_consumption_proof_sha256"
        ),
        "runtime_content_tree_sha256": update.get("runtime_content_tree_sha256"),
        "chain_stage_identities": [
            dict(item)
            for item in update.get("chain_stage_identities") or []
            if isinstance(item, Mapping)
        ],
    }
    feature_vector = {
        "materialization_signal_kind": "repair_exact_failure_rebudgeting_update",
        "candidate_archive_materialized": bool(update.get("candidate_archive_sha256")),
        "runtime_consumption_proof_present": bool(
            update.get("runtime_consumption_proof_sha256")
        ),
        "component_response_replayed": False,
        "expected_local_improvement_score_units": 0.0,
        "improvement_per_allocated_byte": 0.0,
        "blocker_count": len(blockers),
        "missing_artifact_count": sum(
            1 for blocker in blockers if "missing" in blocker or "incomplete" in blocker
        ),
        "entropy_position_label": update.get("entropy_position_label"),
        "entropy_stage_order": entropy_stage_order,
        "entropy_pipeline_stage_index": entropy_stage_order,
        "fractal_active_levels": levels,
        "fractal_ordered_levels": [level for level in _LEVEL_ORDER if level in set(levels)],
        "interaction_order": len(levels),
        "selection_blocker_class": blocker_class,
        "responsible_failure_surface": update.get("responsible_failure_surface"),
        "demote_responsible_family_stage_scope": demote,
        "receiver_credit_exhausted": (
            update.get("rebudget_receiver_closed_credit") is True
        ),
        "stackability_remeasure_required": demote,
        "entropy_stage_contract_miss": (
            blocker_class == "exact_dispatch_preclaim_failed"
            and family == "entropy_stage_chain"
        ),
    }
    signal = {
        "schema": REPAIR_FAMILY_STACK_LEARNING_SIGNAL_SCHEMA,
        "learning_signal_kind": "repair_exact_failure_rebudgeting_feedback",
        "typed_response_id": typed_response_id,
        "candidate_id": candidate_id,
        "family_id": family,
        "component_response_axis": "[contest-exact-axis-blocked]",
        "evidence_grade": "exact_preclaim_or_bridge_failure_routing_signal_only",
        "source_artifacts": [],
        "replay_identity": {
            "schema": "repair_exact_failure_learning_replay_identity.v1",
            "replay_identity_kind": "repair_exact_failure_rebudgeting_feedback",
            "hash_manifest_sha256": _stable_sha256(identity),
            "source_records_sha256": _stable_sha256(
                {
                    "schema": "repair_exact_failure_learning_sources.v1",
                    "failure_rebudgeting_update": dict(update),
                }
            ),
            "replay_argv_sha256": None,
            "execution_context_sha256": None,
            "environment_sha256": None,
        },
        "local_planning_update": {
            "schema": REPAIR_FAMILY_STACK_LOCAL_PLANNING_UPDATE_SCHEMA,
            "posterior_surface": "repair_exact_failure_rebudgeting_posterior_feedback",
            "local_planning_update_ready": True,
            "recommended_acquisition_policy": policy,
            "recommended_stackability_followup": (
                "rebudget_or_demote_exact_failure_surface_before_next_dispatch"
            ),
            "planner_feature_vector": feature_vector,
            "posterior_update_blockers": [
                "repair_exact_failure_rebudgeting_signal_is_not_score_authority",
                "exact_axis_component_response_required_before_budget_spend",
            ],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "blockers": blockers,
        "missing_artifacts": [
            blocker for blocker in blockers if "missing" in blocker or "incomplete" in blocker
        ],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_exact_failure_rebudgeting_update_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        signal,
        context=f"repair_exact_failure_learning_signal:{candidate_id}",
    )
    return signal


def _learning_signals_for_failure_rebudgeting_updates(
    failure_rebudgeting_updates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _learning_signal_for_failure_rebudgeting_update(update)
        for update in failure_rebudgeting_updates
        if isinstance(update, Mapping)
    ]


def build_repair_family_stack_learning_signal_report(
    *,
    stack_plan: Mapping[str, Any],
    bridge_report: Mapping[str, Any] | None = None,
    chain_execution_bundle: Mapping[str, Any] | None = None,
    failure_rebudgeting_updates: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build posterior-consumable local learning signals from stack outcomes."""

    if stack_plan.get("schema") != REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA:
        raise RepairFamilyStackSearchError("stack learning signal report requires repair family stack plan")
    require_no_truthy_authority_fields(
        stack_plan,
        context="repair_family_stack_learning_signal_source_plan",
    )
    if bridge_report is not None:
        require_no_truthy_authority_fields(
            bridge_report,
            context="repair_family_stack_learning_signal_bridge_report",
        )
    rows = [row for row in stack_plan.get("stack_rows") or [] if isinstance(row, Mapping)]
    stack_signals = (
        [_learning_signal_for_stack_row(row) for row in rows] if rows else [_missing_execution_report_learning_signal()]
    )
    chain_signals = _learning_signals_for_entropy_stage_chain_bundle(
        chain_execution_bundle
    )
    failure_signals = _learning_signals_for_failure_rebudgeting_updates(
        failure_rebudgeting_updates
    )
    signals = [*stack_signals, *chain_signals, *failure_signals]
    bridge_blockers = list(_string_list(_mapping(bridge_report).get("blockers"))) if bridge_report is not None else []
    report = {
        "schema": REPAIR_FAMILY_STACK_LEARNING_SIGNAL_REPORT_SCHEMA,
        "generated_at_utc": _utc_now(),
        "learning_signal_kind": "repair_family_stack_search_feedback",
        "source_stack_plan_schema": stack_plan.get("schema"),
        "source_bridge_report_schema": _mapping(bridge_report).get("schema"),
        "source_chain_execution_bundle_schema": (
            _mapping(chain_execution_bundle).get("schema")
        ),
        "stack_row_count": len(rows),
        "stack_learning_signal_count": len(stack_signals),
        "entropy_stage_chain_learning_signal_count": len(chain_signals),
        "exact_failure_rebudgeting_learning_signal_count": len(failure_signals),
        "learning_signal_count": len(signals),
        "learning_signal_rows": signals,
        "blockers": ordered_unique(
            [
                *bridge_blockers,
                "repair_family_stack_learning_signal_is_not_score_authority",
                "exact_axis_component_response_required_before_budget_spend",
            ]
        ),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_family_stack_feedback_posterior_append_input",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="repair_family_stack_learning_signal_report",
    )
    return report


__all__ = [
    "REPAIR_FAMILY_EXACT_HANDOFF_CANDIDATE_ROW_SCHEMA",
    "REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA",
    "REPAIR_FAMILY_STACK_LEARNING_SIGNAL_REPORT_SCHEMA",
    "REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA",
    "REPAIR_FAMILY_STACK_SEARCH_ROW_SCHEMA",
    "RepairFamilyStackSearchError",
    "build_repair_family_exact_handoff_plan",
    "build_repair_family_stack_learning_signal_report",
    "plan_repair_family_stack_search",
]
