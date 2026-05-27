# SPDX-License-Identifier: MIT
"""Planner-consumable learning signals from repair stackability replay bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_chain_contract import (
    require_interaction_aware_optimizer_decision,
)
from tac.optimization.repair_campaign_replay_bundle import (
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
)
from tac.optimization.repair_campaign_scorer import (
    REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
)
from tac.repo_io import json_text, sha256_bytes, sha256_file

REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA = "repair_campaign_learning_signal.v1"
REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA = (
    "repair_campaign_local_planning_update.v1"
)
REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA = (
    "repair_campaign_blocked_learning_signal_report.v1"
)
REPAIR_CAMPAIGN_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA = (
    "frontier_final_rate_attack_child_queue_activation_plan.v1"
)


class RepairCampaignLearningSignalError(ValueError):
    """Raised when a repair campaign learning signal cannot be built."""


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


def _safe_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _policy_blocker_class(blockers: Sequence[Any], missing_artifacts: Sequence[Any]) -> str:
    joined = " ".join([*_string_list(blockers), *_string_list(missing_artifacts)])
    if "receiver_closed_rate_credit_exhausted" in joined:
        return "receiver_credit_exhausted"
    if "receiver_closed" in joined or "saved_bytes" in joined:
        return "receiver_credit_missing"
    if "stacking" in joined or "interaction" in joined or "remeasure" in joined:
        return "stackability_interaction_remeasure"
    if "entropy_pipeline" in joined or "entropy_stage" in joined:
        return "entropy_stage_contract_miss"
    if "exact_auth_eval" in joined:
        return "exact_axis_handoff_missing"
    if "local_mlx" in joined:
        return "local_mlx_custody_missing"
    if "component_response" in joined or "targeted_component" in joined:
        return "component_response_missing"
    return "generic_artifact_or_queue_blocker"


def _entropy_pipeline(row: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = _mapping(row.get("entropy_pipeline_position"))
    if direct:
        return direct
    return _mapping(_mapping(row.get("multiscale_action_row")).get("entropy_pipeline_position"))


def _interaction_order(row: Mapping[str, Any]) -> int:
    direct = _safe_int(row.get("interaction_order"))
    if direct:
        return direct
    return _safe_int(_mapping(row.get("multiscale_action_row")).get("interaction_order"))


def _resolve(path: str | Path, *, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, *, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _file_record(label: str, path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root=repo_root)
    if not resolved.is_file():
        raise RepairCampaignLearningSignalError(f"required artifact missing: {label}={path}")
    return {
        "label": label,
        "path": _repo_rel(resolved, repo_root=repo_root),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(json_text(payload).encode("utf-8"))


def _blocked_policy(blockers: Sequence[Any], missing_artifacts: Sequence[Any]) -> str:
    joined = " ".join([*_string_list(blockers), *_string_list(missing_artifacts)])
    if "receiver_closed_rate_credit_exhausted" in joined:
        return "increase_receiver_closed_rate_credit_or_rebudget_earlier_entropy_stage"
    if "stacking" in joined or "interaction" in joined or "remeasure" in joined:
        return "prioritize_stackability_remeasurement_before_additional_budget"
    if "entropy_pipeline" in joined or "entropy_stage" in joined:
        return "rebuild_entropy_stage_chain_contract_before_budget_spend"
    if "targeted_component" in joined or "component_response" in joined:
        return "increase_priority_for_targeted_component_response_harvest"
    if "receiver_closed_rate_credit" in joined:
        return "hold_until_receiver_closed_rate_credit_available"
    if "receiver_closed" in joined or "saved_bytes" in joined:
        return "hold_until_receiver_closed_rate_credit_available"
    if "exact_auth_eval" in joined:
        return "hold_until_byte_closed_exact_auth_handoff_available"
    if "non_improving_local_objective_delta" in joined:
        return "decrease_family_priority_until_new_component_response_signal"
    if "local_mlx" in joined:
        return "materialize_missing_local_mlx_custody_before_stackability"
    return "materialize_missing_repair_campaign_artifacts"


def _materialization_policy(
    *,
    blockers: Sequence[Any],
    archive_materialized: bool,
    runtime_proof_present: bool,
    receiver_consumed: bool,
    component_response_replayed: bool,
) -> str:
    joined = " ".join(_string_list(blockers))
    if component_response_replayed and not archive_materialized:
        return "prioritize_byte_closed_family_materializer_implementation"
    if archive_materialized and not runtime_proof_present:
        return "prioritize_archive_bound_runtime_consumption_proof"
    if runtime_proof_present and not receiver_consumed:
        return "prioritize_receiver_decode_only_proof"
    if not component_response_replayed and "component_response" in joined:
        return "increase_priority_for_targeted_component_response_harvest"
    if (
        archive_materialized
        and runtime_proof_present
        and receiver_consumed
        and component_response_replayed
    ):
        return "hold_until_byte_closed_exact_auth_handoff_available"
    return _blocked_policy(blockers, blockers)


def build_repair_campaign_materialization_learning_signal_report(
    *,
    materialization_execution_report_path: str | Path,
    materialization_execution_report: Mapping[str, Any],
    materialization_gate_path: str | Path,
    materialization_gate: Mapping[str, Any],
    repo_root: str | Path,
    family_materializer_manifest_path: str | Path | None = None,
    family_materializer_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build posterior-consumable signals from repair materialization gates."""

    if (
        materialization_execution_report.get("schema")
        != "frontier_rate_attack_repair_budget_materialization_execution_report.v1"
    ):
        raise RepairCampaignLearningSignalError(
            "materialization execution report has unexpected schema"
        )
    if (
        materialization_gate.get("schema")
        != "repair_campaign_byte_closed_materialization_gate.v1"
    ):
        raise RepairCampaignLearningSignalError(
            "materialization gate has unexpected schema"
        )
    for label, payload in (
        ("materialization_execution_report", materialization_execution_report),
        ("materialization_gate", materialization_gate),
    ):
        require_no_truthy_authority_fields(
            payload,
            context=f"repair_materialization_learning_signal_input:{label}",
        )
    manifest = dict(family_materializer_manifest or {})
    if manifest:
        require_no_truthy_authority_fields(
            manifest,
            context="repair_materialization_learning_signal_family_manifest",
        )
    source_artifacts = [
        _file_record(
            "repair_budget_materialization_execution_report",
            materialization_execution_report_path,
            repo_root=repo_root,
        ),
        _file_record(
            "repair_campaign_byte_closed_materialization_gate",
            materialization_gate_path,
            repo_root=repo_root,
        ),
    ]
    if family_materializer_manifest_path is not None:
        source_artifacts.append(
            _file_record(
                "repair_campaign_family_materializer_manifest",
                family_materializer_manifest_path,
                repo_root=repo_root,
            )
        )
    typed_response_id = str(materialization_gate.get("typed_response_id") or "").strip()
    candidate_id = str(materialization_gate.get("candidate_id") or "").strip()
    family_id = str(
        manifest.get("family_id")
        or manifest.get("target_kind")
        or materialization_gate.get("family_id")
        or "unclassified_repair_family"
    )
    execution_rows = [
        row
        for row in materialization_execution_report.get("execution_rows") or []
        if isinstance(row, Mapping)
    ]
    child_rows = [
        row
        for row in execution_rows
        if row.get("candidate_kind") == "spent_budget_repair_child"
    ]
    target_row = child_rows[0] if child_rows else (execution_rows[0] if execution_rows else {})
    blockers = ordered_unique(
        [
            *_string_list(materialization_gate.get("blockers")),
            *_string_list(materialization_execution_report.get("blockers")),
            *_string_list(target_row.get("blockers")),
            "materialization_learning_signal_is_not_score_authority",
            "exact_axis_component_response_required_before_budget_spend",
        ]
    )
    archive_materialized = (
        materialization_gate.get("candidate_archive_materialized") is True
        or target_row.get("candidate_archive_materialized") is True
    )
    runtime_proof_present = (
        materialization_gate.get("archive_bound_runtime_consumption_proof_ready")
        is True
        or target_row.get("runtime_consumption_proof_present") is True
    )
    receiver_consumed = target_row.get("receiver_consumed") is True
    component_response_replayed = (
        materialization_gate.get("component_response_replayed") is True
        or target_row.get("component_response_replayed") is True
        or manifest.get("component_response_replayed") is True
    )
    policy = _materialization_policy(
        blockers=blockers,
        archive_materialized=archive_materialized,
        runtime_proof_present=runtime_proof_present,
        receiver_consumed=receiver_consumed,
        component_response_replayed=component_response_replayed,
    )
    entropy_stage = _mapping(manifest.get("active_entropy_stage"))
    fractal_scope = _mapping(manifest.get("fractal_optimization_scope"))
    policy_blocker_class = _policy_blocker_class(blockers, blockers)
    identity = {
        "schema": "repair_materialization_learning_identity.v1",
        "typed_response_id": typed_response_id,
        "candidate_id": candidate_id,
        "family_id": family_id,
        "candidate_chain_id": target_row.get("candidate_chain_id"),
        "entropy_position_label": manifest.get("entropy_position_label"),
        "archive_materialized": archive_materialized,
        "runtime_proof_present": runtime_proof_present,
        "receiver_consumed": receiver_consumed,
        "component_response_replayed": component_response_replayed,
        "blockers": blockers,
        "source_artifact_sha256s": [row["sha256"] for row in source_artifacts],
    }
    feature_vector = {
        "materialization_signal_kind": "repair_campaign_byte_closed_materialization_gate",
        "candidate_archive_materialized": archive_materialized,
        "runtime_consumption_proof_present": runtime_proof_present,
        "receiver_consumed": receiver_consumed,
        "component_response_replayed": component_response_replayed,
        "missing_archive_after_component_replay": (
            component_response_replayed and not archive_materialized
        ),
        "blocker_count": len(blockers),
        "missing_artifact_count": sum(
            1
            for blocker in blockers
            if "missing" in blocker or "not_materialized" in blocker
        ),
        "entropy_position_label": manifest.get("entropy_position_label"),
        "entropy_stage_order": entropy_stage.get("order"),
        "entropy_stage_class": entropy_stage.get("class"),
        "entropy_pipeline_stage_index": _safe_int(
            entropy_stage.get("stage_index") or entropy_stage.get("order")
        ),
        "entropy_pipeline_information_effect_class": entropy_stage.get(
            "information_effect"
        ),
        "fractal_active_levels": _string_list(fractal_scope.get("active_levels")),
        "fractal_ordered_levels": _string_list(fractal_scope.get("ordered_levels")),
        "selection_blocker_class": policy_blocker_class,
        "receiver_credit_exhausted": policy_blocker_class == "receiver_credit_exhausted",
        "stackability_remeasure_required": (
            policy_blocker_class == "stackability_interaction_remeasure"
        ),
        "entropy_stage_contract_miss": (
            policy_blocker_class == "entropy_stage_contract_miss"
        ),
    }
    component_axis = target_row.get("component_response_replay_axis_tag")
    if component_response_replayed and not component_axis:
        component_axis = "[macOS-MLX research-signal]"
    signal = {
        "schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
        "learning_signal_kind": "repair_campaign_materialization_gate",
        "typed_response_id": (
            typed_response_id
            or "materialization:"
            f"{materialization_execution_report.get('chain_id')}:"
            f"{target_row.get('candidate_chain_id')}"
        ),
        "candidate_id": candidate_id or target_row.get("candidate_chain_id"),
        "family_id": family_id,
        "component_response_axis": (
            component_axis or "blocked_or_unmeasured_component_response_axis"
        ),
        "evidence_grade": "materialization_gate_local_custody_signal_only",
        "source_artifacts": source_artifacts,
        "replay_identity": {
            "schema": "repair_materialization_learning_replay_identity.v1",
            "replay_identity_kind": "repair_materialization_gate",
            "hash_manifest_sha256": _stable_sha256(identity),
            "source_records_sha256": _stable_sha256(
                {
                    "schema": "repair_materialization_learning_source.v1",
                    "source_artifacts": source_artifacts,
                }
            ),
            "replay_argv_sha256": None,
            "execution_context_sha256": None,
            "environment_sha256": None,
        },
        "local_planning_update": {
            "schema": REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA,
            "posterior_surface": "repair_campaign_stackability_local_mlx_posterior",
            "local_planning_update_ready": True,
            "recommended_acquisition_policy": policy,
            "recommended_stackability_followup": (
                "materialize_archive_and_runtime_proof_then_rerun_component_stackability"
            ),
            "planner_feature_vector": feature_vector,
            "posterior_update_blockers": [
                "materialization_learning_signal_is_not_score_authority",
                "exact_axis_component_response_required_before_budget_spend",
            ],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "blockers": blockers,
        "missing_artifacts": [
            blocker
            for blocker in blockers
            if "missing" in blocker or "not_materialized" in blocker
        ],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_materialization_gate_acquisition_update_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        signal,
        context="repair_campaign_materialization_learning_signal",
    )
    report = {
        "schema": REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
        "source_materialization_execution_report": source_artifacts[0],
        "source_materialization_gate": source_artifacts[1],
        "source_family_materializer_manifest": (
            source_artifacts[2] if len(source_artifacts) > 2 else None
        ),
        "blocked_signal_count": 1,
        "learning_signal_rows": [signal],
        "learning_signal_schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_materialization_learning_signal_bundle_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="repair_campaign_materialization_learning_signal_report",
    )
    return report


def build_repair_campaign_activation_plan_learning_signal_report(
    *,
    activation_plan_path: str | Path,
    activation_plan: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build posterior-consumable signals from a frozen child-queue activation plan."""

    if activation_plan.get("schema") != REPAIR_CAMPAIGN_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA:
        raise RepairCampaignLearningSignalError(
            "activation plan must be frontier_final_rate_attack_child_queue_activation_plan.v1"
        )
    require_no_truthy_authority_fields(
        activation_plan,
        context="repair_campaign_activation_plan_learning_signal_input",
    )
    source_artifact = _file_record(
        "child_queue_activation_plan",
        activation_plan_path,
        repo_root=repo_root,
    )
    signal_rows: list[dict[str, Any]] = []
    hard_blockers = [
        "activation_plan_learning_signal_is_not_score_authority",
        "component_response_or_receiver_credit_required_before_budget_spend",
        "exact_axis_eval_required_before_promotion_or_dispatch",
    ]
    queue_id = str(activation_plan.get("queue_id") or "unknown_child_queue")
    queue_sha = str(activation_plan.get("queue_sha256") or "")
    for index, experiment in enumerate(
        activation_plan.get("blocked_experiments") or [],
        start=1,
    ):
        if not isinstance(experiment, Mapping):
            continue
        experiment_id = str(experiment.get("experiment_id") or f"blocked_{index}")
        blockers = ordered_unique(
            [
                *_string_list(experiment.get("activation_blockers")),
                *hard_blockers,
            ]
        )
        action_rows = [
            row
            for row in experiment.get("activation_actions") or []
            if isinstance(row, Mapping)
        ]
        action_labels = ordered_unique(
            _string_list(
                [
                    row.get("activation_action")
                    for row in action_rows
                    if row.get("activation_action")
                ]
            )
        )
        evidence_surfaces = ordered_unique(
            _string_list(
                [
                    row.get("evidence_surface")
                    for row in action_rows
                    if row.get("evidence_surface")
                ]
            )
        )
        step_refs = [
            row
            for row in experiment.get("step_evidence_refs") or []
            if isinstance(row, Mapping)
        ]
        telemetry_inputs: list[str] = []
        postcondition_paths: list[str] = []
        for step in step_refs:
            telemetry_inputs.extend(
                _string_list(step.get("telemetry_input_artifact_paths"))
            )
            postcondition_paths.extend(_string_list(step.get("postcondition_paths")))
        missing_artifacts = ordered_unique([*telemetry_inputs, *postcondition_paths])
        policy_blocker_class = _policy_blocker_class(blockers, missing_artifacts)
        tags = _string_list(experiment.get("tags"))
        family_id = next(
            (
                tag
                for tag in tags
                if tag
                not in {
                    "frontier-rate-attack",
                    "no-score-authority",
                    "blocked-no-actionable-response",
                }
            ),
            "child_queue_activation",
        )
        identity = {
            "schema": "repair_campaign_activation_plan_learning_identity.v1",
            "queue_id": queue_id,
            "queue_sha256": queue_sha,
            "experiment_id": experiment_id,
            "blockers": blockers,
            "activation_actions": action_labels,
            "evidence_surfaces": evidence_surfaces,
            "missing_artifacts": missing_artifacts,
            "source_activation_plan_sha256": source_artifact.get("sha256"),
        }
        feature_vector = {
            "blocked_experiment_count": _safe_int(
                activation_plan.get("blocked_experiment_count")
            ),
            "activation_blocker_count": len(blockers),
            "activation_action_count": len(action_labels),
            "evidence_surface_count": len(evidence_surfaces),
            "missing_artifact_count": len(missing_artifacts),
            "step_count": _safe_int(experiment.get("step_count")),
            "has_targeted_component_response_request": any(
                "targeted_component" in item or "component_response" in item
                for item in [*blockers, *evidence_surfaces, *missing_artifacts]
            ),
            "has_receiver_closed_budget_request": any(
                "receiver_closed" in item or "saved_bytes" in item
                for item in [*blockers, *evidence_surfaces, *missing_artifacts]
            ),
            "has_exact_auth_eval_request": any(
                "exact_auth_eval" in item for item in [*blockers, *evidence_surfaces]
            ),
            "activation_actions": action_labels,
            "evidence_surfaces": evidence_surfaces,
            "selection_blocker_class": policy_blocker_class,
            "receiver_credit_exhausted": (
                policy_blocker_class == "receiver_credit_exhausted"
            ),
            "stackability_remeasure_required": (
                policy_blocker_class == "stackability_interaction_remeasure"
            ),
            "entropy_stage_contract_miss": (
                policy_blocker_class == "entropy_stage_contract_miss"
            ),
        }
        signal = {
            "schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
            "learning_signal_kind": "blocked_child_queue_activation_plan",
            "typed_response_id": f"activation_plan:{queue_id}:{experiment_id}",
            "candidate_id": experiment_id,
            "family_id": family_id,
            "component_response_axis": "blocked_or_unmeasured_component_response_axis",
            "evidence_grade": "blocked_queue_activation_plan_only",
            "source_artifacts": [source_artifact],
            "replay_identity": {
                "schema": "repair_campaign_activation_plan_replay_identity.v1",
                "replay_identity_kind": "blocked_child_queue_activation_plan",
                "hash_manifest_sha256": _stable_sha256(identity),
                "source_records_sha256": _stable_sha256(
                    {
                        "schema": "repair_campaign_activation_plan_source.v1",
                        "source_artifact": source_artifact,
                    }
                ),
                "replay_argv_sha256": None,
                "execution_context_sha256": None,
                "environment_sha256": None,
            },
            "local_planning_update": {
                "schema": REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA,
                "posterior_surface": "repair_campaign_stackability_local_mlx_posterior",
                "local_planning_update_ready": True,
                "recommended_acquisition_policy": _blocked_policy(
                    blockers,
                    missing_artifacts,
                ),
                "recommended_stackability_followup": (
                    "do_not_run_stackability_until_activation_blockers_clear"
                ),
                "planner_feature_vector": feature_vector,
                "posterior_update_blockers": hard_blockers,
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            },
            "blockers": blockers,
            "missing_artifacts": missing_artifacts,
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "allowed_use": "blocked_activation_plan_acquisition_update_only",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            signal,
            context=f"repair_campaign_activation_plan_learning_signal:{experiment_id}",
        )
        signal_rows.append(signal)
    report = {
        "schema": REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
        "source_activation_plan": source_artifact,
        "blocked_signal_count": len(signal_rows),
        "learning_signal_rows": signal_rows,
        "learning_signal_schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "blocked_activation_plan_learning_signal_bundle_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="repair_campaign_activation_plan_learning_signal_report",
    )
    return report


def build_repair_campaign_blocked_learning_signal_report(
    *,
    score_report_path: str | Path,
    score_report: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build posterior-consumable signals for blocked repair allocations."""

    if score_report.get("schema") != REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        raise RepairCampaignLearningSignalError(
            "score_report must be repair_campaign_score_report.v1"
        )
    require_no_truthy_authority_fields(
        score_report,
        context="repair_campaign_blocked_learning_signal_report_input",
    )
    decision = _mapping(score_report.get("optimizer_decision"))
    require_interaction_aware_optimizer_decision(
        decision,
        context="repair_campaign_blocked_learning_signal_report",
    )
    source_artifact = _file_record(
        "repair_campaign_score_report",
        score_report_path,
        repo_root=repo_root,
    )
    blocked_rows = [
        row
        for row in decision.get("blocked_allocation_rows") or []
        if isinstance(row, Mapping)
    ]
    signals: list[dict[str, Any]] = []
    hard_blockers = [
        "blocked_repair_learning_signal_is_not_score_authority",
        "exact_axis_component_response_required_before_budget_spend",
        "receiver_runtime_materialization_required_before_exact_dispatch",
    ]
    for index, row in enumerate(blocked_rows, start=1):
        typed_response_id = (
            str(row.get("typed_response_id") or "").strip()
            or f"blocked_repair_allocation_{index}"
        )
        missing_artifacts = ordered_unique(
            [
                *_string_list(row.get("missing_artifacts")),
                *_string_list(
                    _mapping(row.get("execution_gate")).get("missing_artifacts")
                ),
                *_string_list(
                    _mapping(row.get("receiver_proof_status")).get(
                        "missing_artifacts"
                    )
                ),
            ]
        )
        blockers = ordered_unique(
            [
                *_string_list(row.get("blockers")),
                *missing_artifacts,
                *hard_blockers,
            ]
        )
        entropy_pipeline = _entropy_pipeline(row)
        policy_blocker_class = _policy_blocker_class(blockers, missing_artifacts)
        identity = {
            "schema": "repair_campaign_blocked_learning_identity.v1",
            "typed_response_id": typed_response_id,
            "candidate_id": row.get("candidate_id"),
            "family_id": row.get("family_id"),
            "campaign_rank": row.get("campaign_rank"),
            "source_score_report_sha256": source_artifact.get("sha256"),
            "blockers": blockers,
            "missing_artifacts": missing_artifacts,
        }
        identity_sha = _stable_sha256(identity)
        component_terms = _mapping(row.get("component_response_terms"))
        palette_context = _mapping(row.get("palette_dynamics_context"))
        feature_vector = {
            "allocated_repair_bytes": 0,
            "requested_repair_bytes": _safe_int(row.get("requested_repair_bytes")),
            "expected_local_improvement_score_units": _safe_float(
                row.get("expected_local_improvement_score_units")
            ),
            "campaign_score": _safe_float(row.get("campaign_score")),
            "per_op_bytes_delta": row.get("per_op_bytes_delta"),
            "segnet_delta_score_units": component_terms.get(
                "segnet_delta_score_units"
            ),
            "posenet_delta_score_units": component_terms.get(
                "posenet_delta_score_units"
            ),
            "missing_artifact_count": len(missing_artifacts),
            "blocker_count": len(blockers),
            "entropy_position_label": row.get("entropy_position_label"),
            "targeted_dimensions": _string_list(row.get("targeted_dimensions")),
            "operation_levels": _string_list(row.get("operation_levels")),
            "palette_frame_asymmetry_multiplier": _safe_float(
                row.get("palette_frame_asymmetry_multiplier")
            ),
            "palette_zero_frame1_modes": palette_context.get("zero_frame1_modes")
            is True,
            "palette_frame0_non_identity_fraction": _safe_float(
                palette_context.get("frame0_non_identity_fraction")
            ),
            "palette_mode_count": _safe_int(palette_context.get("mode_count")),
            "entropy_pipeline_stage_index": _safe_int(
                row.get("entropy_pipeline_stage_index")
                or entropy_pipeline.get("stage_index")
            ),
            "entropy_position_class": entropy_pipeline.get("entropy_position_class"),
            "entropy_pipeline_information_effect_class": entropy_pipeline.get(
                "information_effect_class"
            ),
            "interaction_order": _interaction_order(row),
            "interaction_aware_selection_score": _safe_float(
                row.get("interaction_aware_selection_score")
            ),
            "remeasure_required_before_budget_spend": (
                row.get("remeasure_required_before_budget_spend") is True
            ),
            "selection_blocker_class": policy_blocker_class,
            "receiver_credit_exhausted": (
                policy_blocker_class == "receiver_credit_exhausted"
            ),
            "stackability_remeasure_required": (
                policy_blocker_class == "stackability_interaction_remeasure"
                or row.get("remeasure_required_before_budget_spend") is True
            ),
            "entropy_stage_contract_miss": (
                policy_blocker_class == "entropy_stage_contract_miss"
            ),
        }
        signal = {
            "schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
            "learning_signal_kind": "blocked_repair_campaign_allocation",
            "typed_response_id": typed_response_id,
            "candidate_id": row.get("candidate_id"),
            "family_id": row.get("family_id") or "unclassified_repair_family",
            "component_response_axis": (
                component_terms.get("response_axis")
                or "blocked_or_unmeasured_component_response_axis"
            ),
            "evidence_grade": "blocked_local_planning_signal_only",
            "source_artifacts": [source_artifact],
            "replay_identity": {
                "schema": "repair_campaign_blocked_learning_replay_identity.v1",
                "replay_identity_kind": "blocked_missing_artifact_no_replay",
                "hash_manifest_sha256": identity_sha,
                "source_records_sha256": _stable_sha256(
                    {
                        "schema": "repair_campaign_blocked_learning_source.v1",
                        "source_artifact": source_artifact,
                    }
                ),
                "replay_argv_sha256": None,
                "execution_context_sha256": None,
                "environment_sha256": None,
            },
            "local_planning_update": {
                "schema": REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA,
                "posterior_surface": (
                    "repair_campaign_stackability_local_mlx_posterior"
                ),
                "local_planning_update_ready": True,
                "recommended_acquisition_policy": _blocked_policy(
                    blockers,
                    missing_artifacts,
                ),
                "recommended_stackability_followup": (
                    "do_not_run_stackability_until_blockers_clear"
                ),
                "planner_feature_vector": feature_vector,
                "posterior_update_blockers": hard_blockers,
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            },
            "blockers": blockers,
            "missing_artifacts": missing_artifacts,
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "allowed_use": "blocked_repair_campaign_acquisition_update_only",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            signal,
            context=f"repair_campaign_blocked_learning_signal:{typed_response_id}",
        )
        signals.append(signal)
    report = {
        "schema": REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
        "source_score_report": source_artifact,
        "blocked_signal_count": len(signals),
        "learning_signal_rows": signals,
        "learning_signal_schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "blocked_repair_campaign_learning_signal_bundle_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="repair_campaign_blocked_learning_signal_report",
    )
    return report


def build_repair_campaign_learning_signal(
    *,
    score_report_path: str | Path,
    probe_path: str | Path,
    replay_bundle_path: str | Path,
    score_report: Mapping[str, Any],
    probe: Mapping[str, Any],
    replay_bundle: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build a false-authority local learning signal from one repair replay."""

    if score_report.get("schema") != REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        raise RepairCampaignLearningSignalError(
            "score_report must be repair_campaign_score_report.v1"
        )
    if probe.get("schema") != REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA:
        raise RepairCampaignLearningSignalError(
            "probe must be repair_campaign_stackability_probe.v1"
        )
    if replay_bundle.get("schema") != REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA:
        raise RepairCampaignLearningSignalError(
            "replay_bundle must be repair_campaign_stackability_replay_bundle.v1"
        )
    for label, payload in (
        ("score_report", score_report),
        ("probe", probe),
        ("replay_bundle", replay_bundle),
    ):
        require_no_truthy_authority_fields(
            payload,
            context=f"repair_campaign_learning_signal_input:{label}",
        )
    require_interaction_aware_optimizer_decision(
        _mapping(score_report.get("optimizer_decision")),
        context="repair_campaign_learning_signal",
    )
    typed_response_id = str(probe.get("typed_response_id") or "").strip()
    if not typed_response_id:
        raise RepairCampaignLearningSignalError("probe missing typed_response_id")
    if str(replay_bundle.get("typed_response_id") or "") != typed_response_id:
        raise RepairCampaignLearningSignalError(
            "replay bundle typed_response_id does not match probe"
        )

    allocation = _mapping(probe.get("optimizer_allocation"))
    score_row = _mapping(probe.get("source_score_row"))
    palette_context = _mapping(probe.get("palette_dynamics_context"))
    entropy_pipeline = _mapping(probe.get("entropy_pipeline_position")) or _entropy_pipeline(allocation)
    family_id = str(allocation.get("family_id") or score_row.get("family_id") or "")
    entropy_position = str(probe.get("entropy_position_label") or "")
    expected_improvement = _safe_float(
        probe.get("expected_local_improvement_score_units")
    )
    allocated_bytes = _safe_int(probe.get("allocated_repair_bytes"))
    improvement_per_allocated_byte = (
        expected_improvement / allocated_bytes if allocated_bytes > 0 else 0.0
    )
    hard_blockers = [
        "local_mlx_learning_signal_is_not_score_authority",
        "exact_axis_component_response_required_before_budget_spend",
        "receiver_runtime_materialization_required_before_exact_dispatch",
        "post_materialization_stackability_must_be_remeasured",
    ]
    signal = {
        "schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
        "typed_response_id": typed_response_id,
        "candidate_id": allocation.get("candidate_id") or score_row.get("candidate_id"),
        "family_id": family_id or "unclassified_repair_family",
        "component_response_axis": "[macOS-MLX research-signal]",
        "evidence_grade": "local_mlx_research_signal_only",
        "source_artifacts": [
            _file_record(
                "repair_campaign_score_report",
                score_report_path,
                repo_root=repo_root,
            ),
            _file_record("repair_campaign_stackability_probe", probe_path, repo_root=repo_root),
            _file_record(
                "repair_campaign_stackability_replay_bundle",
                replay_bundle_path,
                repo_root=repo_root,
            ),
        ],
        "replay_identity": {
            "hash_manifest_sha256": replay_bundle.get("hash_manifest_sha256"),
            "source_records_sha256": replay_bundle.get("source_records_sha256"),
            "replay_argv_sha256": replay_bundle.get("replay_argv_sha256"),
            "execution_context_sha256": replay_bundle.get("execution_context_sha256"),
            "environment_sha256": replay_bundle.get("environment_sha256"),
        },
        "local_planning_update": {
            "schema": REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA,
            "posterior_surface": "repair_campaign_stackability_local_mlx_posterior",
            "local_planning_update_ready": probe.get("stackability_ready") is True,
            "recommended_acquisition_policy": (
                "increase_priority_for_exact_axis_component_response_replay"
                if expected_improvement > 0.0
                else "hold_until_additional_local_response_signal"
            ),
            "recommended_stackability_followup": (
                "remeasure_after_parent_rate_materialization_and_sibling_repairs"
            ),
            "planner_feature_vector": {
                "allocated_repair_bytes": allocated_bytes,
                "receiver_closed_rate_credit_bytes": _safe_int(
                    probe.get("receiver_closed_rate_credit_bytes")
                ),
                "remaining_receiver_closed_rate_credit_bytes_after": _safe_int(
                    probe.get("remaining_receiver_closed_rate_credit_bytes_after")
                ),
                "expected_local_improvement_score_units": expected_improvement,
                "improvement_per_allocated_byte": improvement_per_allocated_byte,
                "interaction_penalty": _safe_float(probe.get("interaction_penalty")),
                "interaction_order": _safe_int(
                    probe.get("interaction_order")
                    or allocation.get("interaction_order")
                    or score_row.get("interaction_order")
                ),
                "interaction_aware_selection_score": _safe_float(
                    allocation.get("interaction_aware_selection_score")
                    or score_row.get("interaction_aware_selection_score")
                ),
                "remeasure_required_before_budget_spend": (
                    allocation.get("remeasure_required_before_budget_spend") is True
                    or score_row.get("remeasure_required_before_budget_spend") is True
                ),
                "entropy_position_label": entropy_position,
                "entropy_position_class": probe.get("entropy_position_class"),
                "entropy_pipeline_stage_index": _safe_int(
                    probe.get("entropy_pipeline_stage_index")
                    or entropy_pipeline.get("stage_index")
                ),
                "entropy_pipeline_information_effect_class": entropy_pipeline.get(
                    "information_effect_class"
                ),
                "targeted_dimensions": _string_list(
                    allocation.get("targeted_dimensions")
                    or score_row.get("targeted_dimensions")
                ),
                "operation_levels": _string_list(
                    allocation.get("operation_levels") or score_row.get("operation_levels")
                ),
                "palette_frame_asymmetry_multiplier": _safe_float(
                    probe.get("palette_frame_asymmetry_multiplier")
                ),
                "palette_zero_frame1_modes": palette_context.get("zero_frame1_modes")
                is True,
                "palette_frame0_non_identity_fraction": _safe_float(
                    palette_context.get("frame0_non_identity_fraction")
                ),
                "palette_mode_count": _safe_int(palette_context.get("mode_count")),
            },
            "posterior_update_blockers": hard_blockers,
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "blockers": ordered_unique([*_string_list(probe.get("blockers")), *hard_blockers]),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "local_repair_campaign_planning_and_acquisition_update_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        signal,
        context="repair_campaign_learning_signal",
    )
    return signal


__all__ = [
    "REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA",
    "REPAIR_CAMPAIGN_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA",
    "REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA",
    "REPAIR_CAMPAIGN_LOCAL_PLANNING_UPDATE_SCHEMA",
    "RepairCampaignLearningSignalError",
    "build_repair_campaign_activation_plan_learning_signal_report",
    "build_repair_campaign_blocked_learning_signal_report",
    "build_repair_campaign_learning_signal",
    "build_repair_campaign_materialization_learning_signal_report",
]
