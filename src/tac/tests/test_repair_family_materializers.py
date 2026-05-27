# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.repair_campaign_chain_contract import (
    RepairCampaignChainContractError,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
    build_repair_campaign_blocked_learning_signal_report,
    build_repair_campaign_materialization_learning_signal_report,
)
from tac.optimization.repair_campaign_posterior import (
    append_repair_campaign_blocked_learning_signal_report,
    load_repair_campaign_stackability_posterior_rows,
)
from tac.optimization.repair_campaign_scorer import (
    build_repair_campaign_posterior_prior_summary,
    score_repair_campaign,
)
from tac.optimization.repair_family_materializers import (
    REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
    build_repair_campaign_family_materializer_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def _repair_payload(tmp_path: Path) -> dict[str, object]:
    mlx = tmp_path / "segnet_mlx_response.json"
    ref = tmp_path / "segnet_reference_mlx_response.json"
    mlx.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    ref.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    return {
        "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
        "chain_id": "unit_repair_chain",
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": 40,
            **_false_authority(),
        },
        "typed_response_ledger": {
            "schema": "frontier_rate_attack_repair_budget_typed_response_ledger.v1",
            "available_receiver_closed_rate_credit_bytes": 40,
            "rows": [
                {
                    "schema": (
                        "frontier_rate_attack_repair_budget_typed_response_row.v1"
                    ),
                    "typed_response_id": "segnet_region_ready",
                    "candidate_id": "segnet_class_region_waterfill",
                    "correction_family": "segnet_class_region_waterfill",
                    "targeted_dimensions": ["segnet", "region"],
                    "operation_levels": ["pixel", "boundary", "region", "frame"],
                    "entropy_position_label": (
                        "before_entropy_coder_distribution_shaping"
                    ),
                    "requested_repair_bytes": 32,
                    "objective_delta_score_units": -0.0010,
                    "local_mlx_response_path": str(mlx),
                    "reference_local_mlx_response_path": str(ref),
                    "segnet_class_region_mask_ids": ["road_boundary"],
                    **_false_authority(),
                }
            ],
            **_false_authority(),
        },
        **_false_authority(),
    }


def _plan_from_score_report(score_report: dict[str, object]) -> dict[str, object]:
    child_id = "repair_budget_spent_child_unit_segnet"
    allocation = score_report["optimizer_decision"]["selected_allocation_rows"][0]  # type: ignore[index]
    return {
        "schema": "frontier_rate_attack_repair_budget_materialization_plan.v1",
        "chain_id": "unit_repair_chain",
        "parent_candidate_chain_id": "rate_parent",
        "candidate_chain_rows": [
            {
                "schema": "frontier_rate_attack_repair_budget_materialization_plan_row.v1",
                "candidate_kind": "rate_only_floor_parent",
                "candidate_chain_id": "rate_parent",
                "materialization_order": 1,
                "candidate_archive_materialized": False,
                "receiver_consumed": False,
                **_false_authority(),
            },
            {
                "schema": "frontier_rate_attack_repair_budget_materialization_plan_row.v1",
                "candidate_kind": "spent_budget_repair_child",
                "candidate_chain_id": child_id,
                "parent_candidate_chain_id": "rate_parent",
                "materialization_order": 2,
                "typed_response_id": allocation["typed_response_id"],
                "allocation_candidate_id": allocation["candidate_id"],
                "correction_family": allocation["correction_family"],
                "operation_levels": ["pixel", "boundary", "region", "frame"],
                "entropy_position_label": allocation["entropy_position_label"],
                "candidate_archive_materialized": False,
                "receiver_consumed": False,
                **_false_authority(),
            },
        ],
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **_false_authority(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def test_segnet_family_materializer_emits_ordered_fail_closed_manifest(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)

    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )

    assert manifest["schema"] == REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA
    assert manifest["target_kind"] == "segnet_class_region_waterfill"
    assert manifest["candidate_chain_ids"] == ["repair_budget_spent_child_unit_segnet"]
    assert manifest["active_entropy_stage"]["order"] == 10
    assert manifest["fractal_optimization_scope"]["ordered_levels"] == [
        "bit",
        "byte",
        "pixel",
        "boundary",
        "region",
        "frame",
        "pair",
        "batch",
        "full_video",
    ]
    assert manifest["component_response_replayed"] is True
    assert manifest["byte_closed_candidate_emitted"] is False
    assert (
        "segnet_class_region_waterfill_byte_closed_candidate_archive_not_materialized"
        in manifest["readiness_blockers"]
    )
    assert "segnet_class_region_mask_ids_missing" not in manifest["readiness_blockers"]
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_family_materializer_cli_writes_manifest(tmp_path: Path) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    score_report_path = _write_json(tmp_path / "score_report.json", score_report)
    plan_path = _write_json(tmp_path / "plan.json", plan)
    manifest_path = tmp_path / "family_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_repair_campaign_family_materializer_manifest.py"),
            "--materialization-plan",
            str(plan_path),
            "--score-report",
            str(score_report_path),
            "--typed-response-id",
            "segnet_region_ready",
            "--candidate-id",
            "segnet_class_region_waterfill",
            "--materializer-manifest-out",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA
    assert manifest["component_response_replayed"] is True
    assert manifest["score_claim"] is False


def test_materialization_gate_learning_signal_updates_posterior(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    family_manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    family_manifest_path = _write_json(tmp_path / "family_manifest.json", family_manifest)
    execution_report = {
        "schema": "frontier_rate_attack_repair_budget_materialization_execution_report.v1",
        "chain_id": "unit_repair_chain",
        "candidate_archive_materialized": False,
        "runtime_consumption_proof_present": False,
        "receiver_consumed": False,
        "component_response_replayed": True,
        "execution_rows": [
            {
                "schema": "frontier_rate_attack_repair_budget_materialization_execution_row.v1",
                "candidate_kind": "spent_budget_repair_child",
                "candidate_chain_id": "repair_budget_spent_child_unit_segnet",
                "candidate_archive_materialized": False,
                "runtime_consumption_proof_present": False,
                "receiver_consumed": False,
                "component_response_replayed": True,
                "component_response_replay_axis_tag": "[macOS-MLX research-signal]",
                "blockers": ["candidate_archive_materialized_false"],
                **_false_authority(),
            }
        ],
        "blockers": ["candidate_archives_not_materialized"],
        **_false_authority(),
    }
    gate = {
        "schema": "repair_campaign_byte_closed_materialization_gate.v1",
        "typed_response_id": "segnet_region_ready",
        "candidate_id": "segnet_class_region_waterfill",
        "candidate_archive_materialized": False,
        "archive_bound_runtime_consumption_proof_ready": False,
        "component_response_replayed": True,
        "blockers": ["candidate_archive_materialized_false"],
        **_false_authority(),
    }
    execution_report_path = _write_json(tmp_path / "execution_report.json", execution_report)
    gate_path = _write_json(tmp_path / "gate.json", gate)

    signal_report = build_repair_campaign_materialization_learning_signal_report(
        materialization_execution_report_path=execution_report_path,
        materialization_execution_report=execution_report,
        materialization_gate_path=gate_path,
        materialization_gate=gate,
        family_materializer_manifest_path=family_manifest_path,
        family_materializer_manifest=family_manifest,
        repo_root=tmp_path,
    )

    assert signal_report["schema"] == REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA
    signal = signal_report["learning_signal_rows"][0]
    update = signal["local_planning_update"]
    assert update["recommended_acquisition_policy"] == (
        "prioritize_byte_closed_family_materializer_implementation"
    )
    assert update["planner_feature_vector"]["entropy_stage_order"] == 10
    signal_report_path = _write_json(tmp_path / "signal_report.json", signal_report)
    posterior_path = tmp_path / "posterior.jsonl"
    append_report = append_repair_campaign_blocked_learning_signal_report(
        blocked_learning_signal_report_path=signal_report_path,
        blocked_learning_signal_report=signal_report,
        posterior_path=posterior_path,
        lock_path=tmp_path / ".posterior.lock",
        repo_root=tmp_path,
    )
    posterior_rows = load_repair_campaign_stackability_posterior_rows(posterior_path)
    summary = build_repair_campaign_posterior_prior_summary(
        posterior_path=posterior_path,
    )

    assert append_report["appended_count"] == 1
    assert posterior_rows[0]["typed_response_id"] == "segnet_region_ready"
    assert posterior_rows[0]["acquisition_policy_delta"][
        "family_priority_direction"
    ] == "increase"
    assert posterior_rows[0]["acquisition_policy_delta"][
        "posterior_budget_routing_hint"
    ] == "route_budget_to_byte_closed_materializer_after_custody"
    route = summary["acquisition_followup_routes"][0]
    assert route["activation_action"] == "implement_or_run_repair_family_byte_transform"


def test_blocked_credit_exhaustion_updates_posterior_budget_routing(
    tmp_path: Path,
) -> None:
    payload = _repair_payload(tmp_path)
    payload["receiver_closed_rate_credit"]["receiver_closed_saved_bytes_total"] = 0
    payload["typed_response_ledger"]["available_receiver_closed_rate_credit_bytes"] = 0
    score_report = score_repair_campaign(payload=payload, repo_root=tmp_path)
    score_report_path = _write_json(tmp_path / "score_report.json", score_report)

    signal_report = build_repair_campaign_blocked_learning_signal_report(
        score_report_path=score_report_path,
        score_report=score_report,
        repo_root=tmp_path,
    )
    signal = signal_report["learning_signal_rows"][0]
    update = signal["local_planning_update"]

    assert update["recommended_acquisition_policy"] == (
        "increase_receiver_closed_rate_credit_or_rebudget_earlier_entropy_stage"
    )
    assert update["planner_feature_vector"]["selection_blocker_class"] == (
        "receiver_credit_exhausted"
    )
    assert update["planner_feature_vector"]["receiver_credit_exhausted"] is True

    signal_report_path = _write_json(tmp_path / "blocked_signals.json", signal_report)
    posterior_path = tmp_path / "posterior.jsonl"
    append_repair_campaign_blocked_learning_signal_report(
        blocked_learning_signal_report_path=signal_report_path,
        blocked_learning_signal_report=signal_report,
        posterior_path=posterior_path,
        lock_path=tmp_path / ".posterior.lock",
        repo_root=tmp_path,
    )
    posterior_rows = load_repair_campaign_stackability_posterior_rows(posterior_path)
    summary = build_repair_campaign_posterior_prior_summary(
        posterior_path=posterior_path,
    )
    route = summary["acquisition_followup_routes"][0]

    assert posterior_rows[0]["acquisition_policy_delta"][
        "posterior_budget_routing_hint"
    ] == "rebudget_receiver_closed_credit_before_exact_axis_spend"
    assert route["activation_action"] == (
        "rebudget_receiver_credit_to_earliest_entropy_stage"
    )
    assert route["queue_artifact_key"] == "repair_budget_waterfill_queue"


def test_family_materializer_rejects_stale_optimizer_solver_contract(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    score_report["optimizer_decision"]["solver"] = "greedy_campaign_score_waterfill_v1"
    plan = _plan_from_score_report(score_report)

    with pytest.raises(RepairCampaignChainContractError, match="requires solver"):
        build_repair_campaign_family_materializer_manifest(
            repo_root=tmp_path,
            materialization_plan=plan,
            score_report=score_report,
            materialization_plan_path=tmp_path / "plan.json",
            score_report_path=tmp_path / "score_report.json",
            typed_response_id="segnet_region_ready",
            candidate_id="segnet_class_region_waterfill",
        )
