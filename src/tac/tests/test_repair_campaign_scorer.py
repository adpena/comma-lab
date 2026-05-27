# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.optimization.repair_campaign_learning_signal import (
    build_repair_campaign_blocked_learning_signal_report,
)
from tac.optimization.repair_campaign_scorer import (
    REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA,
    REPAIR_CAMPAIGN_POSTERIOR_PRIOR_SUMMARY_SCHEMA,
    REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
    REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
    REPAIR_OPERATOR_FAMILY_PRIORS_SCHEMA,
    build_repair_campaign_stackability_probe,
    repair_operator_family_priors,
    score_repair_campaign,
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


def _work_order(tmp_path: Path) -> dict[str, object]:
    mlx = tmp_path / "segnet_mlx_response.json"
    ref = tmp_path / "segnet_reference_mlx_response.json"
    mlx.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    ref.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    return {
        "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
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
                    "acquisition_id": "segnet_region_acq",
                    "correction_family": "segnet_class_region_waterfill",
                    "targeted_dimensions": ["segnet", "region"],
                    "operation_levels": ["frame", "region"],
                    "entropy_position_label": (
                        "before_entropy_coder_distribution_shaping"
                    ),
                    "requested_repair_bytes": 32,
                    "objective_delta_score_units": -0.0010,
                    "allocation_action_term": {
                        "schema": (
                            "frontier_rate_attack_repair_budget_waterfill_"
                            "allocation_action_term.v1"
                        ),
                        "T_i": {
                            "archive_byte_delta_vs_baseline": -4,
                        },
                        "legal_runtime_constraints": [
                            "receiver_consumes_materialized_runtime_output",
                            "component_response_replayed_before_budget_spend",
                        ],
                        **_false_authority(),
                    },
                    "local_mlx_component_terms": {
                        "segnet_delta_score_units": -0.0007,
                        "posenet_delta_score_units": -0.0003,
                        **_false_authority(),
                    },
                    "local_mlx_response_path": str(mlx),
                    "reference_local_mlx_response_path": str(ref),
                    "interaction_scope": {
                        "pair_indices": [7, 9],
                        "region_ids": ["road_boundary"],
                        **_false_authority(),
                    },
                    "stacking_interaction_terms": {
                        "must_remeasure_with_parent_and_sibling_repairs": True,
                        **_false_authority(),
                    },
                    **_false_authority(),
                },
                {
                    "schema": (
                        "frontier_rate_attack_repair_budget_typed_response_row.v1"
                    ),
                    "typed_response_id": "selector_missing",
                    "candidate_id": "per_region_selector_codec",
                    "acquisition_id": "selector_acq",
                    "correction_family": "per_region_selector_codec",
                    "targeted_dimensions": ["selector_stream", "region"],
                    "operation_levels": ["entropy_coder"],
                    "entropy_position_label": "selector_codec_entropy",
                    "requested_repair_bytes": 8,
                    "objective_delta_score_units": -0.0008,
                    **_false_authority(),
                },
            ],
            **_false_authority(),
        },
        **_false_authority(),
    }


def test_repair_operator_family_priors_are_first_class_false_authority() -> None:
    priors = repair_operator_family_priors()

    assert priors["schema"] == REPAIR_OPERATOR_FAMILY_PRIORS_SCHEMA
    family_ids = {row["family_id"] for row in priors["rows"]}
    assert {
        "posenet_null_bottom_decile",
        "segnet_class_region_waterfill",
        "per_region_selector_codec",
        "palette_frame_asymmetry_prior",
        "entropy_position_cascade",
    }.issubset(family_ids)
    assert priors["ready_for_exact_eval_dispatch"] is False
    assert all(row["score_claim"] is False for row in priors["rows"])


def test_score_repair_campaign_preserves_cascade_opportunity_as_blocked_signal(
    tmp_path: Path,
) -> None:
    work_order = {
        "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": 24,
            **_false_authority(),
        },
        "repair_cascade_opportunity_rows": [
            {
                "schema": "frontier_rate_attack_repair_cascade_opportunity_row.v1",
                "cascade_id": "cascade_c_posenet_null_segnet_region_selector_codec",
                "label": "Cascade C",
                "source_relation": "PR110-OPT-5+7+10+12_UNTOUCHED",
                "pipeline_position": "scorer_entropy_repair_before_selector_codec",
                "targeted_positions": [
                    {"position_id": "P19", "entropy_surface": "scorer_entropy"},
                    {"position_id": "P18", "entropy_surface": "scorer_entropy"},
                    {"position_id": "P11", "entropy_surface": "selector_codec_entropy"},
                ],
                "required_probe_measurements": [
                    "posenet_null_bottom_decile_pair_ids",
                    "segnet_class_region_mask_ids",
                    "selector_payload_bits_per_region",
                ],
                "next_queue_action": (
                    "build_cascade_c_mlx_local_probe_queue_and_emit_component_"
                    "response_rows"
                ),
                "blockers": [
                    "cascade_c_empirical_component_response_missing",
                    "per_region_selector_codec_materializer_missing",
                ],
                **_false_authority(),
            }
        ],
        **_false_authority(),
    }

    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)

    assert report["row_count"] == 1
    assert report["structural_repair_opportunity_count"] == 1
    assert report["ready_for_local_mlx_advisory_execution_count"] == 0
    row = report["rows"][0]
    assert row["source_row_kind"] == "repair_cascade_opportunity"
    assert row["family_id"] == "entropy_position_cascade"
    assert row["cascade_id"] == "cascade_c_posenet_null_segnet_region_selector_codec"
    assert row["entropy_position_label"] == (
        "scorer_entropy_repair_before_selector_codec"
    )
    assert "cascade_c_empirical_component_response_missing" in (
        row["execution_gate"]["missing_artifacts"]
    )
    assert "entropy_position_cascade_probe_missing" in (
        row["execution_gate"]["missing_artifacts"]
    )
    decision = report["optimizer_decision"]
    assert decision["selected_allocation_count"] == 0
    assert decision["blocked_allocation_count"] == 1
    blocked = decision["blocked_allocation_rows"][0]
    assert blocked["family_id"] == "entropy_position_cascade"
    assert "local_mlx_structural_cascade_probe_missing" in (
        blocked["missing_artifacts"]
    )
    assert "requested_repair_bytes_missing" in blocked["blockers"]

    score_report_path = tmp_path / "repair_campaign_score_report.json"
    score_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    blocked_learning = build_repair_campaign_blocked_learning_signal_report(
        score_report_path=score_report_path,
        score_report=report,
        repo_root=tmp_path,
    )
    signal = blocked_learning["learning_signal_rows"][0]
    assert signal["family_id"] == "entropy_position_cascade"
    assert "cascade_c_empirical_component_response_missing" in (
        signal["missing_artifacts"]
    )
    assert signal["ready_for_exact_eval_dispatch"] is False


def test_score_repair_campaign_ranks_ready_mlx_and_names_missing_artifacts(
    tmp_path: Path,
) -> None:
    report = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)

    assert report["schema"] == REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA
    assert report["default_campaign_scorer"] is True
    assert report["ready_for_local_mlx_advisory_execution_count"] == 1
    assert report["blocked_missing_artifact_count"] == 1
    assert report["ready_for_exact_eval_dispatch"] is False
    decision = report["optimizer_decision"]
    assert decision["schema"] == REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA
    assert decision["receiver_closed_rate_credit_bytes"] == 40
    assert decision["selected_allocation_count"] == 1
    assert decision["allocated_repair_bytes_total"] == 32
    assert decision["unallocated_receiver_closed_rate_credit_bytes"] == 8
    assert decision["selected_allocation_rows"][0]["typed_response_id"] == (
        "segnet_region_ready"
    )
    assert decision["budget_spend_allowed"] is False
    assert decision["ready_for_exact_eval_dispatch"] is False
    first = report["rows"][0]
    assert first["schema"] == REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA
    assert first["typed_response_id"] == "segnet_region_ready"
    assert first["family_id"] == "segnet_class_region_waterfill"
    assert first["execution_gate"]["recommended_queue_status"] == (
        "ready_for_local_mlx_advisory_execution"
    )
    assert first["per_op_bytes_delta"] == -4
    assert first["component_response_terms"]["segnet_delta_score_units"] == -0.0007
    assert first["component_response_terms"]["posenet_delta_score_units"] == -0.0003
    assert (
        "runtime_consumption_proof_path:missing_or_unverified"
        in first["receiver_proof_status"]["missing_artifacts"]
    )
    assert "receiver_consumes_materialized_runtime_output" in (
        first["hard_legal_runtime_constraints"]
    )
    assert first["campaign_score"] > 0.0
    allocation = decision["selected_allocation_rows"][0]
    assert allocation["per_op_bytes_delta"] == -4
    assert allocation["component_response_terms"]["segnet_delta_score_units"] == (
        -0.0007
    )
    assert (
        "runtime_consumption_proof_path:missing_or_unverified"
        in allocation["receiver_proof_status"]["missing_artifacts"]
    )
    blocked_allocation = decision["blocked_allocation_rows"][0]
    assert blocked_allocation["typed_response_id"] == "selector_missing"
    assert blocked_allocation["family_id"] == "per_region_selector_codec"
    assert blocked_allocation["execution_gate"]["recommended_queue_status"] == (
        "blocked_missing_artifact"
    )
    assert "per_region_selector_codec_replay_missing" in (
        blocked_allocation["missing_artifacts"]
    )
    assert "runtime_consumption_proof_path:missing_or_unverified" in (
        blocked_allocation["receiver_proof_status"]["missing_artifacts"]
    )
    assert blocked_allocation["component_response_terms"]["response_axis"] == (
        "unknown_or_unmeasured_component_response_axis"
    )
    assert "component_response_replayed_before_budget_spend" in (
        blocked_allocation["hard_legal_runtime_constraints"]
    )
    blocked = report["rows"][1]
    assert blocked["typed_response_id"] == "selector_missing"
    assert blocked["execution_gate"]["recommended_queue_status"] == (
        "blocked_missing_artifact"
    )
    assert "runtime_consumption_proof_path" in " ".join(
        blocked["execution_gate"]["missing_artifacts"]
    )
    assert "per_region_selector_codec_replay_missing" in report["missing_artifacts"]


def test_score_repair_campaign_folds_stackability_posterior_into_priors(
    tmp_path: Path,
) -> None:
    posterior_path = tmp_path / "repair_campaign_stackability_posterior.jsonl"
    posterior_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema": "repair_campaign_stackability_posterior_row.v1",
                        "typed_response_id": "segnet_region_prior_a",
                        "candidate_id": "segnet_class_region_waterfill",
                        "family_id": "segnet_class_region_waterfill",
                        "evidence_grade": "local_mlx_research_signal_only",
                        "acquisition_policy_delta": {
                            "recommended_acquisition_policy": (
                                "increase_priority_for_exact_axis_component_response_replay"
                            ),
                            "family_priority_direction": "increase",
                            **_false_authority(),
                        },
                        "planner_feature_vector": {
                            "expected_local_improvement_score_units": 0.002,
                            "improvement_per_allocated_byte": 0.00005,
                            "missing_artifact_count": 0,
                            "blocker_count": 4,
                        },
                        **_false_authority(),
                    }
                ),
                json.dumps(
                    {
                        "schema": "repair_campaign_stackability_posterior_row.v1",
                        "typed_response_id": "selector_blocked_prior",
                        "candidate_id": "per_region_selector_codec",
                        "family_id": "per_region_selector_codec",
                        "evidence_grade": "blocked_local_planning_signal_only",
                        "acquisition_policy_delta": {
                            "recommended_acquisition_policy": (
                                "materialize_missing_local_mlx_custody_before_stackability"
                            ),
                            "family_priority_direction": "hold",
                            **_false_authority(),
                        },
                        "planner_feature_vector": {
                            "expected_local_improvement_score_units": 0.0002,
                            "improvement_per_allocated_byte": 0.0,
                            "missing_artifact_count": 6,
                            "blocker_count": 10,
                        },
                        **_false_authority(),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    baseline = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)
    with_posterior = score_repair_campaign(
        payload=_work_order(tmp_path),
        repo_root=tmp_path,
        posterior_path=posterior_path,
    )

    summary = with_posterior["posterior_prior_summary"]
    assert summary["schema"] == REPAIR_CAMPAIGN_POSTERIOR_PRIOR_SUMMARY_SCHEMA
    assert summary["posterior_row_count"] == 2
    assert summary["family_prior_count"] == 2
    row = with_posterior["rows"][0]
    assert row["typed_response_id"] == "segnet_region_ready"
    assert row["posterior_prior_multiplier"] > 1.0
    assert row["posterior_family_prior"]["observation_count"] == 1
    assert row["campaign_score"] > baseline["rows"][0]["campaign_score"]
    allocation = with_posterior["optimizer_decision"]["selected_allocation_rows"][0]
    assert allocation["posterior_prior_multiplier"] == row[
        "posterior_prior_multiplier"
    ]
    assert with_posterior["optimizer_decision"]["posterior_prior_summary"][
        "posterior_row_count"
    ] == 2


def test_repair_campaign_stackability_probe_requires_mlx_custody(
    tmp_path: Path,
) -> None:
    report = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)

    ready = build_repair_campaign_stackability_probe(
        score_report=report,
        typed_response_id="segnet_region_ready",
        repo_root=tmp_path,
    )

    assert ready["schema"] == REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA
    assert ready["status"] == "ready_for_local_mlx_stackability_probe"
    assert ready["stackability_ready"] is True
    assert ready["allocated_repair_bytes"] == 32
    assert ready["budget_spend_allowed"] is False
    assert ready["ready_for_exact_eval_dispatch"] is False

    blocked = build_repair_campaign_stackability_probe(
        score_report=report,
        typed_response_id="selector_missing",
        repo_root=tmp_path,
    )

    assert blocked["schema"] == REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA
    assert blocked["status"] == "blocked_missing_artifact"
    assert blocked["stackability_ready"] is False
    assert "optimizer_selected_allocation_missing" in blocked["blockers"]
    assert "local_mlx_advisory_custody_missing" in blocked["blockers"]
    assert "per_region_selector_codec_replay_missing" in blocked["missing_artifacts"]


def test_score_repair_campaign_cli_writes_report(tmp_path: Path) -> None:
    work_order_path = tmp_path / "work_order.json"
    report_path = tmp_path / "repair_campaign_score_report.json"
    work_order_path.write_text(
        json.dumps(_work_order(tmp_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/score_repair_campaign.py",
            "--work-order",
            str(work_order_path),
            "--score-report-out",
            str(report_path),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_result = json.loads(result.stdout)
    assert cli_result["ready_for_exact_eval_dispatch"] is False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA
    assert report["rows"][0]["typed_response_id"] == "segnet_region_ready"


def test_repair_campaign_stackability_probe_cli_writes_probe(tmp_path: Path) -> None:
    report_path = tmp_path / "repair_campaign_score_report.json"
    probe_path = tmp_path / "repair_campaign_stackability_probe.json"
    report = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_repair_campaign_stackability_probe.py",
            "--score-report",
            str(report_path),
            "--typed-response-id",
            "segnet_region_ready",
            "--output",
            str(probe_path),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_result = json.loads(result.stdout)
    assert cli_result["ready_for_exact_eval_dispatch"] is False
    assert cli_result["stackability_ready"] is True
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    assert probe["schema"] == REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA
    assert probe["typed_response_id"] == "segnet_region_ready"
