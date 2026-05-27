# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tac.fec6_selector_operator_space import FEC6_FIXED_K16_MODE_IDS
from tac.optimization.family_agnostic_materializers import (
    RUNTIME_CONSUMPTION_PROOF_SCHEMA,
)
from tac.optimization.repair_campaign_learning_signal import (
    build_repair_campaign_blocked_learning_signal_report,
)
from tac.optimization.repair_campaign_scorer import (
    REPAIR_CAMPAIGN_MULTISCALE_ACTION_LEDGER_SCHEMA,
    REPAIR_CAMPAIGN_MULTISCALE_ACTION_ROW_SCHEMA,
    REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA,
    REPAIR_CAMPAIGN_POSTERIOR_ACQUISITION_FOLLOWUP_SCHEMA,
    REPAIR_CAMPAIGN_POSTERIOR_PRIOR_SUMMARY_SCHEMA,
    REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
    REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
    REPAIR_OPERATOR_FAMILY_PRIORS_SCHEMA,
    build_repair_campaign_posterior_prior_summary,
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
                            "compressed_bit_delta_vs_baseline": -32,
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
                    "segnet_class_region_mask_ids": ["road_boundary"],
                    "interaction_scope": {
                        "pair_indices": [7, 9],
                        "region_ids": ["road_boundary"],
                        "pixel_count": 2048,
                        "batch_count": 1,
                        "full_video_id": "unit_video",
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


def _palette_work_order(tmp_path: Path) -> dict[str, object]:
    mlx = tmp_path / "palette_mlx_response.json"
    ref = tmp_path / "palette_reference_mlx_response.json"
    probe_matrix = tmp_path / "repair_dynamics_palette_probe_matrix.json"
    mlx.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    ref.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    probe_matrix.write_text(
        '{"schema":"frontier_rate_attack_repair_dynamics_palette_probe_matrix.v1"}\n',
        encoding="utf-8",
    )
    return {
        "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": 16,
            **_false_authority(),
        },
        "typed_response_ledger": {
            "schema": "frontier_rate_attack_repair_budget_typed_response_ledger.v1",
            "available_receiver_closed_rate_credit_bytes": 16,
            "rows": [
                {
                    "schema": (
                        "frontier_rate_attack_repair_budget_typed_response_row.v1"
                    ),
                    "typed_response_id": "palette_frame0_k16_ready",
                    "candidate_id": "repair_dynamics_frame0_palette_interaction",
                    "acquisition_id": "palette_frame0_acq",
                    "correction_family": (
                        "repair_dynamics_frame0_palette_interaction_waterfill"
                    ),
                    "targeted_dimensions": ["palette", "frame0", "posenet"],
                    "operation_levels": [
                        "pixel",
                        "boundary",
                        "region",
                        "frame",
                        "pair",
                        "batch",
                    ],
                    "entropy_position_label": (
                        "before_entropy_coder_distribution_shaping"
                    ),
                    "requested_repair_bytes": 12,
                    "objective_delta_score_units": -0.0009,
                    "local_mlx_component_terms": {
                        "segnet_delta_score_units": -0.0002,
                        "posenet_delta_score_units": -0.0007,
                        **_false_authority(),
                    },
                    "local_mlx_response_path": str(mlx),
                    "reference_local_mlx_response_path": str(ref),
                    "repair_dynamics_palette_probe_matrix_path": str(probe_matrix),
                    "repair_dynamics_palette_prior": {
                        "schema": (
                            "frontier_rate_attack_repair_dynamics_palette_prior.v1"
                        ),
                        "source": "unit_live_6bae0201_archive_manifest",
                        "palette_modes": list(FEC6_FIXED_K16_MODE_IDS),
                        **_false_authority(),
                    },
                    "interaction_scope": {
                        "pixel_count": 4096,
                        "region_ids": ["pose_sensitive_boundary"],
                        "pair_indices": [3, 4, 5],
                        "batch_count": 1,
                        **_false_authority(),
                    },
                    "stacking_interaction_terms": {
                        "must_remeasure_with_parent_and_sibling_repairs": True,
                        **_false_authority(),
                    },
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
    palette = next(
        row for row in priors["rows"] if row["family_id"] == "palette_frame_asymmetry_prior"
    )
    assert palette["empirical_canonical_palette"]["mode_count"] == 16
    assert palette["empirical_canonical_palette"]["frame0_mode_count"] == 15
    assert palette["empirical_canonical_palette"]["frame1_mode_count"] == 0
    assert palette["empirical_canonical_palette"]["zero_frame1_modes"] is True


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
    custody_values = {
        item["key"] for item in first["execution_gate"]["local_mlx_custody_values"]
    }
    assert "segnet_class_region_mask_ids" in custody_values
    action = first["multiscale_action_row"]
    assert action["schema"] == REPAIR_CAMPAIGN_MULTISCALE_ACTION_ROW_SCHEMA
    assert action["active_scales"] == [
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
    assert action["component_axes"] == [
        "segnet",
        "posenet",
        "rate_bytes",
        "selector_bits",
    ]
    assert action["action_functional"]["bit_delta_vs_baseline"] == -32.0
    assert action["entropy_position_class"] == "pre_entropy_distribution_shaping"
    assert action["remeasure_required_before_budget_spend"] is True
    assert report["multiscale_action_ledger"]["schema"] == (
        REPAIR_CAMPAIGN_MULTISCALE_ACTION_LEDGER_SCHEMA
    )
    assert report["multiscale_action_ledger"]["row_count"] == 2
    assert report["multiscale_action_ledger"]["scale_histogram"]["region"] == 2
    assert report["multiscale_action_ledger"]["scale_histogram"]["bit"] == 1
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
    assert allocation["multiscale_action_row"]["active_scales"] == (
        action["active_scales"]
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


def test_score_repair_campaign_blocks_missing_family_required_value_artifacts(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    first = work_order["typed_response_ledger"]["rows"][0]  # type: ignore[index]
    del first["segnet_class_region_mask_ids"]

    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)

    row = next(item for item in report["rows"] if item["typed_response_id"] == "segnet_region_ready")
    assert row["execution_gate"]["recommended_queue_status"] == "blocked_missing_artifact"
    assert "segnet_class_region_mask_ids:missing_or_empty" in (
        row["execution_gate"]["missing_artifacts"]
    )
    assert report["optimizer_decision"]["selected_allocation_count"] == 0
    blocked = next(
        item
        for item in report["optimizer_decision"]["blocked_allocation_rows"]
        if item["typed_response_id"] == "segnet_region_ready"
    )
    assert "local_mlx_advisory_custody_missing" in blocked["blockers"]
    assert "segnet_class_region_mask_ids:missing_or_empty" in blocked["missing_artifacts"]


def test_score_repair_campaign_blocks_symlinked_local_mlx_custody(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    first = work_order["typed_response_ledger"]["rows"][0]  # type: ignore[index]
    local_mlx_path = Path(first["local_mlx_response_path"])
    local_mlx_link = tmp_path / "segnet_mlx_response_link.json"
    local_mlx_link.symlink_to(local_mlx_path)
    first["local_mlx_response_path"] = str(local_mlx_link)

    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)

    row = next(item for item in report["rows"] if item["typed_response_id"] == "segnet_region_ready")
    assert row["execution_gate"]["recommended_queue_status"] == "blocked_missing_artifact"
    assert "local_mlx_response_path:path_is_symlink" in (
        row["execution_gate"]["missing_artifacts"]
    )
    assert report["optimizer_decision"]["selected_allocation_count"] == 0


def test_score_repair_campaign_blocks_truthy_authority_in_local_mlx_custody(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    first = work_order["typed_response_ledger"]["rows"][0]  # type: ignore[index]
    local_mlx_path = Path(first["local_mlx_response_path"])
    local_mlx_path.write_text(
        json.dumps(
            {
                "schema": "mlx_scorer_response.v1",
                "score_claim": True,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )

    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)

    row = next(item for item in report["rows"] if item["typed_response_id"] == "segnet_region_ready")
    assert row["execution_gate"]["recommended_queue_status"] == "blocked_missing_artifact"
    assert any(
        str(item).startswith("local_mlx_response_path:false_authority_violation:")
        for item in row["execution_gate"]["missing_artifacts"]
    )
    assert report["optimizer_decision"]["selected_allocation_count"] == 0


def test_score_repair_campaign_revalidates_existing_runtime_proof_artifact(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    first = work_order["typed_response_ledger"]["rows"][0]  # type: ignore[index]
    archive_path = tmp_path / "candidate_archive.zip"
    archive_path.write_bytes(b"candidate archive bytes")
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    proof_path = tmp_path / "runtime_consumption_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": RUNTIME_CONSUMPTION_PROOF_SCHEMA,
                "proof_kind": "unit_repair_campaign_runtime_proof",
                "candidate_archive_sha256": archive_sha,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "runtime_consumption_probe": {
                    "schema": "unit_repair_campaign_runtime_probe.v1",
                    "passed": True,
                },
                **_false_authority(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    first["candidate_archive_path"] = str(archive_path)
    first["candidate_archive_sha256"] = archive_sha
    first["runtime_consumption_proof_path"] = str(proof_path)

    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)

    row = report["rows"][0]
    proof_status = row["receiver_proof_status"]
    assert proof_status["runtime_consumption_proof"]["exists"] is True
    assert proof_status["runtime_consumption_proof_validation"][
        "receiver_contract_satisfied"
    ] is True
    assert (
        "runtime_consumption_proof_path:missing_or_unverified"
        not in proof_status["missing_artifacts"]
    )
    allocation = report["optimizer_decision"]["selected_allocation_rows"][0]
    assert (
        "runtime_consumption_proof_path:missing_or_unverified"
        not in allocation["receiver_proof_status"]["missing_artifacts"]
    )


def test_score_repair_campaign_rejects_path_only_runtime_proof_artifact(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    first = work_order["typed_response_ledger"]["rows"][0]  # type: ignore[index]
    archive_path = tmp_path / "candidate_archive.zip"
    archive_path.write_bytes(b"candidate archive bytes")
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    proof_path = tmp_path / "runtime_consumption_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": RUNTIME_CONSUMPTION_PROOF_SCHEMA,
                "candidate_archive_sha256": archive_sha,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "score_claim": True,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    first["candidate_archive_path"] = str(archive_path)
    first["candidate_archive_sha256"] = archive_sha
    first["runtime_consumption_proof_path"] = str(proof_path)

    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)

    row = report["rows"][0]
    validation = row["receiver_proof_status"]["runtime_consumption_proof_validation"]
    assert validation["proof_exists"] is True
    assert validation["receiver_contract_satisfied"] is False
    assert any(
        str(blocker).startswith("runtime_consumption_proof_false_authority:")
        for blocker in validation["blockers"]
    )
    assert "runtime_consumption_proof_path:missing_or_unverified" in (
        row["receiver_proof_status"]["missing_artifacts"]
    )


def test_score_repair_campaign_rejects_symlink_candidate_archive_custody(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    first = work_order["typed_response_ledger"]["rows"][0]  # type: ignore[index]
    archive_path = tmp_path / "candidate_archive.zip"
    archive_path.write_bytes(b"candidate archive bytes")
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    linked_archive_path = tmp_path / "candidate_archive_link.zip"
    linked_archive_path.symlink_to(archive_path)
    proof_path = tmp_path / "runtime_consumption_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": RUNTIME_CONSUMPTION_PROOF_SCHEMA,
                "proof_kind": "unit_repair_campaign_runtime_proof",
                "candidate_archive_sha256": archive_sha,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "runtime_consumption_probe": {
                    "schema": "unit_repair_campaign_runtime_probe.v1",
                    "passed": True,
                },
                **_false_authority(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    first["candidate_archive_path"] = str(linked_archive_path)
    first["candidate_archive_sha256"] = archive_sha
    first["runtime_consumption_proof_path"] = str(proof_path)

    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)

    row = report["rows"][0]
    archive_status = row["receiver_proof_status"]["receiver_consumed_candidate_archive"]
    assert archive_status["exists"] is True
    assert archive_status["is_symlink"] is True
    assert (
        "candidate_archive_path:path_is_symlink"
        in row["receiver_proof_status"]["missing_artifacts"]
    )
    assert (
        row["receiver_proof_status"]["runtime_consumption_proof_validation"][
            "receiver_contract_satisfied"
        ]
        is True
    )


def test_score_repair_campaign_uses_palette_frame_asymmetry_context(
    tmp_path: Path,
) -> None:
    report = score_repair_campaign(
        payload=_palette_work_order(tmp_path),
        repo_root=tmp_path,
    )

    assert report["ready_for_local_mlx_advisory_execution_count"] == 1
    row = report["rows"][0]
    assert row["typed_response_id"] == "palette_frame0_k16_ready"
    assert row["family_id"] == "palette_frame_asymmetry_prior"
    context = row["palette_dynamics_context"]
    assert context["mode_count"] == 16
    assert context["identity_mode_count"] == 1
    assert context["frame0_mode_count"] == 15
    assert context["frame1_mode_count"] == 0
    assert context["frame0_non_identity_fraction"] == 1.0
    assert context["zero_frame1_modes"] is True
    assert context["dominant_dynamics_interpretation"] == (
        "frame0_global_color_geometry_calibration_prior"
    )
    assert row["palette_frame_asymmetry_multiplier"] > 1.0
    assert "frame0_palette_repairs_are_global_interaction_terms" in (
        row["hard_legal_runtime_constraints"]
    )
    assert "frame1_palette_repairs_require_counterfactual_probe" in (
        row["hard_legal_runtime_constraints"]
    )
    custody_keys = {
        item["key"] for item in row["execution_gate"]["local_mlx_custody_paths"]
    }
    assert {
        "local_mlx_response_path",
        "reference_local_mlx_response_path",
        "repair_dynamics_palette_probe_matrix_path",
    }.issubset(custody_keys)
    action = row["multiscale_action_row"]
    assert action["palette_dynamics_context"]["mode_count"] == 16
    assert action["action_functional"]["palette_frame_asymmetry_multiplier"] == (
        row["palette_frame_asymmetry_multiplier"]
    )
    assert action["active_scales"] == [
        "byte",
        "pixel",
        "boundary",
        "region",
        "frame",
        "pair",
        "batch",
    ]
    allocation = report["optimizer_decision"]["selected_allocation_rows"][0]
    assert allocation["palette_dynamics_context"]["zero_frame1_modes"] is True
    assert allocation["palette_frame_asymmetry_multiplier"] == (
        row["palette_frame_asymmetry_multiplier"]
    )


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
                json.dumps(
                    {
                        "schema": "repair_campaign_stackability_posterior_row.v1",
                        "typed_response_id": (
                            "activation_plan:repair_queue:repair_waterfill_chain"
                        ),
                        "candidate_id": "repair_waterfill_chain",
                        "family_id": "segnet-posenet-waterfill",
                        "evidence_grade": "blocked_queue_activation_plan_only",
                        "acquisition_policy_delta": {
                            "recommended_acquisition_policy": (
                                "increase_priority_for_targeted_component_response_harvest"
                            ),
                            "family_priority_direction": "hold",
                            **_false_authority(),
                        },
                        "planner_feature_vector": {
                            "missing_artifact_count": 3,
                            "blocker_count": 5,
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
    assert summary["posterior_row_count"] == 3
    assert summary["family_prior_count"] == 3
    assert summary["acquisition_followup_route_count"] == 3
    followup = summary["acquisition_followup_routes"][0]
    assert followup["schema"] == REPAIR_CAMPAIGN_POSTERIOR_ACQUISITION_FOLLOWUP_SCHEMA
    assert followup["recommended_acquisition_policy"] == (
        "increase_priority_for_targeted_component_response_harvest"
    )
    assert followup["activation_action"] == "harvest_targeted_component_response_rows"
    assert followup["queue_artifact_key"] == (
        "targeted_component_correction_response_harvest"
    )
    assert followup["required_evidence_surface"] == (
        "targeted_component_correction_response_harvest"
    )
    assert followup["family_ids"] == ["segnet-posenet-waterfill"]
    assert followup["typed_response_ids"] == [
        "activation_plan:repair_queue:repair_waterfill_chain"
    ]
    assert followup["missing_artifact_total"] == 3
    assert followup["blocker_total"] == 5
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
    ] == 3
    public_summary = build_repair_campaign_posterior_prior_summary(
        posterior_path=posterior_path,
    )
    assert public_summary["posterior_row_count"] == 3
    assert public_summary["acquisition_followup_routes"][0][
        "recommended_acquisition_policy"
    ] == "increase_priority_for_targeted_component_response_harvest"


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
    assert ready["multiscale_action_row"]["active_scales"][:3] == [
        "bit",
        "byte",
        "pixel",
    ]
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
