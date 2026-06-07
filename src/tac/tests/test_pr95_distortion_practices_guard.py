# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.analysis.nerv_pair_local_distortion_servo import (
    PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA,
    PR95_SERVO_CURRICULUM_STAGES,
)
from tac.analysis.pr95_distortion_practices_guard import (
    AXIS_TRACE_CONTRACT_SCHEMA,
    PACT_NERV_RECEIVER_COMPILER_DAG_SCHEMA,
    PAYLOAD_GUARD_SCHEMA,
    POSE_MARGINAL_TELEMETRY_CONTRACT_SCHEMA,
    PRACTICE_DAG_SCHEMA,
    PRACTICES,
    SCHEMA,
    SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA,
    SCORER_ATOM_ACTUATOR_EXECUTION_EVIDENCE_SCHEMA,
    SOURCE_INVENTORY_SCHEMA,
    STAGE_DAG_SCHEMA,
    TELEMETRY_CONTRACT_SCHEMA,
    build_pact_nerv_receiver_compiler_dag,
    build_pr95_distortion_axis_trace_contract,
    build_pr95_distortion_practices_payload_guard,
    build_pr95_distortion_practices_row_guard,
    build_pr95_distortion_source_inventory,
    build_pr95_evaluate_scorer_domain_telemetry_contract,
    build_pr95_posenet_marginal_telemetry_contract,
    build_pr95_scorer_atom_actuator_contract,
)
from tac.tests.snerv_source_forward_fixtures import valid_snerv_source_forward_action_effect

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pr95_distortion_source_inventory_is_source_derived() -> None:
    inventory = build_pr95_distortion_source_inventory(REPO_ROOT)

    assert inventory["schema"] == SOURCE_INVENTORY_SCHEMA
    assert inventory["source_ready"] is True
    assert inventory["blockers"] == []
    check_ids = {row["check_id"] for row in inventory["source_records"]}
    assert "upstream_frame_utils_seq_len_2" in check_ids
    assert "upstream_posenet_uses_yuv6_pair" in check_ids
    assert "pr95_training_eval_roundtrip_ste" in check_ids
    assert "pr95_eight_stage_curriculum_present" in check_ids
    assert len(inventory["practice_source_rows"]) == len(PRACTICES)


def test_pr95_distortion_guard_accepts_hinerv_pr95_curriculum_row() -> None:
    guard = build_pr95_distortion_practices_row_guard(
        _hinerv_row(),
        repo_root=REPO_ROOT,
    )

    assert guard["schema"] == SCHEMA
    assert guard["family"] == "hi_nerv"
    assert guard["launch_allowed"] is True
    assert guard["blockers"] == []
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    assert rows["scorer_preprocess_eval_roundtrip_yuv6"]["observed"] is True
    assert rows["dual_component_real_scorer_pressure"]["observed"] is True
    assert rows["official_evaluate_archive_byte_price"]["observed"] is True
    assert rows["scorer_domain_telemetry_contract"]["observed"] is True
    assert rows["posenet_marginal_vjp_telemetry_contract"]["observed"] is True
    assert rows["family_local_scorer_atom_actuator_contract"]["observed"] is True
    assert rows["pr95_staged_qat_coder_curriculum"]["observed"] is True
    assert rows["archive_parseback_distortion_axis_trace"]["observed"] is True
    assert guard["practice_dag"]["schema"] == PRACTICE_DAG_SCHEMA
    assert guard["practice_dag"]["all_nodes_green"] is True
    assert guard["dag_blockers"] == []
    assert guard["optimization_stage_dag"]["schema"] == STAGE_DAG_SCHEMA
    assert guard["optimization_stage_dag"][
        "all_required_stage_signals_observed"
    ] is True
    compiler_dag = guard["receiver_compiler_dag"]
    assert compiler_dag["schema"] == PACT_NERV_RECEIVER_COMPILER_DAG_SCHEMA
    assert compiler_dag["pre_long_run_ready"] is True
    assert compiler_dag["promotion_compiler_ready"] is False
    assert compiler_dag["policy"]["primary_problem"] == (
        "minimum_description_length_under_frozen_receivers"
    )
    compiler_nodes = {node["node_id"]: node for node in compiler_dag["nodes"]}
    assert compiler_nodes["exact_evaluator_atom_oracle"]["green"] is True
    assert compiler_nodes["receiver_surface_integer_search"]["green"] is True
    assert compiler_nodes["sufficient_statistic_oracle_baselines"]["green"] is False
    assert compiler_nodes["byte_compiler_value_per_byte"]["green"] is False
    assert (
        "hinerv_section_output_delta_per_byte_rows"
        in compiler_nodes["receiver_surface_integer_search"]["observed_evidence"]
    )
    assert compiler_nodes["byte_compiler_value_per_byte"][
        "required_before_promotion"
    ] is True
    assert guard["score_claim"] is False
    assert guard["ready_for_exact_eval_dispatch"] is False


def test_pact_nerv_receiver_compiler_dag_preserves_partner_oracle_ladder() -> None:
    dag = build_pact_nerv_receiver_compiler_dag(_hinerv_row(), family="hi_nerv")

    assert dag["schema"] == PACT_NERV_RECEIVER_COMPILER_DAG_SCHEMA
    assert dag["pre_long_run_ready"] is True
    nodes = {node["node_id"]: node for node in dag["nodes"]}
    assert nodes["exact_evaluator_atom_oracle"]["required_before_long_run"] is True
    assert nodes["receiver_surface_integer_search"]["required_before_long_run"] is True
    assert nodes["sufficient_statistic_oracle_baselines"]["title"] == (
        "mask-first and pose-trajectory sufficient-statistic oracles"
    )
    assert nodes["sufficient_statistic_oracle_baselines"]["depends_on"] == [
        "seg_only_mask_witness_oracle",
        "pose_only_yuv6_witness_oracle",
    ]
    assert nodes["witness_family_pareto_frontier"]["depends_on"] == [
        "sufficient_statistic_oracle_baselines",
        "receiver_surface_integer_search",
    ]
    assert nodes["scorer_equivalence_witness_search"]["depends_on"] == [
        "witness_family_pareto_frontier",
        "cell_volume_compressibility_estimator",
        "receiver_surface_integer_search",
    ]
    assert "HNeRV pair latent" in nodes["family_backend_residualization"][
        "math_surface"
    ]
    assert nodes["multi_authority_replay"]["required_before_promotion"] is True
    assert dag["promotion_compiler_ready"] is False


def test_receiver_compiler_byte_value_requires_score_delta_ledger() -> None:
    row = _hinerv_row()
    row.update(
        {
            "pr95_distortion_axis_trace_contract": {"required_axes": []},
            "sufficient_statistic_oracle_baselines_ready": True,
            "seg_only_mask_witness_oracle_ready": True,
            "pose_only_yuv6_witness_oracle_ready": True,
            "witness_family_pareto_ready": True,
            "cell_volume_compressibility_estimate_ready": True,
            "scorer_equivalence_witness_search_ready": True,
            "dual_certificate_ready": True,
            "legal_code_data_boundary_contract_ready": True,
            "family_backend_residualization_ready": True,
            "section_value_per_byte_rows": [
                {
                    "section": "pair_local_latents_fine",
                    "score_value_per_byte": 0.0004,
                    "delta_total_score": -0.0008,
                    "delta_seg": -2.0e-6,
                    "delta_pose": -1.0e-7,
                    "delta_archive_bytes": 2,
                }
            ],
        }
    )

    dag = build_pact_nerv_receiver_compiler_dag(row, family="hi_nerv")
    nodes = {node["node_id"]: node for node in dag["nodes"]}

    assert nodes["byte_compiler_value_per_byte"]["green"] is True
    assert "score_value_per_byte:pair_local_latents_fine" in nodes[
        "byte_compiler_value_per_byte"
    ]["observed_evidence"]
    assert nodes["multi_authority_replay"]["green"] is False
    assert dag["promotion_compiler_ready"] is False


def test_pr95_distortion_guard_rejects_metadata_only_actuator_contract() -> None:
    row = _hinerv_row()
    del row["pr95_scorer_atom_actuator_execution_evidence"]

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "hi_nerv_pr95_distortion_family_local_scorer_atom_actuator_contract_missing"
        in guard["blockers"]
    )
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    actuator_row = rows["family_local_scorer_atom_actuator_contract"]
    assert actuator_row["observed"] is False
    assert "actuator_execution_evidence_missing" in actuator_row["observed_evidence"]


def test_pr95_distortion_guard_rejects_global_hinerv_actuator_evidence() -> None:
    row = _hinerv_row()
    evidence = dict(row["pr95_scorer_atom_actuator_execution_evidence"])
    evidence["updated_tensor_names"] = ["latents_fine", "feature_grids.0"]
    evidence["state_mutation_scope"] = "late_decoder_global_update"
    evidence["pair_locality_verified"] = False
    evidence["non_target_pair_output_delta_l2_max"] = 0.001
    row["pr95_scorer_atom_actuator_execution_evidence"] = evidence

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "hi_nerv_pr95_distortion_family_local_scorer_atom_actuator_contract_missing"
        in guard["blockers"]
    )
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    actuator_row = rows["family_local_scorer_atom_actuator_contract"]
    assert actuator_row["observed"] is False
    assert "hinerv_pair_local_latents_fine_row_only_update" not in actuator_row[
        "observed_evidence"
    ]
    assert "hinerv_pair_local_non_target_delta_zero" not in actuator_row[
        "observed_evidence"
    ]


def test_pr95_distortion_guard_rejects_subquantum_hinerv_actuator_evidence() -> None:
    row = _hinerv_row()
    evidence = dict(row["pr95_scorer_atom_actuator_execution_evidence"])
    evidence["pair_local_output_delta_max_abs"] = 1.0e-6
    evidence["pair_local_output_delta_max_abs_uint8"] = 2.55e-4
    evidence["receiver_uint8_crossing_potential"] = False
    row["pr95_scorer_atom_actuator_execution_evidence"] = evidence

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "hi_nerv_pr95_distortion_family_local_scorer_atom_actuator_contract_missing"
        in guard["blockers"]
    )
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    actuator_row = rows["family_local_scorer_atom_actuator_contract"]
    assert actuator_row["observed"] is False
    assert "hinerv_pair_local_receiver_uint8_crossing_potential" not in actuator_row[
        "observed_evidence"
    ]


def test_pr95_distortion_guard_rejects_boolean_only_snerv_source_forward_evidence() -> None:
    row = _snerv_row()
    evidence = dict(row["pr95_scorer_atom_actuator_execution_evidence"])
    evidence.pop("source_forward_replay_proof")
    row["pr95_scorer_atom_actuator_execution_evidence"] = evidence

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "snerv_pr95_distortion_family_local_scorer_atom_actuator_contract_missing"
        in guard["blockers"]
    )
    compiler_nodes = {
        node["node_id"]: node for node in guard["receiver_compiler_dag"]["nodes"]
    }
    assert compiler_nodes["receiver_surface_integer_search"]["green"] is False
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    actuator_row = rows["family_local_scorer_atom_actuator_contract"]
    assert "snerv_complete_numerical_source_forward_proof_present" not in (
        actuator_row["observed_evidence"]
    )


def test_pr95_distortion_guard_rejects_hash_only_snerv_source_forward_metadata() -> None:
    row = _snerv_row()
    evidence = dict(row["pr95_scorer_atom_actuator_execution_evidence"])
    evidence["source_forward_replay_proof"] = _legacy_snerv_source_forward_metadata()
    row["pr95_scorer_atom_actuator_execution_evidence"] = evidence

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "snerv_pr95_distortion_family_local_scorer_atom_actuator_contract_missing"
        in guard["blockers"]
    )
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    actuator_row = rows["family_local_scorer_atom_actuator_contract"]
    assert "snerv_legacy_source_forward_metadata_rejected" in actuator_row[
        "observed_evidence"
    ]
    assert "snerv_complete_numerical_source_forward_proof_present" not in (
        actuator_row["observed_evidence"]
    )


def test_pr95_distortion_guard_rejects_snerv_without_official_authority_gate() -> None:
    row = _snerv_row()
    evidence = dict(row["pr95_scorer_atom_actuator_execution_evidence"])
    evidence.pop("snerv_official_tub_lf_hf_decoder_replacement_authority_gate")
    row["pr95_scorer_atom_actuator_execution_evidence"] = evidence

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "snerv_pr95_distortion_family_local_scorer_atom_actuator_contract_missing"
        in guard["blockers"]
    )
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    actuator_row = rows["family_local_scorer_atom_actuator_contract"]
    assert "snerv_official_replacement_authority_gate_missing" in actuator_row[
        "observed_evidence"
    ]


def test_pr95_distortion_guard_rejects_inline_only_hinerv_actuator_evidence() -> None:
    row = _hinerv_row()
    evidence = dict(row["pr95_scorer_atom_actuator_execution_evidence"])
    evidence.pop("pair_local_smoke_artifact_schema")
    evidence.pop("pair_local_smoke_artifact_path")
    evidence.pop("pair_local_smoke_artifact_sha256")
    evidence.pop("pair_local_smoke_artifact_bytes")
    row["pr95_scorer_atom_actuator_execution_evidence"] = evidence

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "hi_nerv_pr95_distortion_family_local_scorer_atom_actuator_contract_missing"
        in guard["blockers"]
    )
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    actuator_row = rows["family_local_scorer_atom_actuator_contract"]
    assert actuator_row["observed"] is False
    assert "hinerv_pair_local_smoke_artifact_path_sha256_bytes" not in (
        actuator_row["observed_evidence"]
    )


def test_pr95_distortion_guard_requires_pr95_grade_pair_local_servo_receipt() -> None:
    row = _hinerv_row()
    evidence = dict(row["pr95_scorer_atom_actuator_execution_evidence"])
    evidence.pop("pair_local_distortion_servo_receipt")
    row["pr95_scorer_atom_actuator_execution_evidence"] = evidence

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "hi_nerv_pr95_distortion_family_local_scorer_atom_actuator_contract_missing"
        in guard["blockers"]
    )
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    actuator_row = rows["family_local_scorer_atom_actuator_contract"]
    assert actuator_row["observed"] is False
    assert "pair_local_distortion_servo_receipt_missing" in actuator_row[
        "observed_evidence"
    ]


def test_pr95_distortion_guard_rejects_servo_receipt_that_loses_parseback() -> None:
    row = _hinerv_row()
    evidence = dict(row["pr95_scorer_atom_actuator_execution_evidence"])
    receipt = dict(evidence["pair_local_distortion_servo_receipt"])
    receipt["parseback_argmax_flipped_pixels"] = 0
    receipt["parseback_segnet_margin_delta"] = 0.0
    evidence["pair_local_distortion_servo_receipt"] = receipt
    row["pr95_scorer_atom_actuator_execution_evidence"] = evidence

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    actuator_row = rows["family_local_scorer_atom_actuator_contract"]
    assert "pair_local_servo_archive_parseback_survival_missing" in (
        actuator_row["observed_evidence"]
    )


def test_pr95_distortion_guard_blocks_snerv_without_eval_roundtrip() -> None:
    row = _snerv_row()
    command = row["command"]
    command.remove("--snerv-score-aware-long-training-eval-roundtrip-ste")
    command.append("--no-snerv-score-aware-long-training-eval-roundtrip-ste")

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert "snerv_pr95_distortion_scorer_preprocess_eval_roundtrip_yuv6_missing" in guard["blockers"]
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    assert rows["scorer_preprocess_eval_roundtrip_yuv6"]["observed"] is False
    dag_nodes = {row["practice_id"]: row for row in guard["practice_dag"]["nodes"]}
    assert dag_nodes["scorer_preprocess_eval_roundtrip_yuv6"]["status"] == "missing"
    assert dag_nodes["dual_component_real_scorer_pressure"]["status"] == (
        "blocked_by_prerequisite"
    )
    assert dag_nodes["archive_parseback_distortion_axis_trace"]["status"] == (
        "blocked_by_prerequisite"
    )
    assert guard["practice_dag"]["first_failed_practice_ids"] == [
        "scorer_preprocess_eval_roundtrip_yuv6"
    ]


def test_pr95_distortion_guard_accepts_current_snerv_pr95_defaults() -> None:
    row = _snerv_row()
    command = row["command"]
    command.remove("--snerv-score-aware-long-training-pr95-faithful-curriculum")
    command.remove("--snerv-score-aware-long-training-eval-roundtrip-ste")

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is True
    assert guard["blockers"] == []
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    assert rows["scorer_preprocess_eval_roundtrip_yuv6"]["observed"] is True
    assert rows["pr95_staged_qat_coder_curriculum"]["observed"] is True
    assert guard["optimization_stage_dag"]["observed_signals"][
        "pr95_curriculum"
    ] is True


def test_pr95_distortion_guard_blocks_every_stage_muon_as_pr95_unfaithful() -> None:
    row = _snerv_row()
    row["command"].extend(
        [
            "--snerv-score-aware-long-training-pr95-muon-policy",
            "every_stage",
        ]
    )

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "snerv_pr95_stage_dag_stage8_muon_finetune_missing_muon_stage8_only"
        in guard["blockers"]
    )
    assert guard["optimization_stage_dag"]["observed_signals"][
        "muon_stage8_only"
    ] is False


def test_pr95_distortion_guard_blocks_fake_parity_without_byte_binding() -> None:
    row = _hinerv_row()
    del row["upstream_evaluate_score_binding"]

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert "hi_nerv_pr95_distortion_official_evaluate_archive_byte_price_missing" in guard["blockers"]
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    assert rows["official_evaluate_archive_byte_price"]["observed"] is False


def test_pr95_distortion_guard_blocks_fake_parity_without_scorer_telemetry_contract() -> None:
    row = _hinerv_row()
    del row["pr95_evaluate_scorer_domain_telemetry_contract"]

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert "hi_nerv_pr95_distortion_scorer_domain_telemetry_contract_missing" in guard["blockers"]
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    assert rows["scorer_domain_telemetry_contract"]["observed"] is False


def test_pr95_distortion_guard_blocks_axis_contract_without_measured_replay() -> None:
    row = _hinerv_row()
    del row["pr95_distortion_axis_trace_measurements"]

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "hi_nerv_pr95_distortion_archive_parseback_distortion_axis_trace_missing"
        in guard["blockers"]
    )
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    axis_row = rows["archive_parseback_distortion_axis_trace"]
    assert axis_row["observed"] is False
    assert not any(
        str(item).startswith("measured_axes=")
        for item in axis_row["observed_evidence"]
    )


def test_pr95_distortion_axis_trace_contract_names_parseback_chain() -> None:
    contract = build_pr95_distortion_axis_trace_contract("hi_nerv")

    assert contract["schema"] == AXIS_TRACE_CONTRACT_SCHEMA
    assert contract["axis_order_is_dependency_order"] is True
    assert contract["required_axes"] == [
        "live_forward",
        "fakequant_forward",
        "archive_parseback",
        "inflate_replay",
        "official_evaluate_py",
    ]
    assert contract["acceptance_policy"]["live_only_improvement_is_false_authority"] is True
    assert contract["acceptance_policy"]["fail_closed_on_axis_divergence"] is True
    assert {row["stage"] for row in contract["stage_gates"]} == {
        "class_birth",
        "margin_crossing",
            "argmax_disagreement",
            "fakequant_survival",
            "archive_parseback_survival",
            "pose_marginal_vjp",
            "late_byte_and_optimizer_pressure",
        }
    assert contract["score_claim"] is False


def test_pr95_posenet_marginal_contract_names_frontier_derivative() -> None:
    contract = build_pr95_posenet_marginal_telemetry_contract("snerv")

    assert contract["schema"] == POSE_MARGINAL_TELEMETRY_CONTRACT_SCHEMA
    assert contract["pose_marginal_formula"] == "5/sqrt(10*d_pose)"
    assert contract["pose_marginal_increases_as_d_pose_decreases"] is True
    assert "pose_direct_live_score_marginal_wrt_raw_mse" in contract[
        "required_telemetry"
    ]
    assert "pose_direct_live_vjp_norm_by_group" in contract["required_telemetry"]
    assert contract["acceptance_policy"][
        "long_run_admission_requires_pose_marginal_telemetry"
    ] is True
    assert contract["score_claim"] is False


def test_pr95_family_actuator_contract_splits_hinerv_and_snerv() -> None:
    hi = build_pr95_scorer_atom_actuator_contract("hi_nerv")
    snerv = build_pr95_scorer_atom_actuator_contract("snerv")

    assert hi["schema"] == SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA
    assert snerv["schema"] == SCORER_ATOM_ACTUATOR_CONTRACT_SCHEMA
    assert "pair_local_film_or_latent_adapter" in hi["family_actuators"]
    assert "official_mfu_hfr_tub_source_forward_parity" in snerv[
        "family_actuators"
    ]
    assert "tub_output2_segnet_last_frame_binding" in snerv["family_actuators"]
    assert hi["family_actuators"] != snerv["family_actuators"]
    assert hi["acceptance_policy"]["cross_family_evidence_rejected"] is True
    assert hi["acceptance_policy"][
        "execution_evidence_required_before_long_run"
    ] is True
    assert "hinerv_pair_local_actuator_smoke.v1" in hi[
        "required_execution_evidence"
    ]
    assert "pair_local_smoke_artifact_path_sha256_bytes" in hi[
        "required_execution_evidence"
    ]
    assert "receiver_uint8_changed_count" in hi["required_execution_evidence"]
    assert "section_output_delta_per_byte_rows" in hi[
        "required_execution_evidence"
    ]
    assert "nerv_pair_local_distortion_servo_receipt.v1" in hi[
        "required_execution_evidence"
    ]
    assert "exact_pair_local_score_delta" in hi["required_execution_evidence"]
    assert hi["acceptance_policy"][
        "pr95_grade_pair_local_distortion_servo_receipt_required"
    ] is True
    assert hi["acceptance_policy"][
        "servo_must_survive_uint8_preprocess_fakequant_parseback"
    ] is True
    assert "section_value_per_byte_rows" not in hi["required_execution_evidence"]
    assert hi["acceptance_policy"][
        "output_delta_per_byte_is_not_score_value_per_byte"
    ] is True
    assert "snerv_official_source_forward_state_artifact.v1" in snerv[
        "required_execution_evidence"
    ]
    assert "snerv_official_tub_lf_hf_decoder_replacement_authority_gate.v1" in snerv[
        "required_execution_evidence"
    ]
    assert "nerv_pair_local_distortion_servo_receipt.v1" in snerv[
        "required_execution_evidence"
    ]
    assert snerv["score_claim"] is False


def test_pr95_distortion_telemetry_contract_names_evaluate_domains() -> None:
    contract = build_pr95_evaluate_scorer_domain_telemetry_contract("snerv")

    assert contract["schema"] == TELEMETRY_CONTRACT_SCHEMA
    assert contract["segnet_scored_frame_index"] == 1
    assert contract["posenet_scored_frame_indices"] == [0, 1]
    assert contract["argmax_occupancy_gate_required"] is True
    assert contract["fail_closed_on_missing_metrics"] is True
    assert any("snerv_segnet_last_frame_distill" in name for name in contract["segnet_last_frame_argmax_metric_names"])
    assert any("occupied_class_fraction" in name for name in contract["segnet_argmax_occupancy_metric_names"])
    assert any("posenet_yuv6_pair" in name for name in contract["posenet_yuv6_pair_metric_names"])
    assert contract["score_claim"] is False


def test_pr95_distortion_payload_guard_extracts_verdict_rows() -> None:
    payload = {"schema": "example", "selected_local_mlx_experiments": [_snerv_row()]}

    guard = build_pr95_distortion_practices_payload_guard(payload, repo_root=REPO_ROOT)

    assert guard["schema"] == PAYLOAD_GUARD_SCHEMA
    assert guard["candidate_row_count"] == 1
    assert guard["launch_allowed"] is True
    assert guard["blockers"] == []


def _base_command(family: str) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family",
        family,
        "--num-pairs",
        "600",
        "--epochs",
        "16",
        "--hard-byte-ceiling",
        "3980000",
        "--distillation-device",
        "gpu",
        "--segnet-distillation-weight",
        "1.0",
        "--pose-distillation-weight",
        "1.0",
        "--coder-aware-qat",
        "--coder-qat-c1a-entropy-weight",
        "0.0001",
        "--mlx-prefilter-scorer-device",
        "gpu",
        "--mlx-prefilter-scorer-batch-pairs",
        "8",
        "--output-dir",
        "/Volumes/VertigoDataTier/pact/test_pr95_guard",
    ]


def _hinerv_actuator_execution_evidence() -> dict:
    return {
        "schema": SCORER_ATOM_ACTUATOR_EXECUTION_EVIDENCE_SCHEMA,
        "family": "hi_nerv",
        "pair_local_smoke_schema": "hinerv_pair_local_actuator_smoke.v1",
        "pair_local_smoke_artifact_schema": (
            "hinerv_pair_local_actuator_smoke_artifact.v1"
        ),
        "pair_local_smoke_artifact_path": (
            "/Volumes/VertigoDataTier/pact/test_pr95_guard/"
            "hi_nerv_pair_local_actuator_smoke/"
            "hinerv_pair_local_actuator_smoke_pair000000_aaaaaaaaaaaa.json"
        ),
        "pair_local_smoke_artifact_sha256": "d" * 64,
        "pair_local_smoke_artifact_bytes": 2048,
        "actuator_kind": "pair_local_latent_row",
        "actuator_tensor_name": "latents_fine",
        "updated_tensor_names": ["latents_fine"],
        "state_mutation_scope": "latents_fine_row_only",
        "runtime_sidecar_bytes": 0,
        "pair_local_adapter_bytes": 128,
        "pair_local_adapter_sha256": "a" * 64,
        "pair_local_grad_norm": 0.25,
        "pair_local_grad_norm_by_group": {"latents_fine": 0.25},
        "pair_local_output_delta_l2": 0.031,
        "pair_local_output_delta_max_abs": 0.004,
        "pair_local_output_delta_max_abs_uint8": 1.02,
        "receiver_uint8_half_step_normalized": 0.5 / 255.0,
        "receiver_uint8_crossing_potential": True,
        "receiver_uint8_changed": True,
        "receiver_uint8_changed_count": 12,
        "receiver_uint8_changed_fraction": 0.001,
        "receiver_uint8_delta_abs_max": 2,
        "non_target_pair_receiver_uint8_changed_count": 0,
        "non_target_pair_receiver_uint8_delta_abs_max": 0,
        "pair_locality_verified": True,
        "non_target_pair_output_delta_l2_max": 0.0,
        "state_restored_after_smoke": True,
        "pair_local_latents_fine_original_row_sha256": "c" * 64,
        "pair_local_latents_fine_restored_row_sha256": "c" * 64,
        "section_output_delta_per_byte_rows": [
            {
                "section": "pair_local_latents_fine",
                "bytes": 128,
                "output_delta_l2_per_byte": 0.004,
                "value_semantics": "receiver_output_l2_per_byte_not_score_value",
                "score_value_per_byte_measured": False,
            }
        ],
        "section_value_per_byte_rows": [],
        "pair_local_distortion_servo_receipt": _pair_local_distortion_servo_receipt(
            "hi_nerv"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _snerv_actuator_execution_evidence() -> dict:
    return {
        "schema": SCORER_ATOM_ACTUATOR_EXECUTION_EVIDENCE_SCHEMA,
        "family": "snerv",
        "state_artifact_schema": "snerv_official_source_forward_state_artifact.v1",
        "official_state_dict_value_artifact_bytes": 512,
        "official_state_dict_value_artifact_sha256": "b" * 64,
        "checkpoint_export_lineage_bound": True,
        "mfu_hfr_tub_source_forward_parity_proven": True,
        "tub_output2_source_forward_parity_proven": True,
        "source_forward_replay_proof": valid_snerv_source_forward_action_effect(),
        "snerv_official_tub_lf_hf_decoder_replacement_authority_gate": (
            _snerv_official_replacement_authority_gate()
        ),
        "pair_local_distortion_servo_receipt": _pair_local_distortion_servo_receipt(
            "snerv"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _legacy_snerv_source_forward_metadata() -> dict:
    return {
        "official_torch_frame_hash": "1" * 64,
        "mlx_frame_hash": "2" * 64,
        "numpy_receiver_frame_hash": "3" * 64,
        "parseback_frame_hash": "4" * 64,
        "tub_output_2_hash": "5" * 64,
        "max_abs_frame_delta_official_mlx": 0.0,
        "max_abs_yuv6_delta_official_numpy": 0.0,
        "seg_logit_linf_official_parseback": 0.0,
        "pose_linf_official_parseback": 0.0,
        "mfu_tensor_hashes": {"mfu.upsample_mid.weight": "6" * 64},
        "hfr_tensor_hashes": {"hfr.lh.conv1.weight": "7" * 64},
    }


def _snerv_official_replacement_authority_gate() -> dict:
    return {
        "schema": "snerv_official_tub_lf_hf_decoder_replacement_authority_gate.v1",
        "generated_utc": "2026-06-05T00:00:00+00:00",
        "_source_path": "/Volumes/VertigoDataTier/pact/snerv_official_replacement_authority_gate.json",
        "_source_sha256": "8" * 64,
        "official_tub_lf_hf_decoder_replacement_ready": True,
        "official_checkpoint_export_binding_ready": True,
        "receiver_output2_frame_replay_ready": True,
        "tub_source_fixture_replay_ready": True,
        "trained_checkpoint_state_dict_mapping_ready": True,
        "tub_temporal_output2_weight_mapping_ready": True,
        "full_tub_source_forward_replay_ready": True,
        "closed_campaign_blockers": [
            "snerv_official_mfu_hfr_tub_export_not_bound",
            "snerv_official_mfu_hfr_tub_receiver_payload_not_bound",
            "snerv_official_mfu_hfr_tub_frame_producing_export_missing",
            "snerv_official_tub_output2_receiver_frame_decode_not_bound",
        ],
        "source_forward_authority_residual_blockers": [],
        "queue_blockers": [],
        "blockers": ["snerv_official_tub_lf_hf_decoder_replacement_false_authority"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _pair_local_distortion_servo_receipt(family: str) -> dict:
    if family == "snerv":
        actuation = {
            "pair_local": True,
            "trained_param_groups": ["tub.output_2", "mfu.adapter"],
            "source_forward_replay_bound": True,
            "mfu_hfr_tub_source_forward_parity_proven": True,
        }
        trained_groups = ["tub.output_2", "mfu.adapter"]
        grad = {"tub.output_2": 0.4, "mfu.adapter": 0.2}
        update = {"tub.output_2": 0.05, "mfu.adapter": 0.02}
        actuator_id = "snerv_tub_output2_pair_birth"
        actuator_kind = "pair_conditioned_mfu_hfr_tub_adapter"
    else:
        actuation = {
            "pair_local": True,
            "trained_param_groups": ["latents_fine", "output_head.rgb_1"],
        }
        trained_groups = ["latents_fine", "output_head.rgb_1"]
        grad = {"latents_fine": 0.25, "output_head.rgb_1": 0.13}
        update = {"latents_fine": 0.04, "output_head.rgb_1": 0.02}
        actuator_id = "hinerv_target_region_birth_pair17"
        actuator_kind = "pair_local_latent_row"
    return {
        "schema": PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA,
        "family": family,
        "pair_ids": [17],
        "stage": "archive_parseback_survival",
        "authority": "parseback_mlx",
        "source_archive_sha256": "1" * 64,
        "candidate_archive_sha256": "2" * 64,
        "payload_sha256": "3" * 64,
        "old_d_seg": 0.020,
        "new_d_seg": 0.019,
        "old_d_pose": 0.0020,
        "new_d_pose": 0.00195,
        "old_archive_bytes": 178_000,
        "new_archive_bytes": 178_128,
        "value_per_byte": 0.0007,
        "float_rgb_delta_linf": 0.007,
        "uint8_changed_pixels": 12,
        "segnet_input_delta_linf": 0.002,
        "posenet_input_delta_linf": 0.001,
        "segnet_margin_delta": 0.32,
        "segnet_argmax_flipped_pixels": 7,
        "pose_output_delta_l2": 0.015,
        "fakequant_segnet_margin_delta": 0.24,
        "fakequant_argmax_flipped_pixels": 5,
        "parseback_segnet_margin_delta": 0.21,
        "parseback_argmax_flipped_pixels": 4,
        "frame_scope": "frame1_seg_pose_joint",
        "actuator_id": actuator_id,
        "actuator_kind": actuator_kind,
        "worst_scorer_debt": {
            "target_id": "pair17_class4_region2",
            "score_debt_before": 2.4,
            "score_debt_after": 1.1,
        },
        "frame_incidence": {
            "frame0_pose_only": True,
            "frame0_posenet_incidence": True,
            "frame0_segnet_incidence": False,
            "frame1_segnet_incidence": True,
            "frame1_posenet_incidence": True,
            "frame0_frame1_control_split": True,
            "separate_frame_heads": True,
        },
        "stage_manifest": {
            "completed_stage_ids": list(PR95_SERVO_CURRICULUM_STAGES),
            "stage_order_respected": True,
            "byte_pressure_after_birth": True,
            "qat_after_round_ste": True,
            "final_optimizer_after_survival": True,
        },
        "actuation": actuation,
        "trained_param_groups": trained_groups,
        "grad_norm_by_group": grad,
        "update_norm_by_group": update,
        "action_algebra_trace": {
            "selected_action_id": "target_region_rgb_bias_then_pair_adapter",
            "frame_scope": "frame1_seg_pose_joint",
            "effect_delta_seg": -0.001,
            "effect_delta_pose": -0.00005,
            "effect_delta_bytes": 128,
            "runtime_delta_ms": 0.4,
            "selector_bits": 12,
            "noncommutative_interactions_checked": True,
        },
        "hardware_margin_trace": {
            "target_authority": "parseback_mlx",
            "target_authority_margin_checked": True,
            "hardware_drift_risk": "bounded",
            "segnet_margin_min": 0.03,
            "pose_error_slack": 0.0002,
        },
    }


def _axis_trace_measurements() -> list[dict[str, float | str | bool]]:
    return [
        {
            "axis": "live_forward",
            "measured": True,
            "score_delta": -0.01,
            "d_seg": 0.02,
            "d_pose": 0.002,
        },
        {
            "axis": "fakequant_forward",
            "measured": True,
            "score_delta": -0.009,
            "d_seg": 0.021,
            "d_pose": 0.0021,
        },
        {
            "axis": "archive_parseback",
            "measured": True,
            "score_delta": -0.008,
            "d_seg": 0.022,
            "d_pose": 0.0022,
        },
        {
            "axis": "inflate_replay",
            "measured": True,
            "score_delta": -0.007,
            "d_seg": 0.023,
            "d_pose": 0.0023,
        },
        {
            "axis": "official_evaluate_py",
            "measured": True,
            "score": 0.2,
            "archive_bytes": 178000,
        },
    ]


def _hinerv_row() -> dict:
    command = _base_command("hi_nerv")
    command.extend(
        [
            "--batch-pairs",
            "8",
            "--hi-nerv-optimizer-policy",
            "pr95_curriculum",
        ]
    )
    return {
        "id": "hi_row",
        "family": "hi_nerv",
        "command": command,
        "hard_byte_ceiling": 3_980_000,
        "upstream_evaluate_score_binding": _upstream_evaluate_score_binding("hi_nerv"),
        "pr95_evaluate_scorer_domain_telemetry_contract": (
            build_pr95_evaluate_scorer_domain_telemetry_contract("hi_nerv")
        ),
        "pr95_distortion_axis_trace_contract": (
            build_pr95_distortion_axis_trace_contract("hi_nerv")
        ),
        "pr95_distortion_axis_trace_measurements": _axis_trace_measurements(),
        "pr95_posenet_marginal_telemetry_contract": (
            build_pr95_posenet_marginal_telemetry_contract("hi_nerv")
        ),
        "pr95_scorer_atom_actuator_contract": (
            build_pr95_scorer_atom_actuator_contract("hi_nerv")
        ),
        "pr95_scorer_atom_actuator_execution_evidence": (
            _hinerv_actuator_execution_evidence()
        ),
        "score_lowering_gate": {
            "schema": "nerv_long_training_score_lowering_gate.v1",
            "local_mlx_executable": True,
        },
    }


def _snerv_row() -> dict:
    command = _base_command("snerv")
    command.extend(
        [
            "--snerv-score-aware-long-training-batch-pairs",
            "8",
            "--snerv-score-aware-long-training-eval-roundtrip-ste",
            "--snerv-score-aware-long-training-pr95-faithful-curriculum",
            "--snerv-score-aware-long-training-optimizer",
            "pact_muon_adamw",
        ]
    )
    return {
        "id": "snerv_row",
        "family": "snerv",
        "command": command,
        "hard_byte_ceiling": 3_980_000,
        "upstream_evaluate_score_binding": _upstream_evaluate_score_binding("snerv"),
        "pr95_evaluate_scorer_domain_telemetry_contract": (
            build_pr95_evaluate_scorer_domain_telemetry_contract("snerv")
        ),
        "pr95_distortion_axis_trace_contract": (
            build_pr95_distortion_axis_trace_contract("snerv")
        ),
        "pr95_distortion_axis_trace_measurements": _axis_trace_measurements(),
        "pr95_posenet_marginal_telemetry_contract": (
            build_pr95_posenet_marginal_telemetry_contract("snerv")
        ),
        "pr95_scorer_atom_actuator_contract": (
            build_pr95_scorer_atom_actuator_contract("snerv")
        ),
        "pr95_scorer_atom_actuator_execution_evidence": (
            _snerv_actuator_execution_evidence()
        ),
        "score_lowering_gate": {
            "schema": "nerv_long_training_score_lowering_gate.v1",
            "local_mlx_executable": True,
        },
    }


def _upstream_evaluate_score_binding(family: str) -> dict:
    return {
        "schema": "nerv_row_upstream_evaluate_binding.v1",
        "family": family,
        "rate": {
            "archive_authority": "submission_dir/archive.zip.stat().st_size",
            "canonical_denominator_bytes": 37_545_489,
            "rate_price_per_archive_byte": 25 / 37_545_489,
            "raw_output_shape_bytes_are_not_rate_denominator": (1200 * 874 * 1164 * 3),
        },
    }
