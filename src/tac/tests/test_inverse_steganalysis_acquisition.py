# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest

from tac.local_acceleration.mlx_acquisition_batch import (
    build_mlx_acquisition_batch_from_selection,
)
from tac.optimization.byte_shaving_campaign import (
    COUPLED_OPERATION_SET_SCHEMA,
    SIGNAL_SURFACE_SCHEMA,
    build_byte_shaving_campaign_plan,
)
from tac.optimization.inverse_steganalysis_acquisition import (
    ACTION_FUNCTIONAL_SCHEMA,
    ATOM_SCHEMA,
    CONTEST_RATE_DENOM_BYTES,
    CONTEST_RATE_SCORE_PER_BYTE,
    INVERSE_SCORER_SURFACE_SCHEMA,
    MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND,
    MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA,
    OBSERVATION_SCHEMA,
    SCHEMA,
    InverseSteganalysisAcquisitionError,
    action_atoms_from_byte_shaving_campaign_plan,
    action_atoms_from_byte_shaving_signal_surface,
    action_atoms_from_inverse_scorer_surface,
    action_atoms_from_mlx_acquisition_batch,
    action_surface_terms,
    build_discrete_scorer_action_functional,
    build_inverse_steganalysis_acquisition_plan,
    compute_acquisition_priority,
    inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection,
    normalize_inverse_steganalysis_atom,
    normalize_inverse_steganalysis_observation,
    observations_from_materializer_chain_manifest,
    observations_from_queue_observation,
    observations_from_queue_performance_summary,
    paired_exact_auth_calibration_observations_from_review_packets,
)
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.score_composition import CANONICAL_RATE_DENOM_BYTES


def _planning_false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _atom(candidate_id: str = "candidate_a", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "atom_id": f"atom_{candidate_id}",
        "candidate_id": candidate_id,
        "scale": "region_frequency",
        "scope_axis": "regions",
        "parent_unit_id": "pair_0007",
        "frame_range": [14, 16],
        "pair_indices": [7],
        "region_bbox": [16, 24, 96, 128],
        "frequency_band": "mid_high_dct",
        "byte_range": [1024, 1536],
        "component": "segnet",
        "coherence_group": "road_edges",
        "sparsity_prior": 0.8,
        "predicted_segnet_gain": 0.0003,
        "predicted_posenet_gain": 0.0001,
        "predicted_rate_gain": 0.00005,
        "predicted_rate_cost": 0.00001,
        "predicted_score_gain": 0.00044,
        "first_order_marginal_effect": 0.0004,
        "second_order_interaction_effect": 0.00004,
        "discontinuity_risk": 0.1,
        "discontinuity_threshold": 0.5,
        "uncertainty": 0.00004,
        "elapsed_seconds": 4.0,
        "artifact_bytes": 1_000_000,
        "resource_kind": "local_mlx",
    }
    row.update(overrides)
    return row


def _observation(candidate_id: str = "candidate_a", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "observation_id": f"obs_{candidate_id}",
        "candidate_id": candidate_id,
        "axis": "[macOS-MLX research-signal]",
        "runtime_identity": {
            "runtime_tree_sha256": "a" * 64,
            "scorer_version": "mlx_scorer_response.v1",
        },
        "cache_identity": {
            "cache_sha256": "b" * 64,
            "array_sha256": {"pair_indices": "c" * 64},
        },
        "observed_score_gain": 0.0005,
        "calibration_error": 0.00002,
        "elapsed_seconds": 8.0,
        "artifact_bytes": 2_000_000,
        "resource_kind": "local_mlx",
    }
    row.update(overrides)
    return row


def _review_packet(
    axis: str,
    *,
    score: float,
    baseline_score: float,
    archive_sha256: str = "1" * 64,
    archive_bytes: int = 181_232,
    runtime_content_tree_sha256: str = "2" * 64,
    inflated_output_aggregate_sha256: str = "3" * 64,
    **overrides: object,
) -> dict[str, object]:
    exact_cuda = axis == "contest_cuda"
    exact_cpu = axis == "contest_cpu"
    status = (
        "measured_config_retired"
        if exact_cuda and score > baseline_score
        else "exact_cuda_result_reviewed"
        if exact_cuda
        else "contest_cpu_result_reviewed"
    )
    failure_class = (
        "legitimate_score_regression_or_component_collapse"
        if status == "measured_config_retired"
        else "contest_cpu_public_leaderboard_anchor_cuda_pending"
        if exact_cpu
        else "not_negative_against_supplied_baseline"
    )
    row: dict[str, object] = {
        "schema": "tac_result_review_packet_v1",
        "tool": "tools/build_result_review_packet.py",
        "technique": "ias1_runtime_parity_top4",
        "lane_id": f"lane_ias1_{axis}",
        "job_id": f"job_ias1_{axis}",
        "source_json_path": f"experiments/results/{axis}/contest_auth_eval.json",
        "source_json_sha256": "4" * 64,
        "score_claim": False,
        "score_axis": axis,
        "score_claim_valid": exact_cuda,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "family_falsified": False,
        "method_family_retired": False,
        "measured_config_status": status,
        "failure_class": failure_class,
        "baseline_score": baseline_score,
        "canonical_score": score,
        "exact_cuda_evidence": exact_cuda,
        "exact_cpu_evidence": exact_cpu,
        "custody": {
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha256,
            "device": "cuda" if exact_cuda else "cpu",
            "gpu_model": "Tesla T4" if exact_cuda else "",
            "n_samples": 600,
            "inflate_script": "/tmp/submission/inflate.sh",
            "command": ["experiments/contest_auth_eval.py", "--device", axis],
        },
        "dispatch_claim_state": {
            "terminal_status_recorded": True,
            "latest_status": f"completed_{axis}_modal_auth_eval_recovered",
        },
        "runtime_custody": {
            "runtime_manifest_present": True,
            "runtime_tree_sha256": ("5" if exact_cuda else "6") * 64,
            "runtime_content_tree_sha256": runtime_content_tree_sha256,
            "runtime_file_count": 12,
            "runtime_files_listed": True,
            "payload_closure_fields_present": True,
            "inflate_script_sha256": "7" * 64,
            "inflated_output_manifest_sha256": ("8" if exact_cuda else "9") * 64,
            "inflated_output_aggregate_sha256": inflated_output_aggregate_sha256,
        },
        "score_recomputation": {
            "available": True,
            "matches_reported": True,
            "avg_segnet_dist": 0.00055979 if exact_cpu else 0.00066252,
            "avg_posenet_dist": 0.00002943 if exact_cpu else 0.00016845,
            "rate_term": 0.12067494979223735,
            "recomputed_score": score,
            "reported_score": score,
            "abs_delta_vs_reported": 0.0,
        },
        "engineering_forensic_audit": {
            "schema": "engineering_forensic_audit_v1",
            "custody_reviewed": True,
            "axis_reviewed": True,
            "runtime_config_reviewed": True,
            "archive_runtime_closure_reviewed": True,
            "score_formula_reviewed": True,
            "dispatch_claim_reviewed": True,
            "engineering_or_config_bug_found": False,
            "audit_blockers": [],
            "classification_after_audit": (
                "measured_config_retired_only"
                if status == "measured_config_retired"
                else "contest_cpu_axis_reviewed_cuda_pending"
                if exact_cpu
                else "exact_cuda_result_reviewed_no_negative_status_change"
            ),
            "dead_or_family_falsification_allowed": False,
            "measured_config_retirement_allowed": status == "measured_config_retired",
        },
        "reactivation_criteria": ["provide a byte-closed implementation change before redispatch"],
    }
    row.update(overrides)
    return row


def _mlx_selection(**overrides: object) -> dict[str, object]:
    false_authority = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    row = {
        "schema": "mlx_effective_spend_triage_candidate_row.v1",
        **false_authority,
        "rank": 1,
        "candidate_generation_only": True,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "selection_basis": "normalized_full_video_mlx_singleton_response_gain",
        "selection_planning_value_accessor": "scorer_response_planning_value_for_target",
        "selection_planning_value_scope": "normalized_full_video",
        "row_id": "best",
        "family": "mlx_decoder_q",
        "candidate_id": "mlx_scorer_response:window:best",
        "pair_indices": [10, 11],
        "source_pair_window": [10, 11],
        "source_path": "candidate_pair_best.json",
        "window_baseline_source_path": "baseline_pair_best.json",
        "archive_sha256": "a" * 64,
        "raw_sha256": "b" * 64,
        "source_inflated_outputs_aggregate_sha256": "c" * 64,
        "source_candidate_cache_array_sha256": {"pair_indices": "d" * 64},
        "source_reference_cache_array_sha256": {"pair_indices": "e" * 64},
        "window_baseline_candidate_cache_array_sha256": {"pair_indices": "f" * 64},
        "window_baseline_reference_cache_array_sha256": {"pair_indices": "1" * 64},
        "observed_scorer_gain_vs_baseline": 0.012,
        "full_video_denominator": 600,
        "normalized_full_video_scorer_gain_vs_baseline": 0.00002,
        "projected_full_video_delta_vs_baseline_score": -0.00002,
        "break_even_added_bytes_from_normalized_full_video_gain": (0.00002 / CONTEST_RATE_SCORE_PER_BYTE),
        "normalized_full_video_byte_budget_margin_vs_break_even": (0.00002 / CONTEST_RATE_SCORE_PER_BYTE),
        "added_archive_bytes": 0,
        "calibrated_min_mlx_gap_for_spend_triage": 0.00001,
        "prediction_field": "ll_predicted_delta_vs_baseline_score",
        "predicted_delta_vs_baseline_score": -0.000015,
        "prediction_agrees_with_observed_gain": True,
    }
    selection = {
        "schema": MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA,
        **false_authority,
        "candidate_generation_only": True,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "allowed_use": ("candidate_generation_filter_after_strict_effective_mlx_spend_triage_gate"),
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
        "score_axis": "[macOS-MLX research-signal]",
        "source_artifacts": {},
        "gates": {
            "effective_mlx_spend_triage_gate": {
                "schema": "ll_effective_mlx_spend_triage_gate.v1",
                "status": "strict_pass",
                "mlx_exact_eval_spend_triage_allowed": True,
                "allowed_use": ("local_exact_eval_spend_triage_filter_after_all_gates"),
            },
            "response_validation_status": "passed",
            "torch_parity_status": "strict_pass",
            "score_calibration_status": "strict_pass",
            "production_contract_status": "strict_pass",
        },
        "selection_policy": {
            "top_k": 1,
            "families": ["mlx_decoder_q"],
            "gate_spend_triage_allowed_families": ["mlx_decoder_q"],
            "min_observed_gain": 0.00001,
            "prediction_field": "ll_predicted_delta_vs_baseline_score",
            "require_prediction_negative": False,
            "require_singleton_windows": True,
            "planning_value_accessor": "scorer_response_planning_value_for_target",
            "planning_value_scope": "normalized_full_video",
        },
        "summary": {
            "dataset_row_count": 1,
            "eligible_row_count": 1,
            "selected_count": 1,
        },
        "selected_rows": [row],
    }
    selection.update(overrides)
    return selection


def _cross_family_byte_shaving_surface() -> dict[str, object]:
    return {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "family_agnostic_inverse_surface",
        "candidate_id": "hnerv_boostnerv_packetir_mix",
        "lane_id": "local_inverse_waterfill_family_mix",
        "combo_beam_width": 32,
        "max_combo_count": 8,
        "units": [
            {
                "unit_id": "hnerv_section_hi_latents",
                "unit_kind": "archive_section",
                "candidate_saved_bytes": 1300,
                "predicted_quality_score_delta": -0.0002,
                "confidence": 0.8,
                "operations": [
                    {
                        "operation_id": "hnerv_section_recode",
                        "operation_family": "section_entropy_recode",
                        "candidate_saved_bytes": 1300,
                        "predicted_quality_score_delta": -0.0002,
                        "materializer": "hnerv_section_recode_adapter",
                        "target_kind": "hnerv_payload_section_recode_v1",
                    }
                ],
            },
            {
                "unit_id": "boostnerv_overlay_tensor_07",
                "unit_kind": "tensor",
                "candidate_saved_bytes": 900,
                "predicted_quality_score_cost": 0.00002,
                "confidence": 0.75,
                "operations": [
                    {
                        "operation_id": "boostnerv_tensor_factorize",
                        "operation_family": "factorize_tensor",
                        "candidate_saved_bytes": 900,
                        "predicted_quality_score_cost": 0.00002,
                        "materializer": "boostnerv_tensor_overlay_adapter",
                        "target_kind": "boostnerv_tensor_factorization_v1",
                    }
                ],
            },
            {
                "unit_id": "packetir_member_merge",
                "unit_kind": "packet_member",
                "candidate_saved_bytes": 400,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.9,
                "operations": [
                    {
                        "operation_id": "packetir_member_merge",
                        "operation_family": "member_merge",
                        "candidate_saved_bytes": 400,
                        "predicted_quality_score_cost": 0.0,
                        "materializer": "packet_member_merge_adapter",
                        "target_kind": "packet_member_merge_v1",
                    }
                ],
            },
        ],
        "interactions": [
            {
                "interaction_id": "shared_hnerv_boostnerv_overlay_header",
                "unit_ids": [
                    "hnerv_section_hi_latents",
                    "boostnerv_overlay_tensor_07",
                ],
                "operation_families": [
                    "section_entropy_recode",
                    "factorize_tensor",
                ],
                "extra_saved_bytes": 180,
                "delta_score": -0.00003,
                "rationale": "shared overlay header disappears when section and tensor are compiled together",
            }
        ],
        **_planning_false_authority(),
    }


def test_atom_normalization_is_false_authority_only() -> None:
    atom = normalize_inverse_steganalysis_atom(_atom())

    assert atom["schema"] == ATOM_SCHEMA
    assert atom["candidate_generation_only"] is True
    assert atom["planning_only"] is True
    assert atom["scope_axis"] == "regions"
    assert atom["pair_indices"] == [7]
    assert atom["region_bbox"] == [16.0, 24.0, 96.0, 128.0]
    assert atom["first_order_marginal_effect"] == pytest.approx(0.0004)
    assert atom["interaction_kind"] == "synergy"
    assert atom["discontinuity_guard"]["blocked"] is False
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert atom[key] is value

    with pytest.raises(InverseSteganalysisAcquisitionError, match="score_claim=truthy"):
        normalize_inverse_steganalysis_atom(_atom(score_claim=True))


def test_action_functional_preserves_operation_set_compiler_hint() -> None:
    atom = normalize_inverse_steganalysis_atom(
        _atom(
            operation_set_compiler={
                "schema": "inverse_action_operation_set_compiler_hint.v1",
                "operation_set_id": "compiled_hint",
                "target_kind": "archive_section_entropy_recode_v1",
                "archive_section": "decoder_blob",
                "materializer": "archive_section_entropy_recode_adapter",
                "receiver_contract_kind": ("family_agnostic_archive_section_entropy_recode"),
            }
        )
    )
    action = build_discrete_scorer_action_functional([atom])
    cell = action["cells"][0]

    assert cell["operation_set_compiler"]["operation_set_id"] == "compiled_hint"
    assert cell["operation_set_compiler"]["target_kind"] == ("archive_section_entropy_recode_v1")
    assert action["water_bucket"]["selected_count"] == 1
    selected = action["water_bucket"]["selected_cells"][0]
    assert selected["operation_set_compiler"]["operation_set_id"] == "compiled_hint"
    assert selected["operation_set_target_kind"] == ("archive_section_entropy_recode_v1")
    assert selected["target_kind"] == "archive_section_entropy_recode_v1"
    assert selected["materializer_id"] == "archive_section_entropy_recode_adapter"
    assert selected["receiver_contract_kind"] == ("family_agnostic_archive_section_entropy_recode")
    assert cell["score_claim"] is False
    assert action["score_claim"] is False


def test_action_functional_preserves_explicit_target_metadata_for_compiler() -> None:
    atom = _atom(
        operation_set_target_kind="archive_section_entropy_recode_v1",
        operation_set_params={"section_name": "decoder_blob"},
    )
    action = build_discrete_scorer_action_functional([atom])
    cell = action["cells"][0]

    assert cell["operation_set_target_kind"] == ("archive_section_entropy_recode_v1")
    assert cell["operation_set_params"] == {"section_name": "decoder_blob"}
    assert action["score_claim"] is False


def test_inverse_scorer_surface_preserves_operation_compiler_metadata() -> None:
    compiler = {
        "schema": "inverse_action_operation_set_compiler_hint.v1",
        "operation_set_id": "inverse_surface_compiler",
        "candidate_saved_bytes": 64,
        "operation_portability": "family_agnostic",
        "selected_operations": [
            {
                "unit_id": "compiled_decoder_blob",
                "target_kind": "archive_section_entropy_recode_v1",
                "archive_section": "decoder_blob",
                "candidate_saved_bytes": 64,
                "representation_family_class": "hnerv_variant",
            }
        ],
        **_planning_false_authority(),
    }
    atoms = action_atoms_from_inverse_scorer_surface(
        {
            "schema": INVERSE_SCORER_SURFACE_SCHEMA,
            "cells": [
                {
                    "cell_id": "pair0007_decoder_blob",
                    "decision_surface_class": "rate_only_null_space",
                    "dominant_receiver_axis": "rate",
                    "candidate_saved_bytes": 64,
                    "median_projected_delta_vs_baseline_score": -0.0001,
                    "best_projected_delta_vs_baseline_score": -0.00012,
                    "worst_projected_delta_vs_baseline_score": -0.00008,
                    "median_scorer_delta_vs_baseline": 0.0,
                    "operation_set_compiler": compiler,
                    "operation_set_target_kind": "archive_section_entropy_recode_v1",
                    "operation_set_params": {"section_name": "decoder_blob"},
                }
            ],
            **_planning_false_authority(),
        }
    )
    action = build_discrete_scorer_action_functional(atoms)
    cell = action["cells"][0]

    assert atoms[0]["operation_set_compiler"]["operation_set_id"] == ("inverse_surface_compiler")
    assert cell["operation_set_compiler"]["operation_set_id"] == ("inverse_surface_compiler")
    assert cell["operation_set_target_kind"] == "archive_section_entropy_recode_v1"
    assert cell["operation_set_params"] == {"section_name": "decoder_blob"}
    assert cell["score_claim"] is False


def test_byte_shaving_plan_producer_preserves_compiler_and_target_metadata() -> None:
    compiler = {
        "schema": "inverse_action_operation_set_compiler_hint.v1",
        "operation_set_id": "producer_compiled_set",
        "candidate_saved_bytes": 128,
        "operation_portability": "family_agnostic",
        "selected_operations": [
            {
                "unit_id": "producer_decoder_blob",
                "target_kind": "archive_section_entropy_recode_v1",
                "archive_section": "decoder_blob",
                "candidate_saved_bytes": 128,
            }
        ],
        **_planning_false_authority(),
    }
    plan = {
        "schema": "byte_shaving_campaign_plan.v1",
        "candidate_id": "producer_candidate",
        "lane_id": "producer_lane",
        "operation_set_ladder": [
            {
                "schema": COUPLED_OPERATION_SET_SCHEMA,
                "operation_set_id": "producer_opset",
                "candidate_saved_bytes": 128,
                "expected_score_gain": 0.0002,
                "selected_operations": [
                    {
                        "unit_id": "producer_decoder_blob",
                        "operation_family": "section_entropy_recode",
                        "target_kind": "archive_section_entropy_recode_v1",
                    }
                ],
                "operation_set_compiler": compiler,
                **_planning_false_authority(),
            }
        ],
        **_planning_false_authority(),
    }

    opset_atoms = action_atoms_from_byte_shaving_campaign_plan(plan)
    action = build_discrete_scorer_action_functional(opset_atoms)
    opset_cell = action["cells"][0]

    assert opset_atoms[0]["operation_set_compiler"]["operation_set_id"] == ("producer_compiled_set")
    assert opset_atoms[0]["source_provenance"]["operation_set_compiler"]["operation_set_id"] == "producer_compiled_set"
    assert opset_cell["operation_set_compiler"]["operation_set_id"] == ("producer_compiled_set")

    ranked_plan = {
        "schema": "byte_shaving_campaign_plan.v1",
        "candidate_id": "ranked_candidate",
        "lane_id": "ranked_lane",
        "ranked_units": [
            {
                "unit_id": "ranked_decoder_blob",
                "unit_kind": "archive_section",
                "candidate_saved_bytes": 64,
                "expected_score_gain": 0.0001,
                "quality_cost_score": 0.0,
                "recommended_operation_family": "section_entropy_recode",
                "recommended_operation_target_kind": ("archive_section_entropy_recode_v1"),
                "recommended_operation_params": {"section_name": "decoder_blob"},
                **_planning_false_authority(),
            }
        ],
        **_planning_false_authority(),
    }
    ranked_atoms = action_atoms_from_byte_shaving_campaign_plan(ranked_plan)
    ranked_action = build_discrete_scorer_action_functional(ranked_atoms)
    ranked_cell = ranked_action["cells"][0]

    assert ranked_cell["operation_set_target_kind"] == ("archive_section_entropy_recode_v1")
    assert ranked_cell["operation_set_operation_family"] == "section_entropy_recode"
    assert ranked_cell["operation_set_params"] == {"section_name": "decoder_blob"}
    assert ranked_atoms[0]["source_provenance"]["recommended_operation_params"] == {"section_name": "decoder_blob"}


def test_mlx_acquisition_batch_compiler_hint_survives_to_action_cell() -> None:
    selection = _mlx_selection()
    row = selection["selected_rows"][0]
    row["operation_set_compiler"] = {
        "schema": "inverse_action_operation_set_compiler_hint.v1",
        "operation_set_id": "mlx_row_compiler",
        "operation_portability": "family_agnostic",
        "selected_operations": [
            {
                "unit_id": "compiled_decoder_blob",
                "target_kind": "archive_section_entropy_recode_v1",
                "archive_section": "decoder_blob",
                "candidate_saved_bytes": 256,
            },
            {
                "unit_id": "compiled_tensor_overlay",
                "target_kind": "tensor_factorize_v1",
                "tensor_name": "decoder.overlay",
                "candidate_saved_bytes": 128,
            },
        ],
    }

    batch = build_mlx_acquisition_batch_from_selection(selection, set_size=1)
    operation_set = batch["operation_sets"][0]
    atoms = action_atoms_from_mlx_acquisition_batch(batch)
    action = build_discrete_scorer_action_functional(atoms)
    cell = action["cells"][0]
    provenance = cell["source_provenance"]

    assert operation_set["operation_set_compiler"]["operation_set_id"].endswith("_compiled")
    assert {
        operation["target_kind"] for operation in operation_set["operation_set_compiler"]["selected_operations"]
    } == {"archive_section_entropy_recode_v1", "tensor_factorize_v1"}
    assert atoms[0]["operation_set_compiler"] == operation_set["operation_set_compiler"]
    assert cell["operation_set_compiler"] == operation_set["operation_set_compiler"]
    assert provenance["operation_set_compiler"] == operation_set["operation_set_compiler"]
    assert cell["score_claim"] is False
    assert provenance["score_claim"] is False
    assert action["score_claim"] is False


def test_mlx_grouped_structural_interactions_survive_to_action_cell() -> None:
    selection = _mlx_selection()
    first = selection["selected_rows"][0]
    first["operation_set_compiler"] = {
        "schema": "inverse_action_operation_set_compiler_hint.v1",
        "selected_operations": [
            {
                "unit_id": "gate_src_a_ch0",
                "operation_family": "dynamic_sparse_channel_gate",
                "target_kind": "tensor_factorize_v1",
                "packet_member": "decoder.bin",
                "tensor_name": "decoder.gate",
                "params": {
                    "dynamic_sparse_channel_gate": {
                        "source_id": "src_a",
                        "channel_id": "ch0",
                    }
                },
            }
        ],
    }
    second = {
        **first,
        "row_id": "second",
        "candidate_id": "mlx_scorer_response:window:second",
        "pair_indices": [11, 12],
        "source_pair_window": [11, 12],
        "operation_set_compiler": {
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "selected_operations": [
                {
                    "unit_id": "gate_src_b_ch0",
                    "operation_family": "dynamic_sparse_channel_gate",
                    "target_kind": "tensor_factorize_v1",
                    "packet_member": "decoder.bin",
                    "tensor_name": "decoder.gate",
                    "params": {
                        "dynamic_sparse_channel_gate": {
                            "source_id": "src_b",
                            "channel_id": "ch0",
                        }
                    },
                }
            ],
        },
    }
    selection["selected_rows"].append(second)

    batch = build_mlx_acquisition_batch_from_selection(selection, set_size=2)
    atoms = action_atoms_from_mlx_acquisition_batch(batch)
    action = build_discrete_scorer_action_functional(atoms)
    cell = action["cells"][0]
    provenance = cell["source_provenance"]
    interaction_kinds = {
        row["interaction_kind"] for row in provenance["active_interactions"]
    }

    assert "dynamic_sparse_same_channel" in interaction_kinds
    assert "shared_pair_index" in interaction_kinds
    assert "shared_packet_member" in interaction_kinds
    assert atoms[0]["second_order_interaction_effect"] == 0.0
    assert cell["second_order_interaction_effect"] == 0.0
    assert provenance["score_claim"] is False


def test_mlx_acquisition_batch_rejects_truthy_nested_compiler_authority() -> None:
    selection = _mlx_selection()
    selection["selected_rows"][0]["operation_set_compiler"] = {
        "schema": "inverse_action_operation_set_compiler_hint.v1",
        "selected_operations": [
            {
                "target_kind": "archive_section_entropy_recode_v1",
                "archive_section": "decoder_blob",
                "score_claim": True,
            }
        ],
    }

    with pytest.raises(ValueError, match="forbidden truthy authority fields"):
        build_mlx_acquisition_batch_from_selection(selection, set_size=1)


def test_mlx_direct_spend_triage_compiler_hint_survives_to_action_cell() -> None:
    selection = _mlx_selection()
    compiler_hint = {
        "schema": "inverse_action_operation_set_compiler_hint.v1",
        "operation_set_id": "direct_mlx_compiler",
        "operation_portability": "family_agnostic",
        "selected_operations": [
            {
                "unit_id": "direct_decoder_blob",
                "target_kind": "archive_section_entropy_recode_v1",
                "archive_section": "decoder_blob",
                "candidate_saved_bytes": 256,
            },
            {
                "unit_id": "direct_packet_member",
                "target_kind": "packet_member_recompress_v1",
                "member_name": "0.bin",
                "candidate_saved_bytes": 128,
            },
        ],
    }
    selection["selected_rows"][0]["operation_set_compiler"] = compiler_hint

    atoms = inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(
        selection,
        source_path="selection.json",
    )
    action = build_discrete_scorer_action_functional(atoms)
    cell = action["cells"][0]
    provenance = cell["source_provenance"]

    assert atoms[0]["operation_set_compiler"] == compiler_hint
    assert cell["operation_set_compiler"] == compiler_hint
    assert provenance["operation_set_compiler"] == compiler_hint
    assert cell["score_claim"] is False
    assert provenance["score_claim"] is False
    assert action["score_claim"] is False


def test_materializer_observation_matches_compiler_target_and_materializer() -> None:
    saved_bytes = 52
    observed_rate_gain = CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes)
    atom = _atom(
        "candidate_parent",
        atom_id="atom_materializer_bridge",
        fragility_penalty=0.0,
        uncertainty=0.0,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "header_elide_group",
            "selected_operations": [
                {
                    "unit_id": "renderer_bin_header_elide",
                    "target_kind": "packet_member_zip_header_elide_v1",
                    "materializer": "packet_member_zip_header_elide_adapter",
                    "receiver_contract_kind": ("packet_member_zip_header_elide_receiver_proof.v1"),
                    "candidate_saved_bytes": saved_bytes,
                }
            ],
        },
    )
    observation = {
        "schema": "family_agnostic_materializer_empirical_observation.v1",
        "observation_kind": "family_agnostic_materializer_empirical_observation",
        "observation_id": "obs_header_elide_renderer_bin",
        "candidate_id": "sweep_row_not_candidate_parent",
        "axis": "[local-materializer-proof]",
        "runtime_identity": {
            "runtime_contract_sha256": "a" * 64,
            "scorer_version": "family_agnostic_materializer_empirical_sweep.v1",
        },
        "cache_identity": {
            "cache_sha256": "b" * 64,
            "source_archive_sha256": "c" * 64,
        },
        "target_kind": "packet_member_zip_header_elide_v1",
        "materializer_id": "packet_member_zip_header_elide_adapter",
        "portability_contract": {
            "schema": "family_agnostic_materializer_portability_contract.v1",
            "materializer_id": "packet_member_zip_header_elide_adapter",
            "target_kind": "packet_member_zip_header_elide_v1",
            "requires_gpu": False,
            "implementation_language": "python",
        },
        "receiver_contract_kind": "packet_member_zip_header_elide_receiver_proof.v1",
        "saved_bytes": saved_bytes,
        "observed_rate_gain": observed_rate_gain,
        "observed_score_gain": observed_rate_gain,
        "artifact_bytes": 345_750,
        "resource_kind": "local_cpu",
        "rate_positive": True,
        "receiver_contract_satisfied": True,
        **_planning_false_authority(),
    }

    normalized = normalize_inverse_steganalysis_observation(observation)
    action = build_discrete_scorer_action_functional([atom], observations=[observation])
    cell = action["cells"][0]

    assert normalized["source_observation_schema"] == ("family_agnostic_materializer_empirical_observation.v1")
    assert normalized["target_kind"] == "packet_member_zip_header_elide_v1"
    assert normalized["materializer_id"] == "packet_member_zip_header_elide_adapter"
    assert normalized["portability_contract"]["requires_gpu"] is False
    assert normalized["saved_bytes"] == saved_bytes
    assert normalized["observed_rate_gain"] == pytest.approx(observed_rate_gain)
    assert normalized["rate_positive"] is True
    assert cell["best_observation_id"] == "obs_header_elide_renderer_bin"
    assert cell["best_observation_kind"] == ("family_agnostic_materializer_empirical_observation")
    assert cell["priority"]["expected_score_gain"] == pytest.approx(observed_rate_gain)
    assert cell["materializer_archive_delta_feedback"]["observation_count"] == 1
    assert cell["materializer_archive_delta_feedback"]["rate_positive_count"] == 1
    assert cell["materializer_archive_delta_feedback"]["blocks_water_bucket"] is False
    assert cell["score_claim"] is False
    assert action["score_claim"] is False


def test_receiver_negative_materializer_sweep_blocks_matching_water_bucket() -> None:
    saved_bytes = 66
    observed_rate_gain = CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes)
    atom = _atom(
        "candidate_parent",
        atom_id="atom_receiver_negative_sweep",
        fragility_penalty=0.0,
        uncertainty=0.0,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "section_recode_group",
            "selected_operations": [
                {
                    "unit_id": "decoder_packed_brotli",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "materializer": "archive_section_entropy_recode_adapter",
                    "receiver_contract_kind": ("family_agnostic_archive_section_entropy_recode"),
                    "candidate_saved_bytes": saved_bytes,
                }
            ],
        },
    )
    observation = {
        "schema": "family_agnostic_materializer_empirical_observation.v1",
        "observation_kind": "family_agnostic_materializer_empirical_observation",
        "observation_id": "receiver_negative_archive_section_obs",
        "candidate_id": "receiver_negative_archive_section_candidate",
        "axis": "[local-materializer-receiver-feedback]",
        "runtime_identity": {
            "runtime_contract_sha256": "a" * 64,
            "scorer_version": "family_agnostic_materializer_empirical_sweep.v1",
        },
        "cache_identity": {"cache_sha256": "b" * 64},
        "target_kind": "archive_section_entropy_recode_v1",
        "materializer_id": "archive_section_entropy_recode_adapter",
        "receiver_contract_kind": "family_agnostic_archive_section_entropy_recode",
        "saved_bytes": saved_bytes,
        "observed_rate_gain": observed_rate_gain,
        "observed_score_gain": observed_rate_gain,
        "artifact_bytes": 143,
        "resource_kind": "local_cpu",
        "rate_positive": True,
        "receiver_contract_satisfied": False,
        "readiness_blockers": [
            "section_length_changed_requires_runtime_consumption_proof",
            "runtime_consumption_proof_not_passed",
        ],
        "source_unit_ids": ["decoder_packed_brotli"],
        "source_selection_ids": ["opset_combo_0001"],
        **_planning_false_authority(),
    }

    action = build_discrete_scorer_action_functional([atom], observations=[observation])
    cell = action["cells"][0]
    feedback = cell["materializer_archive_delta_feedback"]

    assert cell["water_bucket_selectable"] is False
    assert cell["materializer_archive_delta_blocked"] is True
    assert feedback["observation_count"] == 1
    assert feedback["rate_positive_count"] == 1
    assert feedback["rate_nonpositive_count"] == 0
    assert feedback["realized_saved_bytes_sum"] == saved_bytes
    assert "receiver_negative_materializer_success" in feedback["blockers"]
    assert "rate_negative_materializer_success" not in feedback["blockers"]
    assert action["water_bucket"]["selected_count"] == 0
    assert action["score_claim"] is False


def test_acquisition_plan_blocks_mixed_receiver_negative_materializer_feedback() -> None:
    saved_bytes = 66
    observed_rate_gain = CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes)
    atom = _atom(
        "candidate_parent",
        atom_id="atom_mixed_receiver_feedback",
        fragility_penalty=0.0,
        uncertainty=0.0,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "section_recode_group",
            "selected_operations": [
                {
                    "unit_id": "decoder_packed_brotli",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "materializer": "archive_section_entropy_recode_adapter",
                    "receiver_contract_kind": ("family_agnostic_archive_section_entropy_recode"),
                    "candidate_saved_bytes": saved_bytes,
                }
            ],
        },
    )
    positive_observation = {
        "schema": "family_agnostic_materializer_empirical_observation.v1",
        "observation_kind": "family_agnostic_materializer_empirical_observation",
        "observation_id": "positive_archive_section_obs",
        "candidate_id": "positive_archive_section_candidate",
        "axis": "[local-materializer-proof]",
        "runtime_identity": {"runtime_contract_sha256": "a" * 64},
        "cache_identity": {"cache_sha256": "b" * 64},
        "target_kind": "archive_section_entropy_recode_v1",
        "materializer_id": "archive_section_entropy_recode_adapter",
        "receiver_contract_kind": "family_agnostic_archive_section_entropy_recode",
        "saved_bytes": saved_bytes,
        "observed_rate_gain": observed_rate_gain,
        "observed_score_gain": observed_rate_gain,
        "artifact_bytes": 143,
        "resource_kind": "local_cpu",
        "rate_positive": True,
        "receiver_contract_satisfied": True,
        **_planning_false_authority(),
    }
    receiver_negative_observation = {
        **positive_observation,
        "observation_id": "receiver_negative_archive_section_obs",
        "candidate_id": "receiver_negative_archive_section_candidate",
        "axis": "[local-materializer-receiver-feedback]",
        "receiver_contract_satisfied": False,
        "readiness_blockers": ["runtime_consumption_proof_not_passed"],
        "source_unit_ids": ["decoder_packed_brotli"],
        "source_selection_ids": ["opset_combo_0001"],
    }

    plan = build_inverse_steganalysis_acquisition_plan(
        [atom],
        observations=[positive_observation, receiver_negative_observation],
    )
    row = plan["ranked_atoms"][0]
    feedback = row["materializer_archive_delta_feedback"]

    assert row["observation_count"] == 2
    assert row["best_observation_id"] == "positive_archive_section_obs"
    assert row["materializer_archive_delta_blocked"] is True
    assert row["priority"]["expected_score_gain"] == 0.0
    assert row["priority"]["acquisition_priority"] == 0.0
    assert feedback["observation_count"] == 2
    assert feedback["blocking_observation_count"] == 1
    assert "receiver_negative_materializer_success" in feedback["blockers"]
    assert plan["summary"]["materializer_archive_delta_blocked_count"] == 1
    assert plan["score_claim"] is False


def test_materializer_chain_realized_cost_blocks_matching_water_bucket() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_materializer_chain.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    atom = _atom(
        "surface_candidate",
        atom_id="atom_materializer_delta",
        predicted_score_gain=0.001,
        first_order_marginal_effect=0.001,
        second_order_interaction_effect=0.0,
        uncertainty=0.0,
        fragility_penalty=0.0,
    )
    unrelated_atom = _atom(
        "surface_candidate",
        atom_id="atom_unrelated_same_candidate",
        predicted_score_gain=0.001,
        first_order_marginal_effect=0.001,
        second_order_interaction_effect=0.0,
        uncertainty=0.0,
        fragility_penalty=0.0,
    )
    manifest = {
        "schema": "inverse_scorer_cell_candidate_chain_v1",
        "serialized_archive_delta": {
            "schema": "serialized_archive_delta_contract.v1",
            "status": "realized_cost",
            "archive_delta_bytes": 1764,
            "source_archive_bytes": 178_592,
            "candidate_archive_bytes": 180_356,
            "realized_saved_bytes": -1764,
            "savings_realized": False,
            **_planning_false_authority(),
        },
        "source_archive_sha256": "a" * 64,
        "candidate_archive_sha256": "b" * 64,
        "source_archive_bytes": 178_592,
        "candidate_archive_bytes": 180_356,
        "receiver_contract_satisfied": True,
        "inflate_parity_satisfied": True,
        "readiness_blockers": ["exact_auth_eval_required_before_score_claim"],
        "dispatch_blockers": ["requires_exact_eval_readiness_gate"],
        **_planning_false_authority(),
    }
    candidate_manifest = {
        "schema": "inverse_scorer_cell_candidate_v1",
        "materializer_id": "inverse_scorer_cell_candidate_adapter",
        "target_kind": "inverse_scorer_cell_candidate_v1",
        "receiver_contract_kind": "inverse_scorer_coordinate_candidate",
        "selected_cells": [
            {
                "atom_id": "atom_materializer_delta",
                "candidate_id": "surface_candidate",
                "source_selection_ids": ["top_0001"],
            }
        ],
        **_planning_false_authority(),
    }

    observations = observations_from_materializer_chain_manifest(
        manifest,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        candidate_manifest=candidate_manifest,
        source_path="scratch/inverse_cell_chain_manifest.json",
    )
    direct_manifest = {
        **manifest,
        "materializer_id": "inverse_scorer_cell_candidate_adapter",
        "target_kind": "inverse_scorer_cell_candidate_v1",
        "receiver_contract_kind": "inverse_scorer_coordinate_candidate",
        "selected_cells": candidate_manifest["selected_cells"],
    }
    direct_observations = observations_from_materializer_chain_manifest(
        direct_manifest,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        source_path="scratch/inverse_cell_candidate_manifest.json",
    )
    action = build_discrete_scorer_action_functional(
        [atom, unrelated_atom],
        observations=[*observations, *direct_observations],
    )
    cells_by_atom = {cell["atom_id"]: cell for cell in action["cells"]}
    cell = cells_by_atom["atom_materializer_delta"]
    unrelated_cell = cells_by_atom["atom_unrelated_same_candidate"]
    feedback = cell["materializer_archive_delta_feedback"]

    assert observations[0]["observation_kind"] == MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND
    assert observations[0]["saved_bytes"] == -1764
    assert observations[0]["observed_rate_gain"] == 0.0
    assert observations[0]["rate_positive"] is False
    assert observations[0]["materializer_rate_outcome"] == "realized_cost"
    assert observations[0]["signal_semantics"] == ("successful_quality_spend_not_byte_saving_progress")
    assert direct_observations[0]["source_unit_ids"] == [
        "atom_materializer_delta",
        "inverse_action_atom_materializer_delta",
    ]
    assert direct_observations[0]["materializer_id"] == ("inverse_scorer_cell_candidate_adapter")
    assert cell["best_observation_id"] == observations[0]["observation_id"]
    assert cell["priority"]["expected_score_gain"] == 0.0
    assert cell["water_bucket_selectable"] is False
    assert cell["materializer_archive_delta_blocked"] is True
    assert unrelated_cell["materializer_archive_delta_blocked"] is False
    assert unrelated_cell["water_bucket_selectable"] is True
    assert feedback["blocks_water_bucket"] is True
    assert feedback["observation_count"] == 1
    assert feedback["realized_saved_bytes_sum"] == -1764
    assert "rate_negative_materializer_success" in feedback["blockers"]
    assert action["integral_totals"]["materializer_archive_delta_blocked_cell_count"] == 1
    assert action["materializer_archive_delta_feedback"]["blocking_observation_count"] == 1
    assert action["score_claim"] is False


def test_materializer_delta_blocks_replanned_byte_shaving_ranked_unit_identity() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_materializer_chain.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    plan = {
        "schema": "byte_shaving_campaign_plan.v1",
        "candidate_id": "surface_candidate",
        "ranked_units": [
            {
                "unit_id": "inverse_action_atom_materializer_delta",
                "unit_kind": "scorer_inverse_surface_cell",
                "atom_ids": ["atom_materializer_delta"],
                "source_candidate_id": "surface_candidate",
                "candidate_saved_bytes": 0,
                "expected_score_gain": 0.001,
                "quality_cost_score": 0.0,
                "recommended_operation_params": {
                    "atom_id": "atom_materializer_delta",
                    "candidate_id": "surface_candidate",
                },
                "operation_candidates": [
                    {
                        "operation_params": {
                            "atom_id": "atom_materializer_delta",
                            "candidate_id": "surface_candidate",
                        }
                    }
                ],
            }
        ],
        **_planning_false_authority(),
    }
    atoms = action_atoms_from_byte_shaving_campaign_plan(plan)
    manifest = {
        "schema": "inverse_scorer_cell_candidate_v1",
        "materializer_id": "inverse_scorer_cell_candidate_adapter",
        "target_kind": "inverse_scorer_cell_candidate_v1",
        "receiver_contract_kind": "inverse_scorer_coordinate_candidate",
        "serialized_archive_delta": {
            "schema": "serialized_archive_delta_contract.v1",
            "status": "realized_cost",
            "archive_delta_bytes": 1805,
            "source_archive_bytes": 178_592,
            "candidate_archive_bytes": 180_397,
            "realized_saved_bytes": -1805,
            "savings_realized": False,
            **_planning_false_authority(),
        },
        "selected_cells": [
            {
                "atom_id": "atom_materializer_delta",
                "candidate_id": "surface_candidate",
            }
        ],
        **_planning_false_authority(),
    }

    observations = observations_from_materializer_chain_manifest(
        manifest,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        source_path="scratch/replanned_direct_materializer_manifest.json",
    )
    action = build_discrete_scorer_action_functional(atoms, observations=observations)
    cell = action["cells"][0]

    assert cell["atom_id"].startswith("byte_shaving_unit_inverse_action_atom_materializer_delta")
    assert atoms[0]["source_provenance"]["atom_ids"] == ["atom_materializer_delta"]
    assert observations[0]["source_unit_ids"] == [
        "atom_materializer_delta",
        "inverse_action_atom_materializer_delta",
    ]
    assert cell["materializer_archive_delta_blocked"] is True
    assert cell["water_bucket_selectable"] is False
    assert cell["priority"]["expected_score_gain"] == 0.0


def test_mlx_direct_spend_triage_rejects_truthy_nested_compiler_authority() -> None:
    selection = _mlx_selection()
    selection["selected_rows"][0]["operation_set_compiler"] = {
        "schema": "inverse_action_operation_set_compiler_hint.v1",
        "selected_operations": [
            {
                "target_kind": "archive_section_entropy_recode_v1",
                "archive_section": "decoder_blob",
                "score_claim": True,
            }
        ],
    }

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match=r"operation_set_compiler.*score_claim=truthy",
    ):
        inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(selection)


def test_action_surface_terms_model_scope_interactions_and_fragility() -> None:
    atom = normalize_inverse_steganalysis_atom(
        _atom(
            scale="byte_range",
            scope_axis="bytes",
            first_order_marginal_effect=0.0006,
            second_order_interaction_effect=-0.0002,
            discontinuity_risk=0.9,
            discontinuity_threshold=0.5,
        )
    )

    terms = action_surface_terms(atom)

    assert terms["scope_axis"] == "bytes"
    assert terms["first_order_marginal_effect"] == pytest.approx(0.0006)
    assert terms["second_order_interaction_effect"] == pytest.approx(-0.0002)
    assert terms["interaction_kind"] == "antagonism"
    assert terms["synergy_effect"] == pytest.approx(0.0)
    assert terms["antagonism_effect"] == pytest.approx(0.0002)
    assert terms["discontinuity_guard"]["blocked"] is True


def test_observation_requires_axis_candidate_runtime_and_cache_identity() -> None:
    assert normalize_inverse_steganalysis_observation(_observation())["schema"] == (OBSERVATION_SCHEMA)

    for missing_key, message in [
        ("candidate_id", "candidate_id"),
        ("axis", "axis"),
        ("runtime_identity", "runtime_identity"),
        ("cache_identity", "cache_identity"),
    ]:
        row = _observation()
        row.pop(missing_key)
        with pytest.raises(InverseSteganalysisAcquisitionError, match=message):
            normalize_inverse_steganalysis_observation(row)

    with pytest.raises(InverseSteganalysisAcquisitionError, match="runtime_identity"):
        normalize_inverse_steganalysis_observation(_observation(runtime_identity={"note": "not identity"}))
    with pytest.raises(InverseSteganalysisAcquisitionError, match="cache_identity"):
        normalize_inverse_steganalysis_observation(_observation(cache_identity={"note": "not identity"}))

    with pytest.raises(InverseSteganalysisAcquisitionError, match="contest auth evidence"):
        normalize_inverse_steganalysis_observation(_observation(axis="[contest-CUDA]"))
    with pytest.raises(InverseSteganalysisAcquisitionError, match="contest auth evidence"):
        normalize_inverse_steganalysis_observation(_observation(resource_kind="contest_exact_eval"))


def test_calibrated_observations_rank_acquisition_candidates() -> None:
    atoms = [
        _atom(
            "candidate_fast_weak",
            predicted_score_gain=0.0002,
            elapsed_seconds=2.0,
        ),
        _atom(
            "candidate_slow_strong",
            predicted_score_gain=0.0002,
            elapsed_seconds=2.0,
        ),
    ]
    observations = [
        _observation(
            "candidate_fast_weak",
            observed_score_gain=0.00024,
            calibration_error=0.00002,
            elapsed_seconds=2.0,
            artifact_bytes=1_000_000,
            resource_kind="local_cpu",
        ),
        _observation(
            "candidate_slow_strong",
            observed_score_gain=0.0009,
            calibration_error=0.00003,
            elapsed_seconds=8.0,
            artifact_bytes=2_000_000,
            resource_kind="local_mlx",
        ),
    ]

    plan = build_inverse_steganalysis_acquisition_plan(
        atoms,
        observations=observations,
    )

    assert plan["schema"] == SCHEMA
    assert [row["candidate_id"] for row in plan["ranked_atoms"]] == [
        "candidate_slow_strong",
        "candidate_fast_weak",
    ]
    top = plan["ranked_atoms"][0]
    assert top["best_observation_id"] == "obs_candidate_slow_strong"
    assert top["priority"]["expected_score_gain"] == pytest.approx(0.000836)
    assert top["priority"]["acquisition_priority"] > 0
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert plan[key] is value
        assert top[key] is value


def test_local_proxy_observations_never_become_promotion_or_rank_authority() -> None:
    observation = normalize_inverse_steganalysis_observation(_observation(axis="[macOS-CPU advisory]"))
    priority = compute_acquisition_priority(_atom(), observation)
    plan = build_inverse_steganalysis_acquisition_plan(
        [_atom()],
        observations=[observation],
    )

    assert observation["score_claim"] is False
    assert observation["promotion_eligible"] is False
    assert observation["rank_or_kill_eligible"] is False
    assert observation["ready_for_exact_eval_dispatch"] is False
    assert observation["promotable"] is False
    assert priority["resource_kind"] == "local_mlx"
    assert plan["ranked_atoms"][0]["acquisition_rank"] == 1
    assert plan["ranked_atoms"][0]["rank_or_kill_eligible"] is False
    assert plan["ranked_atoms"][0]["promotion_eligible"] is False

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="rank_or_kill_eligible=truthy",
    ):
        normalize_inverse_steganalysis_observation(_observation(rank_or_kill_eligible=True))


def test_mlx_effective_spend_triage_selection_becomes_false_authority_atoms() -> None:
    atoms = inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(
        _mlx_selection(),
        source_path="artifacts/selection.json",
        elapsed_seconds=2.5,
    )

    assert len(atoms) == 1
    atom = atoms[0]
    assert atom["schema"] == ATOM_SCHEMA
    assert atom["candidate_generation_only"] is True
    assert atom["resource_kind"] == "local_mlx"
    assert atom["scope_axis"] == "pairs"
    assert atom["pair_indices"] == [10, 11]
    assert atom["predicted_score_gain"] == pytest.approx(0.00002)
    assert atom["calibration_error"] == pytest.approx(0.00001)
    provenance = atom["source_provenance"]
    assert provenance["selection_source_path"] == "artifacts/selection.json"
    assert provenance["source_row_id"] == "best"
    assert provenance["selection_planning_value_scope"] == "normalized_full_video"
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert atom[key] is value
        assert provenance[key] is value

    action = build_discrete_scorer_action_functional(
        atoms,
        total_byte_budget=64,
    )
    cell = action["cells"][0]
    assert cell["source_provenance"]["source_row_id"] == "best"
    assert cell["candidate_generation_only"] is True
    assert cell["water_bucket_selectable"] is True
    assert action["water_bucket"]["selected_count"] == 1
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert cell[key] is value
        assert action[key] is value


def test_byte_shaving_surface_becomes_family_agnostic_action_atoms() -> None:
    atoms = action_atoms_from_byte_shaving_signal_surface(
        _cross_family_byte_shaving_surface(),
        source_path="artifacts/family_surface.json",
        resource_kind="local_mlx",
        elapsed_seconds=6.0,
    )
    action = build_discrete_scorer_action_functional(atoms)
    top_atom = atoms[0]
    provenance = top_atom["source_provenance"]

    assert top_atom["schema"] == ATOM_SCHEMA
    assert top_atom["candidate_id"] == "hnerv_boostnerv_packetir_mix"
    assert top_atom["scale"] == "multiscale"
    assert top_atom["scope_axis"] == "full_video"
    assert top_atom["resource_kind"] == "local_mlx"
    assert top_atom["second_order_interaction_effect"] > 0.0
    assert provenance["schema"] == ("inverse_steganalysis_byte_shaving_operation_set_provenance.v1")
    assert provenance["source_path"] == "artifacts/family_surface.json"
    assert "section_entropy_recode" in provenance["operation_families"]
    assert "factorize_tensor" in provenance["operation_families"]
    assert provenance["active_interactions"][0]["interaction_id"] == ("shared_hnerv_boostnerv_overlay_header")
    assert provenance["chosen_operation_sequence"]
    assert action["water_bucket"]["selected_count"] >= 1
    assert action["cells"][0]["source_provenance"]["score_claim"] is False
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert top_atom[key] is value
        assert provenance[key] is value
        assert action[key] is value


def test_byte_shaving_campaign_plan_bridge_preserves_operation_sets() -> None:
    plan = build_byte_shaving_campaign_plan(_cross_family_byte_shaving_surface())
    atoms = action_atoms_from_byte_shaving_campaign_plan(
        plan,
        source_path="artifacts/family_plan.json",
    )

    assert atoms
    assert atoms[0]["source_provenance"]["source_plan_schema"] == ("byte_shaving_campaign_plan.v1")
    assert atoms[0]["source_provenance"]["operation_set_id"].startswith("opset_")
    assert atoms[0]["source_provenance"]["partial_materialization_allowed"] is False
    assert atoms[0]["score_claim"] is False

    bad = dict(plan)
    bad["ready_for_exact_eval_dispatch"] = True
    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="ready_for_exact_eval_dispatch=truthy",
    ):
        action_atoms_from_byte_shaving_campaign_plan(bad)


def test_mlx_effective_spend_triage_selection_bridge_rejects_wrong_schema() -> None:
    selection = _mlx_selection(schema="some_other_schema.v1")

    with pytest.raises(InverseSteganalysisAcquisitionError, match="selection schema"):
        inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(selection)


def test_mlx_effective_spend_triage_selection_bridge_requires_strict_gates() -> None:
    selection = _mlx_selection()
    gates = selection["gates"]
    assert isinstance(gates, dict)
    gates["score_calibration_status"] = "blocked"

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="score calibration gate must be strict_pass",
    ):
        inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(selection)


def test_mlx_effective_spend_triage_selection_bridge_rejects_truthy_authority() -> None:
    selection = _mlx_selection()
    rows = selection["selected_rows"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    rows[0]["ready_for_exact_eval_dispatch"] = True

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="ready_for_exact_eval_dispatch=truthy",
    ):
        inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(selection)


def test_mlx_effective_spend_triage_selection_bridge_rejects_bad_geometry() -> None:
    selection = _mlx_selection()
    rows = selection["selected_rows"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    rows[0]["pair_indices"] = [10]

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="pair_indices must match source_pair_window",
    ):
        inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(selection)


def test_mlx_effective_spend_triage_selection_bridge_requires_denominator() -> None:
    selection = _mlx_selection()
    rows = selection["selected_rows"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    rows[0]["full_video_denominator"] = 599

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="full_video_denominator must be 600",
    ):
        inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(selection)


def test_mlx_effective_spend_triage_selection_bridge_rejects_bad_sha() -> None:
    selection = _mlx_selection()
    rows = selection["selected_rows"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    rows[0]["source_candidate_cache_array_sha256"] = {"pair_indices": "not-sha"}

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match=r"source_candidate_cache_array_sha256\.pair_indices must be sha256 hex",
    ):
        inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(selection)


def test_mlx_effective_spend_triage_selection_bridge_rejects_bad_normalization() -> None:
    selection = _mlx_selection()
    rows = selection["selected_rows"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    rows[0]["normalized_full_video_scorer_gain_vs_baseline"] = 0.001

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="normalized gain inconsistent with full_video_denominator",
    ):
        inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(selection)


def test_queue_performance_observations_calibrate_acquisition_denominator() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {
        "cache_sha256": "e" * 64,
    }
    performance = {
        "schema": "experiment_queue_performance_summary.v1",
        "queue_id": "byte_shave_queue",
        "event_count": 2,
        "candidate_id_by_experiment": {"candidate_a": ["candidate_a"]},
        "by_resource_kind": {},
        "by_step": {
            "candidate_a.materialize": {
                "run_count": 2,
                "success_count": 2,
                "failure_count": 0,
                "resource_kind_counts": {"local_mlx": 2},
                "dominant_resource_kind": "local_mlx",
                "elapsed_seconds_mean": 3.5,
                "artifact_record_count": 4,
                "artifact_record_bytes_mean": 2_250_000.1,
                "artifact_record_raw_bytes_mean": 9_000_000.0,
            }
        },
    }

    observations = observations_from_queue_performance_summary(
        performance,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        source_path="configs/experiment_queues/byte_shave.yaml",
    )
    plan = build_inverse_steganalysis_acquisition_plan(
        [
            _atom(
                "candidate_a",
                predicted_score_gain=0.001,
                elapsed_seconds=99.0,
                artifact_bytes=99_000_000,
                resource_kind="local_cpu",
            )
        ],
        observations=observations,
    )
    top = plan["ranked_atoms"][0]

    assert observations[0]["schema"] == OBSERVATION_SCHEMA
    assert observations[0]["observation_kind"] == "queue_performance_step"
    assert observations[0]["observed_score_gain"] is None
    assert observations[0]["queue_id"] == "byte_shave_queue"
    assert observations[0]["step_id"] == "materialize"
    assert observations[0]["run_count"] == 2
    assert observations[0]["artifact_bytes"] == 2_250_001
    assert top["best_observation_id"] == ("queue_perf_byte_shave_queue_candidate_a_materialize")
    assert top["priority"]["elapsed_seconds"] == pytest.approx(3.5)
    assert top["priority"]["artifact_bytes"] == 2_250_001
    assert top["priority"]["resource_kind"] == "local_mlx"
    assert top["score_claim"] is False

    with pytest.raises(InverseSteganalysisAcquisitionError, match="runtime_identity"):
        observations_from_queue_performance_summary(
            performance,
            runtime_identity={"note": "missing identity key"},
            cache_identity=cache_identity,
        )


def test_queue_performance_candidate_map_can_expand_bundle_steps() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {
        "cache_sha256": "e" * 64,
    }
    performance = {
        "schema": "experiment_queue_performance_summary.v1",
        "queue_id": "byte_shave_queue",
        "event_count": 1,
        "candidate_id_by_experiment": {"bundle": ["candidate_a", "candidate_b"]},
        "by_resource_kind": {},
        "by_step": {
            "bundle.materialize": {
                "run_count": 1,
                "success_count": 1,
                "failure_count": 0,
                "resource_kind_counts": {"local_mlx": 1},
                "elapsed_seconds_mean": 4.25,
                "artifact_record_bytes_mean": 8192.0,
            }
        },
    }

    observations = observations_from_queue_performance_summary(
        performance,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
    )
    action = build_discrete_scorer_action_functional(
        [_atom("candidate_a"), _atom("candidate_b")],
        observations=observations,
    )
    cells_by_candidate = {cell["candidate_id"]: cell for cell in action["cells"]}

    assert [observation["candidate_id"] for observation in observations] == [
        "candidate_a",
        "candidate_b",
    ]
    assert observations[0]["observation_id"] == ("queue_perf_byte_shave_queue_bundle_materialize_candidate_a")
    assert observations[1]["observation_id"] == ("queue_perf_byte_shave_queue_bundle_materialize_candidate_b")
    assert cells_by_candidate["candidate_a"]["best_observation_id"] == (observations[0]["observation_id"])
    assert cells_by_candidate["candidate_b"]["best_observation_id"] == (observations[1]["observation_id"])
    assert cells_by_candidate["candidate_a"]["priority"]["elapsed_seconds"] == pytest.approx(4.25)
    assert cells_by_candidate["candidate_b"]["priority"]["artifact_bytes"] == 8192


def test_queue_performance_observation_matches_inverse_action_source_unit() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {
        "cache_sha256": "e" * 64,
    }
    source_unit_id = "inverse_action_atom_direct_decoder_blob_0000"
    performance = {
        "schema": "experiment_queue_performance_summary.v1",
        "queue_id": "materializer_queue",
        "event_count": 1,
        "candidate_id_by_experiment": {"packetir_opset": ["packetir_opset"]},
        "by_resource_kind": {},
        "by_step": {
            "packetir_opset.materialize": {
                "run_count": 1,
                "success_count": 1,
                "failure_count": 0,
                "resource_kind_counts": {"local_cpu": 1},
                "dominant_resource_kind": "local_cpu",
                "elapsed_seconds_mean": 1.25,
                "artifact_record_bytes_mean": 4096.0,
                "source_unit_ids": [source_unit_id],
                "source_selection_ids": ["compiled_direct_selection"],
            }
        },
    }
    atom = _atom(
        "candidate_parent",
        atom_id="atom_direct",
        elapsed_seconds=99.0,
        artifact_bytes=99_000_000,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "compiled_direct_selection",
            "selected_operations": [
                {
                    "unit_id": "decoder_blob",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "archive_section": "decoder_blob",
                    "candidate_saved_bytes": 256,
                }
            ],
        },
    )

    observations = observations_from_queue_performance_summary(
        performance,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
    )
    action = build_discrete_scorer_action_functional([atom], observations=observations)
    cell = action["cells"][0]

    assert observations[0]["candidate_id"] == "packetir_opset"
    assert observations[0]["source_unit_ids"] == [source_unit_id]
    assert observations[0]["source_selection_ids"] == ["compiled_direct_selection"]
    assert cell["candidate_id"] == "candidate_parent"
    assert cell["best_observation_id"] == ("queue_perf_materializer_queue_packetir_opset_materialize")
    assert cell["priority"]["elapsed_seconds"] == pytest.approx(1.25)
    assert cell["priority"]["artifact_bytes"] == 4096


def test_queue_observation_health_blocker_suppresses_water_bucket_cell() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    observation_payload = {
        "schema": "experiment_queue_observation.v1",
        "queue_id": "materializer_queue",
        "mode": "running",
        "healthy": False,
        "blockers": ["experiment_queue_observation_blocked_steps:1"],
        "performance": {
            "schema": "experiment_queue_performance_summary.v1",
            "queue_id": "materializer_queue",
            "event_count": 1,
            "candidate_id_by_experiment": {"packetir_opset": ["candidate_parent"]},
            "by_resource_kind": {},
            "by_step": {
                "packetir_opset.materialize": {
                    "run_count": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "resource_kind_counts": {"local_cpu": 1},
                    "elapsed_seconds_mean": 1.25,
                    "artifact_record_bytes_mean": 4096.0,
                }
            },
        },
        "blocked_steps": [
            {
                "experiment_id": "packetir_opset",
                "step_id": "materialize",
                "status": "blocked",
                "resource_kind": "local_cpu",
                "expected_artifacts": [
                    {
                        "path": "candidate.json",
                        "exists": True,
                        "bytes": 4096,
                        "postcondition_passed": False,
                        "receiver_contract_satisfied": False,
                        "readiness_blockers": [
                            "runtime_consumption_proof_not_passed",
                        ],
                        "receiver_verification": {
                            "receiver_contract_satisfied": False,
                            "blockers": ["archive_section_entropy_recode_receiver_contract_not_satisfied"],
                        },
                    }
                ],
            }
        ],
        **_planning_false_authority(),
    }
    atom = _atom(
        "candidate_parent",
        uncertainty=0.0,
        calibration_error=0.0,
        fragility_penalty=0.0,
        first_order_marginal_effect=0.01,
        second_order_interaction_effect=0.0,
        predicted_score_gain=0.01,
    )

    observations = observations_from_queue_observation(
        observation_payload,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        source_path="queue_observation.json",
    )
    action = build_discrete_scorer_action_functional([atom], observations=observations)
    cell = action["cells"][0]

    assert [row["observation_kind"] for row in observations] == [
        "queue_performance_step",
        "queue_observation_health_blocker",
    ]
    assert observations[1]["observed_score_gain"] == 0.0
    assert observations[1]["calibration_error"] == 1.0
    assert observations[1]["queue_observation_blockers"] == [
        "queue_observation_blocked_step",
        "queue_observation_step_status:blocked",
        "experiment_queue_observation_blocked_steps:1",
        "queue_observation_artifact_postcondition_failed:candidate.json",
        "queue_observation_receiver_contract_unsatisfied:candidate.json",
        "queue_observation_materializer_readiness_blocker:runtime_consumption_proof_not_passed",
        "queue_observation_receiver_verification_blocker:"
        "archive_section_entropy_recode_receiver_contract_not_satisfied",
    ]
    assert observations[1]["receiver_contract_satisfied"] is False
    assert observations[1]["readiness_blockers"] == [
        "runtime_consumption_proof_not_passed",
        ("receiver_verification:archive_section_entropy_recode_receiver_contract_not_satisfied"),
    ]
    assert cell["best_observation_kind"] == "queue_observation_health_blocker"
    assert cell["priority"]["expected_score_gain"] == 0.0
    assert cell["water_bucket_selectable"] is False
    assert action["water_bucket"]["selected_count"] == 0
    assert action["score_claim"] is False


def test_queue_health_blocker_hard_blocks_uncertainty_bonus() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    observation_payload = {
        "schema": "experiment_queue_observation.v1",
        "queue_id": "materializer_queue",
        "healthy": False,
        "blockers": ["experiment_queue_observation_blocked_steps:1"],
        "performance": {
            "schema": "experiment_queue_performance_summary.v1",
            "queue_id": "materializer_queue",
            "event_count": 1,
            "candidate_id_by_experiment": {"packetir_opset": ["candidate_parent"]},
            "by_resource_kind": {},
            "by_step": {
                "packetir_opset.materialize": {
                    "run_count": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "resource_kind_counts": {"local_cpu": 1},
                    "elapsed_seconds_mean": 1.25,
                    "artifact_record_bytes_mean": 4096.0,
                }
            },
        },
        "blocked_steps": [
            {
                "experiment_id": "packetir_opset",
                "step_id": "materialize",
                "status": "blocked",
                "resource_kind": "local_cpu",
            }
        ],
        **_planning_false_authority(),
    }
    atom = _atom(
        "candidate_parent",
        uncertainty=10.0,
        calibration_error=0.0,
        fragility_penalty=0.0,
        first_order_marginal_effect=0.01,
        second_order_interaction_effect=0.0,
        predicted_score_gain=0.01,
    )

    observations = observations_from_queue_observation(
        observation_payload,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
    )
    action = build_discrete_scorer_action_functional([atom], observations=observations)
    cell = action["cells"][0]

    assert cell["best_observation_kind"] == "queue_observation_health_blocker"
    assert cell["queue_health_blocked"] is True
    assert cell["queue_health_feedback"]["blocks_water_bucket"] is True
    assert cell["queue_health_penalty_applied"] is True
    assert cell["priority"]["expected_score_gain"] == 0.0
    assert cell["priority"]["queue_health_penalty_multiplier"] == 0.0
    assert cell["water_bucket_selectable"] is False
    assert action["integral_totals"]["queue_health_blocked_cell_count"] == 1
    assert action["water_bucket"]["selected_count"] == 0


def test_queue_observation_health_blocker_matches_precise_source_unit() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    source_unit_id = "inverse_action_atom_direct_decoder_blob_0000"
    observation_payload = {
        "schema": "experiment_queue_observation.v1",
        "queue_id": "materializer_queue",
        "healthy": False,
        "blockers": ["experiment_queue_observation_blocked_steps:1"],
        "performance": {
            "schema": "experiment_queue_performance_summary.v1",
            "queue_id": "materializer_queue",
            "event_count": 1,
            "candidate_id_by_experiment": {"packetir_opset": ["candidate_parent"]},
            "source_unit_ids_by_experiment": {"packetir_opset": [source_unit_id]},
            "source_selection_ids_by_experiment": {"packetir_opset": ["compiled_direct_selection"]},
            "by_resource_kind": {},
            "by_step": {
                "packetir_opset.materialize": {
                    "run_count": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "resource_kind_counts": {"local_cpu": 1},
                    "elapsed_seconds_mean": 1.25,
                    "artifact_record_bytes_mean": 4096.0,
                    "source_unit_ids": [source_unit_id],
                    "source_selection_ids": ["compiled_direct_selection"],
                }
            },
        },
        "blocked_steps": [
            {
                "experiment_id": "packetir_opset",
                "step_id": "materialize",
                "status": "blocked",
                "resource_kind": "local_cpu",
            }
        ],
        **_planning_false_authority(),
    }
    matching_atom = _atom(
        "candidate_parent",
        atom_id="atom_direct",
        uncertainty=0.0,
        fragility_penalty=0.0,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "compiled_direct_selection",
            "selected_operations": [
                {
                    "unit_id": "decoder_blob",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "archive_section": "decoder_blob",
                    "candidate_saved_bytes": 256,
                }
            ],
        },
    )
    sibling_atom = _atom(
        "candidate_parent",
        atom_id="atom_other",
        uncertainty=0.0,
        fragility_penalty=0.0,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "other_selection",
            "selected_operations": [
                {
                    "unit_id": "other_blob",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "archive_section": "other_blob",
                    "candidate_saved_bytes": 128,
                }
            ],
        },
    )

    observations = observations_from_queue_observation(
        observation_payload,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
    )
    action = build_discrete_scorer_action_functional(
        [matching_atom, sibling_atom],
        observations=observations,
    )
    cells = {cell["atom_id"]: cell for cell in action["cells"]}
    health = next(row for row in observations if row["observation_kind"] == "queue_observation_health_blocker")

    assert health["source_unit_ids"] == [source_unit_id]
    assert cells["atom_direct"]["best_observation_kind"] == ("queue_observation_health_blocker")
    assert cells["atom_direct"]["priority"]["expected_score_gain"] == 0.0
    assert cells["atom_other"]["best_observation_kind"] == "queue_performance_step"
    assert cells["atom_other"]["priority"]["expected_score_gain"] > 0.0


def test_queue_observation_receiver_negative_materializer_artifact_blocks_water_bucket() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    source_unit_id = "inverse_action_atom_direct_decoder_blob_0000"
    observation_payload = {
        "schema": "experiment_queue_observation.v1",
        "queue_id": "materializer_queue",
        "healthy": False,
        "blockers": [
            "experiment_queue_observation_failed_steps:1",
            "experiment_queue_observation_artifact_postcondition_failures:1",
        ],
        "performance": {
            "schema": "experiment_queue_performance_summary.v1",
            "queue_id": "materializer_queue",
            "event_count": 1,
            "candidate_id_by_experiment": {"packetir_opset": ["candidate_parent"]},
            "source_unit_ids_by_experiment": {"packetir_opset": [source_unit_id]},
            "source_selection_ids_by_experiment": {"packetir_opset": ["compiled_direct_selection"]},
            "by_resource_kind": {},
            "by_step": {
                "packetir_opset.materialize": {
                    "run_count": 1,
                    "success_count": 0,
                    "failure_count": 1,
                    "resource_kind_counts": {"local_cpu": 1},
                    "elapsed_seconds_mean": 1.25,
                    "artifact_record_bytes_mean": 4096.0,
                    "source_unit_ids": [source_unit_id],
                    "source_selection_ids": ["compiled_direct_selection"],
                }
            },
        },
        "failed_steps": [
            {
                "experiment_id": "packetir_opset",
                "step_id": "materialize",
                "status": "failed",
                "resource_kind": "local_cpu",
                "target_kind": "archive_section_entropy_recode_v1",
                "materializer_id": "archive_section_entropy_recode_adapter",
                "receiver_contract_kind": "family_agnostic_archive_section_entropy_recode",
                "expected_artifacts": [
                    {
                        "path": "candidate_manifest.json",
                        "exists": True,
                        "bytes": 4096,
                        "postcondition_passed": False,
                        "candidate_id": "candidate_parent",
                        "target_kind": "archive_section_entropy_recode_v1",
                        "materializer_id": "archive_section_entropy_recode_adapter",
                        "receiver_contract_kind": ("family_agnostic_archive_section_entropy_recode"),
                        "receiver_contract_satisfied": False,
                        "readiness_blockers": ["section_length_changed_requires_runtime_consumption_proof"],
                        "serialized_archive_delta_status": "realized_saving",
                        "serialized_archive_delta_realized_saved_bytes": 64,
                        "serialized_archive_delta_savings_realized": True,
                        "serialized_archive_delta_source_archive_bytes": 1024,
                        "serialized_archive_delta_candidate_archive_bytes": 960,
                        "candidate_archive": {
                            "bytes": 960,
                            "sha256": "f" * 64,
                        },
                    },
                    {
                        "path": "candidate_manifest.json",
                        "exists": True,
                        "bytes": 4096,
                        "postcondition_passed": True,
                        "candidate_id": "candidate_parent",
                        "target_kind": "archive_section_entropy_recode_v1",
                        "materializer_id": "archive_section_entropy_recode_adapter",
                        "receiver_contract_kind": ("family_agnostic_archive_section_entropy_recode"),
                        "receiver_contract_satisfied": False,
                        "receiver_verification": {
                            "receiver_contract_satisfied": False,
                            "blockers": ["runtime_consumption_proof_not_passed"],
                        },
                        "candidate_archive": {
                            "bytes": 960,
                            "sha256": "f" * 64,
                        },
                    },
                ],
            }
        ],
        **_planning_false_authority(),
    }
    atom = _atom(
        "candidate_parent",
        atom_id="atom_direct",
        uncertainty=0.0,
        fragility_penalty=0.0,
        first_order_marginal_effect=0.01,
        second_order_interaction_effect=0.0,
        predicted_score_gain=0.01,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "compiled_direct_selection",
            "selected_operations": [
                {
                    "unit_id": "decoder_blob",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "materializer": "archive_section_entropy_recode_adapter",
                    "receiver_contract_kind": ("family_agnostic_archive_section_entropy_recode"),
                }
            ],
        },
    )

    observations = observations_from_queue_observation(
        observation_payload,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        source_path="queue_observation.json",
    )
    action = build_discrete_scorer_action_functional([atom], observations=observations)
    cell = action["cells"][0]
    materializer_observations = [
        row for row in observations if row["observation_kind"] == MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND
    ]
    assert len(materializer_observations) == 1
    materializer_observation = materializer_observations[0]

    assert materializer_observation["receiver_contract_satisfied"] is False
    assert materializer_observation["rate_positive"] is True
    assert materializer_observation["saved_bytes"] == 64
    assert materializer_observation["source_unit_ids"] == [source_unit_id]
    assert "receiver_contract_not_satisfied" in materializer_observation["readiness_blockers"]
    assert (
        "receiver_verification:runtime_consumption_proof_not_passed" in materializer_observation["readiness_blockers"]
    )
    assert cell["materializer_archive_delta_blocked"] is True
    assert cell["water_bucket_selectable"] is False
    assert cell["priority"]["expected_score_gain"] == 0.0
    assert "receiver_negative_materializer_success" in (cell["materializer_archive_delta_feedback"]["blockers"])
    assert "rate_negative_materializer_success" not in (cell["materializer_archive_delta_feedback"]["blockers"])
    assert cell["materializer_archive_delta_feedback"]["realized_saved_bytes_sum"] == 64
    assert action["integral_totals"]["materializer_archive_delta_blocked_cell_count"] == 1
    assert action["score_claim"] is False


def test_queue_materializer_delta_reads_family_local_archive_delta_fields() -> None:
    observations = observations_from_queue_observation(
        {
            "schema": "experiment_queue_observation.v1",
            "queue_id": "materializer_queue",
            "healthy": True,
            "failed_steps": [
                {
                    "experiment_id": "packet_recompress",
                    "step_id": "materialize",
                    "resource_kind": "local_cpu",
                    "target_kind": "packet_member_recompress_v1",
                    "materializer_id": "packet_member_recompress_adapter",
                    "receiver_contract_kind": ("family_agnostic_packet_member_recompress"),
                    "expected_artifacts": [
                        {
                            "path": "packet_recompress_manifest.json",
                            "exists": True,
                            "candidate_id": "candidate_packet",
                            "target_kind": "packet_member_recompress_v1",
                            "materializer_id": "packet_member_recompress_adapter",
                            "receiver_contract_kind": ("family_agnostic_packet_member_recompress"),
                            "receiver_contract_satisfied": True,
                            "serialized_archive_delta_status": "realized_saving",
                            "selected_compression_saved_bytes": 75,
                            "selected_compression_source_archive_bytes": 500,
                            "selected_compression_candidate_archive_bytes": 425,
                            "candidate_archive": {
                                "bytes": 425,
                                "sha256": "f" * 64,
                            },
                        }
                    ],
                }
            ],
            **_planning_false_authority(),
        },
        runtime_identity={"runtime_tree_sha256": "d" * 64},
        cache_identity={"cache_sha256": "e" * 64},
        source_path="queue_observation.json",
    )

    materializer_observations = [
        row for row in observations if row["observation_kind"] == MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND
    ]
    assert len(materializer_observations) == 1
    row = materializer_observations[0]
    assert row["candidate_id"] == "candidate_packet"
    assert row["saved_bytes"] == 75
    assert row["source_archive_bytes"] == 500
    assert row["candidate_archive_bytes"] == 425
    assert row["rate_positive"] is True
    assert row["receiver_contract_satisfied"] is True
    assert row["score_claim"] is False


def test_queue_observation_reads_succeeded_materializer_observation_jsonl(
    tmp_path,
) -> None:
    observation_jsonl = tmp_path / "observations.jsonl"
    observation_jsonl.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_materializer_empirical_observation.v1",
                "observation_kind": ("family_agnostic_materializer_empirical_observation"),
                "observation_id": "obs_packet_recompress_success",
                "candidate_id": "sweep_packet_recompress_success",
                "axis": "[local-materializer-proof]",
                "runtime_identity": {
                    "runtime_contract_sha256": "a" * 64,
                    "scorer_version": ("family_agnostic_materializer_empirical_sweep.v1"),
                },
                "cache_identity": {"cache_sha256": "b" * 64},
                "target_kind": "packet_member_recompress_v1",
                "materializer_id": "packet_member_recompress_adapter",
                "receiver_contract_kind": ("family_agnostic_packet_member_recompress"),
                "saved_bytes": 75,
                "observed_rate_gain": CONTEST_RATE_SCORE_PER_BYTE * 75,
                "observed_score_gain": CONTEST_RATE_SCORE_PER_BYTE * 75,
                "artifact_bytes": 425,
                "resource_kind": "local_cpu",
                "rate_positive": True,
                "receiver_contract_satisfied": True,
                "candidate_archive_sha256": "f" * 64,
                **_planning_false_authority(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observations = observations_from_queue_observation(
        {
            "schema": "experiment_queue_observation.v1",
            "queue_id": "materializer_queue",
            "healthy": True,
            "succeeded_artifact_steps": [
                {
                    "experiment_id": "packet_recompress",
                    "step_id": "materialize",
                    "status": "succeeded",
                    "resource_kind": "local_cpu",
                    "candidate_ids": ["candidate_packet"],
                    "source_unit_ids": ["packet_payload_bin"],
                    "source_selection_ids": ["selection_packet_payload_bin"],
                    "expected_artifacts": [
                        {
                            "path": str(observation_jsonl),
                            "exists": True,
                            "postcondition_type": "jsonl_false_authority",
                        }
                    ],
                }
            ],
            **_planning_false_authority(),
        },
        runtime_identity={"runtime_tree_sha256": "d" * 64},
        cache_identity={"cache_sha256": "e" * 64},
        source_path=str(tmp_path / "queue_observation.json"),
    )
    atom = _atom(
        "candidate_packet",
        atom_id="atom_packet_recompress",
        fragility_penalty=0.0,
        uncertainty=0.0,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "packet_recompress_group",
            "selected_operations": [
                {
                    "unit_id": "packet_payload_bin",
                    "target_kind": "packet_member_recompress_v1",
                    "materializer": "packet_member_recompress_adapter",
                    "receiver_contract_kind": ("family_agnostic_packet_member_recompress"),
                    "candidate_saved_bytes": 75,
                }
            ],
        },
    )
    action = build_discrete_scorer_action_functional([atom], observations=observations)
    materializer_observations = [
        row for row in observations if row["observation_kind"] == "family_agnostic_materializer_empirical_observation"
    ]

    assert len(materializer_observations) == 1
    row = materializer_observations[0]
    assert row["candidate_id"] == "sweep_packet_recompress_success"
    assert row["candidate_ids"] == [
        "sweep_packet_recompress_success",
        "candidate_packet",
    ]
    assert row["source_unit_ids"] == ["packet_payload_bin"]
    assert row["source_selection_ids"] == ["selection_packet_payload_bin"]
    assert row["expected_artifact_paths"] == [str(observation_jsonl)]
    assert row["saved_bytes"] == 75
    assert row["rate_positive"] is True
    assert row["receiver_contract_satisfied"] is True
    assert action["cells"][0]["materializer_archive_delta_feedback"]["rate_positive_count"] == 1
    assert action["score_claim"] is False


def test_queue_observation_deduplicates_materializer_sweep_json_and_jsonl(
    tmp_path,
) -> None:
    materializer_row = {
        "schema": "family_agnostic_materializer_empirical_observation.v1",
        "observation_kind": "family_agnostic_materializer_empirical_observation",
        "observation_id": "obs_packet_recompress_success",
        "candidate_id": "sweep_packet_recompress_success",
        "axis": "[local-materializer-proof]",
        "target_kind": "packet_member_recompress_v1",
        "materializer_id": "packet_member_recompress_adapter",
        "receiver_contract_kind": "family_agnostic_packet_member_recompress",
        "saved_bytes": 75,
        "observed_rate_gain": CONTEST_RATE_SCORE_PER_BYTE * 75,
        "observed_score_gain": CONTEST_RATE_SCORE_PER_BYTE * 75,
        "artifact_bytes": 425,
        "resource_kind": "local_cpu",
        "rate_positive": True,
        "receiver_contract_satisfied": True,
        "candidate_archive_sha256": "f" * 64,
        **_planning_false_authority(),
    }
    sweep_json = tmp_path / "sweep.json"
    observations_jsonl = tmp_path / "observations.jsonl"
    sweep_json.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_materializer_empirical_sweep.v1",
                "observations": [materializer_row],
                **_planning_false_authority(),
            }
        ),
        encoding="utf-8",
    )
    observations_jsonl.write_text(json.dumps(materializer_row) + "\n", encoding="utf-8")

    observations = observations_from_queue_observation(
        {
            "schema": "experiment_queue_observation.v1",
            "queue_id": "materializer_queue",
            "healthy": True,
            "succeeded_artifact_steps": [
                {
                    "experiment_id": "packet_recompress",
                    "step_id": "materialize",
                    "status": "succeeded",
                    "resource_kind": "local_cpu",
                    "candidate_ids": ["candidate_packet"],
                    "source_unit_ids": ["packet_payload_bin"],
                    "source_selection_ids": ["selection_packet_payload_bin"],
                    "expected_artifacts": [
                        {"path": str(sweep_json), "exists": True},
                        {
                            "path": str(observations_jsonl),
                            "exists": True,
                            "postcondition_type": "jsonl_false_authority",
                        },
                    ],
                }
            ],
            **_planning_false_authority(),
        },
        runtime_identity={"runtime_tree_sha256": "d" * 64},
        cache_identity={"cache_sha256": "e" * 64},
        source_path=str(tmp_path / "queue_observation.json"),
    )

    materializer_observations = [
        row for row in observations if row["observation_kind"] == "family_agnostic_materializer_empirical_observation"
    ]
    assert len(materializer_observations) == 1
    row = materializer_observations[0]
    assert row["candidate_id"] == "sweep_packet_recompress_success"
    assert row["saved_bytes"] == 75
    assert row["expected_artifact_paths"] == [
        str(sweep_json),
        str(observations_jsonl),
    ]


def test_queue_observation_rejects_truthy_authority_in_materializer_jsonl(
    tmp_path,
) -> None:
    observation_jsonl = tmp_path / "observations.jsonl"
    observation_jsonl.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_materializer_empirical_observation.v1",
                "observation_id": "obs_bad_authority",
                "candidate_id": "bad_authority",
                "axis": "[local-materializer-proof]",
                "target_kind": "packet_member_recompress_v1",
                "materializer_id": "packet_member_recompress_adapter",
                "saved_bytes": 1,
                "artifact_bytes": 1,
                **_planning_false_authority(),
                "score_claim": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match=r"queue materializer observation artifact .*score_claim",
    ):
        observations_from_queue_observation(
            {
                "schema": "experiment_queue_observation.v1",
                "queue_id": "materializer_queue",
                "healthy": True,
                "succeeded_artifact_steps": [
                    {
                        "experiment_id": "packet_recompress",
                        "step_id": "materialize",
                        "status": "succeeded",
                        "candidate_ids": ["candidate_packet"],
                        "expected_artifacts": [
                            {
                                "path": str(observation_jsonl),
                                "exists": True,
                                "postcondition_type": "jsonl_false_authority",
                            }
                        ],
                    }
                ],
                **_planning_false_authority(),
            },
            runtime_identity={"runtime_tree_sha256": "d" * 64},
            cache_identity={"cache_sha256": "e" * 64},
            source_path=str(tmp_path / "queue_observation.json"),
        )


def test_queue_health_feedback_groups_repeated_blockers_by_source_selection() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    source_unit_id = "inverse_action_atom_direct_decoder_blob_0000"
    observation_payload = {
        "schema": "experiment_queue_observation.v1",
        "queue_id": "materializer_queue",
        "healthy": False,
        "blockers": ["experiment_queue_observation_blocked_steps:2"],
        "performance": {
            "schema": "experiment_queue_performance_summary.v1",
            "queue_id": "materializer_queue",
            "event_count": 2,
            "candidate_id_by_experiment": {"packetir_opset": ["candidate_parent"]},
            "source_unit_ids_by_experiment": {"packetir_opset": [source_unit_id]},
            "source_selection_ids_by_experiment": {"packetir_opset": ["compiled_direct_selection"]},
            "by_resource_kind": {},
            "by_step": {
                "packetir_opset.materialize": {
                    "run_count": 1,
                    "success_count": 0,
                    "failure_count": 1,
                    "resource_kind_counts": {"local_cpu": 1},
                    "elapsed_seconds_mean": 1.25,
                    "artifact_record_bytes_mean": 4096.0,
                    "source_unit_ids": [source_unit_id],
                    "source_selection_ids": ["compiled_direct_selection"],
                },
                "packetir_opset.proof": {
                    "run_count": 1,
                    "success_count": 0,
                    "failure_count": 1,
                    "resource_kind_counts": {"local_cpu": 1},
                    "elapsed_seconds_mean": 0.75,
                    "artifact_record_bytes_mean": 1024.0,
                    "source_unit_ids": [source_unit_id],
                    "source_selection_ids": ["compiled_direct_selection"],
                },
            },
        },
        "blocked_steps": [
            {
                "experiment_id": "packetir_opset",
                "step_id": "materialize",
                "status": "blocked",
                "resource_kind": "local_cpu",
            },
            {
                "experiment_id": "packetir_opset",
                "step_id": "proof",
                "status": "blocked",
                "resource_kind": "local_cpu",
            },
        ],
        **_planning_false_authority(),
    }
    atom = _atom(
        "candidate_parent",
        atom_id="atom_direct",
        uncertainty=0.0,
        fragility_penalty=0.0,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "compiled_direct_selection",
            "selected_operations": [
                {
                    "unit_id": "decoder_blob",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "archive_section": "decoder_blob",
                    "candidate_saved_bytes": 256,
                }
            ],
        },
    )

    observations = observations_from_queue_observation(
        observation_payload,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
    )
    action = build_discrete_scorer_action_functional([atom], observations=observations)
    cell = action["cells"][0]
    root_feedback = action["queue_health_feedback"]
    cell_feedback = cell["queue_health_feedback"]

    assert action["observation_feedback"]["queue_health_group_priors"] == root_feedback
    assert root_feedback["observation_count"] == 2
    assert root_feedback["repeated_group_count"] == 1
    assert root_feedback["groups"][0]["group_key"] == ("source_selection_ids:compiled_direct_selection")
    assert root_feedback["groups"][0]["observation_count"] == 2
    assert root_feedback["groups"][0]["hard_blocking_observation_count"] == 2
    assert cell_feedback["groups"] == root_feedback["groups"]
    assert cell_feedback["source_unit_ids"] == [source_unit_id]
    assert cell_feedback["source_selection_ids"] == ["compiled_direct_selection"]
    assert cell["queue_health_group_ids"] == ["source_selection_ids:compiled_direct_selection"]
    assert cell["queue_health_repeat_count"] == 2
    assert cell["queue_health_penalty_applied"] is True
    assert root_feedback["score_claim"] is False
    assert root_feedback["promotion_eligible"] is False
    assert cell["water_bucket_selectable"] is False


def test_repeated_queue_health_group_penalizes_related_materializer_only() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    observation_payload = {
        "schema": "experiment_queue_observation.v1",
        "queue_id": "materializer_queue",
        "healthy": False,
        "blockers": ["experiment_queue_observation_blocked_steps:2"],
        "performance": {
            "schema": "experiment_queue_performance_summary.v1",
            "queue_id": "materializer_queue",
            "event_count": 2,
            "candidate_id_by_experiment": {"failed_opset": ["failed_candidate"]},
            "by_resource_kind": {},
            "by_step": {
                "failed_opset.materialize": {
                    "run_count": 1,
                    "success_count": 0,
                    "failure_count": 1,
                    "resource_kind_counts": {"local_cpu": 1},
                    "elapsed_seconds_mean": 1.25,
                    "artifact_record_bytes_mean": 4096.0,
                },
                "failed_opset.proof": {
                    "run_count": 1,
                    "success_count": 0,
                    "failure_count": 1,
                    "resource_kind_counts": {"local_cpu": 1},
                    "elapsed_seconds_mean": 0.75,
                    "artifact_record_bytes_mean": 1024.0,
                },
            },
        },
        "blocked_steps": [
            {
                "experiment_id": "failed_opset",
                "step_id": "materialize",
                "status": "blocked",
                "resource_kind": "local_cpu",
                "target_kind": "archive_section_entropy_recode_v1",
                "materializer_id": "entropy_adapter",
            },
            {
                "experiment_id": "failed_opset",
                "step_id": "proof",
                "status": "blocked",
                "resource_kind": "local_cpu",
                "target_kind": "archive_section_entropy_recode_v1",
                "materializer_id": "entropy_adapter",
            },
        ],
        **_planning_false_authority(),
    }
    related_atom = _atom(
        "new_candidate",
        atom_id="atom_related",
        uncertainty=0.0,
        fragility_penalty=0.0,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "new_entropy_selection",
            "selected_operations": [
                {
                    "unit_id": "decoder_blob",
                    "target_kind": "archive_section_entropy_recode_v1",
                    "materializer_id": "entropy_adapter",
                    "candidate_saved_bytes": 256,
                }
            ],
        },
    )
    unrelated_atom = _atom(
        "other_candidate",
        atom_id="atom_unrelated",
        uncertainty=0.0,
        fragility_penalty=0.0,
        operation_set_compiler={
            "schema": "inverse_action_operation_set_compiler_hint.v1",
            "operation_set_id": "other_selection",
            "selected_operations": [
                {
                    "unit_id": "sidecar_member",
                    "target_kind": "packet_member_recompress_v1",
                    "materializer_id": "zip_adapter",
                    "candidate_saved_bytes": 128,
                }
            ],
        },
    )

    observations = observations_from_queue_observation(
        observation_payload,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
    )
    action = build_discrete_scorer_action_functional(
        [related_atom, unrelated_atom],
        observations=observations,
    )
    cells = {cell["atom_id"]: cell for cell in action["cells"]}
    related = cells["atom_related"]
    unrelated = cells["atom_unrelated"]

    assert related["queue_health_group_ids"] == [
        "materializer_target:entropy_adapter:archive_section_entropy_recode_v1"
    ]
    assert related["queue_health_blocked"] is False
    assert related["queue_health_feedback"]["blocks_water_bucket"] is False
    assert related["queue_health_penalty_applied"] is True
    assert related["priority"]["queue_health_penalty_multiplier"] == pytest.approx(1.0 / 3.0)
    assert unrelated["queue_health_group_ids"] == []
    assert unrelated["queue_health_penalty_applied"] is False
    assert unrelated["priority"]["expected_score_gain"] > related["priority"]["expected_score_gain"]
    assert unrelated["water_bucket_selectable"] is True


def test_queue_observation_global_blocker_suppresses_mapped_candidates() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    observation_payload = {
        "schema": "experiment_queue_observation.v1",
        "queue_id": "materializer_queue",
        "healthy": False,
        "blockers": ["experiment_queue_observation_state_missing"],
        "performance": {
            "schema": "experiment_queue_performance_summary.v1",
            "queue_id": "materializer_queue",
            "event_count": 1,
            "candidate_id_by_experiment": {"candidate_a": ["candidate_a"]},
            "by_resource_kind": {},
            "by_step": {
                "candidate_a.materialize": {
                    "run_count": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "resource_kind_counts": {"local_cpu": 1},
                    "elapsed_seconds_mean": 1.25,
                    "artifact_record_bytes_mean": 4096.0,
                }
            },
        },
        **_planning_false_authority(),
    }

    observations = observations_from_queue_observation(
        observation_payload,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
    )
    action = build_discrete_scorer_action_functional(
        [_atom("candidate_a", uncertainty=0.0, fragility_penalty=0.0)],
        observations=observations,
    )

    assert observations[-1]["observation_kind"] == ("queue_observation_global_health_blocker")
    assert observations[-1]["queue_observation_blockers"] == ["experiment_queue_observation_state_missing"]
    assert action["cells"][0]["best_observation_kind"] == ("queue_observation_global_health_blocker")
    assert action["cells"][0]["priority"]["expected_score_gain"] == 0.0


def test_queue_observation_skips_nonblocking_orphan_without_candidate_identity() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    observation_payload = {
        "schema": "experiment_queue_observation.v1",
        "queue_id": "materializer_queue",
        "healthy": False,
        "blockers": ["experiment_queue_observation_orphaned_steps:1"],
        "orphaned_steps": [
            {
                "experiment_id": "orphaned_materializer",
                "step_id": "materialize",
                "status": "skipped",
            }
        ],
        **_planning_false_authority(),
    }

    assert (
        observations_from_queue_observation(
            observation_payload,
            runtime_identity=runtime_identity,
            cache_identity=cache_identity,
        )
        == []
    )


def test_queue_observation_blocking_orphan_requires_candidate_identity() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {"cache_sha256": "e" * 64}
    observation_payload = {
        "schema": "experiment_queue_observation.v1",
        "queue_id": "materializer_queue",
        "healthy": False,
        "blockers": ["experiment_queue_observation_orphaned_steps:1"],
        "orphaned_steps": [
            {
                "experiment_id": "orphaned_materializer",
                "step_id": "materialize",
                "status": "running",
            }
        ],
        **_planning_false_authority(),
    }

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="missing candidate identity",
    ):
        observations_from_queue_observation(
            observation_payload,
            runtime_identity=runtime_identity,
            cache_identity=cache_identity,
        )


def test_queue_performance_summary_requires_candidate_identity() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {
        "cache_sha256": "e" * 64,
    }
    legacy_summary = {
        "schema": "experiment_queue_performance_summary.v1",
        "queue_id": "legacy_queue",
        "event_count": 1,
        "by_resource_kind": {},
        "by_step": {
            "anonymous.materialize": {
                "run_count": 1,
                "success_count": 1,
                "failure_count": 0,
                "elapsed_seconds_mean": 1.0,
            }
        },
    }

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="missing candidate_id_by_experiment",
    ):
        observations_from_queue_performance_summary(
            legacy_summary,
            runtime_identity=runtime_identity,
            cache_identity=cache_identity,
        )

    observations = observations_from_queue_performance_summary(
        legacy_summary,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        candidate_id_by_experiment={"anonymous": "candidate_a"},
    )

    assert observations[0]["candidate_id"] == "candidate_a"


def test_queue_performance_observations_do_not_override_scorer_observations() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {
        "cache_sha256": "e" * 64,
    }
    queue_observations = observations_from_queue_performance_summary(
        {
            "schema": "experiment_queue_performance_summary.v1",
            "queue_id": "byte_shave_queue",
            "event_count": 1,
            "candidate_id_by_experiment": {"candidate_a": ["candidate_a"]},
            "by_resource_kind": {},
            "by_step": {
                "candidate_a.materialize": {
                    "run_count": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "resource_kind_counts": {"local_mlx": 1},
                    "elapsed_seconds_mean": 1.0,
                    "artifact_record_bytes_mean": 1.0,
                }
            },
        },
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
    )
    scorer_observation = _observation(
        "candidate_a",
        observation_id="obs_scorer_candidate_a",
        observed_score_gain=0.0002,
        elapsed_seconds=30.0,
        artifact_bytes=5_000_000,
        resource_kind="local_cpu",
    )

    plan = build_inverse_steganalysis_acquisition_plan(
        [_atom("candidate_a", predicted_score_gain=0.001)],
        observations=[*queue_observations, scorer_observation],
    )

    assert plan["ranked_atoms"][0]["best_observation_id"] == "obs_scorer_candidate_a"
    assert plan["ranked_atoms"][0]["priority"]["elapsed_seconds"] == pytest.approx(30.0)
    assert plan["ranked_atoms"][0]["priority"]["resource_kind"] == "local_cpu"


def test_paired_exact_auth_calibration_demotes_regressed_measured_config() -> None:
    observations = paired_exact_auth_calibration_observations_from_review_packets(
        [
            _review_packet(
                "contest_cpu",
                score=0.19380912393883232,
                baseline_score=0.192028283,
                inflated_output_aggregate_sha256="a" * 64,
            ),
            _review_packet(
                "contest_cuda",
                score=0.2279696105246996,
                baseline_score=0.205330029,
                inflated_output_aggregate_sha256="b" * 64,
            ),
        ],
        candidate_id="ias1_runtime_parity_top4",
        packet_paths=[
            ".omx/research/ias1_cpu_result_review.json",
            ".omx/research/ias1_cuda_result_review.json",
        ],
    )
    scorer_observation = _observation(
        "ias1_runtime_parity_top4",
        observed_score_gain=0.01,
        calibration_error=0.0,
    )

    action = build_discrete_scorer_action_functional(
        [
            _atom(
                "ias1_runtime_parity_top4",
                predicted_score_gain=0.005,
                first_order_marginal_effect=0.005,
                uncertainty=0.0,
            )
        ],
        observations=[scorer_observation, *observations],
        total_byte_budget=1024,
    )
    cell = action["cells"][0]
    calibration = observations[0]["exact_auth_calibration"]

    assert observations[0]["observation_kind"] == "paired_exact_auth_calibration"
    assert observations[0]["observed_score_gain"] == 0.0
    assert observations[0]["calibration_error"] == pytest.approx(
        (0.19380912393883232 - 0.192028283) + (0.2279696105246996 - 0.205330029)
    )
    assert calibration["pair_status"] == ("paired_exact_auth_regressed_vs_axis_baselines")
    assert calibration["axis_rows"]["contest_cpu"]["score_delta_status"] == ("regresses_vs_axis_baseline")
    assert calibration["axis_rows"]["contest_cuda"]["score_delta_status"] == ("regresses_vs_axis_baseline")
    assert calibration["score_claim"] is False
    assert calibration["promotion_eligible"] is False
    assert calibration["rank_or_kill_eligible"] is False
    assert calibration["ready_for_exact_eval_dispatch"] is False
    assert calibration.get("family_falsified") is not True
    assert calibration.get("method_family_retired") is not True
    assert cell["best_observation_id"] == observations[0]["observation_id"]
    assert cell["best_observation_kind"] == "paired_exact_auth_calibration"
    assert cell["exact_auth_calibration"]["pair_status"] == ("paired_exact_auth_regressed_vs_axis_baselines")
    assert cell["priority"]["expected_score_gain"] == 0.0
    assert action["observation_feedback"]["exact_auth_calibration_count"] == 1
    assert action["water_bucket"]["selected_count"] == 0
    assert action["score_claim"] is False
    assert action["promotion_eligible"] is False
    assert action["rank_or_kill_eligible"] is False
    assert action["ready_for_exact_eval_dispatch"] is False
    assert action.get("family_falsified") is not True
    assert action.get("method_family_retired") is not True


def test_paired_exact_auth_calibration_requires_shared_archive_custody() -> None:
    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="share archive_sha256",
    ):
        paired_exact_auth_calibration_observations_from_review_packets(
            [
                _review_packet(
                    "contest_cpu",
                    score=0.193,
                    baseline_score=0.192,
                    archive_sha256="a" * 64,
                ),
                _review_packet(
                    "contest_cuda",
                    score=0.228,
                    baseline_score=0.205,
                    archive_sha256="b" * 64,
                ),
            ],
            candidate_id="ias1_runtime_parity_top4",
        )


def test_paired_exact_auth_calibration_refuses_family_retirement_authority() -> None:
    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="must not retire a family",
    ):
        paired_exact_auth_calibration_observations_from_review_packets(
            [
                _review_packet(
                    "contest_cpu",
                    score=0.193,
                    baseline_score=0.192,
                    family_falsified=True,
                ),
                _review_packet(
                    "contest_cuda",
                    score=0.228,
                    baseline_score=0.205,
                ),
            ],
            candidate_id="ias1_runtime_parity_top4",
        )


def test_discrete_action_functional_water_fills_positive_euler_cells() -> None:
    atoms = [
        _atom(
            "candidate_high",
            atom_id="atom_high",
            byte_range=[0, 100],
            predicted_score_gain=0.001,
            first_order_marginal_effect=0.0009,
            second_order_interaction_effect=0.0002,
            discontinuity_risk=0.0,
        ),
        _atom(
            "candidate_low",
            atom_id="atom_low",
            byte_range=[100, 400],
            predicted_score_gain=0.00003,
            first_order_marginal_effect=0.00002,
            second_order_interaction_effect=0.0,
            discontinuity_risk=0.0,
        ),
        _atom(
            "candidate_blocked",
            atom_id="atom_blocked",
            byte_range=[400, 500],
            predicted_score_gain=0.001,
            first_order_marginal_effect=0.001,
            second_order_interaction_effect=0.0,
            discontinuity_risk=0.9,
            discontinuity_threshold=0.5,
        ),
    ]

    functional = build_discrete_scorer_action_functional(
        atoms,
        total_byte_budget=128,
        lambda_rate=CONTEST_RATE_SCORE_PER_BYTE,
    )

    assert functional["schema"] == ACTION_FUNCTIONAL_SCHEMA
    assert functional["math_model"]["representation"] == ("discrete_riemann_sum_with_second_order_interactions")
    assert functional["integral_totals"]["cell_count"] == 3
    assert functional["integral_totals"]["blocked_cell_count"] == 1
    assert functional["integral_totals"]["second_order_interaction_effect_sum"] == pytest.approx(0.0002)
    assert functional["water_bucket"]["selected_count"] == 1
    assert functional["water_bucket"]["selected_cells"][0]["atom_id"] == "atom_high"
    assert functional["water_bucket"]["selected_water_fill_cost_bytes"] == 100
    assert functional["cells"][0]["euler_lagrange_residual"] > 0
    assert all(row["atom_id"] != "atom_blocked" for row in functional["water_bucket"]["selected_cells"])
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert functional[key] is value


def test_water_bucket_uses_portfolio_search_not_scalar_greedy() -> None:
    atoms = [
        _atom(
            "greedy_single",
            atom_id="atom_greedy_single",
            byte_range=[0, 60],
            predicted_score_gain=0.00018,
            first_order_marginal_effect=0.00018,
            second_order_interaction_effect=0.0,
            uncertainty=0.0,
            calibration_error=0.0,
            fragility_penalty=0.0,
            discontinuity_risk=0.0,
        ),
        _atom(
            "portfolio_a",
            atom_id="atom_portfolio_a",
            byte_range=[60, 110],
            predicted_score_gain=0.00014,
            first_order_marginal_effect=0.00014,
            second_order_interaction_effect=0.0,
            uncertainty=0.0,
            calibration_error=0.0,
            fragility_penalty=0.0,
            discontinuity_risk=0.0,
        ),
        _atom(
            "portfolio_b",
            atom_id="atom_portfolio_b",
            byte_range=[110, 160],
            predicted_score_gain=0.00014,
            first_order_marginal_effect=0.00014,
            second_order_interaction_effect=0.0,
            uncertainty=0.0,
            calibration_error=0.0,
            fragility_penalty=0.0,
            discontinuity_risk=0.0,
        ),
    ]

    functional = build_discrete_scorer_action_functional(
        atoms,
        total_byte_budget=100,
    )
    bucket = functional["water_bucket"]

    assert bucket["selection_strategy"] == "bounded_lagrangian_portfolio_search"
    assert bucket["greedy_baseline_atom_ids"] == ["atom_greedy_single"]
    assert [row["atom_id"] for row in bucket["selected_cells"]] == [
        "atom_portfolio_a",
        "atom_portfolio_b",
    ]
    assert bucket["selected_water_fill_cost_bytes"] == 100
    assert bucket["selected_expected_score_gain"] > bucket["greedy_baseline_expected_score_gain"]
    assert bucket["selected_lagrangian_gain"] > 0.0
    assert bucket["score_claim"] is False
    assert bucket["ready_for_exact_eval_dispatch"] is False


def test_discrete_action_functional_uses_canonical_contest_rate_denominator() -> None:
    stale_fifty_million_lambda = 25.0 / 50_000_000.0
    boundary_gain = stale_fifty_million_lambda * 100.0 * 1.05
    atoms = [
        _atom(
            "boundary_candidate",
            atom_id="boundary_atom",
            byte_range=[0, 100],
            predicted_score_gain=boundary_gain,
            first_order_marginal_effect=boundary_gain,
            second_order_interaction_effect=0.0,
            uncertainty=0.0,
            discontinuity_risk=0.0,
        )
    ]

    canonical = build_discrete_scorer_action_functional(atoms)
    stale = build_discrete_scorer_action_functional(
        atoms,
        lambda_rate=stale_fifty_million_lambda,
    )

    assert CONTEST_RATE_DENOM_BYTES == CANONICAL_RATE_DENOM_BYTES == 37_545_489
    assert pytest.approx(25.0 / 37_545_489) == CONTEST_RATE_SCORE_PER_BYTE
    assert canonical["math_model"]["lambda_rate"] == pytest.approx(CONTEST_RATE_SCORE_PER_BYTE)
    assert canonical["water_bucket"]["selected_count"] == 0
    assert stale["water_bucket"]["selected_count"] == 1
