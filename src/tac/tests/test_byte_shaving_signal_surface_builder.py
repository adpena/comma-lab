# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.byte_shaving_campaign import (
    ByteShavingCampaignError,
    build_byte_shaving_campaign_plan,
)
from tac.optimization.byte_shaving_signal_surface_builder import (
    build_byte_shaving_signal_surface,
)
from tac.optimization.scorer_response_dataset import RATE_SCORE_PER_BYTE

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "build_byte_shaving_signal_surface.py"


def _candidate_queue(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_queue_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "top_k": [
                    {
                        "candidate_id": "drop_pair_0371",
                        "source_candidate_id": "trained_seed7",
                        "candidate_saved_bytes": 120,
                        "predicted_quality_score_cost": 0.00001,
                        "confidence": 0.8,
                        "operation_families": ["drop_pair"],
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _engineered_correction_targeting(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "master_gradient_consumer_engineered_correction_targeting_v1",
                "consumer_id": "engineered_correction_targeting",
                "archive_sha256": "e" * 64,
                "measurement_axis": "contest_cuda",
                "measurement_hardware": "linux_x86_64_t4",
                "n_bytes": 128,
                "n_pairs": 2,
                "targets_per_pair": 1,
                "total_targets": 2,
                "top_per_pair_targets": [
                    {
                        "pair_index": 0,
                        "byte_index": 7,
                        "per_pair_distortion_magnitude": 0.25,
                        "per_pair_variance_rank": 3,
                    },
                    {
                        "pair_index": 1,
                        "byte_index": 11,
                        "per_pair_distortion_magnitude": 0.5,
                        "per_pair_variance_rank": 1,
                    },
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        encoding="utf-8",
    )


def _auth_eval(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "canonical_score": 0.2,
                "score_recomputed_from_components": 0.2,
                "avg_posenet_dist": 0.01,
                "avg_segnet_dist": 0.001,
                "rate_unscaled": 0.004,
                "archive_size_bytes": 123,
                "n_samples": 600,
                "canonical_score_source": "score_recomputed_from_components",
                "actual_device": "cpu",
                "evidence_grade": "contest-CPU",
                "lane_tag": "[contest-CPU]",
                "score_axis": "contest_cpu",
                "evidence_semantics": "public_leaderboard_cpu_reproduction",
                "cpu_leaderboard_reproduction_eligible": True,
                "score_claim": True,
                "score_claim_valid": True,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "provenance": {
                    "actual_device": "cpu",
                    "platform_system": "Linux",
                    "platform_machine": "x86_64",
                },
            }
        ),
        encoding="utf-8",
    )


def _mlx_calibration(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "mlx_score_calibration.v1",
                "row_count": 3,
                "evidence_grade": "macOS-MLX",
                "evidence_tag": "[macOS-MLX research-signal]",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "promotable": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "summary": {"cpu_pairwise_order_accuracy": 1.0},
                "decision_policy": {
                    "allowed_use": "local_spend_triage_only_after_strict_auth_axis_calibration"
                },
            }
        ),
        encoding="utf-8",
    )


def _scorer_response_dataset(path: Path, *, rows: list[dict[str, object]]) -> None:
    false_authority = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    path.write_text(
        json.dumps(
            {
                "schema": "scorer_response_dataset.v1",
                "producer": "test",
                **false_authority,
                "authority": {
                    **false_authority,
                    "evidence_grade": "macOS-MLX research-signal",
                },
                "summary": {"row_count": len(rows)},
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def _mlx_response_row(*, projected_delta: float, raw_delta: float) -> dict[str, object]:
    false_authority = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    normalized_gain = -float(projected_delta)
    return {
        "schema": "scorer_response_row.v1",
        "row_id": "mlx-row-a",
        "candidate_id": "mlx-row-a",
        "family": "mlx_decoder_q",
        "source_schema": "mlx_scorer_response.v1",
        **false_authority,
        "authority_source_score_claim": False,
        "delta_vs_baseline_score": float(raw_delta),
        "scorer_delta_vs_baseline": float(raw_delta),
        "observed_scorer_gain_vs_baseline": normalized_gain,
        "added_archive_bytes": 0,
        "source_n_samples": 600,
        "full_video_denominator": 600,
        "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
        "projected_full_video_delta_vs_baseline_score": float(projected_delta),
        "break_even_added_bytes_from_normalized_full_video_gain": (
            normalized_gain / RATE_SCORE_PER_BYTE
        ),
        "normalized_full_video_byte_budget_margin_vs_break_even": (
            normalized_gain / RATE_SCORE_PER_BYTE
        ),
    }


def _inverse_response_row(
    *,
    projected_delta: float,
    scorer_delta: float,
    saved_bytes: int,
) -> dict[str, object]:
    false_authority = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    return {
        "schema": "scorer_response_row.v1",
        "row_id": "inverse-row-a",
        "candidate_id": "inverse-row-a",
        "family": "decoder_q",
        **false_authority,
        "authority_source_score_claim": False,
        "delta_vs_baseline_score": float(projected_delta),
        "scorer_delta_vs_baseline": float(scorer_delta),
        "observed_scorer_gain_vs_baseline": -float(scorer_delta),
        "added_archive_bytes": -int(saved_bytes),
        "byte_budget_margin_vs_break_even": float(saved_bytes),
        "source_pair_window": [7, 8],
        "diagnostic_seg_share": 0.15,
        "diagnostic_pose_share": 0.85,
    }


def _inverse_action_payload() -> dict[str, object]:
    false_authority = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }
    return {
        "schema": "inverse_steganalysis_discrete_action_functional.v1",
        "tool": "tac.optimization.inverse_steganalysis_acquisition",
        "math_model": {
            "representation": "discrete_riemann_sum_with_second_order_interactions",
            "stationarity_rule": "select positive euler_lagrange_residual cells",
            "lambda_rate": RATE_SCORE_PER_BYTE,
        },
        "integral_totals": {"cell_count": 1, "blocked_cell_count": 0},
        "water_bucket": {
            "schema": "inverse_steganalysis_water_bucket_plan.v1",
            "selected_count": 1,
            "selected_expected_score_gain": 0.0004,
            "selected_cells": [
                {
                    "atom_id": "inverse_surface_pair0007",
                    "candidate_id": "candidate_pair0007",
                    "scope_axis": "pairs",
                    "component": "posenet",
                    "water_fill_cost_bytes": 32,
                    "expected_score_gain": 0.0004,
                    "euler_lagrange_residual": 0.00039,
                }
            ],
            **false_authority,
        },
        "cells": [],
        **false_authority,
    }


def _pairset_acquisition(
    path: Path,
    *,
    truthy_authority: bool = False,
    malformed_pair_id: bool = False,
) -> None:
    false_authority = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }
    repair_budget = {
        "schema": "decoder_q_pairset_rate_saved_distortion_repair_budget.v1",
        "active": True,
        "saved_bytes_vs_source_selector": 6,
        "score_budget": 3.995153718733028e-6,
        "segnet_distortion_budget_at_fixed_pose": 3.995153718733028e-8,
        "posenet_score_term_budget_at_fixed_seg": 3.995153718733028e-6,
        "candidate_payload_bytes": 37,
        "source_selector_payload_bytes": 43,
        "allowed_use": "planning_only_rate_savings_budget_for_segnet_posenet_repair",
        "forbidden_use": "score_claim_or_distortion_authority",
        **false_authority,
    }
    if truthy_authority:
        repair_budget["score_claim"] = True
    dropped_pair_indices: list[object] = [371, 376, 479]
    if malformed_pair_id:
        dropped_pair_indices[1] = 376.5
    path.write_text(
        json.dumps(
            {
                "schema": "decoder_q_pairset_acquisition.v1",
                **false_authority,
                "summary": {
                    "candidate_count": 1,
                    "drop_many_candidate_count": 1,
                    "pair_frame_geometry_candidate_count": 1,
                },
                "candidates": [
                    {
                        "schema": "decoder_q_pairset_acquisition_candidate.v1",
                        **false_authority,
                        "acquisition_id": "pairset_geometry_lowimpact_k006_habc123",
                        "selector_kind": "pair_frame_geometry_low_impact_drop_many",
                        "selected_pair_indices": [1, 2, 3],
                        "selected_pair_count": 3,
                        "payload_bytes": 37,
                        "distortion_repair_budget_from_rate_savings": repair_budget,
                        "acquisition_operation": {
                            "op": "pair_frame_geometry_low_impact_drop_many",
                            "dropped_pair_indices": dropped_pair_indices,
                            "dropped_pair_ranks": [21, 10, 3],
                            "geometry_coverage": 1.0,
                            "queue_executable": True,
                            "queue_family": "dqs1_pairset_local_first",
                            "master_gradient_status": (
                                "rank_order_proxy_until_pair_gradient_binding_lands"
                            ),
                            "inverse_scorer_status": (
                                "planner_consumer_requested_not_score_authority"
                            ),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _dqs1_observations(
    path: Path,
    *,
    truthy_authority: bool = False,
    malformed_pair_id: bool = False,
) -> None:
    false_authority = {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
    }
    dropped_pair_indices: list[object] = [371, 376, 479]
    if malformed_pair_id:
        dropped_pair_indices[2] = {"pair": 479}
    row = {
        "schema": "mlx_dynamic_sweep_observation.v1",
        **false_authority,
        "candidate_id": "pairset_geometry_lowimpact_k006_habc123",
        "source_schema": "dqs1_local_first_harvest.v1",
        "sweep_config_id": "dqs1_local_first_macos_cpu_advisory",
        "optimization_pass_id": "local_cpu_advisory_harvest",
        "family": "decoder_q_pairset_drop_many",
        "observed_axis": "macos_cpu_advisory",
        "evidence_tag": "[macOS-CPU advisory only]",
        "evidence_grade": "macOS-CPU advisory",
        "observed_score_or_delta": 0.1912,
        "archive_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "raw_output_or_cache_sha256": "c" * 64,
        "component_deltas": {
            "segnet_delta": 0.000003,
            "posenet_delta": -0.000002,
            "rate_delta": -RATE_SCORE_PER_BYTE * 6,
        },
        "source_artifact_path": ".omx/research/local_advisory.json",
        "source_artifact_sha256": "d" * 64,
        "planner_artifact_path": ".omx/research/dqs1_local_first_harvest.json",
        "planner_artifact_sha256": "e" * 64,
        "baseline_artifact_path": ".omx/research/top32_baseline_advisory.json",
        "baseline_artifact_sha256": "f" * 64,
        "score_delta_vs_baseline": 0.000001 - (RATE_SCORE_PER_BYTE * 6),
        "archive_byte_delta_vs_baseline": -6,
        "selected_pair_indices": [1, 2, 3],
        "selected_pair_count": 3,
        "selector_kind": "pair_frame_geometry_low_impact_drop_many",
        "acquisition_operation": {
            "op": "pair_frame_geometry_low_impact_drop_many",
            "dropped_pair_indices": dropped_pair_indices,
            "geometry_coverage": 1.0,
            "queue_executable": True,
            "queue_family": "dqs1_pairset_local_first",
        },
    }
    if truthy_authority:
        row["promotion_eligible"] = True
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def test_builder_merges_queue_and_sanitized_refs_into_plannable_surface(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.json"
    engineered = tmp_path / "engineered.json"
    auth = tmp_path / "auth.json"
    mlx = tmp_path / "mlx.json"
    _candidate_queue(queue)
    _engineered_correction_targeting(engineered)
    _auth_eval(auth)
    _mlx_calibration(mlx)

    surface = build_byte_shaving_signal_surface(
        repo_root=tmp_path,
        campaign_id="fixture_surface",
        candidate_queue_paths=[queue],
        engineered_correction_targeting_paths=[engineered],
        engineered_correction_max_targets=1,
        auth_eval_paths=[auth],
        mlx_calibration_paths=[mlx],
        xray_hooks=["bit_allocator"],
    )
    plan = build_byte_shaving_campaign_plan(surface, repo_root=tmp_path)

    assert surface["schema"] == "byte_shaving_signal_surface.v1"
    assert surface["score_claim"] is False
    assert surface["units"][0]["unit_id"] == "drop_pair_0371"
    assert surface["auth_eval_refs"][0]["source_score_claim_present"] is True
    assert "score_claim" not in surface["auth_eval_refs"][0]["metrics"]
    assert surface["source_signal_refs"][0]["score_claim_valid"] is False
    assert surface["engineered_correction_refs"][0]["surface_unit_count"] == 1
    assert surface["mlx_calibration_refs"][0]["score_claim"] is False
    assert surface["mlx_calibration_refs"][0]["score_claim_valid"] is False
    assert surface["xray_refs"][0]["primitive_count"] >= 0
    assert any(unit["unit_kind"] == "correction_target" for unit in surface["units"])
    assert plan["ranked_units"][0]["unit_id"] == "drop_pair_0371"
    assert plan["score_claim"] is False


def test_builder_promotes_pairset_acquisition_repair_budget_rows_to_units(
    tmp_path: Path,
) -> None:
    pairset = tmp_path / "pairset_acquisition.json"
    _pairset_acquisition(pairset)

    surface = build_byte_shaving_signal_surface(
        repo_root=tmp_path,
        campaign_id="pairset_surface",
        pairset_acquisition_paths=[pairset],
    )
    plan = build_byte_shaving_campaign_plan(surface, repo_root=tmp_path)

    ref = surface["pairset_acquisition_refs"][0]
    unit = surface["units"][0]
    ranked = plan["ranked_units"][0]
    assert ref["kind"] == "decoder_q_pairset_acquisition"
    assert ref["surface_unit_count"] == 1
    assert ref["repair_budget_row_count"] == 1
    assert ref["score_claim"] is False
    assert unit["unit_kind"] == "pair"
    assert unit["candidate_saved_bytes"] == 6
    assert unit["operations"][0]["materializer"] == "dqs1_pairset_drop_pair_adapter"
    assert unit["operations"][0]["target_kind"] == "dqs1_pairset_drop_pair"
    assert unit["operations"][0]["params"]["dropped_pair_indices"] == [371, 376, 479]
    assert unit["distortion_repair_budget_from_rate_savings"]["score_claim"] is False
    assert ranked["recommended_operation_materializer"] == "dqs1_pairset_drop_pair_adapter"
    assert ranked["recommended_operation_target_kind"] == "dqs1_pairset_drop_pair"
    assert ranked["recommended_operation_params"]["score_budget"] == pytest.approx(
        3.995153718733028e-6
    )
    assert "requires_receiver_runtime_consumption_proof" in ranked["blockers"]
    assert plan["score_claim"] is False


def test_builder_rejects_pairset_acquisition_truthy_nested_authority(
    tmp_path: Path,
) -> None:
    pairset = tmp_path / "pairset_acquisition.json"
    _pairset_acquisition(pairset, truthy_authority=True)

    with pytest.raises(ByteShavingCampaignError, match="score_claim"):
        build_byte_shaving_signal_surface(
            repo_root=tmp_path,
            campaign_id="pairset_surface",
            pairset_acquisition_paths=[pairset],
        )


def test_builder_rejects_pairset_acquisition_malformed_pair_id(
    tmp_path: Path,
) -> None:
    pairset = tmp_path / "pairset_acquisition.json"
    _pairset_acquisition(pairset, malformed_pair_id=True)

    with pytest.raises(ByteShavingCampaignError, match="expected integer id"):
        build_byte_shaving_signal_surface(
            repo_root=tmp_path,
            campaign_id="pairset_surface",
            pairset_acquisition_paths=[pairset],
        )


def test_builder_promotes_dqs1_outcomes_to_solver_anchor_units(
    tmp_path: Path,
) -> None:
    observations = tmp_path / "dqs1_observations.jsonl"
    _dqs1_observations(observations)

    surface = build_byte_shaving_signal_surface(
        repo_root=tmp_path,
        campaign_id="dqs1_outcome_surface",
        dqs1_observation_paths=[observations],
    )
    plan = build_byte_shaving_campaign_plan(surface, repo_root=tmp_path)

    ref = surface["pair_frame_geometry_outcome_refs"][0]
    unit = surface["units"][0]
    ranked = plan["ranked_units"][0]
    assert ref["kind"] == "dqs1_pair_frame_geometry_outcome_anchor"
    assert ref["dqs1_row_count"] == 1
    assert ref["k_gt_one_row_count"] == 1
    assert ref["emitted_unit_count"] == 1
    assert "inverse_steganalysis_action_functional" in ref["planner_consumers"]
    assert "bit_allocator_pair_budget" in ref["planner_consumers"]
    assert unit["candidate_saved_bytes"] == 6
    assert unit["dropped_pair_indices"] == [371, 376, 479]
    assert unit["dqs1_outcome_signal"]["score_claim"] is False
    assert unit["master_gradient_signal"]["status"] == (
        "empirical_pairset_outcome_anchor_ready"
    )
    assert ranked["recommended_operation_materializer"] == (
        "dqs1_pairset_drop_pair_adapter"
    )
    assert ranked["inverse_scorer_signal"]["consumer"] == (
        "tac.optimization.scorer_inverse_decision_surface"
    )
    assert ranked["bit_allocator_signal"]["consumer"] == "tac.optimization.bit_allocator"
    assert ranked["dqs1_outcome_signal"]["candidate_id"] == (
        "pairset_geometry_lowimpact_k006_habc123"
    )
    assert "macos_cpu_advisory_is_not_contest_authority" in ranked["blockers"]
    assert plan["score_claim"] is False


def test_builder_rejects_dqs1_outcome_truthy_authority(tmp_path: Path) -> None:
    observations = tmp_path / "dqs1_observations.jsonl"
    _dqs1_observations(observations, truthy_authority=True)

    with pytest.raises(ByteShavingCampaignError, match="promotion_eligible"):
        build_byte_shaving_signal_surface(
            repo_root=tmp_path,
            campaign_id="dqs1_outcome_surface",
            dqs1_observation_paths=[observations],
        )


def test_builder_rejects_dqs1_outcome_malformed_pair_id(tmp_path: Path) -> None:
    observations = tmp_path / "dqs1_observations.jsonl"
    _dqs1_observations(observations, malformed_pair_id=True)

    with pytest.raises(ByteShavingCampaignError, match="expected integer id"):
        build_byte_shaving_signal_surface(
            repo_root=tmp_path,
            campaign_id="dqs1_outcome_surface",
            dqs1_observation_paths=[observations],
        )


def test_builder_merges_inverse_action_functional_into_mixed_signal_surface(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.json"
    inverse_action = tmp_path / "inverse_action.json"
    _candidate_queue(queue)
    inverse_action.write_text(
        json.dumps(_inverse_action_payload()),
        encoding="utf-8",
    )

    surface = build_byte_shaving_signal_surface(
        repo_root=tmp_path,
        campaign_id="mixed_surface",
        candidate_queue_paths=[queue],
        inverse_action_functional_paths=[inverse_action],
    )
    plan = build_byte_shaving_campaign_plan(surface, repo_root=tmp_path)

    kinds = {unit["unit_kind"] for unit in surface["units"]}
    combo_ids = set(plan["recommended_combination"]["selected_unit_ids"])
    assert "scorer_inverse_surface_cell" in kinds
    assert "archive_section" in kinds
    assert surface["inverse_action_functional_refs"][0]["selected_count"] == 1
    assert surface["inverse_action_functional_refs"][0]["score_claim"] is False
    assert surface["inverse_action_materialization_portfolios"][0][
        "actuation_modes"
    ] == ["high_level_operation_compiler_required"]
    assert surface["inverse_action_materialization_portfolios"][0][
        "source_path"
    ] == "inverse_action.json"
    assert any(
        ref["kind"] == "inverse_steganalysis_action_functional"
        for ref in surface["source_signal_refs"]
    )
    assert {
        "drop_pair_0371",
        "inverse_action_inverse_surface_pair0007",
    }.issubset(combo_ids)
    assert plan["score_claim"] is False
    assert plan["recommended_combination"]["score_claim"] is False


def test_builder_canonicalizes_scorer_response_ref_planning_targets(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.json"
    scorer = tmp_path / "scorer.json"
    _candidate_queue(queue)
    _scorer_response_dataset(
        scorer,
        rows=[_mlx_response_row(projected_delta=0.001, raw_delta=-10.0)],
    )

    surface = build_byte_shaving_signal_surface(
        repo_root=tmp_path,
        campaign_id="fixture_surface",
        candidate_queue_paths=[queue],
        scorer_response_paths=[scorer],
    )

    ref = surface["scorer_response_refs"][0]
    summary = ref["planning_summary"]
    scorer_unit = next(
        unit
        for unit in surface["units"]
        if unit["unit_kind"] == "scorer_response_row"
    )
    plan = build_byte_shaving_campaign_plan(surface, repo_root=tmp_path)
    ranked_scorer_unit = next(
        unit
        for unit in plan["ranked_units"]
        if unit["unit_kind"] == "scorer_response_row"
    )
    assert ref["planning_target_accessor"] == "scorer_response_planning_value_for_target"
    assert ref["score_claim_valid"] is False
    assert ref["mlx_scorer_response_row_count"] == 1
    assert summary["improved_total_score_count"] == 0
    assert summary["improved_scorer_term_count"] == 0
    assert summary["best_delta"]["delta_vs_baseline_score"] == 0.001
    assert summary["best_scorer_delta"]["scorer_delta_vs_baseline"] == 0.001
    assert scorer_unit["projected_full_video_delta_vs_baseline_score"] == 0.001
    assert scorer_unit["planning_value_scope"] == "normalized_full_video"
    assert ranked_scorer_unit["expected_delta_score"] == pytest.approx(0.001)
    assert ranked_scorer_unit["recommended_operation_family"] == (
        "materialize_scorer_response_candidate"
    )


def test_builder_promotes_normalized_scorer_response_ref_to_ranked_unit(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer.json"
    _scorer_response_dataset(
        scorer,
        rows=[_mlx_response_row(projected_delta=-0.001, raw_delta=10.0)],
    )

    surface = build_byte_shaving_signal_surface(
        repo_root=tmp_path,
        campaign_id="fixture_surface",
        scorer_response_paths=[scorer],
    )
    plan = build_byte_shaving_campaign_plan(surface, repo_root=tmp_path)

    assert surface["units"][0]["unit_kind"] == "scorer_response_row"
    assert surface["units"][0]["candidate_saved_bytes"] == 0
    assert surface["units"][0]["projected_full_video_delta_vs_baseline_score"] == (
        pytest.approx(-0.001)
    )
    assert surface["units"][0]["planning_value_accessor"] == (
        "scorer_response_planning_value_for_target"
    )
    assert plan["ranked_units"][0]["unit_kind"] == "scorer_response_row"
    assert plan["ranked_units"][0]["expected_delta_score"] == pytest.approx(-0.001)
    assert plan["recommended_prefix"]["selected_unit_ids"] == [
        surface["units"][0]["unit_id"]
    ]


def test_builder_promotes_inverse_scorer_surface_cells_to_ranked_units(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer.json"
    _scorer_response_dataset(
        scorer,
        rows=[
            _inverse_response_row(
                projected_delta=-0.0001,
                scorer_delta=0.0,
                saved_bytes=32,
            )
        ],
    )

    surface = build_byte_shaving_signal_surface(
        repo_root=tmp_path,
        campaign_id="fixture_surface",
        inverse_scorer_response_paths=[scorer],
    )
    plan = build_byte_shaving_campaign_plan(surface, repo_root=tmp_path)

    ref = surface["inverse_scorer_surface_refs"][0]
    unit = surface["units"][0]
    assert ref["kind"] == "scorer_inverse_decision_surface"
    assert ref["decision_surface_classes"] == ["rate_only_null_space"]
    assert unit["unit_kind"] == "scorer_inverse_surface_cell"
    assert unit["decision_surface_class"] == "rate_only_null_space"
    assert unit["dominant_receiver_axis"] == "pose"
    assert unit["candidate_saved_bytes"] == 32
    assert plan["ranked_units"][0]["unit_kind"] == "scorer_inverse_surface_cell"
    assert plan["ranked_units"][0]["expected_delta_score"] == pytest.approx(-0.0001)
    assert plan["ranked_units"][0]["recommended_operation_family"] == (
        "probe_inverse_scorer_surface_cell"
    )
    assert plan["ranked_units"][0]["recommended_operation_materializer"] == (
        "inverse_scorer_action_functional_adapter"
    )
    assert plan["ranked_units"][0]["recommended_operation_target_kind"] == (
        "inverse_scorer_action_functional_v1"
    )
    assert plan["score_claim"] is False


def test_builder_inverse_surface_can_use_native_mlx_window_with_blocker(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer.json"
    row = _mlx_response_row(projected_delta=-0.0001, raw_delta=-0.0001)
    for key in (
        "normalized_full_video_scorer_gain_vs_baseline",
        "projected_full_video_delta_vs_baseline_score",
        "break_even_added_bytes_from_normalized_full_video_gain",
        "normalized_full_video_byte_budget_margin_vs_break_even",
    ):
        row.pop(key)
    _scorer_response_dataset(scorer, rows=[row])

    surface = build_byte_shaving_signal_surface(
        repo_root=tmp_path,
        campaign_id="fixture_surface",
        inverse_scorer_response_paths=[scorer],
        inverse_scorer_allow_native_mlx_window_objective=True,
    )

    ref = surface["inverse_scorer_surface_refs"][0]
    plan = build_byte_shaving_campaign_plan(surface, repo_root=tmp_path)
    ranked = plan["ranked_units"][0]
    selected = plan["recommended_prefix"]["selected_operations"][0]

    assert ref["allow_native_mlx_window_objective"] is True
    assert "native_mlx_window_objective_not_full_video_normalized" in ref["blockers"]
    assert "native_mlx_window_objective_not_full_video_normalized" in surface["blockers"]
    assert surface["units"][0]["unit_kind"] == "scorer_inverse_surface_cell"
    assert surface["units"][0]["planning_value_scope"] == "native_mlx_window"
    assert "native_window_delta_vs_baseline_score" in surface["units"][0]
    assert "projected_full_video_delta_vs_baseline_score" not in surface["units"][0]
    assert "native_mlx_window_objective_not_full_video_normalized" in surface["units"][0]["blockers"]
    assert "native_mlx_window_objective_not_full_video_normalized" in ranked["blockers"]
    assert "native_mlx_window_objective_not_full_video_normalized" in selected["blockers"]
    assert surface["score_claim"] is False


def test_builder_rejects_mlx_scorer_response_ref_missing_normalized_target(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.json"
    scorer = tmp_path / "scorer.json"
    _candidate_queue(queue)
    row = _mlx_response_row(projected_delta=0.001, raw_delta=-10.0)
    for key in (
        "normalized_full_video_scorer_gain_vs_baseline",
        "projected_full_video_delta_vs_baseline_score",
        "break_even_added_bytes_from_normalized_full_video_gain",
        "normalized_full_video_byte_budget_margin_vs_break_even",
    ):
        row.pop(key)
    _scorer_response_dataset(scorer, rows=[row])

    with pytest.raises(ByteShavingCampaignError, match="missing normalized full-video objective"):
        build_byte_shaving_signal_surface(
            repo_root=tmp_path,
            campaign_id="fixture_surface",
            candidate_queue_paths=[queue],
            scorer_response_paths=[scorer],
        )


def test_builder_rejects_truthy_proxy_sources(tmp_path: Path) -> None:
    queue = tmp_path / "queue.json"
    _candidate_queue(queue)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["top_k"][0]["score_claim"] = True
    queue.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ByteShavingCampaignError, match="score_claim"):
        build_byte_shaving_signal_surface(
            repo_root=tmp_path,
            campaign_id="fixture_surface",
            candidate_queue_paths=[queue],
        )


def test_cli_writes_surface_and_markdown(tmp_path: Path) -> None:
    queue = tmp_path / "queue.json"
    pairset = tmp_path / "pairset_acquisition.json"
    dqs1_observations = tmp_path / "dqs1_observations.jsonl"
    engineered = tmp_path / "engineered.json"
    inverse_action = tmp_path / "inverse_action.json"
    output = tmp_path / "surface.json"
    md_out = tmp_path / "surface.md"
    _candidate_queue(queue)
    _pairset_acquisition(pairset)
    _dqs1_observations(dqs1_observations)
    _engineered_correction_targeting(engineered)
    inverse_action.write_text(json.dumps(_inverse_action_payload()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--candidate-queue",
            str(queue),
            "--pairset-acquisition",
            str(pairset),
            "--dqs1-observation-jsonl",
            str(dqs1_observations),
            "--engineered-correction-targeting",
            str(engineered),
            "--engineered-correction-max-targets",
            "1",
            "--inverse-action-functional",
            str(inverse_action),
            "--output",
            str(output),
            "--md-out",
            str(md_out),
            "--repo-root",
            str(tmp_path),
            "--campaign-id",
            "fixture_surface",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "score_claim=false" in result.stdout
    surface = json.loads(output.read_text(encoding="utf-8"))
    assert surface["units"][0]["unit_id"] == "drop_pair_0371"
    assert surface["pairset_acquisition_refs"][0]["surface_unit_count"] == 1
    assert surface["pair_frame_geometry_outcome_refs"][0]["emitted_unit_count"] == 1
    assert surface["engineered_correction_refs"][0]["surface_unit_count"] == 1
    assert surface["inverse_action_functional_refs"][0]["surface_unit_count"] == 1
    assert len(surface["inverse_action_materialization_portfolios"]) == 1
    md = md_out.read_text(encoding="utf-8")
    assert "pairset_acquisition_refs" in md
    assert "pair_frame_geometry_outcome_refs" in md
    assert "inverse_action_materialization_portfolios" in md
    assert "Authority Boundary" in md


def test_cli_can_build_surface_from_scorer_response_only(tmp_path: Path) -> None:
    scorer = tmp_path / "scorer.json"
    output = tmp_path / "surface.json"
    _scorer_response_dataset(
        scorer,
        rows=[_mlx_response_row(projected_delta=-0.001, raw_delta=10.0)],
    )

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--scorer-response",
            str(scorer),
            "--output",
            str(output),
            "--repo-root",
            str(tmp_path),
            "--campaign-id",
            "fixture_surface",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    surface = json.loads(output.read_text(encoding="utf-8"))
    assert surface["units"][0]["unit_kind"] == "scorer_response_row"
    assert surface["units"][0]["planning_value_scope"] == "normalized_full_video"
    assert surface["score_claim"] is False


def test_cli_can_build_inverse_scorer_surface(tmp_path: Path) -> None:
    scorer = tmp_path / "scorer.json"
    output = tmp_path / "surface.json"
    _scorer_response_dataset(
        scorer,
        rows=[
            _inverse_response_row(
                projected_delta=-0.0001,
                scorer_delta=0.0,
                saved_bytes=32,
            )
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--inverse-scorer-response",
            str(scorer),
            "--output",
            str(output),
            "--repo-root",
            str(tmp_path),
            "--campaign-id",
            "fixture_surface",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    surface = json.loads(output.read_text(encoding="utf-8"))
    assert surface["inverse_scorer_surface_refs"][0]["emitted_unit_count"] == 1
    assert surface["units"][0]["unit_kind"] == "scorer_inverse_surface_cell"
