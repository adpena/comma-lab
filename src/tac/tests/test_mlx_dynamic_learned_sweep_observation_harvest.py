# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.mlx_dynamic_learned_sweep import (
    build_mlx_dynamic_learned_sweep_plan,
)
from tac.optimization.mlx_dynamic_learned_sweep_observation_harvest import (
    MLXDynamicLearnedSweepObservationHarvestError,
    build_observation_harvest_manifest,
    build_observation_rows_from_learned_sweep_plan,
)
from tac.optimization.mlx_dynamic_sweep_observations import load_observation_rows
from tac.optimization.mlx_effective_spend_triage_learned_sweep_adapter import (
    build_mlx_effective_spend_triage_learned_sweep_candidates,
)
from tac.optimization.normalized_objective import RATE_SCORE_PER_BYTE

REPO_ROOT = Path(__file__).resolve().parents[3]
INCUMBENT_SCORE = 0.1920513168811056


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _selection(source_path: Path) -> dict[str, object]:
    observed_gain = 0.012
    normalized_gain = observed_gain / 600.0
    return {
        "schema": "mlx_effective_spend_triage_candidate_selection.v1",
        **_false_authority(),
        "candidate_generation_only": True,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "gates": {
            "effective_mlx_spend_triage_gate": {
                "schema": "ll_effective_mlx_spend_triage_gate.v1",
                "status": "strict_pass",
                "mlx_exact_eval_spend_triage_allowed": True,
            },
            "response_validation_status": "passed",
            "torch_parity_status": "strict_pass",
            "score_calibration_status": "strict_pass",
            "production_contract_status": "strict_pass",
        },
        "selected_rows": [
            {
                "schema": "mlx_effective_spend_triage_candidate_row.v1",
                **_false_authority(),
                "candidate_generation_only": True,
                "archive_materialization_required": True,
                "requires_exact_auth_eval_before_score_claim": True,
                "candidate_id": "mlx_scorer_response:window:501:502",
                "row_id": "source_row_501_502",
                "family": "mlx_decoder_q",
                "rank": 1,
                "selection_basis": "normalized_full_video_mlx_singleton_response_gain",
                "observed_delta_vs_baseline_score": -observed_gain,
                "observed_scorer_delta_vs_baseline": -observed_gain,
                "observed_scorer_gain_vs_baseline": observed_gain,
                "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
                "projected_full_video_delta_vs_baseline_score": -normalized_gain,
                "break_even_added_bytes_from_normalized_full_video_gain": (
                    normalized_gain / RATE_SCORE_PER_BYTE
                ),
                "normalized_full_video_byte_budget_margin_vs_break_even": (
                    normalized_gain / RATE_SCORE_PER_BYTE
                ),
                "full_video_denominator": 600,
                "source_n_samples": 1,
                "source_batch_pairs": 1,
                "source_pair_window": [501, 502],
                "pair_indices": [501, 502],
                "added_archive_bytes": 0,
                "calibrated_min_mlx_gap_for_spend_triage": 0.00001,
                "source_path": str(source_path),
                "archive_sha256": "a" * 64,
                "raw_sha256": "b" * 64,
                "source_inflated_outputs_aggregate_sha256": "c" * 64,
                "source_candidate_cache_array_sha256": {
                    "pair_indices": "d" * 64,
                    "posenet_yuv6_pair": "e" * 64,
                    "segnet_last_rgb": "f" * 64,
                },
                "source_reference_cache_array_sha256": {
                    "pair_indices": "d" * 64,
                    "posenet_yuv6_pair": "1" * 64,
                    "segnet_last_rgb": "2" * 64,
                },
                "source_posenet_sha256": "3" * 64,
                "source_segnet_sha256": "4" * 64,
                "source_evidence_grade": EVIDENCE_GRADE_MLX,
                "source_evidence_tag": EVIDENCE_TAG_MLX,
                "source_schema": "mlx_scorer_response.v1",
                "seg_term": 0.069,
                "pose_term": 0.0163,
                "rate_term": 0.118,
                "window_baseline_seg_term": 0.071,
                "window_baseline_pose_term": 0.0162,
                "window_baseline_rate_term": 0.118,
                "rate_delta_vs_baseline": 0.0,
            }
        ],
    }


def _plan(selection: dict[str, object]) -> dict[str, object]:
    candidates = build_mlx_effective_spend_triage_learned_sweep_candidates(
        selection,
        incumbent_score=INCUMBENT_SCORE,
    )
    return build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=INCUMBENT_SCORE,
        candidate_payloads=[candidates],
        execution_configs=[
            {
                "config_id": "mlx_local_response",
                "substrate": EVIDENCE_TAG_MLX,
                "execution_layer": "local_mlx",
                "cost_units": 1.0,
                "signal_quality": 0.45,
                "parallelizable": True,
                "exact_eval_candidate": False,
            }
        ],
        optimization_passes=[
            {
                "pass_id": "smoke",
                "scale": "smoke",
                "recursive_stage": 0,
                "sample_budget": 1,
                "cost_multiplier": 0.25,
                "expected_improvement_weight": 0.25,
                "exploration_weight": 1.75,
                "freeze_candidate": False,
            }
        ],
        top_k=1,
    )


def test_harvest_builds_observation_and_replanner_suppresses_tuple(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate_pair_0501_0502.json"
    source.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    selection = _selection(source)
    plan = _plan(selection)

    observations = build_observation_rows_from_learned_sweep_plan(
        plan,
        selection,
        planner_artifact_path=tmp_path / "plan.json",
        planner_artifact_sha256="5" * 64,
        max_rows=1,
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation["schema"] == "mlx_dynamic_sweep_observation.v1"
    assert observation["score_claim"] is False
    assert observation["candidate_id"] == "mlx_scorer_response:window:501:502"
    assert observation["sweep_config_id"] == "mlx_local_response"
    assert observation["optimization_pass_id"] == "smoke"
    assert observation["observed_score_or_delta"] == pytest.approx(-0.012 / 600.0)
    assert observation["segnet_delta"] == pytest.approx((0.069 - 0.071) / 600.0)
    assert observation["posenet_delta"] == pytest.approx((0.0163 - 0.0162) / 600.0)
    assert len(observation["runtime_sha256"]) == 64
    assert len(observation["raw_output_or_cache_sha256"]) == 64

    replanned = build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=INCUMBENT_SCORE,
        candidate_payloads=[
            build_mlx_effective_spend_triage_learned_sweep_candidates(
                selection,
                incumbent_score=INCUMBENT_SCORE,
            )
        ],
        execution_configs=[
            {
                "config_id": "mlx_local_response",
                "substrate": EVIDENCE_TAG_MLX,
                "execution_layer": "local_mlx",
                "cost_units": 1.0,
                "signal_quality": 0.45,
                "parallelizable": True,
                "exact_eval_candidate": False,
            }
        ],
        optimization_passes=[
            {
                "pass_id": "smoke",
                "scale": "smoke",
                "recursive_stage": 0,
                "sample_budget": 1,
                "cost_multiplier": 0.25,
                "expected_improvement_weight": 0.25,
                "exploration_weight": 1.75,
                "freeze_candidate": False,
            }
        ],
        observations=observations,
        top_k=1,
    )

    assert replanned["summary"]["suppressed_observed_row_count"] == 1
    assert replanned["suppressed_observed_sweep_rows"][0]["observation_feedback"][
        "suppression_reason"
    ] == "already_observed_candidate_config_pass_family"


def test_harvest_derives_component_deltas_from_source_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "candidate_pair_0501_0502.json"
    baseline = tmp_path / "baseline_pair_0501_0502.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "mlx_scorer_response.v1",
                "avg_segnet_dist": 0.00069,
                "avg_posenet_dist": 0.0000264,
                "score_rate_contribution": 0.118,
            }
        ),
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(
            {
                "schema_version": "mlx_scorer_response.v1",
                "avg_segnet_dist": 0.00071,
                "avg_posenet_dist": 0.0000265,
                "score_rate_contribution": 0.118,
            }
        ),
        encoding="utf-8",
    )
    selection = _selection(source)
    source_row = selection["selected_rows"][0]
    assert isinstance(source_row, dict)
    for key in (
        "seg_term",
        "window_baseline_seg_term",
        "pose_term",
        "window_baseline_pose_term",
        "rate_term",
        "window_baseline_rate_term",
        "rate_delta_vs_baseline",
    ):
        source_row.pop(key, None)
    source_row["window_baseline_source_path"] = str(baseline)

    observations = build_observation_rows_from_learned_sweep_plan(
        _plan(selection),
        selection,
        max_rows=1,
    )

    observation = observations[0]
    expected_seg_delta = (100.0 * 0.00069 - 100.0 * 0.00071) / 600.0
    expected_pose_delta = (
        (10.0 * 0.0000264) ** 0.5 - (10.0 * 0.0000265) ** 0.5
    ) / 600.0
    assert observation["segnet_delta"] == pytest.approx(expected_seg_delta)
    assert observation["posenet_delta"] == pytest.approx(expected_pose_delta)
    assert observation["rate_delta"] == pytest.approx(0.0)
    assert observation["component_deltas"]["scorer_delta"] == pytest.approx(
        expected_seg_delta + expected_pose_delta
    )


def test_harvest_requires_explicit_false_plan_authority(tmp_path: Path) -> None:
    source = tmp_path / "candidate_pair_0501_0502.json"
    source.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    selection = _selection(source)
    plan = _plan(selection)
    del plan["score_claim"]

    with pytest.raises(
        MLXDynamicLearnedSweepObservationHarvestError,
        match="score_claim",
    ):
        build_observation_rows_from_learned_sweep_plan(
            plan,
            selection,
            max_rows=1,
        )


def test_harvest_cli_writes_jsonl_and_summary(tmp_path: Path) -> None:
    source = tmp_path / "candidate_pair_0501_0502.json"
    source.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    selection = _selection(source)
    plan = _plan(selection)
    selection_path = tmp_path / "selection.json"
    plan_path = tmp_path / "plan.json"
    jsonl_out = tmp_path / "observations.jsonl"
    summary_out = tmp_path / "summary.json"
    selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "tools/harvest_mlx_dynamic_learned_sweep_observations.py",
            "--plan",
            str(plan_path),
            "--selection",
            str(selection_path),
            "--jsonl-out",
            str(jsonl_out),
            "--summary-json-out",
            str(summary_out),
            "--max-rows",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["row_count"] == 1
    rows = load_observation_rows(jsonl_out)
    assert len(rows) == 1
    manifest = json.loads(summary_out.read_text(encoding="utf-8"))
    assert manifest["schema"] == "mlx_dynamic_learned_sweep_observation_harvest.v1"
    assert manifest["summary"]["row_count"] == 1
    assert build_observation_harvest_manifest(rows)["row_count"] == 1
