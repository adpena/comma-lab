# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tac.optimization.mlx_dynamic_learned_sweep import (
    build_mlx_dynamic_learned_sweep_plan,
)
from tac.optimization.mlx_dynamic_learned_sweep_local_actuator import (
    MLXDynamicLearnedSweepLocalActuatorError,
    execute_local_mlx_sweep_rows,
    replan_after_local_actuation,
)
from tac.optimization.mlx_dynamic_sweep_observations import load_observation_rows
from tac.optimization.mlx_effective_spend_triage_learned_sweep_adapter import (
    build_mlx_effective_spend_triage_learned_sweep_candidates,
)
from tac.optimization.normalized_objective import RATE_SCORE_PER_BYTE

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


def _source_response_artifact(
    path: Path,
    *,
    reference_cache: Path,
    candidate_cache: Path,
    archive_size_bytes: int,
    archive_sha256: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "mlx_scorer_response.v1",
                "archive_size_bytes": archive_size_bytes,
                "archive_sha256": archive_sha256,
                "start_pair": 10,
                "total_cache_pairs": 600,
                "cache_identity": {
                    "reference": {"path": str(reference_cache)},
                    "candidate": {"path": str(candidate_cache)},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _selection(tmp_path: Path) -> dict[str, Any]:
    reference_cache = tmp_path / "reference_cache"
    candidate_cache = tmp_path / "candidate_cache"
    baseline_cache = tmp_path / "baseline_cache"
    for path in (reference_cache, candidate_cache, baseline_cache):
        path.mkdir()
    candidate_source = tmp_path / "candidate_source.json"
    baseline_source = tmp_path / "baseline_source.json"
    _source_response_artifact(
        candidate_source,
        reference_cache=reference_cache,
        candidate_cache=candidate_cache,
        archive_size_bytes=1024,
        archive_sha256="a" * 64,
    )
    _source_response_artifact(
        baseline_source,
        reference_cache=reference_cache,
        candidate_cache=baseline_cache,
        archive_size_bytes=1024,
        archive_sha256="b" * 64,
    )
    observed_gain = 0.012
    normalized_gain = observed_gain / 600.0
    return {
        "schema": "mlx_effective_spend_triage_candidate_selection.v1",
        **_false_authority(),
        "candidate_generation_only": True,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
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
                "candidate_id": "mlx_scorer_response:window:10:11",
                "row_id": "source_row_10_11",
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
                "source_pair_window": [10, 11],
                "pair_indices": [10, 11],
                "added_archive_bytes": 0,
                "calibrated_min_mlx_gap_for_spend_triage": 0.00001,
                "source_path": str(candidate_source),
                "window_baseline_source_path": str(baseline_source),
                "archive_sha256": "a" * 64,
                "raw_sha256": "c" * 64,
                "source_inflated_outputs_aggregate_sha256": "d" * 64,
                "source_candidate_cache_array_sha256": {
                    "pair_indices": "e" * 64,
                    "posenet_yuv6_pair": "f" * 64,
                    "segnet_last_rgb": "1" * 64,
                },
                "source_reference_cache_array_sha256": {
                    "pair_indices": "e" * 64,
                    "posenet_yuv6_pair": "2" * 64,
                    "segnet_last_rgb": "3" * 64,
                },
                "source_posenet_sha256": "4" * 64,
                "source_segnet_sha256": "5" * 64,
                "source_evidence_grade": "macOS-MLX-research-signal",
                "source_evidence_tag": "[macOS-MLX research-signal]",
                "source_schema": "mlx_scorer_response.v1",
            }
        ],
    }


def _candidate_payload(selection: dict[str, Any]) -> dict[str, Any]:
    return build_mlx_effective_spend_triage_learned_sweep_candidates(
        selection,
        incumbent_score=INCUMBENT_SCORE,
    )


def _plan(selection: dict[str, Any]) -> dict[str, Any]:
    return build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=INCUMBENT_SCORE,
        candidate_payloads=[_candidate_payload(selection)],
        execution_configs=[
            {
                "config_id": "mlx_local_response",
                "substrate": "[macOS-MLX research-signal]",
                "execution_layer": "local_mlx",
                "cost_units": 1.0,
                "signal_quality": 0.45,
                "parallelizable": True,
                "exact_eval_candidate": False,
            },
            {
                "config_id": "macos_cpu_advisory",
                "substrate": "[macOS-CPU advisory]",
                "execution_layer": "local_cpu",
                "cost_units": 4.0,
                "signal_quality": 0.6,
                "parallelizable": True,
                "exact_eval_candidate": False,
            },
        ],
        optimization_passes=[
            {
                "pass_id": "micro",
                "scale": "micro",
                "recursive_stage": 1,
                "sample_budget": 8,
                "cost_multiplier": 1.0,
                "expected_improvement_weight": 0.75,
                "exploration_weight": 1.25,
                "freeze_candidate": False,
            }
        ],
        top_k=2,
    )


def _fake_response_builder(**kwargs: Any) -> dict[str, Any]:
    candidate_path = str(kwargs["candidate_cache_dir"])
    is_baseline = "baseline" in candidate_path
    seg = 0.00071 if is_baseline else 0.00069
    pose = 0.0000265 if is_baseline else 0.0000264
    archive_sha = "b" * 64 if is_baseline else "a" * 64
    return {
        "schema_version": "mlx_scorer_response.v1",
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
        "hardware_substrate": "MLX cpu",
        **_false_authority(),
        "candidate_generation_only": True,
        "canonical_score_source": "score_recomputed_from_components",
        "canonical_score": 0.2,
        "score_recomputed_from_components": 0.2,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": 1024,
        "archive_sha256": archive_sha,
        "inflated_outputs_aggregate_sha256": "6" * 64,
        "raw_sha256": "7" * 64,
        "n_samples": int(kwargs["max_pairs"]),
        "start_pair": int(kwargs["start_pair"]),
        "max_pairs": int(kwargs["max_pairs"]),
        "batch_pairs": int(kwargs["batch_pairs"]),
        "cache_identity": {
            "candidate": {
                "array_sha256": {
                    "pair_indices": "8" * 64,
                    "posenet_yuv6_pair": "9" * 64,
                    "segnet_last_rgb": "0" * 64,
                },
                "inflated_outputs_aggregate_sha256": "6" * 64,
            }
        },
        "device_contract": {
            "allowed_uses": ["unit_test"],
            "forbidden_uses": ["score_claim"],
        },
    }


def test_local_actuator_executes_mlx_row_appends_observation_and_replans(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    plan = _plan(selection)
    observation_jsonl = tmp_path / "observations.jsonl"

    summary = execute_local_mlx_sweep_rows(
        plan=plan,
        selection=selection,
        output_dir=tmp_path / "actuation",
        observation_jsonl=observation_jsonl,
        max_rows=1,
        response_builder=_fake_response_builder,
    )

    assert summary["schema"] == "mlx_dynamic_learned_sweep_local_actuation.v1"
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["executed_row_count"] == 1
    assert summary["executed_filter_match"] is True
    assert summary["executed_filter_violation_count"] == 0
    rows = load_observation_rows(observation_jsonl)
    assert len(rows) == 1
    observation = rows[0]
    assert observation["candidate_id"] == "mlx_scorer_response:window:10:11"
    assert observation["sweep_config_id"] == "mlx_local_response"
    assert observation["optimization_pass_id"] == "micro"
    assert observation["score_claim"] is False
    assert observation["ready_for_exact_eval_dispatch"] is False
    assert observation["segnet_delta"] == pytest.approx(
        (100.0 * 0.00069 - 100.0 * 0.00071) * (8 / 600)
    )
    assert observation["posenet_delta"] == pytest.approx(
        ((10.0 * 0.0000264) ** 0.5 - (10.0 * 0.0000265) ** 0.5) * (8 / 600)
    )

    replanned = replan_after_local_actuation(
        incumbent_score=INCUMBENT_SCORE,
        candidate_payloads=[_candidate_payload(selection)],
        source_plan=plan,
        observation_jsonl_paths=[observation_jsonl],
        json_out=tmp_path / "replan.json",
    )
    assert replanned["summary"]["observation_row_count"] == 1
    assert replanned["summary"]["suppressed_observed_row_count"] == 1


def test_replan_cli_merges_parallel_observation_ledgers(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    plan = _plan(selection)
    candidate_payload = _candidate_payload(selection)
    plan_path = tmp_path / "plan.json"
    candidate_path = tmp_path / "candidate_payload.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    candidate_path.write_text(
        json.dumps(candidate_payload, sort_keys=True),
        encoding="utf-8",
    )
    observation_a = tmp_path / "observations_a.jsonl"
    observation_b = tmp_path / "observations_b.jsonl"
    execute_local_mlx_sweep_rows(
        plan=plan,
        selection=selection,
        output_dir=tmp_path / "actuation_a",
        observation_jsonl=observation_a,
        max_rows=1,
        response_builder=_fake_response_builder,
    )
    execute_local_mlx_sweep_rows(
        plan=plan,
        selection=selection,
        output_dir=tmp_path / "actuation_b",
        observation_jsonl=observation_b,
        max_rows=1,
        response_builder=_fake_response_builder,
    )
    replan_json = tmp_path / "merged_replan.json"
    replan_md = tmp_path / "merged_replan.md"
    summary_json = tmp_path / "merged_replan_summary.json"
    repo_root = Path(__file__).resolve().parents[3]

    completed = subprocess.run(
        [
            sys.executable,
            "tools/replan_mlx_dynamic_learned_sweep_from_observations.py",
            "--plan",
            str(plan_path),
            "--candidate-payload",
            str(candidate_path),
            "--observation-jsonl",
            str(observation_a),
            "--observation-jsonl",
            str(observation_b),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--json-out",
            str(replan_json),
            "--md-out",
            str(replan_md),
            "--summary-json-out",
            str(summary_json),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    assert stdout["schema"] == "mlx_dynamic_learned_sweep_replan_from_observations.v1"
    assert stdout["observation_jsonl_count"] == 2
    assert stdout["raw_observation_row_count"] == 2
    assert stdout["observation_row_count"] == 1
    assert stdout["duplicate_observation_row_count"] == 1
    assert stdout["score_claim"] is False
    assert replan_json.is_file()
    assert replan_md.is_file()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["observation_jsonl_count"] == 2
    assert summary["raw_observation_row_count"] == 2
    assert summary["observation_row_count"] == 1
    assert summary["duplicate_observation_row_count"] == 1
    assert summary["replan"]["json_out"] == str(replan_json)
    replanned = json.loads(replan_json.read_text(encoding="utf-8"))
    assert replanned["summary"]["observation_row_count"] == 1
    assert replanned["summary"]["suppressed_observed_row_count"] == 1


def test_local_actuator_filters_exact_queue_candidate_id(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    plan = _plan(selection)
    queue_candidate_id = "mlx_scorer_response:window:10:11::mlx_local_response::micro"

    summary = execute_local_mlx_sweep_rows(
        plan=plan,
        selection=selection,
        output_dir=tmp_path / "actuation",
        max_rows=1,
        queue_candidate_ids=[queue_candidate_id],
        response_builder=_fake_response_builder,
    )

    assert summary["executed_filter_match"] is True
    assert summary["queue_candidate_id_filters"] == [queue_candidate_id]
    assert summary["executed_unique_queue_candidate_id"] == queue_candidate_id
    assert summary["executed_rows"][0]["queue_candidate_id"] == queue_candidate_id


def test_local_actuator_filters_multiple_exact_queue_candidates(
    tmp_path: Path,
) -> None:
    selection = _selection(tmp_path)
    plan = _plan(selection)
    first = dict(plan["ranked_sweep_rows"][0])
    second = dict(first)
    second["queue_candidate_id"] = (
        "mlx_scorer_response:window:10:11::mlx_local_response::macro"
    )
    second["optimization_pass_id"] = "macro"
    plan["ranked_sweep_rows"].insert(1, second)
    queue_candidate_ids = [
        first["queue_candidate_id"],
        second["queue_candidate_id"],
    ]

    summary = execute_local_mlx_sweep_rows(
        plan=plan,
        selection=selection,
        output_dir=tmp_path / "actuation",
        max_rows=2,
        queue_candidate_ids=queue_candidate_ids,
        response_builder=_fake_response_builder,
    )

    assert summary["executed_filter_match"] is True
    assert summary["executed_queue_candidate_id_count"] == 2
    assert summary["executed_queue_candidate_id_set"] == sorted(queue_candidate_ids)
    assert summary["executed_unique_queue_candidate_id"] is None


def test_local_actuator_refuses_missing_queue_candidate_filter(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    plan = _plan(selection)

    with pytest.raises(
        MLXDynamicLearnedSweepLocalActuatorError,
        match="queue_candidate_id filters",
    ):
        execute_local_mlx_sweep_rows(
            plan=plan,
            selection=selection,
            output_dir=tmp_path / "actuation",
            max_rows=1,
            queue_candidate_ids=["missing::mlx_local_response::micro"],
            response_builder=_fake_response_builder,
        )


def test_local_actuator_refuses_unsupported_config(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    plan = _plan(selection)
    with pytest.raises(
        MLXDynamicLearnedSweepLocalActuatorError,
        match="unsupported sweep_config_id",
    ):
        execute_local_mlx_sweep_rows(
            plan=plan,
            selection=selection,
            output_dir=tmp_path / "actuation",
            sweep_config_id="macos_cpu_advisory",
            response_builder=_fake_response_builder,
        )


def test_local_actuator_rejects_truthy_authority(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    plan = _plan(selection)
    plan["score_claim"] = True
    with pytest.raises(MLXDynamicLearnedSweepLocalActuatorError, match="score_claim"):
        execute_local_mlx_sweep_rows(
            plan=plan,
            selection=selection,
            output_dir=tmp_path / "actuation",
            response_builder=_fake_response_builder,
        )
