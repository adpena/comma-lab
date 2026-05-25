# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.distortion_axis_probe_learned_sweep_adapter import (
    SCHEMA,
    DistortionAxisProbeLearnedSweepAdapterError,
    build_distortion_axis_probe_learned_sweep_candidates,
)
from tac.optimization.distortion_axis_probe_learned_sweep_feedback import (
    build_distortion_axis_probe_feedback_observation,
)
from tac.optimization.mlx_dynamic_learned_sweep import (
    build_mlx_dynamic_learned_sweep_plan,
)
from tac.optimization.mlx_dynamic_sweep_observations import load_observation_rows
from tac.optimization.normalized_objective import RATE_SCORE_PER_BYTE

REPO_ROOT = Path(__file__).resolve().parents[3]
INCUMBENT_SCORE = 0.19202828295713675


def _positive_uniward_verdict() -> dict[str, object]:
    return {
        "probe_id": "tier_1_distortion_uniward_per_instance_multi_scale_wavelet_combined_smoke",
        "probe_name": "Probe 9 combined",
        "verdict": "POSITIVE_SIGNAL_BREAKS_THRESHOLD",
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "macOS-CPU-advisory",
        "lane_id": "lane_wave3",
        "actual_signature": {
            "min_segment_textured_avg_weight_combined": 0.25968116521835327,
            "spread_segment_textured_avg_weight_combined": 0.35557132959365845,
            "valid_segment_count": 22,
            "any_segment_below_threshold": True,
            "wavelet_name": "db8",
            "wavelet_levels": 3,
            "pair_count": 4,
            "n_frames_decoded": 8,
            "per_class_segment_count": {"0": 7, "1": 7, "2": 4, "4": 4},
            "per_segment_metrics": [
                {
                    "pair_index": 0,
                    "class_index": 1,
                    "instance_id": 2,
                    "instance_pixel_count": 594,
                    "instance_textured_count": 149,
                    "instance_textured_avg_weight_combined": 0.25968116521835327,
                },
                {
                    "pair_index": 1,
                    "class_index": 0,
                    "instance_id": 2,
                    "instance_pixel_count": 11742,
                    "instance_textured_count": 2936,
                    "instance_textured_avg_weight_combined": 0.3452233672142029,
                },
            ],
        },
        "recommendation": (
            "POSITIVE_SIGNAL_BREAKS_THRESHOLD: Tier-2 paid dispatch; "
            "predicted DS -0.010 to -0.025 [predicted]."
        ),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _motion_neutral_verdict() -> dict[str, object]:
    return {
        "probe_id": "tier_1_distortion_hinton_kl_motion_aware_temporal_context_w6_smoke",
        "probe_name": "Probe 10 motion aware",
        "verdict": "NEGATIVE_MOTION_NEUTRAL",
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "macOS-CPU-advisory",
        "actual_signature": {
            "motion_amplification_ratio": 1.0013685713896916,
            "motion_kl_pearson": 0.08990300558401604,
            "motion_weighted_kl_W6": 0.01468249961394509,
            "uniform_kl_W6": 0.01466243302760025,
        },
        "score_claim": False,
        "promotable": False,
    }


def test_adapter_turns_probe9_into_learned_sweep_candidate_and_suppresses_probe10() -> None:
    payload = build_distortion_axis_probe_learned_sweep_candidates(
        [_positive_uniward_verdict(), _motion_neutral_verdict()],
        incumbent_score=INCUMBENT_SCORE,
    )

    assert payload["schema"] == SCHEMA
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["summary"]["adapted_candidate_count"] == 1
    assert payload["summary"]["suppressed_candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["candidate_id"].endswith("uniward_per_instance_multi_scale_wavelet_combined_v1")
    assert candidate["predicted_score_mean"] == pytest.approx(INCUMBENT_SCORE - 0.010)
    assert candidate["quality_evidence"]["gate_statuses"] == {
        "calibration": "strict_pass",
        "parity": "strict_pass",
        "production_contract": "strict_pass",
        "effective_spend_triage": "strict_pass",
    }
    assert candidate["segnet_context"]["threshold_broken"] is True
    assert candidate["posenet_context"]["status"] == "repair_budget_candidate_not_measured"
    assert candidate["component_axis_context"][
        "non_authoritative_rate_budget_bytes_equivalent"
    ] == pytest.approx(0.010 / RATE_SCORE_PER_BYTE)
    assert candidate["rate_distortion_lattice_context"]["covered_levels"] == [
        "bit",
        "byte",
        "pixel",
        "region",
        "boundary",
        "frame",
        "pair",
        "batch",
        "full_video",
    ]
    assert payload["suppressed_candidates"][0]["suppression_reason"] == (
        "NEGATIVE_MOTION_NEUTRAL"
    )

    plan = build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=INCUMBENT_SCORE,
        candidate_payloads=[payload],
        top_k=2,
    )
    assert plan["summary"]["candidate_count"] == 1
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    row = plan["ranked_sweep_rows"][0]
    assert row["candidate_id"] == candidate["candidate_id"]
    assert row["component_axis_context"]["primary_axis"] == "segnet"


def test_adapter_rejects_truthy_authority_in_source_verdict() -> None:
    verdict = _positive_uniward_verdict()
    verdict["ready_for_exact_eval_dispatch"] = True

    with pytest.raises(
        DistortionAxisProbeLearnedSweepAdapterError,
        match="ready_for_exact_eval_dispatch",
    ):
        build_distortion_axis_probe_learned_sweep_candidates(
            [verdict],
            incumbent_score=INCUMBENT_SCORE,
        )


def test_adapter_cli_writes_candidate_payload_and_plan(tmp_path: Path) -> None:
    probe9 = tmp_path / "probe_9_verdict.json"
    probe10 = tmp_path / "probe_10_verdict.json"
    payload_path = tmp_path / "distortion_axis_probe_learned_sweep_candidates.json"
    plan_path = tmp_path / "distortion_axis_probe_learned_sweep_plan.json"
    md_path = tmp_path / "distortion_axis_probe_learned_sweep_plan.md"
    probe9.write_text(json.dumps(_positive_uniward_verdict()), encoding="utf-8")
    probe10.write_text(json.dumps(_motion_neutral_verdict()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/adapt_distortion_axis_probes_to_learned_sweep.py",
            "--verdict",
            str(probe9),
            "--verdict",
            str(probe10),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--json-out",
            str(payload_path),
            "--plan-json-out",
            str(plan_path),
            "--plan-md-out",
            str(md_path),
            "--plan-top-k",
            "2",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["adapted_candidate_count"] == 1
    assert summary["suppressed_candidate_count"] == 1
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert json.loads(payload_path.read_text(encoding="utf-8"))["schema"] == SCHEMA
    assert json.loads(plan_path.read_text(encoding="utf-8"))["schema"] == (
        "mlx_dynamic_learned_sweep_plan.v1"
    )
    assert "MLX Dynamic Learned Sweep Plan" in md_path.read_text(encoding="utf-8")


def test_feedback_harvester_appends_observation_and_replans(
    tmp_path: Path,
) -> None:
    source_verdict = tmp_path / "probe_9_verdict.json"
    source_verdict.write_text(
        json.dumps(_positive_uniward_verdict(), sort_keys=True),
        encoding="utf-8",
    )
    source_sha = __import__("hashlib").sha256(source_verdict.read_bytes()).hexdigest()
    payload = build_distortion_axis_probe_learned_sweep_candidates(
        [_positive_uniward_verdict(), _motion_neutral_verdict()],
        incumbent_score=INCUMBENT_SCORE,
        source_artifacts={
            "verdict_000": {
                "path": str(source_verdict),
                "sha256": source_sha,
                "bytes": source_verdict.stat().st_size,
            }
        },
    )
    plan = build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=INCUMBENT_SCORE,
        candidate_payloads=[payload],
        top_k=8,
    )

    observation = build_distortion_axis_probe_feedback_observation(
        plan=plan,
        candidate_payload=payload,
    )

    assert observation["schema"] == "mlx_dynamic_sweep_observation.v1"
    assert observation["candidate_id"] == (
        "distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1"
    )
    assert observation["sweep_config_id"] == "macos_cpu_advisory"
    assert observation["optimization_pass_id"] == "smoke"
    assert observation["observed_axis"] == "macos_cpu_advisory"
    assert observation["observed_score_or_delta"] == pytest.approx(-0.010)
    assert observation["segnet_delta"] == pytest.approx(-0.010)
    assert observation["posenet_delta"] == 0.0
    assert observation["rate_delta"] == 0.0
    assert observation["score_claim"] is False
    assert observation["ready_for_exact_eval_dispatch"] is False
    assert observation["archive_identity_semantics"] == (
        "hash_identity_for_probe_feedback_not_submission_archive"
    )
    assert observation["feedback_semantics"] == (
        "local_distortion_probe_feedback_only_not_scorer_execution"
    )

    replan = build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=INCUMBENT_SCORE,
        candidate_payloads=[payload],
        observations=[observation],
        top_k=8,
    )
    assert replan["summary"]["observation_row_count"] == 1
    assert replan["summary"]["suppressed_observed_row_count"] == 1
    suppressed = replan["suppressed_observed_sweep_rows"][0]
    assert suppressed["sweep_config_id"] == "macos_cpu_advisory"
    assert suppressed["optimization_pass_id"] == "smoke"


def test_feedback_cli_writes_observation_summary_and_replan(tmp_path: Path) -> None:
    probe9 = tmp_path / "probe_9_verdict.json"
    probe10 = tmp_path / "probe_10_verdict.json"
    payload_path = tmp_path / "distortion_axis_probe_learned_sweep_candidates.json"
    plan_path = tmp_path / "distortion_axis_probe_learned_sweep_plan.json"
    observation_jsonl = tmp_path / "distortion_axis_probe_learned_sweep_observations.jsonl"
    summary_path = tmp_path / "distortion_axis_probe_learned_sweep_feedback_summary.json"
    replan_path = tmp_path / "distortion_axis_probe_learned_sweep_replan.json"
    probe9.write_text(json.dumps(_positive_uniward_verdict()), encoding="utf-8")
    probe10.write_text(json.dumps(_motion_neutral_verdict()), encoding="utf-8")
    adapt = subprocess.run(
        [
            sys.executable,
            "tools/adapt_distortion_axis_probes_to_learned_sweep.py",
            "--verdict",
            str(probe9),
            "--verdict",
            str(probe10),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
            "--json-out",
            str(payload_path),
            "--plan-json-out",
            str(plan_path),
            "--plan-top-k",
            "8",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert adapt.returncode == 0, adapt.stderr

    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_distortion_axis_probe_learned_sweep_feedback.py",
            "--plan",
            str(plan_path),
            "--candidate-payload",
            str(payload_path),
            "--observation-jsonl",
            str(observation_jsonl),
            "--summary-json-out",
            str(summary_path),
            "--replan-json-out",
            str(replan_path),
            "--incumbent-score",
            str(INCUMBENT_SCORE),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["schema"] == "distortion_axis_probe_learned_sweep_feedback.v1"
    assert stdout["score_claim"] is False
    assert stdout["ready_for_exact_eval_dispatch"] is False
    rows = load_observation_rows(observation_jsonl)
    assert len(rows) == 1
    assert rows[0]["candidate_id"].startswith("distortion_axis:uniward")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["replan"]["suppressed_observed_row_count"] == 1
    replan = json.loads(replan_path.read_text(encoding="utf-8"))
    assert replan["summary"]["suppressed_observed_row_count"] == 1
