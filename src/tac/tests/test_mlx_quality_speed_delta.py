# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.auth_eval_schema import contest_formula_score
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_quality_speed_delta import (
    SCHEMA_VERSION,
    build_quality_speed_delta_manifest,
)

REPO = Path(__file__).resolve().parents[3]


def _anchor() -> dict[str, object]:
    return {
        "schema_version": "local_cpu_advisory.v1",
        "score_axis": "cpu_advisory",
        "evidence_grade": "[macOS-CPU advisory]",
        "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
        "canonical_score": contest_formula_score(
            seg_dist=0.00055991,
            pose_dist=0.00002943,
            archive_bytes=178_558,
        ),
        "avg_segnet_dist": 0.00055991,
        "avg_posenet_dist": 0.00002943,
        "archive_size_bytes": 178_558,
        "archive_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": "b" * 64,
        "raw_sha256": "e" * 64,
        "n_samples": 600,
        "contest_auth_eval_elapsed_seconds": 529.5,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _mlx_response(
    *,
    score_offset: float = 1.0e-5,
    batch_pairs: int = 1,
    hardware_substrate: str = "MLX cpu",
    audited: bool = True,
) -> dict[str, object]:
    seg = 0.00055991 + score_offset / 100.0
    return {
        "schema_version": "mlx_scorer_response.v1",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "hardware_substrate": hardware_substrate,
        "canonical_score": contest_formula_score(
            seg_dist=seg,
            pose_dist=0.00002943,
            archive_bytes=178_558,
        ),
        "avg_segnet_dist": seg,
        "avg_posenet_dist": 0.00002943,
        "archive_size_bytes": 178_558,
        "archive_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": "b" * 64,
        "raw_sha256": "e" * 64,
        "n_samples": 600,
        "batch_pairs": batch_pairs,
        "elapsed_seconds": 33.0,
        "pair_window": [0, 600],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "components": {
            "posenet_sha256": "c" * 64,
            "segnet_sha256": "d" * 64,
        },
        "cache_identity": {
            "candidate": {
                "path": "cache",
                "archive_sha256": "a" * 64,
                "inflated_outputs_aggregate_sha256": "b" * 64,
                "raw_sha256": "e" * 64,
                "pair_count": 600,
                "eligible_for_local_mlx_transfer_calibration": audited,
                "auth_eval_identity_audit": {
                    "verdict": "PASS_CACHE_AUTH_EVAL_IDENTITY",
                    "passed": audited,
                    "identity_residual": 0,
                }
                if audited
                else None,
            }
        },
    }


def _calibration() -> dict[str, object]:
    return {
        "schema_version": "mlx_score_calibration.v1",
        "rows": [{}, {}, {}, {}],
        "summary": {
            "recommended_min_mlx_gap_for_spend_triage": 1.0e-4,
            "calibration_uncertainty_score": 2.0e-5,
            "mlx_minus_cpu_max_abs": 2.0e-5,
            "mlx_minus_local_cpu_max_abs": 2.0e-6,
            "mlx_cpu_rank_inversions": 0,
            "cuda_cpu_rank_inversions": 0,
            "mlx_spend_triage_pairwise_uncertain_count": 0,
            "mlx_spend_triage_pairwise_certified_count": 3,
        },
        "decision_policy": {
            "allowed_use": "local_spend_triage_only_after_strict_auth_axis_calibration",
            "recommended_min_mlx_gap_for_spend_triage": 1.0e-4,
            "calibration_uncertainty_score": 2.0e-5,
            "forbidden_use": "score_claim_or_rank_or_kill_or_promotion",
        },
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
    }


def _weak_calibration() -> dict[str, object]:
    return {
        "schema_version": "mlx_public_frontier_score_calibration.v1",
        "rows": [{}, {}, {}, {}],
        "summary": {
            "max_abs_mlx_minus_cpu": 2.0e-5,
            "max_abs_mlx_minus_local_cpu": 2.0e-6,
            "mean_mlx_minus_cpu": 1.0e-6,
            "mlx_cpu_rank_inversions": 0,
            "cuda_cpu_rank_inversions": 0,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_quality_speed_delta_allows_audited_cpu_singleton_inside_band() -> None:
    manifest = build_quality_speed_delta_manifest(
        anchor_payload=_anchor(),
        mlx_payloads=[_mlx_response(score_offset=5.0e-5)],
        calibration_payload=_calibration(),
        calibration_safety_factor=5.0,
    )

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["score_claim"] is False
    row = manifest["rows"][0]
    assert row["spend_triage_allowed"] is True
    assert row["speedup_vs_anchor_elapsed"] and row["speedup_vs_anchor_elapsed"] > 10.0
    assert manifest["calibration"]["decision_band"] == 1.0e-4


def test_quality_speed_delta_blocks_weak_calibration_for_spend_triage() -> None:
    manifest = build_quality_speed_delta_manifest(
        anchor_payload=_anchor(),
        mlx_payloads=[_mlx_response(score_offset=5.0e-5)],
        calibration_payload=_weak_calibration(),
        calibration_safety_factor=5.0,
    )

    row = manifest["rows"][0]
    assert row["spend_triage_allowed"] is False
    assert "strict_cuda_auth_axis_calibration_missing" in row["blockers"]
    assert manifest["calibration"]["decision_band"] is None


def test_quality_speed_delta_blocks_gpu_batch_and_unaudited_out_of_band() -> None:
    manifest = build_quality_speed_delta_manifest(
        anchor_payload=_anchor(),
        mlx_payloads=[
            _mlx_response(
                score_offset=4.0e-4,
                batch_pairs=8,
                hardware_substrate="MLX gpu",
                audited=False,
            )
        ],
        calibration_payload=_calibration(),
    )

    row = manifest["rows"][0]
    assert row["spend_triage_allowed"] is False
    assert "candidate_cache_missing_pass_cache_auth_eval_identity" in row["blockers"]
    assert "mlx_gpu_response_requires_separate_cpu_transfer_calibration" in row["blockers"]
    assert "mlx_non_singleton_batch_shape_requires_passing_invariance_gate" in row["blockers"]
    assert "score_delta_exceeds_calibration_decision_band" in row["blockers"]
    assert manifest["summary"]["all_rows_blocked_for_spend_triage"] is True


def test_quality_speed_delta_labels_local_advisory_identity_as_non_auth_axis() -> None:
    mlx = _mlx_response(audited=False)
    candidate = mlx["cache_identity"]["candidate"]  # type: ignore[index]
    candidate["eligible_for_local_mlx_transfer_calibration"] = False
    candidate["eligible_for_local_mlx_local_advisory_debug"] = True
    candidate["local_cpu_advisory_cache_identity_audit"] = {
        "verdict": "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY",
        "passed": True,
    }

    manifest = build_quality_speed_delta_manifest(
        anchor_payload=_anchor(),
        mlx_payloads=[mlx],
        calibration_payload=_calibration(),
    )

    row = manifest["rows"][0]
    assert row["spend_triage_allowed"] is False
    assert "local_advisory_cache_identity_not_auth_axis" in row["blockers"]
    assert "candidate_cache_missing_pass_cache_auth_eval_identity" not in row["blockers"]


def test_quality_speed_delta_blocks_identity_and_sample_mismatch() -> None:
    mlx = _mlx_response()
    mlx["archive_sha256"] = "f" * 64
    mlx["n_samples"] = 1
    mlx["pair_window"] = [10, 11]
    mlx["cache_identity"]["candidate"]["archive_sha256"] = "f" * 64  # type: ignore[index]
    mlx["cache_identity"]["candidate"]["pair_count"] = 1  # type: ignore[index]

    manifest = build_quality_speed_delta_manifest(
        anchor_payload=_anchor(),
        mlx_payloads=[mlx],
        calibration_payload=_calibration(),
        calibration_safety_factor=5.0,
    )

    row = manifest["rows"][0]
    assert row["spend_triage_allowed"] is False
    assert "archive_sha256_identity_mismatch" in row["blockers"]
    assert "mlx_n_samples_not_full_contest" in row["blockers"]
    assert "mlx_n_samples_mismatch_anchor" in row["blockers"]
    assert "mlx_pair_window_not_full_contest" in row["blockers"]
    assert "candidate_cache_pair_count_not_full_contest" in row["blockers"]


def test_quality_speed_delta_cli(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.json"
    mlx = tmp_path / "mlx.json"
    calibration = tmp_path / "calibration.json"
    output = tmp_path / "delta.json"
    anchor.write_text(json.dumps(_anchor()), encoding="utf-8")
    mlx.write_text(json.dumps(_mlx_response()), encoding="utf-8")
    calibration.write_text(json.dumps(_calibration()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "compare_mlx_quality_speed_delta.py"),
            "--anchor",
            str(anchor),
            "--mlx-response",
            str(mlx),
            "--calibration-summary",
            str(calibration),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout = json.loads(completed.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["row_count"] == 1
    assert stdout["score_claim"] is False
    assert payload["rows"][0]["spend_triage_allowed"] is True
