# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_score_calibration import (
    build_mlx_score_calibration_manifest,
)

REPO = Path(__file__).resolve().parents[3]


def test_mlx_score_calibration_preserves_order_and_false_authority(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "a", mlx_score=0.2, cpu_score=0.201, cuda_score=0.201),
        _row(tmp_path, "b", mlx_score=0.3, cpu_score=0.301, cuda_score=0.301),
    ]

    manifest = build_mlx_score_calibration_manifest(rows, repo_root=tmp_path, run_id="unit")

    assert manifest["run_id"] == "unit"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["summary"]["mlx_cpu_rank_inversions"] == 0
    assert manifest["summary"]["mlx_cuda_rank_inversions"] == 0
    assert manifest["summary"]["mlx_minus_cpu_max_abs"] == 0.0010000000000000009
    assert manifest["decision_policy"]["recommended_min_mlx_gap_for_spend_triage"] == (
        0.0010000000000000009 * 5.0
    )
    assert manifest["summary"]["mlx_spend_triage_pairwise_certified_count"] == 1
    assert manifest["pairwise_order"][0]["mlx_spend_triage_decision_certified"] is True
    assert manifest["rows"][0]["mlx_rank"] == 1
    assert manifest["rows"][0]["cpu_rank"] == 1
    assert manifest["rows"][0]["cuda_rank"] == 1


def test_mlx_score_calibration_counts_rank_inversions(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "a", mlx_score=0.2, cpu_score=0.31, cuda_score=0.21),
        _row(tmp_path, "b", mlx_score=0.3, cpu_score=0.20, cuda_score=0.32),
    ]

    manifest = build_mlx_score_calibration_manifest(rows, repo_root=tmp_path)

    assert manifest["summary"]["mlx_cpu_rank_inversions"] == 1
    assert manifest["summary"]["mlx_cuda_rank_inversions"] == 0


def test_mlx_score_calibration_marks_close_pairs_uncertain(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "a", mlx_score=0.200, cpu_score=0.201, cuda_score=0.201),
        _row(tmp_path, "b", mlx_score=0.203, cpu_score=0.204, cuda_score=0.204),
    ]

    manifest = build_mlx_score_calibration_manifest(rows, repo_root=tmp_path)

    assert manifest["summary"]["mlx_cpu_rank_inversions"] == 0
    assert manifest["summary"]["recommended_min_mlx_gap_for_spend_triage"] == (
        0.0010000000000000009 * 5.0
    )
    assert manifest["summary"]["mlx_spend_triage_pairwise_certified_count"] == 0
    assert manifest["summary"]["mlx_spend_triage_pairwise_uncertain_count"] == 1
    assert manifest["pairwise_order"][0]["mlx_spend_triage_uncertain"] is True
    assert manifest["pairwise_order"][0]["mlx_matches_cpu"] is True


def test_mlx_score_calibration_rejects_authoritative_mlx_response(tmp_path: Path) -> None:
    row = _row(tmp_path, "bad", mlx_score=0.2, cpu_score=0.2, cuda_score=0.3)
    response_path = tmp_path / row["mlx_response_path"]
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    payload["score_claim"] = True
    response_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_mlx_score_calibration_manifest([row], repo_root=tmp_path)
    except ValueError as exc:
        assert "score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("authoritative MLX response was accepted")


def test_mlx_score_calibration_rejects_direct_auth_score_scalars(tmp_path: Path) -> None:
    row = _row(tmp_path, "bad", mlx_score=0.2, cpu_score=0.2, cuda_score=0.3)
    row["cpu_score"] = 0.2
    row.pop("cpu_auth_eval_path")

    try:
        build_mlx_score_calibration_manifest([row], repo_root=tmp_path)
    except ValueError as exc:
        assert "cpu_score direct scalar is not accepted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("direct scalar CPU score was accepted")


def test_mlx_score_calibration_rejects_non_auth_payload(tmp_path: Path) -> None:
    row = _row(tmp_path, "bad", mlx_score=0.2, cpu_score=0.2, cuda_score=0.3)
    cpu_path = tmp_path / row["cpu_auth_eval_path"]
    payload = json.loads(cpu_path.read_text(encoding="utf-8"))
    payload["evidence_grade"] = "macOS-MLX"
    payload["score_claim_valid"] = False
    cpu_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_mlx_score_calibration_manifest([row], repo_root=tmp_path)
    except ValueError as exc:
        assert "not a strict cpu contest auth-axis source" in str(exc)
        assert "auth_eval_evidence_grade_not_contest_cpu_or_cuda" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-auth CPU payload was accepted")


def test_mlx_score_calibration_cli(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "a", mlx_score=0.2, cpu_score=0.201, cuda_score=0.201),
        _row(tmp_path, "b", mlx_score=0.3, cpu_score=0.301, cuda_score=0.301),
    ]
    input_path = tmp_path / "rows.json"
    output_path = tmp_path / "calibration.json"
    input_path.write_text(json.dumps({"run_id": "cli", "rows": rows}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "calibrate_mlx_scorer_response_scores.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--repo-root",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"score_claim": false' in completed.stdout
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "cli"
    assert manifest["summary"]["mlx_cpu_rank_inversions"] == 0
    assert manifest["summary"]["recommended_min_mlx_gap_for_spend_triage"] == (
        0.0010000000000000009 * 5.0
    )


def _row(
    tmp_path: Path,
    label: str,
    *,
    mlx_score: float,
    cpu_score: float,
    cuda_score: float,
) -> dict:
    response_path = tmp_path / f"{label}_mlx.json"
    cpu_path = tmp_path / f"{label}_cpu_auth.json"
    cuda_path = tmp_path / f"{label}_cuda_auth.json"
    response_path.write_text(
        json.dumps(
            {
                "schema_version": "mlx_scorer_response.v1",
                "evidence_grade": EVIDENCE_GRADE_MLX,
                "evidence_tag": EVIDENCE_TAG_MLX,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "candidate_generation_only": True,
                "canonical_score": mlx_score,
                "avg_posenet_dist": 1.0e-4,
                "avg_segnet_dist": 2.0e-4,
                "archive_size_bytes": 123,
                "n_samples": 600,
                "batch_pairs": 1,
                "archive_sha256": "a" * 64,
                "inflated_outputs_aggregate_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    cpu_path.write_text(json.dumps(_auth_eval_payload("cpu", cpu_score)), encoding="utf-8")
    cuda_path.write_text(json.dumps(_auth_eval_payload("cuda", cuda_score)), encoding="utf-8")
    return {
        "label": label,
        "mlx_response_path": response_path.name,
        "cpu_auth_eval_path": cpu_path.name,
        "cuda_auth_eval_path": cuda_path.name,
    }


def _auth_eval_payload(axis: str, score: float) -> dict:
    archive_size = 0
    pose = 0.0
    seg = float(score) / 100.0
    canonical_score = contest_formula_score(
        seg_dist=seg,
        pose_dist=pose,
        archive_bytes=archive_size,
    )
    payload = {
        "canonical_score": canonical_score,
        "canonical_score_source": "score_recomputed_from_components",
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": archive_size,
        "score_rate_contribution": 25.0 * archive_size / ORIGINAL_VIDEO_BYTES,
        "rate_unscaled": archive_size / ORIGINAL_VIDEO_BYTES,
        "n_samples": 600,
        "score_claim": True,
        "score_claim_valid": True,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    if axis == "cpu":
        payload.update(
            {
                "evidence_grade": "contest-CPU",
                "lane_tag": "[contest-CPU]",
                "score_axis": "contest_cpu",
                "evidence_semantics": "public_leaderboard_cpu_reproduction",
                "cpu_leaderboard_reproduction_eligible": True,
                "provenance": {
                    "device": "cpu",
                    "platform_system": "Linux",
                    "platform_machine": "x86_64",
                },
            }
        )
    elif axis == "cuda":
        payload.update(
            {
                "evidence_grade": "contest-CUDA",
                "lane_tag": "[contest-CUDA]",
                "score_axis": "contest_cuda",
                "evidence_semantics": "contest_cuda_exact_auth_eval",
                "exact_cuda_eval_complete": True,
                "provenance": {
                    "device": "cuda",
                    "gpu_t4_match": True,
                },
            }
        )
    else:  # pragma: no cover
        raise ValueError(axis)
    return payload
