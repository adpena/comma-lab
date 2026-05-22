# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

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
        _row(tmp_path, "c", mlx_score=0.4, cpu_score=0.401, cuda_score=0.401),
    ]

    manifest = build_mlx_score_calibration_manifest(rows, repo_root=tmp_path, run_id="unit")

    assert manifest["run_id"] == "unit"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["summary"]["mlx_cpu_rank_inversions"] == 0
    assert manifest["summary"]["mlx_cuda_rank_inversions"] == 0
    assert manifest["summary"]["mlx_minus_cpu_max_abs"] == pytest.approx(0.001)
    assert manifest["summary"]["axis_specific_min_gap_for_spend_triage"][
        "cpu"
    ] == pytest.approx(0.001 * 5.0)
    assert manifest["summary"]["axis_specific_min_gap_for_spend_triage"][
        "cuda"
    ] == pytest.approx(0.001 * 5.0)
    assert manifest["decision_policy"][
        "recommended_min_mlx_gap_for_spend_triage"
    ] == pytest.approx(0.001 * 5.0)
    assert manifest["summary"]["mlx_spend_triage_pairwise_certified_count"] == 3
    assert manifest["pairwise_order"][0]["mlx_spend_triage_decision_certified"] is True
    assert manifest["pairwise_order"][0]["mlx_cpu_spend_triage_decision_certified"] is True
    assert manifest["pairwise_order"][0]["mlx_cuda_spend_triage_decision_certified"] is True
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
        _row(tmp_path, "c", mlx_score=0.204, cpu_score=0.205, cuda_score=0.205),
    ]

    manifest = build_mlx_score_calibration_manifest(rows, repo_root=tmp_path)

    assert manifest["summary"]["mlx_cpu_rank_inversions"] == 0
    assert manifest["summary"]["recommended_min_mlx_gap_for_spend_triage"] == pytest.approx(
        0.001 * 5.0
    )
    assert manifest["summary"]["mlx_spend_triage_pairwise_certified_count"] == 0
    assert manifest["summary"]["mlx_spend_triage_pairwise_uncertain_count"] == 3
    assert manifest["pairwise_order"][0]["mlx_spend_triage_uncertain"] is True
    assert manifest["pairwise_order"][0]["mlx_matches_cpu"] is True


def test_mlx_score_calibration_reports_dual_axis_discordance_without_conflation(
    tmp_path: Path,
) -> None:
    rows = [
        _row(tmp_path, "a", mlx_score=0.20, cpu_score=0.201, cuda_score=0.31),
        _row(tmp_path, "b", mlx_score=0.25, cpu_score=0.251, cuda_score=0.26),
        _row(tmp_path, "c", mlx_score=0.30, cpu_score=0.301, cuda_score=0.36),
    ]

    manifest = build_mlx_score_calibration_manifest(rows, repo_root=tmp_path)

    summary = manifest["summary"]
    axis_min_gap = summary["axis_specific_min_gap_for_spend_triage"]
    assert axis_min_gap["cpu"] == pytest.approx(0.001 * 5.0)
    assert axis_min_gap["cuda"] == pytest.approx(0.11 * 5.0)
    assert manifest["decision_policy"][
        "recommended_min_mlx_gap_for_spend_triage"
    ] == pytest.approx(axis_min_gap["cuda"])
    assert summary["mlx_cpu_rank_inversions"] == 0
    assert summary["mlx_cuda_rank_inversions"] == 1
    dual = summary["planning_advisory_dual_axis_summary"]["overall"]
    assert dual["cuda_regression_given_cpu_improvement_count"] == 1
    assert dual["cpu_improvement_comparison_count"] == 3
    assert dual["p_cuda_regression_given_cpu_improvement_empirical"] == pytest.approx(1 / 3)
    assert dual[
        "p_cuda_regression_given_cpu_improvement_conservative_count_based"
    ] == pytest.approx(2 / 5)
    assert dual["sample_scarce"] is False


def test_mlx_score_calibration_cpu_only_does_not_authorize_cuda_routing(
    tmp_path: Path,
) -> None:
    rows = [
        _row(tmp_path, "a", mlx_score=0.20, cpu_score=0.201, cuda_score=0.301),
        _row(tmp_path, "b", mlx_score=0.30, cpu_score=0.301, cuda_score=0.401),
        _row(tmp_path, "c", mlx_score=0.40, cpu_score=0.401, cuda_score=0.501),
    ]
    for row in rows:
        row.pop("cuda_auth_eval_path")

    manifest = build_mlx_score_calibration_manifest(rows, repo_root=tmp_path)

    decision = manifest["decision_policy"]
    assert decision["recommended_min_mlx_gap_for_spend_triage"] is None
    assert decision["allowed_use"] == (
        "diagnostic_only_cuda_auth_axis_calibration_missing_or_insufficient"
    )
    assert "cpu_only_calibration_cannot_authorize_cuda_routing" in decision["blockers"]
    assert decision["axis_decision_policies"]["cpu"]["spend_triage_allowed"] is True
    assert decision["axis_decision_policies"]["cuda"]["spend_triage_allowed"] is False
    assert manifest["summary"]["axis_specific_min_gap_for_spend_triage"][
        "cpu"
    ] == pytest.approx(0.001 * 5.0)
    assert manifest["summary"]["axis_specific_min_gap_for_spend_triage"]["cuda"] is None
    assert manifest["summary"]["mlx_cpu_spend_triage_pairwise_certified_count"] == 3
    assert manifest["summary"]["mlx_spend_triage_pairwise_certified_count"] == 0


def test_mlx_score_calibration_sample_scarcity_fails_closed_and_warns(
    tmp_path: Path,
) -> None:
    rows = [
        _row(tmp_path, "a", mlx_score=0.2, cpu_score=0.201, cuda_score=0.201),
        _row(tmp_path, "b", mlx_score=0.3, cpu_score=0.301, cuda_score=0.301),
    ]

    manifest = build_mlx_score_calibration_manifest(rows, repo_root=tmp_path)

    decision = manifest["decision_policy"]
    assert decision["recommended_min_mlx_gap_for_spend_triage"] is None
    assert "cuda_auth_axis_calibration_sample_scarce" in decision["blockers"]
    assert any("cuda_auth_axis_calibration_sample_scarce" in item for item in decision["warnings"])
    assert decision["axis_decision_policies"]["cuda"]["sample_scarce"] is True
    dual = manifest["summary"]["planning_advisory_dual_axis_summary"]["overall"]
    assert dual["sample_scarce"] is True
    assert any("dual_axis_cpu_cuda_sample_scarce_fail_closed" in item for item in dual["warnings"])
    assert manifest["summary"]["mlx_spend_triage_pairwise_certified_count"] == 0


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


def test_mlx_score_calibration_accepts_matching_direct_scalar_with_strict_payload(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path, "a", mlx_score=0.2, cpu_score=0.201, cuda_score=0.201)
    row["cpu_score"] = 0.201

    manifest = build_mlx_score_calibration_manifest([row], repo_root=tmp_path)

    assert manifest["rows"][0]["cpu_score"] == pytest.approx(0.201)
    assert manifest["rows"][0]["cpu_source"] == row["cpu_auth_eval_path"]
    assert manifest["decision_policy"]["allowed_use"] == (
        "diagnostic_only_cuda_auth_axis_calibration_missing_or_insufficient"
    )
    assert "cuda_auth_axis_calibration_sample_scarce" in manifest["decision_policy"]["blockers"]
    assert manifest["decision_policy"]["score_claim"] is False


def test_mlx_score_calibration_rejects_direct_scalar_auth_payload_mismatch(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path, "bad", mlx_score=0.2, cpu_score=0.2, cuda_score=0.3)
    row["cpu_score"] = 0.25

    try:
        build_mlx_score_calibration_manifest([row], repo_root=tmp_path)
    except ValueError as exc:
        assert "cpu_score direct scalar does not match strict auth-eval payload" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched direct scalar CPU score was accepted")


def test_mlx_score_calibration_rejects_non_auth_payload(tmp_path: Path) -> None:
    row = _row(tmp_path, "bad", mlx_score=0.2, cpu_score=0.2, cuda_score=0.3)
    cpu_path = tmp_path / row["cpu_auth_eval_path"]
    payload = json.loads(cpu_path.read_text(encoding="utf-8"))
    payload["evidence_grade"] = "macOS-MLX"
    payload["score_axis"] = "macOS-MLX"
    payload["score_claim_valid"] = False
    cpu_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_mlx_score_calibration_manifest([row], repo_root=tmp_path)
    except ValueError as exc:
        assert "not a strict cpu contest auth-axis source" in str(exc)
        assert "auth_eval_evidence_grade_not_contest_cpu_or_cuda" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-auth CPU payload was accepted")


def test_mlx_score_calibration_rejects_auth_payload_archive_size_mismatch(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path, "bad", mlx_score=0.2, cpu_score=0.2, cuda_score=0.3)
    cpu_path = tmp_path / row["cpu_auth_eval_path"]
    payload = json.loads(cpu_path.read_text(encoding="utf-8"))
    payload["archive_size_bytes"] = 124
    payload["rate_unscaled"] = 124 / ORIGINAL_VIDEO_BYTES
    payload["score_rate_contribution"] = 25.0 * 124 / ORIGINAL_VIDEO_BYTES
    payload["canonical_score"] = contest_formula_score(
        seg_dist=payload["avg_segnet_dist"],
        pose_dist=payload["avg_posenet_dist"],
        archive_bytes=124,
    )
    cpu_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_mlx_score_calibration_manifest([row], repo_root=tmp_path)
    except ValueError as exc:
        assert "archive_size_bytes_mismatch:manifest=124:actual=123" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched auth archive size was accepted")


def test_mlx_score_calibration_rejects_same_size_wrong_archive_sha(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path, "bad", mlx_score=0.2, cpu_score=0.2, cuda_score=0.3)
    cpu_path = tmp_path / row["cpu_auth_eval_path"]
    payload = json.loads(cpu_path.read_text(encoding="utf-8"))
    payload["archive_sha256"] = "f" * 64
    provenance = payload["provenance"]
    provenance["archive_sha256"] = "f" * 64
    cpu_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_mlx_score_calibration_manifest([row], repo_root=tmp_path)
    except ValueError as exc:
        assert "cpu_auth_eval_archive_sha256_mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("same-size wrong-archive auth payload was accepted")


def test_mlx_score_calibration_rejects_same_archive_wrong_inflated_surface(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path, "bad", mlx_score=0.2, cpu_score=0.2, cuda_score=0.3)
    cpu_path = tmp_path / row["cpu_auth_eval_path"]
    payload = json.loads(cpu_path.read_text(encoding="utf-8"))
    payload["inflated_outputs_aggregate_sha256"] = "c" * 64
    provenance = payload["provenance"]
    provenance["inflated_output_manifest"]["payload"]["aggregate_sha256"] = "c" * 64
    cpu_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_mlx_score_calibration_manifest([row], repo_root=tmp_path)
    except ValueError as exc:
        assert "cpu_auth_eval_inflated_outputs_aggregate_sha256_mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrong inflated-surface auth payload was accepted")


def test_mlx_score_calibration_rejects_incomplete_mlx_authority_tags(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path, "bad", mlx_score=0.2, cpu_score=0.2, cuda_score=0.3)
    response_path = tmp_path / row["mlx_response_path"]
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    payload.pop("score_axis")
    response_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_mlx_score_calibration_manifest([row], repo_root=tmp_path)
    except ValueError as exc:
        assert "score axis is not local MLX" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("MLX response with incomplete authority tags was accepted")


def test_mlx_score_calibration_rejects_unaudited_candidate_cache(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path, "bad", mlx_score=0.2, cpu_score=0.2, cuda_score=0.3)
    response_path = tmp_path / row["mlx_response_path"]
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    payload["cache_identity"]["candidate"]["eligible_for_local_mlx_transfer_calibration"] = False
    response_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_mlx_score_calibration_manifest([row], repo_root=tmp_path)
    except ValueError as exc:
        assert "candidate cache is not eligible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("MLX response with unaudited candidate cache was accepted")


def test_mlx_score_calibration_cli(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "a", mlx_score=0.2, cpu_score=0.201, cuda_score=0.201),
        _row(tmp_path, "b", mlx_score=0.3, cpu_score=0.301, cuda_score=0.301),
        _row(tmp_path, "c", mlx_score=0.4, cpu_score=0.401, cuda_score=0.401),
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
    assert manifest["summary"]["recommended_min_mlx_gap_for_spend_triage"] == pytest.approx(
        0.001 * 5.0
    )


def test_mlx_score_calibration_cli_rejects_partial_full_auth_rows(
    tmp_path: Path,
) -> None:
    rows = [_row(tmp_path, "a", mlx_score=0.2, cpu_score=0.201, cuda_score=0.201)]
    input_path = tmp_path / "rows.json"
    output_path = tmp_path / "calibration.json"
    input_path.write_text(
        json.dumps(
            {
                "run_id": "partial",
                "strict_full_calibration": False,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )

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
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "strict_full_calibration=false" in completed.stderr


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
    archive_size = 123
    response_path.write_text(
        json.dumps(
            {
                "schema_version": "mlx_scorer_response.v1",
                "evidence_grade": EVIDENCE_GRADE_MLX,
                "evidence_tag": EVIDENCE_TAG_MLX,
                "score_axis": EVIDENCE_TAG_MLX,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "promotable": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "candidate_generation_only": True,
                "requires_exact_eval_before_promotion": True,
                "canonical_score": mlx_score,
                "avg_posenet_dist": 1.0e-4,
                "avg_segnet_dist": 2.0e-4,
                "archive_size_bytes": archive_size,
                "n_samples": 600,
                "batch_pairs": 1,
                "pair_window": [0, 600],
                "response_family": "unit_calibration",
                "components": {
                    "posenet_shape": [600],
                    "segnet_shape": [600],
                    "posenet_sha256": "1" * 64,
                    "segnet_sha256": "2" * 64,
                },
                "archive_sha256": "a" * 64,
                "inflated_outputs_aggregate_sha256": "b" * 64,
                "cache_identity": {
                    "candidate": {
                        "archive_sha256": "a" * 64,
                        "inflated_outputs_aggregate_sha256": "b" * 64,
                        "raw_sha256": "c" * 64,
                        "path": "candidate/cache",
                        "pair_count": 600,
                        "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
                        "array_sha256": {
                            "pair_indices": "3" * 64,
                            "posenet_yuv6_pair": "4" * 64,
                            "segnet_last_rgb": "5" * 64,
                        },
                        "pair_indices_shape": [600, 2],
                        "posenet_yuv6_pair_shape": [600, 12, 192, 256],
                        "segnet_last_rgb_shape": [600, 3, 384, 512],
                        "eligible_for_local_mlx_transfer_calibration": True,
                        "auth_eval_identity_audit": {
                            "schema_version": "mlx_scorer_input_cache_auth_eval_audit.v1",
                            "verdict": "PASS_CACHE_AUTH_EVAL_IDENTITY",
                            "passed": True,
                            "identity_residual": 0,
                            "score_claim": False,
                            "score_claim_valid": False,
                            "promotion_eligible": False,
                            "promotable": False,
                            "rank_or_kill_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                        },
                    },
                    "reference": {
                        "pair_count": 600,
                        "path": "reference/cache",
                        "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
                        "array_sha256": {
                            "pair_indices": "6" * 64,
                            "posenet_yuv6_pair": "7" * 64,
                            "segnet_last_rgb": "8" * 64,
                        },
                        "pair_indices_shape": [600, 2],
                        "posenet_yuv6_pair_shape": [600, 12, 192, 256],
                        "segnet_last_rgb_shape": [600, 3, 384, 512],
                    },
                    "pair_indices_equal": True,
                },
            }
        ),
        encoding="utf-8",
    )
    cpu_path.write_text(
        json.dumps(_auth_eval_payload("cpu", cpu_score, archive_size)),
        encoding="utf-8",
    )
    cuda_path.write_text(
        json.dumps(_auth_eval_payload("cuda", cuda_score, archive_size)),
        encoding="utf-8",
    )
    return {
        "label": label,
        "mlx_response_path": response_path.name,
        "cpu_auth_eval_path": cpu_path.name,
        "cuda_auth_eval_path": cuda_path.name,
    }


def _auth_eval_payload(axis: str, score: float, archive_size: int) -> dict:
    pose = 0.0
    rate_score = 25.0 * archive_size / ORIGINAL_VIDEO_BYTES
    seg = (float(score) - rate_score) / 100.0
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
        "archive_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": "b" * 64,
        "score_rate_contribution": rate_score,
        "rate_unscaled": archive_size / ORIGINAL_VIDEO_BYTES,
        "n_samples": 600,
        "score_claim": True,
        "score_claim_valid": True,
        "promotion_eligible": False,
        "promotable": False,
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
                    "archive_sha256": "a" * 64,
                    "inflated_output_manifest": {
                        "payload": {"aggregate_sha256": "b" * 64}
                    },
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
                    "archive_sha256": "a" * 64,
                    "inflated_output_manifest": {
                        "payload": {"aggregate_sha256": "b" * 64}
                    },
                },
            }
        )
    else:  # pragma: no cover
        raise ValueError(axis)
    return payload
