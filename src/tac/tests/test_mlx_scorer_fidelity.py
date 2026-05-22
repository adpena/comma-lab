# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_fidelity import (
    FAIL_VERDICT,
    PASS_VERDICT,
    MLXScorerFidelityThresholds,
    build_mlx_scorer_training_signal_fidelity_manifest,
)

REPO = Path(__file__).resolve().parents[3]


def _payload(
    *,
    archive_sha256: str = "a" * 64,
    inflated_sha256: str = "b" * 64,
    archive_size_bytes: int = 178_417,
    seg_avg: float = 0.00056029,
    pose_avg: float = 0.00029137,
    n_samples: int = 600,
    evidence_grade: str | None = None,
    score_claim: bool = False,
) -> dict[str, object]:
    score = contest_formula_score(
        seg_dist=seg_avg,
        pose_dist=pose_avg,
        archive_bytes=archive_size_bytes,
    )
    payload: dict[str, object] = {
        "canonical_score": score,
        "canonical_score_source": "score_recomputed_from_components",
        "avg_segnet_dist": seg_avg,
        "avg_posenet_dist": pose_avg,
        "score_rate_contribution": 25.0 * archive_size_bytes / ORIGINAL_VIDEO_BYTES,
        "rate_unscaled": archive_size_bytes / ORIGINAL_VIDEO_BYTES,
        "archive_size_bytes": archive_size_bytes,
        "n_samples": n_samples,
        "archive_sha256": archive_sha256,
        "inflated_outputs_aggregate_sha256": inflated_sha256,
        "score_claim": score_claim,
    }
    if evidence_grade is not None:
        payload["evidence_grade"] = evidence_grade
        payload["evidence_tag"] = EVIDENCE_TAG_MLX
        payload["score_axis"] = EVIDENCE_TAG_MLX
    return payload


def test_mlx_fidelity_passes_near_exact_byte_closed_signal() -> None:
    manifest = build_mlx_scorer_training_signal_fidelity_manifest(
        _payload(evidence_grade=EVIDENCE_GRADE_MLX),
        _payload(),
        run_id="fixture",
    )

    assert manifest["passed"] is True
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["score_claim"] is False
    assert manifest["score_claim_valid"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["candidate_generation_only"] is True
    assert "prepaid_dispatch_spend_filter" in manifest["device_contract"]["allowed_uses"]
    assert manifest["byte_closure"]["archive_sha256"]["match"] is True
    assert manifest["signal_exposure"]["tightest_failed_axis"] is None


def test_mlx_fidelity_fails_archive_identity_mismatch() -> None:
    manifest = build_mlx_scorer_training_signal_fidelity_manifest(
        _payload(archive_sha256="c" * 64, evidence_grade=EVIDENCE_GRADE_MLX),
        _payload(),
    )

    assert manifest["passed"] is False
    assert manifest["verdict"] == FAIL_VERDICT
    assert "archive_sha256_identity_mismatch_or_missing" in manifest["blockers"]


def test_mlx_fidelity_fails_seg_axis_drift() -> None:
    manifest = build_mlx_scorer_training_signal_fidelity_manifest(
        _payload(seg_avg=0.00057029, evidence_grade=EVIDENCE_GRADE_MLX),
        _payload(),
    )

    assert manifest["passed"] is False
    assert any(
        blocker.startswith("score_delta_exceeds_threshold")
        or blocker.startswith("seg_contribution_delta_exceeds_threshold")
        for blocker in manifest["blockers"]
    )
    assert manifest["signal_exposure"]["tightest_failed_axis"] in {"score", "seg_contribution"}


def test_mlx_fidelity_fails_false_authority_flags() -> None:
    manifest = build_mlx_scorer_training_signal_fidelity_manifest(
        _payload(evidence_grade=EVIDENCE_GRADE_MLX, score_claim=True),
        _payload(),
    )

    assert manifest["passed"] is False
    assert "mlx_payload_attempts_score_claim" in manifest["blockers"]
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False


def test_mlx_fidelity_requires_mlx_evidence_grade() -> None:
    manifest = build_mlx_scorer_training_signal_fidelity_manifest(
        _payload(),
        _payload(),
    )

    assert manifest["passed"] is False
    assert "mlx_evidence_grade_missing" in manifest["blockers"]


def test_mlx_fidelity_requires_mlx_axis_tags() -> None:
    mlx_payload = _payload(evidence_grade=EVIDENCE_GRADE_MLX)
    mlx_payload["evidence_tag"] = "[contest-CUDA]"
    mlx_payload["score_axis"] = "[contest-CUDA]"

    manifest = build_mlx_scorer_training_signal_fidelity_manifest(
        mlx_payload,
        _payload(),
    )

    assert manifest["passed"] is False
    assert f"mlx_evidence_tag_not_{EVIDENCE_TAG_MLX}" in manifest["blockers"]
    assert f"mlx_score_axis_not_{EVIDENCE_TAG_MLX}" in manifest["blockers"]


def test_mlx_fidelity_allow_missing_inflated_identity_does_not_allow_present_mismatch() -> None:
    manifest = build_mlx_scorer_training_signal_fidelity_manifest(
        _payload(evidence_grade=EVIDENCE_GRADE_MLX, inflated_sha256="b" * 64),
        _payload(inflated_sha256="c" * 64),
        thresholds=MLXScorerFidelityThresholds(require_inflated_output_identity=False),
    )

    assert manifest["passed"] is False
    assert "inflated_outputs_aggregate_sha256_identity_present_mismatch" in manifest["blockers"]


def test_mlx_fidelity_cli_writes_manifest(tmp_path: Path) -> None:
    mlx_path = tmp_path / "mlx.json"
    auth_path = tmp_path / "auth.json"
    out_path = tmp_path / "manifest.json"
    mlx_path.write_text(json.dumps(_payload(evidence_grade=EVIDENCE_GRADE_MLX)), encoding="utf-8")
    auth_path.write_text(json.dumps(_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "check_mlx_scorer_fidelity.py"),
            "--mlx-payload",
            str(mlx_path),
            "--auth-eval",
            str(auth_path),
            "--output",
            str(out_path),
            "--run-id",
            "fixture_cli",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"passed": true' in completed.stdout
    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert manifest["passed"] is True
    assert manifest["run_id"] == "fixture_cli"
