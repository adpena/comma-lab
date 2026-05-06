"""Static safety checks for the Alpha-Geo-0 pose-regeneration lane."""
from __future__ import annotations

from pathlib import Path


SCRIPT = Path("scripts/remote_lane_12_alpha_geo0_pose_regen.sh")


def _text() -> str:
    return SCRIPT.read_text()


def test_alpha_geo0_script_exists_and_is_not_nerv_retraining() -> None:
    text = _text()

    assert "lane_12_alpha_geo0_pose_regen" in text
    assert "experiments/train_nerv_mask.py" not in text
    assert "masks.nrv" in text
    assert "--candidate-member masks.nrv" in text


def test_alpha_geo0_regenerates_poses_against_decoded_candidate_masks() -> None:
    text = _text()

    assert "CANDIDATE_MASKS_PT" in text
    assert "--threshold-preset promotion" in text
    assert "experiments/optimize_poses.py" in text
    assert '--masks "$CANDIDATE_MASKS_PT"' in text
    assert '--gt-pose-targets "$GT_POSE_TARGETS"' in text
    assert "--device cuda" in text


def test_alpha_geo0_runs_canonical_cuda_auth_eval_and_adjudication() -> None:
    text = _text()

    assert "experiments/contest_auth_eval.py" in text
    assert "--inflate-sh submissions/robust_current/inflate.sh" in text
    assert "--upstream-dir upstream" in text
    assert "--device cuda" in text
    assert "scripts/adjudicate_contest_auth_eval.py" in text
    assert "--required-device cuda" in text
    assert "--required-samples 600" in text


def test_alpha_geo0_archive_members_are_deterministic() -> None:
    text = _text()

    assert 'members = ("renderer.bin", "masks.nrv", "optimized_poses.bin")' in text
    assert "zipfile.ZIP_DEFLATED" in text
    assert "date_time=(1980, 1, 1, 0, 0, 0)" in text
    assert "validate_archive(dst, manifest=detect_pose_manifest(dst), strict=True)" in text
