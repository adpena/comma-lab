"""Controls for per-pair distortion retention.

The load-bearing property is not "a file was written" — it is "the retained vector reduces to
the number the scorer actually reported". Both directions are executed: a faithful vector
verifies, and a vector that does not reproduce the scalar is marked UNVERIFIED rather than
quietly persisted as though it were a map of the scored quantity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tac.pose_per_pair_retention import (
    retain_per_pair_distortion,
    verify_reduction,
)


def test_green_a_faithful_vector_reduces_to_the_reported_scalar() -> None:
    per_pair = [1e-5, 3e-5, 2e-5, 4e-5]

    check = verify_reduction(per_pair, reported=2.5e-5, label="posenet")

    assert check.verified is True
    assert check.recomputed == 2.5e-5
    assert check.relative_error is not None and check.relative_error < 1e-12
    assert "reproduces the reported scalar" in check.note


def test_red_a_vector_that_does_not_reproduce_the_scalar_is_unverified() -> None:
    """A per-pair map that does not add up to the scored number is worse than no map."""
    check = verify_reduction([1e-5, 1e-5], reported=9e-5, label="posenet")

    assert check.verified is False
    assert "UNVERIFIED" in check.note
    assert "NOT reproducing the scored computation" in check.note


def test_an_unanchored_map_is_not_silently_accepted() -> None:
    check = verify_reduction([1.0, 2.0], reported=None, label="segnet")

    assert check.verified is False
    assert "unanchored" in check.note


def test_an_empty_retention_is_a_failure_not_a_pass() -> None:
    check = verify_reduction([], reported=1.0, label="posenet")

    assert check.verified is False
    assert "nothing to verify" in check.note


def test_payload_is_written_with_sha256_before_anything_is_reported(tmp_path: Path) -> None:
    pose = [1e-5, 3e-5]
    seg = [2e-4, 4e-4]

    retention = retain_per_pair_distortion(
        tmp_path / "retained",
        per_pair_pose=pose,
        per_pair_seg=seg,
        reported_pose=2e-5,
        reported_seg=3e-4,
    )

    assert retention.verified is True
    assert retention.pairs == 2
    # ALWAYS KEEP THE PAYLOAD: the bytes exist, and their sha is recorded beside them.
    for key in ("per_pair_posenet_distortion", "per_pair_segnet_distortion"):
        path = Path(retention.payload_paths[key])
        assert path.is_file()
        assert len(retention.payload_sha256[key]) == 64
        assert retention.payload_bytes[key] == path.stat().st_size
    assert np.load(Path(retention.payload_paths["per_pair_posenet_distortion"])).tolist() == pose


def test_manifest_records_the_verification_and_disclaims_score_authority(tmp_path: Path) -> None:
    retain_per_pair_distortion(
        tmp_path / "retained",
        per_pair_pose=[1.0, 3.0],
        per_pair_seg=[1.0, 1.0],
        reported_pose=2.0,
        reported_seg=1.0,
    )
    manifest = json.loads((tmp_path / "retained" / "manifest.json").read_text())

    assert manifest["schema"] == "pact.per_pair_distortion_retention.v1"
    assert manifest["score_claim"] is False
    assert manifest["promotable"] is False
    assert manifest["verified"] is True
    assert manifest["pose_verification"]["reported"] == 2.0
    assert manifest["seg_verification"]["recomputed"] == 1.0


def test_a_mismatched_pass_is_persisted_but_flagged(tmp_path: Path) -> None:
    """Keep the bytes (they are evidence of the disagreement) and refuse to call them verified."""
    retention = retain_per_pair_distortion(
        tmp_path / "retained",
        per_pair_pose=[1.0, 1.0],
        per_pair_seg=[1.0, 1.0],
        reported_pose=5.0,
        reported_seg=1.0,
    )

    assert retention.verified is False
    assert retention.pose_verification.verified is False
    assert retention.seg_verification.verified is True
    assert Path(retention.payload_paths["per_pair_posenet_distortion"]).is_file()


def test_optional_pose_vectors_are_retained_when_supplied(tmp_path: Path) -> None:
    vectors = np.arange(12, dtype=np.float64).reshape(2, 6)

    retention = retain_per_pair_distortion(
        tmp_path / "retained",
        per_pair_pose=[1.0, 1.0],
        per_pair_seg=[1.0, 1.0],
        reported_pose=1.0,
        reported_seg=1.0,
        pose_vectors=vectors,
    )

    stored = np.load(Path(retention.payload_paths["per_pair_pose_vectors"]))
    assert stored.shape == (2, 6)
    assert np.array_equal(stored, vectors)
