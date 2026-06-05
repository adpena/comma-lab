# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.analysis.nerv_distortion_crux import (
    NERV_DISTORTION_CRUX_SCHEMA,
    build_nerv_distortion_crux_report,
)


def test_distortion_crux_ranks_real_segnet_frame_and_posenet_pair_errors(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()

    ref_seg = np.zeros((3, 3, 2, 2), dtype=np.float32)
    cand_seg = ref_seg.copy()
    cand_seg[0] = 10.0

    ref_pose = np.zeros((3, 12, 2, 2), dtype=np.float32)
    cand_pose = ref_pose.copy()
    cand_pose[1, 6:12, :, :] = 20.0
    cand_pose[2, :, :, :] = 30.0
    pair_indices = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)

    for root, seg, pose in (
        (reference, ref_seg, ref_pose),
        (candidate, cand_seg, cand_pose),
    ):
        np.save(root / "segnet_last_rgb.npy", seg)
        np.save(root / "posenet_yuv6_pair.npy", pose)
        np.save(root / "pair_indices.npy", pair_indices)

    report = build_nerv_distortion_crux_report(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        sample_pairs=3,
        top_k=3,
        min_routable_pairs=3,
        max_segnet_last_frame_mae_for_fit_gate=5.0,
        max_posenet_yuv6_pair_mae_for_fit_gate=10.0,
        max_posenet_temporal_delta_mae_for_fit_gate=8.0,
    )

    assert report["schema"] == NERV_DISTORTION_CRUX_SCHEMA
    assert report["hard_pair_rows"][0]["pair_index"] == 2
    assert report["hard_pair_rows"][0]["dominant_domain"] == "posenet_yuv6_pair"
    assert report["hard_pair_rows"][1]["pair_index"] == 1
    assert report["hard_pair_rows"][1]["dominant_domain"] == (
        "posenet_temporal_delta"
    )
    assert report["hard_pair_rows"][2]["pair_index"] == 0
    assert report["hard_pair_rows"][2]["dominant_domain"] == "segnet_last_frame"
    assert report["hard_pair_coverage"]["prioritized_pair_indices"] == [2, 1, 0]
    assert report["hard_pair_coverage"]["score_axis_hard_pair_coverage"] is True
    assert report["fit_gate_passed"] is False
    assert "nerv_distortion_crux_posenet_yuv6_pair_mae_too_high" in report[
        "blockers"
    ]
    assert report["aggregate"]["segnet_last_frame_mae_255"]["mean"] == pytest.approx(
        10.0 / 3.0
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_distortion_crux_hitlist_is_not_routable_before_min_pair_count(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    for root in (candidate, reference):
        np.save(root / "segnet_last_rgb.npy", np.zeros((1, 3, 2, 2), dtype=np.float32))
        np.save(root / "posenet_yuv6_pair.npy", np.zeros((1, 12, 1, 1), dtype=np.float32))
        np.save(root / "pair_indices.npy", pair_indices)

    report = build_nerv_distortion_crux_report(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        sample_pairs=1,
        min_routable_pairs=16,
    )

    coverage = report["hard_pair_coverage"]
    assert coverage["prioritized_pair_indices"] == [0]
    assert coverage["score_axis_hard_pair_coverage"] is False
    assert "nerv_distortion_crux_min16_pairs_missing_for_launch_routing" in coverage[
        "blockers"
    ]
    assert report["fit_gate_passed"] is True
