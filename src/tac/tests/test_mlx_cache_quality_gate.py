# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np

from tac.analysis.mlx_cache_quality_gate import build_mlx_cache_quality_gate


def test_constant_candidate_cache_fails_as_fundamental_renderer_output_bug(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    _write_cache(candidate, seg_value=127.0, constant=True)
    _write_cache(reference, seg_value=24.0, constant=False)

    report = build_mlx_cache_quality_gate(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        sample_pairs=2,
    )

    assert report["verdict"] == "FUNDAMENTAL_RENDERER_OUTPUT_DEGENERATE"
    assert report["candidate_cache_nondegenerate"] is False
    assert report["fit_gate_passed"] is False
    assert "candidate_segnet_last_rgb_degenerate_constant_or_flat" in report["blockers"]
    assert "candidate_segnet_last_rgb_dynamic_range_too_low" in report["blockers"]
    assert report["stats"]["candidate_segnet_last_rgb"]["std"] == 0.0
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_nondegenerate_close_cache_passes_local_quality_gate(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    _write_cache(reference, seg_value=24.0, constant=False)
    _write_cache(candidate, seg_value=25.0, constant=False)

    report = build_mlx_cache_quality_gate(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        sample_pairs=2,
    )

    assert report["verdict"] == "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY"
    assert report["candidate_cache_nondegenerate"] is True
    assert report["fit_gate_passed"] is True
    assert report["distance_to_reference"]["segnet_last_rgb_mae"] < 64.0
    assert report["score_claim"] is False


def test_dynamic_range_blocker_fails_fit_gate_without_constant_verdict(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    small = np.linspace(20.0, 24.0, num=2 * 3 * 8 * 8, dtype=np.float32).reshape(
        2, 3, 8, 8
    )
    reference_seg = small + 1.0
    _write_cache_arrays(candidate, seg=small)
    _write_cache_arrays(reference, seg=reference_seg)

    report = build_mlx_cache_quality_gate(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        sample_pairs=2,
    )

    assert report["verdict"] == "RENDER_OUTPUT_DYNAMIC_RANGE_TOO_LOW"
    assert report["candidate_cache_nondegenerate"] is True
    assert report["fit_gate_passed"] is False
    assert "candidate_segnet_last_rgb_dynamic_range_too_low" in report["blockers"]
    assert "candidate_segnet_last_rgb_degenerate_constant_or_flat" not in report[
        "blockers"
    ]


def test_posenet_yuv6_collapse_fails_even_when_segnet_cache_is_healthy(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    base = np.arange(2 * 3 * 8 * 8, dtype=np.float32).reshape(2, 3, 8, 8)
    seg = (base % 96.0) + 24.0
    reference_pose = np.concatenate(
        [seg[:, :, ::2, ::2], seg[:, :, ::2, ::2] + 128.0],
        axis=1,
    )
    candidate_pose = np.zeros_like(reference_pose)
    _write_cache_arrays(candidate, seg=seg, pose=candidate_pose)
    _write_cache_arrays(reference, seg=seg + 0.25, pose=reference_pose)

    report = build_mlx_cache_quality_gate(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        sample_pairs=2,
    )

    assert report["verdict"] == "FUNDAMENTAL_POSE_INPUT_DEGENERATE"
    assert report["candidate_cache_nondegenerate"] is True
    assert report["fit_gate_passed"] is False
    assert "candidate_segnet_last_rgb_far_from_reference_fit_gate" not in report[
        "blockers"
    ]
    assert "candidate_posenet_yuv6_pair_degenerate_constant_or_flat" in report[
        "blockers"
    ]
    assert "candidate_posenet_yuv6_pair_dynamic_range_too_low" in report["blockers"]
    assert "candidate_posenet_yuv6_pair_far_from_reference_fit_gate" in report[
        "blockers"
    ]


def _write_cache(root: Path, *, seg_value: float, constant: bool) -> None:
    if constant:
        seg = np.full((2, 3, 8, 8), seg_value, dtype=np.float32)
    else:
        base = np.arange(2 * 3 * 8 * 8, dtype=np.float32).reshape(2, 3, 8, 8)
        seg = (base % 96.0) + seg_value
    _write_cache_arrays(root, seg=seg)


def _write_cache_arrays(
    root: Path,
    *,
    seg: np.ndarray,
    pose: np.ndarray | None = None,
) -> None:
    if pose is None:
        pose = np.concatenate([seg[:, :, ::2, ::2], seg[:, :, ::2, ::2]], axis=1)
    np.save(root / "segnet_last_rgb.npy", seg)
    np.save(root / "posenet_yuv6_pair.npy", pose)
