"""Tests for tac.lane_mark_pose — zero-archive-cost 6-DOF pose conditioning.

These tests pin the property invariants of the inflate-time analytical pose
path. The shape of the output, the dim-0 dominance, the zero-rest convention,
the graceful no-lane-mark fallback, and the build/inflate wiring grep-asserts
are all covered. Smoke-tested against real archive masks at the end.
"""
from __future__ import annotations

from pathlib import Path

import torch

from tac.lane_mark_pose import (
    POSENET_DIM0_MEAN,
    POSENET_DIM0_PER_LOGZOOM,
    ZERO_COST_POSES_SENTINEL,
    compute_zero_cost_poses_from_masks,
)
from tac.lane_mark_speed import LANE_MARK_CLASS

REPO = Path(__file__).resolve().parents[3]


def _mask_with_lane_at(
    h: int, w: int, lane_col: int, lane_row: int,
    lane_class: int = LANE_MARK_CLASS,
) -> torch.Tensor:
    """Build a class-index mask with a 5x5 lane patch centered at (row, col)."""
    mask = torch.zeros(h, w, dtype=torch.long)
    r0 = max(0, lane_row - 2)
    r1 = min(h, lane_row + 3)
    c0 = max(0, lane_col - 2)
    c1 = min(w, lane_col + 3)
    mask[r0:r1, c0:c1] = lane_class
    return mask


# ── Shape + dtype invariants ───────────────────────────────────────────────


def test_compute_zero_cost_poses_returns_correct_shape() -> None:
    """The output must be (num_pairs, 6) with num_pairs = N // 2."""
    n_frames = 12
    masks = torch.stack([
        _mask_with_lane_at(384, 512, 300 + 2 * i, 250 + i)
        for i in range(n_frames)
    ])
    poses = compute_zero_cost_poses_from_masks(masks)
    assert poses.shape == (n_frames // 2, 6), (
        f"expected ({n_frames // 2}, 6), got {tuple(poses.shape)}"
    )
    assert poses.dtype == torch.float32


def test_compute_zero_cost_poses_zero_dim_dominant() -> None:
    """Dim 0 must carry the lane-mark signal; dims 1-5 must be exactly zero.

    Per memory project_posenet_rank1_discovery: 99.8% of variance lives in
    dim 0. Per memory project_yousfi_geometric_analysis: predicting dims 1-5
    as zero costs at most 0.18 distortion points.
    """
    n_frames = 20
    # Forward driving: lane drifts away from FoE → positive log-zoom →
    # pose_dim0 ABOVE the per-clip mean.
    masks = torch.stack([
        _mask_with_lane_at(384, 512, 300 + 2 * i, 250 + i)
        for i in range(n_frames)
    ])
    poses = compute_zero_cost_poses_from_masks(masks)

    # Dims 1-5 are exactly zero (Fridrich strategy)
    for d in range(1, 6):
        assert (poses[:, d] == 0.0).all(), (
            f"dim {d} must be zero, got std={poses[:, d].std().item():.6f}"
        )

    # Dim 0 must be in the empirical PoseNet range (around 31.295 +- ~10)
    assert poses[:, 0].mean().item() > 25.0
    assert poses[:, 0].mean().item() < 36.0


def test_compute_zero_cost_poses_handles_missing_lane_marks() -> None:
    """Frames with no lane-marking pixels must not produce NaNs or out-of-
    range pose values. The graceful fallback is pose_dim0 = per-clip mean
    (no inferred motion) and dims 1-5 = 0."""
    n_frames = 8
    # All frames are pure road (class 0), zero lane pixels
    masks = torch.zeros(n_frames, 384, 512, dtype=torch.long)
    poses = compute_zero_cost_poses_from_masks(masks)
    assert poses.shape == (n_frames // 2, 6)
    assert torch.isfinite(poses).all(), "NaN/Inf in zero-lane fallback"
    # All zoom = 0 → all pose_dim0 = POSENET_DIM0_MEAN exactly
    assert torch.allclose(
        poses[:, 0],
        torch.full((n_frames // 2,), POSENET_DIM0_MEAN, dtype=torch.float32),
        atol=1e-5,
    )
    # And the rest of the columns are still zero
    assert (poses[:, 1:] == 0.0).all()


def test_compute_zero_cost_poses_clamps_extreme_values() -> None:
    """An anomalous lane-mark spike must NOT push pose_dim0 outside the
    empirical [23.0, 36.0] envelope — that would put FiLM conditioning
    OOD vs training data."""
    # Construct a pair where the lane mark jumps from very near the FoE
    # (small radial dist) to very far (large radial dist) — this maximises
    # the log_zoom estimate.
    m_t = _mask_with_lane_at(384, 512, lane_col=258, lane_row=176)  # ~at FoE
    m_t1 = _mask_with_lane_at(384, 512, lane_col=500, lane_row=380)  # corner
    masks = torch.stack([m_t, m_t1])
    poses = compute_zero_cost_poses_from_masks(masks, smoothing=0.0)
    # Even with extreme jump, dim 0 stays clamped
    assert poses[0, 0].item() >= 23.0 - 1e-5
    assert poses[0, 0].item() <= 36.0 + 1e-5


def test_compute_zero_cost_poses_rejects_odd_frame_count() -> None:
    masks = torch.zeros(7, 384, 512, dtype=torch.long)
    try:
        compute_zero_cost_poses_from_masks(masks)
    except ValueError as e:
        assert "even frame count" in str(e)
        return
    raise AssertionError("expected ValueError on odd frame count")


def test_compute_zero_cost_poses_rejects_wrong_ndim() -> None:
    masks = torch.zeros(384, 512, dtype=torch.long)
    try:
        compute_zero_cost_poses_from_masks(masks)
    except ValueError as e:
        assert "(N, H, W)" in str(e)
        return
    raise AssertionError("expected ValueError on wrong ndim")


def test_compute_zero_cost_poses_honors_custom_lane_class() -> None:
    """When the operator passes lane_mark_class != 1, the function must use
    THAT class index, not the module default. Verifies the remap path."""
    # Build masks where class 3 carries the lane signal (instead of class 1)
    n_frames = 8
    masks = torch.stack([
        _mask_with_lane_at(384, 512, 300 + 2 * i, 250 + i, lane_class=3)
        for i in range(n_frames)
    ])
    # If we pass lane_mark_class=3 we get a real signal:
    poses_class3 = compute_zero_cost_poses_from_masks(masks, lane_mark_class=3)
    # If we pass lane_mark_class=1 (default, no class-1 pixels exist), we get
    # the no-signal fallback: pose_dim0 = POSENET_DIM0_MEAN exactly.
    poses_class1 = compute_zero_cost_poses_from_masks(masks, lane_mark_class=1)
    assert torch.allclose(
        poses_class1[:, 0],
        torch.full((n_frames // 2,), POSENET_DIM0_MEAN, dtype=torch.float32),
        atol=1e-5,
    )
    # And the class-3 result has nonzero std (real signal)
    assert poses_class3[:, 0].std().item() > 0.0


# ── Source-grep regression tests for inflate / build wiring ────────────────


def test_inflate_renderer_uses_zero_cost_when_env_set() -> None:
    """Source-grep: the inflate renderer must contain an env-gated branch
    that calls compute_zero_cost_poses_from_masks() when the sentinel is
    present and INFLATE_ZERO_COST_POSES is enabled. This catches a future
    refactor that silently drops the wiring."""
    inflate_path = REPO / "submissions" / "robust_current" / "inflate_renderer.py"
    assert inflate_path.exists(), f"inflate_renderer.py not found at {inflate_path}"
    src = inflate_path.read_text()
    assert "INFLATE_ZERO_COST_POSES" in src, (
        "inflate_renderer.py is missing INFLATE_ZERO_COST_POSES env gate"
    )
    assert "compute_zero_cost_poses_from_masks" in src, (
        "inflate_renderer.py is missing the zero-cost pose call site"
    )
    assert ZERO_COST_POSES_SENTINEL in src, (
        f"inflate_renderer.py is missing the {ZERO_COST_POSES_SENTINEL} "
        "sentinel detection"
    )


def test_build_baseline_archive_skips_poses_with_flag() -> None:
    """Source-grep: build_baseline_archive must define --use-zero-cost-poses
    and write the sentinel file when the flag is set."""
    build_path = REPO / "experiments" / "build_baseline_archive.py"
    assert build_path.exists(), f"build_baseline_archive.py not found at {build_path}"
    src = build_path.read_text()
    assert "--use-zero-cost-poses" in src, (
        "build_baseline_archive.py is missing --use-zero-cost-poses flag"
    )
    assert "use_zero_cost_poses" in src, (
        "build_baseline_archive.py is missing args.use_zero_cost_poses access"
    )
    assert ZERO_COST_POSES_SENTINEL in src, (
        f"build_baseline_archive.py is missing the {ZERO_COST_POSES_SENTINEL} "
        "sentinel write"
    )


# ── Smoke test against real archive masks ─────────────────────────────────


def test_zero_cost_pose_smoke_on_real_masks() -> None:
    """End-to-end sanity: decode masks from the archive_fullres.zip (the
    only checked-in archive with full 384x512 masks), run the helper, and
    assert the output is sane (no NaNs, dim 0 in empirical PoseNet range,
    other dims zero)."""
    import os
    import tempfile
    import zipfile

    archive = (
        REPO / "submissions" / "robust_current" / "archive_fullres.zip"
    )
    if not archive.exists():
        # Real-mask smoke is best-effort — if the ungraded archive is not
        # in the working tree we skip rather than fail. The synthetic-mask
        # tests above cover the algorithm itself.
        import pytest
        pytest.skip(f"{archive} not present; smoke test skipped")

    try:
        import av  # type: ignore[import-not-found]
    except ImportError:
        import pytest
        pytest.skip("PyAV not installed; smoke test skipped")

    with zipfile.ZipFile(archive) as z:
        with tempfile.TemporaryDirectory() as td:
            z.extract("masks.mkv", td)
            masks_path = os.path.join(td, "masks.mkv")
            container = av.open(masks_path)
            frames = []
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format="gray")
                frames.append(torch.from_numpy(arr))
            container.close()
    masks_uint8 = torch.stack(frames)

    # Decode pixel-grayscale back to class indices (matches inflate_renderer
    # _decode_argmax_codec: scale_factor = 255 // 4 = 63, round, clamp 0..4)
    scale_factor = 255 // 4
    masks_class = (
        (masks_uint8.float() / scale_factor).round().long().clamp(0, 4)
    )

    poses = compute_zero_cost_poses_from_masks(masks_class)
    assert poses.shape == (masks_class.shape[0] // 2, 6)
    assert torch.isfinite(poses).all(), "NaN/Inf in real-mask smoke"

    # Dim 0 must land in the empirical PoseNet envelope
    d0 = poses[:, 0]
    assert d0.min().item() >= 23.0 - 1e-5
    assert d0.max().item() <= 36.0 + 1e-5
    # Mean should be within +/-2 of the empirical 31.295 (center of clamp)
    assert abs(d0.mean().item() - POSENET_DIM0_MEAN) < 2.0, (
        f"dim0 mean {d0.mean().item():.3f} drifted too far from "
        f"empirical {POSENET_DIM0_MEAN}"
    )
    # Dims 1-5 stay zero
    assert (poses[:, 1:] == 0.0).all()


def test_per_logzoom_slope_is_documented_value() -> None:
    """Pin the calibrated coupling constant. If a future commit changes
    POSENET_DIM0_PER_LOGZOOM the test forces an explicit acknowledgement
    rather than a silent recalibration that could shift FiLM conditioning."""
    assert POSENET_DIM0_PER_LOGZOOM == 8.0, (
        "POSENET_DIM0_PER_LOGZOOM was changed without updating the test. "
        "Re-derive the slope (POSENET_DIM0_STD / lane_mark_logzoom_std) "
        "from a fresh measurement and update both the constant and this "
        "regression test."
    )


def test_posenet_dim0_mean_matches_baseline_optimized_poses() -> None:
    """Pin the per-clip mean against the baseline optimized_poses.pt the
    renderer was conditioned on. Drift here means the renderer FiLM is
    being driven outside its training distribution."""
    baseline = (
        REPO / "submissions" / "baseline_dilated_h64_0_90"
        / "optimized_poses.pt"
    )
    if not baseline.exists():
        import pytest
        pytest.skip(f"{baseline} not present; mean-match test skipped")
    poses = torch.load(str(baseline), weights_only=True)
    actual_mean = float(poses[:, 0].mean().item())
    # Constant matches the empirical mean to within 0.01
    assert abs(actual_mean - POSENET_DIM0_MEAN) < 0.01, (
        f"baseline pose dim0 mean is {actual_mean:.4f}; "
        f"POSENET_DIM0_MEAN constant is {POSENET_DIM0_MEAN}. "
        "Recalibrate the constant or re-anchor the baseline."
    )
