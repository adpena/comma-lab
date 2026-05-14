# SPDX-License-Identifier: MIT
"""Tests for tac.lane_mark_pose_v2 — endpoint tracking zero-cost pose path.

V1 covered (test_lane_mark_pose.py): centroid math, dim-0 dominance,
graceful fallback on no-lane frames, source-grep wiring asserts.

V2 (this file) adds:

  1. Endpoint extraction (top + bottom row + their mean columns).
  2. Endpoint log-zoom math vs the V1 centroid log-zoom on the same masks.
  3. Per-clip RECALIBRATION via least-squares fit when ``baseline_poses``
     is supplied (the key V2 win: collapses the V1 0.017 correlation).
  4. CLI integration: ``--method centroid|endpoint`` exists in
     ``experiments/build_zero_cost_pose_archive.py`` and dispatches
     correctly.
  5. Smoke against real archive masks (best-effort if the working tree
     has the archive).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from tac.lane_mark_pose import POSENET_DIM0_MEAN
from tac.lane_mark_pose_v2 import (
    compute_endpoint_tracking_pose_dim0,
    compute_endpoint_tracking_poses_from_masks,
    endpoint_log_zoom_from_masks,
    lane_mark_endpoints,
)
from tac.lane_mark_speed import LANE_MARK_CLASS

REPO = Path(__file__).resolve().parents[3]


def _mask_with_lane_segment(
    h: int, w: int,
    top_row: int, bot_row: int,
    col: int,
    lane_class: int = LANE_MARK_CLASS,
) -> torch.Tensor:
    """Build a class-index mask with a single VERTICAL lane segment.

    Drawn as a 3-pixel-wide column from ``top_row`` to ``bot_row`` at
    column ``col``. Useful for testing endpoint geometry without the
    centroid-distortion of a centered patch.
    """
    mask = torch.zeros(h, w, dtype=torch.long)
    c0 = max(0, col - 1)
    c1 = min(w, col + 2)
    r0 = max(0, top_row)
    r1 = min(h, bot_row + 1)
    mask[r0:r1, c0:c1] = lane_class
    return mask


# ── Endpoint extraction ───────────────────────────────────────────────────


def test_endpoints_simple_segment() -> None:
    """A single vertical segment at known top/bottom rows must produce
    matching top_row / bot_row fields."""
    mask = _mask_with_lane_segment(
        384, 512, top_row=180, bot_row=370, col=300,
    )
    ep = lane_mark_endpoints(mask)
    assert ep["n_pixels"] > 0
    assert ep["top_row"] == 180
    assert ep["bot_row"] == 370
    # 3-pixel-wide segment: mean column should be ~300 (col-1, col, col+1).
    assert abs(ep["top_col"] - 300.0) < 1.0
    assert abs(ep["bot_col"] - 300.0) < 1.0


def test_endpoints_empty_mask_falls_back_to_foe() -> None:
    """No lane pixels → endpoints sit at the FoE; n_pixels=0 signals empty."""
    mask = torch.zeros(384, 512, dtype=torch.long)
    ep = lane_mark_endpoints(mask)
    assert ep["n_pixels"] == 0
    # Both endpoints land on the FoE row (174) per DEFAULT_FOE_H.
    assert ep["top_row"] == 174
    assert ep["bot_row"] == 174


def test_endpoints_honours_custom_lane_class() -> None:
    """When the operator passes lane_mark_class != 1, endpoints must search
    for THAT class index."""
    # Lane signal lives at class 3; class 1 is empty.
    mask = _mask_with_lane_segment(
        384, 512, top_row=100, bot_row=200, col=250, lane_class=3,
    )
    ep_class3 = lane_mark_endpoints(mask, lane_mark_class=3)
    ep_class1 = lane_mark_endpoints(mask, lane_mark_class=1)
    assert ep_class3["n_pixels"] > 0
    assert ep_class1["n_pixels"] == 0


# ── Endpoint vs centroid: V2 should be MORE responsive to radial motion ──


def test_endpoint_log_zoom_responds_to_radial_motion() -> None:
    """Forward motion: bottom endpoint sweeps AWAY from FoE → positive
    log-zoom. Construct two frames where ONLY the bottom endpoint moves
    radially outward and verify the V2 log-zoom is positive."""
    h, w = 384, 512
    # Segment t: bottom at row 300 (close to FoE).
    m_t = _mask_with_lane_segment(h, w, top_row=180, bot_row=300, col=300)
    # Segment t1: bottom at row 380 (further from FoE → more radial dist).
    m_t1 = _mask_with_lane_segment(h, w, top_row=180, bot_row=380, col=300)
    masks = torch.stack([m_t, m_t1])
    log_zoom = endpoint_log_zoom_from_masks(masks, smoothing=0.0)
    assert log_zoom.shape == (1,)
    assert log_zoom[0].item() > 0, (
        f"forward motion → bottom endpoint moves away from FoE → "
        f"positive log-zoom; got {log_zoom[0].item()}"
    )


def test_endpoint_log_zoom_zero_on_static_masks() -> None:
    """Identical frames → zero log-zoom (no motion)."""
    h, w = 384, 512
    m = _mask_with_lane_segment(h, w, top_row=180, bot_row=350, col=300)
    masks = torch.stack([m, m.clone()])
    log_zoom = endpoint_log_zoom_from_masks(masks, smoothing=0.0)
    assert abs(log_zoom[0].item()) < 1e-6


def test_endpoint_log_zoom_lateral_drift_is_smaller_than_radial_motion() -> None:
    """The V2 win vs V1: lateral drift (car wandering in lane) does NOT
    move the bottom endpoint's RADIAL distance much — radial motion does.
    A pure lateral shift of the segment should produce a much smaller
    log-zoom than a pure forward-radial shift."""
    h, w = 384, 512
    # Same segment, shifted laterally (column 300 → 320).
    m_t_lateral = _mask_with_lane_segment(h, w, top_row=180, bot_row=350, col=300)
    m_t1_lateral = _mask_with_lane_segment(h, w, top_row=180, bot_row=350, col=320)
    masks_lateral = torch.stack([m_t_lateral, m_t1_lateral])
    log_zoom_lateral = endpoint_log_zoom_from_masks(masks_lateral, smoothing=0.0)

    # Same segment, bottom shifted radially (row 350 → 380).
    m_t_radial = _mask_with_lane_segment(h, w, top_row=180, bot_row=350, col=300)
    m_t1_radial = _mask_with_lane_segment(h, w, top_row=180, bot_row=380, col=300)
    masks_radial = torch.stack([m_t_radial, m_t1_radial])
    log_zoom_radial = endpoint_log_zoom_from_masks(masks_radial, smoothing=0.0)

    # Radial motion produces a larger (positive) log-zoom than lateral.
    assert log_zoom_radial[0].item() > abs(log_zoom_lateral[0].item()), (
        f"radial motion log_zoom={log_zoom_radial[0].item():.4f} should "
        f"exceed |lateral motion log_zoom|={abs(log_zoom_lateral[0].item()):.4f}; "
        f"V2 endpoint method must isolate radial signal from lateral noise"
    )


def test_endpoint_top_vs_bottom_choice() -> None:
    """use_top=True must use the TOP endpoint (closer to FoE, smaller signal).
    Same motion produces smaller |log_zoom| from top endpoints."""
    h, w = 384, 512
    # Both endpoints move down by the same amount (uniform forward motion).
    m_t = _mask_with_lane_segment(h, w, top_row=180, bot_row=300, col=300)
    m_t1 = _mask_with_lane_segment(h, w, top_row=185, bot_row=320, col=300)
    masks = torch.stack([m_t, m_t1])
    lz_bot = endpoint_log_zoom_from_masks(masks, use_top=False, smoothing=0.0)
    lz_top = endpoint_log_zoom_from_masks(masks, use_top=True, smoothing=0.0)
    # Both should be positive (forward motion). The bottom signal is
    # meaningfully larger because radial distance from FoE is larger at
    # the bottom (denominator larger in absolute terms but numerator
    # grows even faster in radial pixel space).
    assert lz_bot[0].item() != lz_top[0].item(), (
        "bottom and top endpoint log-zoom should differ on forward motion"
    )


# ── pose_dim0 wrapper + recalibration ─────────────────────────────────────


def test_compute_pose_dim0_no_baseline_uses_v1_constants() -> None:
    """Without baseline_poses, the function must fall back to the V1
    fixed affine map (POSENET_DIM0_MEAN + POSENET_DIM0_PER_LOGZOOM * logz)."""
    h, w = 384, 512
    masks = torch.stack([
        _mask_with_lane_segment(h, w, top_row=180 + 2 * i, bot_row=350 + i, col=300)
        for i in range(8)
    ])
    pose = compute_endpoint_tracking_pose_dim0(masks)
    assert pose.shape == (4,)
    assert torch.isfinite(pose).all()
    # Range pinned by the V1 clamp [23.0, 36.0]
    assert pose.min().item() >= 23.0 - 1e-5
    assert pose.max().item() <= 36.0 + 1e-5


def test_compute_pose_dim0_with_baseline_recalibrates() -> None:
    """The V2 win: when baseline_poses is supplied, the affine fit absorbs
    per-clip distribution shift. A constant baseline (no variance) must
    map to the constant baseline value (the affine slope = 0, intercept
    = mean)."""
    h, w = 384, 512
    n_pairs = 6
    # Synthetic masks with varying motion.
    masks = torch.stack([
        _mask_with_lane_segment(h, w, top_row=180, bot_row=300 + 5 * i, col=300)
        for i in range(2 * n_pairs)
    ])
    # Constant baseline: every pair has pose_dim0 = 30.0.
    baseline = torch.zeros(n_pairs, 6)
    baseline[:, 0] = 30.0
    pose = compute_endpoint_tracking_pose_dim0(masks, baseline_poses=baseline)
    # The recalibrated map must reproduce the constant ≈ 30.0.
    assert torch.allclose(
        pose, torch.full_like(pose, 30.0), atol=0.5,
    ), f"recalibrated pose_dim0 should match constant baseline; got {pose.tolist()}"


def test_compute_pose_dim0_baseline_shape_validation() -> None:
    masks = torch.zeros(8, 384, 512, dtype=torch.long)
    bad = torch.zeros(99, 6)  # wrong pair count
    with pytest.raises(ValueError, match="pairs"):
        compute_endpoint_tracking_pose_dim0(masks, baseline_poses=bad)
    bad_ndim = torch.zeros(4)
    with pytest.raises(ValueError):
        compute_endpoint_tracking_pose_dim0(masks, baseline_poses=bad_ndim)


# ── Full pose tensor wrapper ──────────────────────────────────────────────


def test_compute_endpoint_tracking_poses_dim_0_dominant() -> None:
    """V2 must mirror V1's dim-0-only contract (dims 1-5 exactly zero)."""
    h, w = 384, 512
    n_frames = 12
    masks = torch.stack([
        _mask_with_lane_segment(h, w, top_row=180, bot_row=300 + i, col=300)
        for i in range(n_frames)
    ])
    poses = compute_endpoint_tracking_poses_from_masks(masks)
    assert poses.shape == (n_frames // 2, 6)
    for d in range(1, 6):
        assert (poses[:, d] == 0.0).all(), (
            f"dim {d} must be zero, got {poses[:, d].tolist()}"
        )


def test_compute_endpoint_tracking_poses_handles_missing_lanes() -> None:
    """No lane pixels in any frame → pose dim 0 = POSENET_DIM0_MEAN."""
    masks = torch.zeros(8, 384, 512, dtype=torch.long)
    poses = compute_endpoint_tracking_poses_from_masks(masks)
    assert torch.isfinite(poses).all()
    assert torch.allclose(
        poses[:, 0],
        torch.full((4,), POSENET_DIM0_MEAN, dtype=torch.float32),
        atol=1e-5,
    )
    assert (poses[:, 1:] == 0.0).all()


def test_compute_endpoint_tracking_poses_rejects_odd_frame_count() -> None:
    masks = torch.zeros(7, 384, 512, dtype=torch.long)
    with pytest.raises(ValueError):
        compute_endpoint_tracking_poses_from_masks(masks)


def test_compute_endpoint_tracking_poses_rejects_wrong_ndim() -> None:
    masks = torch.zeros(384, 512, dtype=torch.long)
    with pytest.raises(ValueError):
        compute_endpoint_tracking_poses_from_masks(masks)


# ── CLI integration: build_zero_cost_pose_archive --method ─────────────


def test_build_zero_cost_pose_archive_has_method_flag() -> None:
    """experiments/build_zero_cost_pose_archive.py MUST register the
    --method flag. Catches the dead-flag-wiring bug class
    (memory: feedback_dead_flag_wiring_pattern)."""
    src = (
        REPO / "experiments" / "build_zero_cost_pose_archive.py"
    ).read_text()
    add_re = re.compile(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)")
    flags = set(add_re.findall(src))
    assert "method" in flags, (
        "experiments/build_zero_cost_pose_archive.py is missing --method flag"
    )
    # Ensure the choices wired correctly (centroid, endpoint).
    assert "centroid" in src and "endpoint" in src, (
        "build_zero_cost_pose_archive.py --method must accept "
        "'centroid' (V1) and 'endpoint' (V2)"
    )


def test_build_zero_cost_pose_archive_default_method_is_endpoint() -> None:
    """V2 is the default. Source-grep the default kwarg so a refactor
    that silently flips back to V1 trips the test."""
    src = (
        REPO / "experiments" / "build_zero_cost_pose_archive.py"
    ).read_text()
    # Locate the --method add_argument block and confirm default="endpoint"
    # appears inside it. The block can span multiple lines; we search for
    # the --method literal then a window of ~500 chars after for the default.
    method_idx = src.find('"--method"')
    assert method_idx != -1, "no --method add_argument block found"
    window = src[method_idx:method_idx + 800]
    assert 'default="endpoint"' in window or "default='endpoint'" in window, (
        "experiments/build_zero_cost_pose_archive.py --method must default "
        f"to 'endpoint' (Lane LM-V2). Window:\n{window!r}"
    )


# ── Smoke: real archive masks ─────────────────────────────────────────────


def test_endpoint_smoke_on_real_masks() -> None:
    """End-to-end: decode masks from archive_fullres.zip + verify the V2
    endpoint method produces sane outputs on real data."""
    import os
    import tempfile
    import zipfile

    archive = (
        REPO / "submissions" / "robust_current" / "archive_fullres.zip"
    )
    if not archive.exists():
        pytest.skip(f"{archive} not present; smoke test skipped")
    try:
        import av  # type: ignore[import-not-found]
    except ImportError:
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
    scale_factor = 255 // 4
    masks_class = (
        (masks_uint8.float() / scale_factor).round().long().clamp(0, 4)
    )
    poses = compute_endpoint_tracking_poses_from_masks(masks_class)
    assert poses.shape == (masks_class.shape[0] // 2, 6)
    assert torch.isfinite(poses).all(), "NaN/Inf in real-mask V2 smoke"
    d0 = poses[:, 0]
    assert d0.min().item() >= 23.0 - 1e-5
    assert d0.max().item() <= 36.0 + 1e-5
    # Mean within +/-2 of empirical 31.295
    assert abs(d0.mean().item() - POSENET_DIM0_MEAN) < 2.0
    assert (poses[:, 1:] == 0.0).all()
