# SPDX-License-Identifier: MIT
"""Tests for lane-mark-speed module — derives per-pair zoom scalars from masks.

The Hotz analytical move: lane-mark centroids encode vehicle speed via radial
displacement from the FoE. At inflate time, given only the masks (already in
the archive), construct the zoom warp WITHOUT needing zoom_scalars.bin.

These tests pin the property invariants. Real-world correlation against
optimize_zoom_scalars() is measured separately at integration time.
"""
from __future__ import annotations

import torch

from tac.lane_mark_speed import (
    DEFAULT_FOE_H,
    DEFAULT_FOE_W,
    LANE_MARK_CLASS,
    _lane_centroid,
    _radial_distance,
    build_zoom_warp_from_masks,
    estimate_correlation_with_optimized,
    lane_displacement_per_pair,
    zoom_from_masks,
)
from tac.radial_zoom import RadialZoomWarp


def _mask_with_lane_at(h: int, w: int, lane_col: int, lane_row: int) -> torch.Tensor:
    """Build a class-index mask with a 5x5 lane patch centered at (row, col)."""
    mask = torch.zeros(h, w, dtype=torch.long)  # class 0 (road) everywhere
    r0 = max(0, lane_row - 2)
    r1 = min(h, lane_row + 3)
    c0 = max(0, lane_col - 2)
    c1 = min(w, lane_col + 3)
    mask[r0:r1, c0:c1] = LANE_MARK_CLASS
    return mask


# ── Centroid + radial distance ────────────────────────────────────────────


def test_lane_centroid_finds_5x5_patch_center() -> None:
    mask = _mask_with_lane_at(384, 512, lane_col=300, lane_row=250)
    cw, ch, n = _lane_centroid(mask)
    assert n == 25, f"expected 25 lane pixels, got {n}"
    assert abs(cw - 300.0) < 0.5
    assert abs(ch - 250.0) < 0.5


def test_lane_centroid_no_lane_returns_foe() -> None:
    """When a frame has zero lane pixels, centroid defaults to FoE.

    This avoids div-by-zero downstream and produces zero-zoom for that pair —
    the only sensible fallback when there's no lane signal.
    """
    mask = torch.zeros(384, 512, dtype=torch.long)
    cw, ch, n = _lane_centroid(mask)
    assert n == 0
    assert cw == DEFAULT_FOE_W
    assert ch == DEFAULT_FOE_H


def test_radial_distance_from_foe() -> None:
    # FoE itself: distance 0
    assert _radial_distance(DEFAULT_FOE_W, DEFAULT_FOE_H) == 0.0
    # Right of FoE by 100: distance 100
    assert abs(_radial_distance(DEFAULT_FOE_W + 100, DEFAULT_FOE_H) - 100.0) < 1e-6
    # 3-4-5 right triangle from FoE
    d = _radial_distance(DEFAULT_FOE_W + 3, DEFAULT_FOE_H + 4)
    assert abs(d - 5.0) < 1e-6


# ── Pair-wise displacement ────────────────────────────────────────────────


def test_pair_displacement_forward_motion_is_positive() -> None:
    """Forward driving: lane mark moves AWAY from FoE → positive displacement."""
    # FoE is (256, 174). Place lane to the right + below FoE; in t1 moves further away.
    m_t = _mask_with_lane_at(384, 512, lane_col=300, lane_row=250)
    m_t1 = _mask_with_lane_at(384, 512, lane_col=320, lane_row=270)
    disp = lane_displacement_per_pair(m_t, m_t1)
    assert disp > 0, f"forward motion should give positive displacement, got {disp}"


def test_pair_displacement_no_lane_returns_zero() -> None:
    """Degenerate (no lane in either frame): displacement = 0 = no zoom inferred."""
    m = torch.zeros(384, 512, dtype=torch.long)
    assert lane_displacement_per_pair(m, m) == 0.0


# ── zoom_from_masks ───────────────────────────────────────────────────────


def test_zoom_from_masks_shape_matches_pairs() -> None:
    n_frames = 12
    masks = torch.stack([
        _mask_with_lane_at(384, 512, lane_col=300 + 2 * i, lane_row=250 + i)
        for i in range(n_frames)
    ])
    z = zoom_from_masks(masks)
    assert z.shape == (n_frames // 2,)
    assert z.dtype == torch.float32


def test_zoom_from_masks_consistent_forward_motion_gives_positive_zoom() -> None:
    """Forward driving (lane drifting away from FoE) gives positive zoom values
    consistently. The exact magnitude varies because radial mean grows with
    distance, but the SIGN is the invariant we care about."""
    n_frames = 20
    masks = []
    for i in range(n_frames):
        # Lane drifts +2 col, +1 row per frame (away from FoE 256,174)
        masks.append(_mask_with_lane_at(384, 512, lane_col=300 + 2 * i, lane_row=250 + i))
    masks = torch.stack(masks)
    z = zoom_from_masks(masks, smoothing=0.0)
    # All pairs should show positive (forward) zoom — sign is the geometric invariant
    assert (z > 0).all(), f"all forward-motion zooms should be positive, got {z}"


def test_zoom_from_masks_rejects_odd_frame_count() -> None:
    masks = torch.zeros(7, 384, 512, dtype=torch.long)
    try:
        zoom_from_masks(masks)
    except ValueError as e:
        assert "even frame count" in str(e)
        return
    raise AssertionError("expected ValueError on odd frame count")


def test_zoom_from_masks_rejects_wrong_ndim() -> None:
    masks = torch.zeros(384, 512, dtype=torch.long)  # missing N dim
    try:
        zoom_from_masks(masks)
    except ValueError as e:
        assert "(N, H, W)" in str(e)
        return
    raise AssertionError("expected ValueError on wrong ndim")


# ── build_zoom_warp_from_masks ────────────────────────────────────────────


def test_build_zoom_warp_from_masks_returns_radial_zoom_warp() -> None:
    n_frames = 10
    masks = torch.stack([
        _mask_with_lane_at(384, 512, 300 + i, 250 + i)
        for i in range(n_frames)
    ])
    zw = build_zoom_warp_from_masks(masks)
    assert isinstance(zw, RadialZoomWarp)
    assert zw.zoom_scalars.shape == (n_frames // 2,)
    # Within RadialZoomWarp's |s| <= max_zoom_log clamp
    assert zw.zoom_scalars.abs().max() <= zw.max_zoom_log + 1e-6


def test_build_zoom_warp_drop_in_for_optimized() -> None:
    """The constructed RadialZoomWarp is interface-identical to one populated
    by optimize_zoom_scalars — same module type, same call signature, same
    output shape from the forward pass.
    """
    n_frames = 8
    masks = torch.stack([
        _mask_with_lane_at(384, 512, 300 + i, 250 + i)
        for i in range(n_frames)
    ])
    zw = build_zoom_warp_from_masks(masks)
    # forward returns (B, 2, H, W) — flow channels first, matching grid_sample input
    pair_indices = torch.arange(2)
    ego_flow = zw(pair_indices, 384, 512)
    assert ego_flow.shape == (2, 2, 384, 512)


# ── correlation against optimized ─────────────────────────────────────────


def test_correlation_with_self_is_one() -> None:
    """If we feed our own analytical estimate back as the 'optimized' baseline
    AND use the same smoothing, correlation must be 1.0 — sanity check on the
    metric itself."""
    n_frames = 20
    masks = torch.stack([
        _mask_with_lane_at(384, 512, 300 + 2 * i, 250 + i)
        for i in range(n_frames)
    ])
    # Use smoothing=0.0 here too so estimated == feedback baseline
    estimated = zoom_from_masks(masks, smoothing=0.0)
    result = estimate_correlation_with_optimized(masks, estimated, smoothing=0.0)
    assert abs(result["correlation"] - 1.0) < 1e-5
    assert result["rmse"] < 1e-5


def test_correlation_metric_shape_validation() -> None:
    """Mismatched shapes raise loudly instead of returning silent garbage."""
    masks = torch.stack([
        _mask_with_lane_at(384, 512, 300, 250) for _ in range(8)
    ])  # 4 pairs
    bad_optimized = torch.zeros(7)  # wrong: should be 4
    try:
        estimate_correlation_with_optimized(masks, bad_optimized)
    except ValueError as e:
        assert "shape mismatch" in str(e)
        return
    raise AssertionError("expected ValueError on mismatched shapes")
