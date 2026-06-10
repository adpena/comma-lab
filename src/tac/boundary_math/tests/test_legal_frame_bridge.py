# SPDX-License-Identifier: MIT
"""Behavior tests for ``tac.boundary_math.legal_frame_bridge`` (tasks #55-react + #56).

These tests verify BEHAVIOR not constants (NO-FAKE class 2):
  - the palette is FIT to the real GT region means (a stub returning gray FAILS the
    region-color assertion);
  - the rasterizer ACTUALLY paints by label map (a stub returning a constant frame
    FAILS the per-region color assertion);
  - the correction field ACTUALLY raises pixels (a no-op correction FAILS);
  - the palette byte cost is the true n_classes*3 (the appearance section is tiny);
  - the label-map coded floor is a real brotli measurement (varies with content);
  - unobserved classes fall back to mid-gray (no crash on missing class).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.legal_frame_bridge import (
    ClassPalette,
    LegalFrameBridgeError,
    _nearest_upsample,
    fit_palette_gt_region_mean,
    lowres_appearance_carrier,
    measure_label_map_coded_bytes,
    rasterize_palette_frame,
    upsample_appearance,
)


def _make_class_correlated_gt(P, H, W, sh, sw):
    """GT frames whose colors are class-correlated so the palette fit is meaningful."""
    am = np.zeros((P, sh, sw), np.uint8)
    am[:, : sh // 2, :] = 0
    am[:, sh // 2 :, :] = 2
    gt = np.zeros((P, H, W, 3), np.uint8)
    for i in range(P):
        amc = _nearest_upsample(am[i], H, W)
        gt[i][amc == 0] = [210, 20, 30]
        gt[i][amc == 2] = [15, 25, 200]
    return gt, am


def test_palette_fits_region_means():
    gt, am = _make_class_correlated_gt(2, 24, 24, 8, 8)
    pal = fit_palette_gt_region_mean(gt, am, n_classes=5)
    assert np.allclose(pal.colors[0], [210, 20, 30], atol=1.0)
    assert np.allclose(pal.colors[2], [15, 25, 200], atol=1.0)
    assert pal.source == "gt_region_mean"


def test_palette_unobserved_class_is_gray():
    gt, am = _make_class_correlated_gt(1, 16, 16, 8, 8)  # only classes 0,2 present
    pal = fit_palette_gt_region_mean(gt, am, n_classes=5)
    assert np.allclose(pal.colors[1], 128.0)  # class 1 never observed
    assert np.allclose(pal.colors[3], 128.0)


def test_rasterizer_paints_by_label_map():
    gt, am = _make_class_correlated_gt(1, 24, 24, 8, 8)
    pal = fit_palette_gt_region_mean(gt, am, n_classes=5)
    frame = rasterize_palette_frame(am[0], pal, camera_h=24, camera_w=24)
    assert frame.shape == (24, 24, 3) and frame.dtype == np.uint8
    # top half (class 0) painted class-0 color; bottom (class 2) class-2 color.
    assert np.allclose(frame[0, 0], [210, 20, 30], atol=1)
    assert np.allclose(frame[-1, -1], [15, 25, 200], atol=1)


def test_rasterizer_not_constant():
    """A constant-frame stub would FAIL: the two regions MUST differ."""
    gt, am = _make_class_correlated_gt(1, 24, 24, 8, 8)
    pal = fit_palette_gt_region_mean(gt, am, n_classes=5)
    frame = rasterize_palette_frame(am[0], pal, camera_h=24, camera_w=24)
    assert not np.allclose(frame[0, 0], frame[-1, -1]), "rasterizer returned constant (fake)"


def test_correction_field_raises_pixels():
    gt, am = _make_class_correlated_gt(1, 24, 24, 8, 8)
    pal = fit_palette_gt_region_mean(gt, am, n_classes=5)
    base = rasterize_palette_frame(am[0], pal, camera_h=24, camera_w=24)
    corr = np.full((8, 8), 10.0)  # +10 luma everywhere
    corrected = rasterize_palette_frame(am[0], pal, camera_h=24, camera_w=24, correction_hw_seg=corr)
    # corrected pixels should be >= base (clamped); on average clearly higher.
    assert corrected.astype(int).mean() > base.astype(int).mean()


def test_palette_byte_cost():
    pal = ClassPalette(colors=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
                                       dtype=np.float64), source="x")
    assert len(pal.to_bytes()) == 15  # 5 classes * 3 channels


def test_palette_bad_shape_raises():
    with pytest.raises(LegalFrameBridgeError):
        ClassPalette(colors=np.zeros((5, 4)), source="x")


def test_label_map_coded_bytes_varies_with_content():
    """A real brotli measurement: a uniform map codes smaller than a noisy one."""
    uniform = np.zeros((4, 8, 8), np.uint8)
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 5, (4, 8, 8)).astype(np.uint8)
    assert measure_label_map_coded_bytes(uniform) < measure_label_map_coded_bytes(noisy)


def test_rasterizer_bad_argmax_shape_raises():
    pal = ClassPalette(colors=np.full((5, 3), 128.0), source="x")
    with pytest.raises(LegalFrameBridgeError):
        rasterize_palette_frame(np.zeros((2, 2, 2)), pal)


# ── lowres appearance carrier (the pose-carrying section diagnostic) ───────
def test_lowres_carrier_downsamples_and_codes():
    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, (64, 96, 3)).astype(np.uint8)
    lowres, coded = lowres_appearance_carrier(frame, factor=4)
    assert lowres.shape == (16, 24, 3) and lowres.dtype == np.uint8
    assert coded > 0  # a real brotli measurement
    # a coarser factor MUST produce fewer coded bytes (real compression behavior).
    _, coded16 = lowres_appearance_carrier(frame, factor=16)
    assert coded16 < coded


def test_lowres_carrier_preserves_structure_not_constant():
    """A structured frame's downsample MUST retain spatial variation (not a constant stub)."""
    frame = np.zeros((64, 64, 3), np.uint8)
    frame[:32] = [200, 50, 50]  # top half red-ish
    frame[32:] = [50, 50, 200]  # bottom half blue-ish
    lowres, _ = lowres_appearance_carrier(frame, factor=8)
    assert not np.allclose(lowres[0, 0], lowres[-1, -1]), "downsample collapsed structure (fake)"


def test_upsample_appearance_roundtrip_shape():
    rng = np.random.default_rng(6)
    lowres = rng.integers(0, 256, (16, 24, 3)).astype(np.uint8)
    up = upsample_appearance(lowres, 64, 96)
    assert up.shape == (64, 96, 3) and up.dtype == np.uint8


def test_lowres_then_upsample_approximates_original():
    """A smooth frame survives downsample->upsample within a tolerance (the pose-carry premise)."""
    yy, xx = np.mgrid[0:80, 0:120]
    plane = (128 + 60 * np.sin(yy / 10.0) + 40 * np.cos(xx / 15.0)).clip(0, 255).astype(np.uint8)
    frame = np.stack([plane, plane, plane], axis=-1)
    lowres, _ = lowres_appearance_carrier(frame, factor=4)
    recon = upsample_appearance(lowres, 80, 120)
    # a smooth field reconstructs closely (the luma motion PoseNet reads is preserved).
    assert float(np.abs(recon.astype(int) - frame.astype(int)).mean()) < 8.0


def test_lowres_carrier_bad_shape_raises():
    with pytest.raises(LegalFrameBridgeError):
        lowres_appearance_carrier(np.zeros((4, 4)), factor=2)
