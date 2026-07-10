# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.through_r.palette_realization` — palette-artifact probe logic.

Uses a deterministic StubSegNet (class = nearest reference colour) so the palette /
decision-geometry / painting / flip-decomposition LOGIC is verified $0 without loading the
real frozen SegNet. One end-to-end arm test proves the perfect-partition round-trip is F==0.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.through_r.palette_realization import (
    ArmResult,
    PaletteRealizationError,
    _dilate_presence,
    arm_result_to_measurement_rows,
    classify_color,
    decompose_flips,
    global_mean_palette,
    map_decision_geometry,
    mixing_robust_palette,
    naive_mean_palette,
    paint_camera_res_uint8,
    paint_render_grid,
    realize_argmax_no_R,
    run_arm,
)
from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W, SEG_H, SEG_W

# 5 well-separated reference colours at the grid corners (levels 0/127.5/255 hit these exactly).
REF = np.array(
    [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]], dtype=np.float32
)


class StubSegNet:
    """Deterministic SegNet stub: per-pixel argmax = nearest REF colour (logit = -dist²)."""

    def __init__(self, ref: np.ndarray = REF):
        import torch

        self.ref = torch.tensor(np.asarray(ref, dtype=np.float32))  # (5,3)

    def preprocess_input(self, x):  # x (b,1,3,H,W) -> (b,3,SEG_H,SEG_W)
        import torch

        x = x[:, -1]
        return torch.nn.functional.interpolate(
            x, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False
        )

    def __call__(self, x):  # (b,3,H,W) -> logits (b,5,H,W)
        xr = x.permute(0, 2, 3, 1)  # (b,H,W,3)
        d = ((xr[..., None, :] - self.ref[None, None, None]) ** 2).sum(-1)  # (b,H,W,5)
        return (-d).permute(0, 3, 1, 2)


@pytest.fixture(scope="module")
def stub():
    return StubSegNet()


@pytest.fixture(scope="module")
def dg(stub):
    return map_decision_geometry(segnet=stub, grid_levels=3)


# --------------------------------------------------------------------------- #
def test_decision_geometry_recovers_reference_classes(dg):
    # every reference colour is a grid corner -> best_color_for_class recovers it.
    for c in range(5):
        best = dg.best_color_for_class(c)
        assert np.allclose(best, REF[c]), (c, best, REF[c])
    # every class wins argmax on at least its own corner.
    assert set(np.unique(dg.argmax_class)) == set(range(5))


def test_classify_color_nearest(dg):
    for c in range(5):
        assert classify_color(dg, REF[c]) == c


def test_naive_mean_palette_per_class_mean():
    rg = np.zeros((SEG_H, SEG_W, 3), np.float32)
    lab = np.zeros((SEG_H, SEG_W), np.int64)
    lab[:, : SEG_W // 2] = 0
    lab[:, SEG_W // 2 :] = 3
    rg[lab == 0] = [10, 20, 30]
    rg[lab == 3] = [200, 100, 50]
    pal = naive_mean_palette(rg, lab)
    assert np.allclose(pal[0], [10, 20, 30])
    assert np.allclose(pal[3], [200, 100, 50])


def test_naive_mean_palette_absent_class_zero():
    rg = np.full((SEG_H, SEG_W, 3), 42.0, np.float32)
    lab = np.zeros((SEG_H, SEG_W), np.int64)  # only class 0 present
    pal = naive_mean_palette(rg, lab)
    assert np.allclose(pal[0], 42.0)
    for c in (1, 2, 3, 4):
        assert np.allclose(pal[c], 0.0)


def test_global_mean_palette_pixel_weighted():
    labs = np.zeros((2, SEG_H, SEG_W), np.int64)
    f0 = np.full((SEG_H, SEG_W, 3), 100.0, np.float32)
    f1 = np.full((SEG_H, SEG_W, 3), 200.0, np.float32)
    pal = global_mean_palette([f0, f1], labs)  # all class 0
    assert np.allclose(pal[0], 150.0)  # equal pixel counts -> mean of 100,200


def test_paint_render_grid_maps_palette():
    pal = REF.copy()
    lab = np.zeros((SEG_H, SEG_W), np.int64)
    lab[10:20, 10:20] = 2
    painted = paint_render_grid(pal, lab)
    assert painted.shape == (SEG_H, SEG_W, 3)
    assert np.allclose(painted[0, 0], REF[0])
    assert np.allclose(painted[15, 15], REF[2])


def test_paint_camera_res_uint8_hard_colors():
    pal = REF.copy()
    lab = np.zeros((SEG_H, SEG_W), np.int64)
    lab[:, SEG_W // 2 :] = 4
    cam = paint_camera_res_uint8(pal, lab)
    assert cam.shape == (CAMERA_H, CAMERA_W, 3)
    assert cam.dtype == np.uint8
    # nearest-up keeps colours hard: every camera pixel is exactly one palette colour.
    uniq = {tuple(c) for c in cam.reshape(-1, 3)[:: 5000]}
    for u in uniq:
        assert any(np.allclose(u, r) for r in REF), u


def test_dilate_presence_radius1():
    b = np.zeros((5, 5), bool)
    b[2, 2] = True
    d = _dilate_presence(b, 1)
    assert d[1:4, 1:4].all()  # 3x3 around center
    assert not d[0, 0]


def test_decompose_flips_third_vs_boundary():
    # class 0 everywhere except col 2 = class 1 (0 and 1 are adjacent).
    gt = np.zeros((1, 5, 5), np.int64)
    gt[0, :, 2] = 1
    realized = gt.copy()
    realized[0, 0, 2] = 0  # gt=1 -> 0 : 0 is locally adjacent -> BOUNDARY
    realized[0, 4, 0] = 3  # gt=0 -> 3 : 3 absent everywhere -> THIRD-CLASS
    d = decompose_flips(realized, gt, radius=1)
    assert d.total_flips == 2
    assert d.boundary_flips == 1
    assert d.third_class_flips == 1
    assert d.third_class_share == pytest.approx(0.5)


def test_decompose_flips_zero_on_match():
    gt = np.zeros((2, 6, 6), np.int64)
    gt[:, 3:, :] = 2
    d = decompose_flips(gt.copy(), gt, radius=2)
    assert d.total_flips == 0
    assert d.third_class_share == 0.0


def test_decompose_flips_empty_stack_no_zerodiv():
    # EMPTY (0,H,W) stack passes ndim==3 + shape-equal guards; total_pixels==0 must NOT raise a
    # raw ZeroDivisionError (int 0/0) — it is a clean all-zero decomposition (r11 F1; sister of the
    # r10 flip_inverse empty-flip-set no-op).
    empty = np.zeros((0, 4, 4), np.int64)
    d = decompose_flips(empty, empty)
    assert d.total_pixels == 0
    assert d.total_flips == 0
    assert d.third_class_frac_of_pixels == 0.0
    assert d.boundary_frac_of_pixels == 0.0
    assert d.third_class_share == 0.0


def test_mixing_robust_palette_abstract_maps_correctly(dg):
    pal = mixing_robust_palette(dg)
    assert pal.shape == (5, 3)
    for c in range(5):
        assert classify_color(dg, pal[c]) == c  # each colour still decodes to its class


def test_mixing_robust_palette_scene_anchored_stays_near_seed(dg):
    seed = np.array([[5, 5, 5], [250, 5, 5], [5, 250, 5], [5, 5, 250], [250, 250, 250]], np.float32)
    pal = mixing_robust_palette(dg, seed_palette=seed, neighbor_dist=64.0)
    # each optimised colour stays within neighbor_dist of its seed (minimal perturbation).
    for c in range(5):
        assert np.linalg.norm(pal[c] - seed[c]) <= 64.0 + 1e-6


def test_realize_no_R_perfect_partition_is_zero(stub):
    # hard REF paint of a partition, zero resolution mixing -> stub reproduces it exactly.
    lab = np.zeros((SEG_H, SEG_W), np.int64)
    lab[:, SEG_W // 2 :] = 1
    frames = [paint_render_grid(REF, lab), paint_render_grid(REF, lab)]
    realized = realize_argmax_no_R(stub, frames, verdict_batch=2)
    gt = np.stack([lab, lab])
    assert (realized != gt).mean() == 0.0


def test_run_arm_end_to_end_no_R(stub):
    lab = np.zeros((SEG_H, SEG_W), np.int64)
    lab[10:30, 10:30] = 3
    frames = [paint_render_grid(REF, lab), paint_render_grid(REF, lab)]
    gt = np.stack([lab, lab])
    res = run_arm("t", frames, gt, segnet=stub, no_R=True, verdict_batch=2,
                  allow_subset_reason="unit test n=2")
    assert isinstance(res, ArmResult)
    assert res.agg_dseg == 0.0
    assert res.decomposition.total_flips == 0


def test_run_arm_n600_discipline_refuses_subset(stub):
    lab = np.zeros((SEG_H, SEG_W), np.int64)
    frames = [paint_render_grid(REF, lab), paint_render_grid(REF, lab)]
    gt = np.stack([lab, lab])
    with pytest.raises(PaletteRealizationError):
        run_arm("t", frames, gt, segnet=stub, no_R=True, verdict_batch=2)  # no reason -> refuse


def test_bad_shapes_raise():
    with pytest.raises(PaletteRealizationError):
        naive_mean_palette(np.zeros((3, 3, 3), np.float32), np.zeros((3, 3), np.int64))
    with pytest.raises(PaletteRealizationError):
        paint_render_grid(REF, np.zeros((3, 3), np.int64))
    with pytest.raises(PaletteRealizationError):
        decompose_flips(np.zeros((2, 4, 4), np.int64), np.zeros((2, 5, 5), np.int64))


def test_arm_result_to_measurement_rows(stub):
    lab = np.zeros((SEG_H, SEG_W), np.int64)
    frames = [paint_render_grid(REF, lab), paint_render_grid(REF, lab)]
    gt = np.stack([lab, lab])
    res = run_arm("t", frames, gt, segnet=stub, no_R=True, verdict_batch=2,
                  allow_subset_reason="unit test n=2")
    rows = arm_result_to_measurement_rows(res, git_sha="deadbeef", review_status="provisional")
    assert len(rows) >= 1
    from tac.verdicts import AxisTag

    assert all(r.axis_tag == AxisTag.THROUGH_R for r in rows)
