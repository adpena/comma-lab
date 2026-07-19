# SPDX-License-Identifier: MIT
"""Tests for the ARM-C #524 Lane stride-2 SKIP-BAND lever (tac.boundary_math.lane_skipband +
the trainer wire-in). Numpy reference is the authority; the MLX twin must match it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.lane_skipband import (
    avg_pool2,
    lane_band_mask_half,
    luma_bt601,
    skip_band_detail,
    skipband_target_and_mask,
    skipband_term_grad_np,
    skipband_term_np,
    up2_nearest,
)

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"

rng = np.random.default_rng(524)


def _rand_rgb(h=16, w=16):
    return rng.uniform(0, 255, size=(h, w, 3)).astype(np.float32)


# ---------------------------------------------------------------- primitives
def test_luma_bt601_range_and_shape():
    lum = luma_bt601(_rand_rgb())
    assert lum.shape == (16, 16)
    assert lum.dtype == np.float32
    assert float(lum.min()) >= 0.0 and float(lum.max()) <= 1.0


def test_luma_bt601_rejects_bad_shape():
    with pytest.raises(ValueError):
        luma_bt601(np.zeros((4, 4), np.float32))


def test_avg_pool2_known_values():
    x = np.array([[1.0, 3.0], [5.0, 7.0]], np.float32)
    assert float(avg_pool2(x)[0, 0]) == pytest.approx(4.0)


def test_avg_pool2_rejects_odd_dims():
    with pytest.raises(ValueError):
        avg_pool2(np.zeros((3, 4), np.float32))


def test_up2_nearest_repeats():
    x = np.array([[1.0, 2.0]], np.float32)
    u = up2_nearest(x)
    assert u.shape == (2, 4)
    assert np.array_equal(u, np.array([[1, 1, 2, 2], [1, 1, 2, 2]], np.float32))


def test_skip_band_zero_on_quarter_representable():
    # image constant over 4x4 blocks -> its half-res version is exactly U2(quarter) -> SB == 0.
    base = rng.uniform(0, 1, size=(4, 4)).astype(np.float32)
    lum = np.repeat(np.repeat(base, 4, axis=0), 4, axis=1)  # (16,16)
    sb = skip_band_detail(lum)
    assert np.max(np.abs(sb)) < 1e-6


def test_skip_band_nonzero_on_halfres_detail():
    # half-res-scale checkerboard survives D2 but is destroyed by D4 -> SB != 0.
    lum = np.indices((16, 16)).sum(axis=0)
    lum = ((lum // 2) % 2).astype(np.float32)  # 2px checkerboard = half-res-scale structure
    sb = skip_band_detail(lum)
    assert float(np.max(np.abs(sb))) > 0.1


def test_skip_band_is_ablation_complement():
    # SB is exactly what the fractal-memo ablation (down-up 2x at skip res) destroys:
    # D2(lum) == SB + U2(D2(D2(lum))).
    lum = luma_bt601(_rand_rgb())
    x2 = avg_pool2(lum)
    sb = skip_band_detail(lum)
    recon = sb + up2_nearest(avg_pool2(x2))
    np.testing.assert_allclose(recon, x2, atol=1e-6)


# ---------------------------------------------------------------- lane band mask
def test_lane_band_mask_selects_class_and_dilates():
    ls = np.zeros((16, 16), np.int64)
    ls[8, 8] = 1
    m0 = lane_band_mask_half(ls, dilate=0)
    m2 = lane_band_mask_half(ls, dilate=2)
    assert m0.shape == (8, 8)
    assert float(m0.sum()) == 1.0          # single lane px -> single half-res cell
    assert float(m2.sum()) > float(m0.sum())  # dilation grows the band


def test_lane_band_mask_maxpool_semantics():
    ls = np.zeros((4, 4), np.int64)
    ls[0, 0] = 1   # one px of the 2x2 cell -> the whole half-res cell is in-band
    m = lane_band_mask_half(ls, dilate=0)
    assert float(m[0, 0]) == 1.0 and float(m.sum()) == 1.0


def test_lane_band_mask_other_class_ignored():
    ls = np.full((8, 8), 3, np.int64)
    assert float(lane_band_mask_half(ls, dilate=3).sum()) == 0.0


# ---------------------------------------------------------------- term + gradient
def test_target_and_mask_shapes():
    rgb = _rand_rgb()
    ls = np.zeros((16, 16), np.int64)
    ls[4:6, :] = 1
    sb, m = skipband_target_and_mask(rgb, ls, dilate=1)
    assert sb.shape == (8, 8) and m.shape == (8, 8)


def test_term_zero_when_render_equals_gt():
    rgb = _rand_rgb()
    ls = np.zeros((16, 16), np.int64)
    ls[4:6, :] = 1
    sb, m = skipband_target_and_mask(rgb, ls, dilate=1)
    assert skipband_term_np(rgb, sb, m) == pytest.approx(0.0, abs=1e-10)


def test_term_positive_when_band_differs_and_ignores_out_of_band():
    gt = _rand_rgb()
    ls = np.zeros((16, 16), np.int64)
    ls[0:2, 0:4] = 1                                     # band in the top-left corner
    sb, m = skipband_target_and_mask(gt, ls, dilate=0)
    render = gt.copy()
    render[0:4, 0:4] += 40.0                             # in-band perturbation
    t_in = skipband_term_np(render, sb, m)
    assert t_in > 0.0
    render2 = gt.copy()
    render2[12:16, 12:16] += 40.0                        # far out-of-band perturbation
    t_out = skipband_term_np(render2, sb, m)
    assert t_out == pytest.approx(0.0, abs=1e-8)


def test_closed_form_gradient_matches_finite_difference():
    gt = _rand_rgb(8, 8)
    render = _rand_rgb(8, 8)
    ls = np.zeros((8, 8), np.int64)
    ls[2:5, 2:6] = 1
    sb, m = skipband_target_and_mask(gt, ls, dilate=1)
    g = skipband_term_grad_np(render, sb, m)             # d(term)/d(luma)
    # finite-difference on the LUMA surface: perturb one full-res luma coordinate by eps by
    # perturbing all three RGB channels equally (delta_rgb = eps*255 on each channel scaled by
    # the BT.601 sum 1.0 -> delta_luma = eps).
    eps = 1e-3
    for (i, j) in [(2, 3), (4, 4), (0, 0)]:
        r_hi = render.copy(); r_hi[i, j, :] += eps * 255.0
        r_lo = render.copy(); r_lo[i, j, :] -= eps * 255.0
        fd = (skipband_term_np(r_hi, sb, m) - skipband_term_np(r_lo, sb, m)) / (2 * eps)
        assert fd == pytest.approx(float(g[i, j]), rel=2e-3, abs=1e-7)


def test_gradient_zero_at_minimum():
    gt = _rand_rgb(8, 8)
    ls = np.zeros((8, 8), np.int64); ls[3, 3] = 1
    sb, m = skipband_target_and_mask(gt, ls, dilate=1)
    g = skipband_term_grad_np(gt, sb, m)
    assert float(np.max(np.abs(g))) < 1e-7


# ---------------------------------------------------------------- MLX twin parity
def test_mlx_twin_matches_numpy_reference():
    mx = pytest.importorskip("mlx.core")
    gt = _rand_rgb()
    render = _rand_rgb()
    ls = np.zeros((16, 16), np.int64); ls[6:9, :] = 1
    sb, m = skipband_target_and_mask(gt, ls, dilate=1)
    # mirror the trainer branch expression exactly
    f1 = mx.array(render[None])                                  # (1,H,W,3) [0,255]
    lum = (0.299 * f1[..., 0] + 0.587 * f1[..., 1] + 0.114 * f1[..., 2]) / 255.0
    H2, W2 = lum.shape[1] // 2, lum.shape[2] // 2
    l2 = mx.mean(lum.reshape(1, H2, 2, W2, 2), axis=(2, 4))
    l4 = mx.mean(l2.reshape(1, H2 // 2, 2, W2 // 2, 2), axis=(2, 4))
    up = mx.repeat(mx.repeat(l4, 2, axis=1), 2, axis=2)
    sb_wit = l2 - up
    term = mx.sum(mx.square(sb_wit - mx.array(sb[None])) * mx.array(m[None])) / (
        mx.sum(mx.array(m[None])) + 1e-6)
    ref = skipband_term_np(render, sb, m)
    assert float(term) == pytest.approx(ref, rel=1e-5, abs=1e-8)


# ---------------------------------------------------------------- trainer wire-in (source-level)
def _trainer_src() -> str:
    return _TRAINER.read_text(encoding="utf-8")


def test_trainer_has_flags_and_default_off():
    src = _trainer_src()
    for flag in ("--lane-skipband-weight", "--lane-skipband-start-epoch", "--lane-skipband-dilate"):
        assert flag in src, f"trainer argparse missing {flag}"
    assert 'ap.add_argument("--lane-skipband-weight", type=float, default=0.0' in src


def test_trainer_branch_gated_and_in_shared_forward_set():
    src = _trainer_src()
    assert 'if skipband_w > 0.0 and skipband_gate["on"] and _skipband_gt_prov is not None:' in src
    assert '(skipband_w > 0.0 and skipband_gate["on"]' in src.split("_nonwa_levers_on")[1]
    assert 'terms_out["lane_skipband"]' in src


def test_trainer_schema_and_persistence():
    src = _trainer_src()
    assert '"lane_skipband"' in src.split("LOSS_TERM_KEYS", 1)[1][:2000]
    assert '__cfg_lane_skipband_weight' in src
    from tac.witness_training_contract import LOSS_TERM_KEYS
    assert "lane_skipband" in LOSS_TERM_KEYS


def test_trainer_microbatch_fail_closed():
    src = _trainer_src()
    assert "--lane-skipband-weight > 0 requires --micro-batch-pairs 1" in src
