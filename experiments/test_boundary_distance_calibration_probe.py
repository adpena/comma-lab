"""Tests for probe_boundary_distance_calibration (CE-window pre-stage calibration harness).

CPU-fast, synthetic; the trainer-imported bd functions are exercised against their documented
semantics (band geometry, tie-at-boundary zero, monotone-in-offset). No live-run access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "upstream"), str(_REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx

mx.set_default_device(mx.cpu)

from probe_boundary_distance_calibration import (
    BD_WEIGHTS,
    bd_ratio,
    recover_tex_effective,
)
from train_levelset_witness_realized_through_R_mlx import (
    boundary_distance_band_map,
    boundary_distance_term_mlx,
)


def _render_np(phi: np.ndarray, tex: np.ndarray, palette: np.ndarray, temp: float) -> np.ndarray:
    z = phi.astype(np.float64) / temp
    z = z - z.max(axis=-1, keepdims=True)
    soft = np.exp(z)
    soft = soft / soft.sum(axis=-1, keepdims=True)
    return (1.0 / (1.0 + np.exp(-(soft @ palette + tex)))) * 255.0


def test_recover_tex_effective_roundtrip():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(64, 5)).astype(np.float32)
    tex = rng.normal(scale=0.5, size=(64, 3)).astype(np.float32)
    palette = rng.normal(size=(5, 3)).astype(np.float32)
    rgb = _render_np(phi, tex.astype(np.float64), palette.astype(np.float64), 1.0).astype(np.float32)
    tex_rec = recover_tex_effective(rgb, phi, palette, 1.0)
    rgb2 = _render_np(phi, tex_rec.astype(np.float64), palette.astype(np.float64), 1.0)
    assert np.max(np.abs(rgb2 - rgb)) < 1e-3  # exact inverse up to fp32 rgb storage


def test_recover_tex_respects_softmax_temp():
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(32, 5)).astype(np.float32)
    tex = rng.normal(scale=0.3, size=(32, 3)).astype(np.float32)
    palette = rng.normal(size=(5, 3)).astype(np.float32)
    for temp in (0.5, 2.0):
        rgb = _render_np(phi, tex.astype(np.float64), palette.astype(np.float64), temp).astype(np.float32)
        tex_rec = recover_tex_effective(rgb, phi, palette, temp)
        rgb2 = _render_np(phi, tex_rec.astype(np.float64), palette.astype(np.float64), temp)
        assert np.max(np.abs(rgb2 - rgb)) < 1e-3


def test_band_map_geometry_two_region():
    # vertical edge between col 3 and 4: straddle pixels (cols 3,4) weight 1.0, ramp to 0 at 2px
    ls = np.zeros((8, 8), np.int64)
    ls[:, 4:] = 2
    band = boundary_distance_band_map(ls, band_px=2.0)
    assert band.shape == (8, 8) and band.dtype == np.float32
    assert np.all(band[:, 3] == 1.0) and np.all(band[:, 4] == 1.0)  # ON the boundary
    assert np.allclose(band[:, 2], 0.5) and np.allclose(band[:, 5], 0.5)  # 1px away: 1-1/2
    assert np.all(band[:, 0] == 0.0) and np.all(band[:, 7] == 0.0)  # outside the band


def test_band_map_degenerate_single_class_is_zero():
    band = boundary_distance_band_map(np.zeros((6, 6), np.int64))
    assert band.shape == (6, 6) and float(band.sum()) == 0.0


def test_bd_term_zero_at_tie_and_monotone_in_offset():
    h = w = 8
    ls = np.zeros((h, w), np.int64)
    ls[:, 4:] = 1
    band = mx.array(boundary_distance_band_map(ls))
    oh = np.zeros((1, h, w, 5), np.float32)
    for k in range(5):
        oh[0, :, :, k] = (ls == k)
    oh_mx = mx.array(oh)
    vals = []
    for off in (0.0, 0.5, 2.0):
        # GT field and top competitor tied everywhere except a uniform offset:
        phi = np.zeros((h * w, 5), np.float32) - 10.0
        flat_ls = ls.reshape(-1)
        phi[np.arange(h * w), flat_ls] = 1.0            # GT field
        phi[np.arange(h * w), 1 - flat_ls] = 1.0 - off  # competitor at gap=off
        vals.append(float(boundary_distance_term_mlx(mx.array(phi), oh_mx, band, h, w)))
    assert abs(vals[0]) < 1e-6                     # tie on the band => zero placement loss
    assert vals[0] < vals[1] < vals[2]             # monotone in |gap|
    assert abs(vals[2] - 2.0) < 1e-5               # band-weighted mean of a uniform gap == gap


def test_bd_term_differentiable_wrt_phi():
    h = w = 6
    ls = np.zeros((h, w), np.int64)
    ls[3:, :] = 2
    band = mx.array(boundary_distance_band_map(ls))
    oh = np.zeros((1, h, w, 5), np.float32)
    for k in range(5):
        oh[0, :, :, k] = (ls == k)
    oh_mx = mx.array(oh)
    rng = np.random.default_rng(2)
    phi0 = mx.array(rng.normal(size=(h * w, 5)).astype(np.float32))
    g = mx.grad(lambda p: boundary_distance_term_mlx(p, oh_mx, band, h, w))(phi0)
    mx.eval(g)
    g_np = np.asarray(g)
    assert g_np.shape == (h * w, 5) and np.isfinite(g_np).all()
    band_np = np.asarray(band)
    on_band = band_np.reshape(-1) > 0
    assert np.abs(g_np[on_band]).sum() > 0.0       # gradient lives on the band
    assert np.abs(g_np[~on_band]).sum() < 1e-6     # and ONLY on the band


def test_bd_ratio_math():
    assert bd_ratio(10.0, 5.0, 0.0) == 0.0
    assert abs(bd_ratio(10.0, 5.0, 1.0) - (5.0 / 15.0)) < 1e-12
    assert abs(bd_ratio(9.0, 20.0, 0.05) - (1.0 / 10.0)) < 1e-12
    assert bd_ratio(0.0, 0.0, 1.0) == 0.0  # degenerate-safe


def test_bd_weights_grid_matches_prompt():
    assert BD_WEIGHTS == (0.0, 0.01, 0.05, 0.1, 0.5, 1.0)


def test_load_live_params_strips_prefix(tmp_path):
    from probe_boundary_distance_calibration import _load_live_params
    p = tmp_path / "resume.npz"
    np.savez(p, **{"liveP__in_proj.weight": np.ones((4, 3), np.float32),
                   "liveP__pose_carrier.xi_stored": np.zeros(6, np.float32),
                   "liveP__code": np.zeros((2, 8), np.float32),
                   "__resume_epoch": np.asarray(100)})
    out = _load_live_params(p)
    assert set(out) == {"in_proj.weight", "code"}  # cfg + pose_carrier excluded


def test_resume_clear_spike_guard_flag_registered():
    # The CE-window pre-stage escape flag: registered in the trainer argparse, default OFF, and
    # the clear branch emits the durable log token. Functional proof = the executed n1 resume
    # smoke (cleared_frozen_window_len=2 then training continued; default path restored 2).
    src = Path("experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    assert '"--resume-clear-spike-guard"' in src
    assert "default=False" in src.split('"--resume-clear-spike-guard"')[1][:400]
    assert "cleared_frozen_window_len" in src


def test_load_live_params_refuses_non_sidecar(tmp_path):
    from probe_boundary_distance_calibration import _load_live_params
    p = tmp_path / "bad.npz"
    np.savez(p, **{"liveP__something": np.zeros(3, np.float32)})
    try:
        _load_live_params(p)
    except ValueError:
        return
    raise AssertionError("expected ValueError on a non-sidecar npz")
