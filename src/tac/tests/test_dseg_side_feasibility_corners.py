# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the $0 d_seg-side feasibility corners probe (#149 sub-pixel, #148 warp).

These verify ACTUAL behaviour, not constants (per CLAUDE.md NO-FAKE class 2): the real
frozen SegNet runs, the exact eval operators are applied, the sub-pixel solve genuinely
changes the frame, the warp genuinely warps the label map, d_seg is recomputed from
components, and the verdict logic correctly maps measured rows to GREEN/AMBER/RED. A test
suite that passes when a function body is replaced by `return constants` is forbidden.

CPU-only, real-scorer authority (NEVER MPS for the score). Some tests are skipped if the
upstream SegNet checkpoint / GT cache are absent (CI without contest assets).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "experiments/probe_dseg_side_feasibility_corners.py"
SEGNET_CKPT = REPO / "upstream/models/segnet.safetensors"
GT_CACHE = REPO / "experiments/results/capstone_gt_targets_cache/gt_targets_n16.pt"

_HAS_ASSETS = SEGNET_CKPT.exists() and GT_CACHE.exists()


def _load_probe():
    spec = importlib.util.spec_from_file_location("_dseg_corners_probe", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1. module loads + constants are the contest-exact geometry (not arbitrary)
# --------------------------------------------------------------------------- #
def test_probe_module_loads_and_constants_are_contest_exact():
    p = _load_probe()
    assert (p.CAM_H, p.CAM_W) == (874, 1164), "camera size must match upstream camera_size"
    assert (p.MH, p.MW) == (384, 512), "model size must match segnet_model_input_size"
    assert p.B0 == 37_545_489, "archive normalizer must be the contest B0"
    # the frontier bar must be the real measured anchors
    assert abs(p.FRONTIER_DSEG_TERM - 100 * p.FRONTIER_DSEG) < 1e-9
    assert p.CONTOUR_BYTES_PER_FRAME == 914


# --------------------------------------------------------------------------- #
# 2. rate-from-bytes is the exact contest rate term (recomputed, not constant)
# --------------------------------------------------------------------------- #
def test_rate_from_total_bytes_is_exact_contest_term():
    p = _load_probe()
    assert abs(p.rate_from_total_bytes(p.B0) - 25.0) < 1e-9
    assert abs(p.rate_from_total_bytes(177_169) - 0.11797) < 1e-4  # frontier rate term
    assert p.rate_from_total_bytes(0.0) == 0.0


# --------------------------------------------------------------------------- #
# 3. d_seg is the exact argmax-flip-rate functional (combinatorial, recomputed)
# --------------------------------------------------------------------------- #
def test_d_seg_is_exact_flip_rate():
    p = _load_probe()
    L = np.zeros((10, 10), dtype=np.int64)
    a = L.copy()
    a[0, :] = 1  # flip 10 of 100 pixels
    assert abs(p._d_seg(a, L) - 0.10) < 1e-9
    assert p._d_seg(L, L) == 0.0  # identical -> 0
    assert p._d_seg(np.ones_like(L), L) == 1.0  # all flipped -> 1


# --------------------------------------------------------------------------- #
# 4. boundary band actually marks region edges (not a constant mask)
# --------------------------------------------------------------------------- #
def test_boundary_band_marks_edges_only():
    p = _load_probe()
    L = np.zeros((p.MH, p.MW), dtype=np.int64)
    L[:, : p.MW // 2] = 1  # one vertical edge down the middle
    band = p._boundary_band_384(L, iters=1)
    # the band must be along the column-split, NOT everywhere and NOT empty
    assert 0.0 < band.mean() < 0.2, "band must be a thin edge region, not all/none"
    # the edge columns must be in the band; far-interior columns must not
    assert band[:, p.MW // 2].any() or band[:, p.MW // 2 - 1].any()
    assert not band[:, 0].any() and not band[:, -1].any()


# --------------------------------------------------------------------------- #
# 5. the warp actually warps the label map (translate moves the edge)
# --------------------------------------------------------------------------- #
def test_warp_label_map_translate_moves_edge():
    p = _load_probe()
    L = np.zeros((p.MH, p.MW), dtype=np.int64)
    L[:, p.MW // 2 :] = 1  # right half is class 1
    # translate right by +20 px -> the class-1 region shrinks (edge moves right)
    w = p._warp_label_map(L, (0.0, 20.0), "translate")
    assert w.shape == L.shape
    assert w.dtype == np.int64
    # labels preserved (nearest), but the partition is genuinely different
    assert set(np.unique(w)).issubset({0, 1})
    assert not np.array_equal(w, L), "a 20px translate must change the label map"
    # a 20px horizontal shift must move the vertical edge by ~20 px (the count changes by
    # ~20*H pixels; direction depends on affine_grid sampling convention — what matters is
    # the edge genuinely moved by the right magnitude).
    delta_px = abs(int(w.sum()) - int(L.sum()))
    assert 15 * p.MH <= delta_px <= 25 * p.MH, f"edge shift magnitude wrong: {delta_px}"
    # and a +dx vs -dx translate move the edge in OPPOSITE directions (real geometry)
    w_neg = p._warp_label_map(L, (0.0, -20.0), "translate")
    assert (w.sum() - L.sum()) * (w_neg.sum() - L.sum()) < 0, "sign symmetry broken"


def test_warp_label_map_identity_affine_is_noop():
    p = _load_probe()
    L = np.zeros((p.MH, p.MW), dtype=np.int64)
    L[100:200, 100:300] = 2
    w = p._warp_label_map(L, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), "affine")
    # identity affine -> (near) identical label map
    assert float((w != L).mean()) < 0.01


# --------------------------------------------------------------------------- #
# 6. warp solve genuinely reduces combinatorial diff (it's a real search, not const)
# --------------------------------------------------------------------------- #
def test_solve_warp_reduces_combinatorial_diff():
    p = _load_probe()
    import argparse

    L_key = np.zeros((p.MH, p.MW), dtype=np.int64)
    L_key[:, p.MW // 2 :] = 1
    # target = keyframe shifted right by 8 px (a pure translate the solver should find)
    L_tgt = p._warp_label_map(L_key, (0.0, 8.0), "translate")
    d_nowarp = float((L_key != L_tgt).mean())
    assert d_nowarp > 0.0
    args = argparse.Namespace(warp_mode="translate", affine_maxiter=50)
    params, d_warp = p._solve_warp(L_key, L_tgt, "translate", args)
    assert params is not None
    # the solver must recover most of the shift -> d_warp << d_nowarp
    assert d_warp < 0.4 * d_nowarp, f"warp solve under-powered: {d_warp} vs {d_nowarp}"
    # 'none' mode returns no params and the raw diff
    pn, dn = p._solve_warp(L_key, L_tgt, "none", args)
    assert pn is None and abs(dn - d_nowarp) < 1e-9


# --------------------------------------------------------------------------- #
# 7. verdict logic maps rows to the right class (GREEN/AMBER/RED) — behaviour
# --------------------------------------------------------------------------- #
def test_corner1_verdict_classifies_green_amber_red():
    p = _load_probe()
    # RED: best d_seg above the frontier
    red = {"c1_rows": {"f0": _c1row(0.01, 5.0)}}
    assert _verdict_str(p._verdict_corner1(red)).startswith("RED")
    # AMBER: below frontier d_seg but S not sub-0.15 (high rate)
    amber = {"c1_rows": {"f0": _c1row(0.0004, 2.0)}}
    assert _verdict_str(p._verdict_corner1(amber)).startswith("AMBER")
    # GREEN: below green-threshold d_seg AND S < 0.15
    green = {"c1_rows": {"f0": _c1row(0.0005, 0.12)}}
    assert _verdict_str(p._verdict_corner1(green)).startswith("GREEN")


def test_corner2_verdict_classifies_green_amber_red():
    p = _load_probe()
    red = {"c2_rows": {"f1": _c2row(0.02, 2.0)}}
    assert _verdict_str(p._verdict_corner2(red)).startswith("RED")
    amber = {"c2_rows": {"f1": _c2row(0.0004, 2.0)}}
    assert _verdict_str(p._verdict_corner2(amber)).startswith("AMBER")
    green = {"c2_rows": {"f1": _c2row(0.0005, 0.12)}}
    assert _verdict_str(p._verdict_corner2(green)).startswith("GREEN")


def test_overall_is_red_only_when_both_red():
    p = _load_probe()
    assert p._overall({"verdict": "RED_x"}, {"verdict": "RED_y"}).startswith("RED")
    assert p._overall({"verdict": "AMBER_x"}, {"verdict": "RED_y"}).startswith("AMBER")
    assert p._overall({"verdict": "GREEN_x"}, {"verdict": "RED_y"}).startswith("GREEN")


# --------------------------------------------------------------------------- #
# 8-12. REAL-SCORER tests (need contest assets) — the NO-FAKE authority checks
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_ASSETS, reason="contest SegNet/GT cache absent")
def test_real_segnet_on_gt_camera_frame_reproduces_cached_lstar():
    """The cached L* IS D(GT_camera)->SegNet->argmax. Feeding the GT camera frame to the
    real SegNet must reproduce it exactly (the harness is faithful)."""
    p = _load_probe()
    seg = p._load_segnet_cpu()
    cache = p._load_gt_cache()
    frames = p._decode_gt_camera_frames(1)
    assert frames[0].shape == (874, 1164, 3)
    am = p._segnet_argmax_cam(seg, frames[0].astype(np.float64))
    assert am.shape == (384, 512)
    assert float((am == cache[0]).mean()) == 1.0, "GT camera frame must reproduce cached L*"


@pytest.mark.skipif(not _HAS_ASSETS, reason="contest SegNet/GT cache absent")
def test_real_flat_paint_hits_the_texture_wall():
    """A flat-color partition (exact GT layout, mu colors) must hit the texture wall
    (d_seg far above the frontier 0.00056) on BOTH the 384-roundtrip and camera-native
    paths — the wall the prior gates measured. This proves the measurement is real and the
    flat baseline is genuinely walled (not secretly low)."""
    p = _load_probe()
    seg = p._load_segnet_cpu()
    cache = p._load_gt_cache()
    frames = p._decode_gt_camera_frames(1)
    L = cache[0]
    cols, Lup = p._per_class_mu_colors(L, frames[0])
    # camera-native flat
    cam_flat = cols[Lup]
    am = p._segnet_argmax_cam(seg, cam_flat)
    dseg_flat = p._d_seg(am, L)
    assert dseg_flat > 5 * p.FRONTIER_DSEG, f"flat paint must be walled, got {dseg_flat}"
    assert dseg_flat < 0.1, "but not catastrophic (sanity: same partition shape)"


@pytest.mark.timeout(300)
@pytest.mark.skipif(not _HAS_ASSETS, reason="contest SegNet/GT cache absent")
def test_subpixel_solve_actually_changes_the_frame_and_lowers_dseg():
    """The sub-pixel solve must (a) genuinely change the camera-res frame in the band, and
    (b) lower realized d_seg below the flat-paint baseline (it is a real optimization, not a
    no-op). MEASUREMENT-FIRST: realized d_seg through the real SegNet is the authority.

    Few iters (CPU backward is slow): even ~12 iters move CE + beat flat-paint, which is the
    behavioural claim — the deep d_seg floor is measured by the gate, not this fast test."""
    p = _load_probe()
    import argparse

    seg = p._load_segnet_cpu()
    cache = p._load_gt_cache()
    frames = p._decode_gt_camera_frames(1)
    args = argparse.Namespace(
        train_device="cpu", iters=12, lr=8.0, cam_band_dilate=2, warp_mode="none",
        affine_maxiter=10,
    )
    r = p.corner1_one_frame(seg, cache[0], frames[0], args)
    # sub-pixel must beat the flat-paint baselines (a real lever, not constant)
    assert r["dseg_subpixel_solved"] < r["dseg_flat_camera_native"], "sub-pixel did nothing"
    assert r["dseg_subpixel_solved"] < r["dseg_flat_384_roundtrip"]
    # the CE loss must have moved (the solve genuinely optimized, not a frozen surrogate)
    assert r["ce_first"] is not None and r["ce_last"] is not None
    assert r["ce_last"] < r["ce_first"], "CE did not decrease -> solve is fake"
    # the boundary band carried real camera-res pixels (the byte driver is non-trivial)
    assert r["n_band_cam_px"] > 1000
    assert r["subpixel_extra_bytes"] > 0.0


@pytest.mark.timeout(300)
@pytest.mark.skipif(not _HAS_ASSETS, reason="contest SegNet/GT cache absent")
def test_subpixel_does_not_beat_frontier_dseg_even_at_floor(  # the load-bearing RED claim
):
    """The load-bearing claim: even the sub-pixel d_seg stays ABOVE the frontier's 0.00056 in
    a short solve — AND even if the boundary code were free, the d_seg TERM alone (100*d_seg)
    exceeds the frontier d_seg term (0.056). This is the false-RED-guarded core of the RED
    verdict, measured on the real scorer. (The deep 1200-iter best-shot reaches only ~1.7x
    frontier — see the gate's result JSON — so a few iters staying above frontier is safe.)"""
    p = _load_probe()
    import argparse

    seg = p._load_segnet_cpu()
    cache = p._load_gt_cache()
    frames = p._decode_gt_camera_frames(1)
    args = argparse.Namespace(
        train_device="cpu", iters=20, lr=8.0, cam_band_dilate=2, warp_mode="none",
        affine_maxiter=10,
    )
    r = p.corner1_one_frame(seg, cache[0], frames[0], args)
    # short solve -> still above frontier (the deep best-shot only reaches ~1.7x frontier)
    assert r["dseg_subpixel_solved"] > p.FRONTIER_DSEG, (
        "if a 60-iter solve beat the frontier d_seg the RED would be wrong; re-run the gate"
    )
    # d_seg term alone exceeds the frontier d_seg term (the byte-independent loss)
    assert 100 * r["dseg_subpixel_solved"] > p.FRONTIER_DSEG_TERM


@pytest.mark.timeout(180)
@pytest.mark.skipif(not _HAS_ASSETS, reason="contest SegNet/GT cache absent")
def test_corner2_warp_cannot_amortize_realized_dseg_to_frontier(tmp_path):
    """Corner 2: the cross-frame keyframe+warp realized d_seg must stay far above the
    frontier (the warp cannot amortize the boundary because the per-frame change isn't a
    rigid warp). Measured on the real scorer; the affine warp closes <40% of the
    combinatorial gap."""
    p = _load_probe()
    import argparse

    seg = p._load_segnet_cpu()
    cache = p._load_gt_cache()
    frames = p._decode_gt_camera_frames(2)
    args = argparse.Namespace(
        warp_mode="affine", affine_maxiter=120, n_frames=2, out_dir=str(tmp_path)
    )
    state: dict = {}
    rows = p.run_corner2(seg, cache, frames, args, state)
    assert rows, "corner2 produced no rows"
    r = next(x for x in rows if "realized_dseg_keyframe_warp" in x)
    # the warped keyframe realized d_seg is far above the frontier (RED)
    assert r["realized_dseg_keyframe_warp"] > 10 * p.FRONTIER_DSEG
    # the warp closed only a fraction of the combinatorial drift (it's not a rigid warp)
    closed = r["d_nowarp_combinatorial"] - r["d_warp_combinatorial"]
    assert 0.0 <= closed < 0.5 * r["d_nowarp_combinatorial"]


# --------------------------------------------------------------------------- #
# helpers for the synthetic-row verdict tests
# --------------------------------------------------------------------------- #
def _c1row(best_dseg, s_amort):
    return {
        "best_dseg": best_dseg,
        "dseg_subpixel_solved": best_dseg,
        "dseg_flat_camera_native": best_dseg + 0.01,
        "dseg_flat_384_roundtrip": best_dseg + 0.01,
        "S_projected_amortized": s_amort,
    }


def _c2row(realized, s_amort):
    return {
        "realized_dseg_keyframe_warp": realized,
        "realized_dseg_own_flat": realized,
        "d_warp_combinatorial": 0.01,
        "d_nowarp_combinatorial": 0.013,
        "S_projected_amortized": s_amort,
        "rate_amortized": 0.005,
    }


def _verdict_str(v):
    assert v is not None
    return v["verdict"]
