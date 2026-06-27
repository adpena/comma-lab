"""Tests for the sky/undrivable + road-complement + joint-containment component (FEED-dw).

NO-FAKE: synthetic-but-faithful fixtures (a static TOP sky band, a static BOTTOM hood band,
a thin lane in the road band, a compact moving blob) exercise the REAL math the functions
name — data-driven region classification, horizon-line fit, half-plane rasterization, scipy-EDT
signed distance, road-as-complement, full per-class confusion decomposition. The decisive
real-data verdict lives in the measure script + memo; these guard the primitives.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.road_horizon_component import (
    HorizonLine,
    build_sky_line_sdf,
    build_static_sky_sdf,
    classify_segnet_regions,
    decompose_full_confusion,
    fit_horizon_line,
    mean_full_decomp,
    rasterize_sky_above_line,
    road_complement_byte_cost,
    road_complement_field,
    sky_line_byte_cost,
)
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields

_H, _W = 384, 512


def _synthetic_scene(n=6, h=_H, w=_W, seed=0):
    """n faithful frames: rows[0,sky_b) = sky(2, static, slight jitter), rows[hood_t,h) =
    hood(4, static), middle = road(0); a thin near-vertical lane(1) in the road band; a
    compact moving blob movable(3, larger area than lane). Returns (n,h,w) int64."""

    rng = np.random.default_rng(seed)
    L = np.zeros((n, h, w), np.int64)
    sky_b = int(h * 0.45)     # sky occupies top ~45% (rows below the horizon 174 for IPM)
    hood_t = int(h * 0.80)
    for i in range(n):
        L[i, :, :] = 0  # road
        jit = int(rng.integers(-3, 4))
        L[i, : sky_b + jit, :] = 2          # static-ish top sky
        L[i, hood_t:, :] = 4                # static bottom hood
        # thin lane lines (class 1) — near-vertical in the road band (below horizon row 174),
        # 1px wide and jittering laterally per frame (real lanes move -> low static IoU)
        ljit = int(rng.integers(-2, 3))
        for u0 in (int(w * 0.35), int(w * 0.62)):
            uu = u0 + ljit + (np.arange(sky_b + 5, hood_t) - sky_b) // 12  # gentle slant + jitter
            for r, u in zip(range(sky_b + 5, hood_t), uu):
                if 0 <= u < w:
                    L[i, r, u] = 1   # 1px wide line (thin -> small area)
        # a compact moving blob (class 3), LARGER area than the thin lanes
        cr = int(rng.integers(int(h * 0.50), int(h * 0.65)))
        cc = int(rng.integers(int(w * 0.20), int(w * 0.78)))
        L[i, cr: cr + 40, cc: cc + 40] = 3
    return L


# ---------------------------------------------------------------------------
# data-driven region classification (NO-FAKE self-detection)
# ---------------------------------------------------------------------------
def test_classify_regions_is_permutation_and_correct_roles():
    L = _synthetic_scene(n=6)
    roles = classify_segnet_regions(L, n_classes=5)
    # permutation
    assert sorted([roles.road, roles.lane, roles.sky, roles.movable, roles.hood]) == [0, 1, 2, 3, 4]
    # the static spatial roles must be detected exactly
    assert roles.sky == 2, roles.as_dict()
    assert roles.hood == 4, roles.as_dict()
    assert roles.road == 0, roles.as_dict()
    # lane (thin, more lane-lines / smaller area) vs movable (compact blob, larger)
    assert roles.lane == 1, roles.as_dict()
    assert roles.movable == 3, roles.as_dict()


def test_classify_regions_self_detects_under_label_permutation():
    """Permute the class labels; the role detection must follow the DATA, not the index."""
    L = _synthetic_scene(n=6)
    perm = {0: 3, 1: 0, 2: 1, 3: 4, 4: 2}  # arbitrary relabel
    Lp = np.vectorize(perm.get)(L)
    roles = classify_segnet_regions(Lp, n_classes=5)
    assert roles.sky == perm[2]      # sky followed its pixels
    assert roles.hood == perm[4]
    assert roles.road == perm[0]
    assert sorted(roles.as_dict().values()) == sorted(perm.values())


def test_region_evidence_top_bottom_shares():
    L = _synthetic_scene(n=4)
    roles = classify_segnet_regions(L, n_classes=5)
    by = {e.cls: e for e in roles.evidence}
    assert by[roles.sky].top_share > 0.5      # sky concentrated up top
    assert by[roles.hood].bottom_share > 0.9  # hood concentrated at bottom
    assert by[roles.sky].static_iou > 0.5     # sky near-static


# ---------------------------------------------------------------------------
# horizon line model
# ---------------------------------------------------------------------------
def test_fit_horizon_line_recovers_flat_boundary():
    h, w = 120, 160
    a = np.zeros((h, w), np.int64)
    a[:50, :] = 2  # flat sky boundary at row 50
    line = fit_horizon_line(a, sky_cls=2, deg=1)
    assert line is not None
    rows = line.horizon_row(np.arange(w))
    assert np.allclose(rows, 50, atol=1.0)
    assert line.n_floats() == 2


def test_fit_horizon_line_recovers_tilt():
    h, w = 120, 200
    a = np.zeros((h, w), np.int64)
    for u in range(w):
        b = int(40 + 0.05 * u)  # tilted horizon
        a[:b, u] = 2
    line = fit_horizon_line(a, sky_cls=2, deg=1)
    assert line is not None
    assert line.coeffs[0] == pytest.approx(0.05, abs=0.02)


def test_fit_horizon_line_none_when_no_sky():
    a = np.zeros((40, 40), np.int64)
    assert fit_horizon_line(a, sky_cls=2, deg=1) is None


def test_rasterize_sky_above_line_is_halfplane():
    line = HorizonLine(coeffs=np.array([0.0, 50.0]), deg=1, n_fit_cols=10, rms_row=0.0)
    m = rasterize_sky_above_line(line, h=100, w=80)
    assert m[:50, :].all()       # above the line = sky
    assert not m[51:, :].any()   # below = not sky


def test_build_sky_line_sdf_sign():
    h, w = 100, 120
    a = np.zeros((h, w), np.int64)
    a[:40, :] = 2
    phi, meta = build_sky_line_sdf(a, sky_cls=2, deg=1, h=h, w=w)
    assert meta["fit"]
    assert phi[5, 60] > 0      # deep in sky -> positive SDF
    assert phi[80, 60] < 0     # deep below -> negative
    assert phi.shape == (h, w)


# ---------------------------------------------------------------------------
# static sky mask SDF (reuses hood template)
# ---------------------------------------------------------------------------
def test_build_static_sky_sdf_sign_and_diag():
    L = _synthetic_scene(n=5)
    phi, sm = build_static_sky_sdf(L, sky_cls=2, agg="majority")
    assert phi.shape == L.shape[1:]
    assert sm.hood_cls == 2
    assert sm.mean_frame_iou > 0.5
    assert phi[5, 256] > 0      # top region (sky) positive
    assert phi[300, 256] < 0    # bottom (hood) negative


# ---------------------------------------------------------------------------
# road as complement (constant background field)
# ---------------------------------------------------------------------------
def test_road_complement_field_constant():
    f = road_complement_field(10, 12, level=0.0)
    assert f.shape == (10, 12)
    assert np.all(f == 0.0)


def test_road_complement_byte_cost_is_zero_video_derived():
    bc = road_complement_byte_cost()
    assert bc["counted_bytes"] == 0
    assert bc["score_rate_contribution"] == 0.0


def test_road_falls_out_as_complement_with_ideal_others():
    """With ideal sky/lane/hood/movable SDFs and road = constant 0, the argmax must
    reconstruct the partition (road is exactly the complement of the positive regions)."""
    L = _synthetic_scene(n=2)[0]
    phi = signed_distance_fields(L, 5)
    phi[..., 0] = 0.0  # road = constant complement
    pred = phi.argmax(-1)
    # ideal others + complement road -> near-exact reconstruction
    assert float(np.mean(pred != L)) < 1e-3


# ---------------------------------------------------------------------------
# byte cost (horizon line)
# ---------------------------------------------------------------------------
def test_sky_line_byte_cost_scales_with_frames():
    coeffs = np.random.default_rng(0).normal(size=(20, 2))
    bc = sky_line_byte_cost(coeffs, n_frames=600)
    assert bc["n_coef_per_frame"] == 2
    assert bc["best_counted_bytes"] > 0
    assert bc["bytes_per_frame"] == pytest.approx(bc["best_counted_bytes"] / 600.0)


# ---------------------------------------------------------------------------
# full per-class confusion decomposition (the joint containment metric)
# ---------------------------------------------------------------------------
def test_decompose_full_confusion_perfect():
    L = _synthetic_scene(n=1)[0]
    d = decompose_full_confusion(L, L, n_classes=5)
    assert d.total_dseg == 0.0
    assert all(x == 0.0 for x in d.fn)
    assert all(x == 0.0 for x in d.fp)


def test_decompose_full_confusion_known():
    L = np.array([[0, 0], [1, 2]], np.int64)
    P = np.array([[0, 1], [1, 2]], np.int64)  # one pixel (0,1): true 0 -> pred 1
    d = decompose_full_confusion(P, L, n_classes=3)
    assert d.total_dseg == pytest.approx(0.25)
    assert d.fn[0] == pytest.approx(0.25)   # class 0 missed one
    assert d.fp[1] == pytest.approx(0.25)   # class 1 over-claimed one
    assert d.confusion[0][1] == pytest.approx(0.25)
    leaks = d.leak_into(1)
    assert leaks[0] == (0, pytest.approx(0.25))


def test_decompose_full_confusion_total_matches_mismatch():
    rng = np.random.default_rng(3)
    L = rng.integers(0, 5, size=(40, 50))
    P = rng.integers(0, 5, size=(40, 50))
    d = decompose_full_confusion(P, L, n_classes=5)
    assert d.total_dseg == pytest.approx(float(np.mean(P != L)))
    # FN sum + diagonal correct == total true; FN total == FP total == total_dseg (closed system)
    assert pytest.approx(sum(d.fn), abs=1e-9) == d.total_dseg
    assert pytest.approx(sum(d.fp), abs=1e-9) == d.total_dseg


def test_mean_full_decomp_averages():
    L = _synthetic_scene(n=1)[0]
    d1 = decompose_full_confusion(L, L, n_classes=5)
    P = L.copy()
    P[0, 0] = (P[0, 0] + 1) % 5
    d2 = decompose_full_confusion(P, L, n_classes=5)
    m = mean_full_decomp([d1, d2], n_classes=5)
    assert m["total_dseg"] == pytest.approx((d1.total_dseg + d2.total_dseg) / 2.0)
    assert len(m["fn"]) == 5
    assert len(m["confusion"]) == 5


def test_ideal_sdf_argmax_reconstructs_partition():
    """Harness sanity: the ideal SDF stack argmax == labels exactly (the baseline)."""
    L = _synthetic_scene(n=1)[0]
    phi = signed_distance_fields(L, 5)
    assert np.array_equal(phi.argmax(-1), L)
