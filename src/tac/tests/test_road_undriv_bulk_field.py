# SPDX-License-Identifier: MIT
"""Tests for the v8 increment-1 Road+Undrivable bulk-boundary field SCAFFOLD.

Covers (task #376, owed after the scaffold's build subagent died pre-tests):
  * the VERIFIED signed lift sign convention (phi_bulk>0 -> Road, <0 -> Undrivable);
  * the byte-close row-span encode/decode is BIT-EXACT (audit row #6 owed the full-slice round-trip);
  * the geometry-native ``horizon_poly_xi_byte_cost`` (curvelet + xi, DAG FEED-v8-realmachinery) fits a
    known synthetic horizon at ~0 residual and honestly flags ``residual_sidecar_owed``;
  * gated real-cache checks (skip if the n600 gt cache is absent) that prove the self-detection returns
    the comma10k Road0/Undriv2 order and the real horizon fit lands in the measured band.

All numbers here are numpy-fp32 geometry on the frozen SegNet argmax — ``[macOS-CPU advisory ·
NON-PROMOTABLE]``. A SCAFFOLD moves no pointer; these tests only prove the composed skeleton is correct.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tac.boundary_math import road_undriv_bulk_field as R

ROAD, UNDRIV = 0, 2  # comma10k canonical (self-detection is exercised on the real cache below)
_CACHE = pathlib.Path(__file__).resolve().parents[3] / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


# ------------------------------- signed lift ----------------------------------
def test_signed_lift_sign_convention():
    phi = np.array([[-2.0, 0.5], [3.0, -1.0]], dtype=np.float32)
    p_road, p_undriv = R.signed_lift(phi, s_road=1.0, s_undriv=1.0)
    # Road channel tracks +phi; Undriv channel tracks -phi (the VERIFIED lift).
    assert np.allclose(p_road, phi)
    assert np.allclose(p_undriv, -phi)


def test_signed_lift_rejects_nonpositive_scales():
    phi = np.zeros((2, 2), np.float32)
    for bad in ((0.0, 1.0), (1.0, -1.0)):
        with pytest.raises(R.RoadUndrivBulkFieldError):
            R.signed_lift(phi, s_road=bad[0], s_undriv=bad[1])


def test_bulk_signed_lift_argmax_reproduces_side():
    # phi_bulk>0 -> Road side; <0 -> Undriv side, for ANY positive scales (asymmetric here).
    phi = np.array([[1.0, -1.0], [-0.01, 2.0]], dtype=np.float32)
    out = R.bulk_signed_lift_argmax(phi, road_cls=ROAD, undriv_cls=UNDRIV, s_road=0.3, s_undriv=5.0)
    expected = np.where(phi >= 0, ROAD, UNDRIV)
    assert np.array_equal(out, expected)


# ------------------------------- byte-close round-trip ------------------------
def test_road_row_span_roundtrip_bit_exact_multiblob():
    # A multi-blob Road mask (two separated horizontal bars + a gap) — the RLE must be exact.
    rng = np.random.default_rng(0)
    mask = np.zeros((24, 40), dtype=bool)
    mask[3:8, 5:15] = True          # blob 1
    mask[3:8, 22:30] = True         # blob 2 (same rows, disjoint columns -> multi-run rows)
    mask[15:20, 10:35] = True       # blob 3
    mask ^= rng.random(mask.shape) < 0.02  # a little salt so runs are non-trivial
    blob = R._road_row_span_encode(mask)
    back = R._road_row_span_decode(blob)
    assert back.shape == mask.shape
    assert np.array_equal(back, mask)


def test_road_row_span_roundtrip_empty_and_full():
    for mask in (np.zeros((8, 12), bool), np.ones((8, 12), bool)):
        assert np.array_equal(R._road_row_span_decode(R._road_row_span_encode(mask)), mask)


# ------------------------------- horizon poly + xi (real machinery) -----------
def _synthetic_horizon_frame(h=64, w=80, *, a=0.004, b=-0.2, c=20.0):
    """Undrivable above a parabolic horizon y(x)=a x^2 + b x + c, Road below (comma10k order)."""
    lab = np.full((h, w), UNDRIV, dtype=np.int64)
    xs = np.arange(w)
    ycut = np.clip(np.round(a * xs**2 + b * xs + c).astype(int), 1, h - 1)
    for x in range(w):
        lab[ycut[x]:, x] = ROAD
    return lab, ycut


def test_horizon_profile_recovers_boundary():
    lab, ycut = _synthetic_horizon_frame()
    ys = R._horizon_profile(lab, ROAD, UNDRIV)
    valid = ys >= 0
    # Every column has a Road-below/Undriv-above boundary at exactly ycut.
    assert valid.all()
    assert np.array_equal(ys[valid], ycut[valid])


def test_horizon_poly_xi_fits_synthetic_at_zero_residual():
    lab, _ = _synthetic_horizon_frame()
    stack = np.stack([lab] * 8)  # 8 identical frames -> delta stream is all-zero (max xi coherence)
    out = R.horizon_poly_xi_byte_cost(stack, road_cls=ROAD, undriv_cls=UNDRIV, degree=3, n_frames=600)
    # A degree-3 poly fits a parabola exactly.
    assert out["median_fit_residual_px"] < 0.5
    assert out["n_frames_fitted"] == 8
    # Honest scope flag is ALWAYS set (dominant-arc only; sidecar owed) — the NO-FAKE guard.
    assert out["residual_sidecar_owed"] is True
    assert "DOMINANT-ARC" in out["scope_note"]
    # Identical frames -> the delta encoding is the cheaper branch (near-free 599/600).
    assert out["delta_coeff_bytes"] <= out["raw_coeff_bytes"]
    assert out["score_rate_contribution_DERIVED_extrapolated"] >= 0.0


# ------------------------------- real n600 cache (gated) ----------------------
@pytest.mark.skipif(not _CACHE.exists(), reason="n600 gt cache absent (portable skip)")
def test_self_detects_comma10k_road_undriv_order():
    L = np.load(_CACHE)["lstars"][:32]
    roles = R.identify_road_undriv_classes(L)
    # Self-detected (NOT hardcoded) — must match the canonical comma10k Road0 / Undriv2.
    assert roles.road == ROAD
    assert roles.undriv == UNDRIV


@pytest.mark.skipif(not _CACHE.exists(), reason="n600 gt cache absent (portable skip)")
def test_real_horizon_fit_lands_in_measured_band():
    L = np.load(_CACHE)["lstars"][:150]
    out = R.horizon_poly_xi_byte_cost(L, road_cls=ROAD, undriv_cls=UNDRIV, degree=3, n_frames=600)
    # DAG FEED-v8-realmachinery: ~1.46 px residual, ~425/512 cols, ~0.003 S dominant-arc.
    assert 0.5 < out["median_fit_residual_px"] < 3.0
    assert out["mean_horizon_columns_covered"] > 300
    assert out["score_rate_contribution_DERIVED_extrapolated"] < 0.026  # below the generic chain-coder
    assert out["residual_sidecar_owed"] is True


@pytest.mark.skipif(not _CACHE.exists(), reason="n600 gt cache absent (portable skip)")
def test_real_byte_close_roundtrip_slice_bit_exact():
    # Audit row #6 owed the full-slice (not 1-frame) round-trip. Prove it on a real 32-frame slice.
    L = np.load(_CACHE)["lstars"][:32]
    for i in range(L.shape[0]):
        road = L[i] == ROAD
        assert np.array_equal(R._road_row_span_decode(R._road_row_span_encode(road)), road)


# ------------------------------- owed-9 lateral extents (F-P5-1 / §3 I1b) ------
def _synthetic_lateral_frame(h=48, w=100):
    """Road band [xL(y), xR(y)] widening with y; OUTSIDE the band is Undrivable (side)."""
    lab = np.full((h, w), UNDRIV, dtype=np.int64)
    for y in range(h):
        xl = 40 - y // 3
        xr = 60 + y // 3
        lab[y, xl:xr] = ROAD
    return lab, np.array([40 - y // 3 for y in range(h)]), np.array([60 + y // 3 - 1 for y in range(h)])


def test_lateral_extents_leftmost_rightmost():
    lab, xl_true, xr_true = _synthetic_lateral_frame()
    xl, xr = R._lateral_extents(lab, ROAD)
    assert np.array_equal(xl, xl_true.astype(np.int32))
    assert np.array_equal(xr, xr_true.astype(np.int32))


def test_lateral_extents_no_road_rows_are_minus_one():
    lab = np.full((10, 20), UNDRIV, dtype=np.int64)
    lab[4:6, 8:12] = ROAD  # only rows 4,5 have road
    xl, xr = R._lateral_extents(lab, ROAD)
    assert xl[0] == -1 and xr[0] == -1
    assert xl[4] == 8 and xr[4] == 11


def test_lateral_extents_multicomponent_collapses_to_hull():
    # Two road blobs in the same row -> leftmost/rightmost spans BOTH (lateral convex hull).
    lab = np.full((6, 30), UNDRIV, dtype=np.int64)
    lab[2, 4:8] = ROAD
    lab[2, 20:24] = ROAD
    xl, xr = R._lateral_extents(lab, ROAD)
    assert xl[2] == 4 and xr[2] == 23  # hull spans the median gap


def test_lateral_byte_cost_no_overflow_and_flags_sidecar():
    import warnings

    lab, _, _ = _synthetic_lateral_frame()
    stack = np.stack([lab, lab, lab, lab, lab])  # 5 identical frames
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any fp overflow now RAISES (the bug this fix closed)
        out = R.lateral_extent_poly_byte_cost(stack, road_cls=ROAD, degree=2, n_frames=600)
    assert out["n_frames_fitted"] == 5
    assert out["best_measured_bytes"] > 0
    assert out["residual_sidecar_owed"] is True
    # identical frames -> the delta stream is ~all-zero -> delta beats raw
    assert out["delta_coeff_bytes"] <= out["raw_coeff_bytes"]


@pytest.mark.skipif(not _CACHE.exists(), reason="n600 gt cache absent (portable skip)")
def test_real_lateral_byte_cost_lands_in_derived_I1b_range():
    # Recess R8: the SPEC_v8.1 §I I1b DERIVED carrier_total_S range is [0.0040, 0.0083].
    # The real-coder anchor here confirms the range; the ~20px fit residual is the honest tell (jagged
    # leftmost/rightmost envelope does NOT fit a smooth low-order poly -> the analytic form hurts).
    L = np.asarray(np.load(_CACHE)["lstars"])
    out = R.lateral_extent_poly_byte_cost(L, road_cls=ROAD, degree=2, n_frames=600)
    assert 0.003 < out["score_rate_contribution_DERIVED_extrapolated"] < 0.010
    assert out["median_fit_residual_px"] > 5.0  # the poor-fit tell (NOT the smooth horizon's ~1.5px)
    assert out["n_frames_fitted"] == 600
