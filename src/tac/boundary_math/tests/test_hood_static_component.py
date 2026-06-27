"""Tests for the static ego-hood level-set component (FEED-du).

NO-FAKE: synthetic-but-faithful fixtures (a known static bottom region + a slowly-moving
top edge) exercise the REAL math the functions name — majority-vote aggregation, scipy-EDT
signed distance, argmax injection, disagreement decomposition, byte cost. The decisive
real-data verdict lives in the measure script + memo; these guard the primitives.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.hood_static_component import (
    StaticHoodMask,
    build_static_hood_sdf,
    compute_static_hood_mask,
    decompose_argmax_disagreement,
    hood_mask_byte_cost,
    identify_static_hood_class,
    inject_hood_sdf,
)
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields


def _synthetic_lstars(n=20, h=64, w=80, hood_cls=4, jitter=2, seed=0):
    """n frames: bottom rows = hood (class hood_cls) with a top edge that jitters +-jitter;
    a static top region (class 2); road (class 0) in the middle; a tiny moving 'movable'
    blob (class 3). Returns (n,h,w) int64."""

    rng = np.random.default_rng(seed)
    L = np.zeros((n, h, w), np.int64)
    base_top = int(h * 0.70)  # hood starts ~row 45/64
    for i in range(n):
        L[i, :, :] = 0  # road
        L[i, : int(h * 0.30), :] = 2  # static top region
        edge = base_top + int(rng.integers(-jitter, jitter + 1))
        L[i, edge:, :] = hood_cls  # hood (bottom)
        # a small moving blob (class 3) in the road region
        r = int(rng.integers(int(h * 0.35), int(h * 0.55)))
        c = int(rng.integers(5, w - 10))
        L[i, r : r + 3, c : c + 5] = 3
    return L


# --------------------------- identify_static_hood_class ---------------------------
def test_identify_picks_bottom_static_class():
    L = _synthetic_lstars(hood_cls=4)
    cls, ev = identify_static_hood_class(L, n_classes=5)
    assert cls == 4
    assert len(ev) == 5


def test_identify_returns_evidence_for_each_class():
    L = _synthetic_lstars(hood_cls=4)
    _, ev = identify_static_hood_class(L, n_classes=5)
    hood_ev = next(e for e in ev if e.cls == 4)
    assert hood_ev.bottom_share > 0.7  # hood concentrated at the bottom (fixture spans rows 44-63)
    assert hood_ev.static_iou > 0.5    # near-static


def test_identify_not_fooled_by_large_top_static_region():
    # class 2 is large + static but at the TOP -> must NOT be picked as the hood.
    L = _synthetic_lstars(hood_cls=4)
    cls, _ = identify_static_hood_class(L, n_classes=5)
    assert cls != 2


def test_identify_hood_at_different_index():
    L = _synthetic_lstars(hood_cls=1)
    cls, _ = identify_static_hood_class(L, n_classes=5)
    assert cls == 1  # detection is data-driven, not hardcoded to 4


# --------------------------- compute_static_hood_mask ---------------------------
def test_majority_mask_shape_and_dtype():
    L = _synthetic_lstars(hood_cls=4)
    sm = compute_static_hood_mask(L, hood_cls=4, agg="majority")
    assert isinstance(sm, StaticHoodMask)
    assert sm.mask.shape == L.shape[1:]
    assert sm.mask.dtype == bool
    assert sm.hood_cls == 4


def test_intersection_subset_of_majority_subset_of_union():
    L = _synthetic_lstars(hood_cls=4)
    inter = compute_static_hood_mask(L, hood_cls=4, agg="intersection").mask
    maj = compute_static_hood_mask(L, hood_cls=4, agg="majority").mask
    uni = compute_static_hood_mask(L, hood_cls=4, agg="union").mask
    assert (inter & ~maj).sum() == 0          # inter ⊆ maj
    assert (maj & ~uni).sum() == 0            # maj ⊆ uni
    assert inter.sum() <= maj.sum() <= uni.sum()


def test_static_iou_high_for_static_hood():
    L = _synthetic_lstars(hood_cls=4, jitter=2)
    sm = compute_static_hood_mask(L, hood_cls=4, agg="majority")
    assert sm.mean_frame_iou > 0.85  # small jitter -> high per-frame IoU


def test_invalid_agg_raises():
    L = _synthetic_lstars(hood_cls=4)
    with pytest.raises(ValueError):
        compute_static_hood_mask(L, hood_cls=4, agg="median")


def test_zero_jitter_mask_is_exact_and_iou_one():
    L = _synthetic_lstars(hood_cls=4, jitter=0)
    sm = compute_static_hood_mask(L, hood_cls=4, agg="majority")
    assert sm.mean_frame_iou == pytest.approx(1.0)
    assert sm.min_frame_iou == pytest.approx(1.0)


# --------------------------- build_static_hood_sdf ---------------------------
def test_sdf_sign_inside_outside():
    L = _synthetic_lstars(hood_cls=4)
    mask = compute_static_hood_mask(L, hood_cls=4, agg="majority").mask
    phi = build_static_hood_sdf(mask)
    assert phi.dtype == np.float32
    assert (phi[mask] > 0).all()      # +EDT inside
    assert (phi[~mask] < 0).all()     # -EDT outside


def test_sdf_one_lipschitz_ish():
    L = _synthetic_lstars(hood_cls=4)
    mask = compute_static_hood_mask(L, hood_cls=4, agg="majority").mask
    phi = build_static_hood_sdf(mask)
    gy, gx = np.gradient(phi.astype(np.float64), axis=0), np.gradient(phi.astype(np.float64), axis=1)
    gmag = np.sqrt(gx * gx + gy * gy)
    # EDT signed distance is ~1-Lipschitz; allow slack at the zero level set
    assert np.percentile(gmag, 90) < 1.6


# --------------------------- inject + decompose (isolation test) ---------------------------
def test_inject_static_hood_into_ideal_is_precise_and_contained():
    L = _synthetic_lstars(hood_cls=4, jitter=2, seed=3)
    mask = compute_static_hood_mask(L, hood_cls=4, agg="majority").mask
    phi_hood = build_static_hood_sdf(mask)
    totals, leaks = [], []
    for i in range(L.shape[0]):
        Li = L[i]
        phi_ideal = signed_distance_fields(Li, 5)
        pred = inject_hood_sdf(phi_ideal, phi_hood, lane_cls=4, mode="replace").argmax(-1)
        d = decompose_argmax_disagreement(pred, Li, lane_cls=4, road_cls=0)
        totals.append(d.total_dseg)
        leaks.append(d.lane_fp_from_road + d.lane_fp_from_other)
    # static mask approximates the jittering hood -> small but nonzero total; low containment leak
    assert 0.0 < float(np.mean(totals)) < 0.05
    assert float(np.mean(leaks)) < float(np.mean(totals))  # leak is the minority term


def test_ideal_injection_is_zero_baseline():
    L = _synthetic_lstars(hood_cls=4)
    Li = L[0]
    phi_ideal = signed_distance_fields(Li, 5)
    d = decompose_argmax_disagreement(phi_ideal.argmax(-1), Li, lane_cls=4, road_cls=0)
    assert d.total_dseg == pytest.approx(0.0)


def test_intersection_more_contained_than_majority():
    L = _synthetic_lstars(hood_cls=4, jitter=3, seed=5)
    maj = build_static_hood_sdf(compute_static_hood_mask(L, hood_cls=4, agg="majority").mask)
    inter = build_static_hood_sdf(compute_static_hood_mask(L, hood_cls=4, agg="intersection").mask)
    leak_maj, leak_int = [], []
    for i in range(L.shape[0]):
        Li = L[i]
        phi_ideal = signed_distance_fields(Li, 5)
        dm = decompose_argmax_disagreement(
            inject_hood_sdf(phi_ideal, maj, lane_cls=4, mode="replace").argmax(-1), Li,
            lane_cls=4, road_cls=0)
        di = decompose_argmax_disagreement(
            inject_hood_sdf(phi_ideal, inter, lane_cls=4, mode="replace").argmax(-1), Li,
            lane_cls=4, road_cls=0)
        leak_maj.append(dm.lane_fp_from_road + dm.lane_fp_from_other)
        leak_int.append(di.lane_fp_from_road + di.lane_fp_from_other)
    assert float(np.mean(leak_int)) <= float(np.mean(leak_maj)) + 1e-9


# --------------------------- byte cost ---------------------------
def test_byte_cost_is_tiny_and_amortizes():
    L = _synthetic_lstars(hood_cls=4)
    mask = compute_static_hood_mask(L, hood_cls=4, agg="majority").mask
    bc = hood_mask_byte_cost(mask, n_frames=600)
    assert bc["best_counted_bytes"] > 0
    assert bc["best_counted_bytes"] <= bc["raw_packed_bits_bytes"]
    assert bc["amortized_bytes_per_frame"] == pytest.approx(bc["best_counted_bytes"] / 600.0)
    assert bc["score_rate_contribution"] < 0.001  # negligible in score units


def test_byte_cost_empty_mask():
    mask = np.zeros((64, 80), bool)
    bc = hood_mask_byte_cost(mask, n_frames=600)
    assert bc["row_span_bytes"] == 0
    assert bc["best_counted_bytes"] >= 0


def test_inject_does_not_mutate_input():
    L = _synthetic_lstars(hood_cls=4)
    Li = L[0]
    phi_ideal = signed_distance_fields(Li, 5)
    before = phi_ideal.copy()
    mask = compute_static_hood_mask(L, hood_cls=4, agg="majority").mask
    _ = inject_hood_sdf(phi_ideal, build_static_hood_sdf(mask), lane_cls=4, mode="replace")
    assert np.array_equal(phi_ideal, before)  # injection returns a copy
