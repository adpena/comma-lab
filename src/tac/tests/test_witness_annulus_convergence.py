# SPDX-License-Identifier: MIT
"""Focused unit tests for tac.witness_annulus_metrics (annulus convergence telemetry).

Pure metric math on tiny synthetic argmax+margin fixtures. No rendering, no torch, no MLX.
Verifies: annulus/interior decomposition, band vs bottom-k definitions, flip-restriction
correctness, per-class split, margin percentiles, Gibbs proxy behaviour, and the sign of the
cross-checkpoint convergence rate.
"""
from __future__ import annotations

import math

import numpy as np

from tac.witness_annulus_metrics import (
    N_CLASSES,
    annulus_mask_bottom_k,
    annulus_mask_threshold,
    checkpoint_metrics,
    convergence_rate,
    convergence_series,
    flip_map,
    flip_mass_share,
    gibbs_ring_proxy,
    margin_percentiles,
    per_class_annulus_flip_frac,
    restricted_flip_frac,
)


def test_annulus_mask_threshold_selects_low_margin():
    gm = np.array([[0.1, 0.5, 3.0, 9.0]], np.float32)
    m = annulus_mask_threshold(gm, band=1.0)
    assert m.tolist() == [[True, True, False, False]]


def test_annulus_mask_threshold_abs_value():
    # top1-top2 is >=0, but be robust: |margin| < band uses absolute value.
    gm = np.array([-0.2, 0.2, -5.0], np.float32)
    m = annulus_mask_threshold(gm, band=1.0)
    assert m.tolist() == [True, True, False]


def test_annulus_mask_bottom_k_fraction_and_area():
    gm = np.arange(100, dtype=np.float32)  # 0..99
    m = annulus_mask_bottom_k(gm, k=0.05)
    # quantile(0.05) of 0..99 == 4.95 -> pixels <=4.95 -> {0,1,2,3,4} = 5 pixels.
    assert int(m.sum()) == 5
    assert m[:5].all() and not m[5:].any()


def test_annulus_mask_bottom_k_rejects_bad_k():
    gm = np.arange(10, dtype=np.float32)
    for bad in (0.0, -0.1, 1.5):
        try:
            annulus_mask_bottom_k(gm, k=bad)
        except ValueError:
            continue
        raise AssertionError(f"bottom-k={bad} should have raised")


def test_flip_map_and_restricted_frac():
    w = np.array([[0, 1, 2, 3]], np.int64)
    g = np.array([[0, 0, 2, 0]], np.int64)  # flips at idx 1 and 3
    fl = flip_map(w, g)
    assert fl.tolist() == [[False, True, False, True]]
    mask = np.array([[True, True, False, False]])  # covers idx 0,1
    # one flip (idx1) among two masked pixels -> 0.5
    assert restricted_flip_frac(fl, mask) == 0.5


def test_restricted_flip_frac_empty_mask_is_zero():
    fl = np.array([True, False])
    assert restricted_flip_frac(fl, np.array([False, False])) == 0.0


def test_flip_mass_share():
    fl = np.array([True, True, False, True])  # 3 flips total
    mask = np.array([True, False, False, True])  # 2 flips inside (idx0, idx3)
    assert flip_mass_share(fl, mask) == 2 / 3
    assert flip_mass_share(np.array([False, False]), np.array([True, True])) == 0.0


def test_interior_is_complement_of_annulus():
    # 3x3, one low-margin center pixel = annulus; the other 8 = interior.
    gm = np.full((3, 3), 5.0, np.float32)
    gm[1, 1] = 0.1
    an = annulus_mask_threshold(gm, band=1.0)
    interior = ~an
    assert int(an.sum()) == 1 and int(interior.sum()) == 8
    # a flip on the interior must NOT count toward annulus_flip_frac.
    w = np.zeros((3, 3), np.int64)
    g = np.zeros((3, 3), np.int64)
    g[0, 0] = 1  # interior flip
    fl = flip_map(w, g)
    assert restricted_flip_frac(fl, an) == 0.0
    assert restricted_flip_frac(fl, interior) == 1 / 8


def test_per_class_annulus_flip_frac_split():
    # 1x4 row: gt classes [0,0,1,1]; all in annulus; flips on one class-0 and one class-1 px.
    g = np.array([[0, 0, 1, 1]], np.int64)
    w = np.array([[0, 2, 1, 3]], np.int64)  # flip at idx1 (class0) and idx3 (class1)
    fl = flip_map(w, g)
    an = np.ones((1, 4), bool)
    pcf = per_class_annulus_flip_frac(fl, an, g, N_CLASSES)
    assert pcf[0] == 0.5  # 1 flip of 2 class-0 px
    assert pcf[1] == 0.5  # 1 flip of 2 class-1 px
    assert pcf[2] == 0.0 and pcf[3] == 0.0 and pcf[4] == 0.0  # no pixels of those classes


def test_margin_percentiles_and_empty():
    wm = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], np.float32)
    mask = np.ones((1, 5), bool)
    pc = margin_percentiles(wm, mask, ps=(10.0, 50.0))
    assert pc["p50"] == 2.0
    empty = margin_percentiles(wm, np.zeros((1, 5), bool))
    assert math.isnan(empty["p10"]) and math.isnan(empty["p50"])


def test_gibbs_ring_proxy_flat_field_low():
    # a flat margin field has zero Laplacian everywhere -> ring energy == interior == 0 -> 0.0
    wm = np.full((6, 6), 2.0, np.float32)
    an = np.zeros((6, 6), bool)
    an[3, 3] = True
    assert gibbs_ring_proxy(wm, an) == 0.0


def test_gibbs_ring_proxy_oscillation_on_ring_high():
    # put a high-frequency checkerboard ONLY on the annulus ring; interior smooth ->
    # ratio should be large (>1). Build a 8x8: annulus = a middle ring, interior elsewhere.
    h = w = 8
    wm = np.zeros((h, w), np.float32)
    an = np.zeros((h, w), bool)
    an[3:5, :] = True  # a horizontal band as the "ring"
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    checker = ((ii + jj) % 2).astype(np.float32) * 10.0
    wm[an] = checker[an]  # oscillation only on the ring; interior stays flat 0
    ratio = gibbs_ring_proxy(wm, an)
    assert ratio > 1.0


def test_checkpoint_metrics_bundle_keys_and_values():
    rng = np.random.default_rng(0)
    V, h, w = 2, 6, 6
    gt = rng.integers(0, N_CLASSES, size=(V, h, w)).astype(np.int64)
    gm = rng.uniform(0.0, 6.0, size=(V, h, w)).astype(np.float32)
    wa = gt.copy()
    wa[0, 0, 0] = (gt[0, 0, 0] + 1) % N_CLASSES  # inject one flip
    wm = rng.uniform(-1.0, 3.0, size=(V, h, w)).astype(np.float32)
    m = checkpoint_metrics(wa, wm, gt, gm, band=2.0, bottom_k=0.25)
    assert m["n_flips"] == 1
    assert 0.0 <= m["threshold"]["annulus_area_frac"] <= 1.0
    # bottom_k=0.25 selects ~25% of pixels as annulus.
    assert m["bottom_k_def"]["annulus_area_frac"] > 0.0
    for defn in ("threshold", "bottom_k_def"):
        assert set(m[defn]["per_class_annulus_flip_frac"].keys()) == set(range(N_CLASSES))


def test_convergence_rate_sign():
    # falling flip fraction across epochs -> negative slope (converging).
    epochs = [100.0, 200.0, 300.0, 400.0]
    falling = [0.04, 0.03, 0.02, 0.01]
    assert convergence_rate(epochs, falling) < 0.0
    rising = [0.01, 0.02, 0.03, 0.04]
    assert convergence_rate(epochs, rising) > 0.0
    # degenerate cases -> 0.0
    assert convergence_rate([1.0], [0.5]) == 0.0
    assert convergence_rate([5.0, 5.0], [0.1, 0.2]) == 0.0


def test_convergence_rate_drops_nan():
    epochs = [1.0, 2.0, 3.0]
    vals = [0.03, float("nan"), 0.01]
    # only two finite points (0.03@1, 0.01@3) -> slope negative.
    assert convergence_rate(epochs, vals) < 0.0


def test_convergence_series_end_to_end():
    # two synthetic checkpoints; annulus flips fall from ckpt A->B => negative rate.
    rng = np.random.default_rng(1)
    V, h, w = 1, 8, 8
    gt = rng.integers(0, N_CLASSES, size=(V, h, w)).astype(np.int64)
    gm = rng.uniform(0.0, 6.0, size=(V, h, w)).astype(np.float32)
    an = annulus_mask_threshold(gm, band=2.0)
    wm = rng.uniform(-1.0, 3.0, size=(V, h, w)).astype(np.float32)

    # ckpt A: several annulus flips. ckpt B: fewer.
    waA = gt.copy()
    idx = np.argwhere(an[0])
    for r, c in idx[:5]:
        waA[0, r, c] = (gt[0, r, c] + 1) % N_CLASSES
    waB = gt.copy()
    for r, c in idx[:1]:
        waB[0, r, c] = (gt[0, r, c] + 1) % N_CLASSES

    mA = checkpoint_metrics(waA, wm, gt, gm, band=2.0)
    mB = checkpoint_metrics(waB, wm, gt, gm, band=2.0)
    series = [
        {"name": "A", "epoch": 100.0, "metrics": mA},
        {"name": "B", "epoch": 200.0, "metrics": mB},
    ]
    cs = convergence_series(series, defn="threshold")
    assert cs["epochs"] == [100.0, 200.0]
    assert cs["annulus_flip_frac_rate"] < 0.0  # converging
    assert len(cs["annulus_flip_frac_curve"]) == 2
    assert set(cs["per_class_annulus_flip_frac_rate"].keys()) == set(range(N_CLASSES))


def test_convergence_series_rejects_bad_defn():
    try:
        convergence_series([], defn="nope")
    except ValueError:
        return
    raise AssertionError("bad defn should raise")


def test_convergence_series_handles_string_class_keys():
    # per_class keys may arrive as strings after a JSON round-trip; series must cope.
    rng = np.random.default_rng(2)
    V, h, w = 1, 6, 6
    gt = rng.integers(0, N_CLASSES, size=(V, h, w)).astype(np.int64)
    gm = rng.uniform(0.0, 6.0, size=(V, h, w)).astype(np.float32)
    wm = rng.uniform(-1.0, 3.0, size=(V, h, w)).astype(np.float32)
    m = checkpoint_metrics(gt.copy(), wm, gt, gm)
    # simulate JSON round-trip: stringify per-class keys.
    for defn in ("threshold", "bottom_k_def"):
        pcf = m[defn]["per_class_annulus_flip_frac"]
        m[defn]["per_class_annulus_flip_frac"] = {str(k): v for k, v in pcf.items()}
    cs = convergence_series([{"name": "A", "epoch": 1.0, "metrics": m}], defn="threshold")
    assert set(cs["per_class_annulus_flip_frac_rate"].keys()) == set(range(N_CLASSES))
