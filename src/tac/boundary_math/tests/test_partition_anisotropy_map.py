"""Tests for the partition anisotropy map — including a synthetic KNOWN-anisotropy control."""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.partition_anisotropy_map import (
    LANE,
    MYCAR,
    ROAD,
    UNDRIV,
    _accumulate_edges,
    compute_anisotropy_map,
    detect_class_order,
    fit_horizon_line,
    hessian_eigs,
    structure_tensor_dH,
)


# --------------------------------------------------------------------------- #
# synthetic control: a KNOWN-anisotropic straight edge -> high d_H;
# an isotropic radial bump -> low d_H.
# --------------------------------------------------------------------------- #
def test_synthetic_straight_edge_is_highly_anisotropic():
    H, W = 128, 128
    yy, _ = np.mgrid[0:H, 0:W]
    # a vertical-gradient step edge (horizontal boundary): margin dips linearly across row 64
    field = np.abs(yy - 64.0).astype(np.float64)  # V-shaped, gradient purely in y away from edge
    st = structure_tensor_dH(field, sigma=3.0)
    # along the edge band (near row 64, away from the exact crease), d_H should be large:
    band = st["dH"][40:60, 20:108]  # off-crease region, gradient is uniform in y
    assert band.mean() > 3.0, f"straight edge should be anisotropic, got {band.mean()}"


def test_synthetic_isotropic_bump_is_low_anisotropy():
    H, W = 128, 128
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (yy - 64.0) ** 2 + (xx - 64.0) ** 2
    field = np.exp(-r2 / (2 * 15.0**2))  # radially symmetric Gaussian bump
    st = structure_tensor_dH(field, sigma=3.0)
    # at the exact center the gradient vanishes and the structure tensor is isotropic;
    # ring around center: gradient is radial but rotating -> the LOCAL structure tensor over a
    # window that spans multiple radial directions is less coherent than a straight edge.
    center = st["dH"][60:68, 60:68]
    # near the peak the structure tensor is near-isotropic (both eigenvalues tiny/comparable)
    assert center.mean() < 3.0, f"isotropic bump center should be low-d_H, got {center.mean()}"


def test_recovered_dH_matches_known_ratio():
    # anisotropic Gaussian field with a controlled eigenvalue ratio in its structure tensor:
    # a plane wave along x -> gradient purely in x -> lam_min ~ floor -> large d_H.
    H, W = 96, 96
    _, xx = np.mgrid[0:H, 0:W]
    field = np.sin(2 * np.pi * xx / 24.0)
    st = structure_tensor_dH(field, sigma=4.0)
    interior = st["dH"][20:76, 20:76]
    # a pure 1-D plane wave has a rank-1 structure tensor -> very high d_H
    assert interior.mean() > 4.0


# --------------------------------------------------------------------------- #
# Hessian saddle: a hyperbolic paraboloid has mixed-sign Hessian everywhere.
# --------------------------------------------------------------------------- #
def test_hyperbolic_paraboloid_is_saddle():
    H, W = 64, 64
    yy, xx = np.mgrid[0:H, 0:W]
    field = (xx - 32.0) ** 2 - (yy - 32.0) ** 2  # z = x^2 - y^2 (saddle)
    he = hessian_eigs(field, sigma=2.0)
    interior = he["is_saddle"][10:54, 10:54]
    assert interior.mean() > 0.9, "hyperbolic paraboloid should be a saddle everywhere"


def test_convex_bowl_is_not_saddle():
    H, W = 64, 64
    yy, xx = np.mgrid[0:H, 0:W]
    field = (xx - 32.0) ** 2 + (yy - 32.0) ** 2  # convex bowl (both Hessian eigs > 0)
    he = hessian_eigs(field, sigma=2.0)
    interior = he["is_saddle"][10:54, 10:54]
    assert interior.mean() < 0.05


# --------------------------------------------------------------------------- #
# edge accumulation on a hand-built label map
# --------------------------------------------------------------------------- #
def test_edge_accumulation_counts_class_pairs():
    L = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [4, 4, 4, 4]], dtype=np.int64)
    dH = np.ones_like(L, dtype=np.float64) * 2.0
    energy = np.ones_like(L, dtype=np.float64)
    acc: dict = {}
    total = _accumulate_edges(L, dH, energy, acc)
    assert total > 0
    # Road(0)-Lane(1) vertical cracks + Road/Lane to MyCar(4) cracks present
    assert (ROAD, LANE) in acc
    assert (ROAD, MYCAR) in acc
    # all d_H were 2.0 -> mean must be 2.0
    assert acc[(ROAD, LANE)].dH_mean == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# class-order self-detection is fail-closed
# --------------------------------------------------------------------------- #
def _canonical_synthetic_labels(T=3, H=48, W=48):
    # Signature-valid mimic of the real cache: Undriv = top + LARGEST, Road = mid,
    # MyCar = bottom + static (smaller than Undriv), Lane = thin + MOVING (unstable IoU).
    L = np.full((T, H, W), UNDRIV, dtype=np.int64)  # top half = sky/undriv (largest)
    L[:, H // 2 : 5 * H // 6, :] = ROAD  # mid = road
    L[:, 5 * H // 6 :, :] = MYCAR  # bottom sixth = hood (static)
    for t in range(T):
        col = (t * 5 + 3) % W  # lane column shifts every frame -> low temporal IoU
        L[t, H // 2 : 5 * H // 6, col] = LANE
    return L


def test_detect_class_order_passes_on_canonical():
    L = _canonical_synthetic_labels()
    res = detect_class_order(L)
    assert all(res["checks"].values())


def test_detect_class_order_fails_on_scrambled():
    L = _canonical_synthetic_labels()
    scrambled = L.copy()
    # swap MyCar(bottom) <-> Undriv(top) indices -> signature must break
    scrambled[L == MYCAR] = UNDRIV
    scrambled[L == UNDRIV] = MYCAR
    with pytest.raises(ValueError):
        detect_class_order(scrambled)


# --------------------------------------------------------------------------- #
# horizon line fit on a synthetic near-horizontal Road/Undriv boundary
# --------------------------------------------------------------------------- #
def test_fit_horizon_recovers_line():
    H, W = 100, 120
    L = np.full((H, W), ROAD, dtype=np.int64)
    # undrivable above a line v = 0.05*u + 30
    us = np.arange(W)
    for u in us:
        row = int(0.05 * u + 30)
        L[:row, u] = UNDRIV
    h = fit_horizon_line(L)
    assert h["fit_ok"] == 1.0
    assert h["slope"] == pytest.approx(0.05, abs=0.02)
    assert h["residual_rows_rms"] < 2.0
    assert h["v_at_center_row"] == pytest.approx(0.05 * (W / 2) + 30, abs=2.0)


# --------------------------------------------------------------------------- #
# end-to-end on a tiny synthetic volume
# --------------------------------------------------------------------------- #
def test_compute_anisotropy_map_end_to_end():
    L = _canonical_synthetic_labels(T=4, H=64, W=64)
    rng = np.random.default_rng(0)
    # margin: high in interiors, dips near boundaries
    M = np.full(L.shape, 5.0, dtype=np.float64)
    M[:, 20:24, :] = 0.2  # near a horizontal boundary band
    M += rng.normal(0, 0.01, M.shape)
    amap = compute_anisotropy_map(L, M, sigma_st=2.0, sigma_hess=2.0)
    assert amap.class_order["order"].startswith("0=Road")
    assert len(amap.edges) >= 1
    # every edge dH is finite and non-negative (structure tensor is SPD)
    for e in amap.edges:
        assert e["dH_mean"] >= 0.0
        assert np.isfinite(e["dH_mean"])
    js = amap.to_json()
    assert "edges" in js and "saddles" in js and "temporal" in js
    # saddle eigenstructure fields present + the rank-1/rank-2 split sums to ~1
    es = js["saddles"]["eigenstructure"]
    assert "n_junctions_sampled" in es
    if es["n_junctions_sampled"]:
        assert es["frac_directionally_codeable"] + es["frac_genuine_2d_hyperbolic"] == pytest.approx(
            1.0, abs=1e-6
        )


def test_saddle_eigenstructure_full_on_synthetic_junction():
    # a hyperbolic margin dip z = -|x*y| centered at a junction -> mixed-sign Hessian; the
    # eigenstructure primitive must flag it as a saddle (rank-2 hyperbolic).
    H, W = 64, 64
    _, xx = np.mgrid[0:H, 0:W]
    yy = np.mgrid[0:H, 0:W][0]
    field = -((xx - 32.0) * (yy - 32.0)) / 100.0  # true hyperbolic saddle
    he = hessian_eigs(field, sigma=2.0)
    interior = he["is_saddle"][12:52, 12:52]
    assert interior.mean() > 0.9, "hyperbolic junction must be rank-2 mixed-sign everywhere"
