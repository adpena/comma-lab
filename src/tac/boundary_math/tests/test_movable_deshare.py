# SPDX-License-Identifier: MIT
"""Tests for Movable-first de-share + the general pairwise archive-dedup audit (v8 T2 Lever-1).

Unit-tests the deterministic primitives (separatrix, footprint, partition guarantee) synthetically;
gates the full n600 measurement + role self-detection to the real cache.  ``[macOS-CPU advisory ·
NON-PROMOTABLE]`` -- a dedup audit moves no pointer.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from tac.boundary_math import movable_deshare as M

_CACHE = pathlib.Path(__file__).resolve().parents[4] / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
H, W = 32, 40


def test_separatrix_mask_symmetric():
    a = np.zeros((H, W), dtype=np.int64)
    a[:, 20:] = 1  # left=0, right=1 -> vertical seam at col 19|20
    sep = M.separatrix_mask(a, 0, 1)
    # both sides of the seam are separatrix pixels
    assert sep[:, 19].all() and sep[:, 20].all()
    assert not sep[:, 5].any() and not sep[:, 35].any()
    # order-invariant
    assert np.array_equal(sep, M.separatrix_mask(a, 1, 0))


def test_dilate_and_footprint():
    a = np.zeros((H, W), dtype=np.int64)
    a[15, 20] = 3  # single movable px
    fp = M.movable_footprint(a, 3, dilate=1)
    assert fp[15, 20] and fp[14, 20] and fp[15, 21] and fp[14, 19]
    assert not fp[12, 20]
    fp2 = M.movable_footprint(a, 3, dilate=2)
    assert fp2[13, 20]  # reaches 2 px out


def test_deshare_partition_is_a_partition():
    # residual = 6 pixels; footprint covers 2 of them
    resid = np.array([10, 25, 40, 55, 70, 85], dtype=np.int64)
    fp = np.zeros(H * W, dtype=bool)
    fp[[25, 70]] = True
    kept, res = M.deshare_partition(resid, fp, "edge")
    assert res.n_residual == 6
    assert res.n_attributed_movable == 2
    assert res.n_kept == 4
    # PARTITION: kept + attributed = input, disjoint
    attributed = np.setdiff1d(resid, kept)
    assert np.array_equal(np.union1d(kept, attributed), resid)
    assert np.intersect1d(kept, attributed).size == 0
    assert set(attributed.tolist()) == {25, 70}
    assert abs(res.frac_attributed - 2 / 6) < 1e-9


def test_deshare_partition_empty():
    kept, res = M.deshare_partition(np.zeros(0, np.int64), np.zeros(H * W, bool), "e")
    assert res.n_residual == 0 and res.n_kept == 0 and res.frac_attributed == 0.0


def test_deshare_determinism():
    resid = np.array([3, 3, 8, 100, 100, 55], dtype=np.int64)  # dups collapse
    fp = np.zeros(H * W, dtype=bool)
    fp[[8, 55]] = True
    k1, r1 = M.deshare_partition(resid, fp, "e")
    k2, r2 = M.deshare_partition(resid, fp, "e")
    assert np.array_equal(k1, k2)
    assert r1 == r2  # dataclass equality
    assert r1.n_residual == 4  # unique {3,8,55,100}


# --------------------------------------------------------------------------- real-cache (gated)
@pytest.mark.skipif(not _CACHE.exists(), reason="gt_n600 cache absent")
def test_roles_self_detect_real_cache():
    lst = np.load(_CACHE)["lstars"][:8]
    roles = M.detect_seg_roles(lst)
    # comma10k canonical Road0/Lane1/Undriv2/Movable3/MyCar4 (self-detected, not hardcoded)
    assert roles.road == 0
    assert roles.undriv == 2
    assert roles.mycar == 4
    assert roles.lane == 1
    assert roles.movable == 3
    assert len(set(roles.as_dict().values())) == 5


@pytest.mark.skipif(not _CACHE.exists(), reason="gt_n600 cache absent")
def test_measure_deshare_smoke_real_cache():
    lst = np.load(_CACHE)["lstars"][:6]
    out = M.measure_deshare_magnitude(lst, seed=0)
    assert set(out["edges"]) == {"horizon", "lane"}
    for e in out["edges"].values():
        # PRIMARY magnitude (amortized) is monotone: 0 <= double-counted <= full residual bytes
        assert 0 <= e["bytes_double_counted"] <= e["bytes_before_deshare_coded"]
        assert 0 <= e["attributed_movable_px"] <= e["residual_px_total"]
    # deterministic
    out2 = M.measure_deshare_magnitude(lst, seed=0)
    assert out2["total_bytes_double_counted"] == out["total_bytes_double_counted"]


@pytest.mark.skipif(not _CACHE.exists(), reason="gt_n600 cache absent")
def test_pairwise_dedup_audit_smoke_real_cache():
    lst = np.load(_CACHE)["lstars"][:4]
    out = M.pairwise_dedup_audit(lst)
    assert len(out["pairs"]) == 10  # C(5,2)
    for p in out["pairs"]:
        assert p["shared_px_total"] >= 0
        assert p["overlap_bytes_double_counted"] >= 0
