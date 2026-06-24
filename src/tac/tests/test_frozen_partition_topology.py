"""NO-FAKE tests for the frozen-instance partition topology probe.

These tests verify BEHAVIOR (the topology signature, boundary-dim, and
explained-fraction helpers actually compute the claimed quantities on real
partition data), not constants. They run on tiny synthetic partitions with
KNOWN topology so the invariants are hand-checkable, plus a guard that the
cached authority argmaps exist and are 5-class.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "experiments/probe_frozen_partition_topology.py"

_spec = importlib.util.spec_from_file_location("_probe_topo", PROBE)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_topology_signature_counts_components_of_known_partition():
    # 4x4: top 2 rows class 0, bottom 2 rows class 1 -> 2 components, adj {(0,1)}
    lab = np.zeros((4, 4), dtype=np.uint8)
    lab[2:, :] = 1
    adj, comps, euler = probe.topology_signature(lab)
    assert adj == frozenset({(0, 1)})
    assert comps[0] == 1 and comps[1] == 1  # one component each
    # two simply-connected regions, no holes -> euler = 2
    assert abs(euler - 2.0) < 1e-6


def test_topology_signature_detects_extra_island():
    # one big class-0 field with a single class-2 island -> class2 has 1 comp,
    # adjacency includes (0,2); adding a 2nd island raises class-2 count to 2.
    lab = np.zeros((10, 10), dtype=np.uint8)
    lab[1:3, 1:3] = 2
    _, comps1, _ = probe.topology_signature(lab)
    lab[6:8, 6:8] = 2
    _, comps2, _ = probe.topology_signature(lab)
    assert comps2[2] == comps1[2] + 1  # second island => +1 component
    assert comps1[2] == 1


def test_topology_signature_not_a_constant_stub():
    # different partitions MUST yield different signatures (no hard-coded return)
    a = np.zeros((8, 8), dtype=np.uint8)
    b = a.copy()
    b[0, 0] = 3  # introduce a class-3 pixel
    sa = probe.topology_signature(a)
    sb = probe.topology_signature(b)
    assert sa != sb


def test_boundary_curve_repr_tracks_horizon_row():
    # horizon at row 5: top class 0, bottom class 4 (road). road_top should be 5.
    lab = np.zeros((10, 12), dtype=np.uint8)
    lab[5:, :] = 4
    cv, road_cls = probe.boundary_curve_repr(lab)
    assert road_cls == 4
    W = 12
    road_top = cv[W:]
    assert np.allclose(road_top, 5.0)
    # raise the horizon to row 3 -> road_top moves to 3 (deformation tracked)
    lab2 = np.zeros((10, 12), dtype=np.uint8)
    lab2[3:, :] = 4
    cv2, _ = probe.boundary_curve_repr(lab2)
    assert np.allclose(cv2[W:], 3.0)
    assert not np.allclose(cv[W:], cv2[W:])  # deformation is real, not constant


def test_explained_fraction_perfect_and_null():
    rng = np.random.default_rng(0)
    ego = rng.standard_normal((50, 2))
    # deform = linear function of ego => R^2 ~ 1
    W = np.array([[2.0, -1.0, 0.5], [0.3, 0.0, 1.0]])
    deform = ego @ W
    deform = deform - deform.mean(0)
    r = probe.explained_fraction(deform, ego)
    assert r["r2_overall"] > 0.99
    # deform independent of ego => R^2 ~ 0 (small)
    deform2 = rng.standard_normal((50, 3))
    deform2 = deform2 - deform2.mean(0)
    r2 = probe.explained_fraction(deform2, ego)
    assert r2["r2_overall"] < 0.3


def test_stage2_boundary_dim_low_rank_for_smooth_deformation():
    # 30 frames where the only motion is a smooth horizon shift => low eff rank
    frames = []
    for t in range(30):
        lab = np.zeros((40, 40), dtype=np.uint8)
        h = 18 + round(3 * float(np.sin(t / 5.0)))
        lab[h:, :] = 4
        frames.append(lab)
    gt = np.stack(frames)
    s2 = probe.stage2_boundary_dim(gt)
    # a single smooth shift mode => effective rank near 1-2
    assert s2["boundary_effective_rank_participation_ratio"] < 5.0
    assert s2["boundary_top1_var_share"] > 0.3


@pytest.mark.skipif(
    not (
        REPO
        / "experiments/results/indep_dseg_bets_20260623_inflated/seg_argmaps.npz"
    ).exists(),
    reason="cached authority argmaps not present",
)
def test_cached_argmaps_are_authority_5class():
    d = np.load(probe.ARGMAPS)
    gt = d["gt"]
    assert gt.shape == (600, 384, 512)
    assert set(np.unique(gt).tolist()) <= set(range(5))
    # the cache's own d_seg (gt vs comp) reproduces the reported authority value
    ds = float((d["gt"] != d["comp"]).mean())
    assert abs(ds - 0.0005598873) < 1e-7
