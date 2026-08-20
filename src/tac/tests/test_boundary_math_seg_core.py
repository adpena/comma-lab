# SPDX-License-Identifier: MIT
"""Behavior tests for the boundary_math seg-core (task #52).

NO-FAKE discipline (CLAUDE.md class 2 — tests verify BEHAVIOR, not constants):
- dense-raster LZMA baseline: assert ENCODE->DECODE is bit-exact AND that it
  compresses (a fake identity codec would fail the compression assertion on a
  low-boundary partition).
- d_seg: assert the popcount form EQUALS the reference argmax-compare on random
  partitions (a fake constant-returning d_seg would fail).
- region-merge: assert the solve actually DROPS the predicted tiny regions and
  rewrites the label map (a no-op merge would fail the merged-count + byte-drop
  assertions).
- margin-polytope: assert the free budget is monotone in margin and that the
  jacobian path actually divides by ||g_p|| (a fake passthrough would fail).
- water level: assert it equals the spec's derived 1.27 within tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

import tac.boundary_math as bm
from tac.boundary_math.bitmask_dseg import (
    class_masks_from_argmax,
    d_seg_bitmask,
    d_seg_reference,
    flip_count,
)
from tac.boundary_math.dense_raster_lzma_baseline import (
    decode_partition,
    encode_partition,
    partition_description_bytes,
)
from tac.boundary_math.margin_polytope import (
    boundary_pixel_mask,
    free_budget_from_margin_jacobian,
)
from tac.boundary_math.partition import (
    build_region_adjacency_graph,
    connected_components,
)
from tac.boundary_math.region_merge import (
    WATER_LEVEL_BYTES_PER_FLIP,
    solve_mdl_region_merge,
)


# ── bitmask d_seg ───────────────────────────────────────────────────────────
def test_d_seg_bitmask_equals_reference_on_random_partitions():
    rng = np.random.default_rng(7)
    for _ in range(20):
        gt = rng.integers(0, 5, size=(16, 24)).astype(np.int64)
        cand = gt.copy()
        # flip a random subset
        mask = rng.random((16, 24)) < 0.3
        cand[mask] = (cand[mask] + rng.integers(1, 5, size=int(mask.sum()))) % 5
        ref = d_seg_reference(cand, gt)
        bit = d_seg_bitmask(cand, gt)
        assert bit == pytest.approx(ref), "bitmask popcount form must equal argmax-compare"


def test_d_seg_zero_iff_identical():
    a = np.array([[0, 1], [2, 3]], dtype=np.int64)
    assert d_seg_reference(a, a) == 0.0
    assert d_seg_bitmask(a, a) == 0.0


def test_d_seg_single_flip_rate():
    gt = np.zeros((4, 4), dtype=np.int64)
    cand = gt.copy()
    cand[0, 0] = 1
    assert flip_count(cand, gt) == 1
    assert d_seg_reference(cand, gt) == pytest.approx(1 / 16)
    assert d_seg_bitmask(cand, gt) == pytest.approx(1 / 16)


def test_class_masks_one_hot_exclusive():
    a = np.array([[0, 4], [2, 2]], dtype=np.int64)
    masks = class_masks_from_argmax(a, n_classes=5)
    # exactly one True per pixel across the class axis
    assert np.array_equal(masks.sum(axis=0), np.ones_like(a))
    assert masks[0, 0, 0] and masks[4, 0, 1] and masks[2, 1, 0]


def test_class_masks_rejects_out_of_range():
    with pytest.raises(ValueError):
        class_masks_from_argmax(np.array([[0, 5]]), n_classes=5)


# ── dense-raster LZMA baseline ──────────────────────────────────────────────
def test_dense_raster_lzma_baseline_roundtrip_bit_exact():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 5, size=(48, 64)).astype(np.int64)
    code = encode_partition(a)
    decoded = decode_partition(code)
    assert np.array_equal(decoded, a), "codec must be a bit-exact reversible identity"


def test_dense_raster_lzma_baseline_compresses_low_boundary_partition():
    # A 2-region partition (one vertical boundary) is mostly constant runs; the codec
    # must compress it well below the raw uint8 size.  A FAKE (identity) codec would
    # store all H*W bytes and fail this.
    a = np.zeros((96, 128), dtype=np.int64)
    a[:, 64:] = 1
    n = partition_description_bytes(a)
    raw = a.size  # 1 byte/pixel raw
    assert n < raw * 0.05, f"low-boundary partition should compress hugely: {n} vs raw {raw}"


def test_dense_raster_lzma_baseline_grows_with_boundary_complexity():
    # A high-boundary (checkerboard) partition has less label-raster regularity than a
    # single-split partition -> more bytes.  This proves the byte cost tracks
    # boundary, not area (both have the same area).
    low = np.zeros((64, 64), dtype=np.int64)
    low[:, 32:] = 1
    checker = (np.indices((64, 64)).sum(axis=0) % 2).astype(np.int64)
    assert partition_description_bytes(checker) > partition_description_bytes(low)


# ── partition / RAG ─────────────────────────────────────────────────────────
def test_connected_components_separates_disjoint_same_class():
    # Two separate blobs of class 1 must be TWO regions, not one.
    a = np.zeros((8, 8), dtype=np.int64)
    a[0:2, 0:2] = 1
    a[6:8, 6:8] = 1
    region_of, regions = connected_components(a, n_classes=5)
    class1_regions = [r for r in regions.values() if r.label == 1]
    assert len(class1_regions) == 2, "disjoint same-class blobs are distinct regions"


def test_connected_components_4connectivity_not_diagonal():
    # Diagonal-only-touching same-class pixels are NOT connected under 4-connectivity.
    a = np.zeros((4, 4), dtype=np.int64)
    a[0, 0] = 1
    a[1, 1] = 1  # diagonal neighbour
    _, regions = connected_components(a, n_classes=5)
    class1 = [r for r in regions.values() if r.label == 1]
    assert len(class1) == 2, "diagonal touch must NOT merge under 4-connectivity"


def test_rag_adjacency_is_symmetric_and_correct():
    a = np.zeros((6, 6), dtype=np.int64)
    a[:, 3:] = 1
    rag = build_region_adjacency_graph(a, n_classes=5)
    assert rag.n_regions() == 2
    ids = list(rag.regions)
    # the two halves touch along the vertical seam -> adjacent
    assert ids[1] in rag.neighbours(ids[0])
    assert ids[0] in rag.neighbours(ids[1])


def test_rag_region_of_covers_all_pixels():
    rng = np.random.default_rng(11)
    a = rng.integers(0, 5, size=(20, 20)).astype(np.int64)
    rag = build_region_adjacency_graph(a, n_classes=5)
    assert (rag.region_of >= 0).all()
    # every pixel's region label matches the source class
    for region in rag.regions.values():
        rows, cols = region.coords
        assert np.all(a[rows, cols] == region.label)


# ── margin polytope ─────────────────────────────────────────────────────────
def test_free_budget_margin_only_is_monotone():
    margin = np.array([[0.0, 1.0], [2.0, 5.0]], dtype=np.float64)
    fb = free_budget_from_margin_jacobian(margin, None)
    assert not fb.jacobian_supplied
    # budget == margin (monotone proxy) when no jacobian
    assert np.array_equal(fb.budget, margin)
    # the largest-margin pixel is in the free set; the zero-margin pixel is not.
    assert fb.free_mask[1, 1]
    assert not fb.free_mask[0, 0]


def test_free_budget_jacobian_divides_by_gradient_norm():
    # A FAKE passthrough would ignore the jacobian; assert b(p) = m/||g|| actually
    # divides — doubling the gradient norm halves the budget.
    margin = np.full((2, 2), 4.0, dtype=np.float64)
    g1 = np.full((2, 2), 1.0, dtype=np.float64)
    g2 = np.full((2, 2), 2.0, dtype=np.float64)
    fb1 = free_budget_from_margin_jacobian(margin, g1)
    fb2 = free_budget_from_margin_jacobian(margin, g2)
    assert fb1.jacobian_supplied
    assert np.allclose(fb1.budget, 4.0)
    assert np.allclose(fb2.budget, 2.0), "budget must scale as 1/||g_p||"


def test_boundary_pixel_mask_thresholds_margin():
    margin = np.array([[0.1, 0.6], [0.49, 1.0]], dtype=np.float64)
    bmask = boundary_pixel_mask(margin, margin_threshold=0.5)
    assert bmask[0, 0] and bmask[1, 0]
    assert not bmask[0, 1] and not bmask[1, 1]


def test_free_budget_rejects_negative_margin():
    with pytest.raises(ValueError):
        free_budget_from_margin_jacobian(np.array([[-1.0]]), None)


# ── water level ─────────────────────────────────────────────────────────────
def test_water_level_matches_spec_derivation():
    # (100 / (600*384*512)) / (25 / 37_545_489) ~ 1.2731 B/flip (spec §10 = 1.27).
    assert pytest.approx(1.2731, abs=0.01) == WATER_LEVEL_BYTES_PER_FLIP


# ── region-merge SOLVE ──────────────────────────────────────────────────────
def test_region_merge_drops_tiny_over_water_regions_and_rewrites_labels():
    # Candidate == GT plus a 1px spurious region (class 2) inside a class-0 sea.  That
    # region costs >1.27 bytes but fixes 0 flips (it's WRONG vs GT) -> must be merged.
    gt = np.zeros((16, 16), dtype=np.int64)
    cand = gt.copy()
    cand[8, 8] = 2  # spurious 1px region disagreeing with GT
    plan = solve_mdl_region_merge(cand, gt)
    # the spurious region must be merged away (its pixel reverts to a neighbour label)
    assert plan.merged_partition[8, 8] != 2, "spurious wrong region must be merged out"
    assert len(plan.merged_region_ids) >= 1, "solve must actually drop >= 1 region"
    # after merge, d_seg vs GT improves (the wrong pixel fixed)
    assert plan.d_seg_after <= plan.d_seg_before


def test_region_merge_keeps_large_rent_paying_region():
    # A large region that AGREES with GT must be kept (it fixes many flips per byte).
    gt = np.zeros((32, 32), dtype=np.int64)
    gt[:, 16:] = 1  # half the frame is class 1 in GT
    cand = gt.copy()  # candidate matches GT exactly
    plan = solve_mdl_region_merge(cand, gt)
    # the candidate already == GT: d_seg stays 0, and the large class-1 region is kept
    assert plan.d_seg_after == 0.0
    # the big region (region covering class 1 half) must not be merged into class 0
    assert (plan.merged_partition[:, 16:] == 1).all(), "large GT-matching region kept"


def test_region_merge_is_not_a_noop():
    # Anti-fake: a no-op merge would leave bytes/flips unchanged.  Construct a
    # candidate with many spurious tiny regions and assert the solve CHANGES the
    # partition (merges some).
    rng = np.random.default_rng(5)
    gt = np.zeros((40, 40), dtype=np.int64)
    gt[:, 20:] = 1
    cand = gt.copy()
    ys = rng.integers(2, 38, 15)
    xs = rng.integers(2, 18, 15)
    for y, x in zip(ys, xs, strict=True):
        cand[y, x] = 3  # spurious tiny regions in the class-0 half
    plan = solve_mdl_region_merge(cand, gt)
    assert not np.array_equal(plan.merged_partition, cand), "solve must alter the partition"
    assert plan.bytes_after < plan.bytes_before, "merging tiny regions must save bytes"
    assert len(plan.merged_region_ids) > 0


def test_region_merge_respects_water_level_threshold():
    # The keep/merge decision must track the water level, not a hardcoded rule.  Use a
    # GT-matching tiny region (it FIXES 1 flip by staying distinct): at a HIGH water
    # level the flip is worth many bytes so the cheap region is kept; at a LOW water
    # level the flip is worth ~nothing so the region's bytes exceed its rent -> merged.
    gt = np.zeros((24, 24), dtype=np.int64)
    gt[12, 12] = 1  # GT has a 1px class-1 region inside the class-0 sea
    cand = gt.copy()  # candidate matches GT exactly
    high = solve_mdl_region_merge(cand, gt, water_level=1e9)
    low = solve_mdl_region_merge(cand, gt, water_level=1e-12)
    # high water level: keep the matching region (rent trivially paid)
    assert high.merged_partition[12, 12] == 1
    # low water level: the 1px region's bytes exceed 1 flip * tiny_wl -> merged away
    assert low.merged_partition[12, 12] != 1


# ── real-SegNet exact-scorer smoke (skipped if checkpoint absent) ────────────
def _segnet_available() -> bool:
    from tac.boundary_math.seg_core import DEFAULT_VIDEO, SEGNET_CKPT

    return SEGNET_CKPT.exists() and DEFAULT_VIDEO.exists()


@pytest.mark.skipif(not _segnet_available(), reason="SegNet checkpoint / video absent")
def test_real_segnet_lstar_roundtrips_at_zero_dseg():
    segnet = bm.load_real_segnet("cpu")
    rows = list(bm.decode_gt_frame1_pairs(n_pairs=1))
    assert rows, "must decode at least one GT pair"
    _idx, _f0, f1 = rows[0]
    res = bm.build_and_measure_lstar(segnet, f1, pair_idx=0)
    # PRE-REGISTERED PREDICTION: stored partition == SegNet argmax -> d_seg == 0 exact
    assert res.d_seg_lstar == 0.0
    assert res.roundtrip_exact
    assert res.partition_bytes > 0
    assert res.n_regions > 1
    assert res.shape == (384, 512)
    assert res.authority == "local-CPU-torch-advisory"
