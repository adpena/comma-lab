# SPDX-License-Identifier: MIT
"""Tests for the per-class-weighted bit-alloc witness apply (#336 SPARC grain).

Covers the PURE, no-render helpers (per-pair class counts, order-independent aggregation, the
de-starving functional). The heavy render+SegNet path is exercised only in the governed measurement
run (advisory, never in unit tests)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from apply_perclass_bitalloc_witness import (  # noqa: E402
    aggregate_perclass,
    per_pair_class_counts,
    perclass_functional,
)


def test_per_pair_class_counts_all_correct_zero_miss():
    gt = np.array([[0, 1], [2, 3]], np.int64)
    realized = gt.copy()
    miss, npx = per_pair_class_counts(realized, gt, n_classes=5)
    assert list(npx) == [1, 1, 1, 1, 0]
    assert list(miss) == [0, 0, 0, 0, 0]


def test_per_pair_class_counts_targeted_lane_miss():
    # gt: class 1 (Lane) at two pixels, class 0 elsewhere; realized flips ONLY the Lane pixels.
    gt = np.array([[1, 1], [0, 0]], np.int64)
    realized = np.array([[0, 0], [0, 0]], np.int64)
    miss, npx = per_pair_class_counts(realized, gt, n_classes=5)
    assert list(npx) == [2, 2, 0, 0, 0]
    assert list(miss) == [0, 2, 0, 0, 0]  # both Lane pixels wrong, Road pixels correct


def test_per_pair_class_counts_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        per_pair_class_counts(np.zeros((2, 2), np.int64), np.zeros((3, 3), np.int64))


def test_aggregate_order_independent():
    # Two pairs; aggregation over [p0, p1] must equal aggregation over [p1, p0].
    rec0 = {"miss": [1, 2, 0, 0, 0], "npx": [10, 4, 0, 0, 0]}
    rec1 = {"miss": [0, 1, 3, 0, 0], "npx": [6, 2, 12, 0, 0]}
    agg_a, pc_a = aggregate_perclass([rec0, rec1])
    agg_b, pc_b = aggregate_perclass([rec1, rec0])
    assert agg_a == pytest.approx(agg_b)
    np.testing.assert_allclose(pc_a, pc_b)


def test_aggregate_matches_manual():
    rec0 = {"miss": [1, 2, 0, 0, 0], "npx": [10, 4, 0, 0, 0]}
    rec1 = {"miss": [0, 1, 3, 0, 0], "npx": [6, 2, 12, 0, 0]}
    agg, pc = aggregate_perclass([rec0, rec1])
    # aggregate = total miss / total px = (1+2+0+1+3) / (10+4+6+2+12) = 7/34
    assert agg == pytest.approx(7 / 34)
    # per-class within-class rate: Road (1+0)/(10+6)=1/16; Lane (2+1)/(4+2)=3/6; Undriv 3/12=0.25
    np.testing.assert_allclose(pc, [1 / 16, 3 / 6, 3 / 12, 0.0, 0.0])


def test_aggregate_absent_class_stays_zero():
    rec = {"miss": [0, 0, 0, 0, 0], "npx": [5, 0, 0, 0, 0]}
    agg, pc = aggregate_perclass([rec])
    assert agg == 0.0
    assert list(pc) == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_perclass_functional_equal_weight_upweights_rare():
    # Aggregate metric barely moves when a rare class collapses, but the per-class functional does.
    # Baseline: everything correct.
    base_pc = np.zeros((5,), np.float64)
    # Rare-class (Lane) fully collapses: within-class miss rate = 1.0 for class 1.
    collapsed_pc = np.array([0.0, 1.0, 0.0, 0.0, 0.0], np.float64)
    d_base = perclass_functional(base_pc)
    d_collapse = perclass_functional(collapsed_pc)
    # Per-class functional jumps by 1/5 = 0.2 regardless of Lane's pixel share.
    assert d_collapse - d_base == pytest.approx(0.2)


def test_perclass_functional_empty():
    assert perclass_functional(np.array([], np.float64)) == 0.0


def test_perclass_functional_is_class_mean_not_pixel_mean():
    # Two classes with wildly different pixel shares but same within-class rate → functional = rate.
    pc = np.array([0.3, 0.3, 0.0, 0.0, 0.0], np.float64)
    assert perclass_functional(pc) == pytest.approx(0.6 / 5)
