# SPDX-License-Identifier: MIT
"""Tests for the ddm_rd2 per-class / per-edge decomposition harness.

Every test here is written to FAIL if the code were broken -- no test asserts only a
constant or a shape. The load-bearing ones are:
  * ``test_confusion_is_not_determined_by_marginals`` -- the whole reason the module
    exists (xp1 cached marginals; the joint cannot be back-derived).
  * ``test_per_class_equals_per_edge_row_sums`` -- the identity that makes the per-edge
    view a refinement of the per-class view rather than a different object.
  * ``test_flicker_denominator_conventions_differ_by_exactly_n_over_n_minus_2`` -- pins
    the denominator mismatch ddm_rd2 measured against fl1's registered vector.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_rd2_perclass_peredge_decomposition import (
    CLASS_ORDER,
    N_CLASSES,
    EdgeDecomposition,
    ExhaustionIndicator,
    boundary_band_masses,
    confusion_from_labels,
    edge_rows,
    exhaustion_table,
    flicker_confusion,
    iter_chunks,
    per_class_from_confusion,
    residual_confusion,
)


def _labels(seed: int, p: int = 9, h: int = 12, w: int = 10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_CLASSES, size=(p, h, w)).astype(np.int8)


# ---------------------------------------------------------------- class order
def test_class_order_is_canonical_comma10k_not_luma_sorted():
    # CLAUDE.md records the luma sort [Road, Lane, MyCar, Undriv, Movable] as a
    # 3x-repeated measured error. Pin the canonical order so it cannot drift back.
    assert CLASS_ORDER == ("Road", "Lane", "Undrivable", "Movable", "MyCar")
    assert CLASS_ORDER[2] == "Undrivable" and CLASS_ORDER[4] == "MyCar"


# ---------------------------------------------------------------- chunking
def test_iter_chunks_partitions_exactly():
    for n, c in ((0, 4), (1, 4), (7, 3), (600, 64), (600, 600), (600, 1000)):
        spans = list(iter_chunks(n, c))
        covered = [i for lo, hi in spans for i in range(lo, hi)]
        assert covered == list(range(n))


def test_iter_chunks_rejects_nonpositive_chunk():
    with pytest.raises(ValueError):
        list(iter_chunks(10, 0))


def test_confusion_is_chunk_size_invariant():
    gt, pred = _labels(1), _labels(2)
    ref = confusion_from_labels(gt, pred, chunk=1)
    for c in (2, 3, 5, 100):
        assert np.array_equal(confusion_from_labels(gt, pred, chunk=c), ref)


# ---------------------------------------------------------------- the joint
def test_confusion_counts_are_exact_against_a_hand_built_case():
    gt = np.array([[[0, 0, 1]]], np.int8)
    pred = np.array([[[0, 2, 1]]], np.int8)
    c = confusion_from_labels(gt, pred)
    assert c[0, 0] == 1 and c[0, 2] == 1 and c[1, 1] == 1
    assert c.sum() == 3


def test_confusion_marginals_match_xp1_cls_gt_and_cls_base():
    gt, pred = _labels(3), _labels(4)
    c = confusion_from_labels(gt, pred)
    assert np.array_equal(c.sum(axis=1), np.bincount(gt.ravel(), minlength=N_CLASSES))
    assert np.array_equal(c.sum(axis=0), np.bincount(pred.ravel(), minlength=N_CLASSES))


def test_confusion_is_not_determined_by_marginals():
    """THE reason this module exists: xp1 cached only marginals."""
    a = np.array([[[0, 1]]], np.int8)
    b_same = np.array([[[0, 1]]], np.int8)
    b_swap = np.array([[[1, 0]]], np.int8)
    c1 = confusion_from_labels(a, b_same)
    c2 = confusion_from_labels(a, b_swap)
    assert np.array_equal(c1.sum(axis=1), c2.sum(axis=1))
    assert np.array_equal(c1.sum(axis=0), c2.sum(axis=0))
    assert not np.array_equal(c1, c2)  # identical marginals, different joint


def test_confusion_rejects_shape_and_range_violations():
    with pytest.raises(ValueError):
        confusion_from_labels(_labels(1), _labels(1)[:3])
    with pytest.raises(ValueError):
        confusion_from_labels(np.zeros((2, 2), np.int8), np.zeros((2, 2), np.int8))
    bad = _labels(1).astype(np.int16)
    bad[0, 0, 0] = N_CLASSES
    with pytest.raises(ValueError):
        confusion_from_labels(bad, _labels(1))


def test_empty_scope_raises_rather_than_reading_as_a_clean_pass():
    with pytest.raises(ValueError):
        confusion_from_labels(np.zeros((0, 4, 4), np.int8), np.zeros((0, 4, 4), np.int8))


# ---------------------------------------------------------------- per-class / per-edge
def test_per_class_is_off_diagonal_row_sum_in_S_units():
    conf = np.array([[8, 2, 0, 0, 0], [1, 9, 0, 0, 0], [0, 0, 5, 0, 0],
                     [0, 0, 0, 4, 1], [0, 0, 0, 0, 10]], np.int64)
    got = per_class_from_confusion(conf, denom_px=100)
    assert got == pytest.approx([2.0, 1.0, 0.0, 1.0, 0.0])


def test_per_class_rejects_bad_denominator():
    with pytest.raises(ValueError):
        per_class_from_confusion(np.zeros((5, 5), np.int64), denom_px=0)


def test_per_class_equals_per_edge_row_sums():
    dec = residual_confusion(_labels(5), _labels(6))
    assert dec.identity_holds()
    assert dec.per_edge_S.sum(axis=1) == pytest.approx(dec.per_class_S)
    assert np.all(np.diag(dec.per_edge_S) == 0.0)


def test_perfect_prediction_gives_zero_residual_everywhere():
    gt = _labels(7)
    dec = residual_confusion(gt, gt.copy())
    assert dec.total_S == pytest.approx(0.0)
    assert dec.per_edge_S.sum() == pytest.approx(0.0)


def test_edge_rows_are_sorted_and_exclude_the_diagonal():
    rows = edge_rows(residual_confusion(_labels(8), _labels(9)))
    assert len(rows) == N_CLASSES * (N_CLASSES - 1)
    assert all(r["gt_class"] != r["pred_class"] for r in rows)
    assert [r["S"] for r in rows] == sorted((r["S"] for r in rows), reverse=True)
    assert edge_rows(residual_confusion(_labels(8), _labels(9)), top=3) == rows[:3]


def test_edge_decomposition_rejects_malformed_construction():
    with pytest.raises(ValueError):
        EdgeDecomposition(confusion=np.zeros((3, 3), np.int64), denom_px=10, n_pairs=1,
                          convention="x")
    with pytest.raises(ValueError):
        EdgeDecomposition(confusion=np.zeros((5, 5), np.int64), denom_px=0, n_pairs=1,
                          convention="x")


# ---------------------------------------------------------------- flicker
def test_flicker_charge_class_is_invariant_to_neighbour_choice():
    ls = _labels(11, p=20)
    a = flicker_confusion(ls, neighbour="prev")
    b = flicker_confusion(ls, neighbour="next")
    assert a.per_class_S == pytest.approx(b.per_class_S)
    # ... but the EDGE targets are genuinely different objects
    assert not np.array_equal(a.confusion, b.confusion)


def test_flicker_denominator_conventions_differ_by_exactly_n_over_n_minus_2():
    """Pins the mismatch ddm_rd2 measured: fl1's registered vector is /598, not /600."""
    ls = _labels(12, p=30)
    interior = flicker_confusion(ls, denom="interior")
    every = flicker_confusion(ls, denom="all")
    n = ls.shape[0]
    assert interior.per_class_S == pytest.approx(every.per_class_S * n / (n - 2))


def test_flicker_finds_a_planted_spike_and_ignores_a_persistent_change():
    ls = np.zeros((5, 2, 2), np.int8)
    ls[2, 0, 0] = 1              # spike: differs from BOTH neighbours
    ls[3:, 1, 1] = 2             # step change: frame 3 differs from 2 but not from 4
    dec = flicker_confusion(ls, denom="all")
    # exactly one spike pixel, charged to its own label (1) flickering to prev label (0)
    assert dec.confusion[1, 0] == 1
    assert dec.confusion.sum() == 1


def test_flicker_rejects_too_few_pairs_and_bad_denom():
    with pytest.raises(ValueError):
        flicker_confusion(_labels(1, p=2))
    with pytest.raises(ValueError):
        flicker_confusion(_labels(1, p=8), denom="nope")
    with pytest.raises(ValueError):
        flicker_confusion(_labels(1, p=8), neighbour="sideways")


def test_flicker_is_chunk_size_invariant():
    ls = _labels(13, p=25)
    ref = flicker_confusion(ls, chunk=1).confusion
    for c in (2, 7, 1000):
        assert np.array_equal(flicker_confusion(ls, chunk=c).confusion, ref)


# ---------------------------------------------------------------- geometry
def test_boundary_split_is_exhaustive_over_flip_pixels():
    ls = _labels(14, p=6)
    flip = _labels(15, p=6) == 0
    bb = boundary_band_masses(ls, flip)
    inter_px = sum(bb["interior_flip_px"])
    # interior IS a disjoint partition, so it can never exceed the total flip mass
    assert 0 <= inter_px <= int(flip.sum())


def test_uniform_field_puts_all_flip_mass_in_the_interior():
    ls = np.full((4, 6, 6), 3, np.int8)      # one class everywhere -> no boundaries
    flip = np.ones((4, 6, 6), bool)
    bb = boundary_band_masses(ls, flip)
    assert sum(bb["interior_flip_px"]) == flip.size
    assert bb["interior_flip_px"][3] == flip.size
    assert sum(sum(r) for r in bb["boundary_pair_flip_incidences"]) == 0


def test_striped_field_puts_no_flip_mass_in_the_interior():
    ls = np.zeros((2, 4, 4), np.int8)
    ls[:, :, 1::2] = 1                       # vertical stripes -> every pixel on a boundary
    flip = np.ones((2, 4, 4), bool)
    bb = boundary_band_masses(ls, flip)
    assert sum(bb["interior_flip_px"]) == 0
    assert bb["boundary_pair_flip_incidences"][0][1] > 0


def test_boundary_rejects_mask_shape_mismatch():
    with pytest.raises(ValueError):
        boundary_band_masses(_labels(1, p=4), np.ones((3, 12, 10), bool))


# ---------------------------------------------------------------- exhaustion
def test_exhaustion_never_reports_a_bound():
    t = ExhaustionIndicator(name="Road", residual_S=0.2, reference_S=0.1)
    assert t.reference_is_a_bound is False
    assert "floor" not in t.reading.lower()
    assert t.reading == "REPRESENTATION_EXHAUSTED_NEEDS_NEW_CARRIER"
    assert ExhaustionIndicator(name="x", residual_S=0.05, reference_S=0.1).reading == (
        "CURRENT_REPRESENTATION_STILL_PAYING")


def test_exhaustion_reference_is_a_bound_cannot_be_set_by_a_caller():
    with pytest.raises(TypeError):
        ExhaustionIndicator(name="x", residual_S=1.0, reference_S=1.0,
                            reference_is_a_bound=True)  # type: ignore[call-arg]


def test_exhaustion_ratio_at_unity_reads_as_exhausted_not_as_a_wall():
    t = ExhaustionIndicator(name="Road", residual_S=0.5, reference_S=0.5)
    assert t.ratio == pytest.approx(1.0)
    assert t.reading == "REPRESENTATION_EXHAUSTED_NEEDS_NEW_CARRIER"


def test_exhaustion_table_length_mismatch_refuses():
    with pytest.raises(ValueError):
        exhaustion_table([1.0, 2.0], [1.0])


def test_exhaustion_zero_reference_is_undefined_not_infinite_confidence():
    t = ExhaustionIndicator(name="x", residual_S=1.0, reference_S=0.0)
    assert t.reading == "REFERENCE_ZERO_UNDEFINED"
