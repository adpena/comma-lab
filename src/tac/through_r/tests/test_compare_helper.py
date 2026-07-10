# SPDX-License-Identifier: MIT
"""Tests for the canonical compare back-half (#389) + numeric-identity of the migration.

The compare helper unifies the "label-stack vs L* -> agg + per-class d_seg" computation
that ``measure_through_r`` and ``measure_mask_dseg`` duplicated. These tests prove:

* the helper reproduces the INLINE formula it replaced, EXACTLY (re-derive, don't
  recognise — operating manual §4);
* the ``mask_dseg_meter`` migration is numeric-identity-preserving (same inputs ->
  identical d_seg before/after, via an independent inline recompute);
* ``ThroughRResult.to_measurement_rows`` (the #389 fence resolution) builds valid rows.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.bitmask_dseg import d_seg_reference
from tac.inc1a_harness.mask_dseg_meter import measure_mask_dseg
from tac.through_r.compare import compare_label_stack_to_lstars
from tac.witness_control.perclass_verdict import (
    CLASS_NAMES,
    per_class_dseg_fields,
    per_class_flip_stats,
)


def _synthetic_stack(n=7, h=12, w=16, seed=0):
    """A tiny synthetic (pred, gt) 5-class label stack with genuine per-class flips."""
    rng = np.random.default_rng(seed)
    gt = [rng.integers(0, 5, size=(h, w)).astype(np.int64) for _ in range(n)]
    pred = []
    for g in gt:
        p = g.copy()
        # flip a deterministic-random ~15% of pixels to a different class
        flip = rng.random((h, w)) < 0.15
        p[flip] = (p[flip] + 1 + rng.integers(0, 4, size=int(flip.sum()))) % 5
        pred.append(p.astype(np.int64))
    return pred, gt


def _inline_reference(pred_list, gt_list, n_classes=5):
    """Re-derive agg + per-class the OLD inline way (the formula the helper replaced)."""
    per_pair = np.empty(len(pred_list), dtype=np.float64)
    for i in range(len(pred_list)):
        per_pair[i] = d_seg_reference(pred_list[i], gt_list[i])
    agg = float(per_pair.mean())
    flips, pixels = per_class_flip_stats(pred_list, gt_list, n_classes=n_classes)
    fields = per_class_dseg_fields(flips, pixels)
    per_class = {CLASS_NAMES[c]: float(fields["d_seg_by_class"][c]) for c in range(n_classes)}
    share = {CLASS_NAMES[c]: float(fields["flip_share_by_class"][c]) for c in range(n_classes)}
    return agg, per_class, share, int(flips.sum()), int(pixels.sum()), float(per_pair.std())


def test_compare_helper_matches_inline_formula_exactly():
    pred, gt = _synthetic_stack()
    cmp = compare_label_stack_to_lstars(pred, gt, n_classes=5)
    agg, per_class, share, tf, tp, std = _inline_reference(pred, gt)
    assert cmp.agg_dseg == agg
    assert cmp.per_class_dseg == per_class
    assert cmp.flip_share_by_class == share
    assert cmp.total_flips == tf
    assert cmp.total_pixels == tp
    assert cmp.per_pair_std == std
    # sum identity: mean of per-pair == agg; per-class flips sum to total flips.
    assert cmp.agg_dseg == float(np.mean(cmp.per_pair_dseg))


def test_compare_helper_length_and_empty_guards():
    with pytest.raises(ValueError):
        compare_label_stack_to_lstars([], [], n_classes=5)
    p = [np.zeros((4, 4), np.int64)]
    with pytest.raises(ValueError):
        compare_label_stack_to_lstars(p, [], n_classes=5)


def test_mask_meter_migration_is_numeric_identity():
    """measure_mask_dseg (now delegating) == the inline reference, exactly."""
    pred, gt = _synthetic_stack(n=5)
    res = measure_mask_dseg(pred, np.stack(gt), require_n600=False, label="[synthetic]")
    agg, per_class, share, tf, tp, std = _inline_reference(pred, gt)
    assert res.agg_dseg == agg
    assert res.per_class_dseg == per_class
    assert res.flip_share_by_class == share
    assert res.total_flips == tf
    assert res.total_pixels == tp
    assert res.extra["per_frame_std"] == std  # key preserved (API unchanged)


def test_mask_meter_shape_mismatch_still_raises_its_own_error():
    from tac.inc1a_harness.mask_dseg_meter import MaskDsegMeterError

    pred = [np.zeros((4, 4), np.int64), np.zeros((5, 5), np.int64)]
    gt = np.zeros((2, 4, 4), np.int64)
    with pytest.raises(MaskDsegMeterError):
        measure_mask_dseg(pred, gt, require_n600=False)


def test_to_measurement_rows_builds_valid_rows():
    from tac.through_r.harness import ThroughRResult
    from tac.verdicts import AxisTag, MeasurementRow, ReviewStatus

    res = ThroughRResult(
        agg_dseg=0.0042,
        per_class_dseg={n: 0.001 * (i + 1) for i, n in enumerate(CLASS_NAMES)},
        flip_share_by_class=dict.fromkeys(CLASS_NAMES, 0.2),
        per_pair_dseg=[0.0042] * 600,
        n_frames=600,
        n_classes=5,
        is_n600=True,
        backend="cpu-torch",
        input_space="camera-uint8",
        total_flips=100,
        total_pixels=10000,
        inputs_sha256="a" * 64,
    )
    rows = res.to_measurement_rows(git_sha="deadbeef", review_status="provisional")
    assert len(rows) == 6  # aggregate + 5 per-class
    agg_row = rows[0]
    assert isinstance(agg_row, MeasurementRow)
    assert agg_row.quantity == "d_seg"
    assert agg_row.value == 0.0042
    assert agg_row.axis_tag == AxisTag.THROUGH_R
    assert agg_row.axis_tag.is_authority is False  # through-R is NEVER authority
    assert agg_row.n_samples == 600
    assert agg_row.review_status == ReviewStatus.PROVISIONAL
    assert agg_row.provenance.inputs_sha256 == "a" * 64
    # per-class rows carry the class-qualified quantity.
    assert rows[1].quantity.startswith("d_seg::")
    # aggregate-only variant.
    only_agg = res.to_measurement_rows(
        git_sha="deadbeef", review_status="reviewed", include_per_class=False
    )
    assert len(only_agg) == 1


def test_to_measurement_rows_subset_carries_reason():
    from tac.through_r.harness import ThroughRResult

    res = ThroughRResult(
        agg_dseg=0.01,
        per_class_dseg=dict.fromkeys(CLASS_NAMES, 0.002),
        flip_share_by_class=dict.fromkeys(CLASS_NAMES, 0.2),
        per_pair_dseg=[0.01] * 8,
        n_frames=8,
        n_classes=5,
        is_n600=False,
        backend="cpu-torch",
        input_space="render-grid",
        total_flips=5,
        total_pixels=500,
        subset_reason="tiny synthetic feasibility probe",
        inputs_sha256=None,
    )
    # n_samples != 600 REQUIRES a reason — the subset_reason must flow through so the row
    # constructs (otherwise MeasurementRow refuses).
    rows = res.to_measurement_rows(git_sha="cafe", review_status="provisional")
    assert rows[0].n_samples == 8
    assert rows[0].n_samples_reason == "tiny synthetic feasibility probe"
