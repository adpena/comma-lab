# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.local_contest_instruments` (win family F5).

Positive controls are EXECUTED, not asserted-about: the score arithmetic is checked
against the live pointer row's own published components, and the axis/lineage binding is
checked against :mod:`tac.gt_lineage`'s constants rather than against re-typed strings.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tac import gt_lineage
from tac import local_contest_instruments as lci

# The live pointer row (ddm_up3 thirteenth move), used as an executed positive control.
POINTER_D_SEG = 0.00030309
POINTER_D_POSE = 7.77e-06
POINTER_ARCHIVE_BYTES = 176_420
POINTER_SCORE = 0.15659459685822907


# --- axis <-> lineage binding ------------------------------------------------


def test_cuda_axis_binds_to_dali_lineage():
    assert lci.required_lineage_for_axis(lci.AXIS_CONTEST_CUDA) == gt_lineage.DALI_NVDEC


def test_cpu_axis_binds_to_pyav_lineage():
    assert lci.required_lineage_for_axis(lci.AXIS_CONTEST_CPU) == gt_lineage.PYAV_YUV420_TO_RGB


def test_advisory_axis_binds_to_pyav_lineage():
    assert (
        lci.required_lineage_for_axis(lci.AXIS_MACOS_CPU_ADVISORY)
        == gt_lineage.PYAV_YUV420_TO_RGB
    )


def test_cuda_axis_uses_the_gt_lineage_authority_constant():
    """The CUDA axis must bind to whatever gt_lineage calls AUTHORITY -- not a copy."""
    assert lci.AXIS_GT_LINEAGE[lci.AXIS_CONTEST_CUDA] == gt_lineage.AUTHORITY_LINEAGE


def test_unknown_axis_refuses():
    with pytest.raises(lci.InstrumentRefusal, match="unknown score axis"):
        lci.required_lineage_for_axis("contest-TPU")


def test_known_axes_are_sorted_and_complete():
    assert lci.known_axes() == tuple(sorted(lci.AXIS_GT_LINEAGE))


def test_every_bound_lineage_is_a_resolved_gt_lineage():
    """No axis may bind to UNKNOWN/ambiguous -- that would launder an unresolved read."""
    for lineage in lci.AXIS_GT_LINEAGE.values():
        assert lineage in gt_lineage.RESOLVED_LINEAGES


# --- the pose-absolute refusal ----------------------------------------------


def test_pose_absolute_allowed_on_cuda_axis():
    lci.assert_pose_absolute_quotable(lci.AXIS_CONTEST_CUDA)


def test_pose_absolute_refused_on_advisory_axis():
    with pytest.raises(lci.PoseAbsoluteRefused, match="additive d_pose floor"):
        lci.assert_pose_absolute_quotable(lci.AXIS_MACOS_CPU_ADVISORY)


def test_pose_absolute_refused_on_contest_cpu_axis():
    with pytest.raises(lci.PoseAbsoluteRefused):
        lci.assert_pose_absolute_quotable(lci.AXIS_CONTEST_CPU)


def test_pose_absolute_allowed_with_explicit_pyav_opt_in():
    lci.assert_pose_absolute_quotable(
        lci.AXIS_CONTEST_CPU, allow_pyav_objective=True
    )


def test_pose_absolute_refusal_names_the_instrument():
    with pytest.raises(lci.PoseAbsoluteRefused, match="my_solver"):
        lci.assert_pose_absolute_quotable(
            lci.AXIS_MACOS_CPU_ADVISORY, instrument="my_solver"
        )


def test_pose_absolute_on_unknown_axis_refuses_as_unknown_axis():
    with pytest.raises(lci.InstrumentRefusal, match="unknown score axis"):
        lci.assert_pose_absolute_quotable("nonsense")


def test_additive_floor_dominates_a_good_carrier():
    """POSITIVE CONTROL for the refusal's premise, executed rather than asserted.

    The refusal exists because the PyAV floor is most of a good carrier's PyAV total.
    Measure that on the live pointer's own d_pose instead of restating it.
    """
    advisory_total = POINTER_D_POSE + lci.ADVISORY_POSE_ADDITIVE_FLOOR
    floor_share = lci.ADVISORY_POSE_ADDITIVE_FLOOR / advisory_total
    assert floor_share > 0.9, floor_share


# --- score arithmetic: executed positive control -----------------------------


def test_score_reproduces_the_live_pointer_row_exactly():
    assert lci.contest_score_from_legs(
        POINTER_D_SEG, POINTER_D_POSE, POINTER_ARCHIVE_BYTES
    ) == pytest.approx(POINTER_SCORE, abs=1e-15)


def test_legs_sum_to_the_score():
    total = (
        lci.seg_leg(POINTER_D_SEG)
        + lci.pose_leg(POINTER_D_POSE)
        + lci.rate_leg(POINTER_ARCHIVE_BYTES)
    )
    assert total == pytest.approx(POINTER_SCORE, abs=1e-12)


def test_seg_leg_is_the_upstream_hundred_times_rule():
    assert lci.seg_leg(0.001) == pytest.approx(0.1)


def test_pose_leg_is_the_upstream_sqrt_rule():
    assert lci.pose_leg(0.4) == pytest.approx(2.0)


def test_pose_report_bound_grows_as_d_pose_falls():
    """The 8dp reporting bound is LARGER on a better carrier -- the sqrt's own geometry."""
    assert lci.pose_report_bound(1e-8) > lci.pose_report_bound(1e-4)


def test_pose_report_bound_at_zero_is_finite():
    assert math.isfinite(lci.pose_report_bound(0.0))


def test_resolvable_floor_is_the_report_half_ulp():
    assert lci.resolvable_d_pose_floor() == 0.5e-8


# --- population selection ----------------------------------------------------


def test_full_field_returns_every_pair_in_order():
    pairs = lci.select_pairs(600, seed=1)
    assert np.array_equal(pairs, np.arange(600))


def test_over_request_still_returns_the_full_field():
    assert len(lci.select_pairs(10_000, seed=1)) == 600


def test_subsample_is_not_a_prefix():
    """The prefix bias inverts sign per axis; a sub-n600 draw must never be [0..n)."""
    pairs = lci.select_pairs(96, seed=20260819)
    assert not np.array_equal(pairs, np.arange(96))
    assert len(pairs) == 96


def test_subsample_is_seeded_and_reproducible():
    assert np.array_equal(lci.select_pairs(50, seed=7), lci.select_pairs(50, seed=7))


def test_different_seeds_give_different_samples():
    assert not np.array_equal(lci.select_pairs(50, seed=7), lci.select_pairs(50, seed=8))


def test_subsample_is_sorted_and_unique():
    pairs = lci.select_pairs(100, seed=3)
    assert np.array_equal(pairs, np.sort(pairs))
    assert len(set(pairs.tolist())) == 100


def test_non_positive_pairs_refuses():
    with pytest.raises(lci.InstrumentRefusal):
        lci.select_pairs(0, seed=1)


# --- the receipt -------------------------------------------------------------


def _receipt(**overrides):
    base = {
        "instrument": "test",
        "axis": lci.AXIS_CONTEST_CUDA,
        "gt_lineage": gt_lineage.DALI_NVDEC,
        "pairs": 600,
        "sampling": "full_field",
    }
    base.update(overrides)
    return lci.InstrumentReceipt(**base)


def test_receipt_is_never_a_score_claim():
    assert _receipt().score_claim is False


def test_receipt_is_never_promotable():
    assert _receipt().promotable is False


def test_receipt_false_authority_flags_are_not_constructor_arguments():
    """They are ``init=False``: there is no local config that makes this a score."""
    with pytest.raises(TypeError):
        lci.InstrumentReceipt(
            instrument="t",
            axis=lci.AXIS_CONTEST_CUDA,
            gt_lineage=gt_lineage.DALI_NVDEC,
            pairs=1,
            sampling="s",
            score_claim=True,
        )


def test_receipt_refuses_axis_lineage_mismatch():
    with pytest.raises(lci.InstrumentRefusal, match="may not record a number"):
        _receipt(gt_lineage=gt_lineage.PYAV_YUV420_TO_RGB)


def test_receipt_refuses_unknown_axis():
    with pytest.raises(lci.InstrumentRefusal, match="unknown score axis"):
        _receipt(axis="contest-TPU")


def test_receipt_refuses_non_positive_pairs():
    with pytest.raises(lci.InstrumentRefusal):
        _receipt(pairs=0)


def test_receipt_full_field_flag():
    assert _receipt(pairs=600).is_full_field
    assert not _receipt(pairs=599).is_full_field


def test_receipt_composes_the_pointer_score():
    receipt = _receipt(
        d_seg=POINTER_D_SEG, d_pose=POINTER_D_POSE, archive_bytes=POINTER_ARCHIVE_BYTES
    )
    assert receipt.score() == pytest.approx(POINTER_SCORE, abs=1e-15)


def test_partial_receipt_refuses_to_compose_a_score():
    with pytest.raises(lci.InstrumentRefusal, match="partial receipt"):
        _receipt(d_seg=POINTER_D_SEG).score()


def test_receipt_json_always_carries_the_false_authority_flags():
    payload = _receipt().to_json()
    assert payload["score_claim"] is False
    assert payload["promotable"] is False
    assert "gt_lineage" in payload and payload["gt_lineage"] == gt_lineage.DALI_NVDEC


# --- per-pair legs -----------------------------------------------------------


def test_d_seg_per_pair_matches_the_upstream_mean_rule():
    argmax = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
    gt = np.array([[[0, 9], [2, 9]]], dtype=np.uint8)
    assert lci.d_seg_per_pair(argmax, gt) == pytest.approx([0.5])


def test_d_seg_per_pair_is_zero_on_identity():
    argmax = np.zeros((3, 4, 4), dtype=np.uint8)
    assert np.all(lci.d_seg_per_pair(argmax, argmax.copy()) == 0.0)


def test_d_seg_per_pair_refuses_shape_mismatch():
    with pytest.raises(lci.InstrumentRefusal, match="disagree in shape"):
        lci.d_seg_per_pair(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)))


def test_d_pose_per_pair_scores_only_the_first_six_components():
    pose = np.zeros((1, 12))
    targets = np.zeros((1, 12))
    targets[0, 6:] = 1000.0  # components 6..11 are NOT scored upstream
    assert lci.d_pose_per_pair(pose, targets) == pytest.approx([0.0])


def test_d_pose_per_pair_is_a_mean_square_error():
    pose = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    targets = np.zeros((1, 6))
    assert lci.d_pose_per_pair(pose, targets) == pytest.approx([1.0])


def test_d_pose_per_pair_refuses_too_few_components():
    with pytest.raises(lci.InstrumentRefusal, match="at least 6"):
        lci.d_pose_per_pair(np.zeros((1, 3)), np.zeros((1, 3)))


def test_population_leg_refuses_empty():
    with pytest.raises(lci.InstrumentRefusal):
        lci.population_leg([])
