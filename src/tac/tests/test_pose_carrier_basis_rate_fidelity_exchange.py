"""Tests for ``pose_carrier_basis_rate_fidelity_exchange_v1``.

The load-bearing test is :func:`test_every_measured_rung_falls_inside_the_band`:
the equation's predictive claim is that a basis payload size prices an archive
delta, and the six MEASURED rungs are the only evidence for it.  If a future body
moves the pass-through band, that test fails first and the band gets re-derived
instead of quietly mispricing the next basis idea.
"""

from __future__ import annotations

import itertools
import math

import pytest

from tac.canonical_equations.pose_carrier_basis_rate_fidelity_exchange_20260905 import (
    BASE_D_POSE_CPU_TORCH,
    BASIS_SYMBOLS,
    CARRIER_STREAM_BYTES,
    DEAD_BASIS_SCALE_BYTES,
    EQUATION_ID,
    FRONTIER_ARCHIVE_BYTES,
    MEASURED_BASIS_RUNGS,
    NEARLY_DEAD_BASIS_SCALE_BYTES,
    PASS_THROUGH_MAX,
    PASS_THROUGH_MIN,
    PASS_THROUGH_TYPICAL,
    RATE_DENOMINATOR_BYTES,
    SHIPPED_BASIS_PAYLOAD_BYTES,
    archive_delta_band,
    archive_delta_from_basis_payload_bytes,
    basis_break_even_d_pose,
    break_even_d_pose,
    build_pose_carrier_basis_rate_fidelity_exchange_v1,
    rate_credit,
)


def test_equation_builds_and_keeps_its_id():
    equation = build_pose_carrier_basis_rate_fidelity_exchange_v1()
    assert equation.equation_id == EQUATION_ID


def test_equation_carries_both_anchors_and_names_its_producer():
    equation = build_pose_carrier_basis_rate_fidelity_exchange_v1()
    assert len(equation.empirical_anchors) == 2
    ids = {anchor.anchor_id for anchor in equation.empirical_anchors}
    assert any("quantiser_step" in name for name in ids)
    assert any("generated_dct_basis_refused" in name for name in ids)
    assert any(
        "ddm_pc1_pose_carrier_efficiency.py" in producer
        for producer in equation.canonical_producers
    )


# --------------------------------------------------------------------------
# the predictive claim
# --------------------------------------------------------------------------


def test_every_measured_rung_falls_inside_the_band():
    """The equation's only predictive claim, checked against all its evidence."""
    for label, (payload, archive_bytes, _c0, _c1) in MEASURED_BASIS_RUNGS.items():
        low, high = archive_delta_band(payload)
        actual = archive_bytes - FRONTIER_ARCHIVE_BYTES
        assert low - 1.0 <= actual <= high + 1.0, (
            f"{label}: measured archive delta {actual} outside predicted "
            f"band ({low:.1f}, {high:.1f})"
        )


def test_the_shipped_rung_predicts_exactly_zero():
    assert archive_delta_from_basis_payload_bytes(SHIPPED_BASIS_PAYLOAD_BYTES) == 0.0
    assert archive_delta_band(SHIPPED_BASIS_PAYLOAD_BYTES) == (0.0, 0.0)


def test_a_smaller_payload_predicts_a_smaller_archive():
    assert archive_delta_from_basis_payload_bytes(0) < 0.0
    assert archive_delta_from_basis_payload_bytes(SHIPPED_BASIS_PAYLOAD_BYTES // 2) < 0.0


def test_a_larger_payload_predicts_a_larger_archive():
    assert archive_delta_from_basis_payload_bytes(SHIPPED_BASIS_PAYLOAD_BYTES + 500) > 0.0


def test_negative_payload_is_refused():
    with pytest.raises(ValueError):
        archive_delta_from_basis_payload_bytes(-1)


def test_band_brackets_the_typical_pass_through():
    low, high = archive_delta_band(0)
    typical = archive_delta_from_basis_payload_bytes(0)
    assert low <= typical <= high
    assert PASS_THROUGH_MIN < PASS_THROUGH_TYPICAL < PASS_THROUGH_MAX


# --------------------------------------------------------------------------
# score arithmetic
# --------------------------------------------------------------------------


def test_rate_credit_is_the_contest_slope():
    assert rate_credit(1) == 25.0 / RATE_DENOMINATOR_BYTES
    assert rate_credit(-1) == -25.0 / RATE_DENOMINATOR_BYTES


def test_rate_credit_refuses_a_nonpositive_denominator():
    with pytest.raises(ValueError):
        rate_credit(100, denominator=0)


def test_break_even_is_the_exact_zero_of_the_score_delta():
    rate = -1.20787e-03
    even = break_even_d_pose(BASE_D_POSE_CPU_TORCH, rate)
    leg = math.sqrt(10.0 * even) - math.sqrt(10.0 * BASE_D_POSE_CPU_TORCH)
    assert abs(rate + leg) < 1e-15


def test_break_even_at_zero_rate_change_is_the_base_itself():
    assert break_even_d_pose(BASE_D_POSE_CPU_TORCH, 0.0) == pytest.approx(
        BASE_D_POSE_CPU_TORCH
    )


def test_a_byte_adding_edit_lowers_the_bar_below_the_base():
    """The sign that an earlier draft got backwards by returning inf."""
    tighter = break_even_d_pose(BASE_D_POSE_CPU_TORCH, +1e-4)
    assert 0.0 < tighter < BASE_D_POSE_CPU_TORCH


def test_added_bytes_beyond_the_whole_pose_term_leave_no_budget():
    whole_pose_term = math.sqrt(10.0 * BASE_D_POSE_CPU_TORCH)
    assert break_even_d_pose(BASE_D_POSE_CPU_TORCH, whole_pose_term * 2.0) == 0.0


def test_break_even_refuses_a_nonpositive_base():
    with pytest.raises(ValueError):
        break_even_d_pose(0.0, -1e-3)


def test_dropping_the_whole_basis_buys_the_most_pose_tolerance():
    """Monotonicity: fewer stored basis bytes, looser d_pose budget."""
    none_stored = basis_break_even_d_pose(0)
    half_stored = basis_break_even_d_pose(SHIPPED_BASIS_PAYLOAD_BYTES // 2)
    all_stored = basis_break_even_d_pose(SHIPPED_BASIS_PAYLOAD_BYTES)
    assert all_stored < half_stored < none_stored
    assert all_stored == pytest.approx(BASE_D_POSE_CPU_TORCH)


def test_the_generated_basis_budget_is_the_one_the_dct_family_missed():
    """The refusal in the second anchor, restated as arithmetic.

    Dropping the basis entirely buys a d_pose budget of order 2.6e-05.  The
    measured DCT re-solve produced an n600 lower bound of 0.9986 -- four orders
    of magnitude past it -- which is why the family is refused rather than tuned.
    """
    budget = basis_break_even_d_pose(0)
    measured_lower_bound = 0.998631
    assert budget < 1e-4
    assert measured_lower_bound / budget > 1e4


# --------------------------------------------------------------------------
# the geometry the whole equation rests on
# --------------------------------------------------------------------------


def test_basis_symbol_count_is_the_carrier_geometry():
    assert BASIS_SYMBOLS == 12 * 3 * 24 * 32 == 27_648


def test_nearly_dead_scale_field_is_recorded_and_clears_the_admit_bar_on_its_own():
    """48 recoverable bytes are worth 3.2e-05 S -- but they are not free.

    The anchor must keep saying so: an earlier draft called them "dead" and the
    render test falsified it (10/24 pairs bit-identical, 14 off by one level).
    """
    assert NEARLY_DEAD_BASIS_SCALE_BYTES == 48
    assert DEAD_BASIS_SCALE_BYTES == NEARLY_DEAD_BASIS_SCALE_BYTES
    assert rate_credit(NEARLY_DEAD_BASIS_SCALE_BYTES) > 2e-5
    equation = build_pose_carrier_basis_rate_fidelity_exchange_v1()
    anchor = equation.empirical_anchors[0]
    text = str(anchor.empirical_output)
    assert "NOT free by construction" in text
    assert "10 of 24" in text


def test_carrier_stream_is_a_real_share_of_the_archive():
    assert 0.10 < CARRIER_STREAM_BYTES / FRONTIER_ARCHIVE_BYTES < 0.15


def test_measured_rungs_are_ordered_consistently_in_payload_and_archive():
    """Smaller payload, smaller archive -- across every measured pair of rungs."""
    rungs = sorted(MEASURED_BASIS_RUNGS.values())
    for (payload_a, archive_a, *_), (payload_b, archive_b, *_) in itertools.pairwise(
        rungs
    ):
        assert payload_a <= payload_b
        assert archive_a <= archive_b
