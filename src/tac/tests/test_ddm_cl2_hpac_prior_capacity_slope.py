"""Re-derivation guards for the ddm_cl2 prior-capacity slope law.

The bytes were MEASURED by the ddm_cl2 arm through the shipped fs2 path; the memo is the
primary artifact.  These tests pin the arithmetic the headline constants are built from and
the three decisions the equation exists to make: the instrument gate on the control, cl1's
adjacent-rung break-even, and the prior-law prediction that the first rung falsified.
"""

from __future__ import annotations

import math

import pytest

from tac.canonical_equations import build_hpac_prior_capacity_slope_v1
from tac.canonical_equations.hpac_prior_capacity_slope_20260905 import (
    BREAK_EVEN_SLOPE,
    CONTROL_JOINT_DELTA_BYTES,
    CONTROL_TOLERANCE_BYTES,
    CPU_CONTROL_JOINT_DELTA_BYTES,
    EQUATION_ID,
    LAMBDA_0P5_DELTA_MODEL_BYTES,
    LAMBDA_0P5_DELTA_STREAM_BYTES,
    LAMBDA_0P5_JOINT_DELTA_VS_CONTROL_BYTES,
    LAMBDA_0P5_JOINT_DELTA_VS_SHIPPED_BYTES,
    LAMBDA_0P5_SLOPE,
    RUNG_MODEL_BYTES,
    RUNG_STREAM_BYTES,
    SHIPPED_JOINT_BYTES,
    adjacent_slope,
    control_reproduces_shipped_family,
    joint_bytes,
    next_rung_admitted,
    prior_law_prediction_holds,
    rate_only_delta_s,
    rung_pays,
)


def test_shipped_joint_and_control_deltas_are_the_measured_bytes() -> None:
    assert SHIPPED_JOINT_BYTES == 126_926
    assert joint_bytes(RUNG_MODEL_BYTES["lambda_1p0"], RUNG_STREAM_BYTES["lambda_1p0"]) == 126_885
    assert CONTROL_JOINT_DELTA_BYTES == -41
    assert CPU_CONTROL_JOINT_DELTA_BYTES == 252
    assert control_reproduces_shipped_family(126_885)
    assert control_reproduces_shipped_family(126_926 + CONTROL_TOLERANCE_BYTES)
    assert not control_reproduces_shipped_family(126_926 + CONTROL_TOLERANCE_BYTES + 1)
    # jf1's epoch-2 instrument (+7,387 B joint) would have been INSTRUMENT-REFUSED by this gate.
    assert not control_reproduces_shipped_family(126_926 + 7_387)


def test_lambda_half_secant_is_positive_and_does_not_pay() -> None:
    assert LAMBDA_0P5_DELTA_MODEL_BYTES == 350
    assert LAMBDA_0P5_DELTA_STREAM_BYTES == 156
    assert pytest.approx(156 / 350) == LAMBDA_0P5_SLOPE
    assert LAMBDA_0P5_SLOPE > 0 > BREAK_EVEN_SLOPE
    assert LAMBDA_0P5_JOINT_DELTA_VS_CONTROL_BYTES == 506
    assert LAMBDA_0P5_JOINT_DELTA_VS_SHIPPED_BYTES == 465
    assert rung_pays(LAMBDA_0P5_DELTA_STREAM_BYTES, LAMBDA_0P5_DELTA_MODEL_BYTES) is False
    assert next_rung_admitted(LAMBDA_0P5_DELTA_STREAM_BYTES, LAMBDA_0P5_DELTA_MODEL_BYTES) is False


def test_break_even_arithmetic_on_synthetic_rungs() -> None:
    # +1,000 B model repaid by -2,500 B stream: slope -2.5 pays.
    assert adjacent_slope(-2_500, 1_000) == -2.5
    assert rung_pays(-2_500, 1_000) is True
    # +1,000 B model repaid by only -600 B stream: slope -0.6 does not pay.
    assert rung_pays(-600, 1_000) is False
    # Exactly break-even does not pay (strict inequality).
    assert rung_pays(-1_000, 1_000) is False
    # No model growth: pays iff the joint fell.
    assert adjacent_slope(-10, 0) is None
    assert rung_pays(-10, 0) is True
    assert rung_pays(+10, -5) is False
    assert rung_pays(-10, -5) is True


def test_prior_law_prediction_is_falsified_by_the_measured_rung() -> None:
    assert prior_law_prediction_holds(1_000, -2_500) is True
    assert prior_law_prediction_holds(1_500, -3_000) is True
    assert prior_law_prediction_holds(1_600, -4_000) is False  # model grew past the cap
    assert prior_law_prediction_holds(1_000, -1_900) is False  # stream saving under 2x
    assert prior_law_prediction_holds(LAMBDA_0P5_DELTA_MODEL_BYTES, LAMBDA_0P5_DELTA_STREAM_BYTES) is False


def test_rate_only_delta_s_of_the_control_candidate() -> None:
    assert rate_only_delta_s(-41) == pytest.approx(-41 * 25.0 / 37_545_489)
    assert math.isclose(rate_only_delta_s(-41), -2.7300217078009027e-05, rel_tol=1e-12)


def test_builder_carries_verified_anchors_with_re_derivable_residuals() -> None:
    """ddm_cl2 landed two anchors; ddm_cl3 appended the smaller-model side (append-only)."""

    equation = build_hpac_prior_capacity_slope_v1()
    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 3
    control, slope, smaller = equation.empirical_anchors
    assert control.residual == 41.0
    assert slope.residual == 506.0 + 1_500.0
    # cl3: measured net +224 B against a prediction whose BEST case was -50 B.
    assert smaller.residual == 224.0 - (-50.0)
    assert set(equation.predicted_vs_empirical_residual) == {
        control.anchor_id,
        slope.anchor_id,
        smaller.anchor_id,
    }
    assert "capacity" in equation.name.lower()


def test_the_ladder_is_now_closed_on_BOTH_sides_of_lambda_equals_one() -> None:
    """The two secants bracket lambda = 1.0, and both neighbours cost bytes.

    cl2 measured the bigger-model side (+506 B) on the fs2/Brotli object; cl3 measured the
    smaller-model side (+224 B) on the rc1-coded successor.  Together they make lambda = 1.0 a
    local optimum, which is a strictly stronger statement than either row alone.
    """

    equation = build_hpac_prior_capacity_slope_v1()
    _control, slope, smaller = equation.empirical_anchors
    bigger_side = slope.empirical_output["joint_delta_vs_control"]
    smaller_side = smaller.empirical_output["joint_delta_vs_control"]
    assert bigger_side > 0 and smaller_side > 0
    assert not rung_pays(
        smaller.empirical_output["delta_stream_bytes"],
        smaller.empirical_output["delta_model_bytes"],
    )
    # The cl3 anchor must say out loud which object it was priced on -- the coder changed.
    assert "rc1" in smaller.inputs["object"].lower()
    assert smaller.empirical_output["lambda_4p0_fired"] is False
