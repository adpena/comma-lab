"""Guards for ``coder_strength_substitutes_for_capacity_v1`` (ddm_cl3, 2026-09-06).

The law says a STRONGER model coder returns LESS for the same weight shrink, so it makes every
capacity rung measured through model bytes worth less.  These guards re-derive the measured
numbers rather than restating them, and pin the two things a future reader could get wrong: that
the stream tax is coder-independent, and that the discount runs the right way.
"""

from __future__ import annotations

import pytest

from tac.canonical_equations import (
    build_coder_strength_substitutes_for_capacity_v1,
    coder_capture_fraction,
    substitution_holds,
    transfer_capacity_delta,
)
from tac.canonical_equations.coder_strength_substitutes_for_capacity_20260906 import (
    CAPTURE_FRACTION,
    CONTROL_MODEL_BYTES_STRONG,
    CONTROL_MODEL_BYTES_WEAK,
    CONTROL_STREAM_BYTES,
    CONTROL_STREAM_BYTES_ON_WEAK_CODER_TREE,
    DELTA_JOINT_STRONG,
    DELTA_JOINT_WEAK,
    DELTA_MODEL_STRONG,
    DELTA_MODEL_WEAK,
    DELTA_STREAM,
    RUNG_MODEL_BYTES_STRONG,
    RUNG_MODEL_BYTES_WEAK,
    RUNG_STREAM_BYTES,
    joint_delta,
    rung_pays,
)


def test_measured_deltas_re_derive_from_the_container_bytes() -> None:
    """Every delta is recomputed from the two container measurements, never restated."""

    assert DELTA_MODEL_WEAK == RUNG_MODEL_BYTES_WEAK - CONTROL_MODEL_BYTES_WEAK == -659
    assert DELTA_MODEL_STRONG == RUNG_MODEL_BYTES_STRONG - CONTROL_MODEL_BYTES_STRONG == -457
    assert DELTA_STREAM == RUNG_STREAM_BYTES - CONTROL_STREAM_BYTES == 681
    assert DELTA_JOINT_WEAK == joint_delta(DELTA_MODEL_WEAK, DELTA_STREAM) == 22
    assert DELTA_JOINT_STRONG == joint_delta(DELTA_MODEL_STRONG, DELTA_STREAM) == 224


def test_the_rung_flips_verdict_between_the_two_coders() -> None:
    """The whole point: break-even on the weak coder, a loss on the object that ships."""

    assert not rung_pays(DELTA_MODEL_WEAK, DELTA_STREAM)
    assert not rung_pays(DELTA_MODEL_STRONG, DELTA_STREAM)
    # ... and the strong coder is strictly worse for the same weights.
    assert DELTA_JOINT_STRONG > DELTA_JOINT_WEAK
    # A 1 B larger model saving on the weak basis would have made it pay; on the strong basis
    # it would still lose by 201 B.  That is the margin the coder consumed.
    assert joint_delta(DELTA_MODEL_WEAK - 23, DELTA_STREAM) < 0
    assert joint_delta(DELTA_MODEL_STRONG - 23, DELTA_STREAM) > 0


def test_the_stream_tax_is_coder_independent() -> None:
    """The stream is a property of the WEIGHTS; only the model side moves with the coder.

    If a future edit ever gives the two bases different stream deltas, the law's premise is gone
    and this guard must fail rather than let the conclusion stand on a broken leg.
    """

    # MEASURED, not assumed: the same control weights gave 113,419 B on cl2's fs2 tree and on the
    # live pc1 tree, which differ in both model coder and carrier.
    assert CONTROL_STREAM_BYTES_ON_WEAK_CODER_TREE == CONTROL_STREAM_BYTES == 113_419
    weak_stream = RUNG_STREAM_BYTES - CONTROL_STREAM_BYTES_ON_WEAK_CODER_TREE
    strong_stream = RUNG_STREAM_BYTES - CONTROL_STREAM_BYTES
    assert weak_stream == strong_stream == DELTA_STREAM
    assert DELTA_JOINT_STRONG - DELTA_JOINT_WEAK == DELTA_MODEL_STRONG - DELTA_MODEL_WEAK


def test_substitution_holds_and_the_capture_fraction_is_below_one() -> None:
    assert substitution_holds(abs(DELTA_MODEL_STRONG), abs(DELTA_MODEL_WEAK))
    assert CAPTURE_FRACTION == pytest.approx(457 / 659, rel=1e-12)
    assert 0.0 < CAPTURE_FRACTION < 1.0
    assert coder_capture_fraction(457.0, 659.0) == pytest.approx(CAPTURE_FRACTION, rel=1e-12)
    # A coder that returned MORE would falsify the law, and the predicate must say so.
    assert not substitution_holds(700.0, 659.0)


def test_transfer_rule_discounts_downward_and_refuses_nonsense() -> None:
    """The transfer rule must shrink a weak-basis saving, never inflate it."""

    assert transfer_capacity_delta(659.0) == pytest.approx(abs(DELTA_MODEL_STRONG), abs=1.0)
    assert transfer_capacity_delta(1000.0) < 1000.0
    with pytest.raises(ValueError):
        transfer_capacity_delta(1000.0, capture_fraction=1.5)
    with pytest.raises(ValueError):
        transfer_capacity_delta(1000.0, capture_fraction=0.0)
    with pytest.raises(ValueError):
        coder_capture_fraction(1.0, 0.0)


def test_equation_carries_its_anchor_scope_and_residual() -> None:
    equation = build_coder_strength_substitutes_for_capacity_v1()
    assert equation.equation_id == "coder_strength_substitutes_for_capacity_v1"
    (anchor,) = equation.empirical_anchors
    # P8 predicted the strong coder would EXCEED the weak one; it fell short by 202 B.
    assert anchor.residual == pytest.approx(202.0)
    assert anchor.empirical_output["delta_joint_weak_bytes"] == 22
    assert anchor.empirical_output["delta_joint_strong_bytes"] == 224
    # The field is held, so the law must never be readable as a distortion or score claim.
    excluded = " ".join(equation.domain_of_validity["excluded"]).lower()
    assert "d_seg" in excluded and "d_pose" in excluded
    assert "instance" in equation.domain_of_validity["verdict_scope"].lower()
    assert equation.canonical_producers
