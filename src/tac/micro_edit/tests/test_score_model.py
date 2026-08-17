# SPDX-License-Identifier: MIT
"""Tests for the ddm_me1 exact-arithmetic score model.

These test BEHAVIOUR against independently-derived receipts, not constants:

* the live rr4 base must reconstruct to its published 17 digits;
* the model must independently reproduce eu4's qs2 net delta (-4.374914e-6),
  which eu4 derived by a different route;
* the union-projection helper must NEVER return a realized delta (the eu4
  union-gating law + the qs4 cross-object compensation refusal).
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from tac.contest_oracle.constants import CONTEST_PER_ARCHIVE_PIXEL_CELLS
from tac.micro_edit.score_model import (
    BREAKEVEN_FLIPS_PER_BYTE,
    CANONICAL_NOISE_BAND_S,
    NAMING_BAR_S,
    RATE_PER_BYTE_S,
    SEG_PER_FLIP_S,
    ScoreDelta,
    ScoreState,
    compose_deltas_unverified,
    pose_marginal,
)

# The LIVE frontier (ddm_rr4_t4_verdict_pointer_move_20260817.md).
RR4_D_SEG = Decimal("0.00029611")
RR4_D_POSE = Decimal("6.88e-06")
RR4_BYTES = 181161
RR4_SCORE = Decimal("0.15853325034789678")
RR4_SHA = "35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956"


def rr4_base() -> ScoreState:
    return ScoreState(
        d_seg=RR4_D_SEG,
        d_pose=RR4_D_POSE,
        archive_bytes=RR4_BYTES,
        label="[contest-CUDA T4 n600] rr4",
        archive_sha256=RR4_SHA,
    )


def test_live_base_reconstructs_to_published_precision() -> None:
    """The published S must fall out of the components, not be typed in."""
    assert abs(rr4_base().score - RR4_SCORE) < Decimal("1e-16")


def test_marginals_match_closed_forms() -> None:
    assert Decimal(100) / Decimal(117_964_800) == SEG_PER_FLIP_S
    assert Decimal(25) / Decimal(37_545_489) == RATE_PER_BYTE_S
    # break-even: how many flips one byte must buy
    assert abs(BREAKEVEN_FLIPS_PER_BYTE - Decimal("0.7854791823")) < Decimal("1e-10")


def test_pose_marginal_is_state_dependent_and_diverges() -> None:
    """A latched pose marginal is the cross-regime transfer bug -- prove it moves."""
    near = pose_marginal(Decimal("1e-8"))
    far = pose_marginal(Decimal("1e-4"))
    assert near > far * 50
    with pytest.raises(ValueError):
        pose_marginal(Decimal(0))


def test_reproduces_eu4_qs2_net_delta_independently() -> None:
    """eu4 derived qs2 net = -4.374914e-6 by its own route; we must agree."""
    base = rr4_base()
    d_seg_delta = Decimal(-32) / Decimal(CONTEST_PER_ARCHIVE_PIXEL_CELLS)
    d_pose_delta = Decimal("1.126177e-7") / base.pose_marginal
    delta = ScoreDelta(
        d_seg_delta=d_seg_delta,
        d_pose_delta=d_pose_delta,
        bytes_delta=34,
        base=base,
        realized=True,
        provenance="qs2_banked_old_coder",
    )
    assert abs(delta.delta_s - Decimal("-4.374914e-6")) < Decimal("1e-11")
    assert delta.net_seg_flips == Decimal(-32)


def test_axis_contributions_sum_to_total() -> None:
    base = rr4_base()
    delta = ScoreDelta(
        d_seg_delta=Decimal("-5e-8"),
        d_pose_delta=Decimal("2e-9"),
        bytes_delta=-17,
        base=base,
        realized=True,
        provenance="synthetic",
    )
    parts = delta.axis_contributions()
    assert abs(parts["total"] - delta.delta_s) < Decimal("1e-40")
    assert abs(parts["seg"] + parts["pose"] + parts["rate"] - delta.delta_s) < Decimal("1e-40")


def test_delta_s_is_nonlinear_in_pose_not_a_linearisation() -> None:
    """A first-order pose estimate is wrong where pose dominates -- prove we differ.

    sqrt(10x) is CONCAVE, so its tangent lies above it: for a pose REDUCTION the
    true gain is LARGER in magnitude than the marginal-times-delta estimate. The
    marginal diverges as d_pose falls, so linearising a pose move systematically
    UNDER-sells it. The engine therefore always evaluates the true difference.
    """
    base = rr4_base()
    big = Decimal("-5e-6")  # most of d_pose removed
    delta = ScoreDelta(Decimal(0), big, 0, base, True, "pose_only")
    linear = base.pose_marginal * big
    assert delta.delta_s != linear
    assert delta.delta_s < linear  # more negative == bigger real gain
    # and a pose INCREASE costs LESS than the linear estimate threatens
    up = ScoreDelta(Decimal(0), Decimal("5e-6"), 0, base, True, "pose_up")
    assert up.delta_s < base.pose_marginal * Decimal("5e-6")


def test_axis_split_recomputes_residual_per_base() -> None:
    """The post-pose residual must be computed from the base, never quoted."""
    split = rr4_base().axis_split(Decimal("0.15"))
    assert split["pose_share_of_gap"] > Decimal("0.97")
    assert Decimal(281) < split["residual_in_seg_flips"] < Decimal(282)
    assert Decimal(358) < split["residual_in_bytes"] < Decimal(359)
    # a different base must give a different residual (no latched constant)
    cheaper = ScoreState(RR4_D_SEG, RR4_D_POSE, RR4_BYTES - 5000, "cheaper")
    assert cheaper.axis_split(Decimal("0.15"))["residual_in_bytes"] < split["residual_in_bytes"]


def test_bars_classify_correctly() -> None:
    base = rr4_base()
    tiny = ScoreDelta(Decimal("-1e-9"), Decimal(0), 0, base, True, "tiny")
    assert tiny.inside_noise_band
    assert not tiny.clears_naming_bar
    big = ScoreDelta(Decimal("-2e-7"), Decimal(0), 0, base, True, "big")
    assert big.clears_naming_bar
    assert abs(big.delta_s) > NAMING_BAR_S > CANONICAL_NOISE_BAND_S


def test_union_projection_is_never_realized() -> None:
    """The eu4 union-gating law, enforced in code."""
    base = rr4_base()
    a = ScoreDelta(Decimal("-1e-8"), Decimal(0), 0, base, True, "a")
    b = ScoreDelta(Decimal("-1e-8"), Decimal(0), 3, base, True, "b")
    union = compose_deltas_unverified([a, b], base)
    assert union.realized is False
    assert "PROJECTION_ONLY" in union.provenance
    assert union.bytes_delta == 3


def test_union_refuses_mixed_bases() -> None:
    """qs4's anti-law: an object solved elsewhere may not be summed in here."""
    base = rr4_base()
    other = ScoreState(RR4_D_SEG, RR4_D_POSE, 186252, "[contest-CUDA T4 n600] cp135")
    a = ScoreDelta(Decimal("-1e-8"), Decimal(0), 0, base, True, "a")
    b = ScoreDelta(Decimal("-1e-8"), Decimal(0), 0, other, True, "b")
    with pytest.raises(ValueError, match="different bases"):
        compose_deltas_unverified([a, b], base)


def test_state_rejects_unlabelled_or_impossible_states() -> None:
    with pytest.raises(ValueError):
        ScoreState(RR4_D_SEG, RR4_D_POSE, RR4_BYTES, "")
    with pytest.raises(ValueError):
        ScoreState(Decimal(-1), RR4_D_POSE, RR4_BYTES, "x")
    with pytest.raises(ValueError):
        ScoreState(RR4_D_SEG, RR4_D_POSE, 0, "x")
