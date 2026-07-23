from __future__ import annotations

import math

import pytest

from tac.optimization.pure_priced_realized_objective import (
    SOURCE_VIDEO_BYTES,
    PurePricedObjectiveError,
    RealizedObjectiveState,
    break_even_distortion_gain_per_byte,
    pure_priced_realized_delta,
)


def test_exact_v17_405_flip_row_is_admitted_without_cap() -> None:
    before = RealizedObjectiveState(0.025053024292, 162.796878513138, 135_328)
    after = RealizedObjectiveState(0.025002797445, 162.797857368493, 135_529)
    row = pure_priced_realized_delta(before, after)
    assert row.seg_term == pytest.approx(-0.0050226847)
    assert row.pose_term > 0.0
    assert row.rate_term == pytest.approx(25 * 201 / SOURCE_VIDEO_BYTES)
    # The receipt publishes d_seg/d_pose at 12 decimals while its stored joint
    # delta was computed from the unrounded arrays.
    assert row.joint_delta == pytest.approx(-0.004767545957001573, abs=5e-11)
    assert row.accepted is True


def test_zero_delta_is_not_strictly_admitted() -> None:
    state = RealizedObjectiveState(0.1, 0.2, 100)
    row = pure_priced_realized_delta(state, state)
    assert row.joint_delta == 0.0
    assert row.accepted is False


@pytest.mark.parametrize(
    "state",
    [
        (math.nan, 0.0, 0),
        (0.0, math.inf, 0),
        (-1.0, 0.0, 0),
        (0.0, 0.0, -1),
        (0.0, 0.0, True),
    ],
)
def test_invalid_measurements_fail_closed(state: tuple[float, float, int]) -> None:
    with pytest.raises(PurePricedObjectiveError):
        RealizedObjectiveState(*state)


def test_registered_rate_break_even_is_exact() -> None:
    assert break_even_distortion_gain_per_byte() == 25 / 37_545_489
