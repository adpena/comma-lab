# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.local_acceleration.argmax_tie_snap import (
    DYADIC_TIE_EPSILONS,
    class_pair_tie_snap_argmax_numpy,
    epsilon_arm_name,
    tie_snap_argmax_numpy,
)


def test_exact_tie_uses_lowest_class_like_torch_argmax() -> None:
    logits = np.asarray([[[5.0, -1.0, 5.0]]], dtype=np.float32)
    observed = tie_snap_argmax_numpy(logits, epsilon=0.0)
    np.testing.assert_array_equal(observed, np.asarray([[0]], dtype=np.int64))


def test_epsilon_snaps_only_after_preregistered_gap_is_reached() -> None:
    logits = np.asarray([[[1.0, 1.0 + 2.0**-20]]], dtype=np.float32)
    below = tie_snap_argmax_numpy(logits, epsilon=2.0**-21)
    above = tie_snap_argmax_numpy(logits, epsilon=2.0**-19)
    assert int(below[0, 0]) == 1
    assert int(above[0, 0]) == 0


def test_nchw_class_axis_and_ladder_names() -> None:
    logits = np.asarray([[[[2.0]], [[2.0]], [[-3.0]]]], dtype=np.float32)
    observed = tie_snap_argmax_numpy(logits, epsilon=0.0, class_axis=1)
    np.testing.assert_array_equal(observed, np.asarray([[[0]]], dtype=np.int64))
    assert epsilon_arm_name(DYADIC_TIE_EPSILONS[0]) == "epsilon_zero"
    assert epsilon_arm_name(2.0**-19) == "epsilon_2m19"


def test_invalid_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        tie_snap_argmax_numpy(np.asarray([[np.nan]], dtype=np.float32), epsilon=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        tie_snap_argmax_numpy(np.asarray([[1.0]], dtype=np.float32), epsilon=-1.0)
    with pytest.raises(ValueError, match="preregistered"):
        epsilon_arm_name(1.0e-5)


def test_class_pair_snap_changes_only_preregistered_ordered_pair() -> None:
    epsilon = 2.0**-19
    logits = np.asarray(
        [
            [
                [1.0, 1.0, -3.0, -4.0, 1.0 + 2.0**-20],
                [1.0, 1.0 + 2.0**-20, -3.0, -4.0, -5.0],
                [1.0, -2.0, -3.0, -4.0, 1.0 + 2.0**-18],
            ]
        ],
        dtype=np.float32,
    )
    observed = class_pair_tie_snap_argmax_numpy(
        logits,
        epsilon=epsilon,
        winner_class=4,
        runner_class=0,
    )
    np.testing.assert_array_equal(observed, np.asarray([[0, 1, 4]], dtype=np.int64))


def test_class_pair_snap_uses_deterministic_runner_and_validates_classes() -> None:
    logits = np.asarray([[[0.0, 0.0, -2.0, -3.0, 2.0**-20]]], dtype=np.float32)
    observed = class_pair_tie_snap_argmax_numpy(
        logits,
        epsilon=2.0**-19,
        winner_class=4,
        runner_class=0,
    )
    assert int(observed[0, 0]) == 0
    with pytest.raises(ValueError, match="must differ"):
        class_pair_tie_snap_argmax_numpy(
            logits,
            epsilon=0.0,
            winner_class=0,
            runner_class=0,
        )
    with pytest.raises(ValueError, match="out of range"):
        class_pair_tie_snap_argmax_numpy(
            logits,
            epsilon=0.0,
            winner_class=5,
            runner_class=0,
        )
