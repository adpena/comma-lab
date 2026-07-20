from __future__ import annotations

import numpy as np

from tools.calibrate_r1b5_xi0_warp import (
    _fit_affine,
    _fit_scalar,
    _policy_shifts,
)


def _losses_for_targets(targets: np.ndarray) -> np.ndarray:
    shifts = np.arange(-16, 17, dtype=np.float64)
    return np.square(shifts[None, :] - targets[:, None])


def test_policy_shift_rules_are_receiver_exact_and_clamped() -> None:
    values = np.asarray([28.0, 31.0, 33.0, 80.0])
    np.testing.assert_array_equal(
        _policy_shifts(values, {"kind": "affine", "center": 31.0, "gain": 1.0}),
        [-3, 0, 2, 16],
    )
    np.testing.assert_array_equal(
        _policy_shifts(values, {"kind": "scalar", "gain": -0.5}),
        [-14, -16, -16, -16],
    )


def test_scalar_fit_uses_only_declared_training_rows() -> None:
    values = np.asarray([20.0, 30.0, 40.0, 50.0])
    losses = _losses_for_targets(np.asarray([2.0, 3.0, -16.0, -16.0]))
    fitted = _fit_scalar(values, losses, (0, 1))
    assert fitted["all_shifts"][:2] == [2, 3]
    assert fitted["all_shifts"][2:] != [-16, -16]


def test_affine_fit_recovers_receiver_native_center_and_gain() -> None:
    values = np.asarray([29.0, 30.0, 32.0, 33.0, 60.0])
    targets = np.asarray([-4.0, -2.0, 2.0, 4.0, -16.0])
    losses = _losses_for_targets(targets)
    fitted = _fit_affine(values, losses, (0, 1, 2, 3))
    assert fitted["center"] == 31.0
    assert fitted["all_shifts"][:4] == [-4, -2, 2, 4]
