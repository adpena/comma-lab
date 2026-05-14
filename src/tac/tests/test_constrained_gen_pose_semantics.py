# SPDX-License-Identifier: MIT
"""Semantic-label regression tests for constrained-gen pose heuristics."""

from __future__ import annotations

import torch

from tac.constrained_gen import estimate_expected_pose


def _mask_pair_with_moving_my_car() -> torch.Tensor:
    masks = torch.zeros(2, 4, 4, dtype=torch.long)
    masks[0, 0, 0] = 4
    masks[1, 2, 2] = 4
    return masks


def _only_my_car_pose_weights(**overrides: float) -> dict[str, float]:
    weights = {
        "tx_road_dx": 0.0,
        "ty_road_dy": 0.0,
        "tz_baseline": 0.0,
        "tz_road_dy": 0.0,
        "rx_my_car_dy": 1.0,
        "ry_my_car_dx": 1.0,
        "rz_road_cx": 0.0,
    }
    weights.update(overrides)
    return weights


def test_estimate_expected_pose_uses_canonical_my_car_class_for_rotation_proxy() -> None:
    poses = estimate_expected_pose(
        _mask_pair_with_moving_my_car(),
        pose_heuristic_weights=_only_my_car_pose_weights(),
    )

    assert poses.shape == (1, 6)
    assert torch.allclose(poses[0, 3], torch.tensor(0.5))
    assert torch.allclose(poses[0, 4], torch.tensor(0.5))


def test_estimate_expected_pose_accepts_legacy_sky_aliases_without_relabeling_class4() -> None:
    legacy_weights = {
        "tx_road_dx": 0.0,
        "ty_road_dy": 0.0,
        "tz_baseline": 0.0,
        "tz_road_dy": 0.0,
        "rx_sky_dy": 2.0,
        "ry_sky_dx": 3.0,
        "rz_road_cx": 0.0,
    }
    poses = estimate_expected_pose(
        _mask_pair_with_moving_my_car(),
        pose_heuristic_weights=legacy_weights,
    )

    assert torch.allclose(poses[0, 3], torch.tensor(1.0))
    assert torch.allclose(poses[0, 4], torch.tensor(1.5))
