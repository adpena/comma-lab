# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.analysis.receiver_surface_metrics import (
    normalize_receiver_surface,
    receiver_surface_receiver_visible,
    receiver_surface_scorer_visible,
    receiver_surface_survival_state,
    receiver_surface_target_support_breakdown,
    surface_value,
)


def test_receiver_surface_normalizes_aliases_to_canonical_keys() -> None:
    surface = normalize_receiver_surface(
        {
            "uint8_changed_pixels": 9,
            "target_hard_won_count": 4,
            "target_hard_lost_count": 1,
            "fakequant_survival": True,
        }
    )

    assert surface["receiver_surface_uint8_changed_pixels"] == 9.0
    assert surface["receiver_surface_target_hard_won_count"] == 4.0
    assert surface["receiver_surface_target_hard_lost_count"] == 1.0
    assert surface["receiver_surface_fakequant_survival"] is True
    assert (
        surface_value(
            surface,
            "receiver_surface_uint8_changed_pixels",
        )
        == 9.0
    )


def test_receiver_surface_distinguishes_preprocess_from_scorer_motion() -> None:
    preprocess_only = normalize_receiver_surface(
        {
            "uint8_changed_pixels": 5,
            "segnet_input_delta_linf": 0.05,
        }
    )
    target_birth = normalize_receiver_surface(
        {
            "uint8_changed_pixels": 5,
            "target_hard_won_count": 2,
            "net_target_support_delta": 2,
        }
    )

    assert receiver_surface_receiver_visible(preprocess_only) is True
    assert receiver_surface_scorer_visible(preprocess_only) is False
    assert receiver_surface_receiver_visible(target_birth) is True
    assert receiver_surface_scorer_visible(target_birth) is True


def test_receiver_surface_derives_net_target_support_when_absent() -> None:
    surface = normalize_receiver_surface(
        {
            "target_hard_won_count": 7,
            "target_hard_lost_count": 2,
        }
    )

    support = receiver_surface_target_support_breakdown(surface)

    assert support["receiver_surface_target_hard_won_count"] == 7.0
    assert support["receiver_surface_target_hard_lost_count"] == 2.0
    assert support["receiver_surface_net_target_support_delta"] == 5.0


def test_receiver_surface_survival_state_reports_conflicts() -> None:
    survived, blockers = receiver_surface_survival_state(
        "parseback",
        {"parseback_survived": False},
        {"receiver_surface_parseback_survival": True},
        blocker_prefix="servo_lift",
    )

    assert survived is False
    assert blockers == ["servo_lift_parseback_survival_conflict"]
