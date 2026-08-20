# SPDX-License-Identifier: MIT

from tools.hunt_ddm_pf3b_joint_improving_edge import (
    neighborhood_identities,
    prescore_rank_key,
)


def _rank_row(
    coordinate: str,
    *,
    event_delta: int,
    outside: int,
    support_outside: int,
    pose_trace: float,
) -> dict:
    return {
        "coordinate_id": coordinate,
        "direction_id": "POSITIVE_ONE_QUANTUM",
        "pf2_ms6_event_direction": {
            "event_delta_errors": event_delta,
            "unique_changed_event_count": 3,
        },
        "joint_spill_guard": {
            "changed_argmax_cells_outside_pf2_ms6_event_union": outside,
            "composite_r_support_cells": support_outside,
            "at1x_camera_input_pose_gram_trace": pose_trace,
        },
    }


def test_prescore_rank_is_correction_first_then_spill_guard() -> None:
    correcting = _rank_row(
        "rg3.finer_event.pair001.class0_1.boundary.static_in_image.band00.fine00.mag1",
        event_delta=-1,
        outside=20,
        support_outside=30,
        pose_trace=4.0,
    )
    neutral = _rank_row(
        "rg3.finer_event.pair002.class0_1.boundary.static_in_image.band00.fine00.mag1",
        event_delta=0,
        outside=0,
        support_outside=0,
        pose_trace=0.0,
    )
    lower_spill = _rank_row(
        "rg3.finer_event.pair003.class0_1.boundary.static_in_image.band00.fine00.mag1",
        event_delta=-1,
        outside=2,
        support_outside=3,
        pose_trace=1.0,
    )
    assert sorted(
        [neutral, correcting, lower_spill],
        key=prescore_rank_key,
    ) == [lower_spill, correcting, neutral]


def test_neighborhood_is_sign_twin_and_adjacent_magnitude_only() -> None:
    prefix = (
        "rg3.finer_event.pair001.class0_1.boundary.static_in_image."
        "band00.fine00"
    )
    inventory = [
        {"receiver_actuator_id": f"{prefix}.mag1", "direction_id": direction}
        for direction in ("NEGATIVE_ONE_QUANTUM", "POSITIVE_ONE_QUANTUM")
    ] + [
        {"receiver_actuator_id": f"{prefix}.mag2", "direction_id": direction}
        for direction in ("NEGATIVE_ONE_QUANTUM", "POSITIVE_ONE_QUANTUM")
    ] + [
        {
            "receiver_actuator_id": (
                "rg3.finer_event.pair002.class0_1.boundary.static_in_image."
                "band00.fine00.mag1"
            ),
            "direction_id": "POSITIVE_ONE_QUANTUM",
        }
    ]
    winner = {
        "coordinate_id": f"{prefix}.mag1",
        "direction_id": "POSITIVE_ONE_QUANTUM",
    }
    assert neighborhood_identities(winner, inventory) == {
        (f"{prefix}.mag1", "NEGATIVE_ONE_QUANTUM"),
        (f"{prefix}.mag1", "POSITIVE_ONE_QUANTUM"),
        (f"{prefix}.mag2", "NEGATIVE_ONE_QUANTUM"),
        (f"{prefix}.mag2", "POSITIVE_ONE_QUANTUM"),
    }
