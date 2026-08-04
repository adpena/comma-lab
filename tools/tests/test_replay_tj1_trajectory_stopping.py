# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

from tools.replay_tj1_trajectory_stopping import (
    _aggregate_curve,
    _eta_interval,
    _ng1_class_map,
)


def test_aggregate_curve_uses_only_steps_present_for_every_row() -> None:
    receipt = {
        "rows": [
            {"solved_convergence_curve": [
                {"step": 0, "proxy_flips": 10},
                {"step": 5, "proxy_flips": 7},
                {"step": 10, "proxy_flips": 6},
            ]},
            {"solved_convergence_curve": [
                {"step": 0, "proxy_flips": 20},
                {"step": 5, "proxy_flips": 14},
                {"step": 15, "proxy_flips": 1},
            ]},
        ]
    }
    assert _aggregate_curve(receipt) == [
        {"step": 0.0, "objective": 30.0},
        {"step": 5.0, "objective": 21.0},
    ]


def test_eta_interval_keeps_order_after_objective_interval_conversion() -> None:
    interval = _eta_interval(
        interval_low_objective=6_786.0,
        interval_high_objective=8_371.0,
        flips_before=27_055,
        described_in_band=23_450,
    )
    assert interval["eta_low"] < interval["eta_high"]
    assert interval["eta_low"] < 0.8620042643923241 < interval["eta_high"]


def test_ng1_class_map_classifies_floor_rows_and_genuine_pose_row() -> None:
    rows = [
        {
            "record_type": "cap_artifact_row",
            "id": "sq1_50_step_uncap_cw1",
            "regrade": "FLOOR-NOT-CONVERGED",
        },
        {
            "record_type": "cap_artifact_row",
            "id": "gn_pose_solve_850",
            "regrade": "GENUINE",
            "current_live_disposition": "FOLDED_STALE_OFF_CHAIN",
        },
    ]
    got = _ng1_class_map(json.loads(json.dumps(rows)))
    assert got[0]["family"] == "cap_bound_floor_not_converged"
    assert got[0]["recipient"] == "sq1 solved-paint loop"
    assert got[1]["family"] == "genuine_stop_or_stale_off_chain"
    assert got[1]["recipient"] == "terminal_pose_gn marginal floor"
