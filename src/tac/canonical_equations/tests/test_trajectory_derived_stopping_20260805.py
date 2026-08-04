# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.canonical_equations.trajectory_derived_stopping_20260805 import (
    EQUATION_ID,
    build_trajectory_derived_stopping_law_v1,
    sq1_prefix25_projection_interval,
)


def test_sq1_prefix25_embedded_anchor_reports_safety_bound_not_convergence() -> None:
    payload = sq1_prefix25_projection_interval()
    assert payload["decision"]["stop_reason"] == "safety_bound_REPORTED"
    assert set(payload["projection"]["fits_used"]) == {"geometric", "power_law"}
    assert payload["projection"]["objective_low"] <= 6_861 <= payload["projection"]["objective_high"]


def test_build_equation_is_non_promoting_and_points_to_executor() -> None:
    equation = build_trajectory_derived_stopping_law_v1(
        source_receipt=".omx/research/ddm_tj1_20260805/trajectory_replay.json"
    )
    assert equation.equation_id == EQUATION_ID
    assert equation.python_callable_module_path.endswith(":evaluate_trajectory_stop")
    assert equation.provenance.promotion_eligible is False
    assert equation.provenance.score_claim_valid is False
    assert equation.empirical_anchors[0].empirical_output["inside_interval"] is True
