# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral_autopilot.operator_correction_meta_pattern (GAP 2)."""
from __future__ import annotations

import pytest

from tac.cathedral_autopilot.operator_correction_meta_pattern import (
    OperatorBindingCorrectionRegistration,
    register_operator_binding_correction,
)


def test_registration_construct_happy_path() -> None:
    reg = OperatorBindingCorrectionRegistration(
        operator_quote="ensure no signal loss",
        anti_pattern_id="manual_main_thread_orchestrator_ranking_drift_v1",
        equation_id="meta_orchestrator_highest_ev_shortest_wall_clock_metric_v1",
        mission_predicted_contribution="apparatus_maintenance",
        registered_at_utc="2026-05-29T00:00:00Z",
        rationale="happy path test",
        canonical_unwind_path="Route through canonical helper",
    )
    assert reg.operator_quote == "ensure no signal loss"
    assert reg.mission_predicted_contribution == "apparatus_maintenance"


def test_registration_refuses_empty_quote() -> None:
    with pytest.raises(ValueError, match="operator_quote"):
        OperatorBindingCorrectionRegistration(
            operator_quote="",
            anti_pattern_id="x_v1",
            equation_id="y_v1",
            mission_predicted_contribution="apparatus_maintenance",
            registered_at_utc="2026-05-29T00:00:00Z",
            rationale="x",
            canonical_unwind_path="x",
        )


def test_registration_refuses_placeholder_quote() -> None:
    """Per Catalog #287 sister discipline."""
    for placeholder in ("<rationale>", "<reason>", "rationale", "reason"):
        with pytest.raises(ValueError, match="placeholder"):
            OperatorBindingCorrectionRegistration(
                operator_quote=placeholder,
                anti_pattern_id="x_v1",
                equation_id="y_v1",
                mission_predicted_contribution="apparatus_maintenance",
                registered_at_utc="2026-05-29T00:00:00Z",
                rationale="x",
                canonical_unwind_path="x",
            )


def test_registration_refuses_short_quote() -> None:
    with pytest.raises(ValueError, match=">= 4 chars"):
        OperatorBindingCorrectionRegistration(
            operator_quote="abc",  # 3 chars
            anti_pattern_id="x_v1",
            equation_id="y_v1",
            mission_predicted_contribution="apparatus_maintenance",
            registered_at_utc="2026-05-29T00:00:00Z",
            rationale="x",
            canonical_unwind_path="x",
        )


def test_registration_refuses_invalid_mission_contribution() -> None:
    with pytest.raises(ValueError, match="mission_predicted_contribution"):
        OperatorBindingCorrectionRegistration(
            operator_quote="real quote",
            anti_pattern_id="x_v1",
            equation_id="y_v1",
            mission_predicted_contribution="bogus_enum",
            registered_at_utc="2026-05-29T00:00:00Z",
            rationale="x",
            canonical_unwind_path="x",
        )


def test_registration_as_dict_round_trip() -> None:
    reg = OperatorBindingCorrectionRegistration(
        operator_quote="real quote",
        anti_pattern_id="x_v1",
        equation_id="y_v1",
        mission_predicted_contribution="frontier_protecting",
        registered_at_utc="2026-05-29T00:00:00Z",
        rationale="x",
        canonical_unwind_path="x",
        sister_consumer_module_path="tac.cathedral_consumers.test_consumer",
    )
    d = reg.as_dict()
    assert d["operator_quote"] == "real quote"
    assert d["mission_predicted_contribution"] == "frontier_protecting"
    assert d["sister_consumer_module_path"] == "tac.cathedral_consumers.test_consumer"


def test_register_helper_basic_invocation() -> None:
    reg = register_operator_binding_correction(
        operator_quote="ensure no signal loss",
        anti_pattern_id="manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1",
        equation_id="meta_orchestrator_highest_ev_shortest_wall_clock_metric_v1",
        canonical_unwind_path="Route through canonical helper",
    )
    assert reg.operator_quote == "ensure no signal loss"
    assert reg.registered_at_utc.endswith("Z")
    assert "Operator binding correction registered" in reg.rationale


def test_register_helper_fires_recalibrator_by_default() -> None:
    reg = register_operator_binding_correction(
        operator_quote="ensure no signal loss",
        anti_pattern_id="manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1",
        equation_id="meta_orchestrator_highest_ev_shortest_wall_clock_metric_v1",
        canonical_unwind_path="Route through canonical helper",
        auto_fire_recalibrator=True,
    )
    # The recalibrator fires; rationale mentions Catalog #371
    assert "Catalog #371" in reg.rationale


def test_register_helper_records_sister_consumer_routing() -> None:
    reg = register_operator_binding_correction(
        operator_quote="ensure no signal loss",
        anti_pattern_id="x_v1",
        equation_id="y_v1",
        canonical_unwind_path="x",
        sister_consumer_module_path="tac.cathedral_consumers.meta_orchestrator_consumer",
    )
    assert "Sister cathedral consumer routing" in reg.rationale
    assert "tac.cathedral_consumers.meta_orchestrator_consumer" in reg.rationale


def test_register_helper_handles_skip_recalibrator() -> None:
    reg = register_operator_binding_correction(
        operator_quote="ensure no signal loss",
        anti_pattern_id="x_v1",
        equation_id="y_v1",
        canonical_unwind_path="x",
        auto_fire_recalibrator=False,
    )
    # When auto_fire is False, recalibrator is NOT mentioned in rationale
    assert "Catalog #371" not in reg.rationale
