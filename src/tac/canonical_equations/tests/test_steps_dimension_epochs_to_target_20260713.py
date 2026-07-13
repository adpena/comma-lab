# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import replace

import pytest

from tac.canonical_equations.steps_dimension_epochs_to_target_20260713 import (
    EQUATION_ID,
    FRESH_ABSOLUTE_TARGET_DSEG,
    FRESH_TICKET,
    HARDNESS_TICKET,
    N_PAIRS,
    TERMINAL_SOLVE_TICKET,
    EpochsToTargetTicket,
    build_steps_dimension_epochs_to_target_v1,
    epochs_saved,
    fixed_quality_threshold_factor,
    measured_accounting,
    step_fraction_saved,
    symbolic_independent_step_fraction_saved,
    wall_fraction,
    wall_fraction_saved,
    wall_seconds,
)
from tac.witness_dsl.curriculum_dsl import (
    FreShFixedQualitySlice,
    FreshFrequencyShift,
    FreShInitControl,
    HardnessOversample,
    TerminalSolve,
)


def test_exact_ticket_rows_preserve_audited_statuses_and_existing_dsl_owners() -> None:
    assert N_PAIRS == 600
    assert FRESH_TICKET.status == "AB_TICKET_ONLY"
    assert HARDNESS_TICKET.status == "WIRING_NEEDED"
    assert TERMINAL_SOLVE_TICKET.status == "WIRING_NEEDED"
    assert "cold" in FRESH_TICKET.start_custody
    assert "same-arm FreSh checkpoints" in FRESH_TICKET.start_custody
    assert "loop consumes P shuffled draws" in HARDNESS_TICKET.verdict_scope
    assert "flags() returns no argv" in TERMINAL_SOLVE_TICKET.existing_dsl_surface
    assert "6dd28a6e295d007ef0e53ae3e0e792a517a5708394a17d2185870e44920dedca" in TERMINAL_SOLVE_TICKET.start_custody
    assert "7515cfe7495526e0dcae656477dc2718180d71f77447e69c23159250ca1afbb2" in TERMINAL_SOLVE_TICKET.start_custody
    for ticket in (FRESH_TICKET, HARDNESS_TICKET, TERMINAL_SOLVE_TICKET):
        assert "all_requested_speed_levers_on=true" in ticket.speed_configuration_rule
        assert "NumPy-fp32" in ticket.measurement_authority_rule

    # The accounting module is a consumer only: it creates no parallel Lever.
    assert FreShInitControl().name == "fresh_init_control"
    assert FreshFrequencyShift().name == "fresh_frequency_shift_init"
    assert FreShFixedQualitySlice().name == "fresh_fixed_quality_slice"
    assert HardnessOversample().name == "hardness_oversample_lever5"
    assert TerminalSolve().flags() == {}


def test_unmeasured_right_censor_stays_none_and_cannot_produce_a_delta() -> None:
    for ticket in (FRESH_TICKET, HARDNESS_TICKET, TERMINAL_SOLVE_TICKET):
        assert ticket.measurement_status == "UNMEASURED"
        assert ticket.control_epochs_to_target is None
        assert ticket.treatment_epochs_to_target is None
        assert ticket.control_optimizer_updates is None
        assert ticket.treatment_optimizer_updates is None
        assert ticket.control_seconds_per_update is None
        assert ticket.treatment_seconds_per_update is None
        assert ticket.source_artifacts
        with pytest.raises(ValueError, match="UNMEASURED or MEASURED_CENSORED"):
            measured_accounting(ticket)


def test_row_invariants_fail_closed_on_wrong_n_or_claimed_missing_measurement() -> None:
    with pytest.raises(ValueError, match="exactly 600"):
        EpochsToTargetTicket(**{**FRESH_TICKET.__dict__, "n_pairs": 8})
    with pytest.raises(ValueError, match="receipt-backed ticket requires"):
        EpochsToTargetTicket(**{**FRESH_TICKET.__dict__, "measurement_status": "MEASURED"})
    with pytest.raises(ValueError, match="cannot claim"):
        EpochsToTargetTicket(**{**FRESH_TICKET.__dict__, "pointer_moved": True})
    with pytest.raises(ValueError, match="cannot exceed"):
        FRESH_TICKET.with_measured_receipt(
            control_epochs_to_target=51,
            treatment_epochs_to_target=42,
            control_optimizer_updates=30_000,
            treatment_optimizer_updates=30_000,
            control_solver_hvp_steps=0,
            treatment_solver_hvp_steps=0,
            control_one_time_overhead_seconds=0.0,
            treatment_one_time_overhead_seconds=0.0,
            control_seconds_per_update=1.0,
            treatment_seconds_per_update=1.0,
            control_recurring_nonupdate_seconds_per_epoch=0.0,
            treatment_recurring_nonupdate_seconds_per_epoch=0.0,
            control_terminal_critical_path_seconds=0.0,
            treatment_terminal_critical_path_seconds=0.0,
            control_async_service_seconds=0.0,
            treatment_async_service_seconds=0.0,
            receipt_custody=("tests/max-window-receipt.json",),
            speed_configuration_custody="tests/max-window-speed-config.json",
            measurement_authority_custody="tests/max-window-authority.json",
        )


def test_epochs_and_update_arithmetic_keeps_extra_visits_visible() -> None:
    assert epochs_saved(50, 42) == 8
    assert step_fraction_saved(30_000, 31_500) == pytest.approx(-0.05)
    assert step_fraction_saved(0, 1) is None
    with pytest.raises(ValueError, match="non-negative"):
        step_fraction_saved(-1, 1)


def test_recurring_cost_aware_wall_composition_prefers_direct_elapsed() -> None:
    fraction = wall_fraction(
        composition_admissible=True,
        composition_refusal_reason=None,
        control_direct_elapsed_seconds_to_crossing=2_200.0,
        treatment_direct_elapsed_seconds_to_crossing=1_760.0,
        control_updates=1_000,
        treatment_updates=800,
        control_epochs=10,
        treatment_epochs=8,
        control_seconds_per_update=2.0,
        treatment_seconds_per_update=1.5,
        control_recurring_nonupdate_seconds_per_epoch=20.0,
        treatment_recurring_nonupdate_seconds_per_epoch=20.0,
        control_one_time_overhead_seconds=0.0,
        treatment_one_time_overhead_seconds=400.0,
        control_terminal_critical_path_seconds=0.0,
        treatment_terminal_critical_path_seconds=0.0,
    )
    assert fraction == pytest.approx(0.8)
    assert wall_seconds(1_000, 10, 2.0, 20.0, 0.0, 0.0) == pytest.approx(2_200.0)
    assert wall_fraction_saved(
        composition_admissible=True,
        composition_refusal_reason=None,
        control_direct_elapsed_seconds_to_crossing=None,
        treatment_direct_elapsed_seconds_to_crossing=None,
        control_updates=1_000,
        treatment_updates=800,
        control_epochs=10,
        treatment_epochs=8,
        control_seconds_per_update=2.0,
        treatment_seconds_per_update=1.5,
        control_recurring_nonupdate_seconds_per_epoch=20.0,
        treatment_recurring_nonupdate_seconds_per_epoch=20.0,
        control_one_time_overhead_seconds=0.0,
        treatment_one_time_overhead_seconds=400.0,
        control_terminal_critical_path_seconds=0.0,
        treatment_terminal_critical_path_seconds=0.0,
    ) == pytest.approx(0.2)
    measured = FRESH_TICKET.with_measured_receipt(
        control_epochs_to_target=50,
        treatment_epochs_to_target=40,
        control_optimizer_updates=30_000,
        treatment_optimizer_updates=24_000,
        control_solver_hvp_steps=0,
        treatment_solver_hvp_steps=0,
        control_one_time_overhead_seconds=20.0,
        treatment_one_time_overhead_seconds=100.0,
        control_seconds_per_update=1.0,
        treatment_seconds_per_update=1.0,
        control_recurring_nonupdate_seconds_per_epoch=0.0,
        treatment_recurring_nonupdate_seconds_per_epoch=0.0,
        control_terminal_critical_path_seconds=0.0,
        treatment_terminal_critical_path_seconds=0.0,
        control_async_service_seconds=100.0,
        treatment_async_service_seconds=200.0,
        control_direct_elapsed_seconds_to_crossing=30_020.0,
        treatment_direct_elapsed_seconds_to_crossing=24_100.0,
        wall_composition_admissible=True,
        wall_composition_refusal_reason=None,
        receipt_custody=("tests/measured-receipt.json",),
        speed_configuration_custody="tests/measured-speed-config.json",
        measurement_authority_custody="tests/measured-authority.json",
    )
    accounting = measured_accounting(measured)
    assert accounting["epochs_saved"] == 10
    assert accounting["step_fraction_saved"] == pytest.approx(0.2)
    assert accounting["wall_fraction"] == pytest.approx(24_100 / 30_020)


def test_completed_but_censored_window_keeps_timing_custody_and_refuses_delta() -> None:
    censored = FRESH_TICKET.with_censored_receipt(
        control_epochs_to_target=None,
        treatment_epochs_to_target=42,
        control_optimizer_updates=30_000,
        treatment_optimizer_updates=30_000,
        control_solver_hvp_steps=0,
        treatment_solver_hvp_steps=0,
        control_one_time_overhead_seconds=20.0,
        treatment_one_time_overhead_seconds=100.0,
        control_seconds_per_update=1.0,
        treatment_seconds_per_update=1.0,
        control_recurring_nonupdate_seconds_per_epoch=0.0,
        treatment_recurring_nonupdate_seconds_per_epoch=0.0,
        control_terminal_critical_path_seconds=0.0,
        treatment_terminal_critical_path_seconds=0.0,
        control_async_service_seconds=0.0,
        treatment_async_service_seconds=0.0,
        receipt_custody=("tests/censored-receipt.json",),
        speed_configuration_custody="tests/censored-speed-config.json",
        measurement_authority_custody="tests/censored-authority.json",
    )
    assert censored.measurement_status == "MEASURED_CENSORED"
    assert censored.control_epochs_to_target is None
    assert censored.control_seconds_per_update == 1.0
    with pytest.raises(ValueError, match="MEASURED_CENSORED"):
        measured_accounting(censored)


def test_epoch_zero_treatment_crossing_is_valid_against_a_positive_control() -> None:
    measured = FRESH_TICKET.with_measured_receipt(
        control_epochs_to_target=1,
        treatment_epochs_to_target=0,
        control_optimizer_updates=1,
        treatment_optimizer_updates=0,
        control_solver_hvp_steps=0,
        treatment_solver_hvp_steps=0,
        control_one_time_overhead_seconds=2.0,
        treatment_one_time_overhead_seconds=3.0,
        control_seconds_per_update=1.0,
        treatment_seconds_per_update=None,
        control_recurring_nonupdate_seconds_per_epoch=0.0,
        treatment_recurring_nonupdate_seconds_per_epoch=0.0,
        control_terminal_critical_path_seconds=0.0,
        treatment_terminal_critical_path_seconds=0.0,
        control_async_service_seconds=0.0,
        treatment_async_service_seconds=0.0,
        control_direct_elapsed_seconds_to_crossing=3.0,
        treatment_direct_elapsed_seconds_to_crossing=3.0,
        wall_composition_admissible=True,
        wall_composition_refusal_reason=None,
        receipt_custody=("tests/epoch-zero-receipt.json",),
        speed_configuration_custody="tests/epoch-zero-speed-config.json",
        measurement_authority_custody="tests/epoch-zero-authority.json",
    )
    accounting = measured_accounting(measured)
    assert accounting["epochs_saved"] == 1
    assert accounting["step_fraction_saved"] == pytest.approx(1.0)
    assert accounting["wall_fraction"] == pytest.approx(1.0)
    assert epochs_saved(1, 0) == 1
    assert step_fraction_saved(0, 0) is None
    assert wall_seconds(0, 0, None, 0.0, 2.0, 0.0) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="zero-update"):
        wall_seconds(0, 0, 1.0, 0.0, 2.0, 0.0)


def test_epoch_zero_control_refuses_update_fraction_and_censored_counterpart_stays_unknown() -> None:
    measured = FRESH_TICKET.with_measured_receipt(
        control_epochs_to_target=0,
        treatment_epochs_to_target=0,
        control_optimizer_updates=0,
        treatment_optimizer_updates=0,
        control_solver_hvp_steps=0,
        treatment_solver_hvp_steps=0,
        control_one_time_overhead_seconds=2.0,
        treatment_one_time_overhead_seconds=3.0,
        control_seconds_per_update=None,
        treatment_seconds_per_update=None,
        control_recurring_nonupdate_seconds_per_epoch=0.0,
        treatment_recurring_nonupdate_seconds_per_epoch=0.0,
        control_terminal_critical_path_seconds=0.0,
        treatment_terminal_critical_path_seconds=0.0,
        control_async_service_seconds=0.0,
        treatment_async_service_seconds=0.0,
        receipt_custody=("tests/epoch-zero-control-receipt.json",),
        speed_configuration_custody="tests/epoch-zero-control-speed-config.json",
        measurement_authority_custody="tests/epoch-zero-control-authority.json",
    )
    assert measured_accounting(measured)["step_fraction_saved"] is None
    censored = FRESH_TICKET.with_censored_receipt(
        control_epochs_to_target=0,
        treatment_epochs_to_target=None,
        control_optimizer_updates=0,
        treatment_optimizer_updates=0,
        control_solver_hvp_steps=0,
        treatment_solver_hvp_steps=0,
        control_one_time_overhead_seconds=2.0,
        treatment_one_time_overhead_seconds=3.0,
        control_seconds_per_update=None,
        treatment_seconds_per_update=None,
        control_recurring_nonupdate_seconds_per_epoch=0.0,
        treatment_recurring_nonupdate_seconds_per_epoch=0.0,
        control_terminal_critical_path_seconds=0.0,
        treatment_terminal_critical_path_seconds=0.0,
        control_async_service_seconds=0.0,
        treatment_async_service_seconds=0.0,
        receipt_custody=("tests/epoch-zero-censored-receipt.json",),
        speed_configuration_custody="tests/epoch-zero-censored-speed-config.json",
        measurement_authority_custody="tests/epoch-zero-censored-authority.json",
    )
    assert censored.control_epochs_to_target == 0
    assert censored.treatment_epochs_to_target is None
    with pytest.raises(ValueError, match="MEASURED_CENSORED"):
        measured_accounting(censored)


def test_fresh_absolute_target_maps_to_existing_threshold_factor_harness() -> None:
    assert fixed_quality_threshold_factor(
        control_epoch0_d_seg=0.05,
        absolute_target_d_seg=FRESH_ABSOLUTE_TARGET_DSEG,
    ) == pytest.approx(FRESH_ABSOLUTE_TARGET_DSEG / 0.05)
    with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
        fixed_quality_threshold_factor(control_epoch0_d_seg=0.04, absolute_target_d_seg=FRESH_ABSOLUTE_TARGET_DSEG)


def test_noncomposable_wall_receipt_returns_none_without_laundering_residual() -> None:
    kwargs = {
        "composition_admissible": False,
        "composition_refusal_reason": (
            "current_wall_receipt: all_requested_speed_levers_on=false and training critical path residual unallocated"
        ),
        "control_direct_elapsed_seconds_to_crossing": None,
        "treatment_direct_elapsed_seconds_to_crossing": None,
        "control_updates": 1_000,
        "treatment_updates": 1_000,
        "control_epochs": 10,
        "treatment_epochs": 10,
        "control_seconds_per_update": 1.0,
        "treatment_seconds_per_update": 1.0,
        "control_recurring_nonupdate_seconds_per_epoch": 0.0,
        "treatment_recurring_nonupdate_seconds_per_epoch": 0.0,
        "control_one_time_overhead_seconds": 0.0,
        "treatment_one_time_overhead_seconds": 0.0,
        "control_terminal_critical_path_seconds": 0.0,
        "treatment_terminal_critical_path_seconds": 0.0,
    }
    assert wall_fraction(**kwargs) is None
    assert wall_fraction_saved(**kwargs) is None


def test_wall_invariants_reject_unmeasured_composition_and_validate_direct_and_fallback_inputs() -> None:
    with pytest.raises(ValueError, match="UNMEASURED ticket cannot admit"):
        replace(FRESH_TICKET, wall_composition_admissible=True)
    direct_kwargs = {
        "composition_admissible": True,
        "composition_refusal_reason": None,
        "control_direct_elapsed_seconds_to_crossing": 2.0,
        "treatment_direct_elapsed_seconds_to_crossing": 1.0,
        "control_updates": -1,
        "treatment_updates": 0,
        "control_epochs": 1,
        "treatment_epochs": 0,
        "control_seconds_per_update": 1.0,
        "treatment_seconds_per_update": None,
        "control_recurring_nonupdate_seconds_per_epoch": 0.0,
        "treatment_recurring_nonupdate_seconds_per_epoch": 0.0,
        "control_one_time_overhead_seconds": 0.0,
        "treatment_one_time_overhead_seconds": 0.0,
        "control_terminal_critical_path_seconds": 0.0,
        "treatment_terminal_critical_path_seconds": 0.0,
    }
    with pytest.raises(ValueError, match="non-negative"):
        wall_fraction(**direct_kwargs)
    direct_kwargs["control_updates"] = 1
    direct_kwargs["control_recurring_nonupdate_seconds_per_epoch"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        wall_fraction(**direct_kwargs)
    zero_fallback = {**direct_kwargs, "control_direct_elapsed_seconds_to_crossing": None,
                     "treatment_direct_elapsed_seconds_to_crossing": None,
                     "control_recurring_nonupdate_seconds_per_epoch": 0.0,
                     "control_updates": 0, "control_epochs": 0,
                     "control_seconds_per_update": None}
    with pytest.raises(ValueError, match="positive control total wall"):
        wall_fraction(**zero_fallback)


def test_wiring_needed_ticket_requires_closure_before_a_measured_row() -> None:
    measured = FRESH_TICKET.with_measured_receipt(
        control_epochs_to_target=1,
        treatment_epochs_to_target=1,
        control_optimizer_updates=1,
        treatment_optimizer_updates=1,
        control_solver_hvp_steps=0,
        treatment_solver_hvp_steps=0,
        control_one_time_overhead_seconds=0.0,
        treatment_one_time_overhead_seconds=0.0,
        control_seconds_per_update=1.0,
        treatment_seconds_per_update=1.0,
        control_recurring_nonupdate_seconds_per_epoch=0.0,
        treatment_recurring_nonupdate_seconds_per_epoch=0.0,
        control_terminal_critical_path_seconds=0.0,
        treatment_terminal_critical_path_seconds=0.0,
        control_async_service_seconds=0.0,
        treatment_async_service_seconds=0.0,
        receipt_custody=("tests/receipt.json",),
        speed_configuration_custody="tests/wiring-speed-config.json",
        measurement_authority_custody="tests/wiring-authority.json",
    )
    assert measured.wall_composition_admissible is False
    assert measured.wall_composition_refusal_reason
    with pytest.raises(ValueError, match="wiring_closure_evidence"):
        replace(measured, status="WIRING_NEEDED")
    assert replace(
        measured,
        status="WIRING_NEEDED",
        wiring_closure_evidence="tests/wiring-closure.json",
    ).status == "WIRING_NEEDED"


def test_independent_composition_is_explicitly_assumed_not_measured() -> None:
    result = symbolic_independent_step_fraction_saved((0.2, 0.5))
    assert result["composition_status"] == "ASSUMED_INDEPENDENT_SYMBOLIC_SCENARIO"
    assert result["step_fraction_saved"] == pytest.approx(0.6)


def test_equation_is_unanchored_ticket_semantic_and_non_promotable() -> None:
    equation = build_steps_dimension_epochs_to_target_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.empirical_anchors == ()
    assert equation.predicted_vs_empirical_residual == {}
    assert equation.domain_of_validity["right_censoring"].endswith("never zero")
    assert equation.provenance.promotion_eligible is False
    assert equation.provenance.score_claim_valid is False
    assert equation.canonical_producers == ()
    assert not hasattr(__import__(
        "tac.canonical_equations.steps_dimension_epochs_to_target_20260713", fromlist=["*"],
    ), "populate_steps_dimension_epochs_to_target_equation")
