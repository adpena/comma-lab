# SPDX-License-Identifier: MIT
"""Tests for tac.cathedral_autopilot.canonical_apparatus_wave_n46 (5 + 5)."""
from __future__ import annotations

from tac.cathedral_autopilot.canonical_apparatus_wave_n46 import (
    build_all_wave_n46_anti_patterns,
    build_all_wave_n46_equations,
)


def test_5_canonical_equations_constructed() -> None:
    eqs = build_all_wave_n46_equations()
    assert len(eqs) == 5
    ids = {eq.equation_id for eq in eqs}
    assert "meta_orchestrator_highest_ev_shortest_wall_clock_metric_v1" in ids
    assert "meta_orchestrator_three_metric_trichotomy_orthogonality_v1" in ids
    assert (
        "meta_orchestrator_variance_acceptance_dominance_at_low_probability_high_leverage_v1"
        in ids
    )
    assert "meta_orchestrator_lesson_set_completeness_lower_bound_v1" in ids
    assert (
        "meta_orchestrator_cap_window_discipline_per_turn_variance_amortization_v1"
        in ids
    )


def test_5_canonical_anti_patterns_constructed() -> None:
    aps = build_all_wave_n46_anti_patterns()
    assert len(aps) == 5
    ids = {ap.anti_pattern_id for ap in aps}
    assert (
        "manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1"
        in ids
    )
    assert "operator_correction_canonical_apparatus_mutation_lag_v1" in ids
    assert "spawn_prompt_boilerplate_duplication_across_subagent_waves_v1" in ids
    assert (
        "hygiene_ev_vs_frontier_breaking_ev_vs_highest_ev_shortest_wall_clock_conflation_canonical_rediscovery_v1"
        in ids
    )
    assert (
        "metric_orthogonality_inversion_safest_incremental_as_default_when_operator_wants_variance_acceptance_v1"
        in ids
    )


def test_equations_carry_canonical_consumers() -> None:
    """Per CLAUDE.md "Subagent coherence-by-default": no orphan equations."""
    for eq in build_all_wave_n46_equations():
        assert (
            eq.canonical_consumers or eq.canonical_producers
        ), f"orphan equation: {eq.equation_id}"


def test_anti_patterns_carry_canonical_consumers() -> None:
    for ap in build_all_wave_n46_anti_patterns():
        assert ap.canonical_consumers, f"orphan anti-pattern: {ap.anti_pattern_id}"


def test_equations_carry_canonical_provenance() -> None:
    """Per Catalog #323 canonical Provenance umbrella."""
    for eq in build_all_wave_n46_equations():
        assert eq.provenance is not None
        assert eq.empirical_anchors  # at least one anchor per equation


def test_equations_recalibrate_on_new_anchors() -> None:
    """Per Catalog #371 auto-recalibration trigger."""
    from tac.canonical_equations.equation import RECALIBRATE_ON_NEW_ANCHORS
    for eq in build_all_wave_n46_equations():
        assert eq.next_recalibration_trigger == RECALIBRATE_ON_NEW_ANCHORS


def test_anti_patterns_severity_valid() -> None:
    from tac.canonical_anti_patterns.anti_pattern import VALID_SEVERITIES
    for ap in build_all_wave_n46_anti_patterns():
        assert ap.severity in VALID_SEVERITIES


def test_anti_patterns_paradigm_class_valid() -> None:
    from tac.canonical_anti_patterns.anti_pattern import VALID_PARADIGM_CLASSES
    for ap in build_all_wave_n46_anti_patterns():
        assert ap.paradigm_class in VALID_PARADIGM_CLASSES


def test_anti_patterns_carry_canonical_unwind_path() -> None:
    for ap in build_all_wave_n46_anti_patterns():
        assert ap.canonical_unwind_path, f"missing unwind: {ap.anti_pattern_id}"


def test_equations_to_dict_round_trip() -> None:
    """Per Catalog #323 canonical Provenance umbrella + JSON safety."""
    for eq in build_all_wave_n46_equations():
        d = eq.to_dict()
        assert d["equation_id"] == eq.equation_id
        assert d["one_line_summary"]
        assert d["latex_form"]
