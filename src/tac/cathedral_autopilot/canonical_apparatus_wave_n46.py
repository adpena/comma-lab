# SPDX-License-Identifier: MIT
"""Wave N+46 canonical equations + anti-patterns for META-orchestrator extension.

Per ``feedback_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_canonicalize_and_harden_for_automation_standing_directive_20260528.md``
5 canonical equations + 5 canonical anti-patterns.

Per Catalog #344 canonical equations + anti-patterns registry:
each registration appends a new ``registered`` event; the latest-row-
wins query semantics in ``query_equations`` / ``query_anti_patterns``
ensure consumers see the most recent payload. APPEND-ONLY per
HISTORICAL_PROVENANCE Catalog #110/#113.

The 5 canonical equations:

1. ``meta_orchestrator_highest_ev_shortest_wall_clock_metric_v1`` —
   ``EV = (predicted_ΔS_magnitude × probability_materializes) / wall_clock_to_validation``
2. ``meta_orchestrator_three_metric_trichotomy_orthogonality_v1`` —
   ``HYGIENE-EV ⊥ FRONTIER-BREAKING-EV ⊥ HIGHEST-EV-SHORTEST-WC``
3. ``meta_orchestrator_variance_acceptance_dominance_at_low_probability_high_leverage_v1`` —
   when predicted_ΔS magnitude exceeds 10× safest-alternative,
   variance-acceptance dominates
4. ``meta_orchestrator_lesson_set_completeness_lower_bound_v1`` —
   canonical 13-lesson HNeRV parity discipline = lower bound; L14-L27
   candidates expansion required
5. ``meta_orchestrator_cap_window_discipline_per_turn_variance_amortization_v1`` —
   cap=1-per-turn under throttle distributes variance across cap-windows

The 5 canonical anti-patterns:

1. ``manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1``
2. ``operator_correction_canonical_apparatus_mutation_lag_v1``
3. ``spawn_prompt_boilerplate_duplication_across_subagent_waves_v1``
4. ``hygiene_ev_vs_frontier_breaking_ev_vs_highest_ev_shortest_wall_clock_conflation_canonical_rediscovery_v1``
5. ``metric_orthogonality_inversion_safest_incremental_as_default_when_operator_wants_variance_acceptance_v1``
"""
from __future__ import annotations

from pathlib import Path

from tac.canonical_equations.equation import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
)
from tac.canonical_equations.registry import register_canonical_equation
from tac.canonical_anti_patterns.anti_pattern import (
    AntiPattern,
    PARADIGM_DISCIPLINE,
    PARADIGM_RIGOR_LOSS,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
)
from tac.canonical_anti_patterns.registry import register_anti_pattern
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)
from tac.provenance.contract import Provenance


_DESIGN_ANCHOR_SHA = "0" * 64
_WAVE_N46_LANDING_UTC = "2026-05-29T00:00:00Z"
_WAVE_N46_MEMO_PATH = (
    "feedback_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_"
    "canonicalize_and_harden_for_automation_standing_directive_20260528.md"
)
_WAVE_N46_CASCADE_MEMO = (
    "feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade_20260528.md"
)
_WAVE_N46_TRICHOTOMY_MEMO = (
    "feedback_canonical_ev_metric_trichotomy_hygiene_vs_frontier_vs_highest_ev_shortest_wall_clock_20260528.md"
)


def _design_provenance(model_id: str) -> Provenance:
    """Build a PREDICTED Provenance for design-only equation registration."""
    return build_provenance_for_predicted(
        model_id=model_id,
        inputs_sha256=_DESIGN_ANCHOR_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )


def _wave_n46_memo_provenance(memo_path: str) -> Provenance:
    """Build a RESEARCH_SIDECAR Provenance for Wave N+46 anchor memos."""
    return build_provenance_for_research_sidecar(
        sidecar_path=memo_path,
        reactivation_criteria="wave_n46_cathedral_autopilot_extension_canonical_apparatus_mutation",
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )


# ---------------------------------------------------------------------------
# 5 canonical equations (per the triple-message standing directive).
# ---------------------------------------------------------------------------


def build_meta_orchestrator_highest_ev_shortest_wall_clock_metric_v1() -> CanonicalEquation:
    """Equation 1: operator's canonical EV metric formalized."""
    anchor = EmpiricalAnchor(
        anchor_id="operator_correction_three_metric_trichotomy_20260528",
        measurement_utc="2026-05-28T23:40:00Z",
        inputs={
            "operator_quote": (
                "we don't ujst want the safest score lowering; we want highest EV "
                "in shortest wall clock"
            ),
            "context": "rapid 3-correction sequence today",
        },
        predicted_output={"canonical_routing_metric_default": "highest_ev_shortest_wall_clock"},
        empirical_output={"operator_canonical_metric_routing_default": "highest_ev_shortest_wall_clock"},
        residual=0.0,
        source_artifact=_WAVE_N46_TRICHOTOMY_MEMO,
        measurement_method="operator_binding_correction_canonical_anchor",
        provenance=_wave_n46_memo_provenance(_WAVE_N46_TRICHOTOMY_MEMO),
    )
    return CanonicalEquation(
        equation_id="meta_orchestrator_highest_ev_shortest_wall_clock_metric_v1",
        name="META-orchestrator highest-EV-shortest-wall-clock canonical metric",
        one_line_summary=(
            "EV = (predicted ΔS magnitude × probability materializes) / wall-clock-to-validation"
        ),
        latex_form=(
            r"\text{EV}_{\text{HE-SW}} = "
            r"\frac{|\Delta S_{\text{predicted}}| \cdot P_{\text{materializes}}}"
            r"{T_{\text{wall-clock-to-validation (hours)}}}"
        ),
        python_callable_module_path=(
            "tac.cathedral_autopilot.three_metric_trichotomy:"
            "_compute_highest_ev_shortest_wall_clock_ev"
        ),
        domain_of_validity={
            "predicted_delta_s_range": "any real number; magnitude consumed",
            "probability_materializes_range": "[0.0, 1.0]",
            "wall_clock_to_validation_hours_range": "(0, +inf)",
            "applies_to": "main-thread spawn-decision per-turn ranking",
        },
        units_in={
            "predicted_delta_s": "float_contest_score",
            "probability_materializes": "float_probability",
            "wall_clock_to_validation_hours": "float_hours",
        },
        units_out={"ev_he_sw": "float_score_per_hour"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"operator_binding_correction_canonical_anchor": 0.0},
        last_calibration_utc=_WAVE_N46_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_autopilot.three_metric_trichotomy:rank_candidates_via_three_metric_trichotomy",
            "tac.cathedral_autopilot.per_turn_spawn_decision:select_canonical_next_spawn_for_main_thread",
            "tools/cathedral_autopilot_autonomous_loop.py",
            "src/tac/cathedral_consumers/meta_orchestrator_consumer/",
        ),
        canonical_producers=(
            "tac.cathedral_autopilot.three_metric_trichotomy:_compute_highest_ev_shortest_wall_clock_ev",
        ),
        provenance=_design_provenance("meta_orchestrator_highest_ev_shortest_wall_clock_metric.v1"),
    )


def build_meta_orchestrator_three_metric_trichotomy_orthogonality_v1() -> CanonicalEquation:
    """Equation 2: 3-metric trichotomy orthogonality + invariant."""
    anchor = EmpiricalAnchor(
        anchor_id="three_correction_sequence_20260528",
        measurement_utc="2026-05-28T23:55:00Z",
        inputs={
            "correction_count_today": 3,
            "metrics_conflated": ["hygiene_ev", "frontier_breaking_ev", "highest_ev_shortest_wall_clock"],
        },
        predicted_output={"distinct_top_candidates_across_3_metrics_lower_bound": 1},
        empirical_output={
            "distinct_top_candidates_across_3_metrics": 3,
            "orthogonality_empirically_confirmed": True,
        },
        residual=0.0,
        source_artifact=_WAVE_N46_TRICHOTOMY_MEMO,
        measurement_method="three_correction_sequence_canonical_anchor",
        provenance=_wave_n46_memo_provenance(_WAVE_N46_TRICHOTOMY_MEMO),
    )
    return CanonicalEquation(
        equation_id="meta_orchestrator_three_metric_trichotomy_orthogonality_v1",
        name="META-orchestrator 3-metric trichotomy orthogonality invariant",
        one_line_summary=(
            "HYGIENE-EV perp FRONTIER-BREAKING-EV perp HIGHEST-EV-SHORTEST-WC; "
            "conflating any 2 produces canonical anti-pattern recurrence"
        ),
        latex_form=(
            r"\text{HYGIENE-EV} \perp \text{FRONTIER-BREAKING-EV} \perp "
            r"\text{HIGHEST-EV-SHORTEST-WC}"
        ),
        python_callable_module_path=(
            "tac.cathedral_autopilot.three_metric_trichotomy:"
            "rank_candidates_via_three_metric_trichotomy"
        ),
        domain_of_validity={
            "applies_to": "main-thread spawn-decision per-turn ranking",
            "metric_count": 3,
        },
        units_in={
            "candidates": "list_of_candidate_mappings",
            "operator_canonical_metric": "str_metric_name",
        },
        units_out={
            "ranking_result": "ThreeMetricTrichotomyRankingResult",
            "per_metric_top_candidate": "dict_metric_to_candidate_id",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"three_correction_sequence_canonical_anchor": 0.0},
        last_calibration_utc=_WAVE_N46_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_autopilot.per_turn_spawn_decision:select_canonical_next_spawn_for_main_thread",
            "tac.cathedral_autopilot.canonical_invariants:validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants",
            "tools/cathedral_autopilot_autonomous_loop.py",
        ),
        canonical_producers=(
            "tac.cathedral_autopilot.three_metric_trichotomy:rank_candidates_via_three_metric_trichotomy",
        ),
        provenance=_design_provenance("meta_orchestrator_three_metric_trichotomy_orthogonality.v1"),
    )


def build_meta_orchestrator_variance_acceptance_dominance_at_low_probability_high_leverage_v1() -> (
    CanonicalEquation
):
    """Equation 3: variance-acceptance dominance at low-probability high-leverage."""
    # Per the trichotomy memo: a 15%-probability +374% leverage candidate
    # beats a 90%-probability +5% leverage candidate per the EV math.
    # 0.374 * 0.15 / 1.5 = 0.0374 vs 0.05 * 0.9 / 4.0 = 0.01125 -> 3.33×
    anchor = EmpiricalAnchor(
        anchor_id="super_additive_alpha_4_74_vs_incremental_4_substrate_cascade_20260528",
        measurement_utc="2026-05-28T23:40:00Z",
        inputs={
            "high_leverage_predicted_delta_pct": 374.0,
            "high_leverage_probability": 0.15,
            "high_leverage_wall_clock_hours": 1.5,
            "incremental_predicted_delta_pct": 5.0,
            "incremental_probability": 0.9,
            "incremental_wall_clock_hours": 4.0,
            "magnitude_ratio_threshold": 10.0,
        },
        predicted_output={"ev_he_sw_ratio_high_over_incremental_lower_bound": 1.0},
        empirical_output={"ev_he_sw_ratio_high_over_incremental": 3.33},
        residual=0.0,
        source_artifact=_WAVE_N46_TRICHOTOMY_MEMO,
        measurement_method="variance_acceptance_dominance_canonical_anchor",
        provenance=_wave_n46_memo_provenance(_WAVE_N46_TRICHOTOMY_MEMO),
    )
    return CanonicalEquation(
        equation_id="meta_orchestrator_variance_acceptance_dominance_at_low_probability_high_leverage_v1",
        name="META-orchestrator variance-acceptance dominance invariant",
        one_line_summary=(
            "When |predicted ΔS| exceeds 10× safest-alternative, "
            "variance-acceptance dominates safest-incremental per EV math"
        ),
        latex_form=(
            r"|\Delta S_{\text{high-leverage}}| > 10 \cdot |\Delta S_{\text{safest}}| "
            r"\implies \text{EV}_{\text{HE-SW}}(\text{high-leverage}) > "
            r"\text{EV}_{\text{HE-SW}}(\text{safest})"
        ),
        python_callable_module_path=(
            "tac.cathedral_autopilot.three_metric_trichotomy:"
            "_compute_highest_ev_shortest_wall_clock_ev"
        ),
        domain_of_validity={
            "applies_to": "low-probability high-leverage candidate ranking",
            "magnitude_ratio_threshold": 10.0,
        },
        units_in={
            "high_leverage_candidate": "candidate_mapping",
            "safest_candidate": "candidate_mapping",
        },
        units_out={"variance_acceptance_dominates": "bool"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"variance_acceptance_dominance_canonical_anchor": 0.0},
        last_calibration_utc=_WAVE_N46_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_autopilot.three_metric_trichotomy:rank_candidates_via_three_metric_trichotomy",
            "tac.cathedral_autopilot.per_turn_spawn_decision:select_canonical_next_spawn_for_main_thread",
        ),
        canonical_producers=(
            "tac.cathedral_autopilot.three_metric_trichotomy:_compute_highest_ev_shortest_wall_clock_ev",
        ),
        provenance=_design_provenance("meta_orchestrator_variance_acceptance_dominance.v1"),
    )


def build_meta_orchestrator_lesson_set_completeness_lower_bound_v1() -> CanonicalEquation:
    """Equation 4: 13-lesson HNeRV parity discipline = lower bound."""
    anchor = EmpiricalAnchor(
        anchor_id="hnerv_parity_13_lesson_count_lower_bound_20260528",
        measurement_utc="2026-05-28T23:00:00Z",
        inputs={
            "claude_md_documented_lesson_count": 13,
            "operator_correction": "more lessons probably; the 13 is a lower bound",
        },
        predicted_output={"L14_L27_candidate_lesson_count_lower_bound": 14},
        empirical_output={"L14_L27_candidate_lesson_count_lower_bound_per_operator": 14},
        residual=0.0,
        source_artifact=(
            "feedback_prioritization_metric_hygiene_vs_frontier_breaking_orthogonal_plus_13_lessons_incomplete_20260528.md"
        ),
        measurement_method="hnerv_parity_lesson_count_lower_bound_canonical_anchor",
        provenance=_wave_n46_memo_provenance(
            "feedback_prioritization_metric_hygiene_vs_frontier_breaking_orthogonal_plus_13_lessons_incomplete_20260528.md"
        ),
    )
    return CanonicalEquation(
        equation_id="meta_orchestrator_lesson_set_completeness_lower_bound_v1",
        name="META-orchestrator HNeRV parity lesson-set completeness lower bound",
        one_line_summary=(
            "Canonical 13-lesson HNeRV parity discipline = lower bound; L14-L27 "
            "candidate expansion required per operator correction"
        ),
        latex_form=(
            r"|\text{lessons}_{\text{honored}}| \geq 13 \text{ (lower bound)}; "
            r"L_{14}\dots L_{27} \text{ candidates pending deep-research expansion}"
        ),
        python_callable_module_path=(
            "tac.cathedral_autopilot.three_metric_trichotomy:_compute_hygiene_ev"
        ),
        domain_of_validity={
            "applies_to": "HYGIENE-EV computation",
            "current_lesson_count": 13,
            "lower_bound": 13,
        },
        units_in={
            "hygiene_lessons_honored": "int_count",
            "hygiene_lessons_total": "int_count_default_13",
        },
        units_out={"hygiene_ev_ratio": "float_ratio_zero_to_one"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"hnerv_parity_lesson_count_lower_bound_canonical_anchor": 0.0},
        last_calibration_utc=_WAVE_N46_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_autopilot.three_metric_trichotomy:_compute_hygiene_ev",
            "Wave N+47 lesson-set expansion deep-research subagent",
        ),
        canonical_producers=(
            "tac.cathedral_autopilot.three_metric_trichotomy:_compute_hygiene_ev",
        ),
        provenance=_design_provenance("meta_orchestrator_lesson_set_completeness_lower_bound.v1"),
    )


def build_meta_orchestrator_cap_window_discipline_per_turn_variance_amortization_v1() -> (
    CanonicalEquation
):
    """Equation 5: cap-window discipline per-turn variance amortization."""
    anchor = EmpiricalAnchor(
        anchor_id="cap_window_discipline_4_in_flight_today_20260528",
        measurement_utc="2026-05-28T22:30:00Z",
        inputs={
            "cap_per_turn_default": 1,
            "in_flight_sister_count_today": 4,
            "rate_limit_cascade_anti_pattern_id": (
                "simultaneous_multi_subagent_spawn_rate_limit_cascade_anti_pattern_v1"
            ),
        },
        predicted_output={"variance_amortization_factor_lower_bound": 1.0},
        empirical_output={"variance_amortization_factor_per_cap_window": 1.0},
        residual=0.0,
        source_artifact=_WAVE_N46_CASCADE_MEMO,
        measurement_method="cap_window_discipline_canonical_anchor",
        provenance=_wave_n46_memo_provenance(_WAVE_N46_CASCADE_MEMO),
    )
    return CanonicalEquation(
        equation_id="meta_orchestrator_cap_window_discipline_per_turn_variance_amortization_v1",
        name="META-orchestrator cap-window discipline per-turn variance amortization",
        one_line_summary=(
            "Cap=1-per-turn under throttle distributes variance across "
            "cap-windows; canonical decision-cycle for highest-EV-shortest-WC"
        ),
        latex_form=(
            r"\text{cap}_{\text{per-turn}} = 1 \text{ (default under throttle)}; "
            r"\text{variance amortized across } N \text{ cap-windows}"
        ),
        python_callable_module_path=(
            "tac.cathedral_autopilot.per_turn_spawn_decision:"
            "select_canonical_next_spawn_for_main_thread"
        ),
        domain_of_validity={
            "applies_to": "main-thread per-turn cap-window discipline",
            "cap_per_turn_default": 1,
            "under_throttle": True,
        },
        units_in={
            "in_flight_subagents": "list_of_subagent_mappings",
            "cap_window_remaining": "int_slots",
        },
        units_out={"candidate_selection": "CandidateSelection"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"cap_window_discipline_canonical_anchor": 0.0},
        last_calibration_utc=_WAVE_N46_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_autopilot.per_turn_spawn_decision:select_canonical_next_spawn_for_main_thread",
            "tools/cathedral_autopilot_autonomous_loop.py",
        ),
        canonical_producers=(
            "tac.cathedral_autopilot.per_turn_spawn_decision:select_canonical_next_spawn_for_main_thread",
        ),
        provenance=_design_provenance("meta_orchestrator_cap_window_discipline.v1"),
    )


# ---------------------------------------------------------------------------
# 5 canonical anti-patterns (per the triple-message standing directive).
# ---------------------------------------------------------------------------


def build_manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1() -> (
    AntiPattern
):
    """Anti-pattern 1: main-thread queue ranking drift across turns."""
    return AntiPattern(
        anti_pattern_id="manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1",
        description=(
            "Main-thread queue priority shifts per-turn based on landed "
            "sister NOT canonical metric. The agent manually re-ranks each "
            "turn rather than routing through the canonical 3-metric "
            "trichotomy + deterministic per-turn helper, producing "
            "operator-visible ranking drift."
        ),
        forbidden_pattern_predicate=(
            "main_thread_decision NOT routes through "
            "tac.cathedral_autopilot.three_metric_trichotomy."
            "rank_candidates_via_three_metric_trichotomy AND "
            "ranking_via_canonical_three_metric_trichotomy != True"
        ),
        falsification_band={
            "ranking_drift_count_across_turns_lo": 1.0,
            "ranking_drift_count_across_turns_hi": 100.0,
        },
        recurrence_conditions=(
            "main-thread queue priority recomputed per-turn from scratch",
            "ranking based on which sister landed first (drift driver)",
            "no canonical metric declared in main_thread_decision",
        ),
        canonical_source_anchor=(
            f"{_WAVE_N46_MEMO_PATH} (anti-pattern #1); "
            "CLAUDE.md 'Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE'"
        ),
        canonical_unwind_path=(
            "Route through tac.cathedral_autopilot."
            "rank_candidates_via_three_metric_trichotomy + "
            "select_canonical_next_spawn_for_main_thread. Ranking becomes "
            "deterministic per canonical helper output."
        ),
        canonical_producers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "main-thread agent spawn-decision code paths",
        ),
        canonical_consumers=(
            "tac.cathedral_autopilot.canonical_invariants:validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants",
            "src/tac/cathedral_consumers/meta_orchestrator_consumer/",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        paradigm_class=PARADIGM_DISCIPLINE,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance(
            "manual_main_thread_orchestrator_ranking_drift.v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_WAVE_N46_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_operator_correction_canonical_apparatus_mutation_lag_v1() -> AntiPattern:
    """Anti-pattern 2: operator correction -> canonical mutation TIME-LAG."""
    return AntiPattern(
        anti_pattern_id="operator_correction_canonical_apparatus_mutation_lag_v1",
        description=(
            "Operator binding correction -> memory file landing TIME-LAG "
            "before canonical apparatus mutation (equation + anti-pattern + "
            "consumer) registered structurally. The corrected behavior is "
            "documented in prose but the canonical apparatus does NOT yet "
            "reflect it, so the same correction can recur silently."
        ),
        forbidden_pattern_predicate=(
            "memo_landing_count(operator_correction) > 0 AND "
            "canonical_apparatus_mutation_count(operator_correction) == 0 "
            "WITHIN same turn"
        ),
        falsification_band={
            "correction_to_mutation_lag_turns_lo": 1.0,
            "correction_to_mutation_lag_turns_hi": 100.0,
        },
        recurrence_conditions=(
            "operator binding correction landed in memory file this turn",
            "no canonical equation registered for the corrected behavior",
            "no canonical anti-pattern registered for the corrected bug class",
        ),
        canonical_source_anchor=(
            f"{_WAVE_N46_MEMO_PATH} (anti-pattern #2); "
            "CLAUDE.md 'Results must become system intelligence' + 'memos must be acted upon'"
        ),
        canonical_unwind_path=(
            "Use tac.cathedral_autopilot.register_operator_binding_correction "
            "within the SAME TURN as the memo landing. The helper auto-fires "
            "Catalog #371 recalibrator + records the canonical equation + "
            "anti-pattern routing."
        ),
        canonical_producers=(
            "main-thread agent post-correction landing surface",
        ),
        canonical_consumers=(
            "tac.cathedral_autopilot.canonical_invariants:validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants",
            "src/tac/cathedral_consumers/meta_orchestrator_consumer/",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        paradigm_class=PARADIGM_DISCIPLINE,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance(
            "operator_correction_canonical_apparatus_mutation_lag.v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_WAVE_N46_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_spawn_prompt_boilerplate_duplication_across_subagent_waves_v1() -> AntiPattern:
    """Anti-pattern 3: spawn prompt boilerplate duplicated across waves."""
    return AntiPattern(
        anti_pattern_id="spawn_prompt_boilerplate_duplication_across_subagent_waves_v1",
        description=(
            "Canonical clauses (Catalog #376 pre-flight + sister scope "
            "DISJOINT + canonical metric declaration + apparatus mutation "
            "requirement) duplicated verbatim per spawn vs canonical helper-"
            "generated. The operator empirically caught a sister instance "
            "where my proposed canonical helper would have BEEN a duplicate "
            "of the cathedral autopilot itself."
        ),
        forbidden_pattern_predicate=(
            "spawn_prompt.contains(canonical_clause_X) AND "
            "spawn_prompt.NOT generated_via_canonical_helper AND "
            "canonical_clause_X.count_across_recent_spawns > 1"
        ),
        falsification_band={
            "duplicate_canonical_clause_count_lo": 1.0,
            "duplicate_canonical_clause_count_hi": 100.0,
        },
        recurrence_conditions=(
            "spawn prompt repeats Catalog #376 PV preamble verbatim",
            "spawn prompt repeats Catalog #117/#157/#174 serializer disclaimers",
            "agent proposes building a NEW canonical helper when an existing "
            "canonical surface already covers the scope (operator-caught today)",
        ),
        canonical_source_anchor=(
            f"{_WAVE_N46_CASCADE_MEMO} + {_WAVE_N46_MEMO_PATH} (anti-pattern #3); "
            "operator empirical catch 2026-05-28 ~23:55Z"
        ),
        canonical_unwind_path=(
            "Use canonical spawn-prompt helper (future Wave N+52 landing); "
            "before proposing NEW canonical surface, check whether cathedral "
            "autopilot covers it (canonical META-orchestrator lesson per "
            "feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade memo). "
            "If yes -> EXTEND. If no -> DOCUMENT THE GAP per Catalog #344 "
            "BEFORE building anything new."
        ),
        canonical_producers=(
            "main-thread agent spawn-prompt composition surface",
        ),
        canonical_consumers=(
            "tac.cathedral_autopilot.canonical_invariants:validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        paradigm_class=PARADIGM_DISCIPLINE,
        severity=SEVERITY_MEDIUM,
        provenance=_design_provenance(
            "spawn_prompt_boilerplate_duplication.v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_WAVE_N46_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_hygiene_ev_vs_frontier_breaking_ev_vs_highest_ev_shortest_wall_clock_conflation_canonical_rediscovery_v1() -> (
    AntiPattern
):
    """Anti-pattern 4: 3-metric conflation canonical REDISCOVERY."""
    return AntiPattern(
        anti_pattern_id="hygiene_ev_vs_frontier_breaking_ev_vs_highest_ev_shortest_wall_clock_conflation_canonical_rediscovery_v1",
        description=(
            "Three-correction sequence today (hygiene-vs-frontier + "
            "mathematical grounding + variance-acceptance) operator-"
            "discovered NOT structurally caught. Canonical evidence for "
            "the conflation of the 3 orthogonal metrics; structural "
            "protection requires canonical orchestrator helper consuming "
            "all 3 metrics independently."
        ),
        forbidden_pattern_predicate=(
            "main_thread_decision.ranking_uses_single_composite_metric AND "
            "NOT main_thread_decision.three_metrics_computed_independently"
        ),
        falsification_band={
            "conflation_recurrence_count_lo": 1.0,
            "conflation_recurrence_count_hi": 100.0,
        },
        recurrence_conditions=(
            "main-thread ranking uses single composite predicted_delta only",
            "main-thread ranking treats hygiene-EV as equivalent to frontier-breaking-EV",
            "main-thread ranking treats frontier-breaking-EV as equivalent to highest-EV-shortest-WC",
            "operator must issue correction to surface the orthogonality",
        ),
        canonical_source_anchor=(
            f"{_WAVE_N46_TRICHOTOMY_MEMO} + {_WAVE_N46_MEMO_PATH} (anti-pattern #4); "
            "3-correction sequence 2026-05-28"
        ),
        canonical_unwind_path=(
            "Route through tac.cathedral_autopilot."
            "rank_candidates_via_three_metric_trichotomy which computes all "
            "3 metrics independently + surfaces per-metric top candidate to "
            "make orthogonality visible. Per "
            "meta_orchestrator_three_metric_trichotomy_orthogonality_v1 "
            "canonical equation."
        ),
        canonical_producers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "main-thread agent ranking surface",
        ),
        canonical_consumers=(
            "tac.cathedral_autopilot.three_metric_trichotomy:rank_candidates_via_three_metric_trichotomy",
            "tac.cathedral_autopilot.canonical_invariants:validate_no_ad_hoc_no_signal_loss_no_rediscovery_no_duplicate_no_drift_invariants",
            "src/tac/cathedral_consumers/meta_orchestrator_consumer/",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        paradigm_class=PARADIGM_RIGOR_LOSS,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance(
            "hygiene_ev_vs_frontier_breaking_ev_vs_highest_ev_shortest_wall_clock_conflation.v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_WAVE_N46_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def build_metric_orthogonality_inversion_safest_incremental_as_default_when_operator_wants_variance_acceptance_v1() -> (
    AntiPattern
):
    """Anti-pattern 5: variance-aversion as default when operator wants variance-acceptance."""
    return AntiPattern(
        anti_pattern_id="metric_orthogonality_inversion_safest_incremental_as_default_when_operator_wants_variance_acceptance_v1",
        description=(
            "Variance-aversion is canonical anti-pattern when operator "
            "explicitly says 'not necessarily the safest'. The agent's "
            "default of ranking by safest-incremental-closest-to-ready "
            "inverts the operator's canonical metric (highest-EV-shortest-"
            "wall-clock)."
        ),
        forbidden_pattern_predicate=(
            "operator_canonical_metric == 'highest_ev_shortest_wall_clock' AND "
            "main_thread_decision.routing_default == 'safest_incremental_closest_to_ready'"
        ),
        falsification_band={
            "variance_aversion_inversion_count_lo": 1.0,
            "variance_aversion_inversion_count_hi": 100.0,
        },
        recurrence_conditions=(
            "main-thread routing default selects high-probability low-leverage candidate",
            "main-thread queue priority systematically deprioritizes low-probability high-leverage candidates",
            "main-thread routing ignores variance-acceptance dominance per EV math",
        ),
        canonical_source_anchor=(
            f"{_WAVE_N46_TRICHOTOMY_MEMO} + {_WAVE_N46_MEMO_PATH} (anti-pattern #5); "
            "operator binding correction 2026-05-28 ~23:40Z"
        ),
        canonical_unwind_path=(
            "Set default operator_canonical_metric=HIGHEST_EV_SHORTEST_WALL_CLOCK "
            "in tac.cathedral_autopilot.three_metric_trichotomy. Per "
            "meta_orchestrator_variance_acceptance_dominance_at_low_probability_high_leverage_v1 "
            "canonical equation: when |ΔS| exceeds 10× safest-alternative, "
            "variance-acceptance dominates."
        ),
        canonical_producers=(
            "tools/cathedral_autopilot_autonomous_loop.py",
            "main-thread agent ranking surface",
        ),
        canonical_consumers=(
            "tac.cathedral_autopilot.three_metric_trichotomy:rank_candidates_via_three_metric_trichotomy",
            "src/tac/cathedral_consumers/meta_orchestrator_consumer/",
            "src/tac/cathedral_consumers/anti_pattern_lookup_consumer/",
        ),
        paradigm_class=PARADIGM_RIGOR_LOSS,
        severity=SEVERITY_HIGH,
        provenance=_design_provenance(
            "metric_orthogonality_inversion_variance_aversion_default.v1"
        ),
        empirical_falsifications=(),
        last_recalibration_utc=_WAVE_N46_LANDING_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# ---------------------------------------------------------------------------
# Populate helpers (idempotent; APPEND-ONLY per Catalog #110/#113).
# ---------------------------------------------------------------------------


def build_all_wave_n46_equations() -> list[CanonicalEquation]:
    """Return the 5 Wave N+46 canonical equations (no registry write)."""
    return [
        build_meta_orchestrator_highest_ev_shortest_wall_clock_metric_v1(),
        build_meta_orchestrator_three_metric_trichotomy_orthogonality_v1(),
        build_meta_orchestrator_variance_acceptance_dominance_at_low_probability_high_leverage_v1(),
        build_meta_orchestrator_lesson_set_completeness_lower_bound_v1(),
        build_meta_orchestrator_cap_window_discipline_per_turn_variance_amortization_v1(),
    ]


def build_all_wave_n46_anti_patterns() -> list[AntiPattern]:
    """Return the 5 Wave N+46 canonical anti-patterns (no registry write)."""
    return [
        build_manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1(),
        build_operator_correction_canonical_apparatus_mutation_lag_v1(),
        build_spawn_prompt_boilerplate_duplication_across_subagent_waves_v1(),
        build_hygiene_ev_vs_frontier_breaking_ev_vs_highest_ev_shortest_wall_clock_conflation_canonical_rediscovery_v1(),
        build_metric_orthogonality_inversion_safest_incremental_as_default_when_operator_wants_variance_acceptance_v1(),
    ]


def populate_wave_n46_canonical_apparatus(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> dict[str, list[str]]:
    """Idempotent population of Wave N+46 canonical apparatus.

    Per CLAUDE.md "Locked writes preserve deletions" (Catalog #132):
    APPEND-ONLY — re-running this helper appends new ``registered``
    events. The latest-row-wins query semantics in ``query_equations``
    + ``query_anti_patterns`` ensure consumers see the most recent
    payload.

    Returns a dict with two lists of canonical ids (equations,
    anti_patterns) actually registered.
    """
    equations_registered: list[str] = []
    for eq in build_all_wave_n46_equations():
        register_canonical_equation(
            eq,
            agent=agent,
            subagent_id=subagent_id,
            notes="wave_n46_cathedral_autopilot_extension_canonical_apparatus_mutation",
        )
        equations_registered.append(eq.equation_id)

    anti_patterns_registered: list[str] = []
    for ap in build_all_wave_n46_anti_patterns():
        register_anti_pattern(
            ap,
            agent=agent,
            subagent_id=subagent_id,
            notes="wave_n46_cathedral_autopilot_extension_canonical_apparatus_mutation",
        )
        anti_patterns_registered.append(ap.anti_pattern_id)

    return {
        "equations_registered": equations_registered,
        "anti_patterns_registered": anti_patterns_registered,
    }


__all__ = [
    "build_all_wave_n46_anti_patterns",
    "build_all_wave_n46_equations",
    "build_hygiene_ev_vs_frontier_breaking_ev_vs_highest_ev_shortest_wall_clock_conflation_canonical_rediscovery_v1",
    "build_manual_main_thread_orchestrator_ranking_drift_across_turns_via_ad_hoc_priority_assignment_v1",
    "build_meta_orchestrator_cap_window_discipline_per_turn_variance_amortization_v1",
    "build_meta_orchestrator_highest_ev_shortest_wall_clock_metric_v1",
    "build_meta_orchestrator_lesson_set_completeness_lower_bound_v1",
    "build_meta_orchestrator_three_metric_trichotomy_orthogonality_v1",
    "build_meta_orchestrator_variance_acceptance_dominance_at_low_probability_high_leverage_v1",
    "build_metric_orthogonality_inversion_safest_incremental_as_default_when_operator_wants_variance_acceptance_v1",
    "build_operator_correction_canonical_apparatus_mutation_lag_v1",
    "build_spawn_prompt_boilerplate_duplication_across_subagent_waves_v1",
    "populate_wave_n46_canonical_apparatus",
]
