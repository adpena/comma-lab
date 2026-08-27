# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""SLOT P closure of Slot N Round 1 corrective actions M1 + M4 + M5.

Registers TWO canonical anti-patterns:

 (1) ``apparatus_maintenance_cascade_dominance_v1`` per Slot N M1 HIGH finding.
     Bug class: apparatus_maintenance dominance 7-of-8 landings in 24h window
     per Catalog #300 §Consequence 5 alert-pending threshold (operator-visible
     alert when rigor_overhead + apparatus_maintenance > 60% of T2+ verdicts
     in any 30-day window). Today's 7-cascade landings (G+E2+F+I+H+J+K) all
     classified ``predicted_mission_contribution=apparatus_maintenance``.
     Canonical unwind path: frontier_breaking dispatch via Slot L symposium
     TOP-3 PROCEED (paired-CUDA $0.18) OR variance-acceptance MLX-LOCAL
     TIME_TRAVELER_Z_FAMILY $0 per Slot M Tier 3 ranking.

 (2) ``equation_one_line_summary_200_char_limit_silently_truncates_registration_v1``
     per Slot N M4 LOW finding. Bug class: ``one_line_summary`` 200-char limit
     per ``src/tac/canonical_equations/equation.py:228-232`` not canonicalized
     as documented contract; Slot I + Slot K both empirically discovered via
     failed registration attempts (recurrence rate 2-of-2 in single session).
     Canonical unwind path: shorten summary preserving semantics; document the
     200-char limit in CLAUDE.md "Canonical equations + models registry"
     non-negotiable; OR extend ``tac.canonical_equations.register_canonical_equation``
     with pre-flight summary-length validation raising operator-friendly error.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable + Catalog #344 canonical registry discipline + Catalog #110/#113
APPEND-ONLY HISTORICAL_PROVENANCE + Catalog #287 placeholder-rationale
rejection sister discipline.

M3 (EmpiricalAnchor schema extension) DEFERRED until post-Slot-O cap-window
to avoid src/tac/canonical_equations/ sister collision risk; M5 (Catalog #363
numbered CLAUDE.md row addition) handled as a sibling Edit operation in the
landing memo's commit batch.
"""
from __future__ import annotations

from tac.canonical_anti_patterns.anti_pattern import (
    AntiPattern,
    PARADIGM_DISCIPLINE,
    PARADIGM_RIGOR_LOSS,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    _utc_now_iso,
)
from tac.canonical_anti_patterns.registry import (
    get_anti_pattern_by_id,
    register_anti_pattern,
)
from tac.provenance.builders import build_provenance_for_research_sidecar


SLOT_P_SUBAGENT_ID = "slot_p_fix_wave_round_1_corrective_actions_m1_m4_m5_per_slot_n_verdict_20260529_0340cst"


# ============================================================================
# M1: apparatus_maintenance_cascade_dominance_v1
# ============================================================================
def build_apparatus_maintenance_cascade_dominance_anti_pattern() -> AntiPattern:
    """Per Slot N Round 1 M1 HIGH cross-landing META-class finding.

    The recurrence is structural: 7-of-7 today's cascade landings declared
    apparatus_maintenance as their mission_predicted_contribution, plus THIS
    Round 1 review itself = 8-of-8 in 24h window. Per Catalog #300 §"Mission
    alignment" Consequence 5: operator-visible alert fires when
    rigor_overhead + apparatus_maintenance > 60% of T2+ verdicts in 30-day
    window. Without structural recognition as a canonical anti-pattern,
    cap-windows can silently accumulate apparatus-maintenance dominance
    indefinitely.
    """
    prov = build_provenance_for_research_sidecar(
        sidecar_path=(
            "~/.claude/projects/-Users-adpena-Projects-pact/memory/"
            "feedback_slot_n_recursive_adversarial_review_round_1_on_today_"
            "7_cascade_landings_landed_20260529.md"
        ),
        reactivation_criteria=(
            "META anti-pattern: apparatus_maintenance dominance across multiple "
            "consecutive cascade landings in a single cap-window, at the expense "
            "of frontier_breaking class-shift substrate work per CLAUDE.md "
            "'Mission alignment - non-negotiable' Consequence 4 ('frontier-breaking "
            "moves DOMINATE rigor budget') + standing directive 'PACT-NeRV + LONG "
            "ORIGINAL SUBSTRATE TRAINING + CLASS/PARADIGM-SHIFT = TOP STANDING "
            "PRIORITY' per feedback_pact_nerv_long_substrate_class_paradigm_shift_"
            "top_priority_20260527.md. The bug class: structurally-sound apparatus "
            "growth (canonical equations + anti-patterns + cathedral consumers + "
            "STRICT preflight gates + canonical posterior anchors) can compound "
            "indefinitely WITHOUT producing frontier-breaking score lowering if "
            "operator-attention budget is silently absorbed by apparatus_maintenance "
            "work instead of routed to frontier-breaking substrate dispatches. "
            "Reactivation criteria: 3+ empirical falsifications across distinct "
            "cap-windows where apparatus-maintenance dominance correlates with "
            "frontier-stagnation OR 1 PARADIGM-LEVEL falsification proving "
            "apparatus_maintenance work itself unblocks frontier-breaking (e.g. "
            "Catalog #335 canonical cathedral consumer auto-discovery genuinely "
            "enabling Slot 1 Compound C-class composition discoveries)."
        ),
    )
    return AntiPattern(
        anti_pattern_id="apparatus_maintenance_cascade_dominance_v1",
        description=(
            "Cascade of consecutive landings declaring mission_predicted_contribution="
            "apparatus_maintenance per Catalog #300 v2 frontmatter, without "
            "frontier_breaking class-shift substrate work in the same cap-window. "
            "Triggers Catalog #300 Consequence 5 operator-visible alert when "
            "rigor_overhead + apparatus_maintenance > 60% of T2+ verdicts in 30d. "
            "Empirical anchor: today's 7-cascade (G+E2+F+I+H+J+K) + Round 1 "
            "review = 8-of-8 apparatus_maintenance in 24h window."
        ),
        forbidden_pattern_predicate=(
            "≥3 consecutive T2+ council deliberations in single cap-window all "
            "declare council_predicted_mission_contribution in {apparatus_maintenance, "
            "rigor_overhead} AND zero declare frontier_breaking alone AND no "
            "frontier_breaking substrate dispatch fired in same cap-window AND "
            "operator-attention budget alert not surfaced per Catalog #300 §"
            "Consequence 5"
        ),
        falsification_band={
            "cascade_landings_apparatus_maintenance_count": 8.0,
            "cascade_landings_frontier_breaking_alone_count": 0.0,
            "operator_attention_budget_apparatus_fraction": 1.0,
            "operator_attention_budget_alert_threshold": 0.60,
        },
        recurrence_conditions=(
            "cap-window receives 3+ consecutive apparatus_maintenance landings",
            "zero frontier_breaking class-shift substrate dispatches in cap-window",
            "operator-attention budget alert per Catalog #300 §Consequence 5 not yet fired",
            "operator routing cascade does not explicitly re-route to frontier_breaking",
            "council Assumption-Adversary axis-8 cargo-cult finding flags pattern but ranks SOFT",
        ),
        canonical_source_anchor=(
            "CLAUDE.md 'Mission alignment - non-negotiable' Consequence 4 "
            "('frontier-breaking moves DOMINATE rigor budget') + Consequence 5 "
            "(60% threshold alert) + standing directive "
            "'PACT-NeRV + LONG ORIGINAL SUBSTRATE TRAINING + CLASS/PARADIGM-SHIFT "
            "= TOP STANDING PRIORITY' per feedback_pact_nerv_long_substrate_class_"
            "paradigm_shift_top_priority_20260527.md; sister memo "
            "feedback_why_our_candidates_lose_to_pr_95_family_canonical_diagnosis_"
            "20260528.md verbatim 'Fix IS NOT more apparatus — per-substrate UNIQUE-"
            "AND-COMPLETE-PER-METHOD discipline EVERY time'; Slot N Round 1 M1 "
            "HIGH finding 2026-05-29 confirmed 8-of-8 apparatus_maintenance in 24h"
        ),
        canonical_unwind_path=(
            "(a) Operator routes next cap-window to frontier_breaking class-shift "
            "substrate work per Slot L symposium TOP-3 PROCEED (paired-CUDA ~$0.18 "
            "total) OR variance-acceptance MLX-LOCAL TIME_TRAVELER_Z_FAMILY $0 per "
            "Slot M Tier 3 ranking; (b) extend tools/audit_council_tier_cadence.py "
            "with apparatus_maintenance dominance alert firing AT (not after) 60% "
            "threshold per Catalog #300 §Consequence 5; (c) cathedral autopilot "
            "ranker per Catalog #379 META-orchestrator 3-metric trichotomy "
            "down-weights apparatus_maintenance candidates when last 5 cap-windows "
            "all apparatus_maintenance; (d) Round 2 self-reflection per Catalog "
            "#363 surfaces 'predicted_mission_contribution distribution audit' as "
            "explicit assumption to verify per cap-window."
        ),
        canonical_producers=(
            "tools/register_slot_p_fix_wave_round_1_corrective_actions_m1_m4_anti_patterns.py",
            "tools/audit_council_tier_cadence.py",
            ".omx/state/council_deliberation_posterior.jsonl",
        ),
        canonical_consumers=(
            "tac.cathedral_consumers.anti_pattern_lookup_consumer",
            "tools/audit_council_tier_cadence.py",
            "tools/list_canonical_anti_patterns.py",
            "tac.cathedral_autopilot.three_metric_trichotomy",
        ),
        paradigm_class=PARADIGM_RIGOR_LOSS,
        severity=SEVERITY_HIGH,
        provenance=prov,
        empirical_falsifications=(),
        last_recalibration_utc=_utc_now_iso(),
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


# ============================================================================
# M4: equation_one_line_summary_200_char_limit_silently_truncates_registration_v1
# ============================================================================
def build_equation_summary_200_char_limit_anti_pattern() -> AntiPattern:
    """Per Slot N Round 1 M4 LOW finding.

    Slot E2 Phase 0 hit the 200-char ``one_line_summary`` limit (7 of 10
    equations failed first-pass); Slot I Phase 1 hit the SAME limit (4 of 11
    failed first-pass). 2-of-2 recurrence in single session × same canonical
    contract. The 200-char limit is enforced at construction time in
    ``tac.canonical_equations.equation.py:228-232`` but NOT documented in
    CLAUDE.md "Canonical equations + models registry" non-negotiable, NOT
    surfaced via pre-flight validator, and NOT canonicalized as a recurrence
    pattern despite 2-of-2 same-session manifestation.
    """
    prov = build_provenance_for_research_sidecar(
        sidecar_path=(
            "~/.claude/projects/-Users-adpena-Projects-pact/memory/"
            "feedback_slot_n_recursive_adversarial_review_round_1_on_today_"
            "7_cascade_landings_landed_20260529.md"
        ),
        reactivation_criteria=(
            "META-meta anti-pattern: canonical contract limit (one_line_summary "
            "<= 200 chars per src/tac/canonical_equations/equation.py:228-232) "
            "rediscovered via failed-then-fixed registration attempts across "
            "consecutive subagents in the same session, instead of being "
            "documented as a canonical contract OR surfaced via pre-flight "
            "validation with operator-friendly error. The bug class is "
            "STRUCTURAL: any canonical contract that surfaces only via "
            "construction-time failure (vs. pre-flight validator OR documented "
            "in CLAUDE.md non-negotiable) will silently recur across subagent "
            "waves until canonicalized. Reactivation criteria: (a) extend "
            "tac.canonical_equations.register_canonical_equation with pre-flight "
            "summary-length validator raising InvalidEquationError BEFORE "
            "construction with operator-friendly hint to shorten; OR (b) document "
            "the 200-char limit in CLAUDE.md 'Canonical equations + models "
            "registry - NON-NEGOTIABLE' section; OR (c) extend Catalog #348 "
            "retroactive sweep memo to flag undocumented canonical contract "
            "limits whenever a 2-of-2 same-session recurrence pattern surfaces."
        ),
    )
    return AntiPattern(
        anti_pattern_id="equation_one_line_summary_200_char_limit_silently_truncates_registration_v1",
        description=(
            "Canonical equation `one_line_summary` field has 200-char limit "
            "enforced at construction time (tac.canonical_equations.equation.py:"
            "228-232) but NOT documented in CLAUDE.md non-negotiable + NOT "
            "surfaced via pre-flight validator. Subagents register, fail with "
            "InvalidEquationError, retry with shortened summary. 2-of-2 recurrence "
            "in same session (Slot E2 7-of-10 first-pass failure + Slot I 4-of-11 "
            "first-pass failure on 2026-05-29) confirms the pattern is STRUCTURAL "
            "not incidental."
        ),
        forbidden_pattern_predicate=(
            "subagent invokes tac.canonical_equations.register_canonical_equation "
            "with one_line_summary > 200 chars AND retries with shortened summary "
            "AFTER InvalidEquationError raised, instead of pre-flight validation "
            "raising operator-friendly hint BEFORE construction OR canonical "
            "limit documented in CLAUDE.md 'Canonical equations + models "
            "registry' non-negotiable"
        ),
        falsification_band={
            "slot_e2_first_pass_failure_count": 7.0,
            "slot_e2_total_equations_registered": 10.0,
            "slot_i_first_pass_failure_count": 4.0,
            "slot_i_total_equations_registered": 11.0,
            "recurrence_rate_same_session": 1.0,
        },
        recurrence_conditions=(
            "subagent registers multiple canonical equations in single session",
            "subagent does not know canonical 200-char one_line_summary limit",
            "no pre-flight summary-length validator in tac.canonical_equations.register_canonical_equation",
            "no documentation of 200-char limit in CLAUDE.md 'Canonical equations + models registry'",
            "InvalidEquationError surfaces limit ONLY at construction time",
        ),
        canonical_source_anchor=(
            "Slot N Round 1 M4 LOW finding 2026-05-29; empirical recurrence "
            "anchors: Slot E2 (commit 4f5e8c2 series; 7-of-10 first-pass failure) "
            "+ Slot I (commit fa48a8 series; 4-of-11 first-pass failure); enforcement "
            "site: tac.canonical_equations.equation.py:228-232; META-class anchor: "
            "CLAUDE.md 'Bugs must be permanently fixed AND self-protected against' "
            "non-negotiable (a 2-of-2 same-session recurrence should be canonicalized "
            "either as anti-pattern OR as documented contract OR as pre-flight gate)"
        ),
        canonical_unwind_path=(
            "(a) Shorten one_line_summary preserving semantics (operator-side fix "
            "per subagent); (b) extend tac.canonical_equations.register_canonical_equation "
            "with pre-flight validator raising operator-friendly hint BEFORE "
            "construction; (c) document the 200-char limit in CLAUDE.md 'Canonical "
            "equations + models registry - NON-NEGOTIABLE' section so future "
            "subagents know the contract at design time; (d) sister Catalog "
            "#287-style same-line placeholder-rejection pattern: refuse summaries "
            "matching `<.*>` placeholder regex even if under 200 chars."
        ),
        canonical_producers=(
            "tools/register_slot_p_fix_wave_round_1_corrective_actions_m1_m4_anti_patterns.py",
            "src/tac/canonical_equations/equation.py",
        ),
        canonical_consumers=(
            "tac.cathedral_consumers.anti_pattern_lookup_consumer",
            "tools/list_canonical_anti_patterns.py",
            "tac.canonical_equations.register_canonical_equation",
        ),
        paradigm_class=PARADIGM_DISCIPLINE,
        severity=SEVERITY_LOW,
        provenance=prov,
        empirical_falsifications=(),
        last_recalibration_utc=_utc_now_iso(),
        next_recalibration_trigger=RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )


def main() -> int:
    # ---- M1: apparatus_maintenance_cascade_dominance_v1 ----
    m1_id = "apparatus_maintenance_cascade_dominance_v1"
    existing_m1 = get_anti_pattern_by_id(m1_id)
    if existing_m1 is None:
        m1 = build_apparatus_maintenance_cascade_dominance_anti_pattern()
        register_anti_pattern(
            m1,
            subagent_id=SLOT_P_SUBAGENT_ID,
            agent="claude",
            notes=(
                "SLOT P closure of Slot N Round 1 M1 HIGH cross-landing META-class "
                "finding (apparatus_maintenance dominance 7-of-7 cascade landings + "
                "Round 1 = 8-of-8 in 24h window)."
            ),
        )
        print(f"REGISTERED anti-pattern: {m1_id}")
    else:
        print(f"SKIPPED already-registered anti-pattern: {m1_id}")

    # ---- M4: equation_one_line_summary_200_char_limit_silently_truncates_registration_v1 ----
    m4_id = "equation_one_line_summary_200_char_limit_silently_truncates_registration_v1"
    existing_m4 = get_anti_pattern_by_id(m4_id)
    if existing_m4 is None:
        m4 = build_equation_summary_200_char_limit_anti_pattern()
        register_anti_pattern(
            m4,
            subagent_id=SLOT_P_SUBAGENT_ID,
            agent="claude",
            notes=(
                "SLOT P closure of Slot N Round 1 M4 LOW META-meta-recurrence finding "
                "(Slot E2 7-of-10 + Slot I 4-of-11 first-pass one_line_summary 200-char "
                "limit failures in single session)."
            ),
        )
        print(f"REGISTERED anti-pattern: {m4_id}")
    else:
        print(f"SKIPPED already-registered anti-pattern: {m4_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
