# SPDX-License-Identifier: MIT
"""SLOT K closure of Slot I op-routables #3 + #2 (sister-equation form).

Registers:
 (1) Canonical anti-pattern ``verdict_re_tagging_kill_to_defer_pending_research_exhaustion_v1``
     per Slot I op-routable #3 (canonical equation form CLOSED Slot I Phase 1; anti-pattern
     form remained queued per cap-discipline + "iterate not force").

 (2) Canonical equation ``uniward_standalone_no_op_on_bitstream_dominated_by_sli1_decoder_cost_v1``
     per Slot I op-routable footnote / Slot E2 Phase D rank 1 (DEFERRED-to-next-cap-window
     from Slot I; THIS wave closes it as the 12th equation in the sister-stack).

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable +
Catalog #344 canonical registry discipline + Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.
"""
from __future__ import annotations

from tac.canonical_anti_patterns.anti_pattern import (
    AntiPattern,
    PARADIGM_PREMATURE_KILL,
    RECALIBRATE_ON_NEW_FALSIFICATIONS as AP_RECAL_ON_NEW_FALS,
    SEVERITY_HIGH,
    _utc_now_iso as ap_utc_now_iso,
)
from tac.canonical_anti_patterns.registry import (
    register_anti_pattern,
    get_anti_pattern_by_id,
)
from tac.canonical_equations.equation import (
    CanonicalEquation,
    RECALIBRATE_ON_NEW_ANCHORS,
    _utc_now_iso as eq_utc_now_iso,
)
from tac.canonical_equations.registry import (
    register_canonical_equation,
    query_equations,
)
from tac.provenance.builders import build_provenance_for_research_sidecar


SLOT_K_SUBAGENT_ID = "slot_k_close_slot_i_deferred_op_routables_20260529_0140cst"


# ============================================================================
# OP-1: anti-pattern verdict_re_tagging_kill_to_defer_pending_research_exhaustion_v1
# ============================================================================
def build_verdict_re_tagging_anti_pattern() -> AntiPattern:
    """Per Slot I op-routable #3 (canonical equation form CLOSED Slot I Phase 1
    entry 2 as ``verdict_re_tagging_kill_to_defer_pending_research_exhaustion_v1``;
    anti-pattern form remains queued per Catalog #344 sister discipline)."""
    prov = build_provenance_for_research_sidecar(
        sidecar_path=(
            "~/.claude/projects/-Users-adpena-Projects-pact/memory/"
            "feedback_adversarial_audit_4_falsifications_DEFERRED_not_killed_20260507.md"
        ),
        reactivation_criteria=(
            "META anti-pattern: prior KILL/FALSIFIED/DEAD/RETIRED verdicts being "
            "re-classified as DEFER-pending-research-exhaustion per CLAUDE.md "
            "'Forbidden premature KILL' non-negotiable hardened 2026-05-05 + "
            "Catalog #307 paradigm-vs-implementation falsification distinction "
            "hardened 2026-05-16. The bug class: verdicts landed pre-discipline-gate "
            "may treat entire technique class as KILLED when only the SPECIFIC "
            "IMPLEMENTATION TESTED was empirically falsified. Reactivation criteria: "
            "Catalog #348 retroactive sweep memo + sister-anchor empirical evidence "
            "of paradigm-vs-implementation distinction proving reactivation paths exist."
        ),
    )
    return AntiPattern(
        anti_pattern_id="verdict_re_tagging_kill_to_defer_pending_research_exhaustion_v1",
        description=(
            "Prior KILL/FALSIFIED/DEAD/RETIRED verdicts being re-classified as "
            "DEFER-pending-research-exhaustion per CLAUDE.md 'Forbidden premature KILL' "
            "+ Catalog #307 paradigm-vs-implementation distinction. Verdicts landed "
            "before 2026-05-05 / 2026-05-16 discipline-gate hardening may treat entire "
            "technique class as KILLED when only the specific implementation tested "
            "was empirically falsified."
        ),
        forbidden_pattern_predicate=(
            "memo body contains KILL/FALSIFIED/DEAD/RETIRED verdict AND landed "
            "BEFORE canonical CLAUDE.md kill-as-last-resort hardening 2026-05-05 OR "
            "Catalog #307 paradigm-vs-implementation gate landing 2026-05-16 AND "
            "no APPEND-ONLY footer re-classification per Catalog #348 retroactive sweep"
        ),
        falsification_band={
            "retroactive_sweep_re_classification_count": 1.0,
            "uniward_kill_2026_04_29_landed_residual": 0.0,
        },
        recurrence_conditions=(
            "verdict landed BEFORE CLAUDE.md 'KILL/FALSIFIED memory verdicts' non-negotiable",
            "verdict landed BEFORE Catalog #307 paradigm-vs-implementation gate landing",
            "verdict landed BEFORE Catalog #308 alternative-probe-methodology enumeration",
            "verdict landed BEFORE Catalog #311 substrate-compatibility-evidence-required gate",
            "verdict assesses standalone-on-current-substrate ONLY; alternative configurations untested",
            "reactivation criteria substantively pinned in source memo but never operationalized",
        ),
        canonical_source_anchor=(
            "CLAUDE.md 'Forbidden premature KILL without research exhaustion' "
            "non-negotiable (hardened 2026-05-05) + 'KILL/FALSIFIED memory verdicts' "
            "non-negotiable + Catalog #307 paradigm-vs-implementation gate (landed "
            "2026-05-16) + Catalog #348 retroactive sweep gate (landed 2026-05-19); "
            "empirical anchor: UNIWARD KILL verdict 2026-04-29 retroactively "
            "re-classified per Slot I Catalog #348 sweep landed 2026-05-29 at "
            ".omx/research/retroactive_sweep_for_uniward_kill_verdict_20260529T055141Z.md"
        ),
        canonical_unwind_path=(
            "(a) Run Catalog #348 retroactive sweep on pre-discipline-gate verdicts; "
            "(b) APPEND-ONLY footer per Catalog #110/#113 HISTORICAL_PROVENANCE "
            "re-classifying KILL-CLASS-LEVEL to IMPLEMENTATION-LEVEL FALSIFICATION "
            "per Catalog #307; (c) update Catalog #313 probe outcomes ledger with "
            "DEFER verdict + 30-day reactivation window; (d) register sister canonical "
            "equation documenting the implementation-level no-op (canonical equation "
            "verdict_re_tagging_kill_to_defer_pending_research_exhaustion_v1 already "
            "CLOSED Slot I Phase 1 entry 2 per Slot E2 axis-5-minor canonical apparatus "
            "mutation gap chain)."
        ),
        canonical_producers=(
            "tools/audit_kill_verdict_compliance_rate.py",
            "tools/register_slot_k_op_routables_anti_pattern_plus_equation.py",
            ".omx/research/retroactive_sweep_for_uniward_kill_verdict_20260529T055141Z.md",
        ),
        canonical_consumers=(
            "tac.cathedral_consumers.anti_pattern_lookup_consumer",
            "tools/audit_kill_verdict_compliance_rate.py",
            "tools/list_canonical_anti_patterns.py",
        ),
        paradigm_class=PARADIGM_PREMATURE_KILL,
        severity=SEVERITY_HIGH,
        provenance=prov,
        empirical_falsifications=(),
        last_recalibration_utc=ap_utc_now_iso(),
        next_recalibration_trigger=AP_RECAL_ON_NEW_FALS,
    )


# ============================================================================
# OP-2: equation uniward_standalone_no_op_on_bitstream_dominated_by_sli1_decoder_cost_v1
# ============================================================================
def build_uniward_standalone_no_op_equation() -> CanonicalEquation:
    """Per Slot I op-routable footnote + Slot E2 Phase D rank 1.

    Mathematical formulation: UNIWARD standalone (encoder-on-current-substrate-
    without-SLI1-decoder) is a NO-OP on the contest bitstream because the
    inflate path never consumes any UNIWARD-emitted side channel. The
    standalone bitstream byte savings ΔS_rate = 0 regardless of UNIWARD
    parameter sweep. SLI1 decoder cost dominates any rate-axis movement.

    Empirical anchor: UNIWARD v8 = 1.14 [Modal-T4-CPU] with full-res masks;
    same archive bytes 694KB as Lane A 1.15 (rate identical; pose+seg worse).
    Bitstream parity-with-baseline IS the empirical receipt that the standalone
    is no-op.
    """
    prov = build_provenance_for_research_sidecar(
        sidecar_path=(
            "~/.claude/projects/-Users-adpena-Projects-pact/memory/"
            "project_council_kill_uniward_20260429.md"
        ),
        reactivation_criteria=(
            "UNIWARD-PARADIGM (adaptive embedding cost per detector blind spot) "
            "MATHEMATICALLY SOUND per Yousfi steelman; standalone-encoder-on-current-"
            "substrate-without-SLI1-decoder IS no-op per empirical receipt (bitstream "
            "parity with baseline). Three reactivation paths pinned per 2026-04-29 "
            "council memo: (a) UNIWARD+SLI1 sister stacking with inflate-time decoder; "
            "(b) Lane LI PoseNet-domain 'learned image' adversarial-cost fork per "
            "Contrarian steelman; (c) UNIWARD-as-TTO-regularizer per Fridrich "
            "implicit-cost observation. THREE empirical anchors required (one per "
            "reactivation path) to demote standalone-no-op falsification to "
            "implementation-context-specific per Catalog #307 + #371 auto-recal."
        ),
    )
    return CanonicalEquation(
        equation_id="uniward_standalone_no_op_on_bitstream_dominated_by_sli1_decoder_cost_v1",
        name="UNIWARD standalone no-op on bitstream dominated by SLI1 decoder cost",
        one_line_summary=(
            "UNIWARD standalone-encoder-without-SLI1-decoder: bitstream byte savings = 0 "
            "(no-op on contest archive); SLI1 decoder cost dominates."
        ),
        latex_form=(
            r"\Delta S_{\text{rate}}^{\text{UNIWARD-standalone}} = 0 \text{ if no SLI1 decoder; "
            r"otherwise } \Delta S \approx \Delta S_{\text{SLI1-decoder-only}}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.builtins"
            ":uniward_standalone_no_op_on_bitstream_dominated_by_sli1_decoder_cost_v1"
        ),
        domain_of_validity={
            "codec_family": "uniward_encoder_family",
            "substrate_class": "pixel_domain_segnet_or_pose_net_scorer_target",
            "payload_class": "bitstream_without_sli1_inflate_decoder",
            "domain_of_validity_included": [
                "uniward_standalone_no_decoder",
                "uniward_v8_pre_sli1_decoder_spec",
            ],
            "domain_of_validity_excluded": [
                "uniward_plus_sli1_inflate_decoder_stacking",
                "uniward_as_tto_regularizer_inflate_unchanged",
                "lane_li_posenet_domain_learned_image_fork",
            ],
        },
        units_in={
            "uniward_parameter_sweep_count": "int",
            "archive_bytes": "bytes",
            "sli1_decoder_present": "bool",
        },
        units_out={
            "delta_s_rate_bytes_predicted": "bytes",
            "delta_s_total_predicted": "score_units",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=eq_utc_now_iso(),
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools/list_canonical_equations.py",
        ),
        canonical_producers=(
            "tools/register_slot_k_op_routables_anti_pattern_plus_equation.py",
            ".omx/research/retroactive_sweep_for_uniward_kill_verdict_20260529T055141Z.md",
        ),
        provenance=prov,
    )


def main() -> int:
    # ---- OP-1: anti-pattern ----
    ap_id = "verdict_re_tagging_kill_to_defer_pending_research_exhaustion_v1"
    existing_ap = get_anti_pattern_by_id(ap_id)
    if existing_ap is None:
        ap = build_verdict_re_tagging_anti_pattern()
        register_anti_pattern(
            ap,
            subagent_id=SLOT_K_SUBAGENT_ID,
            agent="claude",
            notes="SLOT K closure of Slot I op-routable #3 (anti-pattern form); equation form closed Slot I Phase 1 entry 2.",
        )
        print(f"REGISTERED anti-pattern: {ap_id}")
    else:
        print(f"SKIPPED already-registered anti-pattern: {ap_id}")

    # ---- OP-2: equation ----
    eq_id = "uniward_standalone_no_op_on_bitstream_dominated_by_sli1_decoder_cost_v1"
    eqs = query_equations()
    existing_eq = next((e for e in eqs if e.equation_id == eq_id), None)
    if existing_eq is None:
        eq = build_uniward_standalone_no_op_equation()
        register_canonical_equation(
            eq,
            subagent_id=SLOT_K_SUBAGENT_ID,
            agent="claude",
            notes="SLOT K closure of Slot I op-routable footnote (12th canonical equation in sister-stack chain).",
        )
        print(f"REGISTERED equation: {eq_id}")
    else:
        print(f"SKIPPED already-registered equation: {eq_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
