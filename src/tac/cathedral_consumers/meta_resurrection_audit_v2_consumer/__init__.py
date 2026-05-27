# SPDX-License-Identifier: MIT
"""Cathedral consumer: META-RESURRECTION-AUDIT-V2 META-bug amplification detector.

Op-routables Item #3 of META-RESURRECTION-AUDIT-V2 per
``.omx/research/meta_resurrection_audit_v2_inherently_broken_implementations_landed_20260526.md``
+ ``.omx/research/meta_resurrection_audit_v2_op_routables_canonicalization_wave_landed_20260527.md``.

Per Catalog #335 canonical ``CathedralConsumerContract`` + Catalog #341
canonical-routing markers + 6-hook wire-in per Catalog #125 + Catalog #287
placeholder-rationale rejection + Catalog #323 canonical Provenance umbrella
sister at the cathedral consumer surface.

Per the operator's 7th META AUTOMATED+COMPOUNDING+OPTIMAL standing directive:
this Tier-A observability-only cathedral consumer auto-discovered per Catalog
#336/#337 surfaces a per-candidate META-bug-class amplification-detector verdict
- WHICH of the 10 META-bug classes (M1-M10) a candidate's prior negative result
(KILL / DEFER / FALSIFIED) may have suffered - WITHOUT mutating the ranker's
predicted delta (Tier A invariant per Catalog #341).

This makes the 85% over-kill insight from META-RESURRECTION-AUDIT-V2 STRUCTURAL:
when a candidate carries a prior negative-result verdict, the consumer routes a
``META_BUG_AMPLIFICATION_SUSPECTED`` annotation citing the matched META-bug class
+ the canonical equation #344 amplification-factor equation + the canonical
reactivation path - so the cathedral autopilot ranker can see WHICH candidates
are resurrection candidates (prior negative result likely amplified by a broken
implementation) vs WHICH are genuine paradigm refutations (Tier-3 KEEP-KILLED).

The 10 META-bug classes (per audit memo §2.1):

* M1 WRONG-BASELINE-mistaken-for-paradigm-failure
* M2 CARGO-CULT-TECHNIQUE-FAMILY-mistaken-for-exhaustive-research
* M3 SYNTHETIC-FALLBACK-mistaken-for-paradigm-failure
* M4 PHANTOM-SCORE-DIRECTORY-mistaken-for-axis-truth
* M5 PREDICTED-BAND-FROM-RANDOM-INIT-mistaken-for-falsified-prediction
* M6 RECIPE-vs-TRAINER-STATE-divergence-mistaken-for-substrate-failure
* M7 OPERATING-POINT-SATURATION-mistaken-for-paradigm-floor
* M8 WRONG-CANONICAL-APPLICATION-SURFACE-mistaken-for-paradigm-null
* M9 GENERIC-LAGRANGIAN-mistaken-for-substrate-unique-optimization
* M10 SILENT-DEFAULT-MASQUERADING-AS-NEGATIVE-RESULT

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog
#307 paradigm-vs-implementation classification: this consumer does NOT reopen
lanes, does NOT change kill verdicts, and does NOT promote any candidate. It is
the canonical disambiguator at the cathedral ranker surface between
implementation-level-falsification-likely-amplified vs genuine-paradigm-
refutation, surfacing the operator-routable reactivation path per Catalog #308.

Sister of:
  - ``tac.cathedral_consumers._example_consumer`` (canonical reference)
  - ``tac.cathedral_consumers.pr_submission_compliance_consumer`` (Phase 8 sister)
  - ``tac.canonical_equations`` (5 FORMALIZATION_PENDING amplification equations)
  - ``tac.probe_outcomes_ledger`` (TOP-5 DEFERRED-pending-resurrection rows)

Hooks (per Catalog #125 6-hook wire-in non-negotiable):
  * Hook #1 SENSITIVITY_MAP - N/A (defensive observability consumer)
  * Hook #2 PARETO_CONSTRAINT - N/A
  * Hook #3 BIT_ALLOCATOR - N/A
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH - ACTIVE PRIMARY (this IS the consumer;
    surfaces per-candidate META-bug amplification-detector verdict for
    resurrection-candidate dispatch ranking)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR - ACTIVE (per-candidate META-bug-class
    verdict feeds the canonical posterior; once a resurrection candidate's
    paired-axis empirical anchor lands, the matched amplification equation gets a
    NEW empirical anchor via tac.canonical_equations.update_equation_with_empirical_anchor)
  * Hook #6 PROBE_DISAMBIGUATOR - ACTIVE (the 10-class META-bug taxonomy IS the
    disambiguator between implementation-falsification-amplified vs genuine-
    paradigm-refutation at the broader META-adjudication-methodology surface)
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "meta_resurrection_audit_v2"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


# Canonical mapping: META-bug class -> (canonical equation #344 id, canonical
# structural-extinction surface, canonical reactivation-path summary). Per
# audit memo §2.1 + §5. Each amplification equation is FORMALIZATION_PENDING
# per Catalog #344 (promotion requires 3+ in-domain empirical anchors).
META_BUG_CLASSES: dict[str, dict[str, str]] = {
    "M1": {
        "label": "WRONG-BASELINE-mistaken-for-paradigm-failure",
        "equation_id": "wrong_baseline_substitution_score_amplification_v1",
        "extinction": "catalog_343_frontier_pointer + catalog_368_substitution_baseline",
        "reactivation": "re-measure delta vs canonical_frontier_pointer baseline per Catalog #343",
    },
    "M2": {
        "label": "CARGO-CULT-TECHNIQUE-FAMILY-mistaken-for-exhaustive-research",
        "equation_id": "cargo_cult_technique_family_selection_negative_result_amplification_v1",
        "extinction": "claude_md_forbidden_premature_kill + catalog_308_alternative_reducer_enumeration",
        "reactivation": "enumerate N>=3 alternative configs in technique family per Catalog #308",
    },
    "M3": {
        "label": "SYNTHETIC-FALLBACK-mistaken-for-paradigm-failure",
        "equation_id": "synthetic_fallback_implementation_negative_result_amplification_v1",
        "extinction": "catalog_369_inflate_real_trained_weights + catalog_220_operational_mechanism",
        "reactivation": "re-run with real-frame-derived-from-trained-weights / correct input flow",
    },
    "M4": {
        "label": "PHANTOM-SCORE-DIRECTORY-mistaken-for-axis-truth",
        "equation_id": "",  # covered by sister Catalog #249; no NEW equation
        "extinction": "catalog_249_filename_auto_redirect + catalog_127_custody_validator",
        "reactivation": "re-run on contest-CUDA axis per CLAUDE.md MPS-auth-eval-is-NOISE",
    },
    "M5": {
        "label": "PREDICTED-BAND-FROM-RANDOM-INIT-mistaken-for-falsified-prediction",
        "equation_id": "",  # covered by sister T3 implementation_level_falsification_recovery
        "extinction": "catalog_324_post_training_tier_c_validation + catalog_296_dykstra_feasibility",
        "reactivation": "re-derive predicted band post-training per Catalog #324",
    },
    "M6": {
        "label": "RECIPE-vs-TRAINER-STATE-divergence-mistaken-for-substrate-failure",
        "equation_id": "",  # covered by sister Catalog #240; no NEW equation
        "extinction": "catalog_240_recipe_trainer_chain_canonical",
        "reactivation": "reconcile recipe-vs-trainer-state per Catalog #240",
    },
    "M7": {
        "label": "OPERATING-POINT-SATURATION-mistaken-for-paradigm-floor",
        "equation_id": "",  # covered by sister T3 operating_point_pose_seg_marginal_crossover
        "extinction": "catalog_356_per_axis_decomposition + catalog_219_z1_mdl_density_threshold",
        "reactivation": "re-attack at operating-point-aware marginal per Catalog #356",
    },
    "M8": {
        "label": "WRONG-CANONICAL-APPLICATION-SURFACE-mistaken-for-paradigm-null",
        "equation_id": "wrong_canonical_application_surface_paradigm_null_amplification_v1",
        "extinction": "catalog_290_canonical_vs_unique_decision_per_layer + catalog_303_cargo_cult_audit",
        "reactivation": "apply technique to substrate whose math structure matches its canonical assumption",
    },
    "M9": {
        "label": "GENERIC-LAGRANGIAN-mistaken-for-substrate-unique-optimization",
        "equation_id": "generic_shared_helper_vs_individually_fractal_negative_amplification_v1",
        "extinction": "claude_md_unique_and_complete_per_method + catalog_290 + 8th_individually_fractal_directive",
        "reactivation": "per-substrate-unique-coupling-aware Lagrangian/loss/codec per UNIQUE-AND-COMPLETE-PER-METHOD",
    },
    "M10": {
        "label": "SILENT-DEFAULT-MASQUERADING-AS-NEGATIVE-RESULT",
        "equation_id": "synthetic_fallback_implementation_negative_result_amplification_v1",
        "extinction": "29th_metabug_check_no_silent_auto_discovery_with_warn",
        "reactivation": "re-run with correct (non-silent-default) inputs; assert internal stats consistency",
    },
}

# Verdict tokens that signal a prior negative-result adjudication on the
# candidate (KILL / DEFER / FALSIFIED). When a candidate's metadata carries any
# of these AND a META-bug-class label, the consumer routes the amplification
# annotation.
_NEGATIVE_VERDICT_TOKENS = (
    "kill",
    "killed",
    "falsified",
    "falsify",
    "defer",
    "deferred",
    "retired",
    "dead",
    "negative_result",
    "withdrawn",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    Per op-routables Item #3 scope: this consumer is observability-only at
    landing. When a resurrection candidate's paired-axis empirical anchor lands,
    a future subagent wires the matched META-bug amplification equation's NEW
    empirical anchor via ``tac.canonical_equations.update_equation_with_empirical_anchor``
    once 3+ in-domain anchors land per Catalog #344 promotion discipline.
    """
    _ = anchor  # explicit acknowledgment per reference _example_consumer pattern


def _detect_meta_bug_class(candidate: Mapping[str, Any]) -> str | None:
    """Return the matched META-bug class key (M1-M10) or None.

    Detection cascade (most specific first):
      (1) explicit ``meta_bug_class`` field (e.g. "M2") on the candidate;
      (2) ``meta_bug_class`` embedded in candidate notes/verdict text;
      (3) None if no META-bug-class signal is present.
    """
    explicit = candidate.get("meta_bug_class")
    if isinstance(explicit, str):
        key = explicit.strip().upper()
        # Accept "M2" or "M2_cargo_cult..." forms.
        if key in META_BUG_CLASSES:
            return key
        for mk in META_BUG_CLASSES:
            if key.startswith(mk + "_") or key.startswith(mk + " "):
                return mk
    # Embedded in free text. Use a word-boundary regex so "M11" does not match
    # "M1" (the bug class keys are M1-M10; a trailing digit must NOT follow).
    import re

    for text_field in ("notes", "verdict", "rationale", "reactivation_criteria"):
        val = candidate.get(text_field)
        if not isinstance(val, str):
            continue
        upper = val.upper()
        for mk in META_BUG_CLASSES:
            # Match "META-BUG M2" / "META-BUG-CLASS M2" / "(M2)" / "(M2 " forms,
            # with a word boundary AFTER the key so M1 != M11/M10-suffix overlap.
            # \b after \d ensures "M1" does not match inside "M11".
            ctx = rf"(META-BUG[ -](?:CLASS )?|\(){re.escape(mk)}\b"
            if re.search(ctx, upper):
                return mk
    return None


def _carries_negative_verdict(candidate: Mapping[str, Any]) -> bool:
    """True if the candidate carries a prior KILL/DEFER/FALSIFIED signal."""
    for field_name in ("prior_verdict", "verdict", "status", "lane_state", "notes"):
        val = candidate.get(field_name)
        if not isinstance(val, str):
            continue
        lower = val.lower()
        for token in _NEGATIVE_VERDICT_TOKENS:
            if token in lower:
                return True
    return False


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Returns the canonical Tier-A observability-only contribution dict per
    Catalog #341 routing markers: zero ``predicted_delta_adjustment``, never
    ``promotable``, ``axis_tag="[predicted]"``. Promotion of any META-bug
    amplification verdict to a contest score is structurally forbidden per
    CLAUDE.md "Forbidden premature KILL" + Catalog #307 paradigm-vs-implementation
    classification - this consumer is a DISAMBIGUATOR, not a score signal.

    The contribution surfaces a per-candidate META-bug-class amplification-
    detector verdict:
      * ``META_BUG_AMPLIFICATION_SUSPECTED`` - candidate carries a prior negative
        verdict AND a matched META-bug class; surfaces the matched class label +
        canonical amplification equation #344 id + reactivation path.
      * ``GENUINE_PARADIGM_REFUTATION_OR_NO_META_BUG`` - candidate carries a prior
        negative verdict but NO matched META-bug class (Tier-3 KEEP-KILLED OR
        unclassified).
      * ``NO_NEGATIVE_VERDICT`` - candidate carries no prior negative verdict;
        not a resurrection candidate.
    """
    if not isinstance(candidate, Mapping):
        return _neutral_contribution(
            "Candidate is not a Mapping; META-bug amplification verdict unknown."
        )

    carries_negative = _carries_negative_verdict(candidate)
    meta_bug_key = _detect_meta_bug_class(candidate)

    if not carries_negative and meta_bug_key is None:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "Candidate carries no prior KILL/DEFER/FALSIFIED verdict and no "
                "META-bug-class signal; not a META-RESURRECTION-AUDIT-V2 "
                "resurrection candidate."
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "meta_resurrection_verdict": "NO_NEGATIVE_VERDICT",
            "matched_meta_bug_class": None,
        }

    if meta_bug_key is not None:
        meta = META_BUG_CLASSES[meta_bug_key]
        eq_id = meta["equation_id"]
        eq_clause = (
            f"canonical amplification equation #344 {eq_id!r} (FORMALIZATION_PENDING; "
            f"promotion requires 3+ in-domain empirical anchors)"
            if eq_id
            else "no dedicated amplification equation (covered by sister Catalog gate)"
        )
        rationale = (
            f"META_BUG_AMPLIFICATION_SUSPECTED: candidate's prior negative-result "
            f"verdict matches META-bug class {meta_bug_key} "
            f"({meta['label']}). Per META-RESURRECTION-AUDIT-V2 85% over-kill "
            f"insight + Catalog #307 paradigm-vs-implementation: this is LIKELY an "
            f"implementation-level falsification amplified by a broken "
            f"implementation, NOT a paradigm refutation. {eq_clause}. "
            f"Structural extinction surface: {meta['extinction']}. "
            f"Operator-routable reactivation path (Catalog #308): "
            f"{meta['reactivation']}. Per CLAUDE.md 'Forbidden premature KILL' "
            f"this is DEFERRED-pending-resurrection, NOT a confirmed kill; final "
            f"resurrection dispatch is operator-gated."
        )
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": rationale,
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "meta_resurrection_verdict": "META_BUG_AMPLIFICATION_SUSPECTED",
            "matched_meta_bug_class": meta_bug_key,
            "matched_meta_bug_label": meta["label"],
            "amplification_equation_id": eq_id or None,
            "structural_extinction_surface": meta["extinction"],
            "reactivation_path": meta["reactivation"],
        }

    # Carries a negative verdict but no matched META-bug class.
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "Candidate carries a prior KILL/DEFER/FALSIFIED verdict but no "
            "matched META-bug class (M1-M10). Either a Tier-3 KEEP-KILLED genuine "
            "paradigm refutation (per audit §3.5) OR an unclassified verdict "
            "operator-routable for re-classification per Catalog #307. No "
            "amplification suspected without a matched META-bug class."
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "meta_resurrection_verdict": "GENUINE_PARADIGM_REFUTATION_OR_NO_META_BUG",
        "matched_meta_bug_class": None,
    }


def _neutral_contribution(rationale: str) -> Mapping[str, Any]:
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "meta_resurrection_verdict": "UNKNOWN",
        "matched_meta_bug_class": None,
    }
