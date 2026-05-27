# SPDX-License-Identifier: MIT
"""Cathedral consumer: PR-submission canonical compliance gate readiness annotation.

Phase 8 (Layer 6) of the canonical submission pipeline per
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
§3 Phase 8 + Catalog #370 STRICT preflight gate self-protection.

Per Catalog #335 canonical ``CathedralConsumerContract`` + Catalog #341
canonical-routing markers + 6-hook wire-in per Catalog #125 + Catalog #287
placeholder-rationale rejection + Catalog #323 canonical Provenance umbrella
sister at the cathedral consumer surface.

Per the 12th canonicalization x standardization x ease-of-contest-compliance
trinity: this Tier-A observability-only cathedral consumer auto-discovered per
Catalog #336/#337 surfaces PR-submission canonical-compliance readiness per
candidate so the cathedral autopilot ranker can see WHICH candidates are
PR-SUBMISSION-READY (all 4 verdicts present + clean) vs WHICH carry missing or
failed canonical verdicts (BLOCKED_ON_<phase>) - without mutating the ranker's
predicted delta (Tier A invariant per Catalog #341).

The 4 canonical verdicts the consumer routes:

* **Phase 4 builder** ``SubmissionBundleResult`` (canonical helper
  ``tac.submission_packet.build_submission_bundle``)
* **Phase 5 linter** ``LintVerdict`` (canonical helper
  ``tac.submission_packet.lint_submission_bundle``)
* **Phase 6 compliance** ``ComplianceVerdict`` (canonical helper
  ``tac.submission_packet.enforce_contest_compliance``)
* **Phase 7 paired_auth_eval** ``PairedAuthEvalVerdict`` (canonical helper
  ``tac.submission_packet.orchestrate_paired_auth_eval``)

Per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA, ON 1:1 CONTEST-
COMPLIANT HARDWARE" non-negotiable + Catalog #192 macOS-CPU non-promotion:
the consumer specifically surfaces a ``BLOCKED_ON_PAIRED_AUTH_EVAL`` verdict
when the Phase 7 paired_auth_eval sidecar is absent OR not in ``PAIRED_PASS``
state. This is the canonical disambiguator at the cathedral ranker surface for
the structural enforcement that no PR submission ships without paired-axis
empirical evidence on 1:1 contest-compliant hardware.

Sister of:
  - ``tac.cathedral_consumers._example_consumer`` (canonical reference)
  - ``tac.cathedral_consumers.compression_pipeline_readiness_consumer`` (Phase 2)
  - ``tac.cathedral_consumers.archive_grammar_builder_consumer`` (Phase 3)
  - ``tac.cathedral_consumers.submission_bundle_builder_consumer`` (Phase 4)
  - ``tac.cathedral_consumers.submission_linter_consumer`` (Phase 5)
  - ``tac.cathedral_consumers.submission_compliance_consumer`` (Phase 6)
  - ``tac.cathedral_consumers.paired_auth_eval_consumer`` (Phase 7)
  - ``tac.preflight.check_no_pr_submission_without_canonical_compliance_verdict``
    (this consumer's structural sister gate)

Hooks (per Catalog #125 6-hook wire-in non-negotiable):
  * Hook #1 SENSITIVITY_MAP - N/A (defensive observability consumer)
  * Hook #2 PARETO_CONSTRAINT - N/A
  * Hook #3 BIT_ALLOCATOR - N/A
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH - ACTIVE PRIMARY (this IS the
    consumer; surfaces per-candidate PR-submission canonical readiness for
    dispatch ranking)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR - ACTIVE (per-candidate PR-submission
    readiness verdict feeds the canonical posterior so Phase 10 PR111-candidate
    landings inherit the apriori canonical-compliance signal)
  * Hook #6 PROBE_DISAMBIGUATOR - ACTIVE (PR_READY vs BLOCKED_ON_<phase> IS
    the canonical disambiguator between full-canonical-pipeline-completion vs
    operator-routable blocker at the cathedral ranker surface)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "pr_submission_compliance_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


# The four canonical verdicts the consumer routes (Layer 2 -> Layer 5 sister
# chain per spec memo). The bug-class anchor: an operator dispatches a PR
# without one or more of these canonical verdicts being present + clean - the
# Phase 8 STRICT preflight gate refuses BEFORE submission; this consumer
# surfaces the readiness state at the autopilot ranker surface for early
# routing.
_REQUIRED_PHASE_VERDICTS = (
    "submission_bundle_result",  # Phase 4 builder
    "lint_verdict",              # Phase 5 linter
    "compliance_verdict",        # Phase 6 compliance
    "paired_auth_eval_verdict",  # Phase 7 paired axes on 1:1 hardware
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    Per Phase 8 scope: this consumer is observability-only at landing.
    Phase 10 future-subagent landings will wire the PR-submission readiness
    verdict anchor into the canonical equation #344 registry via
    ``tac.canonical_equations.update_equation_with_empirical_anchor`` once 3+
    empirical PR-submission anchors land per Catalog #344.
    """
    _ = anchor  # explicit acknowledgment per reference _example_consumer pattern


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Returns the canonical Tier-A observability-only contribution dict per
    Catalog #341 routing markers: zero ``predicted_delta_adjustment``, never
    ``promotable``, ``axis_tag="[predicted]"``. Promotion of any PR-submission
    readiness verdict to a contest score is structurally forbidden per
    CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA" non-negotiable +
    Catalog #192 macOS-CPU non-promotion.

    The contribution surfaces a per-candidate readiness verdict derived from
    which of the 4 canonical verdicts (Phase 4 + 5 + 6 + 7) are present + clean
    on the candidate's metadata. When the candidate carries no metadata, the
    consumer returns a neutral observation without claiming readiness or
    non-readiness.
    """
    if not isinstance(candidate, Mapping):
        return _neutral_contribution(
            "Candidate is not a Mapping; PR-submission readiness unknown."
        )

    # Collect per-phase presence + clean signals from the candidate metadata.
    phase_states: dict[str, dict[str, Any]] = {}
    for verdict_key in _REQUIRED_PHASE_VERDICTS:
        meta = candidate.get(verdict_key)
        if not isinstance(meta, Mapping):
            phase_states[verdict_key] = {"present": False, "clean": False}
            continue
        # Each verdict's canonical helper persists with overall_clean / overall_pass
        # boolean (Phase 4 PASS / Phase 5 overall_clean / Phase 6 overall_clean
        # / Phase 7 PAIRED_PASS). The consumer treats overall_clean=True as
        # canonical clean.
        overall_clean = bool(
            meta.get("overall_clean")
            or meta.get("overall_pass")
            or meta.get("verdict") == "PAIRED_PASS"
            or meta.get("verdict") == "PASS"
        )
        forbidden_macos = bool(
            meta.get("forbidden_macos_axis_detected", False)
        )
        phase_states[verdict_key] = {
            "present": True,
            "clean": overall_clean and not forbidden_macos,
            "forbidden_macos": forbidden_macos,
        }

    # Determine readiness.
    any_forbidden_macos = any(
        s.get("forbidden_macos", False) for s in phase_states.values()
    )
    missing_phases = tuple(
        k for k, s in phase_states.items() if not s["present"]
    )
    unclean_phases = tuple(
        k for k, s in phase_states.items() if s["present"] and not s["clean"]
    )

    if any_forbidden_macos:
        readiness_verdict = "BLOCKED_FORBIDDEN_MACOS_AXIS"
        rationale = (
            "PR-submission BLOCKED per Catalog #192 + CLAUDE.md 'Submission "
            "auth eval - BOTH CPU AND CUDA' non-negotiable: one of the "
            "canonical verdicts references macOS / Darwin ARM64 substrate "
            "as authoritative axis. Re-run on Linux x86_64 1:1 contest-"
            "compliant hardware before submission."
        )
    elif missing_phases:
        # Missing one or more canonical verdicts -> blocked on the earliest.
        first_missing = missing_phases[0]
        readiness_verdict = f"BLOCKED_ON_{first_missing.upper()}"
        rationale = (
            f"PR-submission BLOCKED: canonical verdict for {first_missing!r} "
            f"is missing on the candidate. Per Phase 1 spec memo Layer 6 + "
            f"Catalog #370 STRICT preflight gate: 4 canonical verdicts "
            f"(Phase 4 builder + Phase 5 linter + Phase 6 compliance + "
            f"Phase 7 paired_auth_eval) MUST all be present + clean before "
            f"any PR submission. Operator-routable: invoke the canonical "
            f"helper for the missing phase."
        )
    elif unclean_phases:
        first_unclean = unclean_phases[0]
        readiness_verdict = f"BLOCKED_ON_{first_unclean.upper()}_FAILED"
        rationale = (
            f"PR-submission BLOCKED: canonical verdict for {first_unclean!r} "
            f"is present but NOT clean (overall_clean / overall_pass / "
            f"verdict marker indicates failure). Operator-routable: review "
            f"the verdict's blockers + re-run the canonical helper after "
            f"resolving each blocker."
        )
    else:
        readiness_verdict = "PR_READY"
        rationale = (
            "PR-submission canonical compliance verdict CLEAN: all 4 "
            "canonical verdicts (Phase 4 builder + Phase 5 linter + "
            "Phase 6 compliance + Phase 7 paired_auth_eval) present + "
            "clean per Phase 1 spec memo Layer 6 + Catalog #370 STRICT "
            "preflight gate. Submission packet is structurally PR-ready per "
            "CLAUDE.md non-negotiables. Final `gh pr create` remains "
            "operator-gated per CLAUDE.md 'Executing actions with care'."
        )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "readiness_verdict": readiness_verdict,
        "phase_state_summary": {
            verdict_key: {
                "present": bool(state["present"]),
                "clean": bool(state["clean"]),
            }
            for verdict_key, state in phase_states.items()
        },
    }


def _neutral_contribution(rationale: str) -> Mapping[str, Any]:
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "readiness_verdict": "UNKNOWN",
        "phase_state_summary": {},
    }
