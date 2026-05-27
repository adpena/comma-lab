# SPDX-License-Identifier: MIT
"""Cathedral consumer: submission linter verdict annotation (Phase 5 sister).

Per Phase 1 audit specification memo §3 Phase 5 acceptance + Catalog #335
canonical CathedralConsumerContract + Catalog #341 canonical-routing
markers + 6-hook wire-in per Catalog #125.

Per the 12th canonicalization × standardization × ease-of-contest-compliance
trinity: this Tier-A observability-only cathedral consumer auto-discovered
per Catalog #336/#337 surfaces submission-linter readiness per candidate so
the cathedral autopilot ranker can see WHICH PR-candidate bundles are
LINT-CLEAN versus WHICH carry forbidden tokens / first-person-plural /
emdash / tone-violation / inflate.py over-budget / archive-sha-mismatch
blockers — without mutating the ranker's predicted delta (Tier A invariant
per Catalog #341).

Sister of:
  * :mod:`tac.cathedral_consumers._example_consumer` (canonical reference)
  * :mod:`tac.cathedral_consumers.compression_pipeline_readiness_consumer`
    (Phase 2 sister)
  * :mod:`tac.cathedral_consumers.archive_grammar_builder_consumer`
    (Phase 3 sister)
  * :mod:`tac.submission_packet.linter` (this consumer's data source)

Hooks (per Catalog #125 6-hook wire-in non-negotiable):
  * Hook #1 SENSITIVITY_MAP — N/A (defensive observability consumer)
  * Hook #2 PARETO_CONSTRAINT — N/A
  * Hook #3 BIT_ALLOCATOR — ACTIVE (lint-clean verdict feeds the
    bit-allocator priority cascade so LINT-CLEAN candidates rank ahead
    of BLOCKED candidates for the same predicted delta band)
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE PRIMARY (this IS the
    consumer)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (per-lint-verdict
    anchor feeds the canonical posterior so Phase 6/Phase 10 empirical
    anchor landings inherit the apriori lint signal)
  * Hook #6 PROBE_DISAMBIGUATOR — N/A
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "submission_linter_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per Phase 5 scope: this consumer is observability-only at landing.
    Phase 6 + Phase 10 future-subagent landings will wire the
    submission-linter anchor into the canonical equation #344 registry
    (see ``tac.canonical_equations.update_equation_with_empirical_anchor``).
    """
    _ = anchor  # explicit acknowledgment per reference _example_consumer pattern


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns the canonical Tier-A observability-only contribution dict per
    Catalog #341 routing markers: zero predicted_delta_adjustment, never
    promotable, axis_tag=[predicted]. Promotion of any submission-linter
    readiness signal to a contest score requires paired-axis empirical
    evidence per CLAUDE.md "Submission auth eval BOTH CPU AND CUDA".

    The contribution surfaces a lint readiness rationale derived from the
    candidate's submission-linter metadata (when present). When the
    candidate lacks submission-linter metadata, the consumer returns a
    neutral observation without claiming clean or unclean status.
    """
    lint_meta = (
        candidate.get("submission_linter_verdict")
        if isinstance(candidate, Mapping)
        else None
    )
    rationale = (
        "Submission linter verdict unknown (no metadata on candidate)"
    )
    readiness_verdict = "UNKNOWN"
    if isinstance(lint_meta, Mapping):
        overall_clean = lint_meta.get("overall_clean")
        error_count = lint_meta.get("error_count", 0)
        warn_count = lint_meta.get("warn_count", 0)
        info_count = lint_meta.get("info_count", 0)
        if overall_clean is True and isinstance(error_count, int) and error_count == 0:
            readiness_verdict = "LINT_CLEAN"
            rationale = (
                f"LINT_CLEAN: 0 ERROR / {warn_count} WARN / {info_count} INFO "
                "per tac.submission_packet.lint_submission_bundle "
                "(canonical Phase 5 Layer 3 lint protocol)."
            )
        elif overall_clean is False:
            readiness_verdict = "BLOCKED_ON_LINT_ERRORS"
            rationale = (
                f"BLOCKED_ON_LINT_ERRORS: {error_count} ERROR / "
                f"{warn_count} WARN / {info_count} INFO. "
                "Operator-routable: fix forbidden-token / first-person-plural / "
                "emdash / inflate.py-LOC / archive-sha violations before "
                "PR-submission readiness."
            )
        else:
            readiness_verdict = "MALFORMED_LINT_METADATA"
            rationale = (
                "MALFORMED_LINT_METADATA: candidate carries submission_linter_verdict "
                "field but overall_clean is not boolean True/False. "
                "Operator-routable: re-emit via tac.submission_packet.lint_submission_bundle."
            )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "submission_linter_readiness_verdict": readiness_verdict,
    }
