# SPDX-License-Identifier: MIT
"""Cathedral consumer: submission compliance verdict readiness annotation (Phase 6 sister).

Per Phase 1 audit specification memo §3 Phase 6 (compliance Layer 4)
acceptance + Catalog #335 canonical CathedralConsumerContract + Catalog
#341 canonical-routing markers + 6-hook wire-in per Catalog #125.

Per the 12th canonicalization × standardization × ease-of-contest-compliance
trinity: this Tier-A observability-only cathedral consumer auto-discovered
per Catalog #336/#337 surfaces submission-compliance readiness per
candidate so the cathedral autopilot ranker can see WHICH candidates are
PR-ready (CLEAN) versus WHICH carry operator-gated D3/D5 dependencies
(OPERATOR_GATED) or structural blockers (STRUCTURAL_BLOCKED) — without
mutating the ranker's predicted delta (Tier A invariant per Catalog
#341).

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
COMPLIANT HARDWARE" non-negotiable + Catalog #192 macOS-CPU non-promotion:
the consumer SPECIFICALLY surfaces a FORBIDDEN_MACOS_AXIS verdict when the
underlying compliance verdict detected Darwin ARM64 / macOS CPU substrate
references. This is the canonical disambiguator at the cathedral ranker
surface for Catalog #192.

Sister of:
  - tac.cathedral_consumers._example_consumer (canonical reference)
  - tac.cathedral_consumers.compression_pipeline_readiness_consumer (Phase 2 sister)
  - tac.cathedral_consumers.archive_grammar_builder_consumer (Phase 3 sister)
  - tac.cathedral_consumers.submission_bundle_builder_consumer (Phase 4 sister)
  - tac.submission_packet.compliance (this consumer's data source)

Hooks (per Catalog #125 6-hook wire-in non-negotiable):
  * Hook #1 SENSITIVITY_MAP — N/A (defensive observability consumer)
  * Hook #2 PARETO_CONSTRAINT — N/A
  * Hook #3 BIT_ALLOCATOR — N/A
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE PRIMARY (this IS the
    consumer; surfaces per-candidate PR-readiness for dispatch ranking)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (per-candidate
    compliance verdict feeds the canonical posterior so Phase 6/Phase 10
    empirical anchor landings inherit the apriori compliance signal)
  * Hook #6 PROBE_DISAMBIGUATOR — ACTIVE (CLEAN vs OPERATOR_GATED vs
    STRUCTURAL_BLOCKED vs FORBIDDEN_MACOS_AXIS IS the canonical
    disambiguator at the cathedral ranker surface)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "submission_compliance_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per Phase 6 scope: this consumer is observability-only at landing.
    Phase 10 future-subagent landings will wire the compliance verdict
    anchor into the canonical equation #344 registry
    (see ``tac.canonical_equations.update_equation_with_empirical_anchor``).
    """
    _ = anchor  # explicit acknowledgment per reference _example_consumer pattern


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns the canonical Tier-A observability-only contribution dict per
    Catalog #341 routing markers: zero predicted_delta_adjustment, never
    promotable, axis_tag=[predicted]. Promotion of any compliance verdict
    to a contest score requires paired-axis empirical evidence per
    CLAUDE.md "Submission auth eval BOTH CPU AND CUDA".

    The contribution surfaces a readiness verdict derived from the
    candidate's compliance verdict metadata (when present). When the
    candidate lacks compliance verdict metadata, the consumer returns a
    neutral observation without claiming readiness or non-readiness.
    """
    compliance_meta = (
        candidate.get("compliance_verdict")
        if isinstance(candidate, Mapping)
        else None
    )
    rationale = (
        "Submission compliance verdict unknown (no metadata on candidate)"
    )
    readiness_verdict = "UNKNOWN"
    if isinstance(compliance_meta, Mapping):
        overall_clean = bool(compliance_meta.get("overall_clean", False))
        forbidden_macos = bool(
            compliance_meta.get("forbidden_macos_axis_detected", False)
        )
        op_gated = compliance_meta.get("operator_gated_remaining", [])
        op_gated_count = len(op_gated) if isinstance(op_gated, list) else 0
        error_count = int(compliance_meta.get("error_count", 0))
        total_checks = int(compliance_meta.get("total_checks", 0))
        passed_count = int(compliance_meta.get("passed_count", 0))
        if forbidden_macos:
            readiness_verdict = "FORBIDDEN_MACOS_AXIS"
            rationale = (
                f"Phase 6 compliance verdict FORBIDDEN per Catalog #192 + "
                f"CLAUDE.md 'Submission auth eval BOTH CPU AND CUDA' "
                f"non-negotiable: candidate references macOS / Darwin ARM64 "
                f"substrate as authoritative axis. Re-run on Linux x86_64 "
                f"1:1 contest-compliant hardware before submission."
            )
        elif overall_clean:
            readiness_verdict = "CLEAN"
            rationale = (
                f"Phase 6 Layer 4 compliance verdict CLEAN: "
                f"{passed_count}/{total_checks} checks passed; zero "
                f"structural blockers + zero operator-gated remaining. "
                f"Submission packet is PR-ready per CLAUDE.md non-negotiable."
            )
        elif op_gated_count > 0 and error_count == op_gated_count:
            # All errors are operator-gated (D3 hosting / D5 paired axis)
            readiness_verdict = "OPERATOR_GATED"
            rationale = (
                f"Phase 6 compliance verdict OPERATOR_GATED: "
                f"{passed_count}/{total_checks} structural checks passed; "
                f"{op_gated_count} operator-gated remaining (D3 hosting / "
                f"D5 paired auth-eval). Operator-routable: run paired axis "
                f"dispatch per Phase 7 paired_auth_eval canonical helper."
            )
        else:
            readiness_verdict = "STRUCTURAL_BLOCKED"
            structural_blockers = error_count - op_gated_count
            rationale = (
                f"Phase 6 compliance verdict STRUCTURAL_BLOCKED: "
                f"{structural_blockers} structural blockers + "
                f"{op_gated_count} operator-gated remaining. "
                f"Operator-routable: rebuild submission_dir via canonical "
                f"tac.submission_packet.build_submission_bundle and re-run "
                f"compliance enforcement."
            )
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "readiness_verdict": readiness_verdict,
    }
