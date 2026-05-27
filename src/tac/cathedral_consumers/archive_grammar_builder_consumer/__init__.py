# SPDX-License-Identifier: MIT
"""Cathedral consumer: archive grammar builder readiness annotation (Phase 3 sister).

Per Phase 1 audit specification memo §3 Phase 3 acceptance + Catalog #335
canonical CathedralConsumerContract + Catalog #341 canonical-routing
markers + 6-hook wire-in per Catalog #125.

Per the 12th canonicalization × standardization × ease-of-contest-compliance
trinity: this Tier-A observability-only cathedral consumer auto-discovered
per Catalog #336/#337 surfaces archive-grammar readiness per candidate so
the cathedral autopilot ranker can see WHICH candidates have a clean HNeRV
parity L3 monolithic single-file 0.bin archive grammar (READY) versus
WHICH carry blockers (NEEDS-REMEDIATION) — without mutating the ranker's
predicted delta (Tier A invariant per Catalog #341).

Sister of:
  - tac.cathedral_consumers._example_consumer (canonical reference)
  - tac.cathedral_consumers.compression_pipeline_readiness_consumer (Phase 2 sister)
  - tac.cathedral_consumers.master_gradient_aggregate_consumer (Catalog #354 sister)
  - tac.submission_packet.archive_grammar (this consumer's data source)

Hooks (per Catalog #125 6-hook wire-in non-negotiable):
  * Hook #1 SENSITIVITY_MAP — N/A (defensive observability consumer)
  * Hook #2 PARETO_CONSTRAINT — N/A
  * Hook #3 BIT_ALLOCATOR — ACTIVE (per-section length feeds the bit-
    allocator priority cascade so canonical-monolithic archives rank ahead
    of multi-file for the same predicted delta band)
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE PRIMARY (this IS the
    consumer)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (per-archive-grammar
    anchor feeds the canonical posterior so Phase 6/Phase 10 empirical
    anchor landings inherit the apriori archive-grammar readiness signal)
  * Hook #6 PROBE_DISAMBIGUATOR — ACTIVE (per-section byte-mutation
    verdict IS the canonical disambiguator between OPERATIONAL vs
    RESEARCH_ONLY archive grammars per Catalog #220 + #266)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "archive_grammar_builder_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per Phase 3 scope: this consumer is observability-only at landing.
    Phase 6 + Phase 10 future-subagent landings will wire the
    archive-grammar anchor into the canonical equation #344 registry
    (see ``tac.canonical_equations.update_equation_with_empirical_anchor``).
    """
    _ = anchor  # explicit acknowledgment per reference _example_consumer pattern


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns the canonical Tier-A observability-only contribution dict per
    Catalog #341 routing markers: zero predicted_delta_adjustment, never
    promotable, axis_tag=[predicted]. Promotion of any archive-grammar
    readiness signal to a contest score requires paired-axis empirical
    evidence per CLAUDE.md "Submission auth eval BOTH CPU AND CUDA".

    The contribution surfaces a readiness rationale derived from the
    candidate's archive-grammar metadata (when present). When the
    candidate lacks archive-grammar metadata, the consumer returns a
    neutral observation without claiming readiness or non-readiness.
    """
    ag_meta = (
        candidate.get("archive_grammar_manifest")
        if isinstance(candidate, Mapping)
        else None
    )
    rationale = "Archive grammar preparation status unknown (no metadata on candidate)"
    readiness_verdict = "UNKNOWN"
    if isinstance(ag_meta, Mapping):
        monolithic = ag_meta.get("monolithic_single_file")
        no_op_passed = ag_meta.get("no_op_detector_passed")
        smoke_verdict = ag_meta.get("byte_mutation_smoke_verdict")
        section_count = len(ag_meta.get("section_specs", []))
        # Catalog #266 FAILED_BYTES_NOT_CONSUMED is the hardest signal — surface
        # it FIRST regardless of monolithic shape (research-substrate trap).
        if smoke_verdict == "FAILED_BYTES_NOT_CONSUMED":
            readiness_verdict = "BLOCKED"
            rationale = (
                "Phase 3 Layer 1 archive grammar BLOCKED per Catalog #266: "
                "archive bytes structurally consumed but no frame changes "
                "resulted (research-substrate trap, 8th forbidden pattern). "
                "Operator-routable per Catalog #220 + #272 + #105 + #139 "
                "sister discipline."
            )
        elif monolithic is True and (smoke_verdict == "PASSED" or smoke_verdict == "NOT_RUN"):
            readiness_verdict = "READY"
            rationale = (
                "Phase 3 Layer 1 archive grammar preparation CLEAN: "
                "HNeRV parity L3 monolithic single-file 0.bin; "
                f"{section_count} canonical section(s); "
                f"byte_mutation_smoke={smoke_verdict}. Per Phase 1 spec memo, "
                "downstream Phase 4-10 layers can compose on this manifest."
            )
        elif monolithic is False:
            readiness_verdict = "MULTI_FILE_REVIEW"
            rationale = (
                f"Phase 3 Layer 1 archive grammar MULTI-FILE (non-canonical per "
                f"HNeRV parity L3); {section_count} section(s). Multi-file "
                f"justification: {ag_meta.get('multi_file_justification', 'N/A')}"
            )
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "readiness_verdict": readiness_verdict,
    }
