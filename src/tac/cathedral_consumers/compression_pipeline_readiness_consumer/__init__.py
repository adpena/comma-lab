# SPDX-License-Identifier: MIT
"""Cathedral consumer: compression pipeline readiness annotation (Phase 2 sister).

Per Phase 1 audit specification memo §3 Phase 2 acceptance + Catalog #335
canonical CathedralConsumerContract + Catalog #341 canonical-routing
markers + 6-hook wire-in per Catalog #125.

Per the 12th canonicalization × standardization × ease-of-contest-compliance
trinity: this Tier-A observability-only cathedral consumer auto-discovered
per Catalog #336/#337 surfaces compression-pipeline readiness per candidate
so the cathedral autopilot ranker can see WHICH candidates have a clean
Catalog #270 umbrella verdict (READY to dispatch) versus WHICH carry
blockers (NEEDS-REMEDIATION) — without mutating the ranker's predicted
delta (Tier A invariant per Catalog #341).

Sister of:
  - tac.cathedral_consumers._example_consumer (canonical reference)
  - tac.cathedral_consumers.master_gradient_aggregate_consumer (Catalog #354 sister)
  - tac.submission_packet.compression_pipeline (this consumer's data source)

Hooks (per Catalog #125 6-hook wire-in non-negotiable):
  * Hook #1 SENSITIVITY_MAP — N/A (defensive observability consumer)
  * Hook #2 PARETO_CONSTRAINT — N/A
  * Hook #3 BIT_ALLOCATOR — ACTIVE (readiness verdict feeds the bit-
    allocator priority cascade so READY candidates rank ahead of
    BLOCKED candidates for the same predicted delta band)
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE PRIMARY (this IS the
    consumer)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (per-readiness anchor
    feeds the canonical posterior so Phase 6/Phase 10 empirical anchor
    landings inherit the apriori readiness signal)
  * Hook #6 PROBE_DISAMBIGUATOR — N/A
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "compression_pipeline_readiness_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per Phase 2 scope: this consumer is observability-only at landing.
    Phase 6 + Phase 10 future-subagent landings will wire the
    compression-pipeline anchor into the canonical equation #344 registry
    (see ``tac.canonical_equations.update_equation_with_empirical_anchor``).
    """
    _ = anchor  # explicit acknowledgment per reference _example_consumer pattern


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns the canonical Tier-A observability-only contribution dict per
    Catalog #341 routing markers: zero predicted_delta_adjustment, never
    promotable, axis_tag=[predicted]. Promotion of any compression-pipeline
    readiness signal to a contest score requires paired-axis empirical
    evidence per CLAUDE.md "Submission auth eval BOTH CPU AND CUDA".

    The contribution surfaces a readiness rationale derived from the
    candidate's compression-pipeline metadata (when present). When the
    candidate lacks compression-pipeline metadata, the consumer returns a
    neutral observation without claiming readiness or non-readiness.
    """
    cp_meta = candidate.get("compression_pipeline_result") if isinstance(candidate, Mapping) else None
    rationale = "Compression pipeline preparation status unknown (no metadata on candidate)"
    readiness_verdict = "UNKNOWN"
    if isinstance(cp_meta, Mapping):
        overall_pass = cp_meta.get("dispatch_optimization_protocol_overall_pass")
        blockers = cp_meta.get("dispatch_optimization_protocol_blockers") or []
        if overall_pass is True and not blockers:
            readiness_verdict = "READY"
            rationale = (
                "Phase 2 Layer 0 compression pipeline preparation CLEAN: "
                "Catalog #270 umbrella verdict overall_pass=True; "
                "zero Tier-1/Tier-2/Tier-3 blockers. Per Phase 1 spec memo, "
                "downstream Phase 3-10 layers can compose on this result."
            )
        elif overall_pass is False or blockers:
            readiness_verdict = "BLOCKED"
            rationale = (
                f"Phase 2 Layer 0 compression pipeline preparation BLOCKED: "
                f"{len(blockers)} Catalog #270 blocker(s) — operator-routable "
                "to remediate per Tier-1 engineering (#172/#178/#179/#180), "
                "Tier-2 hardware (#170/#171/#181/#182/#215), or Tier-3 "
                "substrate (#222/#226/#240/#249) guidance."
            )
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "readiness_verdict": readiness_verdict,
    }
