# SPDX-License-Identifier: MIT
"""Cathedral consumer: submission bundle builder readiness annotation (Phase 4 sister).

Per Phase 1 audit specification memo §3 Phase 4 acceptance + Catalog #335
canonical CathedralConsumerContract + Catalog #341 canonical-routing
markers + 6-hook wire-in per Catalog #125.

Per the 12th canonicalization × standardization × ease-of-contest-compliance
trinity: this Tier-A observability-only cathedral consumer auto-discovered
per Catalog #336/#337 surfaces submission-bundle readiness per candidate so
the cathedral autopilot ranker can see WHICH candidates have a clean HNeRV
parity L4 bundle (READY) versus WHICH carry blockers (NEEDS-REMEDIATION) —
without mutating the ranker's predicted delta (Tier A invariant per Catalog
#341).

Sister of:
  - tac.cathedral_consumers._example_consumer (canonical reference)
  - tac.cathedral_consumers.compression_pipeline_readiness_consumer (Phase 2 sister)
  - tac.cathedral_consumers.archive_grammar_builder_consumer (Phase 3 sister)
  - tac.submission_packet.builder (this consumer's data source)

Hooks (per Catalog #125 6-hook wire-in non-negotiable):
  * Hook #1 SENSITIVITY_MAP — N/A (defensive observability consumer)
  * Hook #2 PARETO_CONSTRAINT — N/A
  * Hook #3 BIT_ALLOCATOR — ACTIVE (per-bundle inflate.py LOC + deps closure
    feeds the bit-allocator priority cascade so canonical numpy-portable
    HNeRV parity L4 bundles rank ahead of multi-dep heavyweight bundles for
    the same predicted delta band)
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE PRIMARY (this IS the
    consumer)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (per-bundle readiness
    anchor feeds the canonical posterior so Phase 6/Phase 10 empirical
    anchor landings inherit the apriori bundle-readiness signal)
  * Hook #6 PROBE_DISAMBIGUATOR — ACTIVE (per-bundle PYTHONPATH self-
    containment status IS the canonical disambiguator between CLEAN vs
    VENDORED_WITH_EXPLICIT_WAIVER bundles per Catalog #295)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "submission_bundle_builder_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per Phase 4 scope: this consumer is observability-only at landing.
    Phase 6 + Phase 10 future-subagent landings will wire the submission-
    bundle anchor into the canonical equation #344 registry
    (see ``tac.canonical_equations.update_equation_with_empirical_anchor``).
    """
    _ = anchor  # explicit acknowledgment per reference _example_consumer pattern


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns the canonical Tier-A observability-only contribution dict per
    Catalog #341 routing markers: zero predicted_delta_adjustment, never
    promotable, axis_tag=[predicted]. Promotion of any submission-bundle
    readiness signal to a contest score requires paired-axis empirical
    evidence per CLAUDE.md "Submission auth eval BOTH CPU AND CUDA".

    The contribution surfaces a readiness rationale derived from the
    candidate's submission-bundle metadata (when present). When the
    candidate lacks submission-bundle metadata, the consumer returns a
    neutral observation without claiming readiness or non-readiness.
    """
    bundle_meta = (
        candidate.get("submission_bundle_result")
        if isinstance(candidate, Mapping)
        else None
    )
    rationale = "Submission bundle preparation status unknown (no metadata on candidate)"
    readiness_verdict = "UNKNOWN"
    if isinstance(bundle_meta, Mapping):
        inflate_loc = bundle_meta.get("inflate_py_loc")
        inflate_budget = bundle_meta.get("inflate_py_loc_budget")
        dep_man = bundle_meta.get("dependency_closure_manifest")
        pythonpath_status = bundle_meta.get("pythonpath_self_containment_status")
        device_routing = bundle_meta.get("select_inflate_device_routing")
        within_loc_budget = (
            inflate_loc is not None
            and inflate_budget is not None
            and inflate_loc <= inflate_budget
        )
        within_deps_budget = (
            isinstance(dep_man, Mapping) and bool(dep_man.get("within_budget"))
        )
        numpy_portable = (
            isinstance(dep_man, Mapping) and bool(dep_man.get("numpy_portable"))
        )
        pythonpath_clean = pythonpath_status == "clean"
        if within_loc_budget and within_deps_budget and pythonpath_clean:
            readiness_verdict = "READY"
            np_marker = "numpy-portable" if numpy_portable else "canonical"
            rationale = (
                f"Phase 4 Layer 2 submission bundle preparation CLEAN: "
                f"inflate.py LOC={inflate_loc}/{inflate_budget} ({np_marker}); "
                f"deps within budget; PYTHONPATH self-containment={pythonpath_status}; "
                f"device routing={device_routing}. Per Phase 1 spec memo, "
                f"downstream Phase 5-10 layers can compose on this bundle."
            )
        elif not within_loc_budget:
            readiness_verdict = "BLOCKED"
            rationale = (
                f"Phase 4 Layer 2 submission bundle BLOCKED per HNeRV parity L4: "
                f"inflate.py LOC={inflate_loc} exceeds budget {inflate_budget}. "
                f"Operator-routable: rewrite inflate.py within budget OR add "
                f"substantive inflate_py_loc_waiver_rationale per Catalog #287."
            )
        elif not within_deps_budget:
            readiness_verdict = "BLOCKED"
            rationale = (
                f"Phase 4 Layer 2 submission bundle BLOCKED per HNeRV parity L4: "
                f"declared dependencies exceed budget. Operator-routable: reduce "
                f"deps OR add substantive inflate_deps_waiver_rationale per Catalog #287."
            )
        elif not pythonpath_clean:
            readiness_verdict = "REVIEW_REQUIRED"
            rationale = (
                f"Phase 4 Layer 2 submission bundle PYTHONPATH self-containment "
                f"status={pythonpath_status} (non-CLEAN per Catalog #295). "
                f"Operator-routable: review vendored sister-package layout."
            )
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "readiness_verdict": readiness_verdict,
    }
