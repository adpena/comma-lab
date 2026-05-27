# SPDX-License-Identifier: MIT
"""Cathedral consumer: paired auth-eval readiness annotation (Phase 7 sister).

Per Phase 1 audit specification memo §3 Phase 6 / Layer 5 (paired_auth_eval)
acceptance + Catalog #335 canonical CathedralConsumerContract + Catalog
#341 canonical-routing markers + 6-hook wire-in per Catalog #125.

Per the 12th canonicalization × standardization × ease-of-contest-compliance
trinity: this Tier-A observability-only cathedral consumer auto-discovered
per Catalog #336/#337 surfaces paired-axis dispatch readiness per
candidate so the cathedral autopilot ranker can see WHICH candidates have
landed paired-CUDA + paired-CPU empirical anchors (PAIRED_PASS) versus
WHICH carry partial-axis-missing dependencies (PARTIAL) or structural
blockers (BLOCKED) — without mutating the ranker's predicted delta (Tier
A invariant per Catalog #341).

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
COMPLIANT HARDWARE" non-negotiable + Catalog #192 macOS-CPU non-promotion:
the consumer SPECIFICALLY surfaces a FORBIDDEN_MACOS_AXIS verdict when the
underlying paired-axis verdict detected Darwin ARM64 / macOS CPU substrate
references. This is the canonical disambiguator at the cathedral ranker
surface for Catalog #192 sister to Phase 6 compliance consumer.

Sister of:
  - tac.cathedral_consumers._example_consumer (canonical reference)
  - tac.cathedral_consumers.compression_pipeline_readiness_consumer (Phase 2 sister)
  - tac.cathedral_consumers.archive_grammar_builder_consumer (Phase 3 sister)
  - tac.cathedral_consumers.submission_bundle_builder_consumer (Phase 4 sister)
  - tac.cathedral_consumers.submission_compliance_consumer (Phase 6 sister)
  - tac.submission_packet.paired_auth_eval (this consumer's data source)

Hooks (per Catalog #125 6-hook wire-in non-negotiable):
  * Hook #1 SENSITIVITY_MAP -- N/A (defensive observability consumer)
  * Hook #2 PARETO_CONSTRAINT -- N/A
  * Hook #3 BIT_ALLOCATOR -- N/A
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH -- ACTIVE PRIMARY (this IS the
    consumer; surfaces per-candidate paired-axis readiness for dispatch
    ranking)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR -- ACTIVE (per-candidate
    paired-axis verdict feeds the canonical posterior so Phase 10 first
    paired empirical anchor landings inherit the apriori paired-axis
    signal AND the canonical equation #344 entry
    paired_auth_eval_canonical_helper_consolidation_savings_v1 promotes
    from FORMALIZATION_PENDING to REGISTERED)
  * Hook #6 PROBE_DISAMBIGUATOR -- ACTIVE (PAIRED_PASS vs PARTIAL_CUDA_ONLY
    vs PARTIAL_CPU_ONLY vs BLOCKED_PRE_DISPATCH vs BLOCKED_HARVEST vs
    BLOCKED_AXIS_MISMATCH vs BLOCKED_HARDWARE_NON_COMPLIANT IS the
    canonical disambiguator at the cathedral ranker surface)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "paired_auth_eval_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 -- continual-learning posterior update.

    Per Phase 7 scope: this consumer is observability-only at landing.
    Phase 10 future-subagent landings will wire the paired-axis verdict
    anchor into the canonical equation #344 registry
    (see ``tac.canonical_equations.update_equation_with_empirical_anchor``).
    """
    _ = anchor  # explicit acknowledgment per reference _example_consumer pattern


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 -- cathedral autopilot ranker contribution.

    Returns the canonical Tier-A observability-only contribution dict per
    Catalog #341 routing markers: zero predicted_delta_adjustment, never
    promotable, axis_tag=[predicted]. Promotion of any paired-axis verdict
    to a contest score requires the structural Tier-B canonical contract
    per Catalog #357 (future Phase 10 sister landing).

    The contribution surfaces a readiness verdict derived from the
    candidate's paired_auth_eval_verdict metadata (when present). When the
    candidate lacks paired-auth-eval verdict metadata, the consumer
    returns a neutral observation without claiming readiness or
    non-readiness.
    """
    paired_meta = (
        candidate.get("paired_auth_eval_verdict")
        if isinstance(candidate, Mapping)
        else None
    )
    rationale = (
        "Paired auth-eval verdict unknown (no metadata on candidate)"
    )
    readiness_verdict = "UNKNOWN"
    if isinstance(paired_meta, Mapping):
        verdict = str(paired_meta.get("verdict", "UNKNOWN"))
        forbidden_macos = bool(
            paired_meta.get("forbidden_macos_axis_detected", False)
        )
        cuda_score = paired_meta.get("cuda_score")
        cpu_score = paired_meta.get("cpu_score")
        cuda_cpu_gap = paired_meta.get("cuda_cpu_gap")
        promotable = bool(paired_meta.get("promotable", False))
        archive_sha_paired = str(paired_meta.get("archive_sha256_paired", ""))[:12]
        if forbidden_macos:
            readiness_verdict = "FORBIDDEN_MACOS_AXIS"
            rationale = (
                f"Phase 7 paired auth-eval verdict FORBIDDEN per Catalog #192 + "
                f"CLAUDE.md 'Submission auth eval BOTH CPU AND CUDA, ON 1:1 "
                f"CONTEST-COMPLIANT HARDWARE' non-negotiable: candidate axes "
                f"reference macOS / Darwin ARM64 substrate. Re-dispatch on "
                f"Linux x86_64 1:1 contest-compliant hardware before submission."
            )
        elif verdict == "PAIRED_PASS" and promotable:
            cuda_str = (
                f"CUDA={float(cuda_score):.6f}" if cuda_score is not None else "CUDA=missing"
            )
            cpu_str = (
                f"CPU={float(cpu_score):.6f}" if cpu_score is not None else "CPU=missing"
            )
            gap_str = (
                f"gap={float(cuda_cpu_gap):+.6f}"
                if cuda_cpu_gap is not None
                else "gap=N/A"
            )
            readiness_verdict = "PAIRED_PASS"
            rationale = (
                f"Phase 7 Layer 5 paired auth-eval verdict PAIRED_PASS: "
                f"archive_sha256={archive_sha_paired}; {cuda_str}; {cpu_str}; "
                f"{gap_str}. Sha-locked invariant held; both axes on 1:1 "
                f"contest-compliant Linux x86_64 hardware. Submission packet "
                f"is PR-ready per CLAUDE.md non-negotiable."
            )
        elif verdict == "PAIRED_PARTIAL_CUDA_ONLY":
            readiness_verdict = "PARTIAL_CUDA_ONLY"
            cuda_str = (
                f"CUDA={float(cuda_score):.6f}" if cuda_score is not None else "CUDA=??"
            )
            rationale = (
                f"Phase 7 paired auth-eval verdict PARTIAL_CUDA_ONLY: "
                f"archive_sha256={archive_sha_paired}; {cuda_str}; CPU axis "
                f"missing/failed. Operator-routable: re-dispatch CPU axis "
                f"per CLAUDE.md 'Submission auth eval BOTH CPU AND CUDA' "
                f"non-negotiable. NOT promotable until paired anchor."
            )
        elif verdict == "PAIRED_PARTIAL_CPU_ONLY":
            readiness_verdict = "PARTIAL_CPU_ONLY"
            cpu_str = (
                f"CPU={float(cpu_score):.6f}" if cpu_score is not None else "CPU=??"
            )
            rationale = (
                f"Phase 7 paired auth-eval verdict PARTIAL_CPU_ONLY: "
                f"archive_sha256={archive_sha_paired}; {cpu_str}; CUDA axis "
                f"missing/failed. Operator-routable: re-dispatch CUDA axis. "
                f"NOT promotable until paired anchor."
            )
        elif verdict == "BLOCKED_PRE_DISPATCH":
            readiness_verdict = "BLOCKED_PRE_DISPATCH"
            rationale = (
                f"Phase 7 paired auth-eval verdict BLOCKED_PRE_DISPATCH: "
                f"pre-dispatch validation failed (likely sha-mismatch / "
                f"missing-archive / insufficient-budget / non-canonical-"
                f"cpu-target). Operator-routable: consult verdict_rationale "
                f"field for specific cause."
            )
        elif verdict == "BLOCKED_HARVEST":
            readiness_verdict = "BLOCKED_HARVEST"
            rationale = (
                f"Phase 7 paired auth-eval verdict BLOCKED_HARVEST: "
                f"dispatch fired but harvest failed. Operator-routable: "
                f"consult Modal dashboard via call_id (per Catalog #245 "
                f"canonical ledger). Retry available per CLAUDE.md "
                f"'Modal .spawn() HARVEST OR LOSE' non-negotiable."
            )
        elif verdict == "BLOCKED_AXIS_MISMATCH":
            readiness_verdict = "BLOCKED_AXIS_MISMATCH"
            rationale = (
                f"Phase 7 paired auth-eval verdict BLOCKED_AXIS_MISMATCH: "
                f"sha-locked invariant violated (CUDA and CPU axes ran on "
                f"different archive bytes) per Catalog #127 custody "
                f"discipline. Operator-routable: investigate dispatch path."
            )
        elif verdict == "BLOCKED_HARDWARE_NON_COMPLIANT":
            readiness_verdict = "BLOCKED_HARDWARE_NON_COMPLIANT"
            rationale = (
                f"Phase 7 paired auth-eval verdict BLOCKED_HARDWARE_NON_COMPLIANT: "
                f"at least one axis landed on non-1:1-contest-compliant "
                f"hardware per Catalog #192. Re-dispatch on canonical Linux "
                f"x86_64 substrate."
            )
        else:
            readiness_verdict = verdict
            rationale = (
                f"Phase 7 paired auth-eval verdict {verdict}: see verdict_rationale "
                f"for operator-routable next-action."
            )
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "readiness_verdict": readiness_verdict,
    }
