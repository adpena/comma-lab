# SPDX-License-Identifier: MIT
"""Layer 4 — contest compliance enforcer (canonical helper).

Wrap ``scripts/pre_submission_compliance_check.py`` (currently 3267 LOC
standalone canonical implementation) in a typed canonical helper per Phase 1
audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
Layer 4 (compliance).

The bug class this layer extincts: per-substrate compliance invocations
diverge on which Catalog gates they actually consult; the operator-facing
CLI surface is a 3267-LOC monolithic script with no typed return shape, no
per-Catalog-gate categorization, no machine-readable downstream consumer
contract, and silently accepts macOS-CPU artifacts as authoritative when
``--submission-score-axis=contest_cpu`` is provided without 1:1 Linux
x86_64 verification per CLAUDE.md "Submission auth eval — BOTH CPU AND
CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable.

Per the 12th canonicalization × standardization × ease-of-contest-
compliance trinity: ONE canonical helper, ONE return shape, ONE per-check
classification protocol. Downstream consumers (Phase 5 linter sister
parallel; Phase 6 paired_auth_eval; Phase 7 operator runbook CLI; Phase 10
PR111-candidate end-to-end regression; Phase 8 Catalog #362 STRICT gate)
consume :class:`ComplianceVerdict` directly without re-parsing the 3267-LOC
script's JSON.

Per the 13th OPTIMAL-TRIO standing directive: helper is canonical-frozen-
dataclass-return + canonical-Provenance-routing + 4-layer canonical-helper-
pattern sister of :mod:`tac.deploy.modal.call_id_ledger` (Catalog #245),
:mod:`tac.probe_outcomes_ledger` (Catalog #313),
:mod:`tac.canonical_equations` (Catalog #344), and
:mod:`tac.submission_packet.builder` (Phase 4 Layer 2).

Per CLAUDE.md "Apples-to-apples evidence discipline" + "Submission auth
eval — BOTH CPU AND CUDA" non-negotiables: this layer is OBSERVABILITY-
ONLY by construction. Every emitted :class:`ComplianceVerdict` carries
``score_claim=False`` + ``promotable=False`` + ``axis_tag=[predicted]``.
The compliance verdict gates SUBMISSION ELIGIBILITY (PR-ready vs
operator-routable-blockers); it does NOT promote any score to a contest
axis. Score promotion REQUIRES Phase 6 paired-CUDA + Linux x86_64 CPU
empirical anchor per CLAUDE.md non-negotiable.

Per CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback
trap)" + Catalog #192 (`check_macos_cpu_advisory_not_promoted_without_
linux_verification`): the helper EXPLICITLY refuses Darwin ARM64 /
``macos_arm64`` hardware substrates as authoritative axes. macOS-CPU
remains a valid macOS-research-signal advisory route per Catalog #192
but is structurally non-promotable.

Quick start::

    from tac.submission_packet import (
        SubmissionBundleResult,
        enforce_contest_compliance,
    )

    bundle: SubmissionBundleResult = ...  # Phase 4 output
    verdict = enforce_contest_compliance(
        submission_bundle_result=bundle,
        contest_final_strict=True,
        expected_lane_id="lane_pr111_candidate_20260601",
        expected_job_id="fc-01KS...",
        output_dir=Path("reports/pr_pre_submission/"),
    )
    if verdict.overall_clean:
        print("PR-ready")
    else:
        for blocker in verdict.error_checks:
            print(f"BLOCKER: {blocker.check_name} -- {blocker.remediation_hint}")
        for og in verdict.operator_gated_remaining:
            print(f"OPERATOR-GATED: {og.check_name} -- {og.remediation_hint}")

Discipline cross-references:
  * CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable
  * Catalog #127 authoritative-tag custody validation
  * Catalog #146 contest-compliant inflate runtime template
  * Catalog #152 required-input-file pre-dispatch validation
  * Catalog #192 macOS-CPU non-promotion enforcement
  * Catalog #221 auth-eval result artifact fail-closed
  * Catalog #226 canonical gate_auth_eval_call helper
  * Catalog #240 recipe-vs-trainer-state consistency
  * Catalog #266 archive bytes consumed by inflate proof
  * Catalog #270 dispatch optimization protocol umbrella
  * Catalog #287 placeholder-rationale rejection
  * Catalog #323 canonical Provenance umbrella
  * Catalog #335 cathedral consumer canonical contract
  * Catalog #341 Tier A canonical-routing markers
  * Catalog #344 canonical equations registry
  * Catalog #245 / #313 / #344 / #355 canonical 4-layer pattern
  * 10th apples-to-apples paired CPU+CUDA on 1:1 contest-compliant hardware
  * 11th ORDER-MATTERS sequencing
  * 12th canonicalization × standardization × ease-of-contest-compliance
  * 13th OPTIMAL-TRIO standing directive
"""
from __future__ import annotations

import datetime
import enum
import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac.submission_packet.builder import SubmissionBundleResult


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Module-level constants — canonical schemas + canonical budgets
# ---------------------------------------------------------------------------

COMPLIANCE_SCHEMA_VERSION = "submission_compliance_v1_20260526"
"""Pinned schema for :class:`ComplianceVerdict` persistence rows."""

PHASE_6_LAYER_VERSION = "phase_6_submission_compliance_canonical_landed_20260526"
"""Operator-readable Phase 6 landing marker per Phase 1 audit spec memo."""

CANONICAL_EQUATION_ID = (
    "submission_compliance_canonical_helper_consolidation_savings_v1"
)
"""Canonical equation registered per Phase 1 audit spec memo §13.

FORMALIZATION_PENDING until Phase 10 first-PR-through-canonical-pipeline
regression lands the first paired-CUDA empirical anchor of per-substrate
compliance-invocation-divergence collapse (predicted: 14+ per-substrate
ad-hoc compliance invocations consolidated to ONE canonical helper, with
typed Catalog-gate-categorized blockers).
"""

CANONICAL_COMPLIANCE_SCRIPT_PATH = (
    "scripts/pre_submission_compliance_check.py"
)
"""Canonical compliance script (3267 LOC standalone)."""

# Per Catalog #341 routing markers (Tier A observability-only).
PREDICTED_AXIS_TAG = "[predicted]"

# Per Catalog #287 placeholder rejection.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset(
    {"<rationale>", "<reason>", "<rationale_here>", "<reason_here>", ""}
)

# Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 — forbidden
# Darwin ARM64 / macOS substrate tokens as authoritative axes.
_FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS: frozenset[str] = frozenset({
    "macos_arm64",
    "darwin_arm64",
    "darwin_arm64_m5_max_macos_cpu_advisory",
    "darwin_arm64_m5_max_macos_cpu",
    "macos_cpu",
    "apple_silicon",
    "macos_m5_max",
    "macos_advisory",
})

# Canonical Catalog gate -> check-name-prefix mapping. The wrapped script
# emits check names that this helper categorizes by Catalog gate.
_CATALOG_GATE_PREFIXES: dict[int, tuple[str, ...]] = {
    127: (
        "authoritative_axis_",
        "auth_eval_custody_",
        "auth_eval_axis_",
    ),
    146: (
        "submission_runtime_",
        "inflate_sh_",
        "inflate_runtime_",
        "submission_runtime_manifest_",
    ),
    152: (
        "expected_archive_",
        "expected_lane_id_",
        "expected_job_id_",
        "expected_runtime_tree_",
    ),
    192: (
        "macos_cpu_advisory_",
        "non_linux_x86_64_",
    ),
    221: (
        "auth_eval_exists",
        "auth_eval_present",
        "auth_eval_score_",
        "auth_eval_n_samples_",
        "contest_cpu_auth_eval_",
    ),
    226: (
        "gate_auth_eval_",
        "auth_eval_paired_",
        "auth_eval_runtime_",
        "contest_final_selected_axis_",
    ),
    240: (
        "recipe_trainer_",
        "submission_runtime_match",
    ),
    266: (
        "archive_bytes_consumed_",
        "runtime_equivalence_proof_",
        "runtime_tree_",
    ),
}

# Canonical operator-gated D3 (hosting) + D5 (paired auth eval) blocker
# patterns per 2026-05-19 sister landing
# `feedback_pr_submission_d5_prerequisites_executed_landed_20260519T182635Z.md`.
_OPERATOR_GATED_BLOCKER_PATTERNS: tuple[str, ...] = (
    "auth_eval_",  # D5: requires paired CUDA + CPU dispatch
    "expected_runtime_tree_",  # D5: requires post-dispatch runtime sha
    "hosted_archive_",  # D3: requires public-URL hosting
    "public_source_",  # D3: requires public source ref manifest
    "runtime_equivalence_proof_",  # D5: requires post-dispatch parity
    "contest_cpu_auth_eval_",  # D5: requires paired CPU dispatch
)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SubmissionComplianceError(RuntimeError):
    """Compliance orchestration error.

    Sister of :class:`tac.submission_packet.builder.SubmissionBundleError`,
    :class:`tac.submission_packet.archive_grammar.ArchiveGrammarError`, and
    :class:`tac.submission_packet.compression_pipeline.CompressionPipelineError`.
    Raised by :func:`enforce_contest_compliance` when the wrapped
    ``scripts/pre_submission_compliance_check.py`` invocation crashes
    structurally (missing script / unparseable JSON / fail-closed-on-
    canonical-protection violation) — NOT when the script returns
    structured ``passed=False`` (that surfaces as a typed
    :class:`ComplianceVerdict` with ``overall_clean=False``).
    """


# ---------------------------------------------------------------------------
# Frozen dataclasses — canonical contract
# ---------------------------------------------------------------------------


class CheckSeverity(enum.StrEnum):
    """Canonical per-check severity taxonomy.

    Mirrors the wrapped script's Check.severity field. Each value maps to
    a downstream consumer's routing decision per Catalog #127/#192/#221
    discipline.
    """

    ERROR = "error"
    """Hard blocker: refuses contest-final --strict invocations."""

    WARNING = "warning"
    """Soft signal: surfaces in human-readable verdict; does not block."""

    INFO = "info"
    """Informational: per-Catalog-gate evidence; not a blocker."""


@dataclass(frozen=True)
class ComplianceCheck:
    """Canonical per-check result.

    Maps one ``scripts/pre_submission_compliance_check.py`` Check entry to
    a typed downstream-consumer-friendly contract that includes:

      * per-Catalog-gate refs (which Catalog gate(s) this check protects)
      * remediation hint (operator-actionable next step)
      * operator-gated classification (D3 hosting / D5 paired-axis-eval)
    """

    check_name: str
    """Canonical check name emitted by the wrapped script."""

    severity: str
    """One of :class:`CheckSeverity` values."""

    passed: bool
    """True iff the check passed."""

    details: str
    """Verbatim per-check details string from the wrapped script."""

    catalog_gate_refs: tuple[int, ...]
    """Canonical sorted tuple of Catalog gate refs this check operationalizes
    (e.g., ``(127, 192)`` for authoritative-axis-custody)."""

    is_operator_gated: bool
    """True iff this check requires operator-side D3 (hosting) or D5
    (paired-axis-eval) artifact to satisfy. Operator-gated blockers are
    surfaced separately from structural blockers in the verdict."""

    remediation_hint: str
    """Operator-actionable remediation hint (e.g., "run paired auth-eval
    via tools/dispatch_modal_paired_auth_eval.py")."""

    def __post_init__(self) -> None:
        if not self.check_name:
            raise ValueError("check_name must be non-empty")
        if self.severity not in {s.value for s in CheckSeverity}:
            raise ValueError(
                f"severity {self.severity!r} must be one of "
                f"{[s.value for s in CheckSeverity]}"
            )
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be bool")
        if not isinstance(self.catalog_gate_refs, tuple):
            raise ValueError("catalog_gate_refs must be a tuple (frozen)")
        for ref in self.catalog_gate_refs:
            if not isinstance(ref, int) or ref < 1 or ref > 1000:
                raise ValueError(
                    f"catalog_gate_refs entries must be positive ints in 1..1000; got {ref!r}"
                )
        sorted_refs = tuple(sorted(set(self.catalog_gate_refs)))
        if sorted_refs != self.catalog_gate_refs:
            raise ValueError(
                f"catalog_gate_refs must be sorted unique tuple; "
                f"got {self.catalog_gate_refs}; canonical {sorted_refs}"
            )
        if not isinstance(self.is_operator_gated, bool):
            raise ValueError("is_operator_gated must be bool")

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "severity": self.severity,
            "passed": bool(self.passed),
            "details": self.details,
            "catalog_gate_refs": list(self.catalog_gate_refs),
            "is_operator_gated": bool(self.is_operator_gated),
            "remediation_hint": self.remediation_hint,
        }


@dataclass(frozen=True)
class ComplianceVerdict:
    """Canonical Phase 6 Layer 4 compliance verdict.

    Sister of :class:`tac.submission_packet.builder.SubmissionBundleResult`
    + :class:`tac.submission_packet.archive_grammar.ArchiveGrammarManifest`
    + :class:`tac.submission_packet.compression_pipeline.CompressionPipelineResult`
    at the compliance enforcement sub-surface.

    Per CLAUDE.md "Apples-to-apples evidence discipline": this verdict is
    OBSERVABILITY-ONLY (score_claim=False + promotable=False + axis_tag=
    [predicted]). Per Catalog #341 routing markers, every consumer that
    reads this verdict inherits the non-promotable invariant.
    """

    schema_version: str
    """Canonical schema version (current: :data:`COMPLIANCE_SCHEMA_VERSION`)."""

    lane_id: str
    """Lane registry id from submission bundle lineage."""

    substrate_id: str
    """Substrate id from submission bundle lineage."""

    archive_sha256: str
    """sha256 hex digest of submission archive (from
    SubmissionBundleResult; passed to the wrapped script as
    ``--expected-archive-sha256``)."""

    archive_bytes: int
    """Total ``archive.zip`` size in bytes (passed as
    ``--expected-archive-size-bytes``)."""

    submission_dir: str
    """Path to bundled submission_dir/ (passed as ``--submission-dir``)."""

    overall_clean: bool
    """True iff every error-severity check passed AND no operator-gated
    blockers remain. False when any error-severity check failed OR any
    operator-gated D3/D5 dependency is unsatisfied."""

    contest_final_strict: bool
    """Caller-set flag indicating whether ``--contest-final --strict``
    was requested. When True, the wrapped script auto-sets
    ``--require-auth-eval`` + ``--require-t4-equivalent`` (CUDA path) +
    ``--require-submission-runtime-match`` per its main()."""

    submission_score_axis: str
    """One of ``contest_cuda`` / ``contest_cpu``. Default ``contest_cuda``
    per the wrapped script's default. Per CLAUDE.md non-negotiable + Catalog
    #192, macOS-CPU is NEVER valid here."""

    total_checks: int
    """Total check count emitted by the wrapped script."""

    passed_count: int
    """Count of checks with ``passed=True``."""

    error_count: int
    """Count of checks with ``severity=error`` AND ``passed=False``."""

    warning_count: int
    """Count of checks with ``severity=warning`` AND ``passed=False``."""

    all_checks: tuple[ComplianceCheck, ...]
    """Canonical-ordered tuple of every check the wrapped script emitted."""

    error_checks: tuple[ComplianceCheck, ...]
    """Subset of all_checks with severity=error AND passed=False."""

    operator_gated_remaining: tuple[ComplianceCheck, ...]
    """Subset of error_checks classified as D3 (hosting) or D5 (paired
    auth-eval) operator-gated blockers per
    :data:`_OPERATOR_GATED_BLOCKER_PATTERNS`. Per Phase 1 spec memo Layer 4
    contract, this is the canonical operator-routable next-action surface."""

    catalog_gate_protection_summary: dict[str, int]
    """Per-Catalog-gate failed-check count (e.g., ``{"127": 0, "192": 0,
    "221": 5, ...}``). Used by Phase 7 operator runbook to surface
    Catalog-specific operator action."""

    forbidden_macos_axis_detected: bool
    """True iff the wrapped script emitted any check whose details
    reference a forbidden Darwin ARM64 / macOS CPU advisory hardware
    substrate per Catalog #192. When True, the verdict is structurally
    refused regardless of other checks."""

    json_report_path: str
    """Path to the JSON report the wrapped script emitted via
    ``--json-out``."""

    measurement_utc: str
    """ISO-8601 UTC timestamp of compliance enforcement."""

    axis_tag: str
    """Always ``"[predicted]"`` per Catalog #341 + canonical Provenance."""

    score_claim: bool
    """Always ``False`` per CLAUDE.md "Apples-to-apples evidence discipline"."""

    promotable: bool
    """Always ``False`` per Catalog #341 + #192."""

    evidence_grade: str
    """Always ``"[predicted; compliance-canonical]"`` per Catalog #287/#323."""

    canonical_helper_invocation: str
    """``"tac.submission_packet.enforce_contest_compliance"`` per Catalog #190."""

    canonical_equation_id: str
    """:data:`CANONICAL_EQUATION_ID` (per Catalog #344)."""

    canonical_equation_status: str
    """``"FORMALIZATION_PENDING"`` until Phase 10 first empirical anchor."""

    elapsed_seconds: float
    """Compliance enforcement elapsed wall-clock."""

    canonical_provenance: Mapping[str, Any] = field(default_factory=dict)
    """Per Catalog #323 canonical Provenance umbrella."""

    written_at_utc: str = ""
    """When persisted to a canonical ledger (caller-fills)."""

    written_pid: int = 0
    """Process PID that emitted the result."""

    written_host: str = ""
    """Host that emitted the result."""

    def __post_init__(self) -> None:
        if self.schema_version != COMPLIANCE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {COMPLIANCE_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}"
            )
        if not self.lane_id:
            raise ValueError("lane_id must be non-empty")
        if not self.substrate_id:
            raise ValueError("substrate_id must be non-empty")
        if len(self.archive_sha256) != 64:
            raise ValueError(
                f"archive_sha256 must be 64-char hex; got len={len(self.archive_sha256)}"
            )
        if self.archive_bytes < 0:
            raise ValueError("archive_bytes must be non-negative")
        if not self.submission_dir:
            raise ValueError("submission_dir must be non-empty")
        if not isinstance(self.overall_clean, bool):
            raise ValueError("overall_clean must be bool")
        if not isinstance(self.contest_final_strict, bool):
            raise ValueError("contest_final_strict must be bool")
        if self.submission_score_axis not in {"contest_cuda", "contest_cpu"}:
            raise ValueError(
                f"submission_score_axis must be contest_cuda or contest_cpu; "
                f"got {self.submission_score_axis!r}"
            )
        for label, value in (
            ("total_checks", self.total_checks),
            ("passed_count", self.passed_count),
            ("error_count", self.error_count),
            ("warning_count", self.warning_count),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be non-negative int")
        if self.passed_count + self.error_count + self.warning_count > self.total_checks:
            raise ValueError(
                "passed_count + error_count + warning_count must not exceed total_checks"
            )
        if not isinstance(self.all_checks, tuple):
            raise ValueError("all_checks must be a tuple (frozen)")
        for c in self.all_checks:
            if not isinstance(c, ComplianceCheck):
                raise ValueError("all_checks entries must be ComplianceCheck")
        if not isinstance(self.error_checks, tuple):
            raise ValueError("error_checks must be a tuple (frozen)")
        for c in self.error_checks:
            if not isinstance(c, ComplianceCheck):
                raise ValueError("error_checks entries must be ComplianceCheck")
            if c.severity != CheckSeverity.ERROR.value:
                raise ValueError(
                    f"error_checks entries must have severity={CheckSeverity.ERROR.value!r}; "
                    f"got {c.severity!r} for {c.check_name!r}"
                )
            if c.passed:
                raise ValueError(
                    f"error_checks entries must have passed=False; "
                    f"got passed=True for {c.check_name!r}"
                )
        if not isinstance(self.operator_gated_remaining, tuple):
            raise ValueError("operator_gated_remaining must be a tuple (frozen)")
        for c in self.operator_gated_remaining:
            if not isinstance(c, ComplianceCheck):
                raise ValueError("operator_gated_remaining entries must be ComplianceCheck")
            if not c.is_operator_gated:
                raise ValueError(
                    f"operator_gated_remaining entries must have is_operator_gated=True; "
                    f"got False for {c.check_name!r}"
                )
        # operator_gated_remaining MUST be a subset of error_checks
        error_check_names = {c.check_name for c in self.error_checks}
        for c in self.operator_gated_remaining:
            if c.check_name not in error_check_names:
                raise ValueError(
                    f"operator_gated_remaining entries must be subset of error_checks; "
                    f"{c.check_name!r} not in error_checks"
                )
        if not isinstance(self.catalog_gate_protection_summary, dict):
            raise ValueError("catalog_gate_protection_summary must be a dict")
        if not isinstance(self.forbidden_macos_axis_detected, bool):
            raise ValueError("forbidden_macos_axis_detected must be bool")
        # Per Catalog #192 + CLAUDE.md non-negotiable: macOS-CPU detection
        # structurally forces overall_clean=False
        if self.forbidden_macos_axis_detected and self.overall_clean:
            raise ValueError(
                "forbidden_macos_axis_detected=True is incompatible with "
                "overall_clean=True per Catalog #192 + CLAUDE.md "
                "'Submission auth eval — BOTH CPU AND CUDA' non-negotiable"
            )
        if not self.json_report_path:
            raise ValueError("json_report_path must be non-empty")
        if not self.measurement_utc:
            raise ValueError("measurement_utc must be non-empty")
        if self.axis_tag != PREDICTED_AXIS_TAG:
            raise ValueError(f"axis_tag must equal {PREDICTED_AXIS_TAG!r}; got {self.axis_tag!r}")
        if self.score_claim is not False:
            raise ValueError("score_claim must be False per Catalog #341")
        if self.promotable is not False:
            raise ValueError("promotable must be False per Catalog #341")
        if not self.evidence_grade.startswith("[predicted;"):
            raise ValueError(
                "evidence_grade must start with '[predicted;' per Catalog #287/#323"
            )
        if self.canonical_equation_id != CANONICAL_EQUATION_ID:
            raise ValueError(
                f"canonical_equation_id must equal {CANONICAL_EQUATION_ID!r}; "
                f"got {self.canonical_equation_id!r}"
            )
        if self.canonical_equation_status not in {"FORMALIZATION_PENDING", "REGISTERED"}:
            raise ValueError(
                "canonical_equation_status must be 'FORMALIZATION_PENDING' or 'REGISTERED' per Catalog #344"
            )
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if not isinstance(self.canonical_provenance, Mapping):
            raise ValueError("canonical_provenance must be a Mapping per Catalog #323")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lane_id": self.lane_id,
            "substrate_id": self.substrate_id,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": int(self.archive_bytes),
            "submission_dir": self.submission_dir,
            "overall_clean": bool(self.overall_clean),
            "contest_final_strict": bool(self.contest_final_strict),
            "submission_score_axis": self.submission_score_axis,
            "total_checks": int(self.total_checks),
            "passed_count": int(self.passed_count),
            "error_count": int(self.error_count),
            "warning_count": int(self.warning_count),
            "all_checks": [c.as_dict() for c in self.all_checks],
            "error_checks": [c.as_dict() for c in self.error_checks],
            "operator_gated_remaining": [c.as_dict() for c in self.operator_gated_remaining],
            "catalog_gate_protection_summary": dict(self.catalog_gate_protection_summary),
            "forbidden_macos_axis_detected": bool(self.forbidden_macos_axis_detected),
            "json_report_path": self.json_report_path,
            "measurement_utc": self.measurement_utc,
            "axis_tag": self.axis_tag,
            "score_claim": bool(self.score_claim),
            "promotable": bool(self.promotable),
            "evidence_grade": self.evidence_grade,
            "canonical_helper_invocation": self.canonical_helper_invocation,
            "canonical_equation_id": self.canonical_equation_id,
            "canonical_equation_status": self.canonical_equation_status,
            "elapsed_seconds": float(self.elapsed_seconds),
            "canonical_provenance": dict(self.canonical_provenance),
            "written_at_utc": self.written_at_utc,
            "written_pid": int(self.written_pid),
            "written_host": self.written_host,
        }


# ---------------------------------------------------------------------------
# Core API helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Canonical UTC timestamp (ISO-8601 with tz)."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def derive_compliance_provenance(
    *,
    lane_id: str,
    substrate_id: str,
    archive_sha256: str,
    measurement_utc: str,
) -> dict[str, Any]:
    """Build the canonical Provenance dict for a compliance verdict.

    Per Catalog #323 canonical Provenance umbrella: every persisted row
    carries (axis_tag + evidence_grade + score_claim + promotable +
    canonical_helper_invocation + captured_at_utc).
    """
    return {
        "axis_tag": PREDICTED_AXIS_TAG,
        "evidence_grade": "[predicted; compliance-canonical]",
        "score_claim": False,
        "promotable": False,
        "canonical_helper_invocation": (
            "tac.submission_packet.enforce_contest_compliance"
        ),
        "captured_at_utc": measurement_utc,
        "lane_id": lane_id,
        "substrate_id": substrate_id,
        "archive_sha256": archive_sha256,
        "canonical_equation_id": CANONICAL_EQUATION_ID,
        "canonical_equation_status": "FORMALIZATION_PENDING",
        "schema_version": COMPLIANCE_SCHEMA_VERSION,
    }


def _classify_check_catalog_gates(check_name: str) -> tuple[int, ...]:
    """Classify a wrapped-script check name by Catalog gate(s).

    Returns the canonical sorted tuple of Catalog gates this check
    operationalizes. Empty tuple when the check name does not match any
    known Catalog gate prefix.
    """
    matched: set[int] = set()
    for gate, prefixes in _CATALOG_GATE_PREFIXES.items():
        for prefix in prefixes:
            if check_name.startswith(prefix):
                matched.add(gate)
                break
    return tuple(sorted(matched))


def _is_operator_gated_blocker(check_name: str) -> bool:
    """Classify a check name as operator-gated (D3 hosting / D5 paired axis).

    Per Phase 1 spec memo Layer 4 + 2026-05-19 sister landing memo
    `feedback_pr_submission_d5_prerequisites_executed_landed_20260519T182635Z.md`:
    18-of-39 checks the wrapped script emits for a baseline submission_dir
    require operator-side D3 (hosting) or D5 (paired axis eval) artifacts
    that this helper cannot produce by construction.
    """
    return any(
        check_name.startswith(pattern)
        for pattern in _OPERATOR_GATED_BLOCKER_PATTERNS
    )


def _derive_remediation_hint(
    *,
    check_name: str,
    catalog_gate_refs: tuple[int, ...],
    is_operator_gated: bool,
) -> str:
    """Derive operator-actionable remediation hint per Catalog gate.

    Returns a canonical short hint citing the Catalog gate(s) and the
    operator-routable next action.
    """
    if is_operator_gated:
        if check_name.startswith(("hosted_archive_", "public_source_")):
            return (
                "OPERATOR-GATED D3 (hosting): host archive.zip + source ref "
                "at public URL (Cloudflare/Lightning/release manifest) and "
                "supply --hosted-archive-manifest-json + "
                "--public-source-ref-manifest-json"
            )
        if check_name.startswith(("auth_eval_", "contest_cpu_auth_eval_")):
            return (
                "OPERATOR-GATED D5 (paired auth-eval): run paired Modal CUDA "
                "+ Linux x86_64 CPU auth-eval via tools/dispatch_modal_"
                "paired_auth_eval.py (Phase 6 sister) and supply "
                "--auth-eval-json + --contest-cpu-auth-eval-json per "
                "CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' "
                "non-negotiable"
            )
        if check_name.startswith("expected_runtime_tree_"):
            return (
                "OPERATOR-GATED D5: runtime tree sha emitted by post-dispatch "
                "auth-eval; supply --expected-runtime-tree-sha256 after "
                "paired Modal dispatch lands"
            )
        if check_name.startswith("runtime_equivalence_proof_"):
            return (
                "OPERATOR-GATED D5: runtime equivalence proof requires "
                "post-dispatch source-vs-candidate byte parity; supply "
                "--runtime-equivalence-proof-json"
            )
        return "OPERATOR-GATED: requires post-dispatch artifact per CLAUDE.md non-negotiable"
    if 192 in catalog_gate_refs:
        return (
            "STRUCTURAL: Catalog #192 macOS-CPU non-promotion violation; "
            "re-run on Linux x86_64 1:1 contest-compliant hardware per "
            "CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' "
            "non-negotiable"
        )
    if 127 in catalog_gate_refs:
        return (
            "STRUCTURAL: Catalog #127 authoritative-tag custody violation; "
            "route through tac.continual_learning.validate_custody before "
            "claiming axis"
        )
    if 152 in catalog_gate_refs:
        return (
            "STRUCTURAL: Catalog #152 required-input-file validation "
            "failure; supply --expected-archive-sha256 / "
            "--expected-archive-size-bytes / --expected-lane-id / "
            "--expected-job-id per the operator-authorize wrapper"
        )
    if 146 in catalog_gate_refs:
        return (
            "STRUCTURAL: Catalog #146 inflate runtime template violation; "
            "rebuild submission_dir via tac.submission_packet.build_submission_bundle "
            "(Phase 4 Layer 2)"
        )
    if 240 in catalog_gate_refs:
        return (
            "STRUCTURAL: Catalog #240 recipe-vs-trainer-state inconsistency; "
            "verify recipe carries research_only=true OR full _full_main "
            "implementation"
        )
    if 266 in catalog_gate_refs:
        return (
            "STRUCTURAL: Catalog #266 archive-bytes-consumed-by-inflate "
            "proof failure; verify byte-mutation smoke via "
            "tac.submission_packet.verify_byte_mutation_smoke_via_canonical_helper"
        )
    if 221 in catalog_gate_refs:
        return (
            "STRUCTURAL: Catalog #221 auth-eval result artifact fail-closed; "
            "re-run auth-eval via canonical gate_auth_eval_call helper"
        )
    if 226 in catalog_gate_refs:
        return (
            "STRUCTURAL: Catalog #226 canonical gate_auth_eval_call helper "
            "routing failure; verify trainer routes through "
            "tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call"
        )
    return "STRUCTURAL: review wrapped-script details and re-emit submission bundle"


def _detect_forbidden_macos_axis(check_details: str) -> bool:
    """Detect Catalog #192 forbidden Darwin ARM64 / macOS CPU hardware
    substrate references in a check details string.
    """
    lowered = check_details.lower()
    return any(
        token in lowered for token in _FORBIDDEN_AUTHORITATIVE_HARDWARE_TOKENS
    )


def _parse_wrapped_script_report(
    *,
    report_payload: Mapping[str, Any],
) -> tuple[
    tuple[ComplianceCheck, ...],
    tuple[ComplianceCheck, ...],
    tuple[ComplianceCheck, ...],
    dict[str, int],
    bool,
]:
    """Parse a wrapped-script JSON report into canonical ComplianceCheck tuples.

    Returns:
        all_checks, error_checks, operator_gated_remaining,
        catalog_gate_protection_summary, forbidden_macos_axis_detected
    """
    raw_checks = report_payload.get("checks", [])
    if not isinstance(raw_checks, list):
        raise SubmissionComplianceError(
            f"wrapped-script report 'checks' field must be a list; "
            f"got {type(raw_checks).__name__}"
        )
    all_checks: list[ComplianceCheck] = []
    error_checks: list[ComplianceCheck] = []
    operator_gated: list[ComplianceCheck] = []
    catalog_summary: dict[int, int] = {gate: 0 for gate in _CATALOG_GATE_PREFIXES}
    forbidden_macos = False
    for raw in raw_checks:
        if not isinstance(raw, Mapping):
            continue
        check_name = str(raw.get("name", ""))
        severity = str(raw.get("severity", "error"))
        # The wrapped script's Check.severity may be 'error' / 'warning' /
        # 'info'; normalize to canonical taxonomy.
        if severity not in {s.value for s in CheckSeverity}:
            severity = CheckSeverity.ERROR.value
        passed = bool(raw.get("passed", False))
        details = str(raw.get("details", ""))
        if not check_name:
            continue
        catalog_refs = _classify_check_catalog_gates(check_name)
        is_op_gated = _is_operator_gated_blocker(check_name) and not passed
        remediation = _derive_remediation_hint(
            check_name=check_name,
            catalog_gate_refs=catalog_refs,
            is_operator_gated=is_op_gated,
        )
        check = ComplianceCheck(
            check_name=check_name,
            severity=severity,
            passed=passed,
            details=details,
            catalog_gate_refs=catalog_refs,
            is_operator_gated=is_op_gated,
            remediation_hint=remediation,
        )
        all_checks.append(check)
        if not passed:
            for gate in catalog_refs:
                catalog_summary[gate] = catalog_summary.get(gate, 0) + 1
            if _detect_forbidden_macos_axis(details):
                forbidden_macos = True
            if severity == CheckSeverity.ERROR.value:
                error_checks.append(check)
                if is_op_gated:
                    operator_gated.append(check)
    return (
        tuple(all_checks),
        tuple(error_checks),
        tuple(operator_gated),
        {str(gate): count for gate, count in catalog_summary.items()},
        forbidden_macos,
    )


def enforce_contest_compliance(
    *,
    submission_bundle_result: SubmissionBundleResult,
    contest_final_strict: bool = True,
    submission_score_axis: str = "contest_cuda",
    expected_lane_id: str | None = None,
    expected_job_id: str | None = None,
    auth_eval_json_path: Path | None = None,
    contest_cpu_auth_eval_json_path: Path | None = None,
    archive_manifest_json_path: Path | None = None,
    runtime_equivalence_proof_json_path: Path | None = None,
    hosted_archive_manifest_json_path: Path | None = None,
    public_source_ref_manifest_json_path: Path | None = None,
    competitive_or_innovative_statement: str | None = None,
    output_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
    canonical_script_path: Path | str | None = None,
    python_executable: str | None = None,
    subprocess_timeout_seconds: float = 120.0,
) -> ComplianceVerdict:
    """Enforce contest compliance for a Phase 4 SubmissionBundleResult.

    Routes through ``scripts/pre_submission_compliance_check.py`` via
    subprocess + structured JSON parsing. Returns a typed
    :class:`ComplianceVerdict` with per-Catalog-gate categorization +
    operator-gated D3/D5 blocker classification + Catalog #192 macOS-CPU
    structural refusal.

    Args:
        submission_bundle_result: Phase 4 Layer 2 output (canonical
            consumer surface; provides archive_sha256 + archive_bytes +
            submission_dir + lane_id + substrate_id).
        contest_final_strict: when True, invokes the wrapped script with
            ``--contest-final --strict`` per CLAUDE.md non-negotiable.
        submission_score_axis: ``contest_cuda`` (default) or
            ``contest_cpu``. Per Catalog #192, never macOS substrate.
        expected_lane_id: caller-supplied lane id (typically matches
            submission_bundle_result.lane_id); when None, defaults to it.
        expected_job_id: optional Modal/Lightning/Vast.ai job id for
            dispatch-claim linkage per Catalog #152.
        auth_eval_json_path: path to CUDA auth-eval JSON artifact (D5
            paired axis; when None, the wrapped script looks for
            ``submission_dir/contest_auth_eval.json``).
        contest_cpu_auth_eval_json_path: path to CPU auth-eval JSON
            artifact (D5 paired axis; when None, the wrapped script looks
            for ``submission_dir/contest_cpu_auth_eval.json``).
        archive_manifest_json_path: optional per-member identity manifest
            (defaults to ``submission_dir/archive_manifest.json``).
        runtime_equivalence_proof_json_path: optional Catalog #266 proof
            artifact (D5 paired-runtime byte-parity).
        hosted_archive_manifest_json_path: optional Catalog D3 public
            hosting manifest.
        public_source_ref_manifest_json_path: optional Catalog D3 public
            source-ref manifest.
        competitive_or_innovative_statement: optional PR101+ contest
            requirement statement.
        output_dir: where to write the wrapped-script JSON report
            (defaults to ``reports/pr_pre_submission/``).
        repo_root: repository root (defaults to
            :data:`REPO_ROOT`).
        canonical_script_path: path to canonical compliance script
            (defaults to
            ``{repo_root}/scripts/pre_submission_compliance_check.py``).
        python_executable: Python interpreter to invoke (defaults to
            ``sys.executable``).
        subprocess_timeout_seconds: hard timeout for wrapped-script
            invocation (default 120s).

    Returns:
        :class:`ComplianceVerdict` with canonical Provenance.

    Raises:
        SubmissionComplianceError: structural failures (missing script,
            unparseable JSON output, subprocess timeout, fail-closed-on-
            macOS-substrate detection).
        ValueError: caller-side argument violations (bad axis, etc.).
    """
    if submission_score_axis not in {"contest_cuda", "contest_cpu"}:
        raise ValueError(
            f"submission_score_axis must be contest_cuda or contest_cpu; "
            f"got {submission_score_axis!r}"
        )
    if not isinstance(submission_bundle_result, SubmissionBundleResult):
        raise ValueError(
            "submission_bundle_result must be a SubmissionBundleResult instance "
            "from tac.submission_packet.build_submission_bundle"
        )
    repo_root_path = Path(repo_root) if repo_root else REPO_ROOT
    canonical_script = (
        Path(canonical_script_path)
        if canonical_script_path
        else repo_root_path / CANONICAL_COMPLIANCE_SCRIPT_PATH
    )
    if not canonical_script.is_file():
        raise SubmissionComplianceError(
            f"canonical compliance script missing at {canonical_script}; "
            f"expected {CANONICAL_COMPLIANCE_SCRIPT_PATH}"
        )
    output_dir_path = (
        Path(output_dir)
        if output_dir
        else repo_root_path / "reports/pr_pre_submission"
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)
    measurement_utc = _utc_now_iso()
    measurement_compact = (
        measurement_utc.replace(":", "").replace("-", "").split(".")[0]
    )
    json_report_path = (
        output_dir_path
        / f"compliance_report_{submission_bundle_result.substrate_id}_{measurement_compact}.json"
    )

    submission_dir = repo_root_path / submission_bundle_result.submission_dir
    if not submission_dir.is_dir():
        # Allow absolute paths too
        submission_dir = Path(submission_bundle_result.submission_dir)
    if not submission_dir.is_dir():
        raise SubmissionComplianceError(
            f"submission_dir not found at {submission_dir}; "
            f"submission_bundle_result.submission_dir="
            f"{submission_bundle_result.submission_dir!r}"
        )

    py_exec = python_executable or sys.executable
    argv: list[str] = [
        py_exec,
        str(canonical_script),
        "--submission-dir", str(submission_dir),
        "--expected-archive-sha256", submission_bundle_result.archive_sha256,
        "--expected-archive-size-bytes", str(submission_bundle_result.archive_bytes),
        "--submission-score-axis", submission_score_axis,
        "--json-out", str(json_report_path),
    ]
    if contest_final_strict:
        argv.extend(["--contest-final", "--strict"])
    lane = expected_lane_id or submission_bundle_result.lane_id
    if lane:
        argv.extend(["--expected-lane-id", lane])
    if expected_job_id:
        argv.extend(["--expected-job-id", expected_job_id])
    if auth_eval_json_path:
        argv.extend(["--auth-eval-json", str(auth_eval_json_path)])
    if contest_cpu_auth_eval_json_path:
        argv.extend([
            "--contest-cpu-auth-eval-json",
            str(contest_cpu_auth_eval_json_path),
        ])
    if archive_manifest_json_path:
        argv.extend([
            "--archive-manifest-json",
            str(archive_manifest_json_path),
        ])
    if runtime_equivalence_proof_json_path:
        argv.extend([
            "--runtime-equivalence-proof-json",
            str(runtime_equivalence_proof_json_path),
        ])
    if hosted_archive_manifest_json_path:
        argv.extend([
            "--hosted-archive-manifest-json",
            str(hosted_archive_manifest_json_path),
        ])
    if public_source_ref_manifest_json_path:
        argv.extend([
            "--public-source-ref-manifest-json",
            str(public_source_ref_manifest_json_path),
        ])
    if competitive_or_innovative_statement:
        argv.extend([
            "--competitive-or-innovative-statement",
            competitive_or_innovative_statement,
        ])
        argv.append("--require-competitive-or-innovative-statement")
    start = datetime.datetime.now()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=str(repo_root_path),
            timeout=subprocess_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SubmissionComplianceError(
            f"wrapped script timed out after {subprocess_timeout_seconds}s; "
            f"argv={argv!r}"
        ) from exc
    except OSError as exc:
        raise SubmissionComplianceError(
            f"wrapped script invocation failed: {exc!r}; argv={argv!r}"
        ) from exc
    elapsed = (datetime.datetime.now() - start).total_seconds()

    # The wrapped script returns rc=0 on PASS or rc=1 on strict-fail; both
    # cases emit a JSON report to --json-out (or stdout when --json-out is
    # absent). A rc!=0/1 indicates a structural script failure (e.g.,
    # uncaught exception) and we surface that as SubmissionComplianceError.
    if completed.returncode not in {0, 1}:
        raise SubmissionComplianceError(
            f"wrapped script crashed with rc={completed.returncode}; "
            f"stderr={completed.stderr[:500]!r}; argv={argv!r}"
        )
    if not json_report_path.is_file():
        raise SubmissionComplianceError(
            f"wrapped script did not emit JSON report at {json_report_path}; "
            f"rc={completed.returncode} stderr={completed.stderr[:500]!r}"
        )
    try:
        report_payload = json.loads(json_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SubmissionComplianceError(
            f"wrapped script emitted unparseable JSON at {json_report_path}: {exc!r}"
        ) from exc
    if not isinstance(report_payload, Mapping):
        raise SubmissionComplianceError(
            f"wrapped script JSON report must be a Mapping; "
            f"got {type(report_payload).__name__}"
        )
    (
        all_checks,
        error_checks,
        operator_gated_remaining,
        catalog_gate_summary,
        forbidden_macos_axis,
    ) = _parse_wrapped_script_report(report_payload=report_payload)
    passed_count = sum(1 for c in all_checks if c.passed)
    warning_count = sum(
        1 for c in all_checks if not c.passed and c.severity == CheckSeverity.WARNING.value
    )
    error_count = len(error_checks)
    total_checks = len(all_checks)
    # Per Catalog #192 + CLAUDE.md non-negotiable: macOS substrate forces
    # structural refusal regardless of wrapped-script overall passed.
    overall_clean = (
        bool(report_payload.get("passed", False))
        and error_count == 0
        and not forbidden_macos_axis
    )
    provenance = derive_compliance_provenance(
        lane_id=submission_bundle_result.lane_id,
        substrate_id=submission_bundle_result.substrate_id,
        archive_sha256=submission_bundle_result.archive_sha256,
        measurement_utc=measurement_utc,
    )
    verdict = ComplianceVerdict(
        schema_version=COMPLIANCE_SCHEMA_VERSION,
        lane_id=submission_bundle_result.lane_id,
        substrate_id=submission_bundle_result.substrate_id,
        archive_sha256=submission_bundle_result.archive_sha256,
        archive_bytes=submission_bundle_result.archive_bytes,
        submission_dir=str(submission_dir),
        overall_clean=overall_clean,
        contest_final_strict=contest_final_strict,
        submission_score_axis=submission_score_axis,
        total_checks=total_checks,
        passed_count=passed_count,
        error_count=error_count,
        warning_count=warning_count,
        all_checks=all_checks,
        error_checks=error_checks,
        operator_gated_remaining=operator_gated_remaining,
        catalog_gate_protection_summary=catalog_gate_summary,
        forbidden_macos_axis_detected=forbidden_macos_axis,
        json_report_path=str(json_report_path),
        measurement_utc=measurement_utc,
        axis_tag=PREDICTED_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; compliance-canonical]",
        canonical_helper_invocation=(
            "tac.submission_packet.enforce_contest_compliance"
        ),
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=elapsed,
        canonical_provenance=provenance,
        written_at_utc=measurement_utc,
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )
    return verdict


__all__ = [
    "CANONICAL_COMPLIANCE_SCRIPT_PATH",
    "CANONICAL_EQUATION_ID",
    "COMPLIANCE_SCHEMA_VERSION",
    "CheckSeverity",
    "ComplianceCheck",
    "ComplianceVerdict",
    "PHASE_6_LAYER_VERSION",
    "PREDICTED_AXIS_TAG",
    "SubmissionComplianceError",
    "derive_compliance_provenance",
    "enforce_contest_compliance",
]
