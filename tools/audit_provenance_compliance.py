#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""tools/audit_provenance_compliance.py — operator-runnable sweep for canonical
Provenance compliance across the repo.

Walks the canonical artifact directories and classifies every score-claiming
row as CLEAN / WARN / VIOLATION with per-violation classifier.

Per CLAUDE.md "Operator gates must be wired and used": this tool is the
operator-facing audit surface for Catalog #323. The companion STRICT
preflight gate ``check_no_score_claim_without_canonical_provenance`` is the
structural enforcement; this tool produces the human-readable report.

Usage:

    .venv/bin/python tools/audit_provenance_compliance.py \\
        --scan-root . \\
        --report-out .omx/state/provenance_compliance_audit_<utc>.json

    .venv/bin/python tools/audit_provenance_compliance.py --summary

    .venv/bin/python tools/audit_provenance_compliance.py --strict-fail-on-violation

Exit codes:
  0 = no violations (or no artifacts to scan)
  1 = at least 1 violation found AND --strict-fail-on-violation set
  2 = CLI error (bad args, etc.)

The audit scans:
  * ``.omx/state/**/*.json`` and ``.omx/state/**/*.jsonl``
  * ``experiments/results/**/build_manifest.json``
  * ``experiments/results/**/auth_eval_*.json``
  * ``experiments/results/**/contest_auth_eval_*.json``
  * ``experiments/results/**/optimal_plan_*.json``
  * ``submissions/*/dual_eval_adjudicated.json``
  * ``reports/**/*.json``

Per-artifact verdict taxonomy:
  * CLEAN — no score-claim keys, OR score-claim keys present with valid Provenance
  * WARN — score-claim keys present with Provenance but minor issues (stale captured_at_utc, etc.)
  * VIOLATION — score-claim keys present without canonical Provenance,
    OR Provenance.score_claim_valid=False but non-zero score (phantom class)

Per-violation classifier:
  * MISSING_PROVENANCE — no 'provenance' sub-object
  * MISSING_CONTEST_COMPLIANCE_RATIONALE — Wyner-Ziv DeliverabilityProof-style
    score-savings artifact lacks a contest-compliance rationale or citation chain
  * BYTE_IDENTITY_ARTIFACT — composed_from parts share sha256 (Catalog #823)
  * RESEARCH_SIDECAR_SCORE_CLAIMED — Catalog #321 anchor
  * AXIS_HARDWARE_MISMATCH — measurement_axis ≠ hardware_substrate canonical pairing
  * FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD — output-affecting payload bytes live
    outside archive.zip; fail-closed before dispatch
  * INVALID_PROVENANCE_SHAPE — cannot reconstruct as canonical Provenance
  * STALE_CAPTURED_AT — captured_at_utc > 90 days old
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Defer tac import so the tool can be run from a fresh checkout w/o setup
REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent


@dataclass
class ArtifactVerdict:
    """Per-artifact audit verdict."""

    path: str
    verdict: str  # "CLEAN" | "WARN" | "VIOLATION"
    score_claim_keys_present: list[str] = field(default_factory=list)
    violation_classifiers: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    row_count: int = 0  # for JSONL files
    rows_with_violations: int = 0


@dataclass
class ProvenanceAuditReport:
    """Aggregate report across all scanned artifacts."""

    schema_version: str = "provenance_audit_v1_20260517"
    scan_root: str = ""
    scanned_at_utc: str = ""
    total_artifacts_scanned: int = 0
    clean_count: int = 0
    warn_count: int = 0
    violation_count: int = 0
    classifier_counts: dict[str, int] = field(default_factory=dict)
    artifact_verdicts: list[ArtifactVerdict] = field(default_factory=list)


# Score-claim key patterns scanned for (must match Catalog #323 gate)
SCORE_CLAIM_KEYS: tuple[str, ...] = (
    "score",
    "score_value",
    "contest_score",
    "final_score",
    "predicted_score",
    "canonical_score",
    "canonical_score_recomputed",
    "canonical_score_contest_cuda",
    "canonical_score_contest_cpu",
    "score_recomputed",
    "score_recomputed_from_components",
    "score_recomputed_from_contest_components",
    "score_recomputed_from_public_components",
    "score_contest_cuda",
    "score_contest_cpu",
    "contest_cuda_score_recomputed",
    "contest_cpu_score_recomputed",
    "empirical_score",
    "diagnostic_cpu_score",
    "deliverable_score_savings_estimate",
    "deliverable_savings",
    "composition_alpha",
    "alpha",
    "alpha_savings_ratio_form",
    "savings",
    "delta_s",
    "delta_score",
    "score_savings",
    "auth_eval_score",
    "auth_eval_recomputed_score",
    "score_recomputed_from_auth_eval",
    "recomputed_score",
)


DELIVERABILITY_PROOF_MARKER_KEYS: tuple[str, ...] = (
    "deliverable_score_savings_estimate",
    "contest_compliance_verdict",
    "contest_compliance_rationale",
    "contest_compliance_citation_chain",
    "candidate_shared_prior_byte_count",
    "tier_1_byte_count",
    "tier_2_byte_count",
    "tier_3_byte_count",
    "tier_4_byte_count",
)


LEGAL_CONTEST_COMPLIANCE_CITATION_ANCHORS: frozenset[str] = frozenset(
    {
        "archive.zip seed inclusion",
        "weight-derived codebook from shipped archive bytes",
        "null-space in-archive byte reduction",
        "reviewability-only no-score-impact",
    }
)


# Path patterns to scan (relative to scan root)
SCAN_PATTERNS: tuple[str, ...] = (
    ".omx/state/**/*.json",
    ".omx/state/**/*.jsonl",
    "experiments/results/**/build_manifest.json",
    "experiments/results/**/auth_eval_*.json",
    "experiments/results/**/contest_auth_eval_*.json",
    "experiments/results/**/optimal_plan_*.json",
    "submissions/*/dual_eval_adjudicated.json",
    "reports/**/*.json",
)


# Path markers to EXCLUDE from scanning (per CLAUDE.md exempt-marker conventions)
EXCLUDE_MARKERS: tuple[str, ...] = (
    "/.venv/",
    "/__pycache__/",
    "/node_modules/",
    "/.git/",
    "/build/lib/",
    "_intake_",
    "/.omx/oss_export/",
    "/vendored/",
    "/reports/raw/",
    "/.omx/state/archive/",  # archived JSONL state per CLAUDE.md "State JSONL archival policy"
    "/.omx/state/quarantine",  # quarantined artifacts
)


def _is_excluded(path: Path) -> bool:
    s = str(path)
    return any(marker in s for marker in EXCLUDE_MARKERS)


def _looks_like_deliverability_proof(payload: dict[str, Any]) -> bool:
    """Return True for persisted Wyner-Ziv DeliverabilityProof-style payloads."""
    marker_count = sum(1 for key in DELIVERABILITY_PROOF_MARKER_KEYS if key in payload)
    has_wz_score_key = "deliverable_score_savings_estimate" in payload
    has_tier_shape = (
        "candidate_shared_prior_byte_count" in payload
        and any(f"tier_{tier}_byte_count" in payload for tier in range(1, 5))
    )
    has_compliance_shape = any(
        key in payload
        for key in (
            "contest_compliance_verdict",
            "contest_compliance_rationale",
            "contest_compliance_citation_chain",
        )
    )
    return has_wz_score_key or (has_tier_shape and has_compliance_shape) or marker_count >= 3


def _contest_compliance_blockers(payload: dict[str, Any]) -> list[str]:
    """Audit persisted DeliverabilityProof rationale fields without importing tac."""
    if not _looks_like_deliverability_proof(payload):
        return []

    blockers: list[str] = []
    rationale = payload.get("contest_compliance_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        blockers.append(
            "DeliverabilityProof-style score-savings artifact is missing "
            "non-empty contest_compliance_rationale"
        )

    citation_chain = payload.get("contest_compliance_citation_chain")
    if not isinstance(citation_chain, (list, tuple)):
        blockers.append(
            "DeliverabilityProof-style score-savings artifact is missing "
            "contest_compliance_citation_chain list/tuple"
        )
        return blockers

    normalized = tuple(
        str(citation).strip() for citation in citation_chain if str(citation).strip()
    )
    if not normalized:
        blockers.append(
            "DeliverabilityProof-style score-savings artifact has empty "
            "contest_compliance_citation_chain"
        )
    elif not (set(normalized) & LEGAL_CONTEST_COMPLIANCE_CITATION_ANCHORS):
        blockers.append(
            "contest_compliance_citation_chain must cite a contest-compliant route: "
            "archive.zip seed inclusion, weight-derived codebook from shipped "
            "archive bytes, null-space in-archive byte reduction, or "
            "reviewability-only no-score-impact"
        )
    return blockers


def _classify_payload(
    payload: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    """Classify a single dict payload.

    Returns: (verdict, violation_classifiers, blockers)
    """
    score_keys_present = [k for k in SCORE_CLAIM_KEYS if k in payload]

    if not score_keys_present:
        return "CLEAN", [], []

    classifiers: list[str] = []
    blockers: list[str] = []

    compliance_blockers = _contest_compliance_blockers(payload)
    if compliance_blockers:
        classifiers.append("MISSING_CONTEST_COMPLIANCE_RATIONALE")
        blockers.extend(compliance_blockers)

    # Score keys present → require provenance
    if "provenance" not in payload:
        classifiers.append("MISSING_PROVENANCE")
        blockers.append(
            f"score-claim keys {score_keys_present!r} but no 'provenance' field"
        )
        return "VIOLATION", list(dict.fromkeys(classifiers)), blockers

    prov_dict = payload["provenance"]
    if not isinstance(prov_dict, dict):
        classifiers.append("INVALID_PROVENANCE_SHAPE")
        blockers.append(
            f"'provenance' field is not a dict (type={type(prov_dict).__name__})"
        )
        return "VIOLATION", list(dict.fromkeys(classifiers)), blockers

    # Reconstruct + validate
    try:
        from tac.provenance import audit_score_claim_dict
        valid, prov_blockers = audit_score_claim_dict(payload)
        if not valid:
            classifiers.append("INVALID_PROVENANCE_SHAPE")
            blockers.extend(prov_blockers)

            # Sub-classifier: byte-identity?
            for b in prov_blockers:
                if "byte-identical" in b.lower() or "byte_identity" in b.lower():
                    classifiers.append("BYTE_IDENTITY_ARTIFACT")
                    break
                if "research" in b.lower() and "non-zero" in b.lower():
                    classifiers.append("RESEARCH_SIDECAR_SCORE_CLAIMED")
                if "stale" in b.lower():
                    classifiers.append("STALE_CAPTURED_AT")
                if "axis" in b.lower() or "hardware" in b.lower():
                    classifiers.append("AXIS_HARDWARE_MISMATCH")
                if "FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD" in b:
                    classifiers.append("FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD")
            return "VIOLATION", sorted(set(classifiers)), blockers
    except Exception as exc:
        classifiers.append("INVALID_PROVENANCE_SHAPE")
        blockers.append(f"audit failed with exception: {exc}")
        return "VIOLATION", list(dict.fromkeys(classifiers)), blockers

    if compliance_blockers:
        return "VIOLATION", list(dict.fromkeys(classifiers)), blockers

    return "CLEAN", [], []


def _audit_file(path: Path) -> ArtifactVerdict:
    """Audit a single JSON or JSONL file."""
    verdict = ArtifactVerdict(path=str(path), verdict="CLEAN")

    try:
        raw = path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        verdict.verdict = "WARN"
        verdict.blockers.append(f"read failed: {exc}")
        return verdict

    if not raw.strip():
        return verdict

    # Detect JSONL vs JSON by extension
    if path.suffix == ".jsonl":
        rows = []
        for _line_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        verdict.row_count = len(rows)

        any_violation = False
        any_warn = False
        all_keys: set[str] = set()
        all_classifiers: set[str] = set()
        all_blockers: list[str] = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            row_verdict, classifiers, blockers = _classify_payload(row)
            if row_verdict == "VIOLATION":
                any_violation = True
                verdict.rows_with_violations += 1
                all_classifiers.update(classifiers)
                all_blockers.extend(blockers[:3])  # cap to avoid noise
            elif row_verdict == "WARN":
                any_warn = True
            # collect keys for transparency
            for k in SCORE_CLAIM_KEYS:
                if k in row:
                    all_keys.add(k)

        verdict.score_claim_keys_present = sorted(all_keys)
        verdict.violation_classifiers = sorted(all_classifiers)
        verdict.blockers = all_blockers[:10]
        if any_violation:
            verdict.verdict = "VIOLATION"
        elif any_warn:
            verdict.verdict = "WARN"
    else:
        # Single JSON document
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            verdict.verdict = "WARN"
            verdict.blockers.append(f"json parse failed: {exc}")
            return verdict

        if isinstance(payload, dict):
            row_verdict, classifiers, blockers = _classify_payload(payload)
            verdict.verdict = row_verdict
            verdict.violation_classifiers = classifiers
            verdict.blockers = blockers
            verdict.score_claim_keys_present = [
                k for k in SCORE_CLAIM_KEYS if k in payload
            ]
            if row_verdict == "VIOLATION":
                verdict.rows_with_violations = 1
            verdict.row_count = 1
        elif isinstance(payload, list):
            any_violation = False
            any_warn = False
            all_keys: set[str] = set()
            all_classifiers: set[str] = set()
            all_blockers: list[str] = []
            verdict.row_count = len(payload)
            for item in payload:
                if not isinstance(item, dict):
                    continue
                row_verdict, classifiers, blockers = _classify_payload(item)
                if row_verdict == "VIOLATION":
                    any_violation = True
                    verdict.rows_with_violations += 1
                    all_classifiers.update(classifiers)
                    all_blockers.extend(blockers[:3])
                elif row_verdict == "WARN":
                    any_warn = True
                for k in SCORE_CLAIM_KEYS:
                    if k in item:
                        all_keys.add(k)
            verdict.score_claim_keys_present = sorted(all_keys)
            verdict.violation_classifiers = sorted(all_classifiers)
            verdict.blockers = all_blockers[:10]
            if any_violation:
                verdict.verdict = "VIOLATION"
            elif any_warn:
                verdict.verdict = "WARN"

    return verdict


def build_audit_report(scan_root: Path) -> ProvenanceAuditReport:
    """Walk patterns and aggregate verdicts."""
    report = ProvenanceAuditReport(
        scan_root=str(scan_root),
        scanned_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    seen: set[Path] = set()
    classifier_counts: dict[str, int] = defaultdict(int)

    for pattern in SCAN_PATTERNS:
        for path in scan_root.glob(pattern):
            if path in seen or _is_excluded(path) or not path.is_file():
                continue
            seen.add(path)

            verdict = _audit_file(path)
            report.artifact_verdicts.append(verdict)
            report.total_artifacts_scanned += 1
            if verdict.verdict == "CLEAN":
                report.clean_count += 1
            elif verdict.verdict == "WARN":
                report.warn_count += 1
            elif verdict.verdict == "VIOLATION":
                report.violation_count += 1
                for c in verdict.violation_classifiers:
                    classifier_counts[c] += 1

    report.classifier_counts = dict(classifier_counts)
    return report


def render_summary(report: ProvenanceAuditReport) -> str:
    """Human-readable summary."""
    lines = []
    lines.append("=== Provenance Audit Compliance ===")
    lines.append(f"scan root: {report.scan_root}")
    lines.append(f"scanned at: {report.scanned_at_utc}")
    lines.append(f"total artifacts: {report.total_artifacts_scanned}")
    lines.append(f"  CLEAN: {report.clean_count}")
    lines.append(f"  WARN:  {report.warn_count}")
    lines.append(f"  VIOLATION: {report.violation_count}")
    if report.classifier_counts:
        lines.append("\nClassifier counts:")
        for k, v in sorted(report.classifier_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k}: {v}")
    if report.violation_count > 0:
        lines.append("\nTop 5 violations:")
        violations = [v for v in report.artifact_verdicts if v.verdict == "VIOLATION"]
        for v in violations[:5]:
            lines.append(f"  - {v.path}")
            lines.append(f"      classifiers: {v.violation_classifiers}")
            if v.blockers:
                lines.append(f"      first blocker: {v.blockers[0][:120]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=REPO_ROOT_DEFAULT,
        help="Root directory to scan (default: repo root).",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Path to write JSON report (default: stdout).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print human-readable summary instead of full JSON.",
    )
    parser.add_argument(
        "--strict-fail-on-violation",
        action="store_true",
        help="Exit rc=1 if any VIOLATION found.",
    )

    args = parser.parse_args(argv)

    if not args.scan_root.exists():
        print(f"Scan root not found: {args.scan_root}", file=sys.stderr)
        return 2

    report = build_audit_report(args.scan_root)

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        report_dict = {
            "schema_version": report.schema_version,
            "scan_root": report.scan_root,
            "scanned_at_utc": report.scanned_at_utc,
            "total_artifacts_scanned": report.total_artifacts_scanned,
            "clean_count": report.clean_count,
            "warn_count": report.warn_count,
            "violation_count": report.violation_count,
            "classifier_counts": report.classifier_counts,
            "artifact_verdicts": [asdict(v) for v in report.artifact_verdicts],
        }
        args.report_out.write_text(json.dumps(report_dict, indent=2, sort_keys=True))

    if args.summary or not args.report_out:
        print(render_summary(report))

    if args.strict_fail_on_violation and report.violation_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
