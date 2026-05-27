#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for Phase 6 Layer 4 contest compliance enforcement.

Wraps :func:`tac.submission_packet.enforce_contest_compliance` for
operator-runbook use. Routes through the canonical 3267-LOC
``scripts/pre_submission_compliance_check.py`` script via the typed
canonical helper, surfacing per-Catalog-gate categorized verdict + D3/D5
operator-gated blockers.

Per the 10th apples-to-apples standing directive: macOS-CPU substrate
references are STRUCTURALLY REFUSED as authoritative axes per Catalog
#192 + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
CONTEST-COMPLIANT HARDWARE" non-negotiable.

Exit codes:
  0 — CLEAN (all checks passed; submission is PR-ready)
  1 — COMPLIANCE-ERROR (one or more structural error checks failed)
  2 — PAIRED-AXIS-MISSING (D5 paired CPU+CUDA auth-eval not satisfied
      per CLAUDE.md non-negotiable; operator-routable next action is
      paired_auth_eval dispatch)
  3 — CUSTODY-MISMATCH (Catalog #127 authoritative-tag custody violation)
  4 — RECIPE-TRAINER-STATE-INCONSISTENT (Catalog #240 recipe vs trainer
      state divergence)
  5 — CLI error (bad arguments, missing files, structural failure)

Usage::

    .venv/bin/python tools/submission_compliance_cli.py \\
        --from-submission-bundle path/to/submission_bundle_result.json \\
        --contest-final-strict

    .venv/bin/python tools/submission_compliance_cli.py \\
        --from-submission-bundle - \\
        --submission-score-axis contest_cuda \\
        --auth-eval-json path/to/cuda_auth_eval.json \\
        --contest-cpu-auth-eval-json path/to/cpu_auth_eval.json \\
        --expected-job-id fc-01KS... \\
        --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure local tac is importable when invoked from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.submission_packet import (  # noqa: E402  (sys.path insert above)
    SubmissionBundleResult,
    SubmissionComplianceError,
    enforce_contest_compliance,
    submission_bundle_result_from_dict,
)

_EXIT_CLEAN = 0
_EXIT_COMPLIANCE_ERROR = 1
_EXIT_PAIRED_AXIS_MISSING = 2
_EXIT_CUSTODY_MISMATCH = 3
_EXIT_RECIPE_TRAINER_STATE_INCONSISTENT = 4
_EXIT_CLI_ERROR = 5


def _load_submission_bundle(path: str) -> SubmissionBundleResult:
    """Load a SubmissionBundleResult from a JSON file or stdin."""
    if path == "-":
        raw = sys.stdin.read()
    else:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"submission bundle JSON not found at {p}")
        raw = p.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"submission bundle JSON unparseable: {exc!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"submission bundle JSON must be a dict; got {type(payload).__name__}"
        )
    return submission_bundle_result_from_dict(payload)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from-submission-bundle",
        required=True,
        help="Path to SubmissionBundleResult JSON (from Phase 4 as_dict); '-' for stdin",
    )
    parser.add_argument(
        "--contest-final-strict",
        action="store_true",
        help=(
            "Invoke wrapped script with --contest-final --strict per "
            "CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' "
            "non-negotiable"
        ),
    )
    parser.add_argument(
        "--submission-score-axis",
        choices=["contest_cuda", "contest_cpu"],
        default="contest_cuda",
        help="Axis the packet is being submitted on (NEVER macOS per Catalog #192)",
    )
    parser.add_argument("--expected-lane-id")
    parser.add_argument("--expected-job-id")
    parser.add_argument("--auth-eval-json", type=Path)
    parser.add_argument("--contest-cpu-auth-eval-json", type=Path)
    parser.add_argument("--archive-manifest-json", type=Path)
    parser.add_argument("--runtime-equivalence-proof-json", type=Path)
    parser.add_argument("--hosted-archive-manifest-json", type=Path)
    parser.add_argument("--public-source-ref-manifest-json", type=Path)
    parser.add_argument("--competitive-or-innovative-statement")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to write the wrapped-script JSON report (default reports/pr_pre_submission/)",
    )
    parser.add_argument(
        "--subprocess-timeout-seconds",
        type=float,
        default=120.0,
        help="Hard timeout for the wrapped-script subprocess (default 120s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit ComplianceVerdict as JSON on stdout (machine-readable)",
    )
    return parser


def _classify_exit_code(verdict) -> int:
    """Classify a ComplianceVerdict into a CLI exit code.

    Per the canonical exit code taxonomy in this CLI's docstring.
    """
    if verdict.overall_clean:
        return _EXIT_CLEAN
    # Operator-gated D3/D5 dependencies dominate; if all errors are D5
    # paired-axis missing, return _EXIT_PAIRED_AXIS_MISSING (2).
    op_gated_count = len(verdict.operator_gated_remaining)
    error_count = verdict.error_count
    structural_count = error_count - op_gated_count
    # Catalog #127 custody mismatch
    cat127_failed = int(
        verdict.catalog_gate_protection_summary.get("127", 0)
    )
    # Catalog #240 recipe-trainer-state inconsistency
    cat240_failed = int(
        verdict.catalog_gate_protection_summary.get("240", 0)
    )
    # If only D3/D5 operator-gated blockers AND no structural blockers, return PAIRED_AXIS_MISSING
    if structural_count == 0 and op_gated_count > 0:
        return _EXIT_PAIRED_AXIS_MISSING
    # Structural blockers present; classify by dominant Catalog
    if cat127_failed > 0:
        return _EXIT_CUSTODY_MISMATCH
    if cat240_failed > 0:
        return _EXIT_RECIPE_TRAINER_STATE_INCONSISTENT
    return _EXIT_COMPLIANCE_ERROR


def _print_human_readable(verdict, args) -> None:
    print("=" * 72, file=sys.stderr)
    print(
        "Phase 6 Layer 4 Contest Compliance Verdict (canonical helper)",
        file=sys.stderr,
    )
    print("=" * 72, file=sys.stderr)
    print(f"lane_id:         {verdict.lane_id}", file=sys.stderr)
    print(f"substrate_id:    {verdict.substrate_id}", file=sys.stderr)
    print(
        f"archive_sha256:  {verdict.archive_sha256[:12]}... "
        f"({verdict.archive_bytes} bytes)",
        file=sys.stderr,
    )
    print(f"submission_dir:  {verdict.submission_dir}", file=sys.stderr)
    print(f"score axis:      {verdict.submission_score_axis}", file=sys.stderr)
    print(f"contest_final:   {verdict.contest_final_strict}", file=sys.stderr)
    print(f"elapsed:         {verdict.elapsed_seconds:.2f}s", file=sys.stderr)
    print("-" * 72, file=sys.stderr)
    print(
        f"VERDICT: {'CLEAN ✓' if verdict.overall_clean else 'BLOCKED ✗'}",
        file=sys.stderr,
    )
    print(
        f"checks:  {verdict.passed_count}/{verdict.total_checks} passed, "
        f"{verdict.error_count} error, {verdict.warning_count} warning",
        file=sys.stderr,
    )
    if verdict.forbidden_macos_axis_detected:
        print(
            "✗ STRUCTURAL REFUSAL: Catalog #192 macOS-CPU axis detected; "
            "re-run on Linux x86_64 1:1 contest-compliant hardware",
            file=sys.stderr,
        )
    if verdict.error_checks:
        print("-" * 72, file=sys.stderr)
        print("ERROR-SEVERITY BLOCKERS:", file=sys.stderr)
        for c in verdict.error_checks:
            op_marker = "[OP-GATED] " if c.is_operator_gated else "[STRUCT]   "
            gate_str = (
                f" (Catalog #{'/#'.join(str(g) for g in c.catalog_gate_refs)})"
                if c.catalog_gate_refs
                else ""
            )
            print(
                f"  {op_marker}{c.check_name}{gate_str}", file=sys.stderr
            )
            print(f"      details:    {c.details[:200]}", file=sys.stderr)
            print(f"      remediation: {c.remediation_hint}", file=sys.stderr)
    print("-" * 72, file=sys.stderr)
    print("Per-Catalog-gate failed-check counts:", file=sys.stderr)
    for gate, count in sorted(
        verdict.catalog_gate_protection_summary.items(),
        key=lambda kv: -kv[1],
    ):
        if count > 0:
            print(f"  Catalog #{gate}: {count}", file=sys.stderr)
    print(f"JSON report:     {verdict.json_report_path}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        bundle = _load_submission_bundle(args.from_submission_bundle)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(
            f"[submission-compliance-cli] argument error: {exc!r}",
            file=sys.stderr,
        )
        return _EXIT_CLI_ERROR
    try:
        verdict = enforce_contest_compliance(
            submission_bundle_result=bundle,
            contest_final_strict=args.contest_final_strict,
            submission_score_axis=args.submission_score_axis,
            expected_lane_id=args.expected_lane_id,
            expected_job_id=args.expected_job_id,
            auth_eval_json_path=args.auth_eval_json,
            contest_cpu_auth_eval_json_path=args.contest_cpu_auth_eval_json,
            archive_manifest_json_path=args.archive_manifest_json,
            runtime_equivalence_proof_json_path=args.runtime_equivalence_proof_json,
            hosted_archive_manifest_json_path=args.hosted_archive_manifest_json,
            public_source_ref_manifest_json_path=args.public_source_ref_manifest_json,
            competitive_or_innovative_statement=args.competitive_or_innovative_statement,
            output_dir=args.output_dir,
            subprocess_timeout_seconds=args.subprocess_timeout_seconds,
        )
    except SubmissionComplianceError as exc:
        print(
            f"[submission-compliance-cli] structural failure: {exc!r}",
            file=sys.stderr,
        )
        return _EXIT_CLI_ERROR
    except ValueError as exc:
        print(
            f"[submission-compliance-cli] caller-side argument error: {exc!r}",
            file=sys.stderr,
        )
        return _EXIT_CLI_ERROR
    if args.json:
        print(json.dumps(verdict.as_dict(), indent=2, sort_keys=True))
    else:
        _print_human_readable(verdict, args)
    return _classify_exit_code(verdict)


if __name__ == "__main__":
    raise SystemExit(main())
