#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for the canonical Phase 5 submission linter.

Per Phase 1 audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
Phase 5 acceptance + the operator-facing 4-layer canonical pattern
(canonical helper module + operator-facing CLI + cathedral consumer +
optional STRICT preflight gate).

Usage::

    # Lint a Phase 4 submission bundle from a JSON sidecar
    .venv/bin/python tools/submission_linter_cli.py \\
        --from-submission-bundle path/to/submission_bundle_result.json \\
        --target-repo commaai/comma_video_compression_challenge

    # Lint a PR body markdown file in isolation
    .venv/bin/python tools/submission_linter_cli.py \\
        --pr-body-only path/to/PR_BODY.md \\
        --target-repo commaai/comma_video_compression_challenge \\
        --json

Exit codes:
  0 LINT-CLEAN
  1 FORBIDDEN-TOKEN (Claude / Anthropic / Co-Authored / claude.com / anthropic.com)
  2 FIRST-PERSON-PLURAL (we / our / us)
  3 EMDASH (U+2014 forbidden)
  4 INFLATE-PY-OVER-BUDGET (HNeRV parity L4 violation)
  5 TONE-VIOLATION (marketing flourish / emoji / sign-off bromide / AI-tell)
  6 CLI ERROR (malformed args / missing files / parse failure)

The CLI delegates ALL parsing + validation to
:mod:`tac.submission_packet.linter`. This file is a thin operator-facing
wrapper sister of :mod:`tools.list_canonical_equations`,
:mod:`tools.check_predecessor_probe_outcome`, and other Phase-1-spec-memo
operator-facing helpers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from repo root via `python tools/...` without install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.submission_packet.builder import (  # noqa: E402
    SubmissionBundleResult,
    submission_bundle_result_from_dict,
)
from tac.submission_packet.linter import (  # noqa: E402
    LINTER_SCHEMA_VERSION,
    LintSeverity,
    LintVerdict,
    SubmissionLinterError,
    lint_pr_body,
    lint_submission_bundle,
)

# Canonical exit codes per CLI docstring above.
EXIT_LINT_CLEAN = 0
EXIT_FORBIDDEN_TOKEN = 1
EXIT_FIRST_PERSON_PLURAL = 2
EXIT_EMDASH = 3
EXIT_INFLATE_PY_OVER_BUDGET = 4
EXIT_TONE_VIOLATION = 5
EXIT_CLI_ERROR = 6


def _exit_code_from_verdict(verdict: LintVerdict) -> int:
    """Map a verdict to the canonical CLI exit code.

    Priority order (highest precedence first):
      1. forbidden_token_* → 1
      2. first_person_plural_* → 2
      3. emdash_u2014 → 3
      4. inflate_py_loc_over_budget → 4
      5. tone_* / emoji_* → 5
      6. any other ERROR finding → 1 (default forbidden-token bucket)
      0. clean → 0
    """
    if verdict.overall_clean:
        return EXIT_LINT_CLEAN
    for finding in verdict.findings:
        if finding.severity != LintSeverity.ERROR.value:
            continue
        if finding.rule.startswith("forbidden_token_"):
            return EXIT_FORBIDDEN_TOKEN
    for finding in verdict.findings:
        if finding.severity != LintSeverity.ERROR.value:
            continue
        if finding.rule.startswith("first_person_plural_"):
            return EXIT_FIRST_PERSON_PLURAL
    for finding in verdict.findings:
        if finding.severity != LintSeverity.ERROR.value:
            continue
        if finding.rule == "emdash_u2014":
            return EXIT_EMDASH
    for finding in verdict.findings:
        if finding.severity != LintSeverity.ERROR.value:
            continue
        if finding.rule == "inflate_py_loc_over_budget":
            return EXIT_INFLATE_PY_OVER_BUDGET
    for finding in verdict.findings:
        if finding.severity != LintSeverity.ERROR.value:
            continue
        if finding.rule.startswith("tone_") or finding.rule.startswith("emoji_"):
            return EXIT_TONE_VIOLATION
    # Any other ERROR finding (archive sha mismatch, missing file, etc.)
    # maps to FORBIDDEN-TOKEN bucket as canonical fallback.
    return EXIT_FORBIDDEN_TOKEN


def _exit_code_from_findings_only(findings) -> int:
    """Sister of _exit_code_from_verdict for pr-body-only mode.

    Operates on a flat tuple of findings (no LintVerdict wrapper).
    """
    errors = [f for f in findings if f.severity == LintSeverity.ERROR.value]
    if not errors:
        return EXIT_LINT_CLEAN
    for f in errors:
        if f.rule.startswith("forbidden_token_"):
            return EXIT_FORBIDDEN_TOKEN
    for f in errors:
        if f.rule.startswith("first_person_plural_"):
            return EXIT_FIRST_PERSON_PLURAL
    for f in errors:
        if f.rule == "emdash_u2014":
            return EXIT_EMDASH
    for f in errors:
        if f.rule == "inflate_py_loc_over_budget":
            return EXIT_INFLATE_PY_OVER_BUDGET
    for f in errors:
        if f.rule.startswith("tone_") or f.rule.startswith("emoji_"):
            return EXIT_TONE_VIOLATION
    return EXIT_FORBIDDEN_TOKEN


def _load_submission_bundle_from_json(path: Path) -> SubmissionBundleResult:
    """Reconstruct a SubmissionBundleResult from a persisted JSON sidecar.

    The Phase 4 builder's :meth:`SubmissionBundleResult.as_dict` is the
    canonical round-trip surface; this loader is its inverse for the
    operator-facing CLI surface.
    """
    if not path.exists():
        raise SubmissionLinterError(f"submission bundle JSON not found at {path}")
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SubmissionLinterError(
            f"submission bundle JSON at {path} is malformed: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SubmissionLinterError(
            f"submission bundle JSON at {path} must be a JSON object; got {type(payload).__name__}"
        )
    try:
        payload_for_parse = dict(payload)
        payload_for_parse.setdefault("vendor_pythonpath_self_containment", True)
        return submission_bundle_result_from_dict(
            payload_for_parse,
            require_schema_version=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SubmissionLinterError(
            f"submission bundle JSON at {path} failed canonical bundle parsing: {exc}"
        ) from exc


def _render_human_readable(verdict: LintVerdict) -> str:
    """Render a verdict as canonical operator-facing summary."""
    lines: list[str] = []
    lines.append(
        f"submission_linter_v={LINTER_SCHEMA_VERSION} target={verdict.target_repo}"
    )
    lines.append(
        f"overall_clean={verdict.overall_clean} "
        f"error={verdict.error_count} warn={verdict.warn_count} info={verdict.info_count}"
    )
    lines.append(f"surfaces_scanned={list(verdict.surfaces_scanned)}")
    if not verdict.findings:
        lines.append("(no findings)")
    else:
        for f in verdict.findings:
            location = f"{f.file_path}:{f.line_number}" if f.line_number else f.file_path
            lines.append(
                f"  [{f.severity}] {f.rule} @ {location} "
                + (f"matched={f.matched_text!r}" if f.matched_text else "")
            )
            lines.append(f"    fix: {f.fix_suggestion}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="submission_linter_cli",
        description=(
            "Canonical Phase 5 Layer 3 submission linter operator-facing CLI. "
            "Delegates to tac.submission_packet.lint_submission_bundle."
        ),
    )
    parser.add_argument(
        "--from-submission-bundle",
        type=Path,
        default=None,
        help=(
            "Path to a persisted SubmissionBundleResult JSON sidecar "
            "(canonical Phase 4 output)."
        ),
    )
    parser.add_argument(
        "--pr-body-only",
        type=Path,
        default=None,
        help=(
            "Path to a PR body markdown file; lint only the PR body "
            "(no Phase 4 bundle required)."
        ),
    )
    parser.add_argument(
        "--pr-body-path",
        type=Path,
        default=None,
        help=(
            "Path to a PR body markdown file to include alongside the Phase 4 "
            "bundle lint (canonical PR_BODY.md sister)."
        ),
    )
    parser.add_argument(
        "--target-repo",
        type=str,
        default="commaai/comma_video_compression_challenge",
        help="Upstream PR target repo (owner/repo).",
    )
    parser.add_argument(
        "--inflate-py-loc-waiver-rationale",
        type=str,
        default=None,
        help=(
            "Substantive waiver rationale (>=4 chars, non-placeholder per "
            "Catalog #287) when inflate.py exceeds the HNeRV parity L4 LOC budget."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON verdict to stdout.",
    )
    args = parser.parse_args(argv)

    if args.from_submission_bundle is None and args.pr_body_only is None:
        parser.error(
            "either --from-submission-bundle or --pr-body-only is required"
        )
    if args.from_submission_bundle is not None and args.pr_body_only is not None:
        parser.error(
            "--from-submission-bundle and --pr-body-only are mutually exclusive"
        )

    try:
        if args.pr_body_only is not None:
            if not args.pr_body_only.exists():
                print(
                    f"submission_linter_cli: PR body path not found: {args.pr_body_only}",
                    file=sys.stderr,
                )
                return EXIT_CLI_ERROR
            body_text = args.pr_body_only.read_text(encoding="utf-8")
            findings = lint_pr_body(
                body_text,
                target_repo=args.target_repo,
                file_path=str(args.pr_body_only),
            )
            if args.json:
                payload = {
                    "mode": "pr_body_only",
                    "target_repo": args.target_repo,
                    "findings": [f.as_dict() for f in findings],
                    "error_count": sum(
                        1 for f in findings if f.severity == LintSeverity.ERROR.value
                    ),
                    "warn_count": sum(
                        1 for f in findings if f.severity == LintSeverity.WARN.value
                    ),
                    "info_count": sum(
                        1 for f in findings if f.severity == LintSeverity.INFO.value
                    ),
                }
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                errors = sum(
                    1 for f in findings if f.severity == LintSeverity.ERROR.value
                )
                warns = sum(
                    1 for f in findings if f.severity == LintSeverity.WARN.value
                )
                infos = sum(
                    1 for f in findings if f.severity == LintSeverity.INFO.value
                )
                print(
                    f"submission_linter_cli pr_body_only target={args.target_repo} "
                    f"error={errors} warn={warns} info={infos}"
                )
                for f in findings:
                    location = (
                        f"{f.file_path}:{f.line_number}"
                        if f.line_number
                        else f.file_path
                    )
                    print(
                        f"  [{f.severity}] {f.rule} @ {location} "
                        + (f"matched={f.matched_text!r}" if f.matched_text else "")
                    )
                    print(f"    fix: {f.fix_suggestion}")
            return _exit_code_from_findings_only(findings)

        bundle = _load_submission_bundle_from_json(args.from_submission_bundle)
        verdict = lint_submission_bundle(
            bundle,
            target_repo=args.target_repo,
            pr_body_path=args.pr_body_path,
            inflate_py_loc_waiver_rationale=args.inflate_py_loc_waiver_rationale,
        )
    except SubmissionLinterError as exc:
        print(f"submission_linter_cli: {exc}", file=sys.stderr)
        return EXIT_CLI_ERROR
    except (KeyError, ValueError, OSError) as exc:
        print(f"submission_linter_cli: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CLI_ERROR

    if args.json:
        print(json.dumps(verdict.as_dict(), indent=2, sort_keys=True))
    else:
        print(_render_human_readable(verdict))
    return _exit_code_from_verdict(verdict)


if __name__ == "__main__":
    sys.exit(main())
