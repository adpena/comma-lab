#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit direct submission ``inflate.py`` files against a source LOC budget.

This is a review-surface guard. The contest score charges ``archive.zip`` bytes;
large ``inflate.py`` files are flagged because they are hard to review and tend
to hide runtime-closure bugs, not because trimming Python source lowers score.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.submission_inflate_loc_budget import (  # noqa: E402
    DEFAULT_MAX_INFLATE_PY_LINES,
    DEFAULT_REVIEW_TARGET_INFLATE_PY_LINES,
    scan_submission_inflate_py_loc_budget,
)


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_INFLATE_PY_LINES)
    parser.add_argument(
        "--review-target-lines",
        type=int,
        default=DEFAULT_REVIEW_TARGET_INFLATE_PY_LINES,
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable findings")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Emit the default human-readable summary explicitly",
    )
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    findings = scan_submission_inflate_py_loc_budget(
        root,
        max_lines=args.max_lines,
        review_target_lines=args.review_target_lines,
    )
    hard_count = sum(1 for finding in findings if finding.budget_tier == "hard_budget")
    default_count = sum(1 for finding in findings if finding.budget_tier == "default_budget")
    if args.json:
        print(
            json.dumps(
                {
                    "schema": "submission_inflate_py_loc_budget_audit_v1",
                    "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "git_head": _git_head(root),
                    "repo_root": str(root),
                    "max_lines": args.max_lines,
                    "review_target_lines": args.review_target_lines,
                    "finding_count": len(findings),
                    "hard_budget_violation_count": hard_count,
                    "default_budget_warning_count": default_count,
                    "findings": [
                        {
                            "budget_tier": f.budget_tier,
                            "rel_path": f.rel_path,
                            "line_count": f.line_count,
                            "max_lines": f.max_lines,
                            "review_target_lines": f.review_target_lines,
                            "severity": f.severity,
                            "shared_runtime_helper_adopted": f.shared_runtime_helper_adopted,
                            "size_driver_categories": list(f.size_driver_categories),
                            "technique_applicability": list(f.technique_applicability),
                        }
                        for f in findings
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif findings:
        print(
            f"[inflate-py-loc-budget] {len(findings)} file(s) exceed "
            f"review target {args.review_target_lines} physical lines "
            f"({hard_count} exceed hard max {args.max_lines}):",
            file=sys.stderr,
        )
        for finding in findings[:40]:
            print(f"  - {finding.format()}", file=sys.stderr)
        if len(findings) > 40:
            print(f"  ... (+{len(findings) - 40} more)", file=sys.stderr)
    else:
        print(
            f"[inflate-py-loc-budget] OK: all direct submissions/inflate.py "
            f"files are <= {args.max_lines} physical lines"
        )

    if findings and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
