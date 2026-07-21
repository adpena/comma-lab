#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compatibility CLI for the retired submission ``inflate.py`` LOC audit.

The operator permanently removed the line-count restriction on 2026-07-21.
This command remains callable for old automation, always reports the retired
status, and always exits zero. Anti-fake enforcement remains in #417 and the
payload-cleanliness audit bundle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
    if args.json:
        print(
            json.dumps(
                {
                    "schema": "submission_inflate_py_loc_budget_audit_v1",
                    "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "git_head": _git_head(root),
                    "repo_root": str(root),
                    "restriction_status": "permanently_removed_2026-07-21",
                    "informational_only": True,
                    "max_lines": args.max_lines,
                    "review_target_lines": args.review_target_lines,
                    "finding_count": len(findings),
                    "hard_budget_violation_count": 0,
                    "default_budget_warning_count": 0,
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
    else:
        print(
            "[inflate-py-loc-budget] RETIRED: source length is informational "
            "and unrestricted (operator 2026-07-21)"
        )

    _ = args.strict
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
