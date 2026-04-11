#!/usr/bin/env python3
"""Pre-commit hook: blocks commits if review coverage is below threshold.

Checks ONLY the files being committed (staged files), not the whole codebase.
If any staged .py file has entities below the review threshold, the commit
is blocked with a clear report of what needs review.

Install:
    ln -sf ../../tools/review_gate_hook.py .git/hooks/pre-commit
    chmod +x tools/review_gate_hook.py

Configuration (environment variables):
    REVIEW_GATE_ENABLED=1       Enable the gate (default: 1)
    REVIEW_GATE_THRESHOLD=0.0   Min reviewed fraction for staged files (0.0-1.0)
    REVIEW_GATE_STRICT=0        If 1, block on ANY unreviewed entity in staged files
    REVIEW_GATE_WARN_ONLY=1     If 1, warn but don't block (default: 1)

The default is warn-only mode — it prints the review status of staged files
but doesn't block the commit. Set REVIEW_GATE_WARN_ONLY=0 to enforce.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKER_DB = REPO_ROOT / ".omx" / "state" / "review_tracker.duckdb"


def get_staged_py_files() -> list[str]:
    """Get list of staged .py files (relative to repo root)."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        return []
    return [f for f in result.stdout.strip().split("\n") if f.endswith(".py") and f.strip()]


def check_review_status(staged_files: list[str]) -> tuple[int, int, int, int, list[str]]:
    """Check review status for entities in staged files.

    Returns: (total, reviewed, unreviewed, needs_fix, details)
    """
    try:
        import duckdb
    except ImportError:
        return 0, 0, 0, 0, ["duckdb not installed — skipping review gate"]

    if not TRACKER_DB.exists():
        return 0, 0, 0, 0, ["No review tracker DB — run: python tools/review_tracker.py scan"]

    con = duckdb.connect(str(TRACKER_DB), read_only=True)
    total = reviewed = unreviewed = needs_fix = 0
    details: list[str] = []

    for fp in staged_files:
        try:
            rows = con.execute("""
                SELECT name, entity_type, line_count, complexity, review_status
                FROM entities WHERE file_path = ?
                ORDER BY start_line
            """, [fp]).fetchall()
        except Exception:
            continue

        if not rows:
            continue

        file_total = len(rows)
        file_reviewed = sum(1 for r in rows if r[4] == "reviewed")
        file_unreviewed = sum(1 for r in rows if r[4] == "unreviewed")
        file_needs_fix = sum(1 for r in rows if r[4] == "needs_fix")
        file_stale = sum(1 for r in rows if r[4] == "stale")

        total += file_total
        reviewed += file_reviewed
        unreviewed += file_unreviewed + file_stale
        needs_fix += file_needs_fix

        cov = file_reviewed / file_total * 100 if file_total else 0
        if cov < 100:
            details.append(f"  {fp}: {file_reviewed}/{file_total} reviewed ({cov:.0f}%)")
            # List unreviewed entities
            for name, etype, lc, cx, st in rows:
                if st != "reviewed":
                    details.append(f"    [{st}] {etype} {name} ({lc}L, C={cx})")

    con.close()
    return total, reviewed, unreviewed, needs_fix, details


def main() -> int:
    enabled = os.environ.get("REVIEW_GATE_ENABLED", "1") == "1"
    if not enabled:
        return 0

    threshold = float(os.environ.get("REVIEW_GATE_THRESHOLD", "0.0"))
    strict = os.environ.get("REVIEW_GATE_STRICT", "0") == "1"
    warn_only = os.environ.get("REVIEW_GATE_WARN_ONLY", "1") == "1"

    staged = get_staged_py_files()
    if not staged:
        return 0

    # Filter to tracked directories only
    tracked_prefixes = ("src/tac/", "experiments/")
    tracked = [f for f in staged if any(f.startswith(p) for p in tracked_prefixes)]
    if not tracked:
        return 0

    total, reviewed, unreviewed, needs_fix, details = check_review_status(tracked)

    if total == 0:
        return 0

    coverage = reviewed / total if total > 0 else 0

    # Determine if we should block
    should_block = False
    if strict and unreviewed > 0:
        should_block = True
    elif coverage < threshold:
        should_block = True
    elif needs_fix > 0:
        should_block = True

    if not details:
        return 0

    # Print report
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    BOLD = "\033[1m"
    RST = "\033[0m"

    color = RED if should_block and not warn_only else YELLOW
    action = "BLOCKED" if should_block and not warn_only else "WARNING"

    print(f"\n{color}{BOLD}[review-gate] {action}: staged files have unreviewed code{RST}")
    print(f"  Coverage: {reviewed}/{total} ({coverage*100:.0f}%)")
    if needs_fix > 0:
        print(f"  {RED}Needs fix: {needs_fix} entities{RST}")
    print()
    for line in details:
        print(line)
    print()

    if should_block and not warn_only:
        print(f"{RED}Commit blocked. Review the above entities first:{RST}")
        print(f"  python tools/review_tracker.py mark-file <file> --status reviewed")
        print(f"  Or skip: REVIEW_GATE_ENABLED=0 git commit ...")
        print()
        return 1
    elif should_block:
        print(f"{YELLOW}To enforce: REVIEW_GATE_WARN_ONLY=0 REVIEW_GATE_STRICT=1{RST}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
