#!/usr/bin/env python3
# no-argparse-OK: git hook entrypoint — controlled via env vars (REVIEW_GATE_ENABLED/REVIEW_GATE_OVERRIDE)
"""Pre-commit hook: policy-enforced review gate.

Checks staged .py files against the review policy (review_policy.json):
- Consecutive clean passes (greenup protocol)
- Minimum approver level (capability-based, L1-L4)
- Distinct approver count
- needs_fix entities block commit

Install:
    ln -sf ../../tools/review_gate_hook.py .git/hooks/pre-commit
    chmod +x tools/review_gate_hook.py

Environment overrides:
    REVIEW_GATE_ENABLED=0    Disable entirely
    REVIEW_GATE_OVERRIDE=1   Override all policy checks (L4 equivalent)
    REVIEW_GATE_WARN_ONLY=1  Warn but don't block (default: 1)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKER_DB = REPO_ROOT / ".omx" / "state" / "review_tracker.duckdb"
POLICY_PATH = REPO_ROOT / ".omx" / "state" / "review_policy.json"


def get_staged_py_files() -> list[str]:
    """Get staged .py files (relative to repo root)."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        return []
    return [f for f in result.stdout.strip().split("\n") if f.endswith(".py") and f.strip()]


def load_policy() -> dict:
    if not POLICY_PATH.exists():
        return {}
    try:
        return json.loads(POLICY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def check_staged_files(staged_files: list[str]) -> tuple[list[str], list[str], dict]:
    """Check policy compliance for staged files.

    Returns: (blocking_issues, warnings, stats)
    """
    try:
        import duckdb
    except ImportError:
        return [], ["duckdb not installed — review gate skipped"], {}

    if not TRACKER_DB.exists():
        return [], ["No tracker DB — run: python tools/review_tracker.py scan"], {}

    policy = load_policy()
    con = duckdb.connect(str(TRACKER_DB), read_only=True)

    # Import policy functions from review_tracker
    _old_path = sys.path[:]
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import review_tracker as _rt
        check_entity_policy = _rt.check_entity_policy
        get_rigor_for_file = _rt.get_rigor_for_file
    except ImportError as _ie:
        sys.path[:] = _old_path
        print(f"WARNING: review_gate_hook: could not import review_tracker: {_ie}",
              file=sys.stderr)
        return [], ["review_tracker import failed — review gate skipped"], {}
    finally:
        sys.path[:] = _old_path

    blocking: list[str] = []
    warnings: list[str] = []
    stats = {"total": 0, "compliant": 0, "violations": 0, "needs_fix": 0}

    for fp in staged_files:
        try:
            rows = con.execute("""
                SELECT qualified_name, name, entity_type, line_count, complexity, review_status
                FROM entities WHERE file_path = ?
                ORDER BY start_line
            """, [fp]).fetchall()
        except Exception:
            continue

        if not rows:
            continue

        rigor = get_rigor_for_file(fp, policy)
        rigor_name = rigor.get("_name", "relaxed")

        file_violations = []
        for qn, name, etype, lc, cx, status in rows:
            stats["total"] += 1

            if status == "needs_fix":
                stats["needs_fix"] += 1
                file_violations.append(f"    [NEEDS_FIX] {etype} {name} ({lc}L, C={cx})")
                continue

            if status in ("unreviewed", "stale"):
                stats["violations"] += 1
                file_violations.append(f"    [{status.upper()}] {etype} {name} ({lc}L, C={cx})")
                continue

            # Entity is "reviewed" — check policy compliance
            result = check_entity_policy(con, qn, fp, policy)
            if result["met"]:
                stats["compliant"] += 1
            else:
                stats["violations"] += 1
                for issue in result["issues"]:
                    file_violations.append(f"    [POLICY] {name}: {issue}")

        if file_violations:
            header = f"  [{rigor_name.upper()}] {fp}"
            # Critical files block; others warn
            if rigor_name in ("critical", "standard"):
                blocking.append(header)
                blocking.extend(file_violations)
            else:
                warnings.append(header)
                warnings.extend(file_violations)

    con.close()
    return blocking, warnings, stats


def main() -> int:
    if os.environ.get("REVIEW_GATE_ENABLED", "1") == "0":
        return 0
    if os.environ.get("REVIEW_GATE_OVERRIDE", "0") == "1":
        return 0

    # Enforcement enabled by default — blocks commits on critical/standard files
    # with unreviewed code. Override: REVIEW_GATE_WARN_ONLY=1 git commit ...
    warn_only = os.environ.get("REVIEW_GATE_WARN_ONLY", "0") == "1"

    staged = get_staged_py_files()
    if not staged:
        return 0

    tracked_prefixes = ("src/tac/", "src/comma_lab/", "experiments/", "submissions/", "tools/")
    tracked = [f for f in staged if any(f.startswith(p) for p in tracked_prefixes)]
    if not tracked:
        return 0

    blocking, warnings, stats = check_staged_files(tracked)

    if not blocking and not warnings:
        return 0

    # ANSI colors
    RED = "\033[31m"; YELLOW = "\033[33m"; GREEN = "\033[32m"; CYAN = "\033[36m"
    BOLD = "\033[1m"; RST = "\033[0m"

    total = stats.get("total", 0)
    compliant = stats.get("compliant", 0)
    violations = stats.get("violations", 0)
    needs_fix = stats.get("needs_fix", 0)

    has_blocking = bool(blocking) and not warn_only
    action = "BLOCKED" if has_blocking else "WARNING"
    color = RED if has_blocking else YELLOW

    print(f"\n{color}{BOLD}[review-gate] {action}{RST}")
    print(f"  Entities: {total} checked, {GREEN}{compliant} compliant{RST}, "
          f"{RED}{violations} violations{RST}, {YELLOW}{needs_fix} needs_fix{RST}")
    print()

    if blocking:
        for line in blocking:
            print(f"{RED}{line}{RST}" if not line.startswith("    ") else line)
        print()

    if warnings:
        for line in warnings:
            print(f"{YELLOW}{line}{RST}" if not line.startswith("    ") else line)
        print()

    if has_blocking:
        print(f"{RED}{BOLD}Commit blocked by review policy.{RST}")
        print("  Fix issues:    python tools/review_tracker.py mark-file <file> --status reviewed")
        print("  Check policy:  python tools/review_tracker.py policy-check <file>")
        print("  Override:      REVIEW_GATE_OVERRIDE=1 git commit ...")
        print("  Disable:       REVIEW_GATE_ENABLED=0 git commit ...")
        print()
        return 1

    if warn_only and (blocking or warnings):
        print(f"{YELLOW}To enforce: REVIEW_GATE_WARN_ONLY=0{RST}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
