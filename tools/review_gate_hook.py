#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
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
    REVIEW_GATE_ENABLED=0           Disable entirely (canonical infra-failure
                                    workaround; should rarely be needed after
                                    lane_duckdb_lock_fix_review_gate_hook_20260514).
    REVIEW_GATE_OVERRIDE=1          Override all policy checks (L4 equivalent).
    REVIEW_GATE_WARN_ONLY=1         Warn but don't block.
    REVIEW_GATE_HOOK_RETRY_SECONDS  Override the hook-specific DuckDB lock
                                    retry budget (default: 1.5s). The hook
                                    uses a TIGHT deadline (vs the 8s
                                    review_tracker default) so concurrent
                                    sister-subagent writers do not stall
                                    the operator's git commit. On retry
                                    exhaustion the hook falls through to
                                    the JSON snapshot (read-only, lock-free).

Lock-contention behavior (lane
``lane_duckdb_lock_fix_review_gate_hook_20260514``, 2026-05-14):
- Empirical anchors: CATALOG-226-REFACTOR observed 84.9s commit stall;
  F3-GTSCORERCACHE observed 5+ min / 10-retry failures forcing
  ``REVIEW_GATE_ENABLED=0`` workaround.
- Mechanism: a sister subagent holds the DuckDB file lock via
  ``review_tracker.py mark-file --status reviewed`` (RW); the hook's
  ``duckdb.connect(read_only=True)`` cross-process call raises
  ``IOException: Could not set lock``.
- Fix: route the hook through the canonical
  ``review_tracker._connect_duckdb`` retry helper with a
  ``retry_seconds=1.5`` budget. On exhaustion, fall through to
  :func:`review_tracker.load_entities_from_json_snapshot` for entity-status
  enforcement (the strongest signal: needs_fix / unreviewed / stale). For
  critical/standard files, the missing full policy evidence itself blocks:
  the JSON snapshot does not contain the ``reviews`` table, so it cannot prove
  consecutive clean passes or distinct approvers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKER_DB = REPO_ROOT / ".omx" / "state" / "review_tracker.duckdb"
TRACKER_JSON = REPO_ROOT / ".omx" / "state" / "review_tracker.json"
POLICY_PATH = REPO_ROOT / ".omx" / "state" / "review_policy.json"

# Hook-specific lock retry budget: short by design. The hook is on the operator's
# critical path (every `git commit` waits for it). Sister subagent writers can
# legitimately hold the DuckDB RW lock for seconds at a time. Rather than block
# the commit for ~8s (the review_tracker default), we wait ~1.5s and then fall
# through to the JSON snapshot. Override via REVIEW_GATE_HOOK_RETRY_SECONDS.
DEFAULT_HOOK_RETRY_SECONDS = 1.5


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


def _import_review_tracker():
    """Import the review_tracker module. Returns the module or None on failure."""
    _old_path = sys.path[:]
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import review_tracker as _rt
        return _rt
    except ImportError as _ie:
        print(
            f"WARNING: review_gate_hook: could not import review_tracker: {_ie}",
            file=sys.stderr,
        )
        return None
    finally:
        sys.path[:] = _old_path


def _resolve_hook_retry_seconds() -> float:
    """Read the hook-specific retry budget (env override or default)."""
    raw = os.environ.get("REVIEW_GATE_HOOK_RETRY_SECONDS", "")
    if not raw:
        return DEFAULT_HOOK_RETRY_SECONDS
    try:
        val = float(raw)
        if val < 0:
            return DEFAULT_HOOK_RETRY_SECONDS
        return val
    except ValueError:
        return DEFAULT_HOOK_RETRY_SECONDS


def _try_connect_tracker_db(rt_module, retry_seconds: float):
    """Open the DuckDB tracker read-only with the canonical retry helper.

    Returns: (connection, error_or_None). On lock-exhaustion the caller falls
    back to JSON snapshot. Any other exception is logged + caller falls back.
    """
    try:
        con = rt_module._connect_duckdb(
            TRACKER_DB, read_only=True, retry_seconds=retry_seconds
        )
        return con, None
    except Exception as exc:
        return None, exc


def _check_staged_via_duckdb(
    rt_module,
    con,
    staged_files: list[str],
    policy: dict,
) -> tuple[list[str], list[str], dict]:
    """Full policy check served from the DuckDB connection (canonical path)."""
    blocking: list[str] = []
    warnings: list[str] = []
    stats = {"total": 0, "compliant": 0, "violations": 0, "needs_fix": 0, "source": "duckdb"}

    check_entity_policy = rt_module.check_entity_policy
    get_rigor_for_file = rt_module.get_rigor_for_file

    for fp in staged_files:
        try:
            rows = con.execute(
                """
                SELECT qualified_name, name, entity_type, line_count, complexity, review_status
                FROM entities WHERE file_path = ?
                ORDER BY start_line
                """,
                [fp],
            ).fetchall()
        except Exception:
            continue

        if not rows:
            continue

        rigor = get_rigor_for_file(fp, policy)
        rigor_name = rigor.get("_name", "relaxed")

        file_violations: list[str] = []
        for qn, name, etype, lc, cx, status in rows:
            stats["total"] += 1

            if status == "needs_fix":
                stats["needs_fix"] += 1
                file_violations.append(f"    [NEEDS_FIX] {etype} {name} ({lc}L, C={cx})")
                continue

            if status in ("unreviewed", "stale"):
                stats["violations"] += 1
                file_violations.append(
                    f"    [{status.upper()}] {etype} {name} ({lc}L, C={cx})"
                )
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
            if rigor_name in ("critical", "standard"):
                blocking.append(header)
                blocking.extend(file_violations)
            else:
                warnings.append(header)
                warnings.extend(file_violations)

    return blocking, warnings, stats


def _check_staged_via_json_snapshot(
    rt_module,
    snapshot: dict,
    staged_files: list[str],
    policy: dict,
) -> tuple[list[str], list[str], dict]:
    """Degraded entity-status check served from the JSON snapshot.

    Used when the DuckDB lock cannot be acquired inside the hook's retry
    budget. Enforces the strongest signal (needs_fix / unreviewed / stale)
    and fails closed for critical/standard files whose reviewed entities cannot
    be proven against the full DuckDB policy.

    NOTE: this preserves the short hook wait while avoiding a false green.
    If DuckDB is locked, critical/standard code waits for a real policy check;
    lower-rigor files get an explicit warning rather than pretending the JSON
    snapshot proved the full review contract.
    """
    blocking: list[str] = []
    warnings: list[str] = []
    stats = {"total": 0, "compliant": 0, "violations": 0, "needs_fix": 0, "source": "json"}

    entities = snapshot.get("entities", {}) if isinstance(snapshot, dict) else {}
    get_rigor_for_file = rt_module.get_rigor_for_file

    # Index entities by file_path for O(1) lookup.
    by_file: dict[str, list[dict]] = {}
    for qn, ent in entities.items():
        if not isinstance(ent, dict):
            continue
        ent_with_qn = dict(ent)
        ent_with_qn["qualified_name"] = qn
        by_file.setdefault(ent.get("file_path", ""), []).append(ent_with_qn)

    # Stable iteration order: by start_line within each file.
    for ent_list in by_file.values():
        ent_list.sort(key=lambda e: e.get("start_line", 0))

    for fp in staged_files:
        ent_list = by_file.get(fp, [])
        if not ent_list:
            continue
        rigor = get_rigor_for_file(fp, policy)
        rigor_name = rigor.get("_name", "relaxed")

        file_violations: list[str] = []
        file_warnings: list[str] = []
        for ent in ent_list:
            stats["total"] += 1
            status = ent.get("review_status", "unreviewed")
            etype = ent.get("entity_type", "")
            name = ent.get("name", "")
            lc = ent.get("line_count", 0)
            cx = ent.get("complexity", 1)

            if status == "needs_fix":
                stats["needs_fix"] += 1
                file_violations.append(f"    [NEEDS_FIX] {etype} {name} ({lc}L, C={cx})")
                continue

            if status in ("unreviewed", "stale"):
                stats["violations"] += 1
                file_violations.append(
                    f"    [{status.upper()}] {etype} {name} ({lc}L, C={cx})"
                )
                continue

            # status == "reviewed": JSON proves only entity status. It cannot
            # prove consecutive clean passes or distinct approvers because the
            # snapshot omits the reviews table.
            if rigor_name in ("critical", "standard"):
                stats["violations"] += 1
                file_violations.append(
                    "    [POLICY_UNPROVEN_JSON_FALLBACK] "
                    f"{etype} {name} ({lc}L, C={cx})"
                )
            else:
                stats["compliant"] += 1
                file_warnings.append(
                    "    [POLICY_UNPROVEN_JSON_FALLBACK] "
                    f"{etype} {name} ({lc}L, C={cx})"
                )

        if file_violations:
            header = f"  [{rigor_name.upper()}] {fp} (JSON-snapshot mode)"
            if rigor_name in ("critical", "standard"):
                blocking.append(header)
                blocking.extend(file_violations)
            else:
                warnings.append(header)
                warnings.extend(file_violations)
        if file_warnings:
            header = f"  [{rigor_name.upper()}] {fp} (JSON-snapshot mode)"
            warnings.append(header)
            warnings.extend(file_warnings)

    return blocking, warnings, stats


def check_staged_files(staged_files: list[str]) -> tuple[list[str], list[str], dict]:
    """Check policy compliance for staged files.

    Returns: (blocking_issues, warnings, stats). On DuckDB lock contention,
    falls back to the JSON snapshot for degraded entity-status enforcement.
    """
    try:
        import duckdb  # noqa: F401 — only used to detect availability
    except ImportError:
        return [], ["duckdb not installed — review gate skipped"], {"source": "none"}

    rt_module = _import_review_tracker()
    if rt_module is None:
        return [], ["review_tracker import failed — review gate skipped"], {"source": "none"}

    policy = load_policy()
    retry_seconds = _resolve_hook_retry_seconds()

    # Try DuckDB first (canonical path).
    con, db_err = (None, None)
    if TRACKER_DB.exists():
        con, db_err = _try_connect_tracker_db(rt_module, retry_seconds=retry_seconds)

    if con is not None:
        try:
            return _check_staged_via_duckdb(rt_module, con, staged_files, policy)
        finally:
            try:
                con.close()
            except Exception:
                pass

    # DuckDB unavailable — fall back to JSON snapshot.
    snapshot = rt_module.load_entities_from_json_snapshot(TRACKER_JSON)
    if snapshot is None:
        # Neither DB nor JSON available. This is the only path where we have
        # zero state — skip the gate rather than block on missing infra.
        if not TRACKER_DB.exists():
            return [], [
                "No tracker DB — run: python tools/review_tracker.py scan",
            ], {"source": "none"}
        # DB exists but couldn't be opened AND no JSON snapshot.
        return [], [
            "review_tracker DB locked + no JSON snapshot — review gate degraded",
            f"  (last DB error: {db_err})",
        ], {"source": "none"}

    blocking, warnings, stats = _check_staged_via_json_snapshot(
        rt_module, snapshot, staged_files, policy
    )
    # Prepend a one-line banner explaining the degraded mode.
    warnings.insert(
        0,
        "[review-gate] DuckDB locked by sister process — using JSON snapshot "
        f"(retry budget: {retry_seconds:.1f}s). Critical/standard files block "
        "until full DuckDB policy evidence is available.",
    )
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
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    BOLD = "\033[1m"
    RST = "\033[0m"

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
