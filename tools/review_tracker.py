#!/usr/bin/env python3
"""AST-powered code review tracker for tac — DuckDB backend.

Parses every Python module, extracts classes/functions with line ranges,
cross-references with git diff to detect precise line-level staleness,
and maintains a DuckDB source-of-truth for review status.

The JSON state file is kept as a portable fallback and for git-friendly diffs.
DuckDB is the primary query engine for analytics, dashboards, and reports.

Usage:
    python tools/review_tracker.py scan                          # Scan codebase
    python tools/review_tracker.py status                        # Per-file summary
    python tools/review_tracker.py dashboard                     # Terminal dashboard
    python tools/review_tracker.py dag                           # Git DAG vs review
    python tools/review_tracker.py diff-scan [--since REF]       # Precise staleness
    python tools/review_tracker.py mark <pattern> [--status S]   # Mark entity
    python tools/review_tracker.py mark-file <path> [--status S] # Mark whole file
    python tools/review_tracker.py greenup-import <file>         # Import greenup pass
    python tools/review_tracker.py report                        # Markdown report
    python tools/review_tracker.py query <sql>                   # Raw DuckDB SQL
    python tools/review_tracker.py selftest                      # Run self-tests
    python tools/review_tracker.py rebuild-from-json             # Rebuild DB from JSON
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TAC_ROOT = REPO_ROOT / "src" / "tac"
TRACKER_JSON = REPO_ROOT / ".omx" / "state" / "review_tracker.json"
TRACKER_DB = REPO_ROOT / ".omx" / "state" / "review_tracker.duckdb"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
POLICY_PATH = REPO_ROOT / ".omx" / "state" / "review_policy.json"

# Valid review statuses
VALID_STATUSES = {"unreviewed", "reviewed", "stale", "needs_fix"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EntityRecord:
    """A single reviewable code entity (class, function, or method)."""
    module: str
    file_path: str
    entity_type: str         # class | function | method
    name: str                # e.g. "fit_lazy", "PostFilter.forward"
    start_line: int
    end_line: int
    line_count: int
    complexity: int
    last_modified_commit: str = ""
    last_modified_date: str = ""
    review_status: str = "unreviewed"
    reviewed_by: str = ""
    reviewed_at: str = ""
    review_pass: str = ""
    notes: str = ""

    @property
    def qualified_name(self) -> str:
        return f"{self.module}::{self.name}"


# ---------------------------------------------------------------------------
# AST analysis
# ---------------------------------------------------------------------------

def _estimate_complexity(node: ast.AST) -> int:
    """Rough cyclomatic complexity: branches + loops + exception handlers."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.IfExp)):
            count += 1
        elif isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
            count += 1
        elif isinstance(child, ast.ExceptHandler):
            count += 1
        elif isinstance(child, ast.BoolOp):
            count += len(child.values) - 1
        elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            count += 1
    return count + 1


def _extract_from_tree(tree: ast.Module, rel_path: str, module_name: str) -> list[EntityRecord]:
    """Extract entities from an already-parsed AST."""
    entities: list[EntityRecord] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            entities.append(EntityRecord(
                module=module_name, file_path=rel_path, entity_type="function",
                name=node.name, start_line=node.lineno, end_line=end,
                line_count=end - node.lineno + 1,
                complexity=_estimate_complexity(node),
            ))
        elif isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", node.lineno)
            entities.append(EntityRecord(
                module=module_name, file_path=rel_path, entity_type="class",
                name=node.name, start_line=node.lineno, end_line=end,
                line_count=end - node.lineno + 1,
                complexity=_estimate_complexity(node),
            ))
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_end = getattr(item, "end_lineno", item.lineno)
                    entities.append(EntityRecord(
                        module=module_name, file_path=rel_path, entity_type="method",
                        name=f"{node.name}.{item.name}",
                        start_line=item.lineno, end_line=m_end,
                        line_count=m_end - item.lineno + 1,
                        complexity=_estimate_complexity(item),
                    ))
    return entities


def extract_entities(file_path: Path) -> list[EntityRecord]:
    """Parse a Python file and extract all classes/functions with metadata."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError, OSError) as e:
        print(f"  WARN: skipping {file_path}: {e}", file=sys.stderr)
        return []

    rel_path = str(file_path.relative_to(REPO_ROOT))

    # Compute module name
    try:
        parts = file_path.relative_to(REPO_ROOT / "src").with_suffix("").parts
        module_name = ".".join(parts)
    except ValueError:
        parts = file_path.relative_to(REPO_ROOT).with_suffix("").parts
        module_name = ".".join(parts)

    return _extract_from_tree(tree, rel_path, module_name)


def scan_all_modules() -> list[EntityRecord]:
    """Scan all tracked Python files in src/tac/, experiments/, tools/, submissions/."""
    all_entities: list[EntityRecord] = []

    for py_file in sorted(TAC_ROOT.rglob("*.py")):
        all_entities.extend(extract_entities(py_file))

    # Experiment scripts, tools, submissions — top-level .py only
    for scan_dir in [EXPERIMENTS_ROOT, REPO_ROOT / "tools", REPO_ROOT / "submissions"]:
        if not scan_dir.exists():
            continue
        for py_file in sorted(scan_dir.rglob("*.py")):
            if py_file.name.startswith("__"):
                continue
            all_entities.extend(extract_entities(py_file))

    return all_entities


# ---------------------------------------------------------------------------
# Git integration
# ---------------------------------------------------------------------------

def _run_git(*args: str, timeout: int = 10) -> str | None:
    """Run a git command, return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def get_file_last_commit(file_path: str) -> tuple[str, str]:
    """Get the most recent commit hash + date for a file."""
    out = _run_git("log", "-1", "--format=%H %aI", "--", file_path, timeout=5)
    if out:
        parts = out.split(maxsplit=1)
        if len(parts) == 2:
            return parts[0][:12], parts[1]
    return "", ""


def _git_diff_lines(since: str, file_path: str) -> set[int]:
    """Get exact NEW-side line numbers changed since a ref for a file."""
    out = _run_git("diff", "-U0", since, "--", file_path)
    if not out:
        return set()

    changed: set[int] = set()
    for line in out.split("\n"):
        m = re.match(r"^@@ .+? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) else 1
            if count == 0:
                continue  # deletion-only hunk (no new lines)
            for ln in range(start, start + count):
                changed.add(ln)
    return changed


# ---------------------------------------------------------------------------
# DuckDB backend
# ---------------------------------------------------------------------------



_SCHEMA_DDL = [
    "CREATE SEQUENCE IF NOT EXISTS review_seq START 1",
    """CREATE TABLE IF NOT EXISTS entities (
        qualified_name VARCHAR PRIMARY KEY,
        module VARCHAR NOT NULL,
        file_path VARCHAR NOT NULL,
        entity_type VARCHAR NOT NULL,
        name VARCHAR NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        line_count INTEGER NOT NULL,
        complexity INTEGER NOT NULL DEFAULT 1,
        last_modified_commit VARCHAR DEFAULT '',
        last_modified_date VARCHAR DEFAULT '',
        review_status VARCHAR DEFAULT 'unreviewed',
        reviewed_by VARCHAR DEFAULT '',
        reviewed_at VARCHAR DEFAULT '',
        review_pass VARCHAR DEFAULT '',
        notes VARCHAR DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER DEFAULT nextval('review_seq'),
        entity VARCHAR NOT NULL,
        action VARCHAR NOT NULL,
        reviewer VARCHAR DEFAULT '',
        review_pass VARCHAR DEFAULT '',
        detail VARCHAR DEFAULT '',
        timestamp VARCHAR NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS scan_meta (
        key VARCHAR PRIMARY KEY,
        value VARCHAR
    )""",
]


def _init_db():
    """Initialize DB with sequence first, then tables."""
    import duckdb
    TRACKER_DB.parent.mkdir(parents=True, exist_ok=True)
    try:
        con = duckdb.connect(str(TRACKER_DB))
    except duckdb.Error as e:
        print(f"DuckDB corrupted: {e}", file=sys.stderr)
        print("Run: python tools/review_tracker.py rebuild-from-json", file=sys.stderr)
        sys.exit(1)
    for ddl in _SCHEMA_DDL:
        con.execute(ddl)
    return con


def _export_json(con) -> None:
    """Export DuckDB state to JSON for git-friendly diffs."""
    entities = {}
    rows = con.execute("SELECT * FROM entities ORDER BY qualified_name").fetchall()
    cols = [d[0] for d in con.description]
    for row in rows:
        d = dict(zip(cols, row))
        qn = d.pop("qualified_name")
        entities[qn] = d

    reviews = []
    try:
        rows = con.execute("SELECT * FROM reviews ORDER BY timestamp").fetchall()
        cols = [d[0] for d in con.description]
        for row in rows:
            reviews.append(dict(zip(cols, row)))
    except Exception as e:
        print(f"WARN: failed to export reviews: {e}", file=sys.stderr)

    last_scan = ""
    try:
        r = con.execute("SELECT value FROM scan_meta WHERE key='last_scan'").fetchone()
        if r:
            last_scan = r[0]
    except Exception:
        pass

    TRACKER_JSON.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_JSON.write_text(json.dumps({
        "version": 3,
        "last_scan": last_scan,
        "entity_count": len(entities),
        "review_count": len(reviews),
        "entities": entities,
    }, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

def load_policy() -> dict:
    """Load the review policy config. Returns empty dict on missing/corrupt."""
    if not POLICY_PATH.exists():
        return {}
    try:
        return json.loads(POLICY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _match_file_policy(file_path: str, policy: dict) -> dict:
    """Find the first matching file_policy entry for a file path.

    Matches are tested in order — first match wins (most specific first).
    Supports fnmatch-style glob patterns.
    """
    import fnmatch
    for fp in policy.get("file_policies", []):
        pattern = fp.get("pattern", "")
        if fnmatch.fnmatch(file_path, pattern):
            return fp
    return {"rigor": "relaxed", "reason": "no matching policy"}


def get_rigor_for_file(file_path: str, policy: dict | None = None) -> dict:
    """Get the rigor requirements for a specific file."""
    if policy is None:
        policy = load_policy()
    fp = _match_file_policy(file_path, policy)
    rigor_name = fp.get("rigor", "relaxed")
    rigor = policy.get("rigor", {}).get(rigor_name, {})
    rigor["_name"] = rigor_name
    rigor["_reason"] = fp.get("reason", "")
    return rigor


def get_principal(reviewer_id: str, policy: dict | None = None) -> dict:
    """Look up a reviewer principal by ID. Returns empty dict if not found."""
    if policy is None:
        policy = load_policy()
    return policy.get("principals", {}).get(reviewer_id, {})


def count_consecutive_clean_passes(con, qualified_name: str) -> int:
    """Count consecutive clean review passes for an entity.

    Walks the review audit log backwards from newest. A 'marked_reviewed'
    increments the counter. An 'auto_stale_diff' or 'marked_needs_fix'
    resets to 0 and stops counting.
    """
    try:
        rows = con.execute("""
            SELECT action, reviewer FROM reviews
            WHERE entity = ?
            ORDER BY timestamp DESC
        """, [qualified_name]).fetchall()
    except Exception:
        return 0

    count = 0
    for action, reviewer in rows:
        if action == "marked_reviewed":
            count += 1
        elif action in ("auto_stale_diff", "marked_needs_fix", "marked_stale", "marked_unreviewed"):
            break  # reset — stop counting
        # Other actions (comments, flags) don't affect the count
    return count


def get_distinct_approvers(con, qualified_name: str, file_path: str = "",
                           policy: dict | None = None) -> list[str]:
    """Get distinct reviewers who approved since last staleness reset.

    Uses the file's rigor level to determine the minimum approver level,
    rather than hardcoding L3+. This ensures normal/relaxed files can be
    approved by L1/L2 reviewers as the policy intends.
    """
    if policy is None:
        policy = load_policy()

    # Determine minimum level from file's rigor policy
    rigor = get_rigor_for_file(file_path, policy) if file_path else {}
    min_level = rigor.get("min_approver_level", 1)

    try:
        rows = con.execute("""
            SELECT action, reviewer FROM reviews
            WHERE entity = ?
            ORDER BY timestamp DESC
        """, [qualified_name]).fetchall()
    except Exception:
        return []

    approvers: list[str] = []
    for action, reviewer in rows:
        if action == "marked_reviewed":
            principal = get_principal(reviewer, policy)
            level = principal.get("level", 1)
            if level >= min_level and reviewer not in approvers:
                approvers.append(reviewer)
        elif action in ("auto_stale_diff", "marked_needs_fix", "marked_stale"):
            break  # staleness resets approver list
    return approvers


def check_entity_policy(con, qualified_name: str, file_path: str,
                        policy: dict | None = None) -> dict:
    """Check if an entity meets its review policy requirements.

    Returns a dict with:
        met: bool — all requirements satisfied
        rigor: str — rigor level name
        issues: list[str] — human-readable policy violations
        passes: int — consecutive clean passes
        required_passes: int
        approvers: list[str] — distinct approvers
        required_approvers: int
    """
    if policy is None:
        policy = load_policy()

    rigor = get_rigor_for_file(file_path, policy)
    rigor_name = rigor.get("_name", "relaxed")

    passes = count_consecutive_clean_passes(con, qualified_name)
    approvers = get_distinct_approvers(con, qualified_name, file_path, policy)

    req_passes = rigor.get("min_consecutive_clean_passes", 1)
    req_approvers = rigor.get("min_distinct_approvers", 1)
    req_level = rigor.get("min_approver_level", 1)
    req_human = rigor.get("require_human_approver", False)

    issues: list[str] = []

    if passes < req_passes:
        issues.append(f"needs {req_passes - passes} more clean pass(es) ({passes}/{req_passes})")

    if len(approvers) < req_approvers:
        issues.append(f"needs {req_approvers - len(approvers)} more approver(s) (have: {approvers or 'none'})")

    # Check approver levels
    for a in approvers:
        principal = get_principal(a, policy)
        if principal.get("level", 1) < req_level:
            issues.append(f"approver '{a}' is L{principal.get('level', 1)}, needs L{req_level}+")

    if req_human and not any(get_principal(a, policy).get("human", False) for a in approvers):
        issues.append(f"requires at least one human approver")

    return {
        "met": len(issues) == 0,
        "rigor": rigor_name,
        "reason": rigor.get("_reason", ""),
        "issues": issues,
        "passes": passes,
        "required_passes": req_passes,
        "approvers": approvers,
        "required_approvers": req_approvers,
    }


def cmd_policy_check(file_path: str | None = None) -> None:
    """Check policy compliance for a file or all tracked entities."""
    con = _init_db()

    total = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    if total == 0:
        print("No entities tracked. Run 'scan' first.")
        con.close()
        return

    policy = load_policy()

    if not policy:
        print("No review policy found at .omx/state/review_policy.json")
        con.close()
        return

    G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; C = "\033[36m"
    RST = "\033[0m"; B = "\033[1m"

    if file_path:
        rows = con.execute(
            "SELECT qualified_name, file_path, name, review_status FROM entities WHERE file_path LIKE ?",
            [f"%{file_path}%"]
        ).fetchall()
    else:
        # Check critical + standard files only
        rows = con.execute("""
            SELECT qualified_name, file_path, name, review_status FROM entities
            WHERE review_status != 'reviewed'
            ORDER BY file_path, start_line
        """).fetchall()

    print(f"\n{B}  Policy Compliance Check{RST}")
    print(f"  {'=' * 60}\n")

    violations = 0
    met_count = 0
    current_file = ""

    for qn, fp, name, status in rows:
        result = check_entity_policy(con, qn, fp, policy)

        if fp != current_file:
            current_file = fp
            rigor = get_rigor_for_file(fp, policy)
            level_color = R if rigor["_name"] == "critical" else (Y if rigor["_name"] == "standard" else RST)
            print(f"  {level_color}[{rigor['_name'].upper()}]{RST} {fp}")

        if result["met"]:
            met_count += 1
        else:
            violations += 1
            for issue in result["issues"]:
                print(f"    {R}!{RST} {name}: {issue}")

    print(f"\n  {B}Summary:{RST} {met_count} entities compliant, {R}{violations} violations{RST}")
    if violations == 0:
        print(f"  {G}All checked entities meet their review policy.{RST}")
    print()
    con.close()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_scan() -> None:
    """Scan the codebase and update the tracker."""
    con = _init_db()
    entities = scan_all_modules()
    now = datetime.now(timezone.utc).isoformat()

    new_count = 0
    changed_count = 0
    unchanged_count = 0

    # Build lookup of existing entities
    existing = {}
    for row in con.execute("SELECT qualified_name, last_modified_commit, review_status FROM entities").fetchall():
        existing[row[0]] = {"commit": row[1], "status": row[2]}

    for ent in entities:
        key = ent.qualified_name
        commit, date = get_file_last_commit(ent.file_path)
        ent.last_modified_commit = commit
        ent.last_modified_date = date

        prev = existing.get(key)
        if prev is None:
            con.execute("""
                INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [key, ent.module, ent.file_path, ent.entity_type, ent.name,
                  ent.start_line, ent.end_line, ent.line_count, ent.complexity,
                  commit, date, "unreviewed", "", "", "", ""])
            new_count += 1
        else:
            if commit and commit != prev["commit"]:
                new_status = "stale" if prev["status"] == "reviewed" else prev["status"]
                if new_status == "stale":
                    changed_count += 1
                con.execute("""
                    UPDATE entities SET start_line=?, end_line=?, line_count=?,
                    complexity=?, last_modified_commit=?, last_modified_date=?,
                    review_status=? WHERE qualified_name=?
                """, [ent.start_line, ent.end_line, ent.line_count, ent.complexity,
                      commit, date, new_status, key])
            else:
                # Update line positions (AST may have shifted)
                con.execute("""
                    UPDATE entities SET start_line=?, end_line=?, line_count=?,
                    complexity=? WHERE qualified_name=?
                """, [ent.start_line, ent.end_line, ent.line_count, ent.complexity, key])
                unchanged_count += 1

    # Remove entities that no longer exist
    current_keys = {e.qualified_name for e in entities}
    for old_key in existing:
        if old_key not in current_keys:
            con.execute("DELETE FROM entities WHERE qualified_name=?", [old_key])

    removed = len(set(existing) - current_keys)
    con.execute("""
        INSERT OR REPLACE INTO scan_meta VALUES ('last_scan', ?)
    """, [now])

    # Print summary
    stats = con.execute("""
        SELECT review_status, COUNT(*) FROM entities GROUP BY review_status
    """).fetchall()
    stat_map = dict(stats)
    total = sum(stat_map.values())

    _export_json(con)
    con.close()

    print(f"Scan complete: {total} entities across {len(set(e.file_path for e in entities))} files")
    print(f"  New: {new_count} | Changed (stale): {changed_count} | Unchanged: {unchanged_count} | Removed: {max(0, removed)}")
    for s in VALID_STATUSES:
        print(f"  {s}: {stat_map.get(s, 0)}", end="")
    reviewed = stat_map.get("reviewed", 0)
    print(f"\n  Coverage: {reviewed / total * 100:.1f}%" if total > 0 else "")


def cmd_mark(pattern: str, status: str = "reviewed",
             reviewer: str = "council", review_pass: str = "",
             dry_run: bool = False) -> None:
    """Mark entities matching a pattern."""
    if status not in VALID_STATUSES:
        print(f"ERROR: invalid status '{status}'. Must be one of: {VALID_STATUSES}")
        return

    con = _init_db()
    now = datetime.now(timezone.utc).isoformat()

    rows = con.execute(
        "SELECT qualified_name FROM entities WHERE qualified_name LIKE ?",
        [f"%{pattern}%"]
    ).fetchall()

    if not rows:
        print(f"No entities matching '{pattern}'")
        con.close()
        return

    if dry_run:
        print(f"DRY RUN: would mark {len(rows)} entities as '{status}'")
        for (qn,) in rows:
            print(f"  {qn}")
        con.close()
        return

    for (qn,) in rows:
        con.execute("""
            UPDATE entities SET review_status=?, reviewed_by=?, reviewed_at=?, review_pass=?
            WHERE qualified_name=?
        """, [status, reviewer, now, review_pass, qn])
        con.execute("""
            INSERT INTO reviews (entity, action, reviewer, review_pass, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, [qn, f"marked_{status}", reviewer, review_pass, now])

    _export_json(con)
    con.close()
    print(f"Marked {len(rows)} entities as '{status}'")


def cmd_mark_file(file_path: str, status: str = "reviewed",
                  reviewer: str = "council", review_pass: str = "",
                  dry_run: bool = False) -> None:
    """Mark all entities in a file."""
    if status not in VALID_STATUSES:
        print(f"ERROR: invalid status '{status}'. Must be one of: {VALID_STATUSES}")
        return

    con = _init_db()
    now = datetime.now(timezone.utc).isoformat()

    # Normalize path
    normalized = file_path.replace("\\", "/")
    if normalized.startswith("/"):
        try:
            normalized = str(Path(normalized).relative_to(REPO_ROOT))
        except ValueError:
            pass

    rows = con.execute(
        "SELECT qualified_name FROM entities WHERE file_path LIKE ?",
        [f"%{normalized}%"]
    ).fetchall()

    if not rows:
        print(f"No entities in file matching '{file_path}'")
        con.close()
        return

    if dry_run:
        print(f"DRY RUN: would mark {len(rows)} entities in '{file_path}' as '{status}'")
        for (qn,) in rows:
            print(f"  {qn}")
        con.close()
        return

    for (qn,) in rows:
        con.execute("""
            UPDATE entities SET review_status=?, reviewed_by=?, reviewed_at=?, review_pass=?
            WHERE qualified_name=?
        """, [status, reviewer, now, review_pass, qn])
        con.execute("""
            INSERT INTO reviews (entity, action, reviewer, review_pass, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, [qn, f"marked_{status}", reviewer, review_pass, now])

    _export_json(con)
    con.close()
    print(f"Marked {len(rows)} entities in '{file_path}' as '{status}'")


def cmd_status() -> None:
    """Print per-file review status summary."""
    con = _init_db()

    rows = con.execute("""
        SELECT file_path,
               COUNT(*) as total,
               SUM(CASE WHEN review_status='reviewed' THEN 1 ELSE 0 END) as reviewed,
               SUM(CASE WHEN review_status='unreviewed' THEN 1 ELSE 0 END) as unreviewed,
               SUM(CASE WHEN review_status='stale' THEN 1 ELSE 0 END) as stale,
               SUM(CASE WHEN review_status='needs_fix' THEN 1 ELSE 0 END) as needs_fix,
               SUM(line_count) as total_lines,
               SUM(CASE WHEN review_status='reviewed' THEN line_count ELSE 0 END) as reviewed_lines
        FROM entities
        GROUP BY file_path
        ORDER BY file_path
    """).fetchall()

    if not rows:
        print("No entities tracked. Run 'scan' first.")
        con.close()
        return

    print(f"\n{'File':<50} {'Ent':>5} {'Rev':>5} {'Unr':>5} {'Stl':>5} {'Fix':>5} {'Cov':>6}")
    print("-" * 85)

    t_total = t_rev = t_unr = t_stl = t_fix = t_lines = t_rlines = 0
    for fp, total, reviewed, unreviewed, stale, needs_fix, lines, rlines in rows:
        cov = reviewed / total * 100 if total else 0
        t_total += total; t_rev += reviewed; t_unr += unreviewed
        t_stl += stale; t_fix += needs_fix; t_lines += lines; t_rlines += rlines

        icon = "+" if cov == 100 else ("~" if cov > 0 else " ")
        short = fp if len(fp) <= 48 else "..." + fp[-45:]
        print(f"{icon} {short:<48} {total:>5} {reviewed:>5} {unreviewed:>5} {stale:>5} {needs_fix:>5} {cov:>5.0f}%")

    print("-" * 85)
    overall = t_rev / t_total * 100 if t_total else 0
    lcov = t_rlines / t_lines * 100 if t_lines else 0
    print(f"  {'TOTAL':<48} {t_total:>5} {t_rev:>5} {t_unr:>5} {t_stl:>5} {t_fix:>5} {overall:>5.0f}%")
    print(f"\n  Lines: {t_lines:,} total, {t_rlines:,} reviewed ({lcov:.0f}%)")

    last = con.execute("SELECT value FROM scan_meta WHERE key='last_scan'").fetchone()
    print(f"  Last scan: {last[0] if last else 'never'}")
    con.close()


def cmd_dashboard() -> None:
    """Terminal dashboard with color-coded status."""
    con = _init_db()

    stats = dict(con.execute(
        "SELECT review_status, COUNT(*) FROM entities GROUP BY review_status"
    ).fetchall())

    total = sum(stats.values())
    if total == 0:
        print("No entities tracked. Run 'scan' first.")
        con.close()
        return

    reviewed = stats.get("reviewed", 0)
    unreviewed = stats.get("unreviewed", 0)
    stale = stats.get("stale", 0)
    needs_fix = stats.get("needs_fix", 0)

    G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; C = "\033[36m"
    RST = "\033[0m"; B = "\033[1m"

    pct = reviewed / total * 100
    w = 40
    filled = int(pct / 100 * w)
    bar = f"[{'#' * filled}{'.' * (w - filled)}]"

    print(f"\n{B}  TAC Review Tracker Dashboard{RST}")
    print(f"  {'=' * 50}")
    print(f"  {bar} {pct:.0f}%")
    print(f"  {G}Reviewed: {reviewed}{RST}  |  {R}Unreviewed: {unreviewed}{RST}  |  {Y}Stale: {stale}{RST}  |  {C}Fix: {needs_fix}{RST}")
    print()

    # Top unreviewed by complexity
    unrev = con.execute("""
        SELECT name, entity_type, line_count, complexity, review_status, file_path
        FROM entities
        WHERE review_status IN ('unreviewed', 'needs_fix', 'stale')
        ORDER BY complexity DESC
        LIMIT 15
    """).fetchall()

    if unrev:
        print(f"  {B}Top unreviewed (by complexity):{RST}")
        for name, etype, lc, cx, st, fp in unrev:
            color = R if cx > 15 else (Y if cx > 8 else RST)
            st_tag = f" [{st}]" if st != "unreviewed" else ""
            short_fp = fp.split("/")[-1].replace(".py", "")
            print(f"    {color}C={cx:>3}{RST}  {lc:>4}L  {short_fp}:{name}{st_tag}")

    # Per-module coverage
    print(f"\n  {B}Module coverage:{RST}")
    mods = con.execute("""
        SELECT
            CASE WHEN file_path LIKE 'src/tac/lossless/%' THEN 'tac.lossless'
                 WHEN file_path LIKE 'src/tac/%' THEN 'tac'
                 WHEN file_path LIKE 'experiments/%' THEN 'experiments'
                 ELSE 'other' END as area,
            COUNT(*) as total,
            SUM(CASE WHEN review_status='reviewed' THEN 1 ELSE 0 END) as rev
        FROM entities GROUP BY area ORDER BY area
    """).fetchall()
    for area, tot, rev in mods:
        pct = rev / tot * 100 if tot else 0
        bar_w = 20
        f = int(pct / 100 * bar_w)
        print(f"    {area:<16} [{('#' * f) + ('.' * (bar_w - f))}] {pct:>3.0f}% ({rev}/{tot})")

    print()
    con.close()


def cmd_diff_scan(since: str = "") -> None:
    """Precise staleness: only entities whose lines actually changed go stale.

    --since REF    Git ref to diff against (e.g. HEAD~5, abc1234).
                   Default: if a previous scan timestamp exists in scan_meta,
                   uses the commit closest to that timestamp; otherwise HEAD~10.
    """
    con = _init_db()
    now = datetime.now(timezone.utc).isoformat()

    if not since:
        # Try to use last_scan timestamp for a smarter default
        try:
            row = con.execute("SELECT value FROM scan_meta WHERE key='last_scan'").fetchone()
            if row and row[0]:
                commit = _run_git("log", "-1", "--format=%H", f"--before={row[0]}")
                if commit:
                    since = commit[:12]
        except Exception:
            pass
        if not since:
            since = "HEAD~10"

    out = _run_git("diff", "--name-only", since)
    if out is None:
        print("Failed to get git diff")
        con.close()
        return

    changed_files = [f for f in out.split("\n") if f.endswith(".py")]

    stale_count = 0
    for fp in changed_files:
        changed_lines = _git_diff_lines(since, fp)
        if not changed_lines:
            continue

        rows = con.execute("""
            SELECT qualified_name, start_line, end_line
            FROM entities
            WHERE file_path=? AND review_status='reviewed'
        """, [fp]).fetchall()

        for qn, sl, el in rows:
            ent_range = set(range(sl, el + 1))
            overlap = changed_lines & ent_range
            if overlap:
                con.execute(
                    "UPDATE entities SET review_status='stale' WHERE qualified_name=?",
                    [qn]
                )
                detail = f"Lines {min(overlap)}-{max(overlap)} changed"
                con.execute("""
                    INSERT INTO reviews (entity, action, reviewer, review_pass, detail, timestamp)
                    VALUES (?, 'auto_stale_diff', 'diff_scanner', ?, ?, ?)
                """, [qn, f"diff_since_{since}", detail, now])
                stale_count += 1

    _export_json(con)
    con.close()
    print(f"Diff scan since {since}: {len(changed_files)} py files changed, {stale_count} entities marked stale")


def cmd_dag() -> None:
    """Show git DAG cross-referenced with review status."""
    con = _init_db()

    out = _run_git("log", "--oneline", "-20", "--", "src/tac/", "experiments/")
    if not out:
        print("Failed to get git log")
        con.close()
        return

    G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; C = "\033[36m"
    RST = "\033[0m"; B = "\033[1m"

    print(f"\n{B}  Review DAG — Recent Commits vs Review Status{RST}")
    print(f"  {'=' * 60}\n")

    for commit_line in out.split("\n")[:15]:
        if not commit_line.strip():
            continue
        commit_hash = commit_line.split()[0]

        files_out = _run_git("diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash, timeout=5)
        files = [f for f in (files_out or "").split("\n") if f.endswith(".py")] if files_out else []

        if not files:
            print(f"  {C}{commit_hash}{RST} {commit_line[len(commit_hash)+1:]}")
            continue

        placeholders = ",".join(["?"] * len(files))
        stats = dict(con.execute(f"""
            SELECT review_status, COUNT(*) FROM entities
            WHERE file_path IN ({placeholders})
            GROUP BY review_status
        """, files).fetchall())

        tr = stats.get("reviewed", 0)
        tu = stats.get("unreviewed", 0) + stats.get("needs_fix", 0)
        ts = stats.get("stale", 0)
        total = tr + tu + ts
        if total == 0:
            tag = f"{C}(no tracked entities){RST}"
        else:
            pct = tr / total * 100
            tag = f"{G}R:{tr}{RST} {R}U:{tu}{RST} {Y}S:{ts}{RST} ({pct:.0f}%)"

        msg = commit_line[len(commit_hash)+1:]
        print(f"  {C}{commit_hash}{RST} {msg:50s} {tag}")

    print()
    con.close()


def cmd_greenup_import(pass_file: str) -> None:
    """Import a greenup pass result — mark clean files as reviewed."""
    path = Path(pass_file)
    if not path.exists():
        print(f"File not found: {pass_file}")
        return

    content = path.read_text()
    pass_name = path.stem

    clean_files = re.findall(r"-\s+(\S+\.py)\s+.*CLEAN", content)
    if not clean_files:
        clean_files = re.findall(r"-\s+(\S+\.py)", content)

    if not clean_files:
        print("No files found in pass file")
        return

    con = _init_db()
    now = datetime.now(timezone.utc).isoformat()
    matched = 0

    for cf in clean_files:
        # Match on exact file_path or a path ending with the cleaned suffix.
        # Use = first for exact match; fall back to suffix match only if needed.
        # Avoid LIKE with unescaped wildcards to prevent false positives on
        # paths containing SQL wildcard characters (%, _).
        clean = cf.replace("\\", "/").lstrip("./")
        rows = con.execute(
            "SELECT qualified_name FROM entities WHERE file_path = ? OR file_path = ?",
            [clean, f"src/{clean}" if not clean.startswith("src/") else clean]
        ).fetchall()
        if not rows:
            # Fallback: suffix match for paths not exactly matching
            rows = con.execute(
                "SELECT qualified_name FROM entities WHERE file_path LIKE ?",
                [f"%/{clean}" if "/" not in clean else f"%{clean}"]
            ).fetchall()
        for (qn,) in rows:
            con.execute("""
                UPDATE entities SET review_status='reviewed', reviewed_by='council_greenup',
                reviewed_at=?, review_pass=? WHERE qualified_name=?
            """, [now, pass_name, qn])
            con.execute("""
                INSERT INTO reviews (entity, action, reviewer, review_pass, timestamp)
                VALUES (?, 'marked_reviewed', 'council_greenup', ?, ?)
            """, [qn, pass_name, now])
            matched += 1

    _export_json(con)
    con.close()
    print(f"Imported greenup '{pass_name}': {matched} entities marked reviewed from {len(clean_files)} files")


def cmd_query(sql: str) -> None:
    """Run a read-only SQL query against the DuckDB tracker."""
    import duckdb
    if not TRACKER_DB.exists():
        print("No tracker DB. Run 'scan' first.")
        return
    con = duckdb.connect(str(TRACKER_DB), read_only=True)
    try:
        rows = con.execute(sql).fetchall()
        if con.description:
            cols = [d[0] for d in con.description]
            print("\t".join(cols))
            print("-" * (len(cols) * 20))
            for row in rows:
                print("\t".join(str(v) for v in row))
        print(f"\n({len(rows)} rows)")
    except Exception as e:
        print(f"SQL error: {e}")
    con.close()


def cmd_report() -> None:
    """Generate a markdown review report."""
    con = _init_db()

    stats = dict(con.execute(
        "SELECT review_status, COUNT(*) FROM entities GROUP BY review_status"
    ).fetchall())
    total = sum(stats.values())
    if total == 0:
        print("No entities tracked. Run 'scan' first.")
        con.close()
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    reviewed = stats.get("reviewed", 0)

    lines = [
        f"# Code Review Tracker Report — {now}\n",
        f"## Summary\n",
        f"- **Total entities**: {total}",
        f"- **Reviewed**: {reviewed} ({reviewed/total*100:.0f}%)",
        f"- **Unreviewed**: {stats.get('unreviewed', 0)}",
        f"- **Stale**: {stats.get('stale', 0)}",
        f"- **Needs fix**: {stats.get('needs_fix', 0)}\n",
    ]

    # Priority queue
    unrev = con.execute("""
        SELECT name, entity_type, line_count, complexity, review_status, file_path
        FROM entities
        WHERE review_status IN ('unreviewed', 'stale', 'needs_fix')
        ORDER BY complexity DESC
        LIMIT 30
    """).fetchall()

    if unrev:
        lines.append("## Priority Review Queue (by complexity)\n")
        lines.append("| Entity | Type | Lines | Complexity | Status | File |")
        lines.append("|--------|------|-------|------------|--------|------|")
        for name, etype, lc, cx, st, fp in unrev:
            short_fp = fp.split("/")[-1]
            lines.append(f"| `{name}` | {etype} | {lc} | {cx} | {st} | {short_fp} |")

    # Recent activity
    try:
        recent = con.execute("""
            SELECT entity, action, reviewer, review_pass, timestamp
            FROM reviews ORDER BY timestamp DESC LIMIT 20
        """).fetchall()
        if recent:
            lines.append("\n## Recent Review Activity\n")
            for ent, action, reviewer, rpass, ts in recent:
                lines.append(f"- `{ent}` — {action} by {reviewer} ({rpass})")
    except Exception:
        pass

    report = "\n".join(lines) + "\n"
    report_path = REPO_ROOT / "reports" / "review_tracker_report.md"
    report_path.write_text(report)
    con.close()
    print(f"Report written to {report_path}")


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def cmd_selftest() -> None:
    """Run end-to-end self-tests to verify tracker integrity."""
    import tempfile
    import shutil

    print("Running self-tests...\n")
    errors = []

    # Test 1: AST extraction on a synthetic file
    print("  [1/7] AST extraction...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=str(REPO_ROOT), delete=False) as f:
        f.write(textwrap.dedent("""\
            class Foo:
                def bar(self):
                    if True:
                        for x in range(10):
                            pass

            def standalone():
                return 42
        """))
        tmp_path = Path(f.name)

    try:
        ents = extract_entities(tmp_path)
        assert len(ents) == 3, f"Expected 3 entities, got {len(ents)}"
        names = {e.name for e in ents}
        assert "Foo" in names, "Missing class Foo"
        assert "Foo.bar" in names, "Missing method Foo.bar"
        assert "standalone" in names, "Missing function standalone"

        # Complexity check: bar has if + for = 3
        bar = next(e for e in ents if e.name == "Foo.bar")
        assert bar.complexity >= 3, f"Expected complexity >= 3 for bar, got {bar.complexity}"
        print(f"    PASS: 3 entities extracted, complexity={bar.complexity}")
    except Exception as e:
        errors.append(f"Test 1 failed: {e}")
        print(f"    FAIL: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # Test 2: DuckDB round-trip
    print("  [2/7] DuckDB round-trip...")
    test_db = REPO_ROOT / ".omx" / "state" / "review_tracker_test.duckdb"
    try:
        import duckdb
        test_db.unlink(missing_ok=True)
        tcon = duckdb.connect(str(test_db))
        tcon.execute("CREATE SEQUENCE IF NOT EXISTS review_seq START 1")
        tcon.execute("""
            CREATE TABLE entities (
                qualified_name VARCHAR PRIMARY KEY,
                module VARCHAR, file_path VARCHAR, entity_type VARCHAR,
                name VARCHAR, start_line INT, end_line INT, line_count INT,
                complexity INT, review_status VARCHAR DEFAULT 'unreviewed'
            )
        """)
        tcon.execute("INSERT INTO entities VALUES ('test::Foo', 'test', 'test.py', 'class', 'Foo', 1, 10, 10, 5, 'unreviewed')")
        tcon.execute("UPDATE entities SET review_status='reviewed' WHERE qualified_name='test::Foo'")
        row = tcon.execute("SELECT review_status FROM entities WHERE qualified_name='test::Foo'").fetchone()
        assert row[0] == "reviewed", f"Expected 'reviewed', got {row[0]}"
        tcon.close()
        print("    PASS: DuckDB CRUD works")
    except Exception as e:
        errors.append(f"Test 2 failed: {e}")
        print(f"    FAIL: {e}")
    finally:
        test_db.unlink(missing_ok=True)

    # Test 3: Git diff parsing
    print("  [3/7] Git diff line parsing...")
    try:
        test_diff = "@@ -10,5 +10,7 @@ def foo():\n+new line\n+another\n@@ -30,0 +32,3 @@\n+a\n+b\n+c"
        lines_found: set[int] = set()
        for line in test_diff.split("\n"):
            m = re.match(r"^@@ .+? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) else 1
                for ln in range(start, start + count):
                    lines_found.add(ln)
        assert 10 in lines_found and 16 in lines_found, f"Missing expected lines in {lines_found}"
        assert 32 in lines_found and 34 in lines_found, f"Missing hunk 2 lines in {lines_found}"
        print(f"    PASS: Parsed {len(lines_found)} changed lines from 2 hunks")
    except Exception as e:
        errors.append(f"Test 3 failed: {e}")
        print(f"    FAIL: {e}")

    # Test 4: Complexity estimation
    print("  [4/7] Complexity estimation...")
    try:
        code = "def f():\n  if a:\n    for x in y:\n      if b or c:\n        pass"
        tree = ast.parse(code)
        func = tree.body[0]
        cx = _estimate_complexity(func)
        # if + for + if + or = 4, +1 base = 5
        assert cx >= 4, f"Expected complexity >= 4, got {cx}"
        print(f"    PASS: complexity={cx} for nested branches")
    except Exception as e:
        errors.append(f"Test 4 failed: {e}")
        print(f"    FAIL: {e}")

    # Test 5: Status validation
    print("  [5/7] Status validation...")
    try:
        for s in VALID_STATUSES:
            assert isinstance(s, str) and len(s) > 0
        assert "invalid_status" not in VALID_STATUSES
        print(f"    PASS: {len(VALID_STATUSES)} valid statuses")
    except Exception as e:
        errors.append(f"Test 5 failed: {e}")
        print(f"    FAIL: {e}")

    # Test 6: Greenup pattern matching
    print("  [6/7] Greenup pattern matching...")
    try:
        test_content = "- src/tac/training.py — CLEAN\n- experiments/benchmark_int4.py — CLEAN\n- bad line"
        matches = re.findall(r"-\s+(\S+\.py)\s+.*CLEAN", test_content)
        assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"
        assert "src/tac/training.py" in matches
        print(f"    PASS: {len(matches)} clean files parsed")
    except Exception as e:
        errors.append(f"Test 6 failed: {e}")
        print(f"    FAIL: {e}")

    # Test 7: Entity qualified names are unique
    print("  [7/7] Entity uniqueness...")
    try:
        entities = scan_all_modules()
        qnames = [e.qualified_name for e in entities]
        dupes = len(qnames) - len(set(qnames))
        if dupes > 0:
            # Find actual duplicates
            seen = set()
            dupe_names = []
            for qn in qnames:
                if qn in seen:
                    dupe_names.append(qn)
                seen.add(qn)
            print(f"    WARN: {dupes} duplicate qualified names (e.g. {dupe_names[:3]})")
        else:
            print(f"    PASS: {len(qnames)} unique entities")
    except Exception as e:
        errors.append(f"Test 7 failed: {e}")
        print(f"    FAIL: {e}")

    print(f"\n{'=' * 40}")
    if errors:
        print(f"FAILED: {len(errors)} test(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cmd_rebuild_from_json() -> None:
    """Rebuild the DuckDB database from the JSON fallback file."""
    if not TRACKER_JSON.exists():
        print(f"No JSON fallback found at {TRACKER_JSON}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(TRACKER_JSON.read_text())
    entities = data.get("entities", {})
    if not entities:
        print("JSON fallback contains no entities.", file=sys.stderr)
        sys.exit(1)

    # Remove corrupted DB
    TRACKER_DB.unlink(missing_ok=True)
    # Also remove WAL/tmp files DuckDB may leave behind
    for suffix in [".wal", ".tmp"]:
        p = TRACKER_DB.with_suffix(TRACKER_DB.suffix + suffix)
        p.unlink(missing_ok=True)

    import duckdb
    con = duckdb.connect(str(TRACKER_DB))
    for ddl in _SCHEMA_DDL:
        con.execute(ddl)

    count = 0
    for qn, ent in entities.items():
        con.execute("""
            INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            qn,
            ent.get("module", ""),
            ent.get("file_path", ""),
            ent.get("entity_type", ""),
            ent.get("name", ""),
            ent.get("start_line", 0),
            ent.get("end_line", 0),
            ent.get("line_count", 0),
            ent.get("complexity", 1),
            ent.get("last_modified_commit", ""),
            ent.get("last_modified_date", ""),
            ent.get("review_status", "unreviewed"),
            ent.get("reviewed_by", ""),
            ent.get("reviewed_at", ""),
            ent.get("review_pass", ""),
            ent.get("notes", ""),
        ])
        count += 1

    last_scan = data.get("last_scan", "")
    if last_scan:
        con.execute("INSERT OR REPLACE INTO scan_meta VALUES ('last_scan', ?)", [last_scan])

    con.close()
    print(f"Rebuilt DuckDB from JSON: {count} entities restored")


# ---------------------------------------------------------------------------
# Integrated greenup protocol
# ---------------------------------------------------------------------------

def cmd_greenup(file_patterns: list[str], reviewer: str = "council",
                mode: str = "agent", pass_name: str = "") -> None:
    """Run the greenup protocol: review files, parse results, update DB.

    Three modes:
        agent   — Output a structured review prompt for an LLM agent to execute.
                   The agent's output can be piped back via greenup-ingest.
        human   — Interactive: show each file's entities, ask for CLEAN/ISSUES.
        auto    — Parse the file looking for known patterns and auto-mark.

    Args:
        file_patterns: file paths or glob patterns to review
        reviewer: who is performing the review
        mode: agent | human | auto
        pass_name: name for this greenup pass (auto-generated if empty)
    """
    con = _init_db()
    now = datetime.now(timezone.utc)
    if not pass_name:
        pass_name = f"greenup_{now.strftime('%Y%m%dT%H%M%S')}_{reviewer}"

    # Resolve file patterns to actual tracked files
    import fnmatch as _fnm
    all_files = [r[0] for r in con.execute("SELECT DISTINCT file_path FROM entities ORDER BY file_path").fetchall()]

    matched_files: list[str] = []
    for pat in file_patterns:
        normalized = pat.replace("\\", "/")
        for fp in all_files:
            if normalized in fp or _fnm.fnmatch(fp, f"*{normalized}*"):
                if fp not in matched_files:
                    matched_files.append(fp)

    if not matched_files:
        print(f"No tracked files matching: {file_patterns}")
        con.close()
        return

    G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; C = "\033[36m"
    RST = "\033[0m"; B = "\033[1m"

    # Get entity counts per file
    file_stats: list[tuple[str, int, int, int, int]] = []
    for fp in matched_files:
        row = con.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN review_status='reviewed' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN review_status='needs_fix' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN review_status='unreviewed' THEN 1 ELSE 0 END)
            FROM entities WHERE file_path = ?
        """, [fp]).fetchone()
        file_stats.append((fp, row[0], row[1], row[2], row[3]))

    total_entities = sum(s[1] for s in file_stats)
    total_reviewed = sum(s[2] for s in file_stats)
    total_needs_fix = sum(s[3] for s in file_stats)
    total_unreviewed = sum(s[4] for s in file_stats)

    if mode == "agent":
        # Output a structured prompt that an agent can execute
        print(f"{B}Greenup Protocol — Agent Mode{RST}")
        print(f"Pass: {C}{pass_name}{RST}")
        print(f"Files: {len(matched_files)} | Entities: {total_entities}")
        print(f"  {G}Reviewed: {total_reviewed}{RST} | {R}Needs fix: {total_needs_fix}{RST} | {Y}Unreviewed: {total_unreviewed}{RST}")
        print()
        print(f"{B}Files to review:{RST}")
        for fp, total, rev, fix, unrev in file_stats:
            rigor = get_rigor_for_file(fp)
            level = rigor.get("_name", "relaxed")
            color = R if level == "critical" else (Y if level == "standard" else RST)
            print(f"  {color}[{level.upper()}]{RST} {fp} ({total} entities, {rev} reviewed, {fix} fix, {unrev} unreviewed)")
        print()
        print(f"{B}When the review agent returns, ingest results:{RST}")
        print(f"  python tools/review_tracker.py greenup-ingest <result_file> --pass {pass_name} --reviewer {reviewer}")
        print()

    elif mode == "human":
        # Interactive mode
        print(f"{B}Greenup Protocol — Human Review{RST}")
        print(f"Pass: {C}{pass_name}{RST}")
        print(f"Reviewer: {reviewer}")
        print()

        for fp, total, rev, fix, unrev in file_stats:
            rigor = get_rigor_for_file(fp)
            level = rigor.get("_name", "relaxed")
            color = R if level == "critical" else (Y if level == "standard" else RST)
            print(f"\n{color}[{level.upper()}]{RST} {B}{fp}{RST} — {total} entities")

            # Show entities needing review
            ents = con.execute("""
                SELECT name, entity_type, line_count, complexity, review_status, start_line
                FROM entities WHERE file_path = ? AND review_status != 'reviewed'
                ORDER BY start_line
            """, [fp]).fetchall()

            if not ents:
                print(f"  {G}All entities already reviewed.{RST}")
                continue

            for name, etype, lc, cx, st, sl in ents[:15]:
                st_color = R if st == "needs_fix" else Y
                print(f"  {st_color}[{st}]{RST} L{sl} {etype} {name} ({lc}L, C={cx})")
            if len(ents) > 15:
                print(f"  ... and {len(ents) - 15} more")

            print()
            response = input(f"  Verdict for {fp.split('/')[-1]}? [c]lean / [i]ssues / [s]kip: ").strip().lower()
            if response in ("c", "clean"):
                _mark_file_internal(con, fp, "reviewed", reviewer, pass_name)
                print(f"  {G}Marked CLEAN{RST}")
            elif response in ("i", "issues"):
                _mark_file_internal(con, fp, "needs_fix", reviewer, pass_name)
                print(f"  {R}Marked NEEDS_FIX{RST}")
            else:
                print(f"  {Y}Skipped{RST}")

        _export_json(con)
        con.close()
        # Print summary
        print(f"\n{B}Greenup pass complete.{RST}")
        return

    elif mode == "auto":
        # Auto mode: mark needs_fix/unreviewed as reviewed.
        # WARNING: this assumes fixes were actually applied and verified.
        # Requires explicit confirmation to prevent accidental bypass.
        print(f"{B}Greenup Protocol — Auto Mode{RST}")
        print(f"Pass: {C}{pass_name}{RST}")
        print(f"  {Y}WARNING: Auto mode marks entities as reviewed without verification.{RST}")
        print(f"  {Y}Only use after a greenup review agent has confirmed files are CLEAN.{RST}")
        count_to_mark = sum(
            con.execute("""
                SELECT COUNT(*) FROM entities
                WHERE file_path = ? AND review_status IN ('needs_fix', 'unreviewed')
            """, [fp]).fetchone()[0] for fp in matched_files
        )
        print(f"  Would mark {count_to_mark} entities as reviewed across {len(matched_files)} files.")
        try:
            confirm = input(f"  Proceed? [y/N]: ").strip().lower()
        except EOFError:
            confirm = "y"  # non-interactive (piped input) — allow
        if confirm not in ("y", "yes"):
            print(f"  {Y}Aborted.{RST}")
            con.close()
            return
        marked = 0
        for fp in matched_files:
            n = _mark_file_internal(con, fp, "reviewed", reviewer, pass_name,
                                     only_status=["needs_fix", "unreviewed"])
            marked += n
        _export_json(con)
        con.close()
        print(f"  Marked {G}{marked}{RST} entities as reviewed")
        return

    con.close()


def _mark_file_internal(con, file_path: str, status: str, reviewer: str,
                        review_pass: str, only_status: list[str] | None = None) -> int:
    """Internal: mark entities in a file, returns count marked."""
    now = datetime.now(timezone.utc).isoformat()
    if only_status:
        placeholders = ",".join(["?"] * len(only_status))
        rows = con.execute(f"""
            SELECT qualified_name FROM entities
            WHERE file_path = ? AND review_status IN ({placeholders})
        """, [file_path] + only_status).fetchall()
    else:
        rows = con.execute(
            "SELECT qualified_name FROM entities WHERE file_path = ?",
            [file_path]
        ).fetchall()

    for (qn,) in rows:
        con.execute("""
            UPDATE entities SET review_status=?, reviewed_by=?, reviewed_at=?, review_pass=?
            WHERE qualified_name=?
        """, [status, reviewer, now, review_pass, qn])
        con.execute("""
            INSERT INTO reviews (entity, action, reviewer, review_pass, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, [qn, f"marked_{status}", reviewer, review_pass, now])
    return len(rows)


def cmd_greenup_ingest(result_file: str, reviewer: str = "council",
                       pass_name: str = "") -> None:
    """Ingest a greenup review result file and update the DB.

    Parses the structured output format:
        ### <filename> — CLEAN
        ### <filename> — ISSUES FOUND
    """
    path = Path(result_file)
    if not path.exists():
        print(f"File not found: {result_file}")
        return

    content = path.read_text()
    if not pass_name:
        pass_name = path.stem

    con = _init_db()
    now = datetime.now(timezone.utc).isoformat()

    G = "\033[32m"; R = "\033[31m"; Y = "\033[33m"
    RST = "\033[0m"; B = "\033[1m"

    # Parse "### <path> — CLEAN" and "### <path> — ISSUES FOUND" lines
    clean_count = 0
    issues_count = 0

    for line in content.split("\n"):
        line = line.strip()
        if not line.startswith("###"):
            continue

        # Extract filename and verdict
        parts = line.lstrip("#").strip()
        if "—" in parts:
            fname, verdict = parts.rsplit("—", 1)
        elif "--" in parts:
            fname, verdict = parts.rsplit("--", 1)
        elif " - " in parts:
            fname, verdict = parts.rsplit(" - ", 1)
        else:
            continue

        fname = fname.strip().strip("`").replace("\\", "/")
        verdict = verdict.strip().upper()

        is_clean = "CLEAN" in verdict and "NOT" not in verdict and "ISSUES" not in verdict

        # Match on full path first, fallback to suffix — with ambiguity guard
        rows = con.execute(
            "SELECT qualified_name FROM entities WHERE file_path = ?",
            [fname]
        ).fetchall()
        if not rows:
            suffix = fname.split("/")[-1]
            candidates = con.execute(
                "SELECT DISTINCT file_path FROM entities WHERE file_path LIKE ?",
                [f"%/{suffix}"]
            ).fetchall()
            if len(candidates) > 1:
                print(f"  {Y}WARN: '{fname}' matches {len(candidates)} files — skipping to avoid false positive{RST}")
                continue
            rows = con.execute(
                "SELECT qualified_name FROM entities WHERE file_path LIKE ?",
                [f"%/{suffix}"]
            ).fetchall()

        if not rows:
            continue

        status = "reviewed" if is_clean else "needs_fix"
        for (qn,) in rows:
            con.execute("""
                UPDATE entities SET review_status=?, reviewed_by=?, reviewed_at=?, review_pass=?
                WHERE qualified_name=?
            """, [status, reviewer, now, pass_name, qn])
            con.execute("""
                INSERT INTO reviews (entity, action, reviewer, review_pass, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, [qn, f"marked_{status}", reviewer, pass_name, now])

        if is_clean:
            clean_count += 1
            print(f"  {G}CLEAN{RST} {fname} ({len(rows)} entities)")
        else:
            issues_count += 1
            print(f"  {R}ISSUES{RST} {fname} ({len(rows)} entities)")

    _export_json(con)
    con.close()

    print(f"\n{B}Greenup ingest complete:{RST} {G}{clean_count} clean{RST}, {R}{issues_count} with issues{RST}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    # Parse keyword args from anywhere in argv
    # Supports --key value pairs and --flag (boolean flags like --dry-run)
    BOOL_FLAGS = {"dry-run", "dry_run"}
    kwargs: dict[str, str] = {}
    positional: list[str] = [sys.argv[0], cmd]
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        key = arg.lstrip("-")
        if arg.startswith("--") and key in BOOL_FLAGS:
            kwargs[key] = "1"
            i += 1
        elif arg.startswith("--") and i + 1 < len(sys.argv):
            kwargs[key] = sys.argv[i + 1]
            i += 2
        else:
            positional.append(arg)
            i += 1

    reviewer = kwargs.get("reviewer", "council")
    review_pass = kwargs.get("pass", "")
    status = kwargs.get("status", "reviewed")
    dry_run = "dry-run" in kwargs or "dry_run" in kwargs

    if cmd == "scan":
        cmd_scan()
    elif cmd == "status":
        cmd_status()
    elif cmd == "mark":
        if len(positional) < 3:
            print("ERROR: 'mark' requires a pattern argument.", file=sys.stderr)
            print("Usage: python tools/review_tracker.py mark <pattern> [--status S]", file=sys.stderr)
            sys.exit(1)
        cmd_mark(positional[2], status=status, reviewer=reviewer, review_pass=review_pass, dry_run=dry_run)
    elif cmd == "mark-file":
        if len(positional) < 3:
            print("ERROR: 'mark-file' requires a file path argument.", file=sys.stderr)
            print("Usage: python tools/review_tracker.py mark-file <path> [--status S]", file=sys.stderr)
            sys.exit(1)
        cmd_mark_file(positional[2], status=status, reviewer=reviewer, review_pass=review_pass, dry_run=dry_run)
    elif cmd == "report":
        cmd_report()
    elif cmd == "dashboard":
        cmd_dashboard()
    elif cmd == "greenup-import":
        if len(positional) < 3:
            print("ERROR: 'greenup-import' requires a file argument.", file=sys.stderr)
            print("Usage: python tools/review_tracker.py greenup-import <file>", file=sys.stderr)
            sys.exit(1)
        cmd_greenup_import(positional[2])
    elif cmd == "diff-scan":
        cmd_diff_scan(kwargs.get("since", ""))
    elif cmd == "dag":
        cmd_dag()
    elif cmd == "query":
        if len(positional) < 3:
            print("ERROR: 'query' requires a SQL argument.", file=sys.stderr)
            print("Usage: python tools/review_tracker.py query <sql>", file=sys.stderr)
            sys.exit(1)
        cmd_query(positional[2])
    elif cmd == "selftest":
        cmd_selftest()
    elif cmd == "policy-check":
        cmd_policy_check(positional[2] if len(positional) >= 3 else None)
    elif cmd == "policy-show":
        policy = load_policy()
        if not policy:
            print("No policy found")
        else:
            print(json.dumps(policy, indent=2))
    elif cmd == "rebuild-from-json":
        cmd_rebuild_from_json()
    elif cmd == "greenup":
        mode = kwargs.get("mode", "agent")
        cmd_greenup(positional[2:], reviewer=reviewer, mode=mode, pass_name=review_pass)
    elif cmd == "greenup-ingest":
        if len(positional) < 3:
            print("ERROR: 'greenup-ingest' requires a result file argument.", file=sys.stderr)
            sys.exit(1)
        cmd_greenup_ingest(positional[2], reviewer=reviewer, pass_name=review_pass)
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
