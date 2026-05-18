# SPDX-License-Identifier: MIT
"""Backfill helpers for the canonical DuckDB read-model.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md]
[verified-against: Catalog #128 fcntl-locked sister discipline]
[verified-against: Catalog #131 no-bare-writes-to-shared-state]
[verified-against: Catalog #245 modal_call_id_ledger 4-layer canonical pattern]
[verified-against: Catalog #265 canonical-contract tokens]

Catalog # citations: the refresh path consumes existing canonicals and writes a
derived DuckDB read-model under a sister lock. JSON/JSONL/Markdown files remain
the source of truth.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from tac.canonical_duckdb.schema import CANONICAL_TABLES, connect


DEFAULT_DB_PATH = Path(".omx/state/canonical.duckdb")
CANONICAL_LOCK_PATH = Path(".omx/state/.canonical_duckdb.lock")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo_path(repo_root: Path, path: Path) -> str:
    return str(path.relative_to(repo_root))


@contextmanager
def canonical_duckdb_lock(repo_root: Path):
    """Acquire the sister lock for derived DuckDB writes."""
    lock_path = repo_root / CANONICAL_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _parse_markdown_frontmatter(text: str) -> dict:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    frontmatter: dict[str, object] = {}
    for raw_line in text[4:end].splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        if value.lower() == "true":
            parsed: object = True
        elif value.lower() == "false":
            parsed = False
        elif value.lower() in {"null", "none"}:
            parsed = None
        else:
            parsed = value
        frontmatter[key.strip()] = parsed
    return frontmatter


def _bool_from_frontmatter(frontmatter: dict, key: str) -> bool:
    value = frontmatter.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def refresh_lanes(con, repo_root: Path, refreshed_at: str) -> int:
    path = repo_root / ".omx/state/lane_registry.json"
    con.execute("DELETE FROM lanes")
    if not path.exists():
        return 0
    payload = json.loads(path.read_text())
    rows = []
    for lane in payload.get("lanes", []):
        rows.append(
            (
                lane.get("id", ""),
                lane.get("name", ""),
                lane.get("phase"),
                lane.get("level"),
                json.dumps(lane.get("gates", {}), sort_keys=True),
                lane.get("notes", ""),
                bool(lane.get("research_only", False)),
                refreshed_at,
                _repo_path(repo_root, path),
            )
        )
    con.executemany(
        """
        INSERT OR REPLACE INTO lanes
        (lane_id, name, phase, level, gates, notes, research_only, refreshed_at_utc, source_path)
        VALUES (?, ?, ?, ?, ?::JSON, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def refresh_research_memos(con, repo_root: Path, refreshed_at: str) -> int:
    research_root = repo_root / ".omx/research"
    con.execute("DELETE FROM research_memos")
    if not research_root.is_dir():
        return 0
    rows = []
    for path in sorted(research_root.glob("*.md")):
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        frontmatter = _parse_markdown_frontmatter(text)
        title = str(frontmatter.get("title") or path.stem)
        lane_id = str(frontmatter.get("lane_id") or frontmatter.get("lane") or "")
        rows.append(
            (
                _repo_path(repo_root, path),
                title,
                str(frontmatter.get("date_utc") or frontmatter.get("date") or ""),
                lane_id,
                _bool_from_frontmatter(frontmatter, "research_only"),
                _bool_from_frontmatter(frontmatter, "score_claim"),
                _bool_from_frontmatter(frontmatter, "promotion_eligible"),
                _sha256_bytes(data),
                len(data),
                json.dumps(frontmatter, sort_keys=True),
                refreshed_at,
            )
        )
    con.executemany(
        """
        INSERT OR REPLACE INTO research_memos
        (path, title, date_utc, lane_id, research_only, score_claim, promotion_eligible,
         sha256, bytes, frontmatter, refreshed_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?::JSON, ?)
        """,
        rows,
    )
    return len(rows)


def refresh_state_json_files(con, repo_root: Path, refreshed_at: str) -> int:
    state_root = repo_root / ".omx/state"
    con.execute("DELETE FROM state_json_files")
    if not state_root.is_dir():
        return 0
    rows = []
    for path in sorted(state_root.rglob("*.json")):
        data = path.read_bytes()
        try:
            payload = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {"parse_error": True}
        rows.append(
            (
                _repo_path(repo_root, path),
                _sha256_bytes(data),
                len(data),
                json.dumps(payload, sort_keys=True),
                refreshed_at,
            )
        )
    con.executemany(
        """
        INSERT OR REPLACE INTO state_json_files
        (path, sha256, bytes, payload, refreshed_at_utc)
        VALUES (?, ?, ?, ?::JSON, ?)
        """,
        rows,
    )
    return len(rows)


def refresh_state_jsonl_files(con, repo_root: Path, refreshed_at: str) -> int:
    state_root = repo_root / ".omx/state"
    con.execute("DELETE FROM state_jsonl_files")
    if not state_root.is_dir():
        return 0
    rows = []
    for path in sorted(state_root.rglob("*.jsonl")):
        data = path.read_bytes()
        row_count = sum(1 for line in data.splitlines() if line.strip())
        rows.append(
            (
                _repo_path(repo_root, path),
                _sha256_bytes(data),
                len(data),
                row_count,
                refreshed_at,
            )
        )
    con.executemany(
        """
        INSERT OR REPLACE INTO state_jsonl_files
        (path, sha256, bytes, row_count, refreshed_at_utc)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _refresh_empty_json_table(con, table: str) -> int:
    con.execute(f"DELETE FROM {table}")
    return 0


REFRESH_DISPATCH: dict[str, Callable] = {
    "lanes": refresh_lanes,
    "research_memos": refresh_research_memos,
    "state_json_files": refresh_state_json_files,
    "state_jsonl_files": refresh_state_jsonl_files,
}
"""Table refresh functions with first implemented read-model tables."""


def refresh_table(
    table: str,
    repo_root: str | Path,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict:
    """Refresh one canonical DuckDB table from source-of-truth files."""
    if table not in CANONICAL_TABLES:
        raise ValueError(f"unknown canonical DuckDB table: {table}")
    root = Path(repo_root)
    db = Path(db_path)
    if not db.is_absolute():
        db = root / db
    refreshed_at = _utc_now_iso()
    with canonical_duckdb_lock(root):
        con = connect(db)
        try:
            if table in REFRESH_DISPATCH:
                row_count = REFRESH_DISPATCH[table](con, root, refreshed_at)
            else:
                row_count = _refresh_empty_json_table(con, table)
        finally:
            con.close()
    return {
        "table": table,
        "row_count": row_count,
        "refreshed_at_utc": refreshed_at,
        "db_path": str(db),
        "source_of_truth": "json_jsonl_markdown",
    }


def refresh_all_tables(
    repo_root: str | Path,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    tables: Iterable[str] = CANONICAL_TABLES,
) -> dict[str, dict]:
    """Refresh all requested canonical DuckDB tables."""
    return {
        table: refresh_table(table, repo_root, db_path=db_path)
        for table in tables
    }


__all__ = [
    "CANONICAL_LOCK_PATH",
    "DEFAULT_DB_PATH",
    "REFRESH_DISPATCH",
    "canonical_duckdb_lock",
    "refresh_all_tables",
    "refresh_table",
]
