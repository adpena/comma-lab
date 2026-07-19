#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit memory files for staleness, supersession, and oversized index lines.

Premortem #6 anchor (`.omx/research/12_month_frustration_premortem_and_
recommendations_20260516.md` Category D + Section 3 #6): without memory
rotation discipline, the memory directory passes 3,000 files in 12 months
and MEMORY.md becomes unreadable. The cross-reference graph rots silently.

This tool scans `~/.claude/projects/-Users-adpena-Projects-pact/memory/`
for five classes of hygiene violation:

1. **Stale-by-age:** files older than `--stale-days` (default 60) with no
   `superseded_by:` frontmatter — candidates for explicit supersession
   marking OR archival.
2. **Index line overflow:** MEMORY.md lines exceeding `--max-index-len`
   characters (default 200) — candidates for triage into a category
   summary file via `tools/cluster_summarize_memory_category.py`.
3. **Broken reference:** memos that cite another memo by name (e.g.
   `see feedback_X_20260510.md`) where the referenced file no longer
   exists on disk — the rename-rot failure mode.
4. **Past freshness window (SLA):** files whose `last_validated:`
   frontmatter (or, absent it, their mtime) is older than
   `--freshness-days` (default 90) and which are NOT tombstoned — the
   monthly "these rows are past their validation SLA" surface (#569 P0-3).
5. **Tombstones:** files carrying `superseded_by:` frontmatter — the
   EXPLICIT retirements, surfaced so a superseded row is visible-and-dated
   rather than silently stale. A superseded row is a tombstone, not a
   freshness violation.

Registry rows (`--registry <path.jsonl>`) get the same freshness-SLA pass
over a timestamp field (`last_validated` / `written_at_utc`), so the
"registry rows carry last-validated metadata" half of the SLA is honored
by the SAME tool rather than a parallel one.

Per CLAUDE.md "Memory file rotation discipline" non-negotiable. This is
operational hygiene only (no STRICT preflight gate) — the operator
runs this monthly and acts on the surfaced candidates.

Usage:
    .venv/bin/python tools/audit_memory_file_freshness.py
    .venv/bin/python tools/audit_memory_file_freshness.py --json
    .venv/bin/python tools/audit_memory_file_freshness.py --stale-days 90
    .venv/bin/python tools/audit_memory_file_freshness.py --memory-dir <path>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MEMORY_DIR = (
    Path.home() / ".claude" / "projects"
    / "-Users-adpena-Projects-pact" / "memory"
)
DEFAULT_STALE_DAYS = 60
DEFAULT_MAX_INDEX_LEN = 200
# Validation-SLA window (#569 P0-3): how long a memo/registry row may go
# without a re-validation touch before it is surfaced as "past freshness
# window". Distinct from stale-by-age (which is about supersession candidacy):
# a file can be recently authored yet never re-validated, and vice-versa.
DEFAULT_FRESHNESS_DAYS = 90

# Per-category rotation window map per CLAUDE.md "Memory file rotation
# discipline" non-negotiable + Wave 2C #8 finding (2026-05-19): a uniform
# 60-day window over-rotates fast-moving categories (catalog gates evolve
# in days; codex review cycles in weeks) and under-rotates slow ones
# (project state is meaningful for months). Each entry maps a filename
# substring (case-insensitive, FIRST match wins by iteration order) to
# its stale-day threshold. Filenames that match NONE of the substrings
# fall through to the global `--stale-days` default (60).
#
# Iteration order matters: more-specific substrings (e.g. `catalog_`)
# come BEFORE more-general ones (e.g. `feedback_`) so a file named
# `feedback_catalog_270_landed_*.md` is classified `catalog_` (7d) not
# `feedback_` (60d).
PER_CATEGORY_STALE_DAYS: dict[str, int] = {
    "catalog_": 7,        # catalog gates evolve fast (per-day landings)
    "fix_wave_": 12,      # fix-wave findings rotate within ~2 weeks
    "codex_": 21,         # codex review cycles run on ~3-week cadence
    "project_": 90,       # project state is meaningful for ~1 quarter
    "feedback_": 60,      # default long-tail memory window
}


def _category_stale_days(
    filename: str,
    *,
    default_days: int,
    category_map: dict[str, int] | None = None,
) -> tuple[int, str | None]:
    """Look up the per-category stale-day threshold for ``filename``.

    Returns ``(days, category_substring)`` where ``category_substring`` is
    ``None`` when no category matched (caller treats as "default window").
    Iteration order of ``category_map`` is preserved (Python 3.7+ insertion
    order); first matching substring wins, so more-specific prefixes must
    appear BEFORE more-general ones in the map definition.
    """
    cmap = category_map if category_map is not None else PER_CATEGORY_STALE_DAYS
    lower = filename.lower()
    for substr, days in cmap.items():
        if substr.lower() in lower:
            return days, substr
    return default_days, None


# Match `feedback_*.md` / `project_*.md` references in memo bodies
_MEMO_REF_RE = re.compile(
    r"\b(feedback_[a-z0-9_]+\.md|project_[a-z0-9_]+\.md)",
    re.IGNORECASE,
)


def _file_age_days(path: Path, now: datetime) -> float:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (now - mtime).total_seconds() / 86400.0


def _has_superseded_frontmatter(text: str) -> bool:
    """Detect `superseded_by:` in YAML frontmatter or first 30 body lines."""
    head = "\n".join(text.splitlines()[:30]).lower()
    return "superseded_by:" in head or "superseded by:" in head


_SUPERSEDED_BY_RE = re.compile(
    r"(?im)^\s*superseded[_ ]by\s*:\s*(.+?)\s*$",
)
_LAST_VALIDATED_RE = re.compile(
    r"(?im)^\s*last[_-]validated\s*:\s*[\"']?(\d{4}-\d{2}-\d{2})",
)


def _superseded_target(text: str) -> str | None:
    """Return the `superseded_by:` target (first 30 lines) if present, else None."""
    head = "\n".join(text.splitlines()[:30])
    m = _SUPERSEDED_BY_RE.search(head)
    if not m:
        return None
    return m.group(1).strip().strip("\"'[]") or "(unnamed)"


def _last_validated(text: str) -> datetime | None:
    """Parse a `last_validated: YYYY-MM-DD` frontmatter date (first 30 lines)."""
    head = "\n".join(text.splitlines()[:30])
    m = _LAST_VALIDATED_RE.search(head)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def audit_memory_files(
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    *,
    stale_days: int = DEFAULT_STALE_DAYS,
    max_index_len: int = DEFAULT_MAX_INDEX_LEN,
    freshness_days: int = DEFAULT_FRESHNESS_DAYS,
    now: datetime | None = None,
    category_map: dict[str, int] | None = None,
) -> dict:
    """Return a dict with the 5 audit classes.

    ``stale_days`` is the FALLBACK threshold for any filename that does NOT
    match an entry in ``category_map`` (default: ``PER_CATEGORY_STALE_DAYS``
    per Wave 2C #8 finding). Pass ``category_map={}`` to disable per-category
    rotation and restore uniform behavior.

    ``freshness_days`` is the validation-SLA window (#569 P0-3): a NON-tombstoned
    file whose ``last_validated:`` (or mtime fallback) age exceeds it is surfaced
    as ``past_freshness_window``; every ``superseded_by:`` file is surfaced as a
    ``tombstoned`` row (an explicit retirement, not a freshness violation).
    """
    now = now or datetime.now(timezone.utc)
    cmap = category_map if category_map is not None else PER_CATEGORY_STALE_DAYS
    result = {
        "memory_dir": str(memory_dir),
        "stale_by_age": [],
        "index_line_overflow": [],
        "broken_references": [],
        "past_freshness_window": [],
        "tombstoned": [],
        "summary": {},
    }
    if not memory_dir.is_dir():
        result["summary"]["error"] = f"memory_dir does not exist: {memory_dir}"
        return result

    # 1. Single per-file pass: stale-by-age + validation-SLA freshness +
    #    tombstones. Collect the filename set for the cross-ref check.
    all_files = sorted(memory_dir.glob("*.md"))
    filename_set: set[str] = {p.name for p in all_files}

    for p in all_files:
        if p.name == "MEMORY.md":
            continue  # the flat index is handled separately (overflow scan)
        try:
            age_days = _file_age_days(p, now)
        except OSError:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        superseded_target = _superseded_target(text)
        is_tombstone = superseded_target is not None
        if is_tombstone:
            result["tombstoned"].append({
                "filename": p.name,
                "superseded_by": superseded_target,
                "age_days": round(age_days, 1),
            })

        # Validation-SLA freshness: non-tombstoned rows whose last-validated
        # (or mtime fallback) age exceeds the freshness window.
        if not is_tombstone:
            lv = _last_validated(text)
            if lv is not None:
                validated_age = (now - lv).total_seconds() / 86400.0
                lv_source = "last_validated"
            else:
                validated_age = age_days
                lv_source = "mtime_fallback"
            if validated_age > freshness_days:
                result["past_freshness_window"].append({
                    "filename": p.name,
                    "validated_age_days": round(validated_age, 1),
                    "validation_source": lv_source,
                    "freshness_days": freshness_days,
                })

        # Stale-by-age (per-category supersession candidacy): tombstoned rows
        # are already explicitly retired, so they are excluded (unchanged
        # behavior from the original _has_superseded_frontmatter skip).
        threshold, category = _category_stale_days(
            p.name, default_days=stale_days, category_map=cmap,
        )
        if age_days <= threshold or is_tombstone:
            continue
        result["stale_by_age"].append({
            "filename": p.name,
            "age_days": round(age_days, 1),
            "size_bytes": p.stat().st_size,
            "category": category,
            "threshold_days": threshold,
        })

    # 2. MEMORY.md index-line overflow.
    memory_md = memory_dir / "MEMORY.md"
    if memory_md.is_file():
        try:
            lines = memory_md.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            lines = []
        for lineno, line in enumerate(lines, start=1):
            if len(line) > max_index_len:
                # Only flag lines that look like index entries (start
                # with `- [` or `- *` etc.) — skip body paragraphs.
                stripped = line.lstrip()
                if stripped.startswith(("- [", "* [", "- ")):
                    result["index_line_overflow"].append({
                        "lineno": lineno,
                        "length": len(line),
                        "preview": line[:80] + "...",
                    })

    # 3. Broken reference scan (sample the freshest 50 files to keep
    # the scan O(N) bounded).
    recent_files = sorted(
        all_files, key=lambda p: p.stat().st_mtime, reverse=True
    )[:50]
    seen_pairs: set[tuple[str, str]] = set()
    for p in recent_files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _MEMO_REF_RE.finditer(text):
            ref = m.group(1)
            if ref not in filename_set and (p.name, ref) not in seen_pairs:
                seen_pairs.add((p.name, ref))
                result["broken_references"].append({
                    "referenced_by": p.name,
                    "missing_target": ref,
                })

    # Per-category breakdown so the report can surface where the cadence
    # is firing most often (the operator decides which categories to
    # rotate-or-supersede first).
    per_category_counts: dict[str, int] = {}
    for row in result["stale_by_age"]:
        key = row.get("category") or "<default>"
        per_category_counts[key] = per_category_counts.get(key, 0) + 1

    result["summary"] = {
        "total_files_scanned": len(all_files),
        "stale_by_age_count": len(result["stale_by_age"]),
        "index_line_overflow_count": len(result["index_line_overflow"]),
        "broken_references_count": len(result["broken_references"]),
        "past_freshness_window_count": len(result["past_freshness_window"]),
        "tombstoned_count": len(result["tombstoned"]),
        "stale_days_threshold": stale_days,
        "max_index_len_threshold": max_index_len,
        "freshness_days_threshold": freshness_days,
        "category_map": dict(cmap),
        "stale_by_category_count": per_category_counts,
    }
    return result


def audit_registry_freshness(
    registry_path: Path,
    *,
    freshness_days: int = DEFAULT_FRESHNESS_DAYS,
    id_fields: tuple[str, ...] = ("task_id", "lane_id", "equation_id", "deliberation_id"),
    ts_fields: tuple[str, ...] = ("last_validated", "written_at_utc", "registered_at_utc",
                                  "event_timestamp_utc"),
    now: datetime | None = None,
) -> dict:
    """Validation-SLA pass over an append-only JSONL registry (#569 P0-3).

    Collapses to latest-row-wins per id, then lists ids whose freshest
    timestamp (first present of ``ts_fields``) is older than ``freshness_days``.
    Rows with a truthy ``superseded_by`` / ``tombstoned`` field are surfaced as
    tombstones, not freshness violations. This gives registry rows the SAME
    last-validated surface as memory files, from the SAME tool.
    """
    now = now or datetime.now(timezone.utc)
    out = {
        "registry_path": str(registry_path),
        "past_freshness_window": [],
        "tombstoned": [],
        "summary": {},
    }
    if not registry_path.is_file():
        out["summary"]["error"] = f"registry does not exist: {registry_path}"
        return out
    latest: dict[str, dict] = {}
    try:
        lines = registry_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        out["summary"]["error"] = f"read failed: {exc}"
        return out
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        rid = next((str(row[f]) for f in id_fields if row.get(f)), f"row{i}")
        latest[rid] = row  # later row wins
    for rid, row in sorted(latest.items()):
        if row.get("superseded_by") or row.get("tombstoned"):
            out["tombstoned"].append({
                "id": rid,
                "superseded_by": str(row.get("superseded_by") or "(tombstoned)"),
            })
            continue
        ts_raw = next((str(row[f]) for f in ts_fields if row.get(f)), "")
        m = re.search(r"(\d{4}-\d{2}-\d{2})", ts_raw)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        age = (now - ts).total_seconds() / 86400.0
        if age > freshness_days:
            out["past_freshness_window"].append({
                "id": rid,
                "validated_age_days": round(age, 1),
                "freshness_days": freshness_days,
            })
    out["summary"] = {
        "total_ids": len(latest),
        "past_freshness_window_count": len(out["past_freshness_window"]),
        "tombstoned_count": len(out["tombstoned"]),
        "freshness_days_threshold": freshness_days,
    }
    return out


def _format_report(audit: dict) -> str:
    summary = audit.get("summary", {})
    cmap = summary.get("category_map", {})
    out = [
        "\n=== Memory file freshness audit ===",
        f"Memory dir: {audit['memory_dir']}",
        f"Total files scanned: {summary.get('total_files_scanned', 0)}",
        f"Stale (default {summary.get('stale_days_threshold', 0)}d, "
        f"per-category map honored): "
        f"{summary.get('stale_by_age_count', 0)}",
        f"MEMORY.md index lines over "
        f"{summary.get('max_index_len_threshold', 0)} chars: "
        f"{summary.get('index_line_overflow_count', 0)}",
        f"Broken cross-references (recent 50 files): "
        f"{summary.get('broken_references_count', 0)}",
        f"Past freshness window (SLA {summary.get('freshness_days_threshold', 0)}d): "
        f"{summary.get('past_freshness_window_count', 0)}",
        f"Tombstoned (superseded_by present): "
        f"{summary.get('tombstoned_count', 0)}",
    ]

    if cmap:
        out.append("\nPer-category rotation map (Wave 2C #8 finding):")
        for substr, days in cmap.items():
            out.append(f"  {substr:<20} -> {days:>3}d window")
        out.append(
            f"  {'<default>':<20} -> "
            f"{summary.get('stale_days_threshold', 0):>3}d "
            "(fallback for non-matching filenames)"
        )

    by_cat = summary.get("stale_by_category_count", {})
    if by_cat:
        out.append("\nStale counts by matched category:")
        for cat, count in sorted(
            by_cat.items(), key=lambda kv: kv[1], reverse=True
        ):
            out.append(f"  {cat:<20} {count:>5}")

    stale = audit.get("stale_by_age", [])
    if stale:
        out.append("\nStale-by-age candidates (top 15 by age):")
        for row in sorted(
            stale, key=lambda r: r["age_days"], reverse=True
        )[:15]:
            cat = row.get("category") or "<default>"
            threshold = row.get("threshold_days", summary.get(
                "stale_days_threshold", 0))
            out.append(
                f"  {row['age_days']:>6.1f}d  [{cat:<14} "
                f">{threshold:>3}d]  {row['filename']}"
            )
        if len(stale) > 15:
            out.append(f"  ... +{len(stale) - 15} more")

    overflow = audit.get("index_line_overflow", [])
    if overflow:
        out.append("\nMEMORY.md index-line overflow (top 5 by length):")
        for row in sorted(
            overflow, key=lambda r: r["length"], reverse=True
        )[:5]:
            out.append(
                f"  L{row['lineno']:>4}  {row['length']:>4}ch  "
                f"{row['preview']}"
            )

    broken = audit.get("broken_references", [])
    if broken:
        out.append("\nBroken references (rename-rot detection):")
        for row in broken[:10]:
            out.append(
                f"  {row['referenced_by']} -> {row['missing_target']} "
                "(missing)"
            )
        if len(broken) > 10:
            out.append(f"  ... +{len(broken) - 10} more")

    freshness = audit.get("past_freshness_window", [])
    if freshness:
        out.append(
            f"\nPast freshness window (validation SLA "
            f"{summary.get('freshness_days_threshold', 0)}d, top 15 by age):"
        )
        for row in sorted(
            freshness, key=lambda r: r["validated_age_days"], reverse=True
        )[:15]:
            out.append(
                f"  {row['validated_age_days']:>6.1f}d  "
                f"[{row['validation_source']:<14}]  {row['filename']}"
            )
        if len(freshness) > 15:
            out.append(f"  ... +{len(freshness) - 15} more")

    tombstoned = audit.get("tombstoned", [])
    if tombstoned:
        out.append("\nTombstones (explicit supersession — visible, not silently stale):")
        for row in tombstoned[:10]:
            out.append(
                f"  {row['filename']} -> superseded_by {row['superseded_by']}"
            )
        if len(tombstoned) > 10:
            out.append(f"  ... +{len(tombstoned) - 10} more")

    out.append("\nRemediation (per CLAUDE.md \"Memory file rotation discipline\"):")
    out.append(
        "  - Stale-by-age: add `superseded_by: <newer_memo.md>` frontmatter "
        "OR move to MEMORY_ARCHIVE_<YYYYQQ>.md."
    )
    out.append(
        "  - Index overflow: triage via `tools/cluster_summarize_memory_category.py "
        "--category <prefix>` to compress N memos into 1 cluster summary."
    )
    out.append(
        "  - Broken references: rename the missing target back OR update the "
        "citing memo to point at the canonical successor."
    )
    out.append(
        "  - Past freshness window: re-validate + stamp `last_validated: "
        "<YYYY-MM-DD>` frontmatter, OR tombstone it with `superseded_by: "
        "<newer_memo.md>` if the claim has moved on."
    )
    out.append(
        "  - Tombstones: these are healthy (explicit retirements). Verify the "
        "`superseded_by:` target still exists (see broken-references above)."
    )
    return "\n".join(out)


def _format_registry_report(audit: dict) -> str:
    summary = audit.get("summary", {})
    out = [
        "\n=== Registry freshness audit ===",
        f"Registry: {audit['registry_path']}",
    ]
    if summary.get("error"):
        out.append(f"  ERROR: {summary['error']}")
        return "\n".join(out)
    out.append(f"Total ids (latest-row-wins): {summary.get('total_ids', 0)}")
    out.append(
        f"Past freshness window (SLA {summary.get('freshness_days_threshold', 0)}d): "
        f"{summary.get('past_freshness_window_count', 0)}"
    )
    out.append(f"Tombstoned: {summary.get('tombstoned_count', 0)}")
    stale = audit.get("past_freshness_window", [])
    if stale:
        out.append("\nPast-SLA rows (top 15 by age):")
        for row in sorted(
            stale, key=lambda r: r["validated_age_days"], reverse=True
        )[:15]:
            out.append(f"  {row['validated_age_days']:>6.1f}d  {row['id']}")
        if len(stale) > 15:
            out.append(f"  ... +{len(stale) - 15} more")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR,
    )
    parser.add_argument(
        "--stale-days", type=int, default=DEFAULT_STALE_DAYS,
    )
    parser.add_argument(
        "--max-index-len", type=int, default=DEFAULT_MAX_INDEX_LEN,
    )
    parser.add_argument(
        "--freshness-days", type=int, default=DEFAULT_FRESHNESS_DAYS,
        help="validation-SLA window in days (#569 P0-3; default 90). A "
        "non-tombstoned file past this since last_validated (or mtime) is "
        "surfaced as past_freshness_window.",
    )
    parser.add_argument(
        "--registry", type=Path, default=None,
        help="also run the validation-SLA pass over a JSONL registry "
        "(e.g. .omx/state/canonical_task_status.jsonl)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON instead of human-readable report",
    )
    parser.add_argument(
        "--disable-category-map", action="store_true",
        help="Disable per-category rotation (use uniform --stale-days for all "
        "filenames). Default honors PER_CATEGORY_STALE_DAYS per Wave 2C #8.",
    )
    parser.add_argument(
        "--show-category-map", action="store_true",
        help="Print PER_CATEGORY_STALE_DAYS map and exit (no audit)",
    )
    args = parser.parse_args()

    if args.show_category_map:
        for substr, days in PER_CATEGORY_STALE_DAYS.items():
            print(f"{substr:<20} {days:>3}d")
        print(f"{'<default>':<20} {args.stale_days:>3}d (fallback)")
        return 0

    category_map = {} if args.disable_category_map else None
    audit = audit_memory_files(
        memory_dir=args.memory_dir,
        stale_days=args.stale_days,
        max_index_len=args.max_index_len,
        freshness_days=args.freshness_days,
        category_map=category_map,
    )
    registry_audit = None
    if args.registry is not None:
        registry_audit = audit_registry_freshness(
            args.registry, freshness_days=args.freshness_days,
        )
    if args.json:
        payload = {"memory": audit}
        if registry_audit is not None:
            payload["registry"] = registry_audit
        print(json.dumps(payload, indent=2))
        return 0
    print(_format_report(audit))
    if registry_audit is not None:
        print(_format_registry_report(registry_audit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
