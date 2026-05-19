#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit memory files for staleness, supersession, and oversized index lines.

Premortem #6 anchor (`.omx/research/12_month_frustration_premortem_and_
recommendations_20260516.md` Category D + Section 3 #6): without memory
rotation discipline, the memory directory passes 3,000 files in 12 months
and MEMORY.md becomes unreadable. The cross-reference graph rots silently.

This tool scans `~/.claude/projects/-Users-adpena-Projects-pact/memory/`
for three classes of hygiene violation:

1. **Stale-by-age:** files older than `--stale-days` (default 60) with no
   `superseded_by:` frontmatter — candidates for explicit supersession
   marking OR archival.
2. **Index line overflow:** MEMORY.md lines exceeding `--max-index-len`
   characters (default 200) — candidates for triage into a category
   summary file via `tools/cluster_summarize_memory_category.py`.
3. **Broken reference:** memos that cite another memo by name (e.g.
   `see feedback_X_20260510.md`) where the referenced file no longer
   exists on disk — the rename-rot failure mode.

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


def audit_memory_files(
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    *,
    stale_days: int = DEFAULT_STALE_DAYS,
    max_index_len: int = DEFAULT_MAX_INDEX_LEN,
    now: datetime | None = None,
    category_map: dict[str, int] | None = None,
) -> dict:
    """Return a dict with the 3 audit classes.

    ``stale_days`` is the FALLBACK threshold for any filename that does NOT
    match an entry in ``category_map`` (default: ``PER_CATEGORY_STALE_DAYS``
    per Wave 2C #8 finding). Pass ``category_map={}`` to disable per-category
    rotation and restore uniform behavior.
    """
    now = now or datetime.now(timezone.utc)
    cmap = category_map if category_map is not None else PER_CATEGORY_STALE_DAYS
    result = {
        "memory_dir": str(memory_dir),
        "stale_by_age": [],
        "index_line_overflow": [],
        "broken_references": [],
        "summary": {},
    }
    if not memory_dir.is_dir():
        result["summary"]["error"] = f"memory_dir does not exist: {memory_dir}"
        return result

    # 1. Stale-by-age (per-category windows) + collect filename set for
    #    cross-ref check.
    all_files = sorted(memory_dir.glob("*.md"))
    filename_set: set[str] = {p.name for p in all_files}

    for p in all_files:
        try:
            age_days = _file_age_days(p, now)
        except OSError:
            continue
        threshold, category = _category_stale_days(
            p.name, default_days=stale_days, category_map=cmap,
        )
        if age_days <= threshold:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _has_superseded_frontmatter(text):
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
        "stale_days_threshold": stale_days,
        "max_index_len_threshold": max_index_len,
        "category_map": dict(cmap),
        "stale_by_category_count": per_category_counts,
    }
    return result


def _format_report(audit: dict) -> str:
    summary = audit.get("summary", {})
    cmap = summary.get("category_map", {})
    out = [
        f"\n=== Memory file freshness audit ===",
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
        out.append(f"\nStale-by-age candidates (top 15 by age):")
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
        out.append(f"\nMEMORY.md index-line overflow (top 5 by length):")
        for row in sorted(
            overflow, key=lambda r: r["length"], reverse=True
        )[:5]:
            out.append(
                f"  L{row['lineno']:>4}  {row['length']:>4}ch  "
                f"{row['preview']}"
            )

    broken = audit.get("broken_references", [])
    if broken:
        out.append(f"\nBroken references (rename-rot detection):")
        for row in broken[:10]:
            out.append(
                f"  {row['referenced_by']} -> {row['missing_target']} "
                "(missing)"
            )
        if len(broken) > 10:
            out.append(f"  ... +{len(broken) - 10} more")

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
        category_map=category_map,
    )
    if args.json:
        print(json.dumps(audit, indent=2))
        return 0
    print(_format_report(audit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
