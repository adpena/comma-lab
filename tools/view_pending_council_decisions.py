#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""View pending council design decisions in a readable table.

Reads ``.omx/state/pending_council_design_decisions.jsonl`` (the canonical
queue per CLAUDE.md "Subagent coherence-by-default" + the standing referral
at ``.omx/research/all_design_decisions_through_grand_council_directive_20260514.md``)
and pretty-prints rows grouped by ``council_priority``.

Per CLAUDE.md "Beauty, simplicity, and developer experience" — the operator
should not need to grep raw JSONL to see what is queued for council review.

Usage:
    .venv/bin/python tools/view_pending_council_decisions.py
    .venv/bin/python tools/view_pending_council_decisions.py --status pending_council
    .venv/bin/python tools/view_pending_council_decisions.py --status all
    .venv/bin/python tools/view_pending_council_decisions.py --json
    .venv/bin/python tools/view_pending_council_decisions.py --decision-id oss_push_hygiene_sweep_2_remediation_option

Exit codes:
    0 — read OK (regardless of how many rows match)
    2 — JSONL file missing or malformed
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

DEFAULT_PATH = Path(".omx/state/pending_council_design_decisions.jsonl")
PRIORITY_ORDER = ("HIGH", "MEDIUM", "LOW", "UNKNOWN")


def load_rows(path: Path) -> list[dict]:
    """Load JSONL rows. Raises FileNotFoundError or json.JSONDecodeError."""
    if not path.exists():
        raise FileNotFoundError(f"queue file not found: {path}")
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(
                f"{path}:{lineno}: {exc.msg}", exc.doc, exc.pos
            ) from exc
    return rows


def filter_rows(
    rows: Iterable[dict],
    *,
    status: str = "pending_council",
    decision_id: str | None = None,
) -> list[dict]:
    """Filter rows by status (or 'all') and optional decision_id substring."""
    out = []
    for r in rows:
        if status != "all" and r.get("status") != status:
            continue
        if decision_id is not None and decision_id not in r.get("decision_id", ""):
            continue
        out.append(r)
    return out


def group_by_priority(rows: Iterable[dict]) -> dict[str, list[dict]]:
    """Group rows by council_priority. Unknown priorities bucket as UNKNOWN."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        prio = r.get("council_priority") or "UNKNOWN"
        if prio not in PRIORITY_ORDER:
            prio = "UNKNOWN"
        grouped[prio].append(r)
    return grouped


def format_row(row: dict) -> str:
    """Format a single row as a multi-line block."""
    decision_id = row.get("decision_id", "<missing>")
    title = row.get("title", "<no title>")
    source = row.get("source_lane") or row.get("source_subagent") or "<unsourced>"
    cost = row.get("cost_usd")
    cost_s = f"${cost}" if cost is not None else "n/a"
    blocking = row.get("blocking") or "—"
    queued = row.get("queued_utc", "<no queue ts>")
    status = row.get("status", "<no status>")
    lines = [
        f"  decision_id: {decision_id}",
        f"  title:       {title}",
        f"  status:      {status}",
        f"  source:      {source}",
        f"  cost:        {cost_s}",
        f"  blocking:    {blocking}",
        f"  queued_utc:  {queued}",
    ]
    options = row.get("options")
    if options:
        lines.append(f"  options ({len(options)}):")
        for opt in options:
            lines.append(f"    - {opt}")
    if row.get("status") == "resolved":
        res = row.get("resolution") or row.get("verdict") or "<no resolution body>"
        resolved_by = row.get("resolved_by", "<unknown adjudicator>")
        lines.append(f"  resolved_by: {resolved_by}")
        lines.append(f"  resolution:  {res}")
    return "\n".join(lines)


def render(grouped: dict[str, list[dict]], *, total: int) -> str:
    """Render grouped rows as a hierarchical text table."""
    parts = [
        "PENDING COUNCIL DESIGN DECISIONS",
        f"Total rows: {total}",
        "",
    ]
    rendered_any = False
    for prio in PRIORITY_ORDER:
        bucket = grouped.get(prio, [])
        if not bucket:
            continue
        rendered_any = True
        parts.append(f"── {prio} ({len(bucket)}) ──")
        for row in bucket:
            parts.append(format_row(row))
            parts.append("")
    if not rendered_any:
        parts.append("(no rows match filter)")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="View pending council design decisions."
    )
    parser.add_argument(
        "--queue-path",
        type=Path,
        default=DEFAULT_PATH,
        help=f"Path to JSONL queue (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--status",
        default="pending_council",
        help=(
            "Filter by status field (default: pending_council). "
            "Pass 'all' to show every row regardless of status."
        ),
    )
    parser.add_argument(
        "--decision-id",
        default=None,
        help="Optional substring filter on decision_id",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human table",
    )
    args = parser.parse_args(argv)

    try:
        rows = load_rows(args.queue_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSONL: {exc}", file=sys.stderr)
        return 2

    filtered = filter_rows(rows, status=args.status, decision_id=args.decision_id)

    if args.json:
        print(json.dumps(filtered, indent=2, sort_keys=True))
        return 0

    grouped = group_by_priority(filtered)
    print(render(grouped, total=len(filtered)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
