#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Warn when ``tools/harvest_modal_calls.py`` has not run recently.

Modal ``.spawn()`` puts artifacts in the FunctionCall return-value cache with a
~24h TTL. Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable
HIGHEST EMPHASIS, every dispatch via ``experiments/modal_train_lane.py`` MUST
be followed by a scheduled harvest within 24h.

This tool reads ``experiments/results/_modal_harvest_summary.json`` mtime and
compares against the configured staleness threshold (default 4h). It also
counts how many ``experiments/results/lane_*_modal/modal_metadata.json``
entries lack a sibling ``harvested_artifacts/`` directory (unharvested).

Use as a shell-prompt hint, manual cron sentinel, or pre-dispatch readiness
check.

Usage:
    .venv/bin/python tools/check_modal_harvest_freshness.py
    .venv/bin/python tools/check_modal_harvest_freshness.py --threshold-hours 8
    .venv/bin/python tools/check_modal_harvest_freshness.py --json

Exit codes:
    0 — fresh (last harvest within threshold) and no unharvested call_ids
    1 — STALE: last harvest exceeded threshold AND at least one unharvested
        modal_metadata.json exists. The user-facing banner instructs the
        operator to run ``.venv/bin/python tools/harvest_modal_calls.py --execute``.
    2 — config / IO error (missing repo, malformed summary, etc.)
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_REPO_ROOT = Path(".")
DEFAULT_SUMMARY_REL = Path("experiments/results/_modal_harvest_summary.json")
DEFAULT_THRESHOLD_HOURS = 4.0
HARVEST_CMD = ".venv/bin/python tools/harvest_modal_calls.py --execute"


@dataclass
class FreshnessReport:
    summary_path: str
    summary_exists: bool
    summary_mtime_utc: str | None
    summary_age_hours: float | None
    threshold_hours: float
    is_stale: bool
    unharvested_count: int
    unharvested_paths: list[str]
    suggested_command: str | None
    note: str


def _utc_iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()


def find_unharvested(repo_root: Path) -> list[Path]:
    """Return modal_metadata.json paths under lane_*_modal/ with no harvested_artifacts/ sibling."""
    results_root = repo_root / "experiments" / "results"
    if not results_root.exists():
        return []
    out: list[Path] = []
    for meta in results_root.glob("lane_*_modal/modal_metadata.json"):
        if not (meta.parent / "harvested_artifacts").is_dir():
            out.append(meta)
    return sorted(out)


def build_report(
    repo_root: Path,
    *,
    summary_rel: Path = DEFAULT_SUMMARY_REL,
    threshold_hours: float = DEFAULT_THRESHOLD_HOURS,
    now: float | None = None,
) -> FreshnessReport:
    if now is None:
        now = time.time()
    summary_path = repo_root / summary_rel
    unharvested = find_unharvested(repo_root)

    if not summary_path.exists():
        # No summary file means harvest never ran (or never completed --execute).
        # Stale only if there's something pending; otherwise informational.
        is_stale = bool(unharvested)
        return FreshnessReport(
            summary_path=str(summary_path),
            summary_exists=False,
            summary_mtime_utc=None,
            summary_age_hours=None,
            threshold_hours=threshold_hours,
            is_stale=is_stale,
            unharvested_count=len(unharvested),
            unharvested_paths=[str(p) for p in unharvested],
            suggested_command=HARVEST_CMD if is_stale else None,
            note=(
                "no harvest summary found — run --execute at least once"
                if is_stale
                else "no harvest summary and no pending dispatches"
            ),
        )

    mtime = summary_path.stat().st_mtime
    age_hours = (now - mtime) / 3600.0
    is_stale = age_hours > threshold_hours and bool(unharvested)
    return FreshnessReport(
        summary_path=str(summary_path),
        summary_exists=True,
        summary_mtime_utc=_utc_iso(mtime),
        summary_age_hours=round(age_hours, 3),
        threshold_hours=threshold_hours,
        is_stale=is_stale,
        unharvested_count=len(unharvested),
        unharvested_paths=[str(p) for p in unharvested],
        suggested_command=HARVEST_CMD if is_stale else None,
        note=(
            f"STALE: last harvest {age_hours:.2f}h ago, "
            f"{len(unharvested)} unharvested dispatch(es) at risk of result-cache eviction"
            if is_stale
            else (
                f"fresh: last harvest {age_hours:.2f}h ago "
                f"(threshold {threshold_hours:.1f}h); "
                f"{len(unharvested)} unharvested dispatch(es)"
                if unharvested
                else f"fresh: last harvest {age_hours:.2f}h ago and nothing pending"
            )
        ),
    )


def render_text(report: FreshnessReport) -> str:
    lines = ["MODAL HARVEST FRESHNESS"]
    lines.append(f"  summary_path:        {report.summary_path}")
    lines.append(f"  summary_exists:      {report.summary_exists}")
    if report.summary_exists:
        lines.append(f"  summary_mtime_utc:   {report.summary_mtime_utc}")
        lines.append(f"  summary_age_hours:   {report.summary_age_hours}")
    lines.append(f"  threshold_hours:     {report.threshold_hours}")
    lines.append(f"  unharvested_count:   {report.unharvested_count}")
    lines.append(f"  is_stale:            {report.is_stale}")
    lines.append(f"  note:                {report.note}")
    if report.suggested_command:
        lines.append("")
        lines.append("ACTION:")
        lines.append(f"  {report.suggested_command}")
    if report.unharvested_paths:
        lines.append("")
        n = len(report.unharvested_paths)
        if n > 10:
            lines.append(f"UNHARVESTED (showing first 10 of {n}):")
        else:
            lines.append(f"UNHARVESTED ({n}):")
        for p in report.unharvested_paths[:10]:
            lines.append(f"  {p}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Warn when Modal harvest is stale and dispatches are pending."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root (default: cwd)",
    )
    parser.add_argument(
        "--threshold-hours",
        type=float,
        default=DEFAULT_THRESHOLD_HOURS,
        help=f"Staleness threshold in hours (default: {DEFAULT_THRESHOLD_HOURS})",
    )
    parser.add_argument(
        "--summary-rel",
        type=Path,
        default=DEFAULT_SUMMARY_REL,
        help=f"Path to harvest summary relative to repo root (default: {DEFAULT_SUMMARY_REL})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human text",
    )
    args = parser.parse_args(argv)

    if not args.repo_root.exists():
        print(f"ERROR: repo-root not found: {args.repo_root}", file=sys.stderr)
        return 2

    report = build_report(
        args.repo_root,
        summary_rel=args.summary_rel,
        threshold_hours=args.threshold_hours,
    )

    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(render_text(report))

    return 1 if report.is_stale else 0


if __name__ == "__main__":
    sys.exit(main())
