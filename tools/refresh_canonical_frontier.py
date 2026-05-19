#!/usr/bin/env python
"""Operator-runnable canonical frontier pointer refresher.

Per CLAUDE.md "Frontier scores are pointer-only - NON-NEGOTIABLE" + Catalog
#343: canonical SoT for OUR LOCAL FRONTIER + PUBLIC LEADERBOARD scores is
``.omx/state/canonical_frontier_pointer.json``. This CLI is the
operator-facing surface for manual refresh.

Examples::

    # Default: refresh from local canonical state and print human summary.
    .venv/bin/python tools/refresh_canonical_frontier.py

    # Opt in to upstream leaderboard fetch (network call).
    .venv/bin/python tools/refresh_canonical_frontier.py --update-upstream

    # Strict mode: exit rc=1 if pointer is stale (>24h since last refresh).
    .venv/bin/python tools/refresh_canonical_frontier.py --strict

    # Machine-readable JSON output.
    .venv/bin/python tools/refresh_canonical_frontier.py --json

Auto-update wire-in (Catalog #343 Layer 4): every successful Modal /
HF Jobs dispatch completion fires
``tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome``
inside ``update_call_id_outcome`` / ``update_hf_jobs_outcome``. So the
operator rarely needs to run this CLI manually - the pointer auto-refreshes
on every harvested dispatch outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.canonical_frontier_pointer import (
    CANONICAL_FRONTIER_POINTER_PATH,
    CanonicalFrontierPointer,
    load_canonical_frontier_pointer_lenient,
    refresh_canonical_frontier_from_local_state,
    refresh_canonical_frontier_from_upstream_leaderboard,
)


REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent


def _render_pointer_summary(pointer: CanonicalFrontierPointer) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("CANONICAL FRONTIER POINTER (single source of truth)")
    lines.append("=" * 72)
    lines.append(f"Pointer file: .omx/state/canonical_frontier_pointer.json")
    lines.append(f"Schema:       {pointer.schema_version}")
    lines.append(f"Refreshed at: {pointer.last_refreshed_utc}")
    lines.append(f"Auto-update:  {pointer.auto_update_on_dispatch_completion}")
    lines.append("")
    lines.append("-" * 72)
    lines.append("OUR LOCAL FRONTIER (qualifying 1:1 contest hardware)")
    lines.append("-" * 72)
    for axis_label, anchor in (
        ("contest-CPU (GHA Linux x86_64)", pointer.our_local_frontier_contest_cpu),
        ("contest-CUDA (NVIDIA T4/A10G/A100/4090/H100/L40S)", pointer.our_local_frontier_contest_cuda),
    ):
        lines.append("")
        if anchor is None:
            lines.append(f"  {axis_label}: <no qualifying anchor>")
            continue
        lines.append(f"  {axis_label}:")
        lines.append(f"    score          = {anchor.score:.10f}")
        lines.append(f"    archive_sha256 = {anchor.archive_sha256}")
        lines.append(f"    lane_id        = {anchor.lane_id or '<none>'}")
        lines.append(f"    hardware       = {anchor.hardware_substrate}")
        lines.append(f"    measured_at    = {anchor.measured_at_utc or '<unknown>'}")
        lines.append(f"    evidence       = {anchor.evidence_grade}")
    lines.append("")
    lines.append("-" * 72)
    lines.append("PR SUBMISSION STATUS FOR CURRENT FRONTIER")
    lines.append("-" * 72)
    pr_number = pointer.submitted_pr_number_for_current_frontier
    if pr_number is None:
        lines.append("  No PR submitted yet for current local frontier.")
        lines.append("  (Operator may be holding pending further improvement.)")
    else:
        lines.append(f"  Submitted PR #{pr_number} for current frontier.")
    lines.append("")
    lines.append("-" * 72)
    lines.append("UPSTREAM PUBLIC LEADERBOARD SNAPSHOT")
    lines.append("-" * 72)
    snapshot = pointer.upstream_leaderboard_snapshot
    if snapshot is None:
        lines.append("  No upstream snapshot fetched yet. Pass --update-upstream to fetch.")
    elif isinstance(snapshot, dict):
        status = snapshot.get("fetch_status", "unknown")
        fetched = snapshot.get("fetched_at_utc", "<unknown>")
        pulls = snapshot.get("pulls", [])
        lines.append(f"  fetch_status: {status}")
        lines.append(f"  fetched_at:   {fetched}")
        lines.append(f"  pulls_count:  {len(pulls) if isinstance(pulls, list) else 0}")
        if status != "ok":
            err = snapshot.get("fetch_error", "<unknown>")
            lines.append(f"  fetch_error:  {err}")
        if isinstance(pulls, list) and pulls:
            lines.append("  most-recent 5 PRs:")
            for entry in pulls[:5]:
                if not isinstance(entry, dict):
                    continue
                num = entry.get("number")
                title = (entry.get("title") or "")[:60]
                state = entry.get("state")
                lines.append(f"    PR #{num} ({state}): {title}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the canonical frontier pointer (Catalog #343).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT_DEFAULT,
        help="Repo root (default: this tool's parent parent).",
    )
    parser.add_argument(
        "--update-local",
        action="store_true",
        default=True,
        help="(default True) Refresh local frontier from canonical state.",
    )
    parser.add_argument(
        "--no-update-local",
        dest="update_local",
        action="store_false",
        help="Skip local-state refresh (print existing pointer only).",
    )
    parser.add_argument(
        "--update-upstream",
        action="store_true",
        default=False,
        help="(opt-in) Fetch upstream public leaderboard snapshot (~30s network call).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=30,
        help="Upstream fetch timeout in seconds (default 30).",
    )
    parser.add_argument(
        "--print",
        dest="print_summary",
        action="store_true",
        default=True,
        help="(default True) Emit human-readable summary.",
    )
    parser.add_argument(
        "--no-print",
        dest="print_summary",
        action="store_false",
        help="Suppress summary output.",
    )
    parser.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit machine-readable JSON (suppresses --print).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit rc=1 if pointer is stale (>24h since last refresh).",
    )
    parser.add_argument(
        "--submitted-pr-number",
        type=int,
        default=None,
        help="Record a PR number for the current frontier (e.g. 108).",
    )
    args = parser.parse_args(argv)

    repo_root: Path = args.repo_root

    pointer: CanonicalFrontierPointer | None = None
    if args.update_upstream:
        pointer = refresh_canonical_frontier_from_upstream_leaderboard(
            repo_root=repo_root, timeout_sec=args.timeout_sec, write=True
        )
        if args.submitted_pr_number is not None:
            # Re-refresh local with PR number to record it (upstream snapshot
            # preserved by the local-refresh path).
            pointer = refresh_canonical_frontier_from_local_state(
                repo_root=repo_root,
                write=True,
                submitted_pr_number_for_current_frontier=args.submitted_pr_number,
                pre_existing_pointer=pointer,
            )
    elif args.update_local:
        pointer = refresh_canonical_frontier_from_local_state(
            repo_root=repo_root,
            write=True,
            submitted_pr_number_for_current_frontier=args.submitted_pr_number,
        )
    else:
        pointer = load_canonical_frontier_pointer_lenient(repo_root=repo_root)

    if pointer is None:
        sys.stderr.write(
            "FATAL: canonical frontier pointer not populated; "
            "run `tools/refresh_canonical_frontier.py --update-local`\n"
        )
        return 2

    if args.emit_json:
        sys.stdout.write(json.dumps(pointer.as_dict(), indent=2, sort_keys=True) + "\n")
    elif args.print_summary:
        sys.stdout.write(_render_pointer_summary(pointer) + "\n")

    if args.strict and pointer.is_stale():
        sys.stderr.write(
            f"STRICT: pointer is stale (last refreshed {pointer.last_refreshed_utc}); "
            "run with --update-local or --update-upstream\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
