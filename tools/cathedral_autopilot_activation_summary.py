#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for the cathedral autopilot consumer verdict ledger.

Per T3 council prioritization 2026-05-19 rank #4 ACTIVATION sprint
(memo ``.omx/research/council_t3_tier_45_backlog_prioritization_20260519.md``
commit ``79bd5695d``): the cathedral autopilot is runtime-activated per
Catalog #336/#337 (32 contract-compliant consumers fire per iteration), but
verdicts were emitted to stdout only and lost when the process exited. This
CLI is the operator-runnable surface for the canonical fcntl-locked verdict
ledger landed in ``src/tac/cathedral/verdict_ledger.py`` (sister of
Catalog #245 Modal call_id ledger / Catalog #313 probe-outcomes ledger /
Catalog #333 codex-to-claude inbox / Catalog #344 canonical equations
registry).

Subcommands
-----------

``latest``      Print the most-recent invocation batch summary.
``sessions``    List recent invocation batches (date-filterable).
``consumers``   Aggregate per-consumer activity counts (cite-chain).
``top``         Top-N ranked candidates from the most recent invocation batch.

All commands honor ``--json`` for machine-readable output.

Usage
-----

::

    .venv/bin/python tools/cathedral_autopilot_activation_summary.py latest
    .venv/bin/python tools/cathedral_autopilot_activation_summary.py top --n 5
    .venv/bin/python tools/cathedral_autopilot_activation_summary.py consumers --json
    .venv/bin/python tools/cathedral_autopilot_activation_summary.py sessions --since 2026-05-19T00:00:00Z

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
the output is observability-only. Every printed score-related field is
labelled with the canonical axis tag ``[predicted]`` and accompanied by
``score_claim=False`` + ``promotion_eligible=False``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.cathedral.verdict_ledger import (  # noqa: E402
    CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH,
    load_verdict_events_lenient,
    query_consumer_activity_summary,
    query_latest_session,
    query_sessions,
)


def _print_table(rows: list[dict[str, Any]], cols: list[str], file=None) -> None:
    """Quick-and-readable text table (no external dep)."""
    if not rows:
        print("(no rows)", file=file or sys.stdout)
        return
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    print(header, file=file or sys.stdout)
    print(sep, file=file or sys.stdout)
    for r in rows:
        line = "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols)
        print(line, file=file or sys.stdout)


def cmd_latest(args: argparse.Namespace) -> int:
    latest = query_latest_session(path=args.ledger_path)
    if latest is None:
        msg = "no invocation batches recorded yet — run cathedral_autopilot with --persist-consumer-verdicts to populate the ledger"
        if args.json:
            print(json.dumps({"empty": True, "message": msg}, indent=2))
        else:
            print(msg, file=sys.stderr)
        return 0

    if args.json:
        # Drop verbose top_candidates_summary for the default scalar view.
        # Caller can use `top` for full candidates.
        slim = {k: v for k, v in latest.items() if k != "top_candidates_summary"}
        print(json.dumps(slim, indent=2, sort_keys=True))
        return 0

    print(f"Most recent cathedral autopilot invocation batch:")
    print(f"  session_id:           {latest.get('session_id', '?')}")
    print(f"  written_at_utc:       {latest.get('written_at_utc', '?')}")
    print(f"  panel_axis:           {latest.get('panel_axis', '?')}")
    print(f"  rank_axis:            {latest.get('rank_axis', '?')}")
    print(f"  consumer_count:       {latest.get('consumer_count', 0)}")
    print(f"  candidates_invoked:   {latest.get('candidates_invoked', 0)}")
    print(f"  n_invocations:        {latest.get('n_invocations', 0)}")
    print(f"  n_non_vacuous:        {latest.get('n_non_vacuous', 0)}")
    print(f"  n_errors:             {latest.get('n_errors', 0)}")
    print(f"  master_gradient_ann:  {latest.get('master_gradient_annotation_count', 0)}")
    print(f"  invocations_summary:  {latest.get('invocations_summary_path', '(none)')}")
    print(f"  axis_tag:             {latest.get('axis_tag', '[predicted]')}  (observability-only per Catalog #287/#323)")
    print(f"  notes:                {latest.get('notes', '')}")
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    rows = query_sessions(
        since_utc=args.since,
        until_utc=args.until,
        path=args.ledger_path,
    )
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("(no sessions in range)", file=sys.stderr)
        return 0
    print(f"Cathedral autopilot invocation batches ({len(rows)} matching):")
    summary_rows = [
        {
            "session_id": r.get("session_id", "?")[:32],
            "written_at_utc": r.get("written_at_utc", "?"),
            "consumers": r.get("consumer_count", 0),
            "candidates": r.get("candidates_invoked", 0),
            "n_non_vac": r.get("n_non_vacuous", 0),
            "errors": r.get("n_errors", 0),
            "axis": r.get("panel_axis", "?"),
        }
        for r in rows
    ]
    _print_table(summary_rows, ["session_id", "written_at_utc", "consumers", "candidates", "n_non_vac", "errors", "axis"])
    return 0


def cmd_consumers(args: argparse.Namespace) -> int:
    summary = query_consumer_activity_summary(
        since_utc=args.since, path=args.ledger_path,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if not summary:
        print("(no consumer activity recorded)", file=sys.stderr)
        return 0
    print(f"Per-consumer activity ({len(summary)} consumers seen):")
    rows = [
        {
            "consumer_name": name[:64],
            "sessions": stats["session_count"],
            "candidates": stats["candidate_count_total"],
            "last_seen": stats["last_seen_utc"],
        }
        for name, stats in sorted(summary.items())
    ]
    _print_table(rows, ["consumer_name", "sessions", "candidates", "last_seen"])
    return 0


def cmd_top(args: argparse.Namespace) -> int:
    latest = query_latest_session(path=args.ledger_path)
    if latest is None:
        msg = "no invocation batches recorded yet — run cathedral_autopilot with --persist-consumer-verdicts to populate the ledger"
        if args.json:
            print(json.dumps({"empty": True, "message": msg}, indent=2))
        else:
            print(msg, file=sys.stderr)
        return 0

    top_summary = latest.get("top_candidates_summary", []) or []
    n = max(1, int(args.n))
    top_n = top_summary[:n]
    if args.json:
        payload = {
            "session_id": latest.get("session_id"),
            "written_at_utc": latest.get("written_at_utc"),
            "panel_axis": latest.get("panel_axis"),
            "rank_axis": latest.get("rank_axis"),
            "axis_tag": latest.get("axis_tag", "[predicted]"),
            "score_claim": False,
            "promotion_eligible": False,
            "top_candidates": top_n,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Top {len(top_n)} candidates from most recent cathedral autopilot invocation:")
    print(f"  session: {latest.get('session_id', '?')}")
    print(f"  written: {latest.get('written_at_utc', '?')}  axis={latest.get('panel_axis', '?')}")
    print(f"  EVIDENCE LABEL: [predicted; observability-only per Catalog #287/#323]")
    print()
    rows = [
        {
            "rank": c.get("rank", "?"),
            "candidate_id": str(c.get("candidate_id", "?"))[:60],
            "archive_sha256": (c.get("archive_sha256") or "(none)")[:12],
            "pred_Δ_raw": f"{c.get('predicted_score_delta_raw', 0.0):+.4f}",
            "cost_$": f"{c.get('estimated_dispatch_cost_usd', 0.0):.2f}",
            "blockers": c.get("blockers_count", 0),
        }
        for c in top_n
    ]
    _print_table(rows, ["rank", "candidate_id", "archive_sha256", "pred_Δ_raw", "cost_$", "blockers"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH,
        help=(
            "Override the canonical ledger path (default "
            f"{CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH.relative_to(REPO_ROOT)})."
        ),
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_latest = sub.add_parser("latest", help="Print most recent invocation batch summary")
    p_latest.add_argument("--json", action="store_true", help="Emit JSON")
    p_latest.set_defaults(func=cmd_latest)

    p_sessions = sub.add_parser("sessions", help="List recent invocation batches")
    p_sessions.add_argument("--since", default=None, help="UTC since (ISO YYYY-MM-DDTHH:MM:SSZ)")
    p_sessions.add_argument("--until", default=None, help="UTC until")
    p_sessions.add_argument("--json", action="store_true", help="Emit JSON")
    p_sessions.set_defaults(func=cmd_sessions)

    p_consumers = sub.add_parser("consumers", help="Per-consumer activity summary")
    p_consumers.add_argument("--since", default=None, help="UTC since")
    p_consumers.add_argument("--json", action="store_true", help="Emit JSON")
    p_consumers.set_defaults(func=cmd_consumers)

    p_top = sub.add_parser("top", help="Top-N ranked candidates from most recent batch")
    p_top.add_argument("--n", type=int, default=10, help="How many candidates (default 10)")
    p_top.add_argument("--json", action="store_true", help="Emit JSON")
    p_top.set_defaults(func=cmd_top)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
