#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recover in-flight agents killed mid-flight (task #388).

Thin CLI over ``tac.session_bus.recovery_manifest`` (which is itself a thin layer over the
canonical ``.omx/state/subagent_progress.jsonl`` crash-resume store). Three subcommands:

    tools/session_recover.py report [--stale-after-seconds N]
    tools/session_recover.py register --subagent-id ID --respawn-context TEXT \
        [--expected-outputs a,b] [--files-touched a,b] [--next-action ...] \
        [--parent-id-or-session ...] [--lane-id ...] [--notes ...] [--step N]
    tools/session_recover.py complete --subagent-id ID [--files-touched a,b] [--notes ...]

``report`` prints a ready-to-paste respawn block for every agent whose latest checkpoint is
in-progress with no heartbeat within the staleness window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.session_bus import recovery_manifest as rm  # noqa: E402


def _csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="subcommand", required=True)

    report_p = sub.add_parser("report", help="Render respawn blocks for crashed agents.")
    report_p.add_argument(
        "--stale-after-seconds",
        type=float,
        default=rm.DEFAULT_STALE_AFTER_SECONDS,
        help="An in-progress agent with no heartbeat older than this is a candidate.",
    )

    reg_p = sub.add_parser("register", help="Register a new in-flight agent.")
    reg_p.add_argument("--subagent-id", required=True)
    reg_p.add_argument("--respawn-context", required=True)
    reg_p.add_argument("--expected-outputs", default="")
    reg_p.add_argument("--files-touched", default="")
    reg_p.add_argument("--next-action", default="")
    reg_p.add_argument("--parent-id-or-session", default=None)
    reg_p.add_argument("--lane-id", default=None)
    reg_p.add_argument("--notes", default="")
    reg_p.add_argument("--step", type=int, default=1)

    comp_p = sub.add_parser("complete", help="Mark an agent complete.")
    comp_p.add_argument("--subagent-id", required=True)
    comp_p.add_argument("--files-touched", default="")
    comp_p.add_argument("--notes", default="")

    args = parser.parse_args(argv)

    if args.subcommand == "report":
        entries = rm.recover_report(stale_after_seconds=args.stale_after_seconds)
        print(rm.render_report(entries))
        return 0

    if args.subcommand == "register":
        rec = rm.register_inflight(
            args.subagent_id,
            args.respawn_context,
            expected_outputs=_csv(args.expected_outputs) or None,
            files_touched=_csv(args.files_touched),
            next_action=args.next_action,
            parent_id_or_session=args.parent_id_or_session,
            lane_id=args.lane_id,
            notes=args.notes,
            step=args.step,
        )
        print(f"[session-recover] registered {rec['subagent_id']} (step {rec['step']})")
        return 0

    if args.subcommand == "complete":
        rec = rm.complete(
            args.subagent_id,
            files_touched=_csv(args.files_touched),
            notes=args.notes,
        )
        print(f"[session-recover] completed {rec['subagent_id']}")
        return 0

    parser.error(f"unknown subcommand {args.subcommand!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
