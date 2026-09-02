"""External read-only telemetry sampler for live burns (operator 2026-09-02
"Need full telemetry and signal always").

Polls a running burn's on-disk artifacts — history.jsonl, the safe_run status
receipt, and the milestones directory — and appends one typed JSONL row per
interval. The burn process is NEVER touched: no signals, no Metal, no imports
of trainer code. This recovers the step-rate CURVE (and milestone-blocking
dips) for runs whose trainer emits no per-step timing, without violating the
pinned-sources boundary of a sealed run.

Liveness reads ARTIFACTS, never buffered logs (the qbr1 run.log 0-byte lesson).
Detach pattern: launch with nohup + disown + pidfile; the loop exits on its own
when the DONE receipt appears or the history file goes stale past
--stale-exit-s (so no orphaned sampler outlives its burn by more than that).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _last_history_row(history_path: Path) -> dict:
    """Return the last JSON row of history.jsonl (empty dict if unreadable).

    Reads only the tail (last 64 KiB) so the poll cost stays flat as the
    history grows over a multi-hour burn.
    """
    try:
        with history_path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            tail = fh.read().decode("utf-8", errors="replace")
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        if not lines:
            return {}
        return json.loads(lines[-1])
    except (OSError, json.JSONDecodeError):
        return {}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def sample_once(args: argparse.Namespace) -> dict:
    now = datetime.now(timezone.utc)
    history = Path(args.history_jsonl)
    row = _last_history_row(history)
    status = _read_json(Path(args.safe_run_status)) if args.safe_run_status else {}
    milestones = sorted(
        d.name for d in Path(args.milestones_dir).iterdir() if d.is_dir()
    ) if args.milestones_dir and Path(args.milestones_dir).is_dir() else []
    try:
        history_mtime = history.stat().st_mtime
    except OSError:
        history_mtime = None
    return {
        "utc": now.isoformat(),
        "completed_steps": row.get("completed_steps"),
        "chunk_index": row.get("chunk_index"),
        "loss_total": (row.get("objective") or {}).get("loss_total"),
        "history_mtime_epoch": history_mtime,
        "safe_run_elapsed_s": status.get("elapsed_s"),
        "safe_run_peak_rss_mib": status.get("peak_rss_mib"),
        "safe_run_exit": status.get("exit"),
        "milestones": milestones,
        "milestones_count": len(milestones),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-jsonl", required=True,
                        help="the burn's history.jsonl (read-only)")
    parser.add_argument("--safe-run-status", default=None,
                        help="resource_safe_run_status.json (read-only)")
    parser.add_argument("--milestones-dir", default=None,
                        help="milestones directory (read-only)")
    parser.add_argument("--done-receipt", default=None,
                        help="exit cleanly once this file exists")
    parser.add_argument("--out-jsonl", required=True,
                        help="append-only output JSONL (typed rows)")
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--stale-exit-s", type=float, default=3600.0,
                        help="exit if history.jsonl mtime is older than this")
    parser.add_argument("--pidfile", default=None)
    parser.add_argument("--once", action="store_true",
                        help="emit one sample and exit (smoke/test mode)")
    args = parser.parse_args(argv)

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.pidfile:
        Path(args.pidfile).write_text(str(os.getpid()))

    missing_history_polls = 0
    try:
        while True:
            row = sample_once(args)
            with out.open("a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            if args.once:
                return 0
            if args.done_receipt and Path(args.done_receipt).exists():
                return 0
            mt = row.get("history_mtime_epoch")
            if mt is None:
                # A wrong path silently sampling nulls forever is the silent-
                # instrument bug; three consecutive misses is a typed exit.
                missing_history_polls += 1
                if missing_history_polls >= 3:
                    return 4
            else:
                missing_history_polls = 0
                if (time.time() - mt) > args.stale_exit_s:
                    return 3  # burn presumed dead/stopped; typed, not silent
            time.sleep(args.interval_s)
    finally:
        if args.pidfile:
            Path(args.pidfile).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
