#!/usr/bin/env python3
"""Fleet watcher — one stdout line per codex-arm terminal/relay event.

Operator directive 2026-08-04: MAIN must RECEIVE arm-completion notifications
(like harness-tracked Claude subagents), never poll on request. This is the
canonical emitter; MAIN arms it as a persistent Monitor at session start —
each printed line becomes a conversation notification that re-invokes MAIN.

Events emitted (one line each, snapshot-baselined so history stays silent):
  ARM <name> FINISHED <rc=0...> [gen=N] elapsed=Ns | last: <final-msg head>
  ARM <name> ALERT <rc!=0|signal=..> [gen=N] elapsed=Ns | last: <final-msg head>
  ARM <name> RELAYED <relay line>            (context-exhaustion relay fired)

A heartbeat file (`_watcher.alive`, mtime-refreshed each poll) lets the
dispatcher (`codex_arm_queue.py status/saturate`) report watcher liveness, so
an unarmed watcher is visible at every queue interaction instead of silently
reintroducing the poll-on-request regime.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

# The session harness delivers SIGURG to long-lived children (~minutes cadence);
# an unhardened watcher dies rc=144 — measured 2026-08-04 on the first armed
# Monitor. Same kill class that felled bare codex spawns (CLAUDE.md pattern-A).
for _sig in ("SIGURG", "SIGPIPE"):
    try:
        signal.signal(getattr(signal, _sig), signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        pass

RUNS = Path(".omx/tmp/codex_runs")
HEARTBEAT = RUNS / "_watcher.alive"
HEARTBEAT_STALE_S = 90  # liveness bar used by the dispatcher


def _snapshot(runs: Path) -> dict[str, tuple[float, int]]:
    """(mtime, size) per watched file — .done terminal receipts + .relay logs."""
    snap: dict[str, tuple[float, int]] = {}
    for pat in ("*.done", "*.relay"):
        for f in runs.glob(pat):
            try:
                st = f.stat()
            except OSError:
                continue
            snap[f.name] = (st.st_mtime, st.st_size)
    return snap


def _final_msg_head(runs: Path, arm: str, limit: int = 140) -> str:
    last = runs / f"{arm}.last.txt"
    try:
        head = last.read_text(errors="replace").strip().replace("\n", " ")
        return head[:limit]
    except OSError:
        return "(no final message file)"


def _terminal_kind(receipt: str) -> tuple[str, str]:
    line = receipt.splitlines()[-1] if receipt else "(empty receipt)"
    if "signal=" in line:
        return "ALERT", line
    for token in line.replace(",", " ").split():
        if token.startswith("rc="):
            try:
                return ("FINISHED" if int(token.split("=", 1)[1]) == 0 else "ALERT", line)
            except ValueError:
                return "ALERT", line
    return "ALERT", line


def format_events(
    runs: Path, before: dict[str, tuple[float, int]], after: dict[str, tuple[float, int]]
) -> list[str]:
    """Pure event diff → notification lines (unit-tested)."""
    lines: list[str] = []
    for name in sorted(after):
        if before.get(name) == after[name]:
            continue
        arm = name.rsplit(".", 1)[0]
        path = runs / name
        try:
            content = path.read_text(errors="replace").strip()
        except OSError:
            continue
        if name.endswith(".done"):
            kind, terminal = _terminal_kind(content)
            lines.append(
                f"ARM {arm} {kind} {terminal}"
                f" | last: {_final_msg_head(runs, arm)}"
            )
        else:  # .relay — report only lines appended since the last poll
            prev_size = before.get(name, (0.0, 0))[1]
            new_part = content.encode()[prev_size:].decode(errors="replace").strip()
            for row in new_part.splitlines():
                if row.strip():
                    lines.append(f"ARM {arm} RELAYED {row.strip()}")
    return lines


def watch(runs: Path, interval_s: float, once: bool) -> int:
    runs.mkdir(parents=True, exist_ok=True)
    before = _snapshot(runs)  # baseline: pre-existing receipts stay silent
    while True:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.touch()
        time.sleep(interval_s)
        after = _snapshot(runs)
        for line in format_events(runs, before, after):
            print(line, flush=True)
        before = after
        if once:
            return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=RUNS)
    p.add_argument("--interval-s", type=float, default=10.0)
    p.add_argument("--once", action="store_true", help="One poll cycle then exit (testing).")
    a = p.parse_args(argv)
    global HEARTBEAT
    HEARTBEAT = a.runs_dir / "_watcher.alive"
    return watch(a.runs_dir, a.interval_s, a.once)


if __name__ == "__main__":
    raise SystemExit(main())
