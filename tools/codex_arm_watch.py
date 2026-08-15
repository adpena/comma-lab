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
import datetime as dt
import hashlib
import json
import os
import signal
import stat
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
DONE_RECEIPT_SCHEMA = "detached_local_process_done.v2"
CONSUMED_RECEIPT_SCHEMA = "detached_local_process_done_consumed.v1"


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


def _structured_terminal(receipt: str) -> tuple[str, str, bool] | None:
    """Parse the additive watched-launch receipt while preserving legacy text."""

    try:
        payload = json.loads(receipt)
    except json.JSONDecodeError:
        kind, line = _terminal_kind(receipt)
        return kind, line, False
    if not isinstance(payload, dict) or payload.get("schema") != DONE_RECEIPT_SCHEMA:
        kind, line = _terminal_kind(receipt)
        return kind, line, False
    try:
        rc = int(payload["rc"])
        elapsed = float(payload["elapsed_s"])
        launch_id = payload["launch_id"]
        counter = int(launch_id["monotonic_launch_counter"])
        pid = int(launch_id["pid"])
        manifest = str(launch_id["manifest_path"])
    except (KeyError, TypeError, ValueError):
        return "ALERT", "malformed structured done receipt", False
    terminal = (
        f"rc={rc} elapsed={int(elapsed)} launch_counter={counter} "
        f"launch_pid={pid} manifest={manifest}"
    )
    return ("FINISHED" if rc == 0 else "ALERT"), terminal, bool(
        payload.get("adjudicated_at_launch", False)
    )


def _mark_consumed(path: Path, *, suppressed: bool) -> bool:
    """Acknowledge only after the real monitor read/delivery path handled it."""

    try:
        data = path.read_bytes()
    except OSError:
        return False
    marker = path.with_name(path.name + ".consumed.json")
    payload = {
        "schema": CONSUMED_RECEIPT_SCHEMA,
        "consumed_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "receipt_path": str(path),
        "receipt_sha256": hashlib.sha256(data).hexdigest(),
        "consumer": "tools/codex_arm_watch.py",
        "suppressed_adjudicated_at_launch": bool(suppressed),
    }
    tmp = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(marker)
    return True


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
            parsed = _structured_terminal(content)
            if parsed is None:
                continue
            kind, terminal, adjudicated_at_launch = parsed
            if adjudicated_at_launch:
                # launch_detached_process already reported this failure
                # synchronously with log tail; re-alerting is the measured D2
                # double-alert class.
                continue
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


def stdout_delivery_channel() -> str:
    """Classify where our stdout GOES — the thing liveness never measured.

    A refreshed heartbeat proves this process is polling; it does NOT prove the
    events reach MAIN. Measured 2026-08-08: a detached watcher held the heartbeat
    at 0s all session (status read ALIVE/green) while MAIN received zero
    notifications, because its stdout was a file, not the Monitor's pipe. That is
    the vacuity genus — a green indicator for a condition nobody checked.

    DELIVERING_CHANNELS = {"socket", "fifo"}: an IPC channel a parent is consuming.
    MEASURED 2026-08-08 against the REAL consumer: the harness Monitor hands the
    child a SOCKETPAIR, not a pipe — a first patch that accepted only "fifo" raised
    a false NOT-DELIVERING alarm on the correct configuration, because its control
    used `| cat` (a stand-in) instead of the actual Monitor. A false alarm on the
    good state is worse than no alarm: it trains the reader to ignore the light.
    Controls for this classifier MUST exercise the real consumer.
    """
    try:
        mode = os.fstat(1).st_mode
    except OSError:
        return "closed"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"   # what the harness Monitor actually hands us
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISCHR(mode):
        return "tty"
    return "other"


def _write_heartbeat() -> None:
    """Heartbeat carries the DELIVERY CHANNEL, not just an mtime."""
    payload = f"channel={stdout_delivery_channel()}\npid={os.getpid()}\n"
    tmp = HEARTBEAT.with_suffix(HEARTBEAT.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(HEARTBEAT)  # atomic; readers never see a torn heartbeat


def watch(runs: Path, interval_s: float, once: bool) -> int:
    runs.mkdir(parents=True, exist_ok=True)
    before = _snapshot(runs)  # baseline: pre-existing receipts stay silent
    while True:
        delivery_channel = stdout_delivery_channel()
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        _write_heartbeat()
        time.sleep(interval_s)
        after = _snapshot(runs)
        for line in format_events(runs, before, after):
            print(line, flush=True)
        # Consumption happens only after formatting and stdout delivery.  A
        # broken monitor pipe kills this process before the acknowledgement,
        # so the launcher will refuse accidental receipt-name reuse.
        for name in sorted(after):
            if not name.endswith(".done") or before.get(name) == after[name]:
                continue
            if delivery_channel not in {"socket", "fifo"}:
                # A manual/file-backed watcher may inspect or log the event,
                # but only the harness-delivering channel may acknowledge it.
                continue
            path = runs / name
            try:
                parsed = _structured_terminal(path.read_text(errors="replace").strip())
            except OSError:
                continue
            suppressed = bool(parsed and parsed[2])
            _mark_consumed(path, suppressed=suppressed)
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
