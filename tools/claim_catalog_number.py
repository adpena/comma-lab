#!/usr/bin/env python3
"""Atomically claim the next CLAUDE.md catalog number.

Background — the bug class this prevents
─────────────────────────────────────────
2026-05-08 hardening burst hit empirical Catalog #N collision: subagent
FIX-A-CUSTODY (commit ``fa604f72``) and subagent FIX-A-SYNTH (commit
``c80162e7``) both grabbed Catalog #114 because they read CLAUDE.md
concurrently to find "next available" without any coordination. Sister
fork manually renumbered after-the-fact at ``000089d1``.

This tool closes the race by serializing every catalog-number claim
through a single ``fcntl.flock(LOCK_EX)`` on
``.omx/state/next_catalog_number.txt``. The state file holds an integer
N; readers atomically read N, write N+1, return N. Concurrent claims
serialize at the lock; each claimant gets a monotonically-increasing
unique number.

Usage
─────

From a subagent that's about to land a new STRICT preflight gate:

    NEXT=$(python tools/claim_catalog_number.py claim)
    echo "Claimed catalog #$NEXT"
    # Now write the entry into CLAUDE.md catalog at "$NEXT.":

To peek without claiming (e.g. for debugging):

    python tools/claim_catalog_number.py peek

To initialize/reset (operator only):

    python tools/claim_catalog_number.py set --value 117

The state file is COMMITTED (under ``.omx/state/``) so all worktrees and
subagents share the canonical counter. Locks are per-machine; the
git-side reconciliation happens via the commit serializer + a preflight
gate (``check_claude_md_catalog_no_duplicate_numbers``) that refuses any
CLAUDE.md with duplicate ``^[0-9]+\\.`` prefixes.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STATE_PATH = REPO_ROOT / ".omx/state/next_catalog_number.txt"
LOG_PATH = REPO_ROOT / ".omx/state/catalog-claim.log"
LOCK_TIMEOUT_SECONDS = 30
DEFAULT_INITIAL_VALUE = 116  # First number after #115 (FIX-A-CUSTODY's renumber)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as e:
        print(f"[claim-catalog] WARNING: log append failed: {e!r}", file=sys.stderr)


def _ensure_initialized() -> None:
    """Create the state file with DEFAULT_INITIAL_VALUE if absent."""
    if STATE_PATH.exists():
        return
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(f"{DEFAULT_INITIAL_VALUE}\n")


def _acquire_lock(timeout_seconds: int):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.touch(exist_ok=True)
    fh = open(STATE_PATH, "r+")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.close()
                raise TimeoutError(
                    f"Could not acquire {STATE_PATH} within {timeout_seconds}s"
                )
            time.sleep(0.05)


def _release_lock(fh) -> None:
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def claim_one() -> int:
    """Atomic: return current N, increment file to N+1."""
    _ensure_initialized()
    fh = _acquire_lock(LOCK_TIMEOUT_SECONDS)
    try:
        fh.seek(0)
        text = fh.read().strip()
        n = int(text) if text else DEFAULT_INITIAL_VALUE
        # Write n+1 back
        fh.seek(0)
        fh.truncate()
        fh.write(f"{n + 1}\n")
        fh.flush()
        os.fsync(fh.fileno())
        _append_log({
            "ts": _now_iso(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "claimed": n,
            "next_will_be": n + 1,
        })
        return n
    finally:
        _release_lock(fh)


def peek() -> int:
    _ensure_initialized()
    text = STATE_PATH.read_text().strip()
    return int(text) if text else DEFAULT_INITIAL_VALUE


def set_value(value: int) -> None:
    _ensure_initialized()
    fh = _acquire_lock(LOCK_TIMEOUT_SECONDS)
    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"{value}\n")
        fh.flush()
        os.fsync(fh.fileno())
        _append_log({
            "ts": _now_iso(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "set_value": value,
        })
    finally:
        _release_lock(fh)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("claim", help="Atomically claim and return next catalog number")
    sub.add_parser("peek", help="Return current next-available number without claiming")
    set_p = sub.add_parser("set", help="Set the counter to a specific value (operator only)")
    set_p.add_argument("--value", type=int, required=True)
    args = parser.parse_args()
    if args.cmd == "claim":
        n = claim_one()
        print(n)
        return 0
    if args.cmd == "peek":
        n = peek()
        print(n)
        return 0
    if args.cmd == "set":
        set_value(args.value)
        print(f"Set next catalog number to {args.value}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
