#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Atomically claim the next CLAUDE.md catalog number.

Background - the bug class this prevents
----------------------------------------
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

CANON-1.E hardening (2026-05-12)
--------------------------------
2026-05-08 collision-at-#158 (DDDD vs FFFF Bug 1) revealed a residual
gap: the fcntl lock arbitrates concurrent claims WITHIN the working
tree, but a mid-session ``git reset`` (or ``git checkout``) that rolls
back ``.omx/state/next_catalog_number.txt`` allows a SECOND claimant to
re-claim a number a FIRST claimant already used. Fix: optional
``--commit-via-serializer`` flag immediately commits the increment via
``tools/subagent_commit_serializer.py`` so the claim is git-transactional
- a later working-tree reset CANNOT erase a claim already in HEAD.

Usage with the new flag::

    NEXT=$(python tools/claim_catalog_number.py claim \\
        --commit-via-serializer \\
        --reason "WAVE-X: claiming for new STRICT preflight gate")
    echo "Claimed catalog #$NEXT"

The flag is OPT-IN to preserve back-compat. Subagents claiming numbers
in fan-out waves SHOULD use the flag; one-off operator claims may skip
it. The flag will become the default in a follow-up if the bug class
re-surfaces empirically. Per CLAUDE.md "Bugs must be permanently fixed
AND self-protected against", the future STRICT gate
``check_catalog_claim_committed_via_serializer`` (planned, deferred -
requires preflight.py edit when no in-flight subagent has it dirty)
will REFUSE bare ``claim`` invocations from subagents in the fan-out
context once the empirical collision recurs.

Usage
-----

From a subagent that's about to land a new STRICT preflight gate:

    NEXT=$(python tools/claim_catalog_number.py claim)
    echo "Claimed catalog #$NEXT"
    # Now write the entry into CLAUDE.md catalog at "$NEXT.":

With git-transactional commit (CANON-1.E hardening):

    NEXT=$(python tools/claim_catalog_number.py claim \\
        --commit-via-serializer --reason "WAVE-X: gate <name>")

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
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STATE_PATH = REPO_ROOT / ".omx/state/next_catalog_number.txt"
LOG_PATH = REPO_ROOT / ".omx/state/catalog-claim.log"
SERIALIZER_PATH = REPO_ROOT / "tools" / "subagent_commit_serializer.py"
LOCK_TIMEOUT_SECONDS = 30
DEFAULT_INITIAL_VALUE = 116  # First number after #115 (FIX-A-CUSTODY's renumber)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    fh = open(STATE_PATH, "r+")  # noqa: SIM115 - caller owns lock lifetime.
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
                ) from None
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


def _commit_state_via_serializer(claimed_n: int, reason: str) -> None:
    """Commit ``.omx/state/next_catalog_number.txt`` via the canonical
    subagent commit serializer so the increment is git-transactional.

    Per CANON-1.E hardening (2026-05-12): the fcntl-locked write to the
    state file is durable on disk but a subsequent ``git reset`` /
    ``git checkout`` can roll it back, leaving a window where a sibling
    subagent re-claims the same number. Routing through the serializer
    captures the increment in a HEAD commit immediately, making it
    impossible for a later working-tree reset to erase the claim.

    This function is intentionally tolerant: if the serializer is missing
    OR the commit fails (e.g., review-gate refuses the commit), we
    raise a RuntimeError with the underlying stderr so the caller can
    decide whether to roll back the in-memory claim. The state file
    increment is NOT rolled back automatically - the caller is
    responsible for that decision because rolling back would re-introduce
    the original race window.
    """
    if not SERIALIZER_PATH.exists():
        raise RuntimeError(
            f"subagent commit serializer not found at {SERIALIZER_PATH}; "
            "cannot make claim git-transactional. Use plain `claim` or "
            "install the serializer first."
        )
    rel_state_path = STATE_PATH.relative_to(REPO_ROOT)
    message = (
        f"state: claim catalog #{claimed_n} (git-transactional)\n\n"
        f"reason: {reason}\n\n"
        "Committed via tools/claim_catalog_number.py --commit-via-serializer "
        "to make the claim git-transactional per CANON-1.E hardening "
        "(.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md). "
        "A later working-tree reset cannot erase a claim already in HEAD.\n\n"
        "# CHECKPOINT_DISCIPLINE_WAIVED:catalog-claim helper, single-line "
        "state file mutation, no in-flight subagent context to checkpoint."
    )
    # Compute the working-tree sha so the serializer's --expected-content-sha256
    # gate refuses the commit if a sibling subagent has edited the state file
    # in our pre-edit window. This is the documented Catalog #157 contract.
    import hashlib
    state_bytes = STATE_PATH.read_bytes()
    state_sha = hashlib.sha256(state_bytes).hexdigest()
    cmd = [
        sys.executable,
        str(SERIALIZER_PATH),
        "--message",
        message,
        "--files",
        str(rel_state_path),
        "--expected-content-sha256",
        f"{rel_state_path}={state_sha}",
    ]
    env = dict(os.environ)
    # Catalog state files are non-code (`.txt` under `.omx/state/`); the
    # review-gate is designed for `.py` files only per CLAUDE.md "Review gate
    # - non-negotiable". Bypass is acceptable here per CLAUDE.md guidance.
    env["REVIEW_GATE_OVERRIDE"] = "1"
    proc = subprocess.run(
        cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"serializer commit failed (rc={proc.returncode}):\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    _append_log({
        "ts": _now_iso(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "claimed": claimed_n,
        "committed_via_serializer": True,
        "reason": reason,
    })


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
    claim_p = sub.add_parser(
        "claim", help="Atomically claim and return next catalog number"
    )
    claim_p.add_argument(
        "--commit-via-serializer",
        action="store_true",
        help=(
            "After atomic claim, commit the state-file increment via "
            "tools/subagent_commit_serializer.py so the claim is "
            "git-transactional. Per CANON-1.E hardening, a later "
            "git reset cannot erase a claim already in HEAD."
        ),
    )
    claim_p.add_argument(
        "--reason",
        type=str,
        default="",
        help=(
            "Required when --commit-via-serializer is set. Brief "
            "description of why the catalog number is being claimed; "
            "appears in the commit body for forensic audit."
        ),
    )
    sub.add_parser("peek", help="Return current next-available number without claiming")
    set_p = sub.add_parser("set", help="Set the counter to a specific value (operator only)")
    set_p.add_argument("--value", type=int, required=True)
    args = parser.parse_args()
    if args.cmd == "claim":
        if args.commit_via_serializer and not args.reason.strip():
            print(
                "ERROR: --commit-via-serializer requires --reason '<brief description>'",
                file=sys.stderr,
            )
            return 2
        n = claim_one()
        if args.commit_via_serializer:
            try:
                _commit_state_via_serializer(n, args.reason.strip())
            except RuntimeError as exc:
                # The on-disk increment IS durable; we surface the error
                # but do NOT roll back, because rolling back would
                # re-introduce the bug class CANON-1.E exists to fix.
                print(
                    f"WARNING: claim {n} succeeded on disk but git-transactional "
                    f"commit failed: {exc}",
                    file=sys.stderr,
                )
                print(
                    "Resolve the commit-side issue (e.g., review-gate, dirty "
                    "tree) and run `tools/subagent_commit_serializer.py` "
                    "manually to commit `.omx/state/next_catalog_number.txt`.",
                    file=sys.stderr,
                )
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
