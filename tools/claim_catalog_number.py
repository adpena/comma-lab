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
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STATE_PATH = REPO_ROOT / ".omx/state/next_catalog_number.txt"
LOG_PATH = REPO_ROOT / ".omx/state/catalog-claim.log"
SERIALIZER_PATH = REPO_ROOT / "tools" / "subagent_commit_serializer.py"
LOCK_TIMEOUT_SECONDS = 30
DEFAULT_INITIAL_VALUE = 116  # First number after #115 (FIX-A-CUSTODY's renumber)


class CatalogStateCorruptError(RuntimeError):
    """Raised when the catalog-counter state file exists but is empty or malformed.

    OP-6 fail-closed protection (codex chunk 8 HIGH, 2026-05-15): pre-fix,
    ``claim_one()`` silently treated an empty state file as
    ``DEFAULT_INITIAL_VALUE`` and reissued numbers that had already been
    claimed. The pre-fix truncate+rewrite path could LEAVE the file empty
    if the process died between ``truncate`` and ``write+fsync``. The fix
    is twofold: (1) write atomically via temp+fsync+rename so the canonical
    file is never empty mid-write; (2) refuse to claim from an empty/
    malformed file so a corrupted state surface fails loudly instead of
    silently re-issuing claimed numbers.

    Operator recovery: inspect the corrupted file. If a sibling
    ``.recover.<utc>`` file exists from quarantine, prefer its contents.
    Otherwise, the highest already-claimed number can be recovered from
    ``.omx/state/catalog-claim.log`` (JSONL) or from the CLAUDE.md catalog
    table itself (max ``^[0-9]+\\.`` row). Once the correct value is known,
    use ``python tools/claim_catalog_number.py set --value <N+1>``.
    """


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as e:
        print(f"[claim-catalog] WARNING: log append failed: {e!r}", file=sys.stderr)


def _lock_path() -> Path:
    """Sibling lockfile path adjacent to STATE_PATH.

    Per CLAUDE.md catalog #131 sister discipline: the canonical pattern is
    ``fcntl.flock(LOCK_EX)`` on a SIBLING lockfile (not on the canonical
    state file itself). Locking on the canonical file is incompatible with
    ``os.replace``-based atomic writes because the rename swaps the inode
    underneath any open fd; a process holding a lock on the OLD inode
    no longer protects the NEW inode. The sibling-lockfile pattern lets
    writers atomically rename the canonical file while every locker still
    contends on the same lockfile inode.

    Mirrors the canonical pattern in
    ``tac.deploy.lightning.active_jobs_state.ACTIVE_JOBS_LOCK`` (catalog #131).
    """
    return STATE_PATH.with_suffix(STATE_PATH.suffix + ".lock")


def _ensure_initialized() -> None:
    """Create the state file with DEFAULT_INITIAL_VALUE if absent.

    The bootstrap write goes through the atomic ``_atomic_write_state``
    helper so even initialization is crash-safe (a process death mid-bootstrap
    leaves the canonical file absent rather than empty).
    """
    if STATE_PATH.exists():
        return
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_state(DEFAULT_INITIAL_VALUE)


def _atomic_write_state(value: int) -> None:
    """Write ``value`` to STATE_PATH atomically: temp+fsync+rename.

    Per codex chunk 8 HIGH (2026-05-15) + CLAUDE.md catalog #131 sister:
    the pre-fix ``seek(0); truncate(); write()`` pattern leaves a window
    where the canonical file is empty if the process dies between
    ``truncate`` and ``write+fsync``. Subsequent readers then either
    (a) see an empty file and silently fall back to ``DEFAULT_INITIAL_VALUE``
    (the original bug, reissuing claimed numbers) OR (b) raise
    ``CatalogStateCorruptError`` (the new fail-closed contract).

    The atomic pattern eliminates window (a) entirely: the canonical file
    transitions from "old contents" to "new contents" in a single
    ``os.replace`` syscall — no intermediate empty state is observable.
    The temp file is fsync'd before rename so the new contents are durable
    on disk before the rename publishes them.

    Mirrors ``tac.deploy.lightning.active_jobs_state._save_active_jobs``
    (catalog #140 sister).

    The CALLER is responsible for holding the sibling-lockfile fcntl
    lock; this helper does NOT acquire it. Concurrent atomic writes
    without serialization would still race at the rename step (last
    rename wins), losing increments.
    """
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{value}\n"
    tmp = STATE_PATH.with_suffix(STATE_PATH.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        # Write payload to temp file, then fsync the temp fd so the bytes
        # are durable on disk BEFORE the rename publishes them. Without
        # the fsync, a power-loss between write and rename could leave
        # the renamed-into-place file with stale or zero content depending
        # on the filesystem's metadata-vs-data ordering.
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_PATH)
    finally:
        # Best-effort cleanup if we never reached os.replace (e.g., write
        # failed before fsync, or fsync raised). The os.replace path
        # already removed the tmp inode by atomically swapping it into
        # STATE_PATH, so tmp.exists() is False on the success path.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _read_state_strict() -> int:
    """Read the catalog counter or raise ``CatalogStateCorruptError``.

    OP-6 fail-closed contract (codex chunk 8 HIGH, 2026-05-15): if the
    state file exists but is empty or non-integer, refuse to return
    a fallback value. The pre-fix ``int(text) if text else DEFAULT_INITIAL_VALUE``
    silently re-issued claimed numbers when the file was empty (e.g., due
    to the truncate-mid-write crash window the atomic-write fix closes).

    The caller MUST hold the sibling-lockfile lock when invoking this
    helper as part of a claim cycle, so the read+validate+write sequence
    is atomic w.r.t. concurrent claimants. Standalone read-only consumers
    (``peek``) may invoke without the lock — they observe a consistent
    snapshot because writers use ``os.replace``.

    Returns ``DEFAULT_INITIAL_VALUE`` only when the canonical file does
    not exist (the bootstrap case is normal and not corruption).

    Raises:
        CatalogStateCorruptError: when the file exists and is empty,
            whitespace-only, or cannot parse as an integer.
    """
    if not STATE_PATH.exists():
        return DEFAULT_INITIAL_VALUE
    try:
        raw = STATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogStateCorruptError(
            f"catalog state file at {STATE_PATH} could not be read: {exc!r}. "
            "OP-6 fail-closed: refusing to claim from unreadable state to "
            "avoid re-issuing claimed numbers. Operator: inspect file "
            "permissions / filesystem health, then `python "
            "tools/claim_catalog_number.py set --value <N>` once the "
            "highest already-claimed number is known."
        ) from exc
    text = raw.strip()
    if not text:
        raise CatalogStateCorruptError(
            f"catalog state file at {STATE_PATH} exists but is empty. "
            "This indicates a crash during a pre-OP-6 truncate+rewrite "
            "(now structurally extincted via atomic temp+fsync+rename), "
            "OR explicit external truncation. OP-6 fail-closed: refusing "
            "to fall back to DEFAULT_INITIAL_VALUE because doing so "
            "silently re-issues numbers already claimed. Operator: "
            "recover the highest claimed number from "
            ".omx/state/catalog-claim.log (JSONL) or from CLAUDE.md "
            "catalog table (max ^[0-9]+\\. row), then run `python "
            "tools/claim_catalog_number.py set --value <N+1>`."
        )
    try:
        return int(text)
    except ValueError as exc:
        raise CatalogStateCorruptError(
            f"catalog state file at {STATE_PATH} contains non-integer "
            f"content {text!r}: {exc!r}. OP-6 fail-closed: refusing to "
            "claim from malformed state. Operator: recover the correct "
            "value as documented in CatalogStateCorruptError docstring."
        ) from exc


def _acquire_lock(timeout_seconds: int):
    """Acquire fcntl exclusive lock on the SIBLING lockfile.

    OP-6 hardening (codex chunk 8 HIGH, 2026-05-15): switched from
    locking the canonical state file directly to locking a sibling
    ``.lock`` file. This is required for the atomic-write fix because
    ``os.replace`` swaps the canonical file's inode; an fd holding a
    lock on the old inode does not arbitrate against writers locking
    the new inode. The sibling-lockfile pattern (canonical per CLAUDE.md
    catalog #131 + Lightning's ``ACTIVE_JOBS_LOCK``) guarantees every
    locker contends on a stable inode that is never renamed.

    Returns the open file handle so callers can release via
    ``_release_lock``. Backwards-compatible API: the prior contract
    "acquire returns a file-like with ``.fileno()``" is preserved.
    """
    lock_path = _lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Open with O_RDWR | O_CREAT so the lockfile is auto-created if absent.
    # The lockfile's contents are NEVER read or written; only its inode
    # is used for fcntl arbitration. Opening for "r+" via Python's open()
    # would fail if the file doesn't exist; using "a+" ensures creation
    # while keeping a normal Python file handle (so .fileno() works for
    # tests that mock the lock acquisition).
    fh = open(lock_path, "a+")  # noqa: SIM115 - caller owns lock lifetime.
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.close()
                raise TimeoutError(
                    f"Could not acquire {lock_path} within {timeout_seconds}s"
                ) from None
            time.sleep(0.05)


def _release_lock(fh) -> None:
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def claim_one() -> int:
    """Atomic: return current N, increment file to N+1.

    Per OP-6 hardening (codex chunk 8 HIGH, 2026-05-15) + CLAUDE.md
    catalog #131 sister discipline:

    1. Acquire fcntl exclusive lock on the SIBLING lockfile (so concurrent
       claimants serialize) — NOT on the canonical state file (whose inode
       is swapped by ``os.replace``).
    2. Read current N via ``_read_state_strict`` (fail-closed on
       empty/malformed; bootstrap-only fallback to ``DEFAULT_INITIAL_VALUE``
       when the canonical file does not exist).
    3. Write N+1 atomically via ``_atomic_write_state`` (temp file +
       fsync + ``os.replace``) — eliminates the truncate-mid-write
       window that pre-fix could leave the canonical file empty.
    4. Append a ``catalog-claim.log`` JSONL audit row.
    5. Release the lock.

    Crash safety: a crash AT ANY point in steps 1-4 leaves the canonical
    file in a consistent state — either with the OLD value (if crash
    happened before ``os.replace``) or the NEW value (if crash happened
    after). The empty-file / partial-write window the pre-fix had is
    structurally extincted.
    """
    _ensure_initialized()
    fh = _acquire_lock(LOCK_TIMEOUT_SECONDS)
    try:
        n = _read_state_strict()
        # Atomic publish of n+1 via temp+fsync+rename. The CALLER
        # (us, here) holds the sibling-lockfile lock for the duration.
        _atomic_write_state(n + 1)
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
    """Return the current next-available catalog number without claiming.

    Read-only; safe to call without the sibling-lockfile lock because
    writers use ``os.replace`` (every individual read sees either the
    pre-write or post-write state, never an intermediate empty file).

    Raises ``CatalogStateCorruptError`` if the file exists but is empty
    or malformed (per OP-6 fail-closed contract — silent fallback to
    ``DEFAULT_INITIAL_VALUE`` would mislead operators inspecting the
    counter into believing the canonical state is intact).
    """
    _ensure_initialized()
    return _read_state_strict()


def set_value(value: int) -> None:
    """Set the catalog counter to a specific value (operator only).

    Atomic write via ``_atomic_write_state`` (temp+fsync+rename) under
    the sibling-lockfile lock, mirroring ``claim_one``'s OP-6 hardening
    (2026-05-15). The pre-fix ``seek(0); truncate(); write()`` pattern
    had the same crash-window vulnerability as ``claim_one`` and is
    extincted via the same fix.
    """
    _ensure_initialized()
    fh = _acquire_lock(LOCK_TIMEOUT_SECONDS)
    try:
        _atomic_write_state(value)
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
