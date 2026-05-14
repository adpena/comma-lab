#!/usr/bin/env python3
"""Subagent crash-resume checkpoint tool.

Background — the bug class this prevents
─────────────────────────────────────────
2026-05-14 operator directive: "why did it die? need to investigate and fix
permanently". Empirical anchor: the Anthropic API returned ``Internal server
error`` mid-subagent-session for a Wyner-Ziv research subagent (id
``a1362a24d986029c3``) that had completed 17 minutes of work / 58 tool uses /
1704 tokens. All in-flight progress was lost; the parent had to re-spawn from
scratch with no resume signal. A second incident this session
(WAVE-3-HNERV-C-RETRY pattern for DSNeRV + HiNeRV trainers) had the same
failure class and was only recoverable because the subagent had already
committed intermediate progress that the successor could grep for.

The bug class: long-running subagents accumulate non-trivial work in-context
(file edits, research synthesis, dispatch plans) that is invisible to the
parent until either (a) a commit lands, or (b) the subagent reports back.
When the API crashes mid-session, every uncommitted-and-unreported byte is
lost. The parent has no canonical place to look for "did predecessor X get
anywhere before crashing?".

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable: structural extinction requires (1) a canonical place to
checkpoint subagent progress + (2) a discipline for every long-running
subagent to write checkpoints + (3) a STRICT preflight gate that refuses
subagent commits without checkpoint traces.

This tool is layer 1: the canonical checkpoint store.

Schema
──────
Records are appended JSONL to ``.omx/state/subagent_progress.jsonl``. Each
record is one line of JSON with these fields::

    {
        "subagent_id": "<freeform string; conventionally the subagent's name>",
        "parent_id_or_session": "<optional parent session id>",
        "step": <integer 1..N or string 'complete'>,
        "status": <"in_progress" | "blocked" | "complete">,
        "files_touched": ["<repo-relative path>", ...],
        "next_action": "<one-line description of what comes next>",
        "notes": "<freeform>",
        "written_at_utc": "<ISO-8601 timestamp>",
        "pid": <integer>,
        "host": "<hostname>",
    }

Per CLAUDE.md Catalog #131 (``check_no_bare_writes_to_shared_state``):
every write acquires ``fcntl.flock(LOCK_EX)`` on ``.omx/state/.subagent_progress.lock``
so concurrent appends from sibling subagents serialize without lost rows.

Usage
─────
Write a checkpoint::

    .venv/bin/python tools/subagent_checkpoint.py \
        --subagent-id WAVE-7-FOO-SUBAGENT \
        --step 3 \
        --status in_progress \
        --files-touched src/tac/foo.py,src/tac/tests/test_foo.py \
        --next-action "wire foo() into preflight_all() and add 5 more tests" \
        --notes "completed 12 of estimated 25 tool uses"

Read latest checkpoints for a subagent::

    .venv/bin/python tools/subagent_checkpoint.py read \
        --subagent-id WAVE-7-FOO-SUBAGENT

Read raises ``SystemExit(2)`` if no records exist for the subagent (so
predecessor-resume scripts can branch on the rc).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import errno
import fcntl
import json
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / ".omx" / "state"
JSONL_PATH = STATE_DIR / "subagent_progress.jsonl"
LOCK_PATH = STATE_DIR / ".subagent_progress.lock"

# Lock acquisition timeout in seconds. A single append is fast (<10ms) so 30s
# is generous even under heavy fan-out contention.
LOCK_TIMEOUT_SECONDS = 30

VALID_STATUSES = ("in_progress", "blocked", "complete")


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _acquire_lock(timeout_seconds: int):
    """Open the lock file and acquire fcntl LOCK_EX with timeout.

    Returns an open file handle that the caller must close to release.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.touch(exist_ok=True)
    fh = open(LOCK_PATH, "r+")  # noqa: SIM115 - caller owns lock lifetime
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.close()
                raise TimeoutError(
                    f"could not acquire {LOCK_PATH} within {timeout_seconds}s"
                ) from None
            time.sleep(0.05)


def _release_lock(fh) -> None:
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def _validate_record(record: dict) -> None:
    """Sanity-check a record before append. Raises ValueError on bad input."""
    sid = record.get("subagent_id")
    if not isinstance(sid, str) or not sid.strip():
        raise ValueError("subagent_id must be a non-empty string")
    if any(c in sid for c in ("\n", "\t", "\x1f")):
        raise ValueError("subagent_id must not contain newlines/tabs/0x1f")

    status = record.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status must be one of {VALID_STATUSES!r}, got {status!r}"
        )

    step = record.get("step")
    if not (isinstance(step, int) or step == "complete"):
        raise ValueError(
            f"step must be int or the literal string 'complete', got {step!r}"
        )

    files = record.get("files_touched", [])
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise ValueError("files_touched must be a list of strings")

    next_action = record.get("next_action", "")
    if not isinstance(next_action, str):
        raise ValueError("next_action must be a string")

    notes = record.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")


def append_checkpoint(
    *,
    subagent_id: str,
    step: int | str,
    status: str,
    files_touched: list[str],
    next_action: str,
    notes: str = "",
    parent_id_or_session: str | None = None,
) -> dict:
    """Append a single checkpoint record under the fcntl lock.

    Returns the record as-written (including server-side fields like
    ``written_at_utc`` / ``pid`` / ``host``).
    """
    # Validate BEFORE the list() coercion below so callers passing a string
    # (or other non-list) for ``files_touched`` get a clear error rather than
    # silently being coerced to a list-of-characters.
    if not isinstance(files_touched, list) or not all(
        isinstance(f, str) for f in files_touched
    ):
        raise ValueError("files_touched must be a list of strings")
    record = {
        "subagent_id": subagent_id,
        "parent_id_or_session": parent_id_or_session,
        "step": step,
        "status": status,
        "files_touched": list(files_touched),
        "next_action": next_action,
        "notes": notes,
        "written_at_utc": _now_iso(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    _validate_record(record)

    fh = _acquire_lock(LOCK_TIMEOUT_SECONDS)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        # Append-only: open in 'a' mode under the lock so multiple appenders
        # don't truncate each other's content. The lock serializes appends so
        # the kernel's append-atomicity is irrelevant; this is belt + suspenders.
        with open(JSONL_PATH, "a") as out:
            out.write(json.dumps(record, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
    finally:
        _release_lock(fh)
    return record


def read_checkpoints(subagent_id: str | None = None) -> list[dict]:
    """Read all checkpoint records, optionally filtered to one subagent.

    Returns the records in the order they appear in the JSONL file (which is
    the order they were written under the lock).
    """
    if not JSONL_PATH.exists():
        return []
    rows: list[dict] = []
    with open(JSONL_PATH, "r") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if subagent_id is not None and rec.get("subagent_id") != subagent_id:
                continue
            rows.append(rec)
    return rows


def latest_checkpoint(subagent_id: str) -> dict | None:
    """Return the most recent record for ``subagent_id`` or None."""
    rows = read_checkpoints(subagent_id)
    if not rows:
        return None
    return rows[-1]


def _parse_files_touched(raw: str | None) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_step(raw: str) -> int | str:
    if raw == "complete":
        return "complete"
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(
            f"--step must be an integer or the literal string 'complete', "
            f"got {raw!r}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Subagent crash-resume checkpoint tool. Writes JSONL to "
            ".omx/state/subagent_progress.jsonl under fcntl lock per "
            "Catalog #131 bare-write discipline."
        )
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # 'read' subcommand
    read_p = subparsers.add_parser(
        "read",
        help="Read latest checkpoint(s) for a subagent.",
    )
    read_p.add_argument(
        "--subagent-id",
        required=True,
        help="Subagent id to look up.",
    )
    read_p.add_argument(
        "--latest-only",
        action="store_true",
        help="Print only the most recent record (default: all records).",
    )

    # Default (write) flags directly on the top-level parser
    parser.add_argument(
        "--subagent-id",
        help="Subagent id (required for writes).",
    )
    parser.add_argument(
        "--step",
        help="Step number (integer >=1) or the literal string 'complete'.",
    )
    parser.add_argument(
        "--status",
        choices=VALID_STATUSES,
        help=f"One of {VALID_STATUSES}.",
    )
    parser.add_argument(
        "--files-touched",
        default="",
        help="Comma-separated repo-relative file paths.",
    )
    parser.add_argument(
        "--next-action",
        default="",
        help="One-line description of the next planned action.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Freeform notes (multi-line allowed; will be JSON-escaped).",
    )
    parser.add_argument(
        "--parent-id-or-session",
        default=None,
        help="Optional parent subagent id or session anchor.",
    )

    args = parser.parse_args(argv)

    if args.subcommand == "read":
        records = read_checkpoints(args.subagent_id)
        if not records:
            print(
                f"[subagent-checkpoint] no records for subagent_id="
                f"{args.subagent_id!r}",
                file=sys.stderr,
            )
            return 2
        out = records[-1:] if args.latest_only else records
        for rec in out:
            print(json.dumps(rec, sort_keys=True))
        return 0

    # Default = write
    if not args.subagent_id:
        parser.error("--subagent-id is required for writes")
    if args.step is None:
        parser.error("--step is required for writes")
    if args.status is None:
        parser.error("--status is required for writes")

    step_val = _parse_step(args.step)
    files = _parse_files_touched(args.files_touched)

    record = append_checkpoint(
        subagent_id=args.subagent_id,
        step=step_val,
        status=args.status,
        files_touched=files,
        next_action=args.next_action,
        notes=args.notes,
        parent_id_or_session=args.parent_id_or_session,
    )
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
