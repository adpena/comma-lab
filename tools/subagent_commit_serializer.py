#!/usr/bin/env python3
# no-argparse-OK: thin CLI wrapper that forwards positional args to git
"""Serialize concurrent subagent commits via a file lock.

Background — the bug class this prevents
─────────────────────────────────────────
Memory ref: feedback_concurrent_subagent_commit_message_swap_20260429.md.

When 2+ subagents reach `git commit` near-simultaneously:
- The first to acquire git's commit lock wins.
- The losing subagent sees the "other" agent's staging area in HEAD already
  (because git index is shared).
- Re-staging + re-committing creates a commit with the LOSER's body but
  contains the WINNER's files (because they were staged first).
- Or: pre-commit hooks (review-gate, preflight) fire on the COMBINED staged
  set, blocking both, and they retry interleaved.

Net effect: code lands intact; commit attribution is shuffled. Forensic
recovery requires `git show <commit> --stat` + grep for the source file.

This wrapper closes the race by serializing every subagent commit through a
single fcntl.flock(LOCK_EX) on .omx/state/.commit-lock. Inside the lock the
wrapper performs `git add <files>` then `git commit -m <msg>` so the index
+ HEAD update is atomic w.r.t. other concurrent invocations.

Usage
─────
From a subagent:

    python tools/subagent_commit_serializer.py \\
        --message "Lane PD-V2: arithmetic-coded pose deltas — 16/16 tests" \\
        --files src/tac/pose_delta_codec_v2.py src/tac/tests/test_pose_delta_codec_v2.py

Or with stdin for message + files-from-stdin:

    python tools/subagent_commit_serializer.py --message "..." --stdin-files <<EOF
    src/tac/foo.py
    src/tac/tests/test_foo.py
    EOF

Behaviour
─────────
1. Acquires fcntl.flock(LOCK_EX) on .omx/state/.commit-lock (blocking; with
   --timeout-seconds N, raises after N seconds of waiting).
2. Logs the attempt (PID, label, files, timestamp, msg head) to
   .omx/state/commit-serializer.log for forensics.
3. Runs `git add -- <files>` (NOT `git add -A`/`git add .`; the wrapper
   refuses to stage files NOT explicitly named, per CLAUDE.md "git add
   specific files by name").
4. Runs `git commit -m <message>`. The pre-commit hook (preflight + review
   gate) runs as usual — IF it fails, the wrapper releases the lock and
   exits non-zero, and the next waiter proceeds.
5. Releases the lock.

Cooperators
───────────
- The lock is FILESYSTEM-ADVISORY: anyone who calls `git commit` directly
  (without going through this wrapper) bypasses the lock. Subagent prompts
  must instruct subagents to use the wrapper.
- The lock is held for the duration of `git add` + `git commit` ONLY — not
  for the work itself. Subagents do their work in parallel; they only
  serialize at the moment of staging+commit.
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

# Repo root: tools/ lives one level under repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent

LOCK_PATH = REPO_ROOT / ".omx/state/.commit-lock"
LOG_PATH = REPO_ROOT / ".omx/state/commit-serializer.log"

# How many seconds to wait for the lock before giving up. The pre-commit
# hook runs full preflight (~5-10s) so 120s easily accommodates 5+ queued
# subagents.
DEFAULT_TIMEOUT_SECONDS = 120


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_log(record: dict) -> None:
    """Append-only JSONL log of every commit attempt."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        # Log failure must never block a commit. Print to stderr instead.
        print(f"[subagent-commit-serializer] WARNING: could not append "
              f"to log {LOG_PATH}: {record!r}", file=sys.stderr)


def _acquire_lock(timeout_seconds: int) -> "io.IOBase":
    """Acquire LOCK_EX on .commit-lock with a soft timeout.

    Returns the open file handle (caller must keep it open until release).
    Raises TimeoutError if the lock can't be acquired within timeout.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.touch(exist_ok=True)
    fh = open(LOCK_PATH, "w")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.close()
                raise TimeoutError(
                    f"Could not acquire {LOCK_PATH} within {timeout_seconds}s "
                    f"— another subagent's commit hook is taking unusually "
                    f"long. Inspect: tail .omx/state/commit-serializer.log"
                )
            # Brief backoff so we don't spin at 100% CPU.
            time.sleep(0.25)


def _release_lock(fh) -> None:
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def _git_add(files: list[str]) -> tuple[int, str]:
    """Run `git add -- <files>` and return (rc, output)."""
    if not files:
        return 0, "(no files)"
    # NEVER use `git add -A` / `git add .` — per CLAUDE.md "Always commit
    # specific files by name" (sensitive-file leakage prevention).
    cmd = ["git", "add", "--"] + files
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _git_commit(message: str, allow_empty: bool = False) -> tuple[int, str, str]:
    """Run `git commit -m <message>` and return (rc, stdout, stderr).

    The pre-commit hook (preflight + review gate) runs here as usual.
    """
    cmd = ["git", "commit", "-m", message]
    if allow_empty:
        cmd.append("--allow-empty")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _git_head_sha() -> str | None:
    """Best-effort: return the current HEAD SHA short form."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return proc.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--message", "-m", required=True,
        help="Commit message (passed to `git commit -m`).",
    )
    parser.add_argument(
        "--files", "-f", nargs="*", default=None,
        help="Files to stage. Required UNLESS --stdin-files OR "
             "--no-stage is passed.",
    )
    parser.add_argument(
        "--stdin-files", action="store_true",
        help="Read newline-separated filenames from stdin (in addition to "
             "any --files).",
    )
    parser.add_argument(
        "--no-stage", action="store_true",
        help="Skip `git add` — assume files are already staged. Use only "
             "when the caller has done its own `git add` AND knows the "
             "concurrency window is safe.",
    )
    parser.add_argument(
        "--allow-empty", action="store_true",
        help="Pass --allow-empty to `git commit`.",
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Max seconds to wait for the lock. "
             f"Default {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--label", default=os.environ.get("SUBAGENT_LABEL", "anonymous"),
        help="Subagent label for log forensics (default: $SUBAGENT_LABEL or "
             "'anonymous').",
    )
    args = parser.parse_args()

    # Resolve file list
    files: list[str] = list(args.files or [])
    if args.stdin_files:
        for line in sys.stdin:
            line = line.strip()
            if line:
                files.append(line)

    if not args.no_stage and not files:
        parser.error(
            "must pass --files or --stdin-files (or --no-stage if files are "
            "already staged)"
        )

    started_iso = _now_iso()
    pid = os.getpid()
    host = socket.gethostname()

    base_record = {
        "started_at_utc": started_iso,
        "pid": pid,
        "host": host,
        "label": args.label,
        "files": files,
        "message_head": args.message.splitlines()[0][:160],
        "no_stage": bool(args.no_stage),
    }

    # Acquire lock
    t0 = time.monotonic()
    try:
        lock_fh = _acquire_lock(args.timeout_seconds)
    except TimeoutError as e:
        _append_log({**base_record, "outcome": "lock_timeout",
                     "error": str(e), "wait_seconds": args.timeout_seconds})
        print(f"[subagent-commit-serializer] FATAL: {e!s}", file=sys.stderr)
        return 2
    wait_seconds = round(time.monotonic() - t0, 3)

    try:
        # Step 1: stage
        if not args.no_stage:
            rc, msg = _git_add(files)
            if rc != 0:
                _append_log({**base_record, "outcome": "git_add_failed",
                             "wait_seconds": wait_seconds,
                             "git_add_rc": rc, "git_add_output": msg})
                print(f"[subagent-commit-serializer] git add failed (rc={rc}):\n{msg}",
                      file=sys.stderr)
                return rc

        # Step 2: commit (pre-commit hook fires here)
        commit_t0 = time.monotonic()
        rc, stdout, stderr = _git_commit(args.message, allow_empty=args.allow_empty)
        commit_seconds = round(time.monotonic() - commit_t0, 3)

        head_after = _git_head_sha()
        outcome = "committed" if rc == 0 else "commit_failed"
        _append_log({
            **base_record,
            "outcome": outcome,
            "wait_seconds": wait_seconds,
            "commit_seconds": commit_seconds,
            "commit_rc": rc,
            "head_after": head_after,
            "stdout_tail": (stdout or "")[-200:],
            "stderr_tail": (stderr or "")[-200:],
        })

        # Surface git output to caller stderr/stdout.
        if stdout:
            sys.stdout.write(stdout)
        if stderr:
            sys.stderr.write(stderr)

        if rc != 0:
            print(f"[subagent-commit-serializer] git commit failed (rc={rc}). "
                  f"Lock released; next waiter (if any) will proceed.",
                  file=sys.stderr)
            return rc

        print(f"[subagent-commit-serializer] OK head={head_after} "
              f"label={args.label} files={len(files)} "
              f"wait={wait_seconds}s commit={commit_seconds}s",
              file=sys.stderr)
        return 0
    finally:
        _release_lock(lock_fh)


if __name__ == "__main__":
    sys.exit(main())
