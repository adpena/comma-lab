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
import hashlib
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

# Canonical Co-Authored-By trailer (FIX-3 2026-05-08).
CO_AUTHOR_TRAILER = (
    "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
)


def _hash_working_tree_files(files: list[str]) -> dict[str, str]:
    """SHA-256 each file's working-tree content (FIX-1 concurrent-edit detection).

    Used to detect a sister subagent modifying our intended-to-commit files
    between the moment we computed the pre-lock snapshot and the moment we
    acquire LOCK_EX. If the working-tree content of any file in our list
    changed during the lock-wait window, that's evidence that a concurrent
    subagent's edit is about to leak into our commit. Refuse rather than
    silently package someone else's changes under our authorship.

    Bug class: META-FIX subagent's `src/tac/preflight.py` edits flowed into
    sister FIX-5 commit `89d6eba2` because both subagents edited the file
    in the working tree concurrently. The temp-index isolates staging but
    `git add` reads the working tree, so concurrent working-tree edits can
    still leak. This hash check catches that leak.
    """
    out: dict[str, str] = {}
    for f in files:
        p = REPO_ROOT / f
        try:
            out[f] = hashlib.sha256(p.read_bytes()).hexdigest()
        except FileNotFoundError:
            out[f] = "MISSING"
        except OSError as e:
            out[f] = f"ERROR:{type(e).__name__}"
    return out


def _append_co_author_trailer(message: str) -> str:
    """Auto-append the canonical Co-Authored-By trailer (FIX-3 2026-05-08).

    Idempotent: if the trailer is already present, return unchanged. Otherwise
    append two newlines + the canonical trailer line. Subagents flagged this
    as a recurring miss across multiple commits this session (FIX-1 00896b43,
    FIX-3+4 c6d09bbb, FIX-5 89d6eba2, etc.).
    """
    if CO_AUTHOR_TRAILER in message:
        return message
    sep = "\n\n" if not message.endswith("\n") else "\n"
    return message + sep + CO_AUTHOR_TRAILER + "\n"

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


def _acquire_lock(timeout_seconds: int):
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


def _make_temp_index() -> tuple[str, dict]:
    """Create a per-invocation temp git index, seeded from HEAD.

    Returns (temp_index_path, env_dict) — env_dict is the subprocess env
    overlay that pins GIT_INDEX_FILE to the temp index. This isolates
    `git add` + `git commit` from the shared `.git/index` so a CONCURRENT
    subagent (or a manual `git add` from the user's shell) cannot inject
    files into our commit's staged set.

    Bug class fixed: 2026-04-29 PM — even with the file-lock serializer,
    Defect #1 from subagent #264 was absorbed into commit 22a2bcd2 (Lane
    Ω-W-V2 work) because subagent #263 staged AND committed files in the
    brief window before #264 acquired the lock — both sets of files were
    in the SHARED index when #263's commit fired. The temp-index
    isolation makes this impossible going forward.
    """
    tmp = REPO_ROOT / ".omx" / "state" / f".subagent-temp-index-{os.getpid()}-{int(time.time() * 1000)}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # Seed the temp index from HEAD (so `git add` adds modifications, not
    # everything already-tracked-and-unchanged).
    env = {**os.environ, "GIT_INDEX_FILE": str(tmp)}
    proc = subprocess.run(
        ["git", "read-tree", "HEAD"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git read-tree HEAD failed: rc={proc.returncode} "
            f"stderr={proc.stderr.strip()}"
        )
    return str(tmp), env


def _cleanup_temp_index(temp_index_path: str) -> None:
    """Remove the temp index file. Safe to call multiple times."""
    try:
        os.unlink(temp_index_path)
    except FileNotFoundError:
        pass


def _git_add(files: list[str], env: dict) -> tuple[int, str]:
    """Run `git add -- <files>` against env's GIT_INDEX_FILE."""
    if not files:
        return 0, "(no files)"
    # NEVER use `git add -A` / `git add .` — per CLAUDE.md "Always commit
    # specific files by name" (sensitive-file leakage prevention).
    cmd = ["git", "add", "--"] + files
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _git_commit(message: str, env: dict, allow_empty: bool = False) -> tuple[int, str, str]:
    """Run `git commit -m <message>` against env's GIT_INDEX_FILE.

    The pre-commit hook (preflight + review gate) runs here as usual; it
    inherits GIT_INDEX_FILE so `git diff --cached` calls inside the hook
    see ONLY this subagent's staged files, not anyone else's.
    """
    cmd = ["git", "commit", "-m", message]
    if allow_empty:
        cmd.append("--allow-empty")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
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
    parser.add_argument(
        "--no-co-author", action="store_true",
        help="Skip auto-appending the Co-Authored-By trailer. Use ONLY for "
             "human-authored commits or commits that intentionally have no "
             "Claude attribution. Default: trailer auto-appended (FIX-3).",
    )
    parser.add_argument(
        "--no-concurrent-edit-check", action="store_true",
        help="Skip the FIX-1 pre-lock vs post-lock content-hash mismatch "
             "check. Use ONLY when intentionally racing edits with a known "
             "sister subagent (rare); default: check enabled.",
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

    # FIX-1: snapshot working-tree content hashes BEFORE acquiring lock.
    # If any file's content changes between this moment and post-lock, a
    # concurrent subagent edited our intended-to-commit files and we refuse.
    pre_lock_hashes: dict[str, str] = {}
    if not args.no_concurrent_edit_check and not args.no_stage and files:
        pre_lock_hashes = _hash_working_tree_files(files)

    # Auto-append Co-Authored-By trailer (FIX-3, idempotent).
    final_message = (
        args.message if args.no_co_author
        else _append_co_author_trailer(args.message)
    )

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

    # FIX-1: re-hash under the lock and compare. If a sister subagent
    # modified our --files content during our lock-wait, refuse.
    if not args.no_concurrent_edit_check and not args.no_stage and files:
        post_lock_hashes = _hash_working_tree_files(files)
        diffs = {
            f: (pre_lock_hashes.get(f, "?"), post_lock_hashes.get(f, "?"))
            for f in files
            if pre_lock_hashes.get(f) != post_lock_hashes.get(f)
        }
        if diffs:
            _release_lock(lock_fh)
            _append_log({
                **base_record,
                "outcome": "concurrent_edit_detected",
                "wait_seconds": wait_seconds,
                "concurrent_edit_diffs": {
                    f: {"pre": pre, "post": post}
                    for f, (pre, post) in diffs.items()
                },
            })
            print(
                "[subagent-commit-serializer] REFUSED: concurrent-edit "
                "detected on these files between pre-lock and post-lock "
                "snapshot. A sister subagent edited our files during the "
                "lock-wait window. Re-stage and retry; do not silently "
                "package their changes under your commit. Files affected: "
                f"{list(diffs.keys())!r}",
                file=sys.stderr,
            )
            return 3

    temp_index_path: str | None = None
    try:
        # Per-invocation temp index — isolates our staging from any
        # concurrent subagent or manual `git add`. See _make_temp_index
        # docstring for the bug class this fixes.
        if args.no_stage:
            # Caller already staged into .git/index; we honor that.
            env = {**os.environ}
        else:
            temp_index_path, env = _make_temp_index()

        # Step 1: stage
        if not args.no_stage:
            rc, msg = _git_add(files, env)
            if rc != 0:
                _append_log({**base_record, "outcome": "git_add_failed",
                             "wait_seconds": wait_seconds,
                             "git_add_rc": rc, "git_add_output": msg,
                             "temp_index": temp_index_path})
                print(f"[subagent-commit-serializer] git add failed (rc={rc}):\n{msg}",
                      file=sys.stderr)
                return rc

        # Step 2: commit (pre-commit hook fires here, inherits GIT_INDEX_FILE)
        # final_message includes the FIX-3 Co-Authored-By trailer unless
        # --no-co-author was passed.
        commit_t0 = time.monotonic()
        rc, stdout, stderr = _git_commit(final_message, env, allow_empty=args.allow_empty)
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
            "temp_index": temp_index_path,
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
              f"wait={wait_seconds}s commit={commit_seconds}s "
              f"temp_index={'YES' if temp_index_path else 'NO (--no-stage)'}",
              file=sys.stderr)
        return 0
    finally:
        if temp_index_path:
            _cleanup_temp_index(temp_index_path)
        _release_lock(lock_fh)


if __name__ == "__main__":
    sys.exit(main())
