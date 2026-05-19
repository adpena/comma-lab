#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
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

SCOPE OF PROTECTION (honest, per 2026-05-12 adversarial-review pass):
This serializer protects against concurrent commits ON A SINGLE MACHINE.
`fcntl.flock` is filesystem-local — multiple machines (e.g., one agent on
local + one agent on a Vast.ai/Modal/Lightning instance) running git operations
on COPIES of the repo do NOT coordinate via this lock. The bug class CLAUDE.md
describes ("2+ subagents commit near-simultaneously") happens at the
operating-system process layer, which fcntl covers; multi-machine git
coordination is a separate problem space requiring (e.g.) a network-side
serializer, push-with-lease semantics, or a distributed lock service. NOT
covered by this tool. Multi-machine subagents should still call this wrapper
on their local machine, then rely on git push-with-lease at sync time.

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

# Catalog #340 STAGING-surface absorption-prevention helper.
# Sister of Catalog #314 (POST-COMMIT detect) — together they extinct the
# bare-commit-absorbs-in-flight-files class bidirectionally.
# Imported BEFORE fcntl-lock acquisition so the guard runs early.
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    from tac.commit_safety import (  # noqa: E402
        check_files_against_sister_checkpoints,
        bare_override_attempted,
        parse_override_env,
    )
    from tac.commit_safety.sister_checkpoint_guard import (  # noqa: E402
        CorruptCheckpointError,
    )
    _CATALOG_340_HELPER_AVAILABLE = True
except ImportError:
    # Test fixtures may stand up a minimal repo without the package.
    _CATALOG_340_HELPER_AVAILABLE = False
    check_files_against_sister_checkpoints = None  # type: ignore[assignment]
    bare_override_attempted = None  # type: ignore[assignment]
    parse_override_env = None  # type: ignore[assignment]
    CorruptCheckpointError = RuntimeError  # type: ignore[misc, assignment]

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


def _parse_expected_content_sha256(arg_values: list[str]) -> dict[str, str]:
    """Parse ``--expected-content-sha256 <file>=<sha>`` flag values.

    Each value must be ``<relpath>=<64-hex>``. Returns a dict mapping
    relpath -> expected SHA-256. Empty list -> empty dict.

    Raises ValueError on malformed input.
    """
    out: dict[str, str] = {}
    for v in arg_values or []:
        if "=" not in v:
            raise ValueError(
                f"--expected-content-sha256 must be '<relpath>=<sha256>'; "
                f"got {v!r}"
            )
        path, _, sha = v.partition("=")
        path = path.strip()
        sha = sha.strip().lower()
        if not path or not sha:
            raise ValueError(
                f"--expected-content-sha256 has empty path or sha in {v!r}"
            )
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(
                f"--expected-content-sha256 sha must be 64 hex chars; "
                f"got {sha!r} for path {path!r}"
            )
        out[path] = sha
    return out


def _expected_content_sha256_check(
    expected: dict[str, str],
) -> dict[str, tuple[str, str]]:
    """FIX-92aba3ca (2026-05-12): pre-lock-vs-EXPECTED-content-sha256 check.

    The 92aba3ca commit-swap incident showed that the FIX-1 pre-lock vs
    post-lock check only catches edits during the lock-wait window. If
    TWO subagents have ALREADY edited the same file in the working tree
    BEFORE either takes its pre-lock snapshot, both subagents observe
    the merged content; both `pre==post` checks pass; the winning
    subagent's `git add <file>` packages BOTH edits.

    The structural fix: callers may pass ``--expected-content-sha256
    <file>=<sha>`` declaring what the file's content SHOULD be at the
    moment the subagent started its work. The serializer hashes the
    current working-tree content and refuses if it differs.

    Returns a dict of mismatches: ``{relpath: (expected_sha, actual_sha)}``.
    Returns an empty dict if every declared expectation matches. Callers
    that don't pass ``--expected-content-sha256`` get an empty expected
    dict and an empty mismatch dict (backward-compatible).
    """
    if not expected:
        return {}
    actual = _hash_working_tree_files(list(expected.keys()))
    diffs: dict[str, tuple[str, str]] = {}
    for path, want in expected.items():
        got = actual.get(path, "MISSING")
        if got != want:
            diffs[path] = (want, got)
    return diffs


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
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    fh = open(LOCK_PATH, "w")  # noqa: SIM115 - caller must hold the handle until explicit unlock.
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
                ) from None
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


def _refresh_real_index_after_temp_commit(files: list[str], repo_root: Path = REPO_ROOT) -> None:
    """Refresh the caller-visible index for files committed via a temp index.

    Alternate-index commits move ``HEAD`` but intentionally do not update the
    shared ``.git/index``. Without this refresh, a successful serialized commit
    can leave the user's real index stale and `git status` may report the just
    committed paths as still modified/staged. ``git reset -- <files>`` updates
    only the named index entries to the new ``HEAD`` while preserving the
    working tree.
    """
    if not files:
        return
    proc = subprocess.run(
        ["git", "reset", "-q", "HEAD", "--", *files],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "git reset real index after temp-index commit failed: "
            f"rc={proc.returncode} stderr={proc.stderr.strip()}"
        )


def _git_add(files: list[str], env: dict) -> tuple[int, str]:
    """Run `git add -- <files>` against env's GIT_INDEX_FILE."""
    if not files:
        return 0, "(no files)"
    # NEVER use `git add -A` / `git add .` — per CLAUDE.md "Always commit
    # specific files by name" (sensitive-file leakage prevention).
    cmd = ["git", "add", "--", *files]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _hash_staged_files(files: list[str], env: dict) -> dict[str, str]:
    """Catalog #216 (FIX-HARDEN-OPT 2026-05-14 P1).

    Return SHA-256 of each file's STAGED content (what's currently in the
    index, NOT the working tree). Uses `git cat-file --batch` on the blob
    OID resolved from `git ls-files --stage <file>`. Honors env's
    GIT_INDEX_FILE so this reads OUR temp index, not the real index.

    Used by Catalog #216's staged-content verification: catches the case
    where two subagents edited the same file in the working tree
    independently and BOTH took their pre-lock snapshot AFTER both edits
    were already present. The pre-lock + post-lock check sees the merged
    content as stable, so the loser silently absorbs the winner's edits.
    The new check verifies the STAGED content matches the caller's
    declared post-edit sha — only one subagent can declare the merged
    content; the other gets refused with rc=5 and must re-base.
    """
    out: dict[str, str] = {}
    for f in files:
        # Step 1: resolve blob OID from the index for this file.
        try:
            ls = subprocess.run(
                ["git", "ls-files", "--stage", "--", f],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                check=False,
            )
        except OSError as exc:
            out[f] = f"ERROR_LS_FILES:{type(exc).__name__}"
            continue
        if ls.returncode != 0 or not ls.stdout.strip():
            out[f] = "NOT_STAGED"
            continue
        # Format: "<mode> <oid> <stage>\t<path>"
        parts = ls.stdout.strip().split(maxsplit=2)
        if len(parts) < 2 or len(parts[1]) != 40:
            # Not a 40-char SHA-1 OID — index entry malformed.
            out[f] = f"ERROR_LS_FILES_PARSE:{ls.stdout.strip()[:80]}"
            continue
        blob_oid = parts[1]
        # Step 2: read blob content via `git cat-file blob`.
        try:
            cat = subprocess.run(
                ["git", "cat-file", "blob", blob_oid],
                cwd=REPO_ROOT, env=env, capture_output=True, check=False,
            )
        except OSError as exc:
            out[f] = f"ERROR_CAT_FILE:{type(exc).__name__}"
            continue
        if cat.returncode != 0:
            out[f] = f"ERROR_CAT_FILE_RC:{cat.returncode}"
            continue
        # SHA-256 the raw bytes (parity with _hash_working_tree_files).
        out[f] = hashlib.sha256(cat.stdout).hexdigest()
    return out


def _staged_content_check(
    expected: dict[str, str],
    env: dict,
) -> dict[str, tuple[str, str]]:
    """Catalog #216 (FIX-HARDEN-OPT 2026-05-14 P1).

    Verify each file's STAGED content sha matches what the caller declared
    via `--expected-content-sha256`. Returns `{relpath: (expected, actual)}`
    for any mismatch. Empty dict means all staged content matches.

    Bug class anchor: 2026-05-14 commit `5d0ec061d` (D4-OOM-FIX Catalog #218)
    absorbed FIX-HARDEN-OPT's Catalog #215 edits to `src/tac/preflight.py`.
    Both subagents edited the same file. D4-OOM-FIX's `git add` packaged
    BOTH edits under their commit body, and the subsequent FIX-HARDEN-OPT
    commit `f7df40f33` showed `preflight.py` as already-clean (their edits
    landed in the previous commit). The pre-lock + post-lock check
    (Catalog #157) saw stable content because BOTH edits were already in
    the working tree before either subagent took its pre-lock snapshot.

    The staged-content check catches this by comparing the INDEX blob
    against the caller's declared sha AFTER `git add`. Only the subagent
    that declared the truly-merged sha can pass; the loser is refused
    with rc=5 and must re-base on whichever winning content is in HEAD.
    """
    if not expected:
        return {}
    actual = _hash_staged_files(list(expected.keys()), env)
    diffs: dict[str, tuple[str, str]] = {}
    for path, want in expected.items():
        got = actual.get(path, "MISSING")
        if got != want:
            diffs[path] = (want, got)
    return diffs


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
    parser.add_argument(
        "--expected-content-sha256",
        action="append",
        default=None,
        help=(
            "FIX-92aba3ca (2026-05-12 Catalog #157): declare the expected "
            "working-tree SHA-256 of a file as observed at the START of "
            "the subagent's work, BEFORE any sister subagent may have "
            "edited the same file. Repeatable per-file as "
            "'<relpath>=<sha256>'. The serializer refuses (rc=4) if the "
            "actual content differs from the declared expectation. "
            "Catches the commit-swap class where both subagents edited "
            "the same file before either took its pre-lock snapshot."
        ),
    )
    parser.add_argument(
        "--no-sister-checkpoint-check",
        action="store_true",
        help=(
            "Catalog #340 STAGING-surface PREVENT escape hatch. Skip the "
            "tac.commit_safety.check_files_against_sister_checkpoints scan "
            "that runs BEFORE fcntl-lock acquisition. Use ONLY when the "
            "operator has confirmed coordination via Catalog #230 ownership "
            "map; the paired-env bypass "
            "(SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE=1 + "
            "SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE_RATIONALE=<text>) is "
            "preferred over this CLI flag because it leaves an audit trail."
        ),
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
        # OMNIBUS GAP-5 (Catalog #289): log whether the caller passed
        # --expected-content-sha256 so Catalog #289 can detect the WAVE-D
        # drop-flag-and-retry pattern (rc=4 mismatch followed by committed
        # WITHOUT the flag = silent absorption of sister's edits).
        "expected_content_sha256_present": bool(args.expected_content_sha256),
        "expected_content_sha256_file_count": (
            len(args.expected_content_sha256) if args.expected_content_sha256 else 0
        ),
    }

    # FIX-92aba3ca (2026-05-12 Catalog #157): pre-lock-vs-EXPECTED check.
    # If the caller declared --expected-content-sha256 <file>=<sha>, verify
    # the working-tree content matches BEFORE doing anything else. This
    # catches the commit-swap class where both subagents have ALREADY
    # edited the same file in the working tree before either took its
    # pre-lock snapshot. The FIX-1 pre-vs-post-lock check would NOT catch
    # that race (both pre and post hashes would match the merged content).
    try:
        expected_content_shas = _parse_expected_content_sha256(
            args.expected_content_sha256 or []
        )
    except ValueError as exc:
        print(f"[subagent-commit-serializer] FATAL: {exc!s}", file=sys.stderr)
        return 2

    # OMNIBUS GAP-5 (Catalog #289): high-risk files MUST carry
    # --expected-content-sha256. Per WAVE-D 2c957c31e forensic analysis: the
    # agent-response failure mode (DROP the flag on rc=4 retry) is the root
    # cause of the recurring commit-swap class. Flip opt-in -> opt-out for
    # files with high concurrency risk so they cannot be committed without
    # the post-edit sha guard.
    _OMNIBUS_GAP5_HIGH_RISK_FILES = (
        "src/tac/preflight.py",
        "CLAUDE.md",
    )
    if files and not expected_content_shas:
        high_risk_in_commit = [
            f for f in files
            if any(f.endswith(hr) or f == hr for hr in _OMNIBUS_GAP5_HIGH_RISK_FILES)
        ]
        if high_risk_in_commit:
            _append_log({
                **base_record,
                "outcome": "high_risk_file_missing_expected_content_sha",
                "high_risk_files": high_risk_in_commit,
            })
            print(
                "[subagent-commit-serializer] REFUSED: high-risk file(s) "
                f"{high_risk_in_commit!r} REQUIRE --expected-content-sha256 "
                "per OMNIBUS GAP-5 (Catalog #289). These files have high "
                "concurrency risk; the WAVE-D 2c957c31e drop-flag-and-retry "
                "pattern is structurally extincted by making the flag "
                "MANDATORY for them. Compute the post-edit sha via "
                "`sha256sum <file> | awk '{print $1}'` and re-pass via "
                "`--expected-content-sha256 <file>=<sha>`.",
                file=sys.stderr,
            )
            return 5
    if expected_content_shas:
        diffs = _expected_content_sha256_check(expected_content_shas)
        if diffs:
            _append_log({
                **base_record,
                "outcome": "expected_content_sha_mismatch",
                "expected_content_sha_diffs": {
                    f: {"expected": want, "actual": got}
                    for f, (want, got) in diffs.items()
                },
            })
            print(
                "[subagent-commit-serializer] REFUSED: "
                "--expected-content-sha256 mismatch. Working-tree content "
                "differs from the SHA the caller declared at the START of "
                "its work. A sister subagent likely edited these files "
                "BEFORE the caller could take its pre-lock snapshot — the "
                "commit-swap class (FIX-92aba3ca / Catalog #157). "
                f"Files affected: {list(diffs.keys())!r}",
                file=sys.stderr,
            )
            return 4

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

    # Catalog #340 STAGING-surface PREVENT: check that no sister subagent
    # has declared the same files as "in_progress" in its checkpoint within
    # the last 60 minutes. Runs BEFORE fcntl-lock acquisition so the
    # operator gets a fast diagnostic (rc=8 ABORT / rc=9 WAIT_AND_RETRY)
    # without burning the lock-wait window on a doomed commit.
    #
    # Sister of Catalog #314 (POST-COMMIT detect) — together they extinct
    # the bare-commit-absorbs-in-flight-files bug class bidirectionally.
    # Bug class anchor (2026-05-19): slot 5 commit `c8d51ebb5` absorbed
    # slot 2's preflight.py + CLAUDE.md edits before slot 2's serializer
    # call ran; Catalog #157 caught the secondary effect but the absorption
    # was downstream of the bare `git add` that the `/commit` slash command
    # does directly (NOT through this wrapper).
    #
    # Bypass options:
    #   * --no-sister-checkpoint-check CLI flag (operator escape; rare)
    #   * Paired-env: SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE=1 AND
    #     SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE_RATIONALE=<text> (≥4 chars,
    #     not a placeholder). Bare flag without rationale → rc=10.
    if (
        not args.no_sister_checkpoint_check
        and not args.no_stage
        and files
        and _CATALOG_340_HELPER_AVAILABLE
    ):
        # rc=10 FIRST: bare override attempt (flag set but no rationale)
        # is a discipline violation distinct from a "real" sister conflict.
        # Surface it whether or not there's an actual conflict so operators
        # get the discipline lesson early.
        if bare_override_attempted is not None and bare_override_attempted(dict(os.environ)):
            _append_log({
                **base_record,
                "outcome": "sister_checkpoint_bare_override_rejected",
            })
            print(
                "[subagent-commit-serializer] REFUSED: bare paired-env "
                "bypass attempt. SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE=1 "
                "REQUIRES paired SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE_"
                "RATIONALE=<text> (≥4 chars, NOT a placeholder like "
                "'<rationale>' or '<reason>'). Per Catalog #199 paired-env "
                "discipline + Catalog #340 STAGING-surface PREVENT. Set "
                "the rationale and retry, OR coordinate via Catalog #230 "
                "ownership map.",
                file=sys.stderr,
            )
            return 10
        bypass_active = False
        bypass_rationale = ""
        if parse_override_env is not None:
            bypass_active, bypass_rationale = parse_override_env(dict(os.environ))
        if not bypass_active:
            try:
                verdict = check_files_against_sister_checkpoints(
                    list(files),
                    current_subagent_id=args.label,
                )
            except CorruptCheckpointError as exc:
                # Per Catalog #138 fail-closed: corrupt checkpoint state
                # must not silently let the commit proceed. rc=11 is the
                # corrupt-state-fail-closed code (distinct from rc=8/9/10).
                _append_log({
                    **base_record,
                    "outcome": "sister_checkpoint_corrupt_jsonl",
                    "error": str(exc)[:400],
                })
                print(
                    "[subagent-commit-serializer] REFUSED: "
                    "subagent_progress.jsonl is corrupt. Per Catalog #138 "
                    "fail-closed pattern + Catalog #340 STAGING-surface "
                    "PREVENT, commit refused until the checkpoint store is "
                    f"repaired or quarantined.\n  {exc!s}",
                    file=sys.stderr,
                )
                return 11
            if verdict.recommendation == "ABORT":
                _append_log({
                    **base_record,
                    "outcome": "sister_checkpoint_abort",
                    "sister_checkpoint_conflicts": [
                        {"sister_id": sid, "overlap": list(files_)}
                        for sid, files_ in verdict.conflicts
                    ],
                })
                print(
                    "[subagent-commit-serializer] REFUSED: ABORT per "
                    "Catalog #340 STAGING-surface PREVENT (sister of "
                    "Catalog #314 POST-COMMIT detect). At least one sister "
                    "subagent has declared one of these files in its "
                    "in-flight `files_touched` checkpoint within the last "
                    "60 minutes. Coordinate via Catalog #230 ownership "
                    "map OR opt out via the paired-env bypass.\n"
                    + verdict.diagnostic,
                    file=sys.stderr,
                )
                return 8
            if verdict.recommendation == "WAIT_AND_RETRY":
                _append_log({
                    **base_record,
                    "outcome": "sister_checkpoint_wait_and_retry",
                    "sister_checkpoint_conflicts": [
                        {"sister_id": sid, "overlap": list(files_)}
                        for sid, files_ in verdict.conflicts
                    ],
                })
                print(
                    "[subagent-commit-serializer] REFUSED: WAIT_AND_RETRY "
                    "per Catalog #340 STAGING-surface PREVENT. Sister "
                    "subagent(s) have older (>30 min) checkpoints "
                    "declaring these files; they may be near completion. "
                    "Retry with exponential backoff (e.g. 30s/60s/120s); "
                    "if still ABORT, escalate to Catalog #230 ownership "
                    "map coordination.\n" + verdict.diagnostic,
                    file=sys.stderr,
                )
                return 9
        else:
            # Bypass active; log it so the audit trail records who bypassed.
            _append_log({
                **base_record,
                "outcome": "sister_checkpoint_paired_env_bypass",
                "sister_checkpoint_bypass_rationale": bypass_rationale,
            })

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

        # Catalog #216 (FIX-HARDEN-OPT 2026-05-14 P1): POST-STAGE
        # content verification. The pre-lock + post-lock check (Catalog #157)
        # only catches working-tree edits DURING the lock-wait window. If
        # both subagents edited the same file BEFORE either took its
        # pre-lock snapshot, both see merged content as stable. The
        # 2026-05-14 D4-OOM-FIX vs FIX-HARDEN-OPT preflight.py race
        # (commit 5d0ec061d absorbed Catalog #215 edits) is the empirical
        # anchor. Verify what's actually STAGED in the temp index matches
        # the caller's declared post-edit sha; refuse with rc=5 on
        # mismatch (separate from rc=4 = pre-lock working-tree mismatch).
        if expected_content_shas and not args.no_stage:
            staged_diffs = _staged_content_check(expected_content_shas, env)
            if staged_diffs:
                _append_log({
                    **base_record,
                    "outcome": "staged_content_sha_mismatch",
                    "wait_seconds": wait_seconds,
                    "staged_content_sha_diffs": {
                        f: {"expected": want, "actual": got}
                        for f, (want, got) in staged_diffs.items()
                    },
                    "temp_index": temp_index_path,
                })
                print(
                    "[subagent-commit-serializer] REFUSED: post-stage "
                    "content sha mismatch. The file currently STAGED in "
                    "the index differs from the SHA the caller declared "
                    "at the start of its work. A sister subagent's "
                    "edits to the same file landed in HEAD between the "
                    "caller's pre-lock snapshot and `git add` (the 5d0ec061d "
                    "P1 anchor 2026-05-14 — D4-OOM-FIX absorbed Catalog #215 "
                    "preflight.py edits). Re-base on the sister's landed "
                    "work and retry with the merged-content sha. "
                    f"Files affected: {list(staged_diffs.keys())!r}",
                    file=sys.stderr,
                )
                return 5

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

        if temp_index_path:
            _refresh_real_index_after_temp_commit(files)

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
