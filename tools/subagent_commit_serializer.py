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

Canonical sha discipline (FIX-ABSORPTION 2026-07-07)
────────────────────────────────────────────────────
For any file that a sibling agent may also be editing (shared hot files:
the levelset trainer, curriculum_dsl.py, the DAG, preflight.py, CLAUDE.md),
declare BOTH shas:

    BASE=$(shasum -a 256 <file> | awk '{print $1}')   # BEFORE your first edit
    # ... your edits ...
    POST=$(shasum -a 256 <file> | awk '{print $1}')   # AFTER all edits
    python tools/subagent_commit_serializer.py \\
        --message "..." --files <file> \\
        --base-content-sha256 <file>=$BASE \\
        --expected-content-sha256 <file>=$POST

The POST sha (Catalog #157/#216) guards the edit-to-commit window; the BASE
sha guards the edit-START surface — it is compared against HEAD's blob, and
a mismatch means the file already contained a sibling's uncommitted hunks
when you began (whole-file `git add` would absorb them under your commit
body — the serializer_whole_file_staging_absorbs_sibling_hunks class,
incident commits 1d6704e5b/049aa0d9f). rc=6 refusal; retry after the
sibling lands (HEAD then matches your base and the check passes).

Post-commit verification + shared-file discipline (FIX-CLOBBER 2026-07-08)
─────────────────────────────────────────────────────────────────────────
Two 2026-07-08 incidents showed the pre-commit sha guards are blind to
clobbers that PRECEDE the caller's snapshot:

  (1) A sibling's file REVERT landed in the working tree BEFORE a builder
      snapshotted its --expected-content-sha256, so the builder declared
      (and every pre-commit check verified) the CLOBBERED content. rc=0
      committed the sibling's copy under the builder's body — caught only
      by a post-commit `git show`.
  (2) A whole-file `git add` swept a DIFFERENT sibling's uncommitted hunks
      into the wrong commit body (mis-attribution).

Two structural additions close/mitigate these:

* POST-COMMIT VERIFICATION (rc=7, automatic when --expected-content-sha256
  is passed): after the commit lands, the serializer re-reads each declared
  file AT HEAD (`git cat-file blob HEAD:<file>`) and compares to the
  declared sha. This is the ONLY check that reads HEAD after the ref moved,
  so it catches the pre-snapshot-clobber gap. The commit is KEPT (not
  auto-reverted — it may be a sibling's newer legitimate landing); the
  serializer prints reconcile guidance (`git show`, re-apply via
  --patch-file, or `git revert --no-commit <sha>`) and returns rc=7.

* --patch-file INTENT-MANIFEST MODE (the real fix for shared hot files):
  supply a patch of EXACTLY your hunks; the serializer applies it with
  `git apply --cached` to a temp index seeded from HEAD, IGNORING the
  working tree, so no sibling hunk can leak in. --expected-diff-lines
  <file>=<N> is a warn-only heuristic that flags a grossly larger staged
  diff for callers still using whole-file `git add` on shared files.

    Return-code map: 0 ok · 2 fatal/malformed/timeout · 3 concurrent-edit
    (lock-wait) · 4 pre-lock expected-sha mismatch · 5 staged-sha mismatch /
    high-risk-file-missing-sha · 6 base-sha mismatch (absorption) · 7 POST-COMMIT
    HEAD mismatch (clobber) · 8/9 sister-checkpoint ABORT/WAIT · 10 bare-override ·
    11 corrupt-checkpoint · 12 review-gate override attempted on Python ·
    13 gitignored path · 14 protected append-doc shrink · 15 undeclared staged file ·
    17 git-object write denial captured as an SSD bundle fallback ·
    18 git-object write denial detected but fallback construction failed ·
    19 bundle fallback refused by its artifact cap or canonical storage reserve.

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
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
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
    from tac.commit_safety import (
        bare_override_attempted,
        check_files_against_sister_checkpoints,
        parse_override_env,
    )
    from tac.commit_safety.sister_checkpoint_guard import (
        CorruptCheckpointError,
        infer_current_subagent_id,
    )
    _CATALOG_340_HELPER_AVAILABLE = True
except ImportError:
    # Test fixtures may stand up a minimal repo without the package.
    _CATALOG_340_HELPER_AVAILABLE = False
    check_files_against_sister_checkpoints = None  # type: ignore[assignment]
    infer_current_subagent_id = None  # type: ignore[assignment]
    bare_override_attempted = None  # type: ignore[assignment]
    parse_override_env = None  # type: ignore[assignment]
    CorruptCheckpointError = RuntimeError  # type: ignore[misc, assignment]

try:
    from comma_lab.storage_tiers import DEFAULT_RESERVE_FREE_GB, bytes_from_gib
    _STORAGE_WATERFALL_AVAILABLE = True
except ImportError:
    # Focused fixtures may copy this file without the repository's ``src``
    # tree. Ordinary commits remain usable; the SSD fallback fails loudly if
    # its canonical reserve source is unavailable.
    DEFAULT_RESERVE_FREE_GB = None  # type: ignore[assignment]
    bytes_from_gib = None  # type: ignore[assignment]
    _STORAGE_WATERFALL_AVAILABLE = False

LOCK_PATH = REPO_ROOT / ".omx/state/.commit-lock"
LOG_PATH = REPO_ROOT / ".omx/state/commit-serializer.log"


def _resolve_effective_repo_root(explicit: str | None = None) -> Path:
    """Resolve the repo root to operate on — WORKTREE-AWARE (FIX 2026-07-17).

    The module-level ``REPO_ROOT`` is derived from THIS FILE's location, i.e. the
    MAIN checkout. But a subagent may run the serializer from inside a git
    WORKTREE (separate working tree + index, shared object store). Operating on
    the main checkout from there silently stages the MAIN copy of the file, not
    the worktree copy — so the caller's post-edit ``--expected-content-sha256``
    never matches (rc=4) even though nothing actually raced, and a legitimate
    worktree commit is impossible via the serializer. Resolve the root the caller
    actually intends, in priority order:

      1. explicit ``--repo-root``
      2. ``$SUBAGENT_SERIALIZER_REPO_ROOT``
      3. ``git rev-parse --show-toplevel`` from CWD (picks the worktree)
      4. this file's checkout (back-compat fallback — unchanged behaviour when
         run from the main checkout, and when git is unavailable)

    A worktree has its OWN index, so operating on it preserves the
    anti-commit-swap guarantee: the race the lock protects against is PER-INDEX,
    and two commits to different working trees never share an index. A linked
    worktree's ``.git`` is a FILE (a ``gitdir:`` pointer), not a dir, so
    ``.exists()`` validates both the main checkout and a worktree.
    """
    for cand in (explicit, os.environ.get("SUBAGENT_SERIALIZER_REPO_ROOT")):
        if cand:
            p = Path(cand).resolve()
            if (p / ".git").exists():
                return p
    # Auto-detect the working tree containing CWD (the worktree case).
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path.cwd(), capture_output=True, text=True,
            check=False, timeout=10,
        )
        if out.returncode == 0:
            top = out.stdout.strip()
            if top:
                p = Path(top).resolve()
                if (p / ".git").exists():
                    return p
    except (OSError, subprocess.SubprocessError):
        pass
    return REPO_ROOT

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


_BASE_NEW_FILE_TOKEN = "new"


def _parse_base_content_sha256(arg_values: list[str]) -> dict[str, str]:
    """Parse ``--base-content-sha256 <file>=<sha|new>`` flag values.

    Each value must be ``<relpath>=<64-hex>`` OR ``<relpath>=new`` (the file
    did not exist when the caller began editing — a caller-created file).
    Returns a dict mapping relpath -> declared base (hex sha or ``new``).

    Raises ValueError on malformed input.
    """
    out: dict[str, str] = {}
    for v in arg_values or []:
        if "=" not in v:
            raise ValueError(
                f"--base-content-sha256 must be '<relpath>=<sha256|new>'; "
                f"got {v!r}"
            )
        path, _, sha = v.partition("=")
        path = path.strip()
        sha = sha.strip().lower()
        if not path or not sha:
            raise ValueError(
                f"--base-content-sha256 has empty path or sha in {v!r}"
            )
        if sha != _BASE_NEW_FILE_TOKEN and (
            len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError(
                f"--base-content-sha256 sha must be 64 hex chars or 'new'; "
                f"got {sha!r} for path {path!r}"
            )
        out[path] = sha
    return out


def _hash_head_blob_files(files: list[str]) -> dict[str, str]:
    """SHA-256 each file's content AT HEAD (the committed blob, not the
    working tree, not any index). Returns ``MISSING`` for paths not present
    at HEAD. Used by the FIX-ABSORPTION base-content check below.
    """
    out: dict[str, str] = {}
    for f in files:
        try:
            proc = subprocess.run(
                ["git", "cat-file", "blob", f"HEAD:{f}"],
                cwd=REPO_ROOT, capture_output=True, check=False,
            )
        except OSError as exc:
            out[f] = f"ERROR_CAT_FILE:{type(exc).__name__}"
            continue
        if proc.returncode != 0:
            out[f] = "MISSING"
            continue
        out[f] = hashlib.sha256(proc.stdout).hexdigest()
    return out


def _files_recorded_by_head_commit() -> set[str] | None:
    """Relpaths the JUST-LANDED commit actually changed, per git itself.

    FIX-ATTRIBUTION (2026-08-02, task #911): the complement of every existing
    check. ``_post_commit_content_check`` asks "is the right CONTENT at HEAD?"
    — it cannot ask "did MY commit put it there?", because when a sibling has
    already committed byte-identical content the HEAD blob matches and the
    check passes by construction. That is exactly what happened: a sibling
    absorbed 215 lines of another arm's trainer work; the serializer printed
    ``files=6`` while git recorded 5 changed, and NOTHING compared the two
    numbers.

    Fail-open (returns ``None`` on any git/OS error or a root commit) so this
    diagnostic can never break a real commit.
    """
    try:
        proc = subprocess.run(
            ["git", "show", "--pretty=format:", "--name-only", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return {ln.strip() for ln in proc.stdout.splitlines() if ln.strip()}


def _base_content_check(
    base: dict[str, str],
) -> dict[str, tuple[str, str]]:
    """FIX-ABSORPTION (2026-07-07): declared-edit-BASE vs HEAD-blob check.

    The 2026-07-07 build-wave absorption incident (commits 1d6704e5b +
    049aa0d9f; harness-failure-ledger id
    ``serializer_whole_file_staging_absorbs_sibling_hunks``, 5th+ firing)
    showed that Catalog #157 (rc=4) + #216 (rc=5) are TAUTOLOGICAL against
    co-mingled content: the caller computes ``--expected-content-sha256``
    on its post-edit WORKING TREE, which already contains any sibling's
    uncommitted hunks, so both checks pass by construction and the
    whole-file ``git add`` stages the sibling's hunks under the caller's
    commit body.

    The missing information is the caller's edit BASE — what the file
    looked like BEFORE the caller's own edits. Callers pass
    ``--base-content-sha256 <file>=<sha>`` (sha computed BEFORE editing;
    the literal ``new`` for a file the caller created). The serializer
    compares the declared base against the file's content AT HEAD:

    - base == HEAD blob → the working-tree delta on this file is exactly
      the caller's own edits; whole-file staging is attribution-safe.
    - base != HEAD blob → the file contained uncommitted foreign hunks
      when the caller began editing (absorption imminent), OR HEAD has
      since moved past content the caller never based on (whole-file
      staging would REVERT the sibling's landed hunks). Refuse (rc=6).

    Natural resolution: once the sibling lands exactly the hunks that were
    in the caller's base, ``HEAD:<file>`` equals the declared base and the
    check passes on retry — WAIT_AND_RETRY semantics, no override needed.

    Returns mismatches ``{relpath: (declared_base, head_sha)}``; empty
    dict when every declared base matches (or nothing was declared).
    """
    if not base:
        return {}
    head = _hash_head_blob_files(list(base.keys()))
    diffs: dict[str, tuple[str, str]] = {}
    for path, want in base.items():
        got = head.get(path, "MISSING")
        if want == _BASE_NEW_FILE_TOKEN:
            if got != "MISSING":
                diffs[path] = (want, got)
        elif got != want:
            diffs[path] = (want, got)
    return diffs


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
    """NO-OP per operator NON-NEGOTIABLE 2026-05-31: "there should be no co-author
    trailer ever in our commit history."

    Returns the message UNCHANGED — no Co-Authored-By trailer is ever appended.
    Retained as a no-op shim so any legacy caller is structurally neutralised
    rather than removed (back-compat). The prior FIX-3 (2026-05-08) auto-append
    behaviour is superseded by this newer explicit operator directive.
    Self-protected by Catalog #119 (now FORBID-trailer; was require-trailer).
    """
    return message


# Catalog #206 body-evidence tokens, lowercased mirror of preflight's
# _CHECKPOINT_DISCIPLINE_TOKENS (the gate matches casefolded). Any of these in
# the caller's message means the caller supplied its own checkpoint evidence
# and the serializer must not add a second line.
_CHECKPOINT_DISCIPLINE_TOKENS_FOLDED = (
    "tools/subagent_checkpoint.py",
    "subagent_checkpoint.py",
    "subagent_progress.jsonl",
    "checkpoint discipline honored",
    "checkpoint discipline: honored",
    "checkpoint_discipline_waived",
)


def _ensure_checkpoint_discipline_line(message: str, label: str) -> str:
    """Catalog #206 evidence, determinized at the serializer (2026-08-25).

    The checkpoint discipline was practiced (307 compliant commits after the
    2026-05-19 cutoff) and then lapsed — the #936 adoption-decay genus. The
    cure lives at the ergonomic layer: the tool that lands every commit also
    carries the discipline evidence, so compliance no longer depends on each
    caller remembering a waiver line.

    Appends one honest, class-specific `# CHECKPOINT_DISCIPLINE_WAIVED:<reason>`
    line when the message carries NO checkpoint token at all. Suppressed the
    moment any token is present — caller-supplied evidence (including a bare,
    reason-less waiver the gate will rightly reject) always wins; the auto-line
    only fills silence, never overrides intent. The reason is factual per
    caller class, not boilerplate: anonymous serializer calls are single-shot
    MAIN landings with no multi-step resume state; labeled calls are keeper-arm
    landings whose resume custody lives in the keeper apparatus (charter +
    done-receipt + persisted final message), not in commit bodies.
    """
    folded = message.casefold()
    if any(tok in folded for tok in _CHECKPOINT_DISCIPLINE_TOKENS_FOLDED):
        return message
    if label and label != "anonymous":
        reason = (
            f"keeper-arm '{label}' landing; resume custody = keeper apparatus "
            "(charter + done-receipt + persisted final message), not "
            "commit-body checkpoints (auto-appended by the serializer)"
        )
    else:
        reason = (
            "single-shot MAIN serializer landing; no multi-step subagent "
            "resume state to checkpoint (auto-appended by the serializer)"
        )
    return f"{message}\n\n# CHECKPOINT_DISCIPLINE_WAIVED:{reason}"

# ------------------------------------------------------------------------------------
# Lock patience.
#
# This is a CONTROL knob (task #847 / #854): it acts on the SELECTION channel, deciding
# which commit attempts survive contention. It is DERIVED from the hook's own declared
# bounds rather than chosen, because the previous literal was derived once and then
# silently outlived its measurement.
#
# WHAT WENT WRONG WITH THE LITERAL. The retired comment read "the pre-commit hook runs
# full preflight (~5-10s) so 120s easily accommodates 5+ queued subagents". That was TRUE
# when written and is STILL true at the median: MEASURED over .omx/state/commit-serializer
# .log (n=9112 attempts carrying `commit_seconds`), p50 = 3.2s, p90 = 7.8s. The hazard
# moved into the TAIL: p99 = 161.0s, max = 468.0s, because the hook grew a CI-blind
# (MLX-gated) pytest step that agents routinely run with
# PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS raised to 1500. A patience derived from the median
# cannot survive a tail that grew, and nothing re-derived it. MEASURED consequence on
# 2026-08-01: 7 of 60 commit attempts (11.7%) died with outcome=lock_timeout — ddm_tr6 x5,
# ddm_vc1 x1, ddm_rt1 x1 — every one of them a healthy commit behind a healthy sibling.
#
# WHY A TIMEOUT AT ALL, AND WHY LARGE. `fcntl.flock` is released by the kernel when the
# holder dies, so a CRASHED holder can never block us: the only thing this timeout can
# protect against is a holder that is alive but wedged, and a timeout cannot tell wedged
# from slow. Setting it below the legitimate hook duration therefore buys no safety and
# converts normal contention into failure. The patience is set to cover legitimate
# contention, and `_acquire_lock` now REPORTS progress while waiting so a wedged holder
# is visible immediately instead of after a silent block.
#
# DERIVATION. Two ways to be starved, so the patience is the max of two terms:
#   (1) ONE sibling running its hook to the full bound. We must not give up while it is
#       still healthy, so the patience is at least
#       preflight_hook.effective_hook_wall_clock_bound_seconds(), read from the SAME
#       environment the child `git commit` inherits — a raised
#       PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS raises this patience with it, so the two can
#       no longer drift apart. This is the term that binds today.
#   (2) A QUEUE of siblings each taking a typical long hook:
#       MAX_CONCURRENT_COMMITTERS x the ledger's p99 hold.
#         queue depth = 6, MEASURED: the maximum number of DISTINCT agent labels with
#           overlapping commit attempts since 2026-07-25 (n=735 attempts; the same
#           statistic for 2026-08-01 alone is 3).
#         p99 hold = 161s, MEASURED: p99 of `commit_seconds` over the ledger (n=9112).
# NOT queue depth x the BOUND: that compounds two worst cases into a patience of hours.
# Six siblings simultaneously running to a 20-minute ceiling is not contention, it is a
# wedged machine — which the progress line below surfaces while it is happening.
# CROSS-CHECK (independent of the derivation): the largest SUCCESSFUL lock wait ever
# recorded is 1021.93s (n=9833 waits), and the arm instructions already tell agents to
# pass `--timeout-seconds 2400` by hand. The derived default sits between the two, and
# making it the DEFAULT is the point: a patience that only works when every caller
# remembers a flag is an orphaned default.
# OWNER: the commit serializer surface (tools/subagent_commit_serializer.py).
# RE-DERIVATION TRIGGER: a lock_timeout rate above ~1% of attempts in any week; a
#   measured max concurrent-label count above MAX_CONCURRENT_COMMITTERS; or a measured
#   p99 `commit_seconds` above MEASURED_P99_HOLD_SECONDS.
MAX_CONCURRENT_COMMITTERS = 6
MEASURED_P99_HOLD_SECONDS = 161

# Used only when tools/preflight_hook.py cannot be read (fixtures stand up minimal repos).
_FALLBACK_HOOK_BOUND_SECONDS = 180


def _hook_wall_clock_bound_seconds() -> int:
    """Whatever bound the pre-commit hook is running under, in THIS environment."""
    hook_path = Path(__file__).resolve().parent / "preflight_hook.py"
    try:
        spec = importlib.util.spec_from_file_location(
            "_serializer_preflight_hook", hook_path)
        if spec is None or spec.loader is None:
            return _FALLBACK_HOOK_BOUND_SECONDS
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return int(module.effective_hook_wall_clock_bound_seconds())
    except Exception:
        # A hook we cannot read must not stop a commit; fall back, never raise.
        return _FALLBACK_HOOK_BOUND_SECONDS


def default_timeout_seconds() -> int:
    """Lock patience = max(one full hook bound, a queue of typical long hooks).

    See the derivation above; both terms are measured, neither is chosen.
    """
    return max(
        _hook_wall_clock_bound_seconds(),
        MAX_CONCURRENT_COMMITTERS * MEASURED_P99_HOLD_SECONDS,
    )


DEFAULT_TIMEOUT_SECONDS = default_timeout_seconds()

# How often to say, out loud, that we are still waiting. A silent multi-minute block is
# indistinguishable from a hang; the periodic line is what makes a wedged holder legible
# without turning legitimate contention into a failure.
LOCK_WAIT_PROGRESS_SECONDS = 30


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


def _git_common_dir() -> Path | None:
    """The real .git directory (worktree-aware); None if git cannot answer."""

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    git_dir = Path(proc.stdout.strip())
    return git_dir if git_dir.is_absolute() else (REPO_ROOT / git_dir)


def merge_in_progress() -> tuple[str, float] | None:
    """(MERGE_HEAD sha, mtime epoch) when a merge is open, else None.

    An open `git merge --no-commit` makes the NEXT commit in this repo a merge
    commit whatever it stages: git reads .git/MERGE_HEAD and records it as a
    second parent.  A serializer commit landing in that window therefore writes
    history that CLAIMS the branch is merged while committing only its own
    files.
    """

    git_dir = _git_common_dir()
    if git_dir is None:
        return None
    merge_head = git_dir / "MERGE_HEAD"
    try:
        text = merge_head.read_text(encoding="utf-8", errors="replace").strip()
        mtime = merge_head.stat().st_mtime
    except OSError:
        return None
    first = text.splitlines()[0].strip() if text else ""
    return (first, mtime) if first else None


def _merge_branch_changed_files(merge_head_sha: str, env: dict) -> list[str] | None:
    """Files the open merge's branch changed since the merge base.

    Fail-open (None) on any git error: this guard must never wedge a commit
    because git could not answer, only because it answered that content is
    missing.
    """

    try:
        base = subprocess.run(
            ["git", "merge-base", "HEAD", merge_head_sha],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
        )
        if base.returncode != 0 or not base.stdout.strip():
            return None
        changed = subprocess.run(
            ["git", "diff", "--name-only", base.stdout.strip(), merge_head_sha],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    if changed.returncode != 0:
        return None
    return [ln.strip() for ln in changed.stdout.splitlines() if ln.strip()]


def _acquire_lock(timeout_seconds: int):
    """Acquire LOCK_EX on .commit-lock with a soft timeout, reporting progress.

    Returns the open file handle (caller must keep it open until release).
    Raises TimeoutError if the lock can't be acquired within timeout.

    The wait is NARRATED every LOCK_WAIT_PROGRESS_SECONDS. Patience without narration is
    the failure mode this replaces: the caller could not tell a healthy sibling's long
    hook from a hang, so the only visible signal was the eventual FATAL. Since `flock` is
    released by the kernel when a holder dies, a wait that keeps growing means a LIVE
    holder — which the periodic line makes actionable while it is still happening.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.touch(exist_ok=True)
    fh = open(LOCK_PATH, "w")  # noqa: SIM115 - caller must hold the handle until explicit unlock.
    started = time.monotonic()
    deadline = started + timeout_seconds
    next_report = started + LOCK_WAIT_PROGRESS_SECONDS
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            now = time.monotonic()
            if now >= deadline:
                fh.close()
                raise TimeoutError(
                    f"Could not acquire {LOCK_PATH} within {timeout_seconds}s "
                    f"(= {MAX_CONCURRENT_COMMITTERS} concurrent committers x the "
                    f"{_hook_wall_clock_bound_seconds()}s pre-commit hook bound). "
                    f"A holder that outlives this is wedged, not slow — flock is "
                    f"released on process death. Inspect: "
                    f"tail .omx/state/commit-serializer.log"
                ) from None
            if now >= next_report:
                waited = int(now - started)
                print(
                    f"[subagent-commit-serializer] waiting for {LOCK_PATH.name}: "
                    f"{waited}s of {timeout_seconds}s — a sibling's pre-commit hook "
                    f"holds it (its CI-blind test step can legitimately run minutes). "
                    f"Inspect: tail .omx/state/commit-serializer.log",
                    file=sys.stderr,
                )
                next_report = now + LOCK_WAIT_PROGRESS_SECONDS
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


def _refresh_real_index_after_temp_commit(files: list[str], repo_root: Path | None = None) -> None:
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
    if repo_root is None:
        # Resolve at CALL time (not def time) so tests that patch the
        # module-level REPO_ROOT to a throwaway repo stay hermetic — a
        # def-time default froze the real repo root and made test commits
        # run `git reset` against the real index (2026-07-07 hermeticity fix).
        repo_root = REPO_ROOT
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


def _check_ignored_files(files: list[str]) -> list[str]:
    """PERMANENT FIX (2026-07-18): gitignored-file silent whole-commit abort.

    THE BUG (self-caught 2026-07-18 B-power-diagram harvest): passing a
    gitignored path to ``--files`` makes ``git add -- <ignored>`` fail with a
    generic "paths are ignored by one of your .gitignore files ... use -f"
    hint and rc!=0, which aborts the ENTIRE commit — the log said "committing
    38 files" but HEAD never moved because ONE gitignored ``storage_plan.json``
    poisoned the whole ``git add``. The failure is easy to miss (HEAD unchanged,
    only a git hint on stderr, no named culprit).

    THE FIX: a pre-lock preflight that names the offending files LOUDLY and
    refuses with a distinct rc BEFORE any staging, so the caller removes the
    gitignored path from ``--files`` (bulk/rebuildable artifacts belong on the
    SSD cold-store per the disk-hygiene non-negotiable, never in git).

    ``git check-ignore -- <files>`` prints the ignored subset (rc=0 if any are
    ignored, rc=1 if none, rc>=128 on error). Returns the ignored relpaths.
    """
    if not files:
        return []
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "--", *files],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
    except OSError:
        # If check-ignore itself can't run, don't block — let the normal
        # `git add` path surface the error (fail-open on the guard, not the
        # commit). This preserves today's behavior when git is unavailable.
        return []
    if proc.returncode not in (0, 1):
        # rc>=128 = check-ignore error (e.g. bad pathspec); fail-open.
        return []
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


# PERMANENT FIX (2026-07-18): protected append-doc whole-file-clobber guard.
# These research docs are MULTI-WRITER + APPEND-HEAVY (many arms/agents add
# rows; they never intentionally SHRINK in normal operation). A whole-file
# ``cp``/overwrite off a STALE base silently drops sibling rows — the exact
# clobber that wiped the compiler arm's factor-1/5 edits from the completeness
# matrix on 2026-07-18. A net line LOSS on one of these is the clobber
# signature; refuse it unless the caller declares an intentional shrink
# (consolidation) via --allow-shared-doc-shrink. Match on relpath substrings.
_PROTECTED_APPEND_DOC_MARKERS: tuple[str, ...] = (
    "inverse_solve_completeness_matrix",  # the 2026-07-18 clobber anchor
    "sub015_DAG",                          # the canonical work-graph DAG
    "_DAG_FEED_",                          # DAG feed blocks
    "canonical_equations_registry",        # append-only equation ledger
)
# Net line loss (HEAD_lines - staged_lines) at/above this is treated as a
# clobber, not an edit. Well above normal correction churn (a few lines),
# well below a factor-block clobber (dozens). Escape via override flag.
_PROTECTED_DOC_SHRINK_LINES = 8

# Rationales REJECTED as placeholder stubs for --allow-shared-doc-shrink: the
# override must carry a REAL reason, not a filler token (review F1 2026-07-18:
# `--allow-shared-doc-shrink TODO` slipped past). Lowercased comparison.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset({
    "", "<rationale>", "<reason>", "rationale", "reason", "tbd", "todo",
    "fixme", "xxx", "wip", "n/a", "na", "none", "null", "-", ".", "?",
})


def _staged_touched_files(env: dict, *, diff_filter: str | None = None) -> list[str]:
    """Paths the temp index differs from HEAD on — i.e. what a --patch-file patch
    actually staged. In patch mode the caller passes no --files, so the harvest
    guards must run against THIS set instead (review F1 2026-07-18: patch-mode
    bypassed both the gitignore and protected-doc-shrink guards).

    ``diff_filter='A'`` restricts to ADDED paths (the gitignore guard only cares
    about newly-introduced ignored bulk, never modifications to tracked files).
    Fail-open: any git error returns [] so the guard never breaks a real commit.
    """
    cmd = ["git", "diff", "--cached", "--name-only"]
    if diff_filter:
        cmd.append(f"--diff-filter={diff_filter}")
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


def _staged_declared_file_set_mismatch(
    files: list[str], env: dict,
) -> tuple[list[str], list[str], list[str]]:
    """Compare the staged file set to the caller's declared file list.

    ``--no-stage`` intentionally honors the real shared index for repair paths,
    which is exactly where an unrelated staged file can be silently swept into a
    commit. The temp-index paths should already be isolated, but running the same
    check there also catches hook/patch surprises before or immediately after the
    commit.

    Returns ``(staged_but_not_declared, declared_but_not_staged, staged_files)``.
    Only ``staged_but_not_declared`` is a hard pre-commit violation: a declared
    file absent from the staged diff can be a legitimate no-op versus HEAD.
    """
    staged = set(_staged_touched_files(env))
    declared = set(files)
    return sorted(staged - declared), sorted(declared - staged), sorted(staged)


def _refuse_staged_declared_file_set_mismatch(
    files: list[str],
    env: dict,
    record: dict,
    *,
    wait_seconds: float | int | None = None,
    temp_index_path: str | None = None,
    context: str,
) -> int | None:
    """rc=15 refusal when the staged diff contains undeclared paths."""
    extra, absent, staged = _staged_declared_file_set_mismatch(files, env)
    if not extra:
        return None
    log_record = {
        **record,
        "outcome": "staged_file_set_mismatch_refused",
        "context": context,
        "declared_files": sorted(set(files)),
        "staged_files": staged,
        "staged_but_not_declared": extra,
        "declared_but_not_staged": absent,
        "temp_index": temp_index_path,
    }
    if wait_seconds is not None:
        log_record["wait_seconds"] = wait_seconds
    _append_log(log_record)
    print(
        "[subagent-commit-serializer] REFUSED (rc=15): the staged file set "
        "contains path(s) not declared by --files. This would let a repair/"
        "--no-stage or hook-mutated commit carry unauthored work under the "
        "wrong message. Declare the exact staged paths or unstage the decoy "
        "before retrying. "
        f"context={context}; staged_but_not_declared={extra!r}; "
        f"declared_but_not_staged={absent!r}",
        file=sys.stderr,
    )
    return 15


def _is_protected_append_doc(relpath: str) -> bool:
    return any(m in relpath for m in _PROTECTED_APPEND_DOC_MARKERS)


def _protected_append_doc_shrink_check(files: list[str], env: dict) -> dict[str, tuple[int, int]]:
    """Return {relpath: (head_lines, staged_lines)} for protected append docs
    whose STAGED content lost >= _PROTECTED_DOC_SHRINK_LINES lines vs HEAD.

    Reads staged content from the temp index (``git show :0:<f>`` honoring
    env's GIT_INDEX_FILE) and HEAD content from the HEAD blob. A file NOT yet
    tracked on HEAD (brand-new doc) can't be clobbered → skipped.
    """
    hits: dict[str, tuple[int, int]] = {}
    for f in files:
        if not _is_protected_append_doc(f):
            continue
        # HEAD blob (old). Missing on HEAD = new file, cannot clobber.
        head = subprocess.run(
            ["git", "show", f"HEAD:{f}"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        if head.returncode != 0:
            continue
        head_lines = head.stdout.count("\n")
        # Staged blob (new) from OUR temp index.
        staged = subprocess.run(
            ["git", "show", f":0:{f}"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
        )
        if staged.returncode != 0:
            # Staged blob ABSENT while HEAD has it = the protected doc is being
            # DELETED (or fully removed from the index). Deletion is the MAXIMAL
            # clobber — treat as staged_lines=0 and flag UNCONDITIONALLY (even a
            # tiny <threshold doc), so the override is required rather than the
            # deletion silently allowed (review F1 2026-07-18: --files deletion
            # and patch-mode deletion both bypassed this by hitting `continue`).
            hits[f] = (head_lines, 0)
            continue
        staged_lines = staged.stdout.count("\n")
        if head_lines - staged_lines >= _PROTECTED_DOC_SHRINK_LINES:
            hits[f] = (head_lines, staged_lines)
    return hits


def _refuse_gitignored(ignored: list[str], record: dict) -> int | None:
    """rc=13 refusal for gitignored paths about to be committed (a single one
    aborts the WHOLE `git add` silently). Returns 13 to refuse, None if clean.
    Shared by the --files pre-lock check and the --patch-file post-apply check
    so both staging surfaces are guarded identically (review F1 2026-07-18)."""
    if not ignored:
        return None
    _append_log({**record, "outcome": "gitignored_files_in_commit_refused",
                 "ignored_files": ignored})
    print(
        "[subagent-commit-serializer] REFUSED (rc=13): "
        f"{len(ignored)} file(s) about to be committed are gitignored — a single "
        "gitignored path aborts the WHOLE `git add` silently (only a git hint, "
        "no named culprit; incident 2026-07-18 'committing N files' with HEAD "
        "unmoved). Remove them; bulk / rebuildable artifacts belong on the SSD "
        f"cold-store, not git. Gitignored: {ignored!r}",
        file=sys.stderr,
    )
    return 13


def _refuse_protected_doc_shrink(
    shrinks: dict, override_raw: str | None, record: dict,
) -> int | None:
    """rc=14 refusal for a protected-append-doc whole-file clobber (net line loss
    OR deletion off a stale base). Returns 14 to refuse, None to allow (logged).
    A REAL --allow-shared-doc-shrink rationale downgrades to allow; a placeholder
    stub is rejected. Shared by the --files and --patch-file staging paths so a
    patch can no longer clobber a protected doc unguarded (review F1 2026-07-18)."""
    if not shrinks:
        return None
    shrink_detail = {f: {"head_lines": h, "staged_lines": s}
                     for f, (h, s) in shrinks.items()}
    override = (override_raw or "").strip()
    if override.lower() in _PLACEHOLDER_RATIONALES:
        _append_log({**record, "outcome": "protected_append_doc_clobber_refused",
                     "shrinks": shrink_detail})
        print(
            "[subagent-commit-serializer] REFUSED (rc=14): the staged version of "
            f"a PROTECTED APPEND doc lost >={_PROTECTED_DOC_SHRINK_LINES} lines "
            "(or was deleted) vs HEAD — the whole-file-clobber-off-a-stale-base "
            "signature (2026-07-18 completeness-matrix incident wiped a sibling "
            "arm's factor edits). Re-base with a 3-way `git merge-file -p <yours> "
            "<base> HEAD:<f> > <f>` (0 conflict markers) so both writers' rows "
            "survive. If the shrink is INTENTIONAL (consolidation) pass "
            "--allow-shared-doc-shrink '<real reason>'. "
            + "; ".join(f"{f}: HEAD={h} lines -> staged={s} lines"
                        for f, (h, s) in shrinks.items()),
            file=sys.stderr,
        )
        return 14
    _append_log({**record, "outcome": "protected_append_doc_shrink_allowed",
                 "allow_shared_doc_shrink_rationale": override,
                 "shrinks": shrink_detail})
    return None


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


def _post_commit_content_check(
    expected: dict[str, str],
) -> dict[str, tuple[str, str]]:
    """FIX-CLOBBER (2026-07-08 Catalog #405): POST-commit HEAD-blob verification.

    Closes the pre-snapshot-clobber gap (incident 1, 2026-07-08): a sibling's
    file REVERT landed in the working tree BEFORE the builder computed its
    ``--expected-content-sha256`` snapshot, so every PRE-commit working-tree
    check (rc=4 pre-lock, rc=5 staged) compared the declared sha against the
    already-clobbered content and PASSED by construction — the serializer
    committed the sibling's copy under the builder's body at rc=0. The gap is
    structural: all prior checks read the WORKING TREE / INDEX, never HEAD
    after the ref moved.

    This check is the ground-truth "did what I declared actually land in HEAD?"
    verification. It re-reads each declared file's content AT HEAD via
    ``git cat-file blob HEAD:<file>`` (the just-committed blob) and compares
    its SHA-256 to the caller's declared ``--expected-content-sha256`` (the
    content the caller intended to commit, observed at the START of its work).
    A mismatch means the committed content is NOT what the caller declared —
    a clobber (or any TOCTOU divergence between intent and reality) slipped
    through the working-tree-based checks.

    Returns ``{relpath: (declared_sha, committed_head_sha)}`` for every
    mismatch; empty dict when every declared file's HEAD blob matches (or
    nothing was declared → backward-compatible no-op).
    """
    if not expected:
        return {}
    head = _hash_head_blob_files(list(expected.keys()))
    diffs: dict[str, tuple[str, str]] = {}
    for path, want in expected.items():
        got = head.get(path, "MISSING")
        if got != want:
            diffs[path] = (want, got)
    return diffs


def _parse_expected_diff_lines(arg_values: list[str]) -> dict[str, int]:
    """Parse ``--expected-diff-lines <file>=<int>`` flag values (Catalog #405).

    Each value must be ``<relpath>=<non-negative-int>`` — the caller's hint for
    how many added+deleted lines its OWN edits to the file comprise. Used by
    the warn-only hunk-attribution heuristic to flag a grossly larger staged
    diff (a whole-file ``git add`` that swept a sibling's hunks in — incident 2,
    2026-07-08). Empty list -> empty dict. Raises ValueError on malformed input.
    """
    out: dict[str, int] = {}
    for v in arg_values or []:
        if "=" not in v:
            raise ValueError(
                f"--expected-diff-lines must be '<relpath>=<int>'; got {v!r}"
            )
        path, _, n = v.partition("=")
        path = path.strip()
        n = n.strip()
        if not path or not n:
            raise ValueError(
                f"--expected-diff-lines has empty path or count in {v!r}"
            )
        try:
            count = int(n)
        except ValueError:
            raise ValueError(
                f"--expected-diff-lines count must be an integer; got {n!r} "
                f"for path {path!r}"
            ) from None
        if count < 0:
            raise ValueError(
                f"--expected-diff-lines count must be >= 0; got {count} for "
                f"path {path!r}"
            )
        out[path] = count
    return out


def _staged_diff_line_count(files: list[str], env: dict) -> dict[str, int]:
    """Return added+deleted line count of each file's STAGED diff vs HEAD.

    Uses ``git diff --cached --numstat HEAD -- <files>`` honoring env's
    GIT_INDEX_FILE (so it reads OUR temp index). Binary files (numstat ``-``)
    map to -1 (skip the heuristic). Files with no staged change map to 0.
    """
    out: dict[str, int] = dict.fromkeys(files, 0)
    if not files:
        return out
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--numstat", "HEAD", "--", *files],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
        )
    except OSError:
        return out
    if proc.returncode != 0:
        return out
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added, deleted, path = parts[0], parts[1], parts[2]
        path = path.strip()
        if added == "-" or deleted == "-":
            out[path] = -1  # binary — heuristic not applicable
            continue
        try:
            out[path] = int(added) + int(deleted)
        except ValueError:
            continue
    return out


def _parse_patch_target_files(patch_text: str) -> list[str]:
    """Extract the target file paths a unified diff / git patch touches.

    Reads the ``+++ b/<path>`` headers (falling back to ``diff --git a/x b/x``)
    so the serializer can log + sister-checkpoint the patch's file set without
    a --files argument. Returns paths in first-seen order, de-duplicated.
    """
    files: list[str] = []
    seen: set[str] = set()

    def _add(p: str) -> None:
        p = p.strip()
        if p and p != "/dev/null" and p not in seen:
            seen.add(p)
            files.append(p)

    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            # strip a leading a/ or b/ prefix
            if target.startswith(("a/", "b/")):
                target = target[2:]
            # strip a trailing tab-timestamp (unified-diff dialects)
            target = target.split("\t", 1)[0].strip()
            _add(target)
        elif line.startswith("diff --git "):
            # 'diff --git a/<path> b/<path>' — use the b/ side.
            rest = line[len("diff --git "):].strip()
            toks = rest.split()
            if len(toks) == 2 and toks[1].startswith("b/"):
                _add(toks[1][2:])
    return files


def _git_apply_cached(patch_path: str, env: dict) -> tuple[int, str]:
    """Apply a patch to env's GIT_INDEX_FILE ONLY (``git apply --cached``).

    Patch-file (intent-manifest) staging path — the real fix for shared files
    (incident 2, 2026-07-08). ``--cached`` applies the patch to the temp index
    (seeded from HEAD) and IGNORES the working tree entirely, so a co-mingled /
    clobbered working tree cannot leak foreign hunks into the commit: only the
    caller's declared patch lands. Context lines are validated against the temp
    index (== HEAD), so a patch not cleanly based on HEAD fails LOUDLY here
    (rc != 0) rather than silently mis-applying — a feature.
    """
    cmd = ["git", "apply", "--cached", "--", patch_path]
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


# #1293 LEG 2 — deterministic custody when the managed sandbox denies writes to
# ``.git/objects``.  Four arms on 2026-08-26 hit the same failure while their
# ordinary workspace/SSD writes remained available.  The intended tree already
# exists outside Git (whole-file mode) or as the caller's exact patch
# (``--patch-file``), so losing it is avoidable: reproduce that tree in a
# tiny isolated Git object store, commit there with the source object database
# read-only, retain a bundle + format-patch + typed receipt, and return a
# distinct rc. The fallback runs only AFTER the caller releases the serializer
# lock; helpers below contain no lock operations.
BUNDLE_FALLBACK_RC = 17
BUNDLE_FALLBACK_FAILED_RC = 18
BUNDLE_FALLBACK_STORAGE_REFUSAL_RC = 19
BUNDLE_FALLBACK_MAX_ARTIFACT_BYTES = 64 * 1024**2
BUNDLE_FALLBACK_RECEIPT_ALLOWANCE_BYTES = 128 * 1024
_GIT_OBJECT_PERMISSION_TOKENS = (
    "operation not permitted",
    "permission denied",
    "insufficient permission",
    "read-only file system",
)
_GIT_OBJECT_WRITE_TOKENS = (
    ".git/objects",
    "object database",
    "failed to insert into database",
    "unable to create temporary file",
    "failed to write object",
    "unable to write tree",
)


def _is_git_object_write_denial(output: str) -> bool:
    """True only for Git object-store writes refused by filesystem policy."""

    folded = output.casefold()
    return any(token in folded for token in _GIT_OBJECT_PERMISSION_TOKENS) and any(
        token in folded for token in _GIT_OBJECT_WRITE_TOKENS
    )


class BundleFallbackStorageRefusal(RuntimeError):
    """The bounded fallback cannot safely write its retained SSD artifacts."""

    def __init__(self, message: str, *, receipt_path: Path) -> None:
        super().__init__(message)
        self.receipt_path = receipt_path


def _git_output(
    args: list[str],
    *,
    cwd: Path,
    text_mode: bool = True,
    env: dict[str, str] | None = None,
    input_data: str | bytes | None = None,
) -> str | bytes:
    """Run a fallback-construction Git command or raise with its real output."""

    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=text_mode,
        check=False,
        env=env,
        input=input_data,
    )
    if proc.returncode != 0:
        stdout = proc.stdout if text_mode else proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr if text_mode else proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{' '.join(args)} failed rc={proc.returncode}: {(stdout or '') + (stderr or '')}"
        )
    return proc.stdout


def _head_full_sha() -> str:
    return str(_git_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)).strip()


def _safe_label_component(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label.strip())
    return safe.strip("._") or "anonymous"


def _fallback_receipt_candidates(explicit: str | None, label: str) -> tuple[Path, ...]:
    """Return candidate receipt tiers without writing to any of them."""

    configured = explicit or os.environ.get("SUBAGENT_SERIALIZER_FALLBACK_RECEIPT_DIR")
    if configured:
        return (Path(configured).expanduser().resolve(),)

    arm = _safe_label_component(label)
    candidates: list[Path] = []
    for volume in (Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact")):
        if not volume.is_dir():
            continue
        candidates.append(volume / arm / "receipts" / "commit_serializer_fallbacks")
    if candidates:
        return tuple(candidates)
    raise RuntimeError(
        "no writable SSD receipt tier for bundle fallback; tried "
        "/Volumes/VertigoDataTier/pact then /Volumes/APDataStore/pact"
    )


def _canonical_storage_reserve_bytes() -> int:
    if not _STORAGE_WATERFALL_AVAILABLE or bytes_from_gib is None:
        raise RuntimeError(
            "canonical storage reserve unavailable: comma_lab.storage_tiers did not import"
        )
    return int(bytes_from_gib(DEFAULT_RESERVE_FREE_GB))


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    if not current.exists():
        raise RuntimeError(f"no existing parent for fallback receipt path {path}")
    return current


def _local_fallback_refusal_receipt(row: dict[str, object]) -> Path:
    """Persist a small refusal receipt locally without consuming SSD reserve."""

    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    parent = REPO_ROOT / ".omx" / "state" / "commit_serializer_fallback_refusals"
    attempt = parent / f"{stamp}-{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    receipt = attempt / "receipts.jsonl"
    receipt.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _select_fallback_receipt_parent(
    candidates: tuple[Path, ...], *, projected_bytes: int
) -> tuple[Path, dict[str, object]]:
    """Select the first tier that remains above the canonical reserve."""

    reserve_bytes = _canonical_storage_reserve_bytes()
    statuses: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            probe = _nearest_existing_parent(candidate)
            usage = shutil.disk_usage(probe)
            writable = os.access(probe, os.W_OK)
            free_bytes: int | None = usage.free
            eligible = writable and usage.free - projected_bytes >= reserve_bytes
            error = None
        except OSError as exc:
            probe = candidate
            writable = False
            free_bytes = None
            eligible = False
            error = f"{type(exc).__name__}: {exc}"
        status: dict[str, object] = {
            "candidate": str(candidate),
            "probe_path": str(probe),
            "free_bytes": free_bytes,
            "projected_bytes": projected_bytes,
            "reserve_bytes": reserve_bytes,
            "writable": writable,
            "eligible": eligible,
            "error": error,
        }
        statuses.append(status)
        if eligible:
            return candidate, {"selected": status, "candidates": statuses}

    refusal = {
        "schema": "subagent_commit_bundle_fallback.v1",
        "event_type": "git_object_write_denial_bundle_fallback",
        "status": "BUNDLE_FALLBACK_STORAGE_REFUSED",
        "written_at_utc": _now_iso(),
        "serializer_rc": BUNDLE_FALLBACK_STORAGE_REFUSAL_RC,
        "projected_bytes": projected_bytes,
        "artifact_cap_bytes": BUNDLE_FALLBACK_MAX_ARTIFACT_BYTES,
        "reserve_bytes": reserve_bytes,
        "storage_candidates": statuses,
        "reason": "projected fallback artifacts would breach every candidate tier's reserve",
    }
    receipt = _local_fallback_refusal_receipt(refusal)
    raise BundleFallbackStorageRefusal(
        f"fallback storage reserve refused projected_bytes={projected_bytes} "
        f"reserve_bytes={reserve_bytes}",
        receipt_path=receipt,
    )


def _path_mount_context(path: Path) -> dict[str, str | None]:
    """Best-effort mountpoint/flags diagnosis; failure is recorded, never hidden."""

    context: dict[str, str | None] = {"path": str(path), "mountpoint": None, "mount_line": None}
    try:
        df = subprocess.run(
            ["df", "-P", str(path)], capture_output=True, text=True, check=False, timeout=10
        )
        lines = [line for line in df.stdout.splitlines() if line.strip()]
        if df.returncode == 0 and len(lines) >= 2:
            context["mountpoint"] = lines[-1].split()[-1]
        mounts = subprocess.run(
            ["mount"], capture_output=True, text=True, check=False, timeout=10
        )
        mountpoint = context["mountpoint"]
        if mounts.returncode == 0 and mountpoint:
            needle = f" on {mountpoint} "
            context["mount_line"] = next(
                (line for line in mounts.stdout.splitlines() if needle in line), None
            )
    except (OSError, subprocess.SubprocessError):
        pass
    return context


def _snapshot_intended_files(
    files: list[str], env: dict, *, from_index: bool
) -> dict[str, tuple[bytes | None, str]]:
    """Capture exactly what the failed serializer invocation meant to land.

    ``None`` means deletion.  Git modes are retained so executable bits and
    symlinks survive the fallback.  Whole-file mode reads the working tree,
    matching ``git add -- <files>``; ``--no-stage`` reads the caller's index.
    """

    snapshot: dict[str, tuple[bytes | None, str]] = {}
    for rel in files:
        if from_index:
            ls = subprocess.run(
                ["git", "ls-files", "--stage", "--", rel],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
            )
            if ls.returncode != 0:
                raise RuntimeError(f"cannot read staged entry for {rel}: {ls.stderr.strip()}")
            if not ls.stdout.strip():
                snapshot[rel] = (None, "000000")
                continue
            first = ls.stdout.splitlines()[0]
            prefix = first.split("\t", 1)[0].split()
            if len(prefix) < 2:
                raise RuntimeError(f"malformed staged entry for {rel}: {first!r}")
            mode, oid = prefix[0], prefix[1]
            blob = subprocess.run(
                ["git", "cat-file", "blob", oid],
                cwd=REPO_ROOT, env=env, capture_output=True, check=False,
            )
            if blob.returncode != 0:
                raise RuntimeError(f"cannot read staged blob for {rel}: rc={blob.returncode}")
            snapshot[rel] = (blob.stdout, mode)
            continue

        path = REPO_ROOT / rel
        if path.is_symlink():
            snapshot[rel] = (os.readlink(path).encode("utf-8"), "120000")
        elif path.is_file():
            mode = "100755" if os.access(path, os.X_OK) else "100644"
            snapshot[rel] = (path.read_bytes(), mode)
        elif not path.exists():
            snapshot[rel] = (None, "000000")
        else:
            raise RuntimeError(
                f"bundle fallback requires explicit files, not directory target {rel!r}"
            )
    return snapshot


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fallback_git_environment(git_dir: Path) -> dict[str, str]:
    source_name = str(
        _git_output(["git", "config", "--get", "user.name"], cwd=REPO_ROOT)
    ).strip()
    source_email = str(
        _git_output(["git", "config", "--get", "user.email"], cwd=REPO_ROOT)
    ).strip()
    env = os.environ.copy()
    env.update(
        {
            "GIT_DIR": str(git_dir),
            "GIT_INDEX_FILE": str(git_dir / "fallback.index"),
            "GIT_AUTHOR_NAME": source_name or "Pact Serializer Fallback",
            "GIT_AUTHOR_EMAIL": source_email or "serializer-fallback@localhost.invalid",
            "GIT_COMMITTER_NAME": source_name or "Pact Serializer Fallback",
            "GIT_COMMITTER_EMAIL": source_email or "serializer-fallback@localhost.invalid",
        }
    )
    return env


def _write_snapshot_to_fallback_index(
    snapshot: dict[str, tuple[bytes | None, str]], env: dict[str, str]
) -> None:
    """Write only the declared blobs into the isolated fallback object store."""

    for rel, (payload, mode) in snapshot.items():
        if payload is None:
            _git_output(["git", "update-index", "--force-remove", "--", rel], cwd=REPO_ROOT, env=env)
            continue
        oid = str(
            _git_output(
                ["git", "hash-object", "-w", "--stdin"],
                cwd=REPO_ROOT,
                text_mode=False,
                env=env,
                input_data=payload,
            )
            .decode("ascii")
            .strip()
        )
        _git_output(
            ["git", "update-index", "--add", "--cacheinfo", f"{mode},{oid},{rel}"],
            cwd=REPO_ROOT,
            env=env,
        )


def _fallback_file_shas(
    fallback_commit: str, files: list[str], env: dict[str, str]
) -> dict[str, str]:
    shas: dict[str, str] = {}
    for rel in files:
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{fallback_commit}:{rel}"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            check=False,
        )
        if exists.returncode != 0:
            shas[rel] = "DELETED"
            continue
        payload = _git_output(
            ["git", "cat-file", "blob", f"{fallback_commit}:{rel}"],
            cwd=REPO_ROOT,
            env=env,
            text_mode=False,
        )
        assert isinstance(payload, bytes)
        shas[rel] = hashlib.sha256(payload).hexdigest()
    return shas


def _artifact_cap_refusal(*, projected_bytes: int, label: str) -> None:
    row: dict[str, object] = {
        "schema": "subagent_commit_bundle_fallback.v1",
        "event_type": "git_object_write_denial_bundle_fallback",
        "status": "BUNDLE_FALLBACK_STORAGE_REFUSED",
        "written_at_utc": _now_iso(),
        "label": label,
        "serializer_rc": BUNDLE_FALLBACK_STORAGE_REFUSAL_RC,
        "projected_bytes": projected_bytes,
        "artifact_cap_bytes": BUNDLE_FALLBACK_MAX_ARTIFACT_BYTES,
        "reason": "fallback artifacts exceed the bounded artifact cap",
    }
    receipt = _local_fallback_refusal_receipt(row)
    raise BundleFallbackStorageRefusal(
        f"fallback artifact cap refused projected_bytes={projected_bytes} "
        f"cap_bytes={BUNDLE_FALLBACK_MAX_ARTIFACT_BYTES}",
        receipt_path=receipt,
    )


def _author_bundle_fallback(
    *,
    files: list[str],
    final_message: str,
    label: str,
    original_rc: int,
    original_output: str,
    fallback_receipt_dir: str | None,
    intended_snapshot: dict[str, tuple[bytes | None, str]] | None,
    patch_bytes: bytes | None,
    allow_empty: bool,
) -> dict[str, object]:
    """Build a thin bundle without cloning or copying the checkout."""

    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    base_head = _head_full_sha()
    attempt: Path | None = None
    receipt_path: Path | None = None

    try:
        with tempfile.TemporaryDirectory(prefix=".serializer-bundle-build-") as scratch_text:
            scratch = Path(scratch_text)
            git_dir = scratch / "fallback.git"
            object_format = str(
                _git_output(["git", "rev-parse", "--show-object-format"], cwd=REPO_ROOT)
            ).strip()
            init_cmd = ["git", "init", "--quiet", "--bare"]
            if object_format and object_format != "sha1":
                init_cmd.append(f"--object-format={object_format}")
            init_cmd.append(str(git_dir))
            _git_output(init_cmd, cwd=scratch)

            common_dir = _git_common_dir()
            if common_dir is None:
                raise RuntimeError("cannot resolve source Git common directory")
            source_objects = (common_dir / "objects").resolve()
            alternates = git_dir / "objects" / "info" / "alternates"
            alternates.parent.mkdir(parents=True, exist_ok=True)
            alternates.write_text(str(source_objects) + "\n", encoding="utf-8")
            env = _fallback_git_environment(git_dir)
            _git_output(["git", "read-tree", base_head], cwd=REPO_ROOT, env=env)

            local_intended_patch = scratch / "intended-tree.patch"
            if patch_bytes is not None:
                local_intended_patch.write_bytes(patch_bytes)
                _git_output(
                    ["git", "apply", "--cached", "--binary", str(local_intended_patch)],
                    cwd=REPO_ROOT,
                    env=env,
                )
            elif intended_snapshot is not None:
                _write_snapshot_to_fallback_index(intended_snapshot, env)
                intended_patch = _git_output(
                    ["git", "diff", "--cached", "--binary", base_head, "--", *files],
                    cwd=REPO_ROOT,
                    text_mode=False,
                    env=env,
                )
                assert isinstance(intended_patch, bytes)
                local_intended_patch.write_bytes(intended_patch)
            else:
                raise RuntimeError("fallback has neither an intended snapshot nor a patch")

            tree = str(_git_output(["git", "write-tree"], cwd=REPO_ROOT, env=env)).strip()
            parent_tree = str(
                _git_output(["git", "rev-parse", f"{base_head}^{{tree}}"], cwd=REPO_ROOT)
            ).strip()
            if tree == parent_tree and not allow_empty:
                raise RuntimeError("fallback intended commit is empty without --allow-empty")
            fallback_commit = str(
                _git_output(
                    ["git", "commit-tree", tree, "-p", base_head, "-m", final_message],
                    cwd=REPO_ROOT,
                    env=env,
                )
            ).strip()
            _git_output(
                ["git", "update-ref", "refs/heads/serializer-fallback", fallback_commit],
                cwd=REPO_ROOT,
                env=env,
            )

            local_bundle = scratch / "intended-commit.bundle"
            _git_output(
                [
                    "git", "bundle", "create", str(local_bundle),
                    "refs/heads/serializer-fallback", f"^{base_head}",
                ],
                cwd=REPO_ROOT,
                env=env,
            )
            local_format_patch = scratch / "intended-commit.format-patch"
            format_patch = _git_output(
                [
                    "git", "format-patch", "--binary", "--stdout", "--no-signature",
                    f"{base_head}..{fallback_commit}",
                ],
                cwd=REPO_ROOT,
                text_mode=False,
                env=env,
            )
            assert isinstance(format_patch, bytes)
            local_format_patch.write_bytes(format_patch)
            _git_output(["git", "bundle", "verify", str(local_bundle)], cwd=REPO_ROOT)
            committed_file_shas = _fallback_file_shas(fallback_commit, files, env)

            projected_bytes = (
                local_bundle.stat().st_size
                + local_format_patch.stat().st_size
                + local_intended_patch.stat().st_size
                + BUNDLE_FALLBACK_RECEIPT_ALLOWANCE_BYTES
            )
            if projected_bytes > BUNDLE_FALLBACK_MAX_ARTIFACT_BYTES:
                _artifact_cap_refusal(projected_bytes=projected_bytes, label=label)
            parent, storage_check = _select_fallback_receipt_parent(
                _fallback_receipt_candidates(fallback_receipt_dir, label),
                projected_bytes=projected_bytes,
            )

            attempt = parent / f"{stamp}-{os.getpid()}"
            attempt.mkdir(parents=True, exist_ok=False)
            bundle_path = attempt / local_bundle.name
            format_patch_path = attempt / local_format_patch.name
            intended_patch_path = attempt / local_intended_patch.name
            receipt_path = attempt / "receipts.jsonl"
            shutil.copyfile(local_bundle, bundle_path)
            shutil.copyfile(local_format_patch, format_patch_path)
            shutil.copyfile(local_intended_patch, intended_patch_path)

        git_dir = _git_common_dir()
        objects = git_dir / "objects" if git_dir is not None else None
        row: dict[str, object] = {
            "schema": "subagent_commit_bundle_fallback.v1",
            "event_type": "git_object_write_denial_bundle_fallback",
            "status": "BUNDLE_READY_MAIN_MUST_LAND",
            "written_at_utc": _now_iso(),
            "label": label,
            "base_head": base_head,
            "fallback_commit": fallback_commit,
            "bundle_path": str(bundle_path),
            "bundle_bytes": bundle_path.stat().st_size,
            "bundle_sha256": _sha256_path(bundle_path),
            "format_patch_path": str(format_patch_path),
            "format_patch_bytes": format_patch_path.stat().st_size,
            "format_patch_sha256": _sha256_path(format_patch_path),
            "intended_patch_path": str(intended_patch_path),
            "intended_patch_bytes": intended_patch_path.stat().st_size,
            "intended_patch_sha256": _sha256_path(intended_patch_path),
            "construction_mode": "isolated_git_plumbing_no_checkout",
            "artifact_cap_bytes": BUNDLE_FALLBACK_MAX_ARTIFACT_BYTES,
            "projected_artifact_bytes": projected_bytes,
            "storage_reserve": storage_check,
            "files": [
                {"path": rel, "content_sha256": committed_file_shas.get(rel, "UNKNOWN")}
                for rel in files
            ],
            "failure": {
                "serializer_rc": original_rc,
                "output": original_output[-4000:],
            },
            "environment": {
                "cwd": str(Path.cwd()),
                "repo_root": str(REPO_ROOT),
                "uid": os.getuid(),
                "gid": os.getgid(),
                "platform": sys.platform,
                "sandbox_markers": {
                    key: os.environ[key]
                    for key in (
                        "CODEX_SANDBOX",
                        "CODEX_SANDBOX_NETWORK_DISABLED",
                        "SUBAGENT_SERIALIZER_REPO_ROOT",
                    )
                    if key in os.environ
                },
                "git_dir": None if git_dir is None else str(git_dir),
                "git_objects": None if objects is None else str(objects),
                "git_objects_mode": (
                    None if objects is None or not objects.exists() else oct(objects.stat().st_mode & 0o777)
                ),
                "repo_mount": _path_mount_context(REPO_ROOT),
                "receipt_mount": _path_mount_context(attempt),
            },
        }
        receipt_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        row["receipt_path"] = str(receipt_path)
        return row
    except BundleFallbackStorageRefusal:
        raise
    except Exception as exc:
        failure = {
            "schema": "subagent_commit_bundle_fallback.v1",
            "event_type": "git_object_write_denial_bundle_fallback",
            "status": "BUNDLE_FALLBACK_FAILED",
            "written_at_utc": _now_iso(),
            "label": label,
            "base_head": base_head,
            "error": f"{type(exc).__name__}: {exc}",
            "attempt_dir": None if attempt is None else str(attempt),
        }
        if receipt_path is None:
            receipt_path = _local_fallback_refusal_receipt(failure)
        else:
            receipt_path.write_text(json.dumps(failure, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"{failure['error']} (receipt {receipt_path})") from exc


# Triality-legs disposition (CANONICALIZATION UNIT 2, task #388). A STRUCTURED
# alternative to the ad-hoc ``[no-triality]`` commit-message token: the caller
# declares which triality legs this commit touched (dag / dsl / equations), or
# ``none`` + a reason for a deliberate chore/apparatus commit. Purely additive —
# absent flag == today's behavior. Recorded into the JSONL log so the
# triality_drift_detector Stop hook can soften a core-drift block when a
# disposition was structurally declared (vs firing on an ad-hoc opt-out token).
_VALID_TRIALITY_LEGS = ("dag", "dsl", "equations", "none")


def _parse_triality_legs(
    raw: str | None, reason: str | None
) -> tuple[list[str] | None, str | None]:
    """Parse ``--triality-legs <csv>`` (+ ``--triality-reason``).

    Returns ``(legs, reason)``. ``raw is None`` (flag absent) ⇒ ``(None, None)``,
    the backward-compatible no-op. Values must be a subset of
    ``dag,dsl,equations,none``; ``none`` REQUIRES a non-empty ``reason`` and cannot
    be combined with other legs. Raises ``ValueError`` on malformed input.
    """
    if raw is None:
        return None, None
    legs = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not legs:
        raise ValueError(
            "--triality-legs was empty — omit the flag entirely, or pass a "
            f"comma-separated subset of {list(_VALID_TRIALITY_LEGS)}"
        )
    bad = [x for x in legs if x not in _VALID_TRIALITY_LEGS]
    if bad:
        raise ValueError(
            f"--triality-legs values must be from {list(_VALID_TRIALITY_LEGS)}; "
            f"got unknown {bad!r}"
        )
    # Dedupe, preserving first-seen order.
    seen: set[str] = set()
    uniq = [x for x in legs if not (x in seen or seen.add(x))]
    if "none" in uniq:
        if len(uniq) > 1:
            raise ValueError(
                "--triality-legs 'none' is a deliberate opt-out and cannot combine "
                f"with other legs; got {uniq!r}"
            )
        if not (reason and reason.strip()):
            raise ValueError(
                "--triality-legs none REQUIRES --triality-reason <str> "
                "(a deliberate chore/apparatus opt-out must state why)"
            )
    return uniq, (reason.strip() if reason and reason.strip() else None)


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


def _review_gate_override_python_targets(
    files: list[str], *, no_stage: bool
) -> list[str]:
    """Return Python targets that make ``REVIEW_GATE_OVERRIDE=1`` illegal.

    The override is reserved for non-code state/document landings.  A mixed
    Markdown+Python serializer invocation previously inherited the override
    into the pre-commit hook and silently bypassed review for the Python files
    (harness failure ``review_gate_override_on_py_commit_20260711``).  Enforce
    the boundary at the serializer, before staging or lock acquisition.

    ``--no-stage`` callers may omit ``--files``; in that mode inspect the real
    staged index too so an empty positional file list cannot bypass the guard.
    """
    if os.environ.get("REVIEW_GATE_OVERRIDE", "0") != "1":
        return []

    candidates = set(files)
    if no_stage:
        try:
            proc = subprocess.run(
                [
                    "git",
                    "diff",
                    "--cached",
                    "--name-only",
                    "--diff-filter=ACMR",
                    "--",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            # Fail closed: an override with an unreadable staged file set has
            # no proof that it is non-code.
            return ["<staged-file-set-unreadable>"]
        if proc.returncode != 0:
            return ["<staged-file-set-unreadable>"]
        candidates.update(
            line.strip() for line in proc.stdout.splitlines() if line.strip()
        )

    return sorted(path for path in candidates if Path(path).suffix == ".py")


def _venv_integrity_blockers(repo_root: Path) -> list[str]:
    """LOUD first-commit canary for metadata-loss damage in the live venv."""

    venv = repo_root / ".venv"
    if not venv.exists():
        if (repo_root / "tools/subagent_commit_serializer.py").is_file():
            return ["venv_missing"]
        return []
    blockers: list[str] = []
    python3 = venv / "bin" / "python3"
    ruff = venv / "bin" / "ruff"
    try:
        python3.resolve(strict=True)
    except OSError as exc:
        blockers.append(f"python3_does_not_resolve:{type(exc).__name__}:{exc}")
    else:
        if not os.access(python3, os.X_OK):
            blockers.append("python3_not_executable")
        else:
            probe = subprocess.run(
                [str(python3), "-c", "import sys; print(sys.executable)"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if probe.returncode != 0:
                blockers.append(f"python3_exec_failed:rc={probe.returncode}:{probe.stderr.strip()}")
    if not ruff.is_file():
        blockers.append("ruff_missing")
    elif not os.access(ruff, os.X_OK):
        blockers.append("ruff_not_executable")
    else:
        probe = subprocess.run(
            [str(ruff), "--version"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if probe.returncode != 0 or not probe.stdout.strip().lower().startswith("ruff "):
            blockers.append(
                f"ruff_exec_failed:rc={probe.returncode}:"
                f"{(probe.stderr or probe.stdout).strip()}"
            )
    return blockers


def main(rebind_root: bool = False) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--message", "-m", required=True,
        help="Commit message (passed to `git commit -m`).",
    )
    parser.add_argument(
        "--files", "-f", nargs="*", action="extend", default=None,
        help="Files to stage. Required UNLESS --stdin-files OR --no-stage is "
             "passed. Repeated --files flags ACCUMULATE (extend) — a repeated "
             "flag never silently drops earlier files. One flag with many "
             "paths (`--files a b c`) is equivalent.",
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
        "--label", default=os.environ.get("SUBAGENT_LABEL"),
        help="Subagent label for log forensics (default: $SUBAGENT_LABEL, then "
             "the unique active checkpoint covering every staged file, then 'anonymous').",
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="Repo/worktree root to operate on (WORKTREE-AWARE FIX 2026-07-17). "
             "Default: auto-detect from CWD via `git rev-parse --show-toplevel`, "
             "so running from inside a git WORKTREE commits into THAT worktree "
             "(its own index) instead of silently staging the main checkout's "
             "copy. Falls back to this file's checkout when git is unavailable. "
             "Also settable via $SUBAGENT_SERIALIZER_REPO_ROOT.",
    )
    parser.add_argument(
        "--fallback-receipt-dir",
        default=None,
        help=(
            "#1293 git-object denial custody root. On an object-store write denial, "
            "the serializer releases its lock, authors the intended commit in a "
            "isolated Git object store, and retains a bundle + format-patch + typed receipt "
            "under this directory (rc=17). Default: "
            "$SUBAGENT_SERIALIZER_FALLBACK_RECEIPT_DIR, then the arm label under "
            "/Volumes/VertigoDataTier/pact, then /Volumes/APDataStore/pact."
        ),
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
        "--merge-commit", action="store_true",
        help="Declare that THIS commit is the completion of an in-progress "
             "`git merge --no-commit`. Without it the serializer REFUSES "
             "(rc=16) whenever .git/MERGE_HEAD exists, because git would "
             "attach the merge as a second parent while committing only your "
             "own files (the ddm_oc2 false-second-parent incident). Even with "
             "it, the staged set must cover every file the merge branch "
             "changed, or the commit is still refused.",
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
        "--base-content-sha256",
        action="append",
        default=None,
        help=(
            "FIX-ABSORPTION (2026-07-07): declare the SHA-256 of a file's "
            "content BEFORE your own edits began ('<relpath>=<sha256>'; the "
            "literal '<relpath>=new' for a file you created). Repeatable "
            "per-file. The serializer refuses (rc=6) when the declared base "
            "differs from the file's content at HEAD — that means the file "
            "contained a sibling's uncommitted hunks when you began editing "
            "(whole-file staging would ABSORB them under your commit body: "
            "the serializer_whole_file_staging_absorbs_sibling_hunks class, "
            "incident commits 1d6704e5b/049aa0d9f), or HEAD moved past your "
            "base (staging would REVERT the sibling's landed hunks). "
            "Retry after the sibling lands: once HEAD matches your base the "
            "check passes. Pair with --expected-content-sha256 (post-edit "
            "sha) — base guards the edit-start surface, expected guards the "
            "edit-to-lock window."
        ),
    )
    parser.add_argument(
        "--patch-file",
        default=None,
        help=(
            "FIX-CLOBBER intent-manifest mode (2026-07-08 Catalog #405): supply "
            "a unified diff / git patch whose hunks are EXACTLY what you intend "
            "to commit. The serializer applies it with `git apply --cached` to a "
            "temp index seeded from HEAD (the WORKING TREE is ignored entirely), "
            "so a co-mingled or clobbered working tree cannot leak a sibling's "
            "hunks into your commit body — the real fix for editing shared hot "
            "files. Generate it with `git diff HEAD -- <file>` (your hunks only) "
            "or `git add -p <file>` then `git diff --cached -- <file>`. When set, "
            "--files is derived from the patch and the working-tree sha checks "
            "(rc=4/5/6) are skipped as inapplicable; post-commit HEAD verification "
            "(rc=7) still runs if you pass --expected-content-sha256. A patch not "
            "cleanly based on HEAD fails LOUDLY at apply time."
        ),
    )
    parser.add_argument(
        "--expected-diff-lines",
        action="append",
        default=None,
        help=(
            "Catalog #405 warn-only hunk-attribution heuristic: hint the "
            "added+deleted line count of your OWN edits per file "
            "('<relpath>=<int>', repeatable). If the actually-staged diff is "
            "grossly larger (>2x) than the hint, the serializer WARNS + logs "
            "(never refuses) — a whole-file `git add` that swept a sibling's "
            "hunks in shows up as a gross diff-size overshoot. For a hard "
            "guarantee on shared files use --patch-file instead."
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
    parser.add_argument(
        "--triality-legs",
        default=None,
        help=(
            "CANONICALIZATION UNIT 2 (task #388): OPTIONAL structured triality "
            "disposition — a comma-separated subset of "
            f"{list(_VALID_TRIALITY_LEGS)!r} declaring which triality legs this "
            "commit touched (dag=trajectory / dsl=control / equations=law), OR "
            "'none' (a deliberate chore/apparatus commit; REQUIRES "
            "--triality-reason). Recorded into the serializer JSONL log so the "
            "triality drift-detector Stop hook softens a core-drift block when a "
            "disposition was structurally declared. Absent flag == today's "
            "behavior (fully backward-compatible)."
        ),
    )
    parser.add_argument(
        "--triality-reason",
        default=None,
        help=(
            "Rationale required when '--triality-legs none' is used (the reason "
            "the commit legitimately touches no triality leg)."
        ),
    )
    parser.add_argument(
        "--allow-shared-doc-shrink",
        default=None,
        metavar="RATIONALE",
        help=(
            "PERMANENT FIX (2026-07-18) escape hatch for the protected "
            "append-doc CLOBBER guard (rc=14). Multi-writer append-heavy "
            "research docs (completeness matrix / sub015_DAG / DAG-FEED blocks "
            "/ canonical_equations_registry) never SHRINK in normal operation; "
            "a net line loss is the whole-file-clobber-off-a-stale-base "
            "signature (2026-07-18 matrix incident). The serializer refuses "
            "such a commit unless you declare an INTENTIONAL shrink here "
            "(e.g. a consolidation pass). Requires a real rationale string "
            "(placeholder '<rationale>'/'<reason>' literals are rejected). The "
            "RIGHT fix for a stale-base clobber is a 3-way `git merge-file`, "
            "NOT this flag — use the flag only for a deliberate reduction."
        ),
    )
    args = parser.parse_args()

    # WORKTREE-AWARE FIX (2026-07-17): resolve the EFFECTIVE repo root from the
    # caller's CWD (or --repo-root / env) and rebind the module globals that
    # every downstream git op, lock, log, temp-index, and content-hash reads.
    # Without this the serializer always operated on the MAIN checkout (root
    # derived from __file__), so a commit attempted from inside a git worktree
    # silently staged main's copy of the file and the caller's post-edit
    # --expected-content-sha256 could never match (spurious rc=4). A worktree
    # has its own index, so per-working-tree locking preserves the anti-swap
    # guarantee (the race is per-index).
    #
    # Rebind ONLY on the real CLI entry path (`rebind_root=True`, set by the
    # __main__ guard). In-process callers (tests that patch REPO_ROOT to a
    # throwaway repo and call main() directly) pass rebind_root=False so their
    # patched globals are respected — cwd-auto-detect must not clobber them.
    if rebind_root:
        global REPO_ROOT, LOCK_PATH, LOG_PATH
        REPO_ROOT = _resolve_effective_repo_root(args.repo_root)
        LOCK_PATH = REPO_ROOT / ".omx/state/.commit-lock"
        LOG_PATH = REPO_ROOT / ".omx/state/commit-serializer.log"

    try:
        _venv_blockers = _venv_integrity_blockers(REPO_ROOT)
    except (OSError, subprocess.SubprocessError) as exc:
        _venv_blockers = [f"probe_failed:{type(exc).__name__}:{exc}"]
    if _venv_blockers:
        print(
            "[subagent-commit-serializer] VENV-INTEGRITY REFUSED: "
            + "; ".join(_venv_blockers),
            file=sys.stderr,
        )
        return 20

    # FIX-CLOBBER (2026-07-08 Catalog #405): patch-file (intent-manifest) mode.
    # When --patch-file is set the working tree is NOT the source of truth; the
    # patch text is. Derive the file set from the patch so logging + the sister
    # checkpoint scan still see it, and stage via `git apply --cached` (not a
    # whole-file `git add`).
    patch_mode = bool(args.patch_file)
    patch_text = ""
    if patch_mode:
        try:
            patch_text = Path(args.patch_file).read_text()
        except OSError as exc:
            parser.error(f"--patch-file could not be read: {exc!s}")
        if not patch_text.strip():
            parser.error(f"--patch-file {args.patch_file!r} is empty")

    # Resolve file list
    files: list[str] = list(args.files or [])
    if args.stdin_files:
        for line in sys.stdin:
            line = line.strip()
            if line:
                files.append(line)

    if patch_mode and not files:
        # Derive the committed file set from the patch headers.
        files = _parse_patch_target_files(patch_text)
        if not files:
            parser.error(
                "--patch-file has no recognizable '+++ b/<path>' / 'diff --git' "
                "target headers — cannot determine which files it commits"
            )

    if not args.no_stage and not patch_mode and not files:
        parser.error(
            "must pass --files or --stdin-files (or --no-stage if files are "
            "already staged, or --patch-file for intent-manifest staging)"
        )

    if not args.label:
        inferred_label = None
        if infer_current_subagent_id is not None and files:
            try:
                inferred_label = infer_current_subagent_id(list(files))
            except CorruptCheckpointError:
                # The canonical guard below owns the fail-closed rc=11 path.
                # Do not turn corrupt state into an inferred self-exclusion.
                inferred_label = None
        args.label = inferred_label or "anonymous"

    # SHIFT-LEFT triality-leg advisory (ddm_hw1, task #785): the Stop-hook legs
    # (missing_legs + consumer_leg in triality_drift_detector) fire AFTER the commit
    # lands, so a missed leg is a re-finish (the "4 backstop fires in one day" tax).
    # This prints the SAME suggestion BEFORE the commit, so the author tags/settles
    # now. Fail-open + read-only: never blocks, never mutates the commit; any error
    # (import failure / git error / worktree quirk) prints nothing.
    try:
        import importlib.util as _ilu

        _tdd_path = Path(__file__).resolve().parent / "triality_drift_detector.py"
        _spec = _ilu.spec_from_file_location("_serializer_tdd_shiftleft", _tdd_path)
        _tdd = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_tdd)
        _dsl_files = [f for f in files if str(f).startswith("src/tac/witness_dsl/")]
        _dsl_diff = ""
        if _dsl_files:
            _dsl_diff = subprocess.run(  # subprocess-no-check-OK: advisory drift-detector diff; the detector is fail-open by design
                ["git", "diff", "HEAD", "--", *_dsl_files],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        _owed = _tdd.owed_legs_line(args.message.splitlines()[0], files, _dsl_diff)
        if _owed:
            print(f"[subagent-commit-serializer] SHIFT-LEFT {_owed}", file=sys.stderr)
    except Exception:
        pass

    # FALSIFIED-PREMISE memo advisory (fpr1/ea1 arc, 2026-08-31): the curated
    # registry guarded charter SPAWNS while the day's propagation vector was
    # MAIN-authored MEMOS (jt1 restated a number its owning memo had published a
    # do-not-cite list against, and crossed no lint — memos never touch the spawn
    # path). This runs the SAME canonical matcher (tools/premise_lint.py) over
    # staged .omx/research/*.md content BEFORE the commit. Registered-premise
    # findings remain advisory. SHA prefix-match/divergent-tail findings are a
    # separate refusing leg: correction memos must abbreviate a bad historical
    # token below 16 hex or remove it, never republish the transcription error.
    _sha_transcription_refusals: list[str] = []
    try:
        import importlib.util as _ilu_pl

        _pl_path = Path(__file__).resolve().parent / "premise_lint.py"
        _pl_spec = _ilu_pl.spec_from_file_location("_serializer_premise_lint", _pl_path)
        _pl = _ilu_pl.module_from_spec(_pl_spec)
        _pl_spec.loader.exec_module(_pl)
        for _memo in files:
            _m = str(_memo)
            if not (_m.startswith(".omx/research/") and _m.endswith(".md")):
                continue
            _memo_path = REPO_ROOT / _m
            if not _memo_path.is_file():
                continue
            _memo_text = _memo_path.read_text(encoding="utf-8")
            for _warn in _pl.lint_text(
                _memo_text,
                subject=f"staged memo {_memo_path.name}",
            ):
                print(f"[subagent-commit-serializer] PREMISE-LINT {_warn}", file=sys.stderr)
            _sha_transcription_refusals.extend(
                _pl.lint_sha_prefix_divergent_tails(
                    _memo_text,
                    canonical_shas=_pl.canonical_frontier_shas(
                        REPO_ROOT / ".omx/state/canonical_frontier_pointer.json"
                    ),
                    subject=f"staged memo {_memo_path.name}",
                )
            )
    except Exception as exc:
        _sha_transcription_refusals.append(
            f"SHA-transcription lint unavailable ({type(exc).__name__}: {exc})"
        )
    if _sha_transcription_refusals:
        for _refusal in _sha_transcription_refusals:
            print(
                f"[subagent-commit-serializer] SHA-LINT REFUSED {_refusal}",
                file=sys.stderr,
            )
        return 21

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
        # FIX-ABSORPTION (2026-07-07): log whether the caller declared its
        # edit BASE so forensics can distinguish base-guarded commits from
        # legacy sha-only commits when the absorption class fires again.
        "base_content_sha256_present": bool(args.base_content_sha256),
        "base_content_sha256_file_count": (
            len(args.base_content_sha256) if args.base_content_sha256 else 0
        ),
        # FIX-CLOBBER (2026-07-08 Catalog #405): record whether the caller used
        # the intent-manifest patch path + the diff-line hint so forensics can
        # distinguish patch-staged commits from whole-file `git add` commits.
        "patch_mode": bool(args.patch_file),
        "expected_diff_lines_present": bool(args.expected_diff_lines),
    }

    # Harness-failure class fix (2026-07-15): REVIEW_GATE_OVERRIDE is allowed
    # only for non-code state/doc landings.  The hook itself retains its
    # operator escape hatch, but the canonical subagent serializer refuses to
    # carry that escape hatch across any Python target.  Python must pass the
    # normal review tracker policy.
    override_python_targets = _review_gate_override_python_targets(
        files, no_stage=bool(args.no_stage)
    )
    if override_python_targets:
        _append_log({
            **base_record,
            "outcome": "review_gate_override_on_python_rejected",
            "python_targets": override_python_targets,
        })
        print(
            "[subagent-commit-serializer] REFUSED (rc=12): "
            "REVIEW_GATE_OVERRIDE=1 cannot be used when the commit includes "
            "Python. Split non-code state/docs into a separate commit or unset "
            "the override and satisfy the review gate. Python targets: "
            f"{override_python_targets!r}",
            file=sys.stderr,
        )
        return 12

    # PERMANENT FIX (2026-07-18): gitignored-file silent whole-commit abort.
    # A single gitignored path in --files poisons `git add` and aborts the
    # entire commit with only a generic git hint (no named culprit). Refuse
    # PRE-LOCK, naming every offending file, so the caller removes it (bulk /
    # rebuildable artifacts belong on the SSD cold-store, not in git). Skipped
    # in patch mode (--files is derived from the patch, not staged directly).
    if not args.no_stage and not args.patch_file:
        rc13 = _refuse_gitignored(_check_ignored_files(files), base_record)
        if rc13 is not None:
            return rc13

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
        base_content_shas = _parse_base_content_sha256(
            args.base_content_sha256 or []
        )
        expected_diff_lines = _parse_expected_diff_lines(
            args.expected_diff_lines or []
        )
        # CANONICALIZATION UNIT 2 (task #388): structured triality disposition.
        # Absent flag → (None, None) → no base_record change (backward-compatible).
        triality_legs, triality_reason = _parse_triality_legs(
            args.triality_legs, args.triality_reason
        )
    except ValueError as exc:
        print(f"[subagent-commit-serializer] FATAL: {exc!s}", file=sys.stderr)
        return 2

    # Only stamp triality keys when the caller actually declared a disposition, so
    # log rows for existing (flag-absent) callers stay byte-identical.
    if triality_legs is not None:
        base_record["triality_legs"] = triality_legs
        base_record["triality_reason"] = triality_reason

    # FIX-ABSORPTION (2026-07-07) pre-lock check: declared edit BASE vs the
    # file's content at HEAD. See _base_content_check docstring — this is the
    # check that would have refused incident commits 1d6704e5b / 049aa0d9f
    # (whole-file staging absorbed a sibling's uncommitted trainer + DSL
    # hunks; Catalog #157/#216 passed by construction because the caller's
    # expected sha was computed on the already-merged working tree).
    if base_content_shas and not patch_mode:
        if not expected_content_shas:
            print(
                "[subagent-commit-serializer] NOTE: --base-content-sha256 "
                "without --expected-content-sha256 — the base guards the "
                "edit-START surface only; pair it with the post-edit sha so "
                "the edit-to-lock window is guarded too.",
                file=sys.stderr,
            )
        base_diffs = _base_content_check(base_content_shas)
        if base_diffs:
            _append_log({
                **base_record,
                "outcome": "base_content_sha_mismatch_pre_lock",
                "base_content_sha_diffs": {
                    f: {"declared_base": want, "head_blob": got}
                    for f, (want, got) in base_diffs.items()
                },
            })
            print(
                "[subagent-commit-serializer] REFUSED (rc=6): "
                "--base-content-sha256 mismatch vs HEAD. Your declared "
                "edit base is NOT what these files look like at HEAD — the "
                "file contained a sibling's uncommitted hunks when you began "
                "editing (whole-file staging would ABSORB them under your "
                "commit body), or HEAD has moved past your base (staging "
                "would REVERT the sibling's landed hunks). The "
                "serializer_whole_file_staging_absorbs_sibling_hunks class "
                "(incident 1d6704e5b/049aa0d9f 2026-07-07). Wait for the "
                "sibling to land, then retry — once HEAD matches your base "
                "this check passes. Files affected: "
                f"{list(base_diffs.keys())!r}",
                file=sys.stderr,
            )
            return 6

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
    if files and not expected_content_shas and not patch_mode:
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
    if expected_content_shas and not patch_mode:
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
    if not args.no_concurrent_edit_check and not args.no_stage and files and not patch_mode:
        pre_lock_hashes = _hash_working_tree_files(files)

    # Operator NON-NEGOTIABLE 2026-05-31: NO co-author trailer EVER — never
    # append a Co-Authored-By line. (`--no-co-author` is a no-op since the
    # trailer is never appended regardless.) ONE structural exception to
    # verbatim commit: the Catalog #206 checkpoint-discipline line is
    # auto-appended when the caller's message carries no checkpoint token
    # (see _ensure_checkpoint_discipline_line — caller evidence always wins).
    final_message = _ensure_checkpoint_discipline_line(args.message, args.label)

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

    def bundle_fallback_for_denial(
        *, phase: str, rc: int, output: str, env: dict, intent_source: str
    ) -> int | None:
        """Release the lock and retain the intended commit when #1293 fires.

        ``None`` means this was an ordinary Git failure and the caller should
        preserve the pre-existing return behavior.  Bundle construction is
        intentionally outside the lock: only the exact intent snapshot is
        captured while serialized.
        """

        nonlocal lock_fh
        if not _is_git_object_write_denial(output):
            return None

        snapshot: dict[str, tuple[bytes | None, str]] | None = None
        patch_payload: bytes | None = None
        snapshot_error: str | None = None
        try:
            if intent_source == "patch":
                patch_payload = Path(args.patch_file).read_bytes()
            else:
                snapshot = _snapshot_intended_files(
                    files, env, from_index=(intent_source == "index")
                )
        except Exception as exc:
            snapshot_error = f"{type(exc).__name__}: {exc}"

        # The diagnosis + isolated bundle work can be slow and touches no shared
        # index.  Release before it, as required by the lock discipline.
        _release_lock(lock_fh)
        lock_fh = None
        failure_output = f"phase={phase}\n{output}"
        if snapshot_error:
            failure_output += f"\nintent_snapshot_error={snapshot_error}"
        try:
            row = _author_bundle_fallback(
                files=files,
                final_message=final_message,
                label=args.label,
                original_rc=rc,
                original_output=failure_output,
                fallback_receipt_dir=args.fallback_receipt_dir,
                intended_snapshot=snapshot,
                patch_bytes=patch_payload,
                allow_empty=args.allow_empty,
            )
        except BundleFallbackStorageRefusal as exc:
            _append_log(
                {
                    **base_record,
                    "outcome": "bundle_fallback_storage_refused",
                    "failure_phase": phase,
                    "git_rc": rc,
                    "error": str(exc),
                    "receipt_path": str(exc.receipt_path),
                }
            )
            print(
                f"BUNDLE_FALLBACK_STORAGE_REFUSED "
                f"rc={BUNDLE_FALLBACK_STORAGE_REFUSAL_RC} "
                f"phase={phase} receipt={exc.receipt_path} error={exc}",
                file=sys.stderr,
            )
            return BUNDLE_FALLBACK_STORAGE_REFUSAL_RC
        except Exception as exc:
            _append_log(
                {
                    **base_record,
                    "outcome": "bundle_fallback_failed",
                    "failure_phase": phase,
                    "git_rc": rc,
                    "error": str(exc),
                }
            )
            print(
                f"BUNDLE_FALLBACK_FAILED rc={BUNDLE_FALLBACK_FAILED_RC} "
                f"phase={phase} error={exc}",
                file=sys.stderr,
            )
            return BUNDLE_FALLBACK_FAILED_RC

        _append_log(
            {
                **base_record,
                "outcome": "bundle_fallback_ready",
                "failure_phase": phase,
                "git_rc": rc,
                "bundle_path": row["bundle_path"],
                "format_patch_path": row["format_patch_path"],
                "receipt_path": row["receipt_path"],
                "fallback_commit": row["fallback_commit"],
            }
        )
        print(
            "BUNDLE_FALLBACK "
            f"rc={BUNDLE_FALLBACK_RC} phase={phase} "
            f"bundle={row['bundle_path']} "
            f"format_patch={row['format_patch_path']} "
            f"receipt={row['receipt_path']} "
            f"fallback_commit={row['fallback_commit']}"
        )
        return BUNDLE_FALLBACK_RC

    # FIX-MERGE-HEAD (2026-08-20, the ddm_oc2 incident): an open
    # `git merge --no-commit` held across tool calls makes OUR commit a merge
    # commit. Git attaches MERGE_HEAD as a second parent regardless of what we
    # staged, so history claims the branch landed while only our own files were
    # committed — 7,637 insertions were nearly lost that way, and `git merge`
    # afterwards reports "Already up to date", so the loss is silent and
    # permanent. Measured class population: 3 of 311 merge commits in this
    # repo's history carry the signature (100% of branch-changed files absent
    # from the merge tree). This is the FIRST post-lock check because an open
    # merge invalidates every downstream guard's premise.
    merge_state = merge_in_progress()
    if merge_state is not None:
        merge_head_sha, merge_head_mtime = merge_state
        merge_age_seconds = round(time.time() - merge_head_mtime, 1)
        # Read-only queries against the REAL index/HEAD: the temp index does not
        # exist yet at this point, and a merge is completed from the real index.
        merge_probe_env = dict(os.environ)
        branch_files = _merge_branch_changed_files(merge_head_sha, merge_probe_env)
        staged_now = _staged_touched_files(merge_probe_env)
        unstaged_branch_files = (
            sorted(set(branch_files) - set(staged_now)) if branch_files is not None else None
        )
        if not args.merge_commit:
            _release_lock(lock_fh)
            _append_log({
                **base_record,
                "outcome": "merge_head_open_refused",
                "wait_seconds": wait_seconds,
                "merge_head_sha": merge_head_sha,
                "merge_head_age_seconds": merge_age_seconds,
                "merge_branch_changed_file_count": (
                    None if branch_files is None else len(branch_files)
                ),
                "merge_branch_files_not_staged": unstaged_branch_files,
            })
            print(
                "[subagent-commit-serializer] REFUSED (rc=16): a merge is IN "
                f"PROGRESS (.git/MERGE_HEAD -> {merge_head_sha}, opened "
                f"{merge_age_seconds}s ago). Committing now would attach that "
                "merge as a SECOND PARENT of your commit. The serializer stages "
                "into a TEMP INDEX built from HEAD, so the merge's content is "
                "dropped even when it is staged in the real index: history "
                "would claim the branch is merged, `git merge` would then "
                "report 'Already up to date', and the branch's content would be "
                "silently lost (the ddm_oc2 incident, 2026-08-20, 7,637 "
                "insertions). Either finish the merge (`git commit` it, or "
                "`git merge --abort`), or — if this commit IS the merge — "
                "re-run with --merge-commit --no-stage and declare every staged "
                "path in --files. The merge branch changed "
                f"{'?' if branch_files is None else len(branch_files)} file(s); "
                f"not currently staged: {unstaged_branch_files!r}",
                file=sys.stderr,
            )
            return 16
        if not args.no_stage:
            _release_lock(lock_fh)
            _append_log({
                **base_record,
                "outcome": "merge_commit_without_no_stage",
                "wait_seconds": wait_seconds,
                "merge_head_sha": merge_head_sha,
            })
            print(
                "[subagent-commit-serializer] REFUSED (rc=16): --merge-commit "
                "requires --no-stage. The serializer stages --files into a "
                "TEMP index built from HEAD, which would discard the merge "
                "result and reproduce the very loss this guard exists to stop. "
                "Resolve and `git add -A` into the real index yourself, then "
                "re-run with --merge-commit --no-stage.",
                file=sys.stderr,
            )
            return 16
        if unstaged_branch_files:
            _release_lock(lock_fh)
            _append_log({
                **base_record,
                "outcome": "merge_commit_staged_set_incomplete",
                "wait_seconds": wait_seconds,
                "merge_head_sha": merge_head_sha,
                "merge_head_age_seconds": merge_age_seconds,
                "merge_branch_files_not_staged": unstaged_branch_files,
            })
            print(
                "[subagent-commit-serializer] REFUSED (rc=16): --merge-commit "
                "was declared, but the staged set does NOT cover the merge. "
                f"{len(unstaged_branch_files)} file(s) the branch "
                f"({merge_head_sha}) changed are not staged, so this commit "
                "would record the merge while dropping its content. Stage the "
                "full merge result (`git add -A` after resolving), then retry. "
                f"Missing: {unstaged_branch_files[:20]!r}",
                file=sys.stderr,
            )
            return 16
        base_record["merge_commit_declared"] = True
        base_record["merge_head_sha"] = merge_head_sha

    # FIX-1: re-hash under the lock and compare. If a sister subagent
    # modified our --files content during our lock-wait, refuse.
    if not args.no_concurrent_edit_check and not args.no_stage and files and not patch_mode:
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

    # FIX-ABSORPTION post-lock re-check: HEAD may have moved during the
    # lock-wait window (a sibling committed). If the sibling landed exactly
    # the hunks that were in our declared base, HEAD now matches and we
    # proceed; if the sibling landed content we never based on, whole-file
    # staging would silently REVERT it — refuse instead.
    if base_content_shas and not patch_mode:
        base_diffs = _base_content_check(base_content_shas)
        if base_diffs:
            _release_lock(lock_fh)
            _append_log({
                **base_record,
                "outcome": "base_content_sha_mismatch_post_lock",
                "wait_seconds": wait_seconds,
                "base_content_sha_diffs": {
                    f: {"declared_base": want, "head_blob": got}
                    for f, (want, got) in base_diffs.items()
                },
            })
            print(
                "[subagent-commit-serializer] REFUSED (rc=6): "
                "--base-content-sha256 mismatch vs HEAD *after* acquiring "
                "the lock — a sibling's commit landed during the lock-wait "
                "and your whole-file stage would revert or mis-attribute "
                "it. Re-base on HEAD and retry. Files affected: "
                f"{list(base_diffs.keys())!r}",
                file=sys.stderr,
            )
            return 6

    temp_index_path: str | None = None
    try:
        # Per-invocation temp index — isolates our staging from any
        # concurrent subagent or manual `git add`. See _make_temp_index
        # docstring for the bug class this fixes.
        if args.no_stage and not patch_mode:
            # Caller already staged into .git/index; we honor that.
            env = {**os.environ}
        else:
            # Patch mode always needs the temp index so `git apply --cached`
            # stages into an isolated index (never the shared .git/index).
            temp_index_path, env = _make_temp_index()

        # Step 1: stage
        if patch_mode:
            # FIX-CLOBBER intent-manifest staging (Catalog #405): apply the
            # caller's exact patch to the temp index; the working tree is not
            # consulted, so no sibling hunks can be swept in.
            rc, msg = _git_apply_cached(str(args.patch_file), env)
            if rc != 0:
                _append_log({**base_record, "outcome": "git_apply_failed",
                             "wait_seconds": wait_seconds,
                             "git_apply_rc": rc, "git_apply_output": msg,
                             "patch_file": str(args.patch_file),
                             "temp_index": temp_index_path})
                print(
                    "[subagent-commit-serializer] git apply --cached failed "
                    f"(rc={rc}) — the patch is likely NOT cleanly based on "
                    f"HEAD. Regenerate it against HEAD and retry:\n{msg}",
                    file=sys.stderr,
                )
                fallback_rc = bundle_fallback_for_denial(
                    phase="git_apply_cached",
                    rc=rc,
                    output=msg,
                    env=env,
                    intent_source="patch",
                )
                if fallback_rc is not None:
                    return fallback_rc
                return rc
            # PERMANENT FIX (review F1 2026-07-18): patch-mode harvest guards.
            # `git apply --cached` stages regardless of .gitignore and can
            # shrink/delete a protected append doc, and the --files guards above
            # never run in patch mode (no --files). Run the SAME guards against
            # the patch's touched files — ADDED paths for the gitignore check,
            # all touched paths for the shrink/deletion clobber check — reading
            # staged content from THIS temp index.
            _patch_ctx = {**base_record, "wait_seconds": wait_seconds,
                          "temp_index": temp_index_path}
            rc13 = _refuse_gitignored(
                _check_ignored_files(_staged_touched_files(env, diff_filter="A")),
                _patch_ctx,
            )
            if rc13 is not None:
                return rc13
            rc14 = _refuse_protected_doc_shrink(
                _protected_append_doc_shrink_check(_staged_touched_files(env), env),
                args.allow_shared_doc_shrink, _patch_ctx,
            )
            if rc14 is not None:
                return rc14
        elif not args.no_stage:
            rc, msg = _git_add(files, env)
            if rc != 0:
                _append_log({**base_record, "outcome": "git_add_failed",
                             "wait_seconds": wait_seconds,
                             "git_add_rc": rc, "git_add_output": msg,
                             "temp_index": temp_index_path})
                print(f"[subagent-commit-serializer] git add failed (rc={rc}):\n{msg}",
                      file=sys.stderr)
                fallback_rc = bundle_fallback_for_denial(
                    phase="git_add",
                    rc=rc,
                    output=msg,
                    env=env,
                    intent_source="working_tree",
                )
                if fallback_rc is not None:
                    return fallback_rc
                return rc

            # PERMANENT FIX (2026-07-18): protected append-doc CLOBBER guard.
            # A whole-file overwrite of a multi-writer append doc off a STALE
            # base silently drops sibling rows (the 2026-07-18 completeness-
            # matrix incident wiped the compiler arm's factor-1/5 edits). A net
            # line LOSS on such a doc is the clobber signature; refuse unless
            # the caller declares an intentional shrink via
            # --allow-shared-doc-shrink <rationale>. The RIGHT fix is a 3-way
            # `git merge-file`, which this guard's message points the caller to.
            rc14 = _refuse_protected_doc_shrink(
                _protected_append_doc_shrink_check(files, env),
                args.allow_shared_doc_shrink,
                {**base_record, "wait_seconds": wait_seconds,
                 "temp_index": temp_index_path},
            )
            if rc14 is not None:
                return rc14

            # Catalog #405 hunk-attribution WARN (never refuses): if the caller
            # hinted --expected-diff-lines, compare the actually-staged diff
            # size. A whole-file `git add` that swept a sibling's uncommitted
            # hunks (incident 2, 2026-07-08) shows up as a gross overshoot.
            if expected_diff_lines:
                staged_lines = _staged_diff_line_count(
                    list(expected_diff_lines.keys()), env
                )
                overshoots: dict[str, tuple[int, int]] = {}
                for f, hint in expected_diff_lines.items():
                    actual = staged_lines.get(f, 0)
                    if actual < 0:
                        continue  # binary — heuristic N/A
                    if actual > 2 * hint:
                        overshoots[f] = (hint, actual)
                if overshoots:
                    _append_log({
                        **base_record,
                        "outcome": "hunk_attribution_overshoot_warned",
                        "wait_seconds": wait_seconds,
                        "diff_line_overshoots": {
                            f: {"hint": h, "staged": a}
                            for f, (h, a) in overshoots.items()
                        },
                        "temp_index": temp_index_path,
                    })
                    print(
                        "[subagent-commit-serializer] WARNING (Catalog #405): "
                        "staged diff is grossly larger (>2x) than the "
                        "--expected-diff-lines hint — a whole-file `git add` "
                        "may have swept a sibling's uncommitted hunks into "
                        "your commit (incident 2, 2026-07-08). NOT refused. "
                        "Inspect `git diff --cached -- <file>`; for a hard "
                        "guarantee on shared files use --patch-file. "
                        + "; ".join(
                            f"{f}: hint={h} staged={a}"
                            for f, (h, a) in overshoots.items()
                        ),
                        file=sys.stderr,
                    )

        if patch_mode:
            staged_set_context = "patch-file temp index"
        elif args.no_stage:
            staged_set_context = "--no-stage real index"
        else:
            staged_set_context = "git-add temp index"
        rc15 = _refuse_staged_declared_file_set_mismatch(
            files,
            env,
            base_record,
            wait_seconds=wait_seconds,
            temp_index_path=temp_index_path,
            context=staged_set_context,
        )
        if rc15 is not None:
            return rc15

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
        if expected_content_shas and not args.no_stage and not patch_mode:
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
        # final_message NEVER carries a co-author trailer (operator 2026-05-31);
        # it MAY carry the auto-appended Catalog #206 checkpoint-discipline
        # waiver line when the caller supplied no checkpoint token.
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
            fallback_rc = bundle_fallback_for_denial(
                phase="git_commit",
                rc=rc,
                output=(stdout or "") + (stderr or ""),
                env=env,
                intent_source=("patch" if patch_mode else "index"),
            )
            if fallback_rc is not None:
                return fallback_rc
            return rc

        if temp_index_path:
            _refresh_real_index_after_temp_commit(files)

        # FIX-CLOBBER post-commit verification (Catalog #405, rc=7): the commit
        # has landed and HEAD moved. Re-read each declared file's content AT
        # HEAD and confirm it matches what the caller declared it should be.
        # This is the ONLY check that reads HEAD-after-the-ref-moved, so it
        # catches the pre-snapshot-clobber gap (incident 1, 2026-07-08) that
        # every working-tree/index check is structurally blind to. The commit
        # is NOT auto-reverted — the committed content may be a sibling's newer
        # legitimate landing; surfacing beats destroying.
        if expected_content_shas:
            post_diffs = _post_commit_content_check(expected_content_shas)
            if post_diffs:
                _append_log({
                    **base_record,
                    "outcome": "post_commit_content_sha_mismatch",
                    "wait_seconds": wait_seconds,
                    "commit_seconds": commit_seconds,
                    "head_after": head_after,
                    "post_commit_content_sha_diffs": {
                        f: {"declared": want, "committed_head": got}
                        for f, (want, got) in post_diffs.items()
                    },
                    "temp_index": temp_index_path,
                })
                print(
                    "[subagent-commit-serializer] REFUSED (rc=7): POST-COMMIT "
                    "content mismatch — the content that landed at HEAD is NOT "
                    "what you declared via --expected-content-sha256. Likely "
                    "cause: a sibling clobbered/reverted these file(s) in the "
                    "working tree BEFORE you snapshotted the sha, so every "
                    "pre-commit check compared against the already-clobbered "
                    "content and passed (incident 1, 2026-07-08). The commit "
                    f"({head_after}) is KEPT — it may be the sibling's newer "
                    "legitimate landing, so it is NOT auto-reverted.\n"
                    + "\n".join(
                        f"  {f}: declared={want} committed={got}"
                        for f, (want, got) in post_diffs.items()
                    )
                    + "\n  RECONCILE (do NOT blind-revert): "
                    "1) inspect what landed: `git show HEAD:<file>`; "
                    "2) if the committed content is the sibling's and yours "
                    "is newer, re-apply your intended content and re-commit "
                    "(ideally via --patch-file); "
                    "3) if a full undo is truly warranted after reconciling, "
                    f"`git revert --no-commit {head_after}` (a revert commit, "
                    "not a history rewrite).",
                    file=sys.stderr,
                )
                return 7

        # FIX-ATTRIBUTION (2026-08-02 task #911, hardened 2026-08-05 task #883):
        # reconcile REQUESTED files against what git RECORDED for this commit.
        # `absent` remains warn-only because a requested file whose content
        # already matched HEAD produces no diff. `extra` is hard rc=15: this
        # commit carries files the caller did not declare, i.e. the wrong-body
        # absorption direction. The commit is KEPT for no-signal-loss forensics.
        recorded = _files_recorded_by_head_commit()
        recorded_n = "?" if recorded is None else str(len(recorded))
        if recorded is not None:
            requested = set(files)
            absent = sorted(requested - recorded)   # sibling absorbed, or no-op edit
            extra = sorted(recorded - requested)    # WE absorbed a sibling's work
            if absent or extra:
                _append_log({
                    **base_record,
                    "outcome": (
                        "post_commit_file_attribution_extra_refused"
                        if extra
                        else "post_commit_file_attribution_absent_warned"
                    ),
                    "wait_seconds": wait_seconds,
                    "commit_seconds": commit_seconds,
                    "head_after": head_after,
                    "requested_files": sorted(requested),
                    "recorded_files": sorted(recorded),
                    "requested_but_not_recorded": absent,
                    "recorded_but_not_requested": extra,
                    "temp_index": temp_index_path,
                })
                print(
                    "[subagent-commit-serializer] ATTRIBUTION MISMATCH "
                    f"({'hard rc=15' if extra else 'warn rc=0'}): "
                    f"you requested {len(requested)} file(s); commit "
                    f"{head_after} recorded {len(recorded)}.",
                    file=sys.stderr,
                )
                if absent:
                    print(
                        "  REQUESTED BUT NOT RECORDED (a sibling may have "
                        "already committed your content — check "
                        "`git log -S<your-symbol> --all`; or your edit was a "
                        "no-op vs HEAD):\n"
                        + "\n".join(f"    {f}" for f in absent),
                        file=sys.stderr,
                    )
                if extra:
                    print(
                        "  RECORDED BUT NOT REQUESTED (YOUR commit carries "
                        "files you did not ask for — the 2026-04-29 absorption "
                        "direction; do NOT rewrite history, record the "
                        "correction):\n"
                        + "\n".join(f"    {f}" for f in extra),
                        file=sys.stderr,
                    )
                    print(
                        "  The commit is KEPT for no-signal-loss forensics. "
                        "Record a correction or follow-up; do not rewrite "
                        "history blindly.",
                        file=sys.stderr,
                    )
                    return 15

        print(f"[subagent-commit-serializer] OK head={head_after} "
              f"label={args.label} files={len(files)} recorded={recorded_n} "
              f"wait={wait_seconds}s commit={commit_seconds}s "
              f"temp_index={'YES' if temp_index_path else 'NO (--no-stage)'}",
              file=sys.stderr)
        return 0
    finally:
        if temp_index_path:
            _cleanup_temp_index(temp_index_path)
        if lock_fh is not None:
            _release_lock(lock_fh)


if __name__ == "__main__":
    sys.exit(main(rebind_root=True))
