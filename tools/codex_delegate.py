#!/usr/bin/env python3
"""codex_delegate.py — canonical wrapper for delegating a task to a codex agent
with AUTOMATIC completion notification back to the Claude main loop.

WHY: codex agents are launched HEADLESS — detached into their own session (setsid via
the shared spawn core spawn_durable_daemon.spawn_detached_verified) so they are durable (survive the harness
SIGURG-144 that kills bg-bash, by escaping claude's process group) WITHOUT opening a GUI
Terminal window (the prior `osascript … do script` mechanism orphaned a completed-but-open
window per arm) — AND with a CONTROLLING PTY (script -q; task #525): a no-TTY codex arm
with stdin=/dev/null matches the fleet claude-code-reaper launchd agent's orphan signature
and is SIGTERM'd at ~5min (measured 2026-07-17, 10/10 kills); the pty classifies the arm
as the live terminal session it actually is. They are fire-and-forget — the main loop had to
manually poll (pgrep / log tail / git log) to know when they finished. Claude
Agent-tool subagents, by contrast, deliver a completion notification. This wrapper
gives codex the same: every delegated run appends START/DONE lines to ONE shared
events log (`.omx/tmp/codex_runs/codex_events.log`); the main loop keeps a SINGLE
persistent `tail -f` Monitor over that log (a blocking tail SURVIVES SIGURG, unlike
a `while/sleep` poll loop — the empirically-verified durable primitive), so each
codex completion notifies the loop like a subagent finishing.

USAGE (launch):
    .venv/bin/python tools/codex_delegate.py \
        --label frozen_segnet \
        --prompt-file .omx/tmp/codex_runs/frozen_segnet_analysis.prompt.txt \
        [--model gpt-5.6-sol] [--effort ultra] [--sandbox danger-full-access] \
        [--no-launch]   # write the launcher + ledger row but do not launch (headless)

    # then ONCE per session, arm the notifier (Claude Monitor tool, persistent):
    #   tail -n 0 -f .omx/tmp/codex_runs/codex_events.log | grep --line-buffered '^DONE'

STATUS:  .venv/bin/python tools/codex_status.py
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# tools/ sibling imports (shared detached-spawn core lives in spawn_durable_daemon).
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
# THE model/effort constants are OWNED by codex_arm_queue -- imported, never
# re-declared here. This file is the SECOND live codex spawn surface (10+ callers);
# a duplicated default is two sources of truth for one fact, and on 2026-08-08 the
# 5.5 REFUSAL was enforced at only one of the two (this one would have spawned a
# banned generation, since _full_model_id only rejects shorthands). One-way import:
# codex_arm_queue does not import this module.
from codex_arm_queue import (  # noqa: E402  (must follow the sys.path insert above)
    ARM_EFFORT_LEVELS,
    ARM_MODEL,
    assert_arm_model_admissible,
)

RUNS = REPO / ".omx" / "tmp" / "codex_runs"
EVENTS = RUNS / "codex_events.log"
LEDGER = REPO / ".omx" / "state" / "codex_delegations.jsonl"
INBOX_DIR = REPO / ".omx" / "tmp" / "codex_inbox"  # per-arm watched inboxes (tools/codex_msg.py producer)
MANIFEST_DIR = REPO / ".omx" / "tmp" / "codex_manifests"  # per-arm files-touched manifest (harvest-commit contract)
WORKTREE_DIR = REPO / ".omx" / "tmp" / "codex_worktrees"  # per-arm isolated git worktree (CFL disjoint domain)

# Back-pressure for legacy shared-tree writers: workspace-write blocks .git/objects,
# so such an arm strands its diff. Above this many existing uncommitted files the
# dispatcher refuses ordinary launches until the pile is harvest-drained.
_MAX_UNCOMMITTED_PILE = 40

# A shared-tree writer has no attribution boundary: two such arms can edit one
# index/worktree and leave an intermingled, unreviewable pile. Isolated worktree
# arms and read-only shared-tree arms cannot create that failure mode.
_MAX_NONISOLATED_WRITERS = 1

# Relaunches retain their label's custody. They may retain/increase authority,
# but must never silently downgrade a commit-capable arm to a sandbox that
# cannot write the worktree's git objects.
_SANDBOX_AUTHORITY = {
    "read-only": 0,
    "workspace-write": 1,
    "danger-full-access": 2,
}

# Bidirectional-channel contract prepended to EVERY delegated prompt so a one-shot codex arm
# stays AMENDABLE while running: it polls a watched inbox at each checkpoint and consumes new
# directives (which SUPERSEDE the prompt, being later information). Producer: tools/codex_msg.py.
_INBOX_CONTRACT = """=== LIVE INBOX — bidirectional channel, CHECK AT EVERY CHECKPOINT (do not skip) ===
You were launched with a one-shot prompt, but the main loop / operator CAN send you NEW
information mid-run through a watched inbox. Two files:
  - per-arm:    {inbox}
  - fleet-wide: {broadcast}
BEFORE each tools/subagent_checkpoint.py call (~every 10 tool uses) AND before your FINAL
message, READ both files (they may be empty/absent = proceed normally). Each line is a JSON
directive {{"utc","from","priority","directive"}}. Track a cursor (last utc consumed) so you
act only on NEW lines. Rules:
  - operator/main directives here are LATER information and SUPERSEDE any conflicting
    instruction in the prompt below (including anti-collision / defer clauses if lifted).
  - priority "stop" -> checkpoint your state and exit cleanly (a newer arm supersedes you).
  - acknowledge which directives you consumed in your checkpoint notes + final message.
Every checkpoint for this delegation MUST use the exact parent/session key:
    {delegation_key}
The transient retry controller refuses to retry without an in_progress row under that key
containing a positive integer step and non-empty next_action.
This is how you stay amendable while running. Absent/empty inbox -> ignore, proceed normally.
=== END LIVE INBOX ===

"""

# Commit-path contract — full-authority + worktree-ISOLATED (operator 2026-07-14: codex is a trusted
# partner with full authority; coherent parallelism = CFL: disjoint write-domains + serialized merge).
# The arm runs in its OWN git worktree on branch codexwt/<label>_<stamp>, commits FREELY there (full
# sandbox), and MAIN reviews the branch diff + merges to main. No trample (disjoint domain); the review
# survives at the merge boundary. Prepended to every delegated prompt.
_WORKTREE_CONTRACT = """=== COMMIT PATH — YOU ARE IN YOUR OWN ISOLATED GIT WORKTREE; COMMIT FREELY HERE ===
You are running in your OWN git worktree at:
    {worktree}
on branch:  {branch}   (NOT main; shares .git/objects but has its own index + HEAD)
You have FULL git authority HERE and you are ISOLATED — no other arm shares this tree, so there is NO
trample risk. Therefore:
  1. Make your edits and COMMIT them on THIS branch (plain `git commit` is fine — you are isolated).
     Commit early + often; your commits ARE your deliverable.
  2. Do NOT `cd` to or touch the main repo ({repo}) or any other worktree — stay in your worktree.
  3. In your FINAL message, list your commit(s) + a one-line verdict + what MAIN should review.
  4. (optional, aids review) write the files you touched to {manifest} as
     {{"files":[...], "code_files":[...], "verdict":"...", "review_notes":"..."}}.
MAIN reviews your branch diff (base..{branch}) and MERGES it to main — that review at the merge boundary
IS the non-negotiable follow-up. Uncommitted edits left in your worktree are ALSO harvested, but
committing is cleaner. Never force, never touch main directly.
=== END COMMIT PATH ===

"""


def _uncommitted_pile_size() -> int:
    r = subprocess.run(["git", "-C", str(REPO), "status", "--short"],  # subprocess-no-check-OK: advisory pile-size census; failure reads 0
                       capture_output=True, text=True)
    return sum(1 for ln in r.stdout.splitlines() if ln.strip())


def _ledger_rows() -> list[dict]:
    """Return valid delegation rows; a corrupt line is not treated as a row."""
    if not LEDGER.is_file():
        return []
    rows: list[dict] = []
    for line in LEDGER.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _prior_sandboxes(label: str) -> list[str]:
    """Every valid sandbox previously recorded for ``label`` (oldest first)."""
    return [
        str(row.get("sandbox"))
        for row in _ledger_rows()
        if row.get("label") == label and row.get("sandbox") in _SANDBOX_AUTHORITY
    ]


def _live_nonisolated_writer_count() -> int:
    """Count live shared-tree arms that can write.

    Legacy rows without an ``isolate`` field are conservatively non-isolated.
    """
    seen: dict[str, dict] = {}
    for row in _ledger_rows():
        seen[f"{row.get('label')}_{row.get('stamp')}"] = row
    count = 0
    for row in seen.values():
        if bool(row.get("isolate", False)):
            continue
        if row.get("sandbox", "workspace-write") == "read-only":
            continue
        label, stamp = row.get("label", ""), row.get("stamp", "")
        if not label or not stamp:
            continue
        r = subprocess.run(
            ["pgrep", "-f", f"{label}_{stamp}"], capture_output=True, text=True
        )
        if r.returncode == 0 and r.stdout.strip():
            count += 1
    return count


def _launch_policy_refusal(
    *,
    label: str,
    requested_sandbox: str,
    isolate: bool,
    prior_sandboxes: list[str],
    live_nonisolated_writers: int,
    nonisolated_writer_cap: int = _MAX_NONISOLATED_WRITERS,
) -> tuple[int, str] | None:
    """Pure fail-closed policy for relaunch authority and shared-tree writers.

    STRICT preflight executes the diagnosed counterexamples against this exact
    function, so a token-only imitation cannot satisfy the gate.
    """
    requested_rank = _SANDBOX_AUTHORITY[requested_sandbox]
    valid_prior = [s for s in prior_sandboxes if s in _SANDBOX_AUTHORITY]
    if valid_prior:
        strongest = max(valid_prior, key=_SANDBOX_AUTHORITY.__getitem__)
        if requested_rank < _SANDBOX_AUTHORITY[strongest]:
            return (
                5,
                f"REFUSED (sandbox downgrade): label '{label}' previously ran with "
                f"sandbox={strongest}, but this relaunch requests {requested_sandbox}. "
                "A retry/relaunch must preserve or increase the original arm's git "
                "authority; use the original sandbox, or use a new label for a genuinely "
                "read-only lineage. --force cannot bypass this custody guard.",
            )

    requested_is_writer = not isolate and requested_sandbox != "read-only"
    if requested_is_writer and live_nonisolated_writers >= nonisolated_writer_cap:
        return (
            6,
            "REFUSED (non-isolated writer cap): "
            f"{live_nonisolated_writers} shared-tree writer arm(s) are already live "
            f"(cap={nonisolated_writer_cap}). Multiple --no-isolate writer arms create "
            "one intermingled, confounded diff with no reviewable attribution. Use the "
            "default isolated worktree, wait for the current writer to finish, or make "
            "this arm --sandbox read-only. --force cannot bypass this safety cap.",
        )
    return None

# Transient-death auto-recovery (apparatus fix for the codex_probe_token_limit_death /
# "Selected model is at capacity" bug class): a capacity/rate-limit/disconnect is a
# SERVER blip, not a token limit — a bump of model_context_window would NOT help. On such
# a death the launcher runs a compact RESUME prompt after backoff, but only after the exact
# delegation checkpoint has been proved. Blind replay of the original prompt is forbidden.
# Fatal errors (bad flag, syntax, non-transient) do NOT match the signature → no retry loop.
_MAX_CAPACITY_RETRIES = 2            # checkpoint-custodied retries; never eight blind restarts
_CAPACITY_BACKOFF_STEP_SECONDS = 20  # attempt N waits N * step seconds (20, 40)
_TRANSIENT_DEATH_SIGNATURE = (
    "at capacity|rate.?limit|429|overloaded|temporarily unavailable|"
    "stream disconnected|error sending request|connection reset|timed out|503|502"
)

# The one-line notifier the main loop arms once (printed on launch for convenience).
# -n 0: new events ONLY. `-n +1` on this long-lived shared log replays the
# ENTIRE fleet history as fresh 'events' into any monitor that arms it
# (2026-07-19 incident). Historical rows are for reading, not watching.
NOTIFIER_CMD = f"tail -n 0 -f {EVENTS} | grep --line-buffered '^DONE'"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _live_labels() -> list[str]:
    """Labels of currently-alive delegated arms (robust: pgrep on the label_stamp
    token per ledger row, never `pgrep -fl | head` which the inlined prompt clips)."""
    seen: dict[str, dict] = {}
    for d in _ledger_rows():
        seen[f"{d.get('label')}_{d.get('stamp')}"] = d
    live: list[str] = []
    for d in seen.values():
        label, stamp = d.get("label", ""), d.get("stamp", "")
        r = subprocess.run(["pgrep", "-f", f"{label}_{stamp}"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            live.append(label)
    return live


def _append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _write_compact_prompts(
    *, label: str, stamp: str, wrapped_prompt: Path, delegation_key: str
) -> tuple[Path, Path]:
    """Externalize the authority prompt and return bounded entry/resume prompts."""
    payload = wrapped_prompt.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    entry = RUNS / f"{label}_{stamp}.entry.prompt.txt"
    resume = RUNS / f"{label}_{stamp}.resume.prompt.txt"
    entry.write_text(
        "Delegated task authority is the file below. Read it completely before acting, "
        "using bounded 32768-byte chunks if needed; do not ask the launcher to inline it.\n"
        f"authority_file={wrapped_prompt}\nsha256={digest}\nbytes={len(payload)}\n"
        f"delegation_checkpoint_key={delegation_key}\n"
        "Checkpoint after each major milestone with that exact key. Land durable artifacts, "
        "keep the final response concise, and require MAIN landing review.\n",
        encoding="utf-8",
    )
    resume.write_text(
        "This is a checkpoint-custodied transient retry, not a fresh execution.\n"
        f"delegation_checkpoint_key={delegation_key}\n"
        f"authority_file={wrapped_prompt}\nsha256={digest}\nbytes={len(payload)}\n"
        "First run: python3 tools/subagent_checkpoint.py read "
        f"--parent-id-or-session {delegation_key} --latest-only\n"
        "Continue only from that checkpoint's next_action and durable files. Do not restart "
        "completed work. Re-read authority in bounded 32768-byte chunks only as needed. "
        "Finish with a concise result and require MAIN landing review.\n",
        encoding="utf-8",
    )
    return entry, resume


def _write_launcher(label: str, stamp: str, prompt_file: Path, model: str,
                    effort: str, sandbox: str, log: Path, last: Path,
                    done: Path, isolate: bool = True,
                    worktree: Path | None = None,
                    allow_network: bool = False,
                    resume_prompt_file: Path | None = None,
                    delegation_key: str | None = None) -> Path:
    launcher = RUNS / f"launch_{label}_{stamp}.sh"
    worktree = worktree or (WORKTREE_DIR / f"{label}_{stamp}")
    resume_prompt_file = resume_prompt_file or prompt_file
    delegation_key = delegation_key or f"codex_delegate:{label}:{stamp}"
    # --- codex-contract fix (2026-07-15): an isolated git worktree writes its objects + index to the
    # SHARED <repo>/.git, which lives OUTSIDE the workspace-write sandbox root (=the worktree dir). So a
    # workspace-write isolated arm literally CANNOT `git add`/`commit` ("Operation not permitted") and
    # lands PASS_UNCOMMITTED, forcing main to harvest by hand (bit 4 arms 2026-07-15). Make the shared
    # git dir a writable_root so isolated arms commit their own work. (danger-full-access already has
    # full write, so this only matters for workspace-write.) network_access is opt-in per --allow-network
    # (least-privilege) — needed for Modal-dispatch arms whose sandbox otherwise cannot resolve DNS.
    #   (2026-07-15 fix #2): the SAME `.git`-read-only block hits --no-isolate arms that must commit to
    #   MAIN's .git directly (e.g. a branch-consolidation/merge arm) — codex's workspace-write sandbox
    #   treats `.git` as read-only by default in BOTH modes. So grant `[REPO/.git]` as a writable_root
    #   whenever sandbox==workspace-write, not just for isolated worktrees. (isolate -> shared .git for
    #   worktree objects+index; --no-isolate -> the same main .git for direct commits.) Operator directive
    #   2026-07-15: "give them the authority they need."
    _extra_c: list[str] = []
    if sandbox == "workspace-write":
        _extra_c.append(f"-c 'sandbox_workspace_write.writable_roots=[\"{REPO}/.git\"]'")
    if allow_network and sandbox == "workspace-write":
        _extra_c.append("-c sandbox_workspace_write.network_access=true")
    extra_c = ("".join(f" \\\n  {c}" for c in _extra_c))
    # Per-arm git-worktree ISOLATION (CFL disjoint write-domain -> no trample). When isolate, the arm
    # runs in its OWN worktree on branch codexwt/<label>_<stamp> (shares .git/objects, own index+HEAD),
    # commits there with full authority, and MAIN merges the branch to main (the single serialized
    # integration point = Courant<=1). Falls back to the shared tree if `git worktree add` fails.
    branch = f"codexwt/{label}_{stamp}"
    if isolate:
        isolate_setup = (
            f'BASE=$(git -C {REPO} rev-parse HEAD 2>/dev/null)\n'
            f'if git -C {REPO} worktree add -b {branch} "{worktree}" "$BASE" >> {log} 2>&1; then\n'
            f'  WORKDIR="{worktree}"\n'
            f'  echo "WORKTREE {label} {stamp} base=$BASE branch={branch} path={worktree}" >> {EVENTS}\n'
            f'else\n'
            f'  RC=12\n'
            f'  echo "WORKTREE-REFUSED {label} {stamp} — isolation setup failed; no shared-tree writer fallback" >> {EVENTS}\n'
            f'  echo "DONE {label} rc=$RC {stamp} $(date -u +%Y-%m-%dT%H:%M:%SZ) :: isolation setup failed closed" >> {EVENTS}\n'
            f"  printf 'rc=%s\\nfinished_utc=%s\\nlog=%s\\nlast=%s\\nworktree=%s\\nbranch=%s\\nisolate=1\\n' \"$RC\" \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \"{log}\" \"{last}\" \"{worktree}\" \"{branch}\" > {done}\n"
            f'  echo "REFUSED: isolated worktree creation failed; shared-tree fallback is forbidden for writer arms"\n'
            f'  exit $RC\n'
            f'fi'
        )
    else:
        isolate_setup = '# --no-isolate: running in the shared repo tree (legacy; use for read-only arms only)'
    # The launcher: (worktree isolate ->) run codex, tee to log, capture final message via -o, then on
    # exit append a DONE line (with a one-line summary from the tail of `last`) to the shared events log
    # + write a per-run .done marker recording the worktree/branch for main to merge. The launcher then
    # `exit`s cleanly (headless — see the Popen start_new_session launch in main); it no longer `exec bash`
    # to hold a Terminal window open (that orphaned a completed-but-open window per arm). Inspect via {last}/{log}.
    launcher.write_text(
        f"""#!/bin/bash
set +e
mkdir -p {RUNS}
echo "START {label} {stamp} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> {EVENTS}
echo "=== codex delegate '{label}' [{model}/{effort}/{sandbox}] {stamp} ==="
echo "log:  {log}"
ORIGINAL_SANDBOX={shlex.quote(sandbox)}
WORKDIR="{REPO}"
{isolate_setup}
cd "$WORKDIR" || cd {REPO} || exit 1
echo "cwd: $WORKDIR"
codex exec \\
  --skip-git-repo-check \\
  --sandbox "$ORIGINAL_SANDBOX" \\
  -m {model} \\
  -c model_reasoning_effort={effort}{extra_c} \\
  -o {last} \\
  "$(cat {prompt_file})" \\
  2>&1 | tee {log}
RC=${{PIPESTATUS[0]}}
# --- transient-death auto-recovery: capacity/rate-limit/disconnect is a SERVER blip, not a
# token limit. Retry only after proving the exact delegation checkpoint, then pass a compact
# resume prompt. Fatal errors or missing custody do not retry.
attempt=0
ATTEMPT_LOG={shlex.quote(str(log))}
while [ "$RC" -ne 0 ] && [ "$attempt" -lt {_MAX_CAPACITY_RETRIES} ] && \\
      grep -qiE '{_TRANSIENT_DEATH_SIGNATURE}' "$ATTEMPT_LOG"; do
  attempt=$((attempt+1))
  python3 "$WORKDIR/tools/codex_retry_checkpoint.py" \\
    --delegation-key {shlex.quote(delegation_key)} \\
    --progress-file "$WORKDIR/.omx/state/subagent_progress.jsonl" >> {log} 2>&1
  CHECKPOINT_RC=$?
  if [ "$CHECKPOINT_RC" -ne 0 ]; then
    RC=$CHECKPOINT_RC
    echo "RETRY-REFUSED-NO-CHECKPOINT {label} attempt=$attempt rc=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> {EVENTS}
    break
  fi
  backoff=$(({_CAPACITY_BACKOFF_STEP_SECONDS} * attempt))
  echo "RETRY {label} transient rc=$RC attempt=$attempt/{_MAX_CAPACITY_RETRIES} backoff=${{backoff}}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> {EVENTS}
  echo "=== TRANSIENT death (rc=$RC) — re-run $attempt/{_MAX_CAPACITY_RETRIES} in ${{backoff}}s; agent self-resumes from checkpoint ===" | tee -a {log}
  sleep $backoff
  # Classify only this retry's output on the next loop.  Grepping the
  # cumulative log would let the original transient signature authorize an
  # additional retry after a later fatal/non-transient failure.
  ATTEMPT_LOG={shlex.quote(str(log))}.retry.$attempt
  : > "$ATTEMPT_LOG"
  codex exec \\
    --skip-git-repo-check \\
    --sandbox "$ORIGINAL_SANDBOX" \\
    -m {model} \\
    -c model_reasoning_effort={effort}{extra_c} \\
    -o {last} \\
    "$(cat {resume_prompt_file})" \\
    2>&1 | tee -a {log} "$ATTEMPT_LOG"
  RC=${{PIPESTATUS[0]}}
done
# one-line summary from the tail of the final-message file (best-effort, sanitized)
SUMMARY=$(tail -c 400 {last} 2>/dev/null | tr '\\n' ' ' | tr -s ' ' | sed 's/[|]/ /g' | tail -c 200)
echo "DONE {label} rc=$RC {stamp} $(date -u +%Y-%m-%dT%H:%M:%SZ) :: ${{SUMMARY}}" >> {EVENTS}
echo "LANDING-REVIEW-REQUIRED {label} {stamp}" >> {EVENTS}
printf 'rc=%s\\nfinished_utc=%s\\nlog=%s\\nlast=%s\\nworktree=%s\\nbranch=%s\\nisolate=%s\\nreview_required=1\\n' "$RC" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "{log}" "{last}" "{worktree if isolate else ''}" "{branch if isolate else ''}" "{'1' if isolate else '0'}" > {done}
echo ""
echo "=== codex delegate '{label}' exited rc=$RC — DONE appended; MAIN: harvest+merge worktree branch to main ==="
echo "(headless; review {last} / {log})"
exit $RC
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def _full_model_id(value: str) -> str:
    """Refuse model shorthands the API rejects (e.g. 'sol' -> 400 after setup cost)."""
    if not value.startswith("gpt-"):
        raise argparse.ArgumentTypeError(
            f"model {value!r} looks like a shorthand; pass the full ID (e.g. 'gpt-5.6-sol')"
        )
    # ALSO fail closed on a banned generation (operator 2026-08-08: "never spawning
    # on five point five again"). Shorthand-refusal alone let 'gpt-5.5' straight
    # through -- it starts with 'gpt-'. Same refusal the arm-queue spawn path uses.
    assert_arm_model_admissible(value)
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Delegate a task to codex with auto completion notification.")
    ap.add_argument("--label", required=True, help="short slug for this delegation (no spaces)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt-file", help="path to a file containing the codex prompt")
    g.add_argument("--prompt", help="inline prompt string (written to a prompt file)")
    # Full model IDs only: the API 400s on shorthands ("sol") AFTER worktree/ledger setup — a
    # 2026-07-20 launcher-side rc=1 came from exactly that. Refuse shorthands at parse time.
    ap.add_argument("--model", default=ARM_MODEL,
                    type=_full_model_id,
                    help="full codex model ID (e.g. gpt-5.6-sol); shorthands like 'sol' are refused")
    ap.add_argument("--effort", default="high", choices=list(ARM_EFFORT_LEVELS),
                    help="reasoning effort, per task (operator range high->ultra). "
                         "Canonical enum owned by codex_arm_queue.ARM_EFFORT_LEVELS.")
    # Sandbox relaxed to FULL AUTHORITY (operator 2026-07-14: "codex is a trusted partner, I trust it
    # with as much authority as you"). danger-full-access lets an arm write .git/objects and COMMIT —
    # coherent (no-trample) because each arm runs in its OWN git worktree (disjoint write-domain, the
    # CFL well-posedness condition), and MAIN is the single serialized merge point (Courant<=1).
    ap.add_argument("--sandbox", default="danger-full-access",
                    choices=["read-only", "workspace-write", "danger-full-access"])
    ap.add_argument("--no-isolate", action="store_true",
                    help="run in the shared repo tree (LEGACY) instead of a per-arm git worktree. "
                         "Isolation (default) is the no-trample guarantee — only use --no-isolate for "
                         "a read-only/analysis arm that writes nothing.")
    ap.add_argument("--allow-network", action="store_true",
                    help="grant the workspace-write sandbox network access (sandbox_workspace_write."
                         "network_access=true). Opt-in per arm (least-privilege) — needed for arms that "
                         "dispatch Modal / hit the network. danger-full-access already has network.")
    ap.add_argument("--no-launch", action="store_true", help="write launcher + ledger but do not osascript-launch")
    ap.add_argument("--no-pty", action="store_true",
                    help="launch WITHOUT the controlling-pty wrap (NOT recommended: a no-TTY codex arm "
                         "matches the fleet claude-code-reaper orphan signature and is SIGTERM'd at "
                         "~5min — the measured task-#525 kill class)")
    ap.add_argument("--force", action="store_true",
                    help="launch even if an arm with this label is already live (de-confliction override)")
    args = ap.parse_args(argv)

    if " " in args.label:
        ap.error("--label must not contain spaces")

    # Operator 2026-07-20 (binding): a codex arm must NEVER again be starved of
    # filesystem access. A prior arm launched with sandbox=workspace-write (writable
    # roots limited to the worktree + .git) could not write the mandatory SSD evidence
    # tier (/Volumes/VertigoDataTier/pact/evidence) -> it doom-looped for hours on
    # in-sandbox apparatus review, burning the operator's codex rate limit. Permanent
    # fix: a COMMIT-CAPABLE (non-read-only) arm is ALWAYS promoted to danger-full-access
    # (full filesystem + network). read-only stays available for pure-analysis arms that
    # write nothing. Worktree isolation (the default) + the commit serializer remain the
    # commit-discipline guarantee -- they are orthogonal to the OS file sandbox, so full
    # FS access does NOT relax how commits are made in the per-arm worktree.
    if args.sandbox == "workspace-write":
        print(
            "[codex_delegate] sandbox workspace-write -> danger-full-access "
            "(operator 2026-07-20: never FS-starve a commit-capable arm; worktree "
            "isolation + serializer still enforce commit discipline).",
            file=sys.stderr,
        )
        args.sandbox = "danger-full-access"

    RUNS.mkdir(parents=True, exist_ok=True)

    isolate = not args.no_isolate
    policy_refusal = _launch_policy_refusal(
        label=args.label,
        requested_sandbox=args.sandbox,
        isolate=isolate,
        prior_sandboxes=_prior_sandboxes(args.label),
        live_nonisolated_writers=_live_nonisolated_writer_count(),
    )
    if policy_refusal is not None:
        rc, message = policy_refusal
        print(message)
        return rc

    # De-confliction preflight: surface the live fleet before adding to it, and REFUSE a
    # duplicate-live-label (the over-launch this hardening extincts) unless --force.
    live = _live_labels()
    if live:
        print(f"[de-conflict] {len(live)} arm(s) live: {', '.join(sorted(live))}")
    if args.label in live and not args.force:
        print(f"REFUSED: an arm labeled '{args.label}' is already live. "
              f"Redirect it via .omx/tmp/codex_inbox/{args.label}.jsonl, or pass --force to run a second.")
        return 3

    # Back-pressure: the shared-tree writer cap above prevents multi-writer
    # confounding; this second bound stops even one legacy writer from growing
    # an already-unreviewable pile.
    pile = _uncommitted_pile_size()
    if pile > _MAX_UNCOMMITTED_PILE and not args.force:
        print(f"REFUSED (back-pressure): {pile} uncommitted files in the shared tree "
              f"(> {_MAX_UNCOMMITTED_PILE}). Launching more ACCRUES drift. "
              f"Harvest-drain done arms first "
              f"(tools/codex_harvest_commit.py --label <L> --stamp <S>), then relaunch — or --force.")
        return 4

    stamp = _utc()
    delegation_key = f"codex_delegate:{args.label}:{stamp}"

    if args.prompt_file:
        prompt_file = Path(args.prompt_file).resolve()
        if not prompt_file.is_file():
            ap.error(f"--prompt-file not found: {prompt_file}")
    else:
        prompt_file = RUNS / f"{args.label}_{stamp}.prompt.txt"
        prompt_file.write_text(args.prompt, encoding="utf-8")

    # ARTIFACT QUARANTINE (operator 2026-07-21): retired-vehicle artifacts are
    # HARVEST-SIGNAL-ONLY — a dispatch prompt anchoring on a quarantined identifier is
    # refused unless it carries the explicit signal-only waiver line. Fail-open only when
    # the manifest is absent; a corrupt manifest refuses (rc=12 either way on violation).
    try:
        from tac.artifact_quarantine import refuse_message, scan_text
        _q_hits = scan_text(prompt_file.read_text(encoding="utf-8"))
    except ImportError:
        print("[codex_delegate] WARNING: tac.artifact_quarantine unavailable — "
              "quarantine scan skipped", file=sys.stderr)
        _q_hits = []
    if _q_hits:
        print(refuse_message(_q_hits), file=sys.stderr)
        return 12

    # RECENCY-FLOOR ADVISORY (operator 2026-07-21 ×2, recurrence of the 07-19 law
    # "no recency floor — oldest math wins"): a research/survey prompt that scopes a
    # FAMILY to a recent year range (e.g. "2024-2026 SOTA") silently excludes the
    # classical corpus (Prony 1795 / Fourier 1807 / Chasles 1830 / KL 1940s — all
    # load-bearing in this campaign). Warn-only: year mentions are fine for "know the
    # latest"; the warning fires so the dispatcher re-checks the scope is not a floor.
    _recency_hits = re.findall(
        r"\b20[12]\d\s*[-–—]\s*(?:20)?[12]\d\b|\b(?:since|post|after)[- ]20[12]\d\b",
        prompt_file.read_text(encoding="utf-8"),
    )
    if _recency_hits:
        print(
            f"[codex_delegate] RECENCY-FLOOR ADVISORY: prompt contains year-range "
            f"scope(s) {_recency_hits[:4]} — verify this means 'include the latest', "
            f"NOT 'exclude the classical corpus' (memory: "
            f"feedback_no_recency_floor_oldest_math_keeps_winning_20260719).",
            file=sys.stderr,
        )

    # METRIC-NAMING ADVISORY (operator 2026-07-24 ×2 "Euclidean is not the only metric
    # we have" + "start at more optimal in our laddering"; recurrence probe 2026-07-24
    # "Did you already forget the more optimal over Euclidean?"): a charter whose work is
    # pricing-shaped (solve/price/rank/trust region/waterfill/proposal) but which never
    # NAMES the metric per surface defaults to identity-Euclidean naive — the ms1 receipt
    # (0/1,200 CVP wins in identity metric) is the empirical anchor. Warn-only per the
    # fmtools never-blocking doctrine: pure read/survey charters legitimately lack metric
    # clauses. Memory: metric_first_charters_never_euclidean_naive_20260724.
    _prompt_text = prompt_file.read_text(encoding="utf-8")
    _pricing_shaped = re.search(
        r"\b(solve|solver|pric(?:e|ing)|rank(?:ing)?|trust[- ]region|waterfill|"
        r"proposal|warm[- ]start|coder race|frontier)\b",
        _prompt_text, re.IGNORECASE,
    )
    _metric_named = re.search(
        r"margin[- ]Fisher|Fisher|pose quadratic|(?:≤6|<=\s*6)[- ]dim.{0,40}quadratic|"
        r"optimal_metric|scorer metric|rank[- ]4 (?:head|hyperplane)|Bregman|"
        r"contest units|score[- ]native",
        _prompt_text, re.IGNORECASE,
    )
    if _pricing_shaped and not _metric_named:
        print(
            "[codex_delegate] METRIC-NAMING ADVISORY: prompt is pricing-shaped "
            f"(matched {_pricing_shaped.group(0)!r}) but NAMES no metric per surface — "
            "identity-Euclidean-naive spawn risk (seg=margin-Fisher/rank-4 · pose=exact "
            "≤6-dim quadratic · rate=real-coder contest units · Euclid=labeled "
            "control ONLY; memory: metric_first_charters_never_euclidean_naive_20260724).",
            file=sys.stderr,
        )

    # SCORER-DERIVATION GATE — BLOCKING (operator 2026-07-24 ×2: "Many implementations
    # were probably naive and not optimized against recursive analysis of upstream
    # evaluate.py" + "How can we prevent that from happening permanently?"). Empirical
    # anchor: dm2's generic disk/global candidate menu produced a 2,524x naive-menu
    # upper bound and dm3's translation-only xi proxy produced an uninterpretable 0/36 —
    # both spawned THROUGH the warn-only advisory above, which the dispatcher habituated
    # to (four consecutive ignored warnings on 2026-07-24). Warnings do not survive
    # habituation; only refusal does. Rule: a charter that CONSTRUCTS candidates/
    # proposals/menus/programs/writes MUST contain a "SCORER-DERIVATION" section naming,
    # per constructed family, the in-tree recursive-evaluate.py surface it derives from
    # (segnet fractal factorization rank-4 head · frozen-scorer exact factorization ·
    # at1 analytic atlas · sn1 telemetry · #580 resize-kernel adjoint/null projector ·
    # #583 corrected inner-Jacobian bank · ERF/stem lattice) or an explicit
    # "[naive-baseline]" label with the scorer-derived replacement named. Pure read/
    # survey/review charters do not match the construction shape and pass untouched.
    # Waiver: TAC_SCORER_DERIVATION_WAIVED=<non-placeholder rationale> (logged loudly).
    _construction_shaped = re.search(
        r"\b(candidate (?:menu|famil\w*|construction)|construct(?:ing)? (?:candidates|"
        r"proposals|writes|programs)|proposal (?:source|ladder|famil\w*)|"
        r"minimal (?:RGB )?write|realiz(?:e|ation) (?:race|cure)|history program|"
        r"preimage solve|emit(?:ter)? architecture|compose .{0,40}archive|"
        r"paint(?:ed)? (?:color|order)|exception (?:stream|aiming))",
        _prompt_text, re.IGNORECASE,
    )
    _scorer_derivation_named = re.search(
        r"SCORER-DERIVATION|scorer[- ]recursi(?:on|ve)|derived? from the (?:resize[- ]"
        r"kernel|rank[- ]4|stem lattice|ERF|corrected inner[- ]Jacobian|frozen[- ]scorer "
        r"factorization|analytic atlas)|\[naive-baseline\]",
        _prompt_text, re.IGNORECASE,
    )
    if _construction_shaped and not _scorer_derivation_named:
        _waiver = os.environ.get("TAC_SCORER_DERIVATION_WAIVED", "").strip()
        if _waiver and _waiver.lower() not in {"<rationale>", "<reason>", "1", "true", "yes"}:
            print(
                "[codex_delegate] SCORER-DERIVATION GATE WAIVED: "
                f"rationale={_waiver!r} (construction-shaped charter spawned without a "
                "scorer-derivation section under explicit waiver).",
                file=sys.stderr,
            )
        else:
            print(
                "[codex_delegate] SCORER-DERIVATION GATE — REFUSED (rc=15): prompt is "
                f"construction-shaped (matched {_construction_shaped.group(0)!r}) but "
                "contains NO 'SCORER-DERIVATION' section. Every constructed candidate/"
                "proposal/menu/program family must name the in-tree recursive-"
                "evaluate.py surface it derives from (rank-4 head hyperplanes · frozen-"
                "scorer factorization · at1 atlas · sn1 telemetry · #580 resize-kernel "
                "adjoint · #583 corrected inner-Jacobian bank · ERF/stem lattice) or "
                "carry an explicit [naive-baseline] label with the scorer-derived "
                "replacement named. Fix the charter, or set "
                "TAC_SCORER_DERIVATION_WAIVED=<real rationale> (placeholders rejected). "
                "Operator law 2026-07-24: spawns 'not optimized against recursive "
                "analysis of upstream evaluate.py' are FORBIDDEN; empirical anchors: "
                "dm2 2,524x naive-menu bound, dm3 xi 0/36 naive-program bound.",
                file=sys.stderr,
            )
            return 15

    # Give the arm a watched inbox + prepend the poll-and-consume contract so it stays
    # amendable mid-run. The wrapped prompt is what the launcher feeds codex; the original
    # prompt_file is preserved untouched (and recorded in the ledger for provenance).
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    worktree = WORKTREE_DIR / f"{args.label}_{stamp}"
    branch = f"codexwt/{args.label}_{stamp}"
    inbox = INBOX_DIR / f"{args.label}.jsonl"
    broadcast = INBOX_DIR / "_broadcast.jsonl"
    manifest = MANIFEST_DIR / f"{args.label}_{stamp}.json"
    inbox.touch(exist_ok=True)
    broadcast.touch(exist_ok=True)
    wrapped = RUNS / f"{args.label}_{stamp}.wrapped.prompt.txt"
    commit_contract = (
        _WORKTREE_CONTRACT.format(worktree=worktree, branch=branch, manifest=manifest, repo=REPO)
        if isolate else ""
    )
    wrapped.write_text(
        commit_contract
        + _INBOX_CONTRACT.format(
            inbox=inbox, broadcast=broadcast, delegation_key=delegation_key
        )
        + prompt_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    entry_prompt, resume_prompt = _write_compact_prompts(
        label=args.label,
        stamp=stamp,
        wrapped_prompt=wrapped,
        delegation_key=delegation_key,
    )
    log = RUNS / f"{args.label}_{stamp}.log"
    last = RUNS / f"{args.label}_{stamp}.last.txt"
    done = RUNS / f"{args.label}_{stamp}.done"
    launcher = _write_launcher(args.label, stamp, entry_prompt, args.model,
                               args.effort, args.sandbox, log, last, done, isolate, worktree,
                               allow_network=args.allow_network,
                               resume_prompt_file=resume_prompt,
                               delegation_key=delegation_key)

    _append_ledger({
        "label": args.label, "stamp": stamp, "model": args.model, "effort": args.effort,
        "sandbox": args.sandbox, "launcher": str(launcher), "log": str(log),
        "last": str(last), "done_marker": str(done), "prompt_file": str(prompt_file),
        "wrapped_prompt": str(wrapped), "entry_prompt": str(entry_prompt),
        "resume_prompt": str(resume_prompt), "delegation_key": delegation_key,
        "isolate": isolate, "worktree": str(worktree) if isolate else None,
        "branch": branch if isolate else None,
        "launched_utc": datetime.now(UTC).isoformat(), "status": "running",
    })

    launched = False
    if not args.no_launch:
        # HEADLESS + REAPER-IMMUNE launch (task #525, measured 2026-07-17). Two layers:
        # 1. HEADLESS (operator 2026-07-16: "They should be launched headless"): detach into a NEW
        #    SESSION (setsid) so the arm survives the parent Claude harness SIGURG-144 process-group
        #    death WITHOUT opening a GUI Terminal window (ends the orphaned-window class, ff0f884b35).
        # 2. CONTROLLING PTY (the ff0f884b35 regression fix): the prior Terminal launch gave arms a
        #    TTY as a side effect; going headless removed it, and the operator's fleet launchd agent
        #    com.vertigo.claude-code-reaper (60s cadence, GRACE=300s) then SIGTERM'd EVERY codex arm
        #    at ~5:20-5:55 (10/10 kills logged in /tmp/com.vertigo.claude-code-reaper.log, incl. a
        #    clean-env low-token probe; a renamed-argv probe survived — argv \bcodex\b + tty=?? +
        #    stdin=/dev/null is the exact reap signature). script(1) restores a controlling pty so
        #    the arm chain is classified as a LIVE terminal session (the reaper's own exclusion
        #    class) — honest: arms ARE live supervised sessions (events-log custody + codex_status).
        # Single shared spawn core — no copy-paste second impl (spawn_durable_daemon owns it).
        # Prefer the package path so tests patching tools.spawn_durable_daemon see one module.
        try:
            from tools.spawn_durable_daemon import spawn_detached_verified
        except ImportError:  # script-mode fallback (tools/ on sys.path above)
            from spawn_durable_daemon import spawn_detached_verified
        wrap_log = launcher.with_suffix(".wrap.log")
        child_env = dict(os.environ)
        # Plain output under the pty: codex detects a TTY and would emit ANSI/spinner noise into
        # the tee'd logs; TERM=dumb keeps the log files machine-readable (parity with pre-pty logs).
        child_env["TERM"] = "dumb"
        try:
            _proc, _pgid, ok, vinfo = spawn_detached_verified(
                ["bash", str(launcher)],
                wrap_log,
                with_pty=not args.no_pty,
                verify_s=3.0,
                env=child_env,
                cwd=str(REPO),
            )
            launched = bool(ok)
            if not ok:
                print(f"ERROR headless launch did not survive the verify window: "
                      f"{vinfo.get('exit_str')} — see {wrap_log}")
        except OSError as e:
            print(f"WARN headless launch failed: {e}")

    print(json.dumps({
        "delegated": args.label, "launched": launched, "launcher": str(launcher),
        "log": str(log), "last": str(last), "done_marker": str(done),
        "events_log": str(EVENTS),
        "arm_notifier_once": NOTIFIER_CMD,
    }, indent=2))
    if not EVENTS.exists():
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        EVENTS.touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
