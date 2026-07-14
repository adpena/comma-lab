#!/usr/bin/env python3
"""codex_delegate.py — canonical wrapper for delegating a task to a codex agent
with AUTOMATIC completion notification back to the Claude main loop.

WHY: codex agents launched via osascript Terminal windows are durable (survive the
harness SIGURG-144 that kills bg-bash) but fire-and-forget — the main loop had to
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
        [--model gpt-5.6-sol] [--effort ultra] [--sandbox workspace-write] \
        [--no-launch]   # write the launcher + ledger row but do not osascript-launch

    # then ONCE per session, arm the notifier (Claude Monitor tool, persistent):
    #   tail -n +1 -f .omx/tmp/codex_runs/codex_events.log | grep --line-buffered '^DONE'

STATUS:  .venv/bin/python tools/codex_status.py
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / ".omx" / "tmp" / "codex_runs"
EVENTS = RUNS / "codex_events.log"
LEDGER = REPO / ".omx" / "state" / "codex_delegations.jsonl"
INBOX_DIR = REPO / ".omx" / "tmp" / "codex_inbox"  # per-arm watched inboxes (tools/codex_msg.py producer)
MANIFEST_DIR = REPO / ".omx" / "tmp" / "codex_manifests"  # per-arm files-touched manifest (harvest-commit contract)

# Back-pressure: a workspace-write arm CANNOT commit (sandbox blocks .git/objects), so every arm
# strands its diff UNCOMMITTED in the shared tree. Above this many uncommitted files the dispatcher
# REFUSES to launch (accruing more drift) until the pile is harvest-drained. --force overrides.
_MAX_UNCOMMITTED_PILE = 40

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
This is how you stay amendable while running. Absent/empty inbox -> ignore, proceed normally.
=== END LIVE INBOX ===

"""

# Commit-path contract (apparatus fix for codex_workspace_write_sandbox_blocks_git_objects_20260712):
# a workspace-write arm CANNOT write .git/objects -> `git add`/`git commit` (and the serializer) fail
# rc=128 -> its work strands UNCOMMITTED in the shared tree. Instead of committing, the arm writes a
# MANIFEST of the files it touched; MAIN (unsandboxed) harvests + REVIEWS + serializer-commits them via
# tools/codex_harvest_commit.py. Prepended to EVERY delegated prompt so no arm silently orphans its work.
_HARVEST_CONTRACT = """=== COMMIT PATH — YOU CANNOT COMMIT; MANIFEST INSTEAD (mandatory; skipping ORPHANS your work) ===
Your sandbox (workspace-write) BLOCKS writes to .git/objects, so `git add` / `git commit` / the
serializer WILL fail rc=128. DO NOT attempt to commit, and DO NOT try to bypass it (no direct-git,
no override). Instead:
  1. Make your file edits normally.
  2. BEFORE your FINAL message, write the EXACT repo-relative paths you created/modified to:
       {manifest}
     as JSON: {{"files": ["path/a.md", "src/tac/b.py", ...],
                "code_files": ["src/tac/b.py", ...],   # subset needing code review
                "verdict": "<one-line result>", "review_notes": "<what MAIN should check>"}}
  3. Repeat your verdict + what-to-review in your FINAL message.
MAIN (unsandboxed) reads the manifest, REVIEWS, and serializer-commits your diff — that harvest IS the
non-negotiable follow-up review. An arm that edits files but writes NO manifest ORPHANS its work into an
un-attributable shared-tree pile. Writing the manifest is MANDATORY.
=== END COMMIT PATH ===

"""


def _uncommitted_pile_size() -> int:
    r = subprocess.run(["git", "-C", str(REPO), "status", "--short"],
                       capture_output=True, text=True)
    return sum(1 for ln in r.stdout.splitlines() if ln.strip())

# Transient-death auto-recovery (apparatus fix for the codex_probe_token_limit_death /
# "Selected model is at capacity" bug class): a capacity/rate-limit/disconnect is a
# SERVER blip, not a token limit — a bump of model_context_window would NOT help. On such
# a death the launcher re-runs the SAME prompt after backoff; the agent self-resumes from
# its tools/subagent_checkpoint.py record (the crash-resume protocol) so no work is orphaned.
# Fatal errors (bad flag, syntax, non-transient) do NOT match the signature → no retry loop.
_MAX_CAPACITY_RETRIES = 8            # linear backoff attempts before giving up
_CAPACITY_BACKOFF_STEP_SECONDS = 20  # attempt N waits N * step seconds (20,40,...,160)
_TRANSIENT_DEATH_SIGNATURE = (
    "at capacity|rate.?limit|429|overloaded|temporarily unavailable|"
    "stream disconnected|error sending request|connection reset|timed out|503|502"
)

# The one-line notifier the main loop arms once (printed on launch for convenience).
NOTIFIER_CMD = f"tail -n +1 -f {EVENTS} | grep --line-buffered '^DONE'"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _live_labels() -> list[str]:
    """Labels of currently-alive delegated arms (robust: pgrep on the label_stamp
    token per ledger row, never `pgrep -fl | head` which the inlined prompt clips)."""
    if not LEDGER.is_file():
        return []
    seen: dict[str, dict] = {}
    for line in LEDGER.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
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


def _write_launcher(label: str, stamp: str, prompt_file: Path, model: str,
                    effort: str, sandbox: str, log: Path, last: Path,
                    done: Path) -> Path:
    launcher = RUNS / f"launch_{label}_{stamp}.sh"
    # The launcher: run codex, tee to log, capture final message via -o, then on
    # exit append a DONE line (with a one-line summary from the tail of `last`) to
    # the shared events log + write a per-run .done marker. `exec bash` keeps the
    # Terminal window open for inspection AFTER the notification has fired.
    launcher.write_text(
        f"""#!/bin/bash
set +e
cd {REPO} || exit 1
mkdir -p {RUNS}
echo "START {label} {stamp} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> {EVENTS}
echo "=== codex delegate '{label}' [{model}/{effort}/{sandbox}] {stamp} ==="
echo "log:  {log}"
codex exec \\
  --skip-git-repo-check \\
  --sandbox {sandbox} \\
  -m {model} \\
  -c model_reasoning_effort={effort} \\
  -o {last} \\
  "$(cat {prompt_file})" \\
  2>&1 | tee {log}
RC=${{PIPESTATUS[0]}}
# --- transient-death auto-recovery: capacity/rate-limit/disconnect is a SERVER blip, not a
# token limit. Re-run the same prompt after backoff; the agent self-resumes from its
# subagent_checkpoint so no work is orphaned. Fatal (non-transient) errors do not match → no loop.
attempt=0
while [ "$RC" -ne 0 ] && [ "$attempt" -lt {_MAX_CAPACITY_RETRIES} ] && \\
      grep -qiE '{_TRANSIENT_DEATH_SIGNATURE}' {log}; do
  attempt=$((attempt+1))
  backoff=$(({_CAPACITY_BACKOFF_STEP_SECONDS} * attempt))
  echo "RETRY {label} transient rc=$RC attempt=$attempt/{_MAX_CAPACITY_RETRIES} backoff=${{backoff}}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> {EVENTS}
  echo "=== TRANSIENT death (rc=$RC) — re-run $attempt/{_MAX_CAPACITY_RETRIES} in ${{backoff}}s; agent self-resumes from checkpoint ===" | tee -a {log}
  sleep $backoff
  codex exec \\
    --skip-git-repo-check \\
    --sandbox {sandbox} \\
    -m {model} \\
    -c model_reasoning_effort={effort} \\
    -o {last} \\
    "$(cat {prompt_file})" \\
    2>&1 | tee -a {log}
  RC=${{PIPESTATUS[0]}}
done
# one-line summary from the tail of the final-message file (best-effort, sanitized)
SUMMARY=$(tail -c 400 {last} 2>/dev/null | tr '\\n' ' ' | tr -s ' ' | sed 's/[|]/ /g' | tail -c 200)
echo "DONE {label} rc=$RC {stamp} $(date -u +%Y-%m-%dT%H:%M:%SZ) :: ${{SUMMARY}}" >> {EVENTS}
printf 'rc=%s\\nfinished_utc=%s\\nlog=%s\\nlast=%s\\n' "$RC" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "{log}" "{last}" > {done}
echo ""
echo "=== codex delegate '{label}' exited rc=$RC — DONE line appended to codex_events.log ==="
echo "(window kept open; review {last})"
exec bash
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Delegate a task to codex with auto completion notification.")
    ap.add_argument("--label", required=True, help="short slug for this delegation (no spaces)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt-file", help="path to a file containing the codex prompt")
    g.add_argument("--prompt", help="inline prompt string (written to a prompt file)")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="ultra", choices=["low", "medium", "high", "xhigh", "ultra"])
    ap.add_argument("--sandbox", default="workspace-write",
                    choices=["read-only", "workspace-write", "danger-full-access"])
    ap.add_argument("--no-launch", action="store_true", help="write launcher + ledger but do not osascript-launch")
    ap.add_argument("--force", action="store_true",
                    help="launch even if an arm with this label is already live (de-confliction override)")
    args = ap.parse_args(argv)

    if " " in args.label:
        ap.error("--label must not contain spaces")
    RUNS.mkdir(parents=True, exist_ok=True)

    # De-confliction preflight: surface the live fleet before adding to it, and REFUSE a
    # duplicate-live-label (the over-launch this hardening extincts) unless --force.
    live = _live_labels()
    if live:
        print(f"[de-conflict] {len(live)} arm(s) live: {', '.join(sorted(live))}")
    if args.label in live and not args.force:
        print(f"REFUSED: an arm labeled '{args.label}' is already live. "
              f"Redirect it via .omx/tmp/codex_inbox/{args.label}.jsonl, or pass --force to run a second.")
        return 3

    # Back-pressure: refuse to launch into a growing drift pile. Arms CANNOT commit (sandbox blocks
    # .git/objects), so every launch accrues more stranded, un-attributable work in the shared tree.
    pile = _uncommitted_pile_size()
    if pile > _MAX_UNCOMMITTED_PILE and not args.force:
        print(f"REFUSED (back-pressure): {pile} uncommitted files in the shared tree "
              f"(> {_MAX_UNCOMMITTED_PILE}). Codex arms CANNOT commit (sandbox blocks .git/objects), so "
              f"launching more ACCRUES drift. Harvest-drain done arms first "
              f"(tools/codex_harvest_commit.py --label <L> --stamp <S>), then relaunch — or --force.")
        return 4

    stamp = _utc()

    if args.prompt_file:
        prompt_file = Path(args.prompt_file).resolve()
        if not prompt_file.is_file():
            ap.error(f"--prompt-file not found: {prompt_file}")
    else:
        prompt_file = RUNS / f"{args.label}_{stamp}.prompt.txt"
        prompt_file.write_text(args.prompt, encoding="utf-8")

    # Give the arm a watched inbox + prepend the poll-and-consume contract so it stays
    # amendable mid-run. The wrapped prompt is what the launcher feeds codex; the original
    # prompt_file is preserved untouched (and recorded in the ledger for provenance).
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    inbox = INBOX_DIR / f"{args.label}.jsonl"
    broadcast = INBOX_DIR / "_broadcast.jsonl"
    manifest = MANIFEST_DIR / f"{args.label}_{stamp}.json"
    inbox.touch(exist_ok=True)
    broadcast.touch(exist_ok=True)
    wrapped = RUNS / f"{args.label}_{stamp}.wrapped.prompt.txt"
    wrapped.write_text(
        _HARVEST_CONTRACT.format(manifest=manifest)
        + _INBOX_CONTRACT.format(inbox=inbox, broadcast=broadcast)
        + prompt_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    log = RUNS / f"{args.label}_{stamp}.log"
    last = RUNS / f"{args.label}_{stamp}.last.txt"
    done = RUNS / f"{args.label}_{stamp}.done"
    launcher = _write_launcher(args.label, stamp, wrapped, args.model,
                               args.effort, args.sandbox, log, last, done)

    _append_ledger({
        "label": args.label, "stamp": stamp, "model": args.model, "effort": args.effort,
        "sandbox": args.sandbox, "launcher": str(launcher), "log": str(log),
        "last": str(last), "done_marker": str(done), "prompt_file": str(prompt_file),
        "launched_utc": datetime.now(UTC).isoformat(), "status": "running",
    })

    launched = False
    if not args.no_launch:
        osa = f'tell application "Terminal"\n    do script "bash {launcher}"\nend tell'
        r = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
        launched = r.returncode == 0
        if not launched:
            print(f"WARN osascript launch failed rc={r.returncode}: {r.stderr.strip()[:200]}")

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
