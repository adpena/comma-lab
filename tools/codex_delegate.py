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
        [--model gpt-5.6-sol] [--effort ultra] [--sandbox danger-full-access] \
        [--no-launch]   # write the launcher + ledger row but do not osascript-launch

    # then ONCE per session, arm the notifier (Claude Monitor tool, persistent):
    #   tail -n +1 -f .omx/tmp/codex_runs/codex_events.log | grep --line-buffered '^DONE'

STATUS:  .venv/bin/python tools/codex_status.py
"""
from __future__ import annotations

import argparse
import fcntl
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
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
    r = subprocess.run(["git", "-C", str(REPO), "status", "--short"],
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


def _write_launcher(label: str, stamp: str, prompt_file: Path, model: str,
                    effort: str, sandbox: str, log: Path, last: Path,
                    done: Path, isolate: bool = True,
                    worktree: Path | None = None,
                    allow_network: bool = False) -> Path:
    launcher = RUNS / f"launch_{label}_{stamp}.sh"
    worktree = worktree or (WORKTREE_DIR / f"{label}_{stamp}")
    # --- codex-contract fix (2026-07-15): an isolated git worktree writes its objects + index to the
    # SHARED <repo>/.git, which lives OUTSIDE the workspace-write sandbox root (=the worktree dir). So a
    # workspace-write isolated arm literally CANNOT `git add`/`commit` ("Operation not permitted") and
    # lands PASS_UNCOMMITTED, forcing main to harvest by hand (bit 4 arms 2026-07-15). Make the shared
    # git dir a writable_root so isolated arms commit their own work. (danger-full-access already has
    # full write, so this only matters for workspace-write.) network_access is opt-in per --allow-network
    # (least-privilege) — needed for Modal-dispatch arms whose sandbox otherwise cannot resolve DNS.
    _extra_c: list[str] = []
    if isolate and sandbox == "workspace-write":
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
            f'  exec bash\n'
            f'fi'
        )
    else:
        isolate_setup = '# --no-isolate: running in the shared repo tree (legacy; use for read-only arms only)'
    # The launcher: (worktree isolate ->) run codex, tee to log, capture final message via -o, then on
    # exit append a DONE line (with a one-line summary from the tail of `last`) to the shared events log
    # + write a per-run .done marker recording the worktree/branch for main to merge. `exec bash` keeps
    # the Terminal window open for inspection AFTER the notification has fired.
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
    --sandbox "$ORIGINAL_SANDBOX" \\
    -m {model} \\
    -c model_reasoning_effort={effort}{extra_c} \\
    -o {last} \\
    "$(cat {prompt_file})" \\
    2>&1 | tee -a {log}
  RC=${{PIPESTATUS[0]}}
done
# one-line summary from the tail of the final-message file (best-effort, sanitized)
SUMMARY=$(tail -c 400 {last} 2>/dev/null | tr '\\n' ' ' | tr -s ' ' | sed 's/[|]/ /g' | tail -c 200)
echo "DONE {label} rc=$RC {stamp} $(date -u +%Y-%m-%dT%H:%M:%SZ) :: ${{SUMMARY}}" >> {EVENTS}
printf 'rc=%s\\nfinished_utc=%s\\nlog=%s\\nlast=%s\\nworktree=%s\\nbranch=%s\\nisolate=%s\\n' "$RC" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "{log}" "{last}" "{worktree if isolate else ''}" "{branch if isolate else ''}" "{'1' if isolate else '0'}" > {done}
echo ""
echo "=== codex delegate '{label}' exited rc=$RC — DONE appended; MAIN: harvest+merge worktree branch to main ==="
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
    ap.add_argument("--force", action="store_true",
                    help="launch even if an arm with this label is already live (de-confliction override)")
    args = ap.parse_args(argv)

    if " " in args.label:
        ap.error("--label must not contain spaces")
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
        + _INBOX_CONTRACT.format(inbox=inbox, broadcast=broadcast)
        + prompt_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    log = RUNS / f"{args.label}_{stamp}.log"
    last = RUNS / f"{args.label}_{stamp}.last.txt"
    done = RUNS / f"{args.label}_{stamp}.done"
    launcher = _write_launcher(args.label, stamp, wrapped, args.model,
                               args.effort, args.sandbox, log, last, done, isolate, worktree,
                               allow_network=args.allow_network)

    _append_ledger({
        "label": args.label, "stamp": stamp, "model": args.model, "effort": args.effort,
        "sandbox": args.sandbox, "launcher": str(launcher), "log": str(log),
        "last": str(last), "done_marker": str(done), "prompt_file": str(prompt_file),
        "isolate": isolate, "worktree": str(worktree) if isolate else None,
        "branch": branch if isolate else None,
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
