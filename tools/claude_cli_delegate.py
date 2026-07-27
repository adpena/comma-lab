#!/usr/bin/env python
"""Spawn a detached headless Claude Code CLI arm — the Opus 5 deployment path.

Operator directives 2026-07-27: "Use Opus sub agents for everything. Only use
Fable for the hardest and most important work." + "Opus 5" + "Deploy them in
such a way as to access opus 5." MEASURED facts this tool encodes:
  - the in-session Agent tool's `opus` alias resolves to claude-opus-4-8;
  - a FRESH headless CLI session accepts `--model claude-opus-5` (probe
    2026-07-27: modelUsage == ['claude-opus-5']).
So arm-class work reaches Opus 5 only through fresh `claude -p` sessions.
This wrapper mirrors the codex_delegate essentials: isolated worktree, prompt
file, detached spawn (nohup+setsid+disown — the rc=143/144 harness-kill-immune
Pattern A), JSON log, `.last.txt` final-message capture, stamp record.

Usage:
  .venv/bin/python tools/claude_cli_delegate.py \
      --label ddm_xyz --prompt-file path/to/charter.md \
      [--model claude-opus-5] [--no-worktree]

The arm runs with --dangerously-skip-permissions INSIDE its isolated worktree
(the established codex danger-full-access equivalent); it inherits the
worktree's CLAUDE.md non-negotiables. MAIN reviews + lands its branch.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / ".omx" / "tmp" / "claude_cli_runs"
WORKTREES = REPO / ".omx" / "tmp" / "claude_cli_worktrees"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--no-worktree", action="store_true",
                    help="run in the main tree cwd (read-only/report arms ONLY)")
    args = ap.parse_args()

    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{args.label}_{stamp}"
    RUNS.mkdir(parents=True, exist_ok=True)
    prompt_path = Path(args.prompt_file).resolve()
    if not prompt_path.is_file():
        print(f"ERROR: prompt file missing: {prompt_path}", file=sys.stderr)
        return 2

    if args.no_worktree:
        workdir = REPO
        branch = None
    else:
        WORKTREES.mkdir(parents=True, exist_ok=True)
        workdir = WORKTREES / run_id
        branch = f"clwt/{run_id}"
        subprocess.run(
            ["git", "worktree", "add", "-b", branch, str(workdir), "HEAD"],
            cwd=REPO, check=True, capture_output=True, text=True,
        )

    log = RUNS / f"{run_id}.json.log"
    last = RUNS / f"{run_id}.last.txt"
    # Detached spawn: setsid + nohup semantics via start_new_session, stdin
    # closed, output to the log; a trailing python step extracts .result into
    # .last.txt so harvest never needs to parse the full JSON stream.
    shell = (
        f"claude -p \"$(cat {prompt_path})\" --model {args.model} "
        f"--dangerously-skip-permissions --output-format json > {log} 2>&1; "
        f"rc=$?; {sys.executable} -c \""
        f"import json,sys;\n"
        f"try: d=json.load(open('{log}')); open('{last}','w').write(d.get('result',''))\n"
        f"except Exception as e: open('{last}','w').write('PARSE-FAIL: '+repr(e))\" ; "
        f"echo DONE rc=$rc >> {log}"
    )
    proc = subprocess.Popen(
        ["bash", "-c", shell],
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    record = {
        "run_id": run_id, "label": args.label, "stamp": stamp,
        "model": args.model, "pid": proc.pid, "worktree": str(workdir),
        "branch": branch, "log": str(log), "last": str(last),
        "prompt_sha_file": str(prompt_path),
    }
    ledger = RUNS / "runs.jsonl"
    with ledger.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
