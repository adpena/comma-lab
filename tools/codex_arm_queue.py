#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Codex arm queue + SATURATION actuator.

The fleet is supposed to stay saturated: whenever a codex arm lands, the next
charter fires. That only happens reliably if it is MECHANICAL. Today (2026-08-04)
proved the alternative twice — a routing law that lived in a memo was applied on
zero spawns, and a sandbox flag omitted by hand killed an arm 25 minutes from a
frontier row. So saturation lives here, not in anyone's memory.

NOT the same object as ``.omx/state/dispatch_queue.md`` (last touched 2026-05-17),
which queues PAID GPU LANE dispatches (``scripts/remote_lane_*.sh``, dollar cost
bands, Vast.ai/Modal). Same word, different thing — see the retrieval-hazard class
(task #867). This queue holds CODEX ARM CHARTERS: prompt files spawned via the
canonical detached ``codex exec`` Pattern A.

Queue file: ``.omx/state/codex_arm_queue.jsonl`` — append-only rows:
    {"name","prompt_path","rank","owns_scorer","status","note", ...}
``status`` ∈ queued | live | landed | dropped. Latest row per ``name`` wins, so
status changes are appends, never rewrites (append-only custody).

Safety, all fail-closed:
  * hard cap (default 4) on concurrent codex arms — never exceeded;
  * at most ONE scorer-owning arm live at a time (the one-full-n600 rule);
  * refuses to spawn a name that is already live;
  * refuses a charter whose prompt file is missing;
  * every spawn carries ``--add-dir`` for the SSD tier (the flag whose absence
    killed fz3) and the canonical Pattern A detachment;
  * ``TAC_CODEX_SATURATE_OFF=1`` is the kill switch;
  * ``--dry-run`` is the default for ``saturate``; spawning requires ``--spawn``.

Usage:
    codex_arm_queue.py status                     # live arms, cap, next charter
    codex_arm_queue.py add --name X --prompt P --rank 30 [--owns-scorer]
    codex_arm_queue.py mark --name X --status landed
    codex_arm_queue.py saturate                   # report the gap (dry run)
    codex_arm_queue.py saturate --spawn           # actually fill the gap
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
QUEUE = _REPO / ".omx" / "state" / "codex_arm_queue.jsonl"
RUNS = _REPO / ".omx" / "tmp" / "codex_runs"
SPAWN_LOG = _REPO / ".omx" / "state" / "codex_arm_spawn_log.jsonl"

DEFAULT_CAP = 4
SSD_ADD_DIR = "/Volumes/VertigoDataTier/pact"
KILL_SWITCH = "TAC_CODEX_SATURATE_OFF"
_LIVE_STATUSES = frozenset({"queued", "live"})


# --- queue state (pure-ish) ------------------------------------------------------


def load_rows(path: Path = QUEUE) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn tail; never crash the actuator
    return rows


def latest_by_name(rows: list[dict]) -> dict[str, dict]:
    """Latest row per name wins FIELD-BY-FIELD — status changes are appends.

    Merging, not replacing: a ``mark`` row carries only {name,status}, so a naive
    last-row-wins silently drops ``prompt_path`` and the charter becomes unspawnable
    (observed 2026-08-04: ``would spawn: fz4 ()``). Later rows override the fields
    they actually set; everything else survives from the row that set it.
    """
    out: dict[str, dict] = {}
    for row in rows:
        name = row.get("name")
        if isinstance(name, str) and name:
            merged = dict(out.get(name, {}))
            merged.update({k: v for k, v in row.items() if v is not None})
            out[name] = merged
    return out


def append_row(row: dict, path: Path = QUEUE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("ts", time.time())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_arm_names(ps_output: str) -> set[str]:
    """Extract arm names from `ps` command lines via the -o receipt path.

    Split from the process listing so it is testable against a real fixture line —
    the earlier version asserted only `isinstance(..., set)`, which passes on a
    permanently broken function (and it WAS broken: see live_arm_names).
    """
    names: set[str] = set()
    for line in ps_output.splitlines():
        if "codex exec" not in line:
            continue
        for token in line.replace("'", " ").replace('"', " ").split():
            if token.endswith(".last.txt"):
                names.add(Path(token).name[: -len(".last.txt")])
    return names


def live_arm_names() -> set[str]:
    """Names of codex arms with a running process, read from the OS not the ledger.

    The ledger can lie (an arm dies without marking itself — fz3 did exactly that);
    the process table cannot.

    `ps -eo pid,command`, NOT `pgrep -af`: macOS pgrep has no `-a` flag, silently
    prints bare PIDs, and the name parser could never match — so this returned an
    empty set on every call while four arms were running (MEASURED 2026-08-04).
    A liveness detector that always reports zero makes the cap vacuous.
    """
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,command"], capture_output=True, text=True, timeout=15
        )
    except Exception:
        return set()
    return parse_arm_names(proc.stdout)


def next_charters(rows: list[dict], live: set[str], slots: int, scorer_taken: bool) -> list[dict]:
    """Rank-ordered charters that may fire right now, honouring the scorer rule."""
    latest = latest_by_name(rows)
    ready = [
        r
        for r in latest.values()
        if r.get("status") in _LIVE_STATUSES and r.get("name") not in live
    ]
    ready.sort(key=lambda r: (r.get("rank", 999), r.get("name", "")))
    picked: list[dict] = []
    for row in ready:
        if len(picked) >= slots:
            break
        if row.get("owns_scorer"):
            if scorer_taken:
                continue  # one full-n600 job at a time, fleet-wide
            scorer_taken = True
        picked.append(row)
    return picked


# --- spawning --------------------------------------------------------------------


# BSD-correct detach: fork + setsid(2) + exec, the mechanism proven in
# tools/codex_companion_spawn.sh. `nohup ... & disown` is NOT sufficient — disown
# removes the job from the shell's JOB TABLE but leaves the child in the shell's
# PROCESS GROUP, so a group-directed SIGURG/SIGTERM (the harness sends one at ~3min,
# the m77/rc=143 class) still reaps it. setsid makes the child its own session
# leader, out of reach of that signal. macOS has NO setsid(1), so it must be done
# in Python. MEASURED 2026-08-04: four hand-rolled nohup+disown arms died within
# 23 seconds of each other, mid-work, with no error and no final message.
_DETACH_PY = (
    "import os,sys\n"
    "log=sys.argv[1]; cmd=sys.argv[2:]\n"
    "if os.fork()>0: os._exit(0)\n"
    "os.setsid()\n"
    "fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)\n"
    "os.dup2(fd,1); os.dup2(fd,2)\n"
    "os.dup2(os.open(os.devnull,os.O_RDONLY),0)\n"
    "os.execvp(cmd[0],cmd)\n"
)


def spawn_command(name: str, prompt_path: str) -> str:
    """The canonical session-detached spawn command for one arm.

    The detached child is a WRAPPER, not codex itself, so every termination
    leaves a typed receipt at ``<name>.done``:

        rc=0            -> codex finished cleanly
        rc=N (N>0)      -> codex exited with an error of its own
        signal=TERM|INT|HUP -> something REAPED it (trap fired)
        no .done file   -> SIGKILL, or the wrapper itself vanished

    Without this the four states are indistinguishable, which is exactly how
    2026-08-04 produced two rounds of guessing: `.last.txt` presence proves a
    clean finish but its ABSENCE proves nothing. The canonical
    codex_companion_spawn.sh has carried an rc receipt since 08-03; this
    dispatcher shipped without one.
    """
    q = shlex.quote
    log = f".omx/tmp/codex_runs/{name}.log"
    done = f".omx/tmp/codex_runs/{name}.done"
    instruction = (
        f"Read and execute the charter at {prompt_path} in full, plus the common "
        f"contract it points to at .omx/tmp/codex_runs/_common_contract.md. "
        f"Follow every constraint in both."
    )
    # codex runs in the BACKGROUND and the wrapper `wait`s on it. This is not
    # cosmetic: bash services traps only BETWEEN commands, so with codex in the
    # foreground a SIGTERM is deferred until codex finishes — i.e. exactly the
    # reap case produces no receipt. MEASURED 2026-08-04 round 2: the fg form
    # wrote rc=0 correctly and dropped the signal receipt entirely. `wait` is
    # interruptible, so the trap fires immediately.
    inner = (
        f"start=$(date +%s); "
        f"codex exec --skip-git-repo-check -s workspace-write "
        f"--add-dir {q(SSD_ADD_DIR)} "
        f"-m gpt-5.5 -c model_reasoning_effort=xhigh "
        f"-o {q(f'.omx/tmp/codex_runs/{name}.last.txt')} "
        f"{q(instruction)} & "
        f"child=$!; "
        f"for s in TERM INT HUP QUIT; do "
        f'trap "kill \\$child 2>/dev/null; '
        f'echo signal=$s elapsed=\\$(( \\$(date +%s) - $start )) > {q(done)}; '
        f'exit 143" $s; '
        f"done; "
        f"wait $child; rc=$?; "
        f"echo rc=$rc elapsed=$(( $(date +%s) - start )) > {q(done)}"
    )
    return " ".join(["python3 -c", q(_DETACH_PY), q(log), "bash", "-c", q(inner)])


def spawn(name: str, prompt_path: str) -> bool:
    RUNS.mkdir(parents=True, exist_ok=True)
    if not (_REPO / prompt_path).exists():
        print(f"  REFUSED {name}: prompt file missing ({prompt_path})", file=sys.stderr)
        return False
    subprocess.run(["bash", "-c", spawn_command(name, prompt_path)], cwd=_REPO, check=False)
    append_row({"name": name, "prompt_path": prompt_path, "status": "live", "event": "spawned"})
    try:
        SPAWN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SPAWN_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": time.time(), "name": name, "prompt": prompt_path}) + "\n")
    except Exception:
        pass
    return True


# --- CLI -------------------------------------------------------------------------


def cmd_status(args) -> int:
    rows = load_rows()
    live = live_arm_names()
    latest = latest_by_name(rows)
    # Report the DENOMINATOR (m50): a charter marked `live` whose process is gone is
    # SPAWNABLE, and listing only status=="queued" hid exactly those — status said
    # "live 0, queued 4" while saturate was about to fire four arms in neither count.
    spawnable = sorted(
        (
            r
            for r in latest.values()
            if r.get("status") in _LIVE_STATUSES and r.get("name") not in live
        ),
        key=lambda r: (r.get("rank", 999), r.get("name", "")),
    )
    n_stale = sum(1 for r in spawnable if r.get("status") == "live")
    scorer_live = any(latest.get(n, {}).get("owns_scorer") for n in live)
    print(f"codex arms live: {len(live)}/{args.cap}  {sorted(live) if live else ''}")
    print(f"scorer slot: {'TAKEN' if scorer_live else 'free'}")
    print(f"spawnable charters: {len(spawnable)} of {len(latest)} tracked", end="")
    print(f"  ({n_stale} marked live but DEAD — resumable)" if n_stale else "")
    for row in spawnable[: args.limit]:
        flag = " [SCORER]" if row.get("owns_scorer") else ""
        stale = " [DIED]" if row.get("status") == "live" else ""
        print(
            f"  rank {row.get('rank', '?'):>3}  {row.get('name')}{flag}{stale}"
            f" — {row.get('note', '')[:80]}"
        )
    gap = max(0, args.cap - len(live))
    print(f"SATURATION GAP: {gap} slot(s) open" if gap else "SATURATED")
    return 0


def cmd_add(args) -> int:
    append_row(
        {
            "name": args.name,
            "prompt_path": args.prompt,
            "rank": args.rank,
            "owns_scorer": bool(args.owns_scorer),
            "status": "queued",
            "note": args.note or "",
        }
    )
    print(f"queued {args.name} (rank {args.rank})")
    return 0


def cmd_mark(args) -> int:
    append_row({"name": args.name, "status": args.status, "event": "mark"})
    print(f"{args.name} -> {args.status}")
    return 0


def cmd_saturate(args) -> int:
    if os.environ.get(KILL_SWITCH) == "1":
        print(f"saturation OFF ({KILL_SWITCH}=1)")
        return 0
    rows = load_rows()
    live = live_arm_names()
    latest = latest_by_name(rows)
    gap = max(0, args.cap - len(live))
    if not gap:
        print(f"SATURATED ({len(live)}/{args.cap}) — nothing to fire")
        return 0
    scorer_taken = any(latest.get(n, {}).get("owns_scorer") for n in live)
    picks = next_charters(rows, live, gap, scorer_taken)
    if not picks:
        print(f"GAP {gap} but QUEUE EMPTY — feed the queue (codex_arm_queue.py add ...)")
        return 0
    for row in picks:
        name, prompt = row.get("name"), row.get("prompt_path", "")
        if not args.spawn:
            print(f"  would spawn: {name} ({prompt})")
            continue
        if spawn(name, prompt):
            print(f"  spawned {name}")
            time.sleep(2)
    if not args.spawn:
        print("dry run — pass --spawn to actually fire")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status"); p.add_argument("--limit", type=int, default=12); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("add")
    p.add_argument("--name", required=True); p.add_argument("--prompt", required=True)
    p.add_argument("--rank", type=int, default=100); p.add_argument("--owns-scorer", action="store_true")
    p.add_argument("--note", default=""); p.set_defaults(fn=cmd_add)
    p = sub.add_parser("mark")
    p.add_argument("--name", required=True)
    p.add_argument("--status", required=True, choices=["queued", "live", "landed", "dropped"])
    p.set_defaults(fn=cmd_mark)
    p = sub.add_parser("saturate"); p.add_argument("--spawn", action="store_true"); p.set_defaults(fn=cmd_saturate)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
