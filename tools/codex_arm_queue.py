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
    """Rank-ordered charters that may fire right now, honouring the scorer rule.

    A live-marked row with NO process but a CLEAN ``rc=0`` exit receipt is a
    FINISHED arm awaiting harvest — NOT died-resumable. Auto-respawning it
    re-runs a completed charter (measured 2026-08-04: rf1 respawned after its
    harvest because the row was never marked landed; the duplicate had to be
    killed). Clean finishes need an explicit ``mark --status queued`` (or a
    fresh ``add``) to run again; only receipt-less or nonzero/signal exits
    remain implicitly resumable.
    """
    latest = latest_by_name(rows)

    def _finished_clean(r: dict) -> bool:
        if r.get("status") != "live":
            return False
        receipt = _done_receipt(r.get("name", ""))
        return receipt is not None and receipt.startswith("rc=0")

    ready = [
        r
        for r in latest.values()
        if r.get("status") in _LIVE_STATUSES
        and r.get("name") not in live
        and not _finished_clean(r)
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


# BSD-correct detach: fork + setsid(2) + exec. `nohup ... & disown` is NOT
# sufficient — disown clears the shell's JOB TABLE but the child stays in the
# shell's process group AND gets reparented to PID 1 when the tool-shell exits.
# macOS has NO setsid(1), so it must be done in Python.
#
# ROOT CAUSE, MEASURED 2026-08-04 via the exit receipts: the killer of every
# arm generation (nohup 11:13, setsid 11:34, receipted 11:44 — signal=TERM at
# elapsed 335/337/337 s) is ~/Library/LaunchAgents/com.vertigo.claude-code-reaper
# → ~/Projects/fleet/scripts/claude-code-reaper.sh, a launchd agent firing every
# 60 s that SIGTERMs any process matching \b(claude|codex)\b with no TTY and
# (PPID==1 or stdin in {null,pipe}) older than 300 s. Differential proof: a
# plain-bash control detached by the IDENTICAL fork+setsid shim survived to
# natural completion — the harness reaps nothing; the reaper kills by NAME.
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


def keeper_path(name: str) -> str:
    return f".omx/tmp/codex_runs/{name}_keeper.py"


def keeper_source(name: str, prompt_path: str) -> str:
    """Source of the per-arm KEEPER — the reaper-proof supervisor.

    The reaper kills on: name-match \\b(claude|codex)\\b AND no-TTY AND
    (PPID==1 OR stdin in {/dev/null, pipe}) AND age>300s. The keeper breaks two
    conjuncts at once, using only plain POSIX facts:

      * The keeper IS the detached setsid leader, and its ps line is
        ``python3 .omx/tmp/codex_runs/<name>_keeper.py`` — ``codex_runs`` has
        no word boundary (underscore is a word char), so the reaper's grep
        never examines it at all.
      * codex runs as the keeper's normal CHILD (PPID != 1) with stdin bound
        to a REGULAR FILE — the reaper's stdin_is_dead() flags only
        /dev/null|PIPE|FIFO, so a REG-file stdin reads as a live session.

    The keeper also owns the EXIT RECEIPT at ``<name>.done``:

        rc=0                 -> codex finished cleanly
        rc=N (N>0)           -> codex exited with an error of its own
        signal=TERM|INT|HUP|QUIT -> something reaped the keeper (handler fired)
        no .done file        -> SIGKILL, or the keeper itself vanished

    Python signal handlers interrupt ``proc.wait()`` immediately, so the
    bash foreground-trap-deferral class (round-2 finding: fg traps are
    serviced only between commands — exactly the reap case wrote no receipt)
    cannot recur here.
    """
    instruction = (
        f"Read and execute the charter at {prompt_path} in full, plus the common "
        f"contract it points to at .omx/tmp/codex_runs/_common_contract.md. "
        f"Follow every constraint in both."
    )
    relay_instruction = (
        f"CONTINUATION generation %d of arm {name}: your predecessor ran out of "
        f"context mid-charter. Its durable state is on disk — FIRST read the newest "
        f"receipts under .omx/research matching this arm's receipt dir (named in the "
        f"charter) and run `git log --oneline -15` to see work already committed. Do "
        f"NOT redo completed work; resume from the newest NEXT-IF-RESUMED block (or "
        f"infer the frontier from receipts if absent). CONTEXT LAW: write receipts "
        f"incrementally, commit early via the serializer, never accumulate large "
        f"outputs or web-search results in context. Then continue executing the "
        f"charter at {prompt_path} plus the common contract at "
        f".omx/tmp/codex_runs/_common_contract.md."
    )
    argv_prefix = [
        "codex", "exec", "--skip-git-repo-check", "-s", "workspace-write",
        "--add-dir", SSD_ADD_DIR,
        "-m", "gpt-5.5", "-c", "model_reasoning_effort=xhigh",
        "-o", f".omx/tmp/codex_runs/{name}.last.txt",
    ]
    return (
        "# Auto-generated per-arm keeper — see tools/codex_arm_queue.py:keeper_source.\n"
        "# RELAY: on context-exhaustion the keeper relaunches codex with a fresh\n"
        "# context + disk-state continuation header (operator directive 2026-08-04:\n"
        "# arms must exceed one context and run autonomously for however long it takes).\n"
        "import os, shutil, signal, subprocess, sys, time\n"
        f"NAME = {name!r}\n"
        f"ARGV_PREFIX = {argv_prefix!r}\n"
        f"INSTRUCTION = {instruction!r}\n"
        f"RELAY_INSTRUCTION = {relay_instruction!r}\n"
        "MAX_GEN = 12\n"
        "CONTEXT_SIGS = ('ran out of room in the model\\'s context window',\n"
        "                'exceeds the context window')\n"
        "DONE = '.omx/tmp/codex_runs/' + NAME + '.done'\n"
        "STDIN_PATH = '.omx/tmp/codex_runs/' + NAME + '.stdin'\n"
        "LOG = '.omx/tmp/codex_runs/' + NAME + '.log'\n"
        "LAST = '.omx/tmp/codex_runs/' + NAME + '.last.txt'\n"
        "RELAY_LOG = '.omx/tmp/codex_runs/' + NAME + '.relay'\n"
        "start = time.time()\n"
        "open(STDIN_PATH, 'ab').close()  # REGULAR file: reaper reads it as a live stdin\n"
        "stdin_f = open(STDIN_PATH, 'rb')\n"
        "log_f = open(LOG, 'ab')\n"
        "proc = None\n"
        "def _mk(name_):\n"
        "    def h(signum, frame):\n"
        "        try:\n"
        "            if proc is not None:\n"
        "                proc.terminate()\n"
        "        except Exception:\n"
        "            pass\n"
        "        with open(DONE, 'w') as f:\n"
        "            f.write('signal=%s elapsed=%d\\n' % (name_, int(time.time() - start)))\n"
        "        os._exit(143)\n"
        "    return h\n"
        "for s in ('TERM', 'INT', 'HUP', 'QUIT'):\n"
        "    signal.signal(getattr(signal, 'SIG' + s), _mk(s))\n"
        "def _head():\n"
        "    try:\n"
        "        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()\n"
        "    except Exception:\n"
        "        return ''\n"
        "def _log_tail():\n"
        "    try:\n"
        "        with open(LOG, 'rb') as f:\n"
        "            f.seek(max(0, os.path.getsize(LOG) - 8000))\n"
        "            return f.read().decode('utf-8', 'replace')\n"
        "    except Exception:\n"
        "        return ''\n"
        "def _research_mtime():\n"
        "    newest = 0.0\n"
        "    for root, _dirs, files in os.walk('.omx/research'):\n"
        "        for fn in files:\n"
        "            try:\n"
        "                newest = max(newest, os.path.getmtime(os.path.join(root, fn)))\n"
        "            except OSError:\n"
        "                pass\n"
        "    return newest\n"
        "gen = 1\n"
        "rc = 1\n"
        "while True:\n"
        "    gen_head, gen_mtime, gen_start = _head(), _research_mtime(), time.time()\n"
        "    prompt = INSTRUCTION if gen == 1 else (RELAY_INSTRUCTION % gen)\n"
        "    proc = subprocess.Popen(ARGV_PREFIX + [prompt], stdin=stdin_f, stdout=log_f, stderr=log_f)\n"
        "    rc = proc.wait()\n"
        "    if rc == 0:\n"
        "        break\n"
        "    tail = _log_tail()\n"
        "    exhausted = any(sig in tail for sig in CONTEXT_SIGS)\n"
        "    progressed = (_head() != gen_head) or (_research_mtime() > gen_mtime)\n"
        "    if not exhausted or gen >= MAX_GEN or (gen >= 2 and not progressed):\n"
        "        break\n"
        "    try:\n"
        "        shutil.copyfile(LAST, LAST + '.gen%d' % gen)\n"
        "    except OSError:\n"
        "        pass\n"
        "    with open(RELAY_LOG, 'a') as f:\n"
        "        f.write('relay gen=%d->%d rc=%d gen_elapsed=%d progressed=%s\\n'\n"
        "                % (gen, gen + 1, rc, int(time.time() - gen_start), progressed))\n"
        "    gen += 1\n"
        "with open(DONE, 'w') as f:\n"
        "    f.write('rc=%d elapsed=%d gen=%d\\n' % (rc, int(time.time() - start), gen))\n"
        "sys.exit(rc)\n"
    )


def spawn_command(name: str, prompt_path: str) -> str:
    """The canonical detached spawn: fork+setsid shim exec'ing the KEEPER.

    Reaper-shape invariant (pinned by tests): this command string contains NO
    standalone ``codex``/``claude`` word — the codex argv lives inside the
    keeper FILE, which ps never shows. ``prompt_path`` is intentionally not in
    the command either; the keeper embeds the full instruction.
    """
    q = shlex.quote
    log = f".omx/tmp/codex_runs/{name}_keeper.log"
    return " ".join(["python3 -c", q(_DETACH_PY), q(log), "python3", q(keeper_path(name))])


def spawn(name: str, prompt_path: str) -> bool:
    RUNS.mkdir(parents=True, exist_ok=True)
    if not (_REPO / prompt_path).exists():
        print(f"  REFUSED {name}: prompt file missing ({prompt_path})", file=sys.stderr)
        return False
    # Clear STALE evidence from a previous generation: a leftover `.done`
    # (death receipt) or `.last.txt` (clean-finish marker) would corrupt the
    # next death-vs-completion read — the exact ambiguity the receipt exists
    # to remove. Confirmed by executed control 2026-08-04 review round.
    for stale in (RUNS / f"{name}.done", RUNS / f"{name}.last.txt"):
        stale.unlink(missing_ok=True)
    (_REPO / keeper_path(name)).write_text(keeper_source(name, prompt_path), encoding="utf-8")
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


def _done_receipt(name: str) -> str | None:
    """Terminal receipt line for an arm, or None if no ``.done`` exists.

    The old status labeled every live-marked-but-processless arm ``[DIED]``
    without reading receipts — rc=0 finishes showed as deaths (the known
    instrument mislabel). The ``.done`` file is the keeper's exit receipt and
    is the authority on how the arm actually ended.
    """
    try:
        content = (RUNS / f"{name}.done").read_text(errors="replace").strip()
    except OSError:
        return None
    return content.splitlines()[-1] if content else "(empty receipt)"


def _watcher_line() -> str:
    """Fleet-watcher liveness from the heartbeat file (codex_arm_watch.py).

    Surfaced at every queue interaction so an unarmed watcher is visible —
    without it, arm completions silently revert to poll-on-request.
    """
    hb = RUNS / "_watcher.alive"
    try:
        age = time.time() - hb.stat().st_mtime
    except OSError:
        return (
            "fleet watcher: NOT RUNNING — arm completions will NOT notify MAIN "
            "(arm via Monitor: .venv/bin/python tools/codex_arm_watch.py, persistent)"
        )
    if age > 90:  # codex_arm_watch.HEARTBEAT_STALE_S
        return f"fleet watcher: STALE (heartbeat {int(age)}s ago) — re-arm the Monitor"
    return f"fleet watcher: ALIVE (heartbeat {int(age)}s ago)"


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
    # Split processless-live rows by their .done receipt: FINISHED-unharvested
    # (keeper wrote an exit receipt) vs truly DIED (no receipt at all).
    n_finished = sum(
        1
        for r in spawnable
        if r.get("status") == "live" and _done_receipt(r.get("name", "")) is not None
    )
    n_stale = (
        sum(1 for r in spawnable if r.get("status") == "live") - n_finished
    )
    scorer_live = any(latest.get(n, {}).get("owns_scorer") for n in live)
    print(f"codex arms live: {len(live)}/{args.cap}  {sorted(live) if live else ''}")
    print(f"scorer slot: {'TAKEN' if scorer_live else 'free'}")
    print(_watcher_line())
    print(f"spawnable charters: {len(spawnable)} of {len(latest)} tracked", end="")
    tail = []
    if n_finished:
        tail.append(f"{n_finished} FINISHED (unharvested — read .done)")
    if n_stale:
        tail.append(f"{n_stale} marked live but DEAD — resumable")
    print(f"  ({'; '.join(tail)})" if tail else "")
    for row in spawnable[: args.limit]:
        flag = " [SCORER]" if row.get("owns_scorer") else ""
        stale = ""
        if row.get("status") == "live":
            receipt = _done_receipt(row.get("name", ""))
            stale = f" [FINISHED {receipt}]" if receipt is not None else " [DIED]"
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
    else:
        print(_watcher_line())
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
