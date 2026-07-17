#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Witness-chain liveness watchdog (B4, p0_launcher_chain_durability_20260717).

THE FAILURE CLASS THIS CLOSES (empirical anchor 2026-07-16/17, run
20260716T211713Z): a dry-start/bench/launch chain judged DEAD from the wrong
signals — a block-buffered daemon log frozen for hours + a mis-fired
``ps | grep`` + a registry row that always says "running" — while the chain was
in fact ALIVE mid-pass (trainer at 100% CPU, 44 GiB). The same signals also
cannot distinguish that from the REAL silent-death class (sandbox teardown /
SIGKILL, spawn_durable_daemon.py docstring), which leaves no receipt. Verdicts
here come ONLY from composite kernel-truth signals, per memory
``launcher_buffered_log_not_hung_orphan_spawn_respawn_id_collision_20260715``
("judge by FILE mtimes; psutil descendants FIRST"):

  pid alive (os.kill 0)  x  descendant tree (ps ppid walk, cross-session)
  x  newest run-dir file mtime  x  receipt presence (dry_start_report.json)

Verdicts per registry row (durable_daemons.json, status == "running"):
  RUNNING_HEALTHY        pid tree alive AND run-dir mtime fresh (< --stale-s)
  RUNNING_QUIET          pid tree alive, mtimes stale -- LOUDLY REPORTED ALIVE
                         (the buffered-log phantom-death case; do NOT re-diagnose
                         death from a quiet log while this verdict shows)
  CHAIN_DEAD_RECEIPTED   tree dead, receipt exists (normal/failed exit -- reconcile)
  CHAIN_DEAD_NO_RECEIPT  tree dead, NO receipt  ->  THE ALARM (rc=2); the silent-
                         death class recurred -- postmortem + relaunch decision
  NO_RUN_DIR             tree dead, no run dir resolvable from the row (label-only)

Single-shot + cron-able (a 15-min cron gives the "flagged within ~15 min"
contract). Appends one JSONL row per scan to
``.omx/state/witness_chain_watchdog.jsonl`` (fcntl-append) so the costate
digest / dashboard can surface the latest verdicts. Read-only + score-neutral:
kills nothing, writes nothing into run dirs.

Usage:
    .venv/bin/python tools/witness_chain_watchdog.py            # scan + table
    .venv/bin/python tools/witness_chain_watchdog.py --json     # machine-readable
    .venv/bin/python tools/witness_chain_watchdog.py --stale-s 900
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_REGISTRY = _REPO / ".omx" / "state" / "durable_daemons.json"
_OUT_JSONL = _REPO / ".omx" / "state" / "witness_chain_watchdog.jsonl"

# Labels the watchdog considers witness-chain custody (bench/dry-start/launch chains +
# their inner safe_run rows). Everything else in the registry is out of scope.
_LABEL_RE = re.compile(r"drystart|dry_start|witness|levelset", re.IGNORECASE)


def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (ValueError, TypeError, OverflowError):
        return False


def _ps_children_map() -> dict[int, list[int]]:
    """ppid -> [pids] from one ``ps`` snapshot (cross-session; no psutil dependency)."""
    out = subprocess.run(["ps", "ax", "-o", "pid=,ppid="], capture_output=True,
                         text=True, timeout=10).stdout
    kids: dict[int, list[int]] = {}
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        kids.setdefault(ppid, []).append(pid)
    return kids


def _descendants(pid: int, kids: dict[int, list[int]]) -> list[int]:
    seen: list[int] = []
    stack = [int(pid)]
    while stack:
        p = stack.pop()
        for c in kids.get(p, ()):
            if c not in seen:
                seen.append(c)
                stack.append(c)
    return seen


def _run_dir_from_row(row: dict) -> Path | None:
    """Resolve the chain's run dir from any cmd token / log path mentioning
    experiments/results/<run>."""
    tokens = list(row.get("cmd") or [])
    if row.get("log"):
        tokens.append(str(row["log"]))
    candidates: list[Path] = []
    for tok in tokens:
        for m in re.finditer(r"((?:/[^\s]*?)?experiments/results/[^\s/]+)", str(tok)):
            frag = m.group(1)
            p = Path(frag) if frag.startswith("/") else _REPO / frag
            if p.is_dir() and p not in candidates:
                candidates.append(p)
    # Prefer a dir that IS a launch run dir (carries launch artifacts) — a cmd can also
    # mention e.g. the gt-cache dir under experiments/results/, which is NOT the chain's
    # run dir (live-fire finding: the outer row matched mlx_fleet_gt_cache first).
    for p in candidates:
        if (p / "launch_manifest.json").exists() or (p / "launch.sh").exists():
            return p
    return None  # a token-only fuzzy match (e.g. the gt-cache dir) is NOT the run dir


def _newest_mtime(root: Path, max_files: int = 4000) -> float | None:
    newest: float | None = None
    n = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            n += 1
            if n > max_files:
                return newest
            try:
                m = os.stat(os.path.join(dirpath, fn)).st_mtime
            except OSError:
                continue
            if newest is None or m > newest:
                newest = m
    return newest


def scan(stale_s: float = 900.0) -> list[dict]:
    try:
        rows = json.loads(_REGISTRY.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return [{"verdict": "REGISTRY_UNREADABLE", "ts": _utc()}]
    if not isinstance(rows, list):
        return [{"verdict": "REGISTRY_UNREADABLE", "ts": _utc()}]
    kids = _ps_children_map()
    now = _dt.datetime.now().timestamp()
    verdicts: list[dict] = []
    for row in rows:
        if not (isinstance(row, dict) and row.get("status") == "running"):
            continue
        label = str(row.get("label") or "")
        if not _LABEL_RE.search(label):
            continue
        pid = row.get("pid")
        alive = _pid_alive(pid) if pid else False
        desc = _descendants(pid, kids) if alive else []
        run_dir = _run_dir_from_row(row)
        receipt = bool(run_dir and (run_dir / "dry_start_report.json").exists())
        newest = _newest_mtime(run_dir) if run_dir else None
        age_s = round(now - newest, 1) if newest else None
        if alive:
            fresh = age_s is not None and age_s < stale_s
            verdict = "RUNNING_HEALTHY" if fresh else "RUNNING_QUIET"
        elif run_dir is None:
            verdict = "NO_RUN_DIR"
        elif receipt:
            verdict = "CHAIN_DEAD_RECEIPTED"
        else:
            verdict = "CHAIN_DEAD_NO_RECEIPT"
        verdicts.append({
            "ts": _utc(), "label": label, "pid": pid, "alive": alive,
            "descendants": desc[:12], "n_descendants": len(desc),
            "run_dir": str(run_dir) if run_dir else None,
            "newest_file_age_s": age_s, "receipt_exists": receipt,
            "verdict": verdict,
            "note": ("ALIVE despite quiet logs/mtimes — judge by THIS row, not the buffered "
                     "daemon log (the 20260716 phantom-death class)"
                     if verdict == "RUNNING_QUIET" else
                     "SILENT DEATH: chain gone with no receipt — postmortem before relaunch"
                     if verdict == "CHAIN_DEAD_NO_RECEIPT" else ""),
        })
    return verdicts


def _append_jsonl(verdicts: list[dict]) -> None:
    try:
        _OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with _OUT_JSONL.open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(json.dumps({"schema": "witness_chain_watchdog.v1", "ts": _utc(),
                                 "verdicts": verdicts}) + "\n")
            fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError as exc:
        print(f"witness_chain_watchdog: WARN jsonl append failed: {exc!r}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stale-s", type=float, default=900.0,
                    help="mtime freshness threshold (s) separating HEALTHY from QUIET (default 900)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--no-append", action="store_true", help="skip the JSONL sidecar append")
    a = ap.parse_args(argv)
    verdicts = scan(stale_s=a.stale_s)
    if not a.no_append:
        _append_jsonl(verdicts)
    if a.json:
        print(json.dumps(verdicts, indent=2))
    else:
        if not verdicts:
            print("witness_chain_watchdog: no running witness-chain rows in the registry.")
        for v in verdicts:
            print(f"{v['verdict']:22s} label={v.get('label')} pid={v.get('pid')} "
                  f"alive={v.get('alive')} desc={v.get('n_descendants')} "
                  f"age_s={v.get('newest_file_age_s')} receipt={v.get('receipt_exists')}"
                  + (f"  <- {v['note']}" if v.get("note") else ""))
    return 2 if any(v.get("verdict") == "CHAIN_DEAD_NO_RECEIPT" for v in verdicts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
