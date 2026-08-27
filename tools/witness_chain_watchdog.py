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
sys.path.insert(0, str(_REPO / "src"))

from tac import process_liveness  # noqa: E402  (needs the sys.path bootstrap above)

_REGISTRY = _REPO / ".omx" / "state" / "durable_daemons.json"
_OUT_JSONL = _REPO / ".omx" / "state" / "witness_chain_watchdog.jsonl"

# Labels the watchdog considers witness-chain custody (bench/dry-start/launch chains +
# their inner safe_run rows). Everything else in the registry is out of scope.
_LABEL_RE = re.compile(r"drystart|dry_start|witness|levelset", re.IGNORECASE)


def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pid_alive(pid: int) -> bool:
    """Delegates to the canonical tri-state read (``tac.process_liveness``).

    PRESERVES this site's deliberate extra: ``pid`` arrives from a JSON
    registry row, so it may be a string or junk -- the ``int()`` coercion and
    its guard stay here, ahead of the canonical call.

    This site already agreed on ``PermissionError`` -> alive.  CHANGED: a
    zombie was ALIVE forever and is now DEAD, which is the point of the
    surrounding ``_pid_alive_cmd`` cross-check -- an exited chain must not read
    as running.  ``_pid_alive_cmd`` itself is UNTOUCHED: its pid-REUSE
    cross-check against the live command line is a real protection that
    canonical liveness does not replace.
    """
    try:
        pid_int = int(pid)
    except (ValueError, TypeError, OverflowError):
        return False
    return process_liveness.pid_state(pid_int) == process_liveness.ALIVE


def _live_cmdline(pid: int) -> str | None:
    """The live process's command line via ps -p (kernel truth; empty/dead -> None)."""
    try:
        out = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="],  # subprocess-no-check-OK: best-effort ps probe; empty/dead reads None via the except arm
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _expected_tokens(row_cmd: list) -> list[str]:
    """Distinctive basenames from the registered argv (for the pid-reuse cross-check)."""
    toks: list[str] = []
    for tok in row_cmd or []:
        base = str(tok).rstrip("/").split("/")[-1]
        if base.endswith((".py", ".sh")) and base not in toks:
            toks.append(base)
    return toks


_CHAIN_TOKENS = ("safe_run.py", "launch_witness_run.py", "train_levelset", "launch.sh",
                 "spawn_durable_daemon.py")


def _pid_alive_cmd(pid, expect_tokens: list[str]) -> tuple[bool, str | None]:
    """F3b (independent review 2026-07-17): bare kill(pid,0) is wrong twice — pid REUSE makes
    a dead chain read alive forever, and PermissionError->True can mask death. Cross-check
    the LIVE command line against the registered argv's distinctive tokens; no match => the
    pid now belongs to someone else => DEAD for this chain's purposes."""
    if not pid or not _pid_alive(pid):
        return False, None
    live = _live_cmdline(pid)
    if live is None:
        return False, None
    if expect_tokens and not any(tok in live for tok in expect_tokens):
        # exec-chains legitimately REPLACE the registered argv (the v3 waiter bash execs
        # safe_run -> launcher): accept any known chain token as fallback before declaring
        # pid-reuse. An UNRELATED process matches neither set => DEAD for this chain.
        if not any(tok in live for tok in _CHAIN_TOKENS):
            return False, live  # pid reused by an unrelated process
    return True, live


def _ps_children_map() -> dict[int, list[int]]:
    """ppid -> [pids] from one ``ps`` snapshot (cross-session; no psutil dependency)."""
    out = subprocess.run(["ps", "ax", "-o", "pid=,ppid="], capture_output=True,  # subprocess-no-check-OK: best-effort ps snapshot; empty output yields no children map
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
    # Keep only dirs that ARE launch run dirs (carry launch artifacts) — a cmd can also
    # mention e.g. the gt-cache dir under experiments/results/, which is NOT the chain's
    # run dir (live-fire finding: the outer row matched mlx_fleet_gt_cache first).
    artifact_dirs = [p for p in candidates
                     if (p / "launch_manifest.json").exists() or (p / "launch.sh").exists()]
    # F3c (independent review 2026-07-17): a v3-waiter-style argv carries PRIOR run dirs
    # (--dry-start-delta-from / --observer-cost-evidence) that DO have launch artifacts —
    # and green receipts — so picking any of them turns a silent death into a false
    # CHAIN_DEAD_RECEIPTED. Multiple candidates = ambiguous = None; the chain MANIFEST
    # (written by the launcher, which knows its out_dir) is the authoritative source.
    if len(artifact_dirs) == 1:
        return artifact_dirs[0]
    return None


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


_MANIFEST = _REPO / ".omx" / "state" / "witness_chain_manifest.jsonl"
_MANIFEST_MAX_AGE_DAYS = 7.0


def _manifest_rows(manifest_path: Path | None = None) -> list[dict]:
    """LAST manifest row per out_dir within the recency window (launcher self-registration,
    F3a): {launcher_pid, out_dir, config, label, ts}. Missing/unreadable -> []."""
    _env_path = os.environ.get("TAC_CHAIN_MANIFEST_PATH")
    if manifest_path is None and not _env_path and os.environ.get("PYTEST_CURRENT_TEST"):
        return []  # hermetic: never read the LIVE manifest from a test (writer parity)
    path = manifest_path or (Path(_env_path) if _env_path else _MANIFEST)
    rows: dict[str, dict] = {}
    try:
        text = path.read_text()
    except OSError:
        return []
    cutoff = _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=_MANIFEST_MAX_AGE_DAYS)
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            d = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if not (isinstance(d, dict) and d.get("out_dir") and d.get("launcher_pid")):
            continue
        try:
            ts = _dt.datetime.strptime(str(d.get("ts")), "%Y%m%dT%H%M%SZ").replace(tzinfo=_dt.UTC)
            if ts < cutoff:
                continue
        except (ValueError, TypeError):
            pass  # unparseable ts: keep (fail-open for the alarm's sake)
        rows[str(d["out_dir"])] = d
    return list(rows.values())


def scan(stale_s: float = 900.0, registry_path: Path | None = None,
         manifest_path: Path | None = None) -> list[dict]:
    try:
        rows = json.loads((registry_path or _REGISTRY).read_text())
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
        alive, live_cmd = _pid_alive_cmd(pid, _expected_tokens(row.get("cmd") or []))
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
    # F3a: chain-manifest verdicts (launcher self-registration — covers chains whose registry
    # rows cannot resolve a run dir, e.g. the v3-waiter outer chain; a silent LAUNCHER death
    # must alarm as CHAIN_DEAD_NO_RECEIPT, never NO_RUN_DIR rc 0).
    for man in _manifest_rows(manifest_path):
        out_dir = Path(str(man["out_dir"]))
        pid = man.get("launcher_pid")
        alive, live_cmd = _pid_alive_cmd(pid, ["launch_witness_run.py"])
        out_dir_exists = out_dir.is_dir()
        receipt = out_dir_exists and (out_dir / "dry_start_report.json").exists()
        newest = _newest_mtime(out_dir) if out_dir_exists else None
        age_s = round(now - newest, 1) if newest else None
        if alive:
            fresh = age_s is not None and age_s < stale_s
            verdict = "RUNNING_HEALTHY" if fresh else "RUNNING_QUIET"
        elif not out_dir_exists:
            # D1b (independent review 2026-07-17): the self-registered out_dir was DELETED
            # (a GC'd scratch proof, or an operator cleanup) — there is nothing to alarm ON.
            # A missing dir is NOT a silent death (no run artifacts => never had a real chain
            # here to die); read NO_RUN_DIR, never CHAIN_DEAD_NO_RECEIPT.
            verdict = "NO_RUN_DIR"
        elif receipt:
            verdict = "CHAIN_DEAD_RECEIPTED"
        else:
            verdict = "CHAIN_DEAD_NO_RECEIPT"
        verdicts.append({
            "ts": _utc(), "source": "manifest", "label": man.get("label"),
            "pid": pid, "alive": alive, "run_dir": str(out_dir),
            "newest_file_age_s": age_s, "receipt_exists": receipt, "verdict": verdict,
            "note": ("SILENT DEATH: launcher gone, no receipt in its self-registered out_dir "
                     "— postmortem before relaunch" if verdict == "CHAIN_DEAD_NO_RECEIPT"
                     else "ALIVE despite quiet logs/mtimes — the phantom-death class"
                     if verdict == "RUNNING_QUIET" else ""),
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
