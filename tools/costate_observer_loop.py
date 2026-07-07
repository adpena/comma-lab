#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""costate_observer_loop — periodic short-lived shadow observations for ONE witness run.

Per the CLAUDE.md "'Off' is a tracked queue" NON-NEGOTIABLE ("observability defaults
ON" — read-only, score-neutral telemetry must never depend on a human remembering to
start it) + the 2026-07-07 operator directive that the #247 costate controller is a
CORE SENSE ORGAN needing no manual activation: every governed launch gets this observer
automatically (wired in tools/launch_witness_run.py step (c.2)).

DESIGN (score-neutral by construction; SENSE only, no actuation):
  * Each tick invokes ``tools/costate_shadow_report.py --run-dir X --write`` as a
    SHORT-LIVED subprocess (bounded timeout) — the observer itself is a tiny sleeper,
    per the prefer-periodic-short-lived-invocation-over-long-daemon lesson.
  * The shadow package (tac.witness_control) READS run.log + launch.sh and writes ONLY
    the advisory sidecar ``<run_dir>/costate_shadow.jsonl`` (the trainer never reads
    it -> byte-identity of training is preserved BY CONSTRUCTION). The package has no
    process-control capability (source-scan-tested containment).
  * SELF-TERMINATING: the loop exits when no live trainer owns the run dir (plus a
    grace tick to capture the final state) — it can never become an orphan daemon.
  * run.log fallback: a resumed run sometimes logs to the daemon registry's log path
    instead of <run_dir>/run.log (observed on the mod32cap resume 2026-07-07). If
    run.log is missing, symlink it to the registry log so the observer can SEE the
    telemetry. The symlink is score-neutral (the trainer never reads run.log).

Usage (normally spawned by launch_witness_run via spawn_durable_daemon):
  .venv/bin/python tools/costate_observer_loop.py --run-dir <dir> [--interval-s 900]
  .venv/bin/python tools/costate_observer_loop.py --run-dir <dir> --once   # one tick
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

_REGISTRY = _REPO / ".omx" / "state" / "durable_daemons.json"
_SHADOW_TOOL = _REPO / "tools" / "costate_shadow_report.py"


def _log(msg: str) -> None:
    print(f"[costate-observer] {msg}", flush=True)


def trainer_alive_for(run_dir: Path) -> bool:
    """True iff a live trainer process owns this run dir (reuses the canonical
    check-in's discovery; fail-open False so the loop errs toward exiting)."""
    try:
        import witness_checkin as wc
        run_dir = run_dir.resolve()
        for proc in wc.find_trainer_procs():
            out_dir = proc.get("out_dir")
            if out_dir and Path(out_dir).resolve() == run_dir:
                return True
        return False
    except Exception:
        return False


def ensure_run_log(run_dir: Path) -> str | None:
    """If <run_dir>/run.log is missing, symlink it to the daemon registry's log for
    the job that launched this run dir (score-neutral: only the OBSERVER reads it).
    Returns a note string when a symlink was created, else None."""
    run_log = run_dir / "run.log"
    if run_log.exists() or run_log.is_symlink():
        return None
    try:
        reg = json.loads(_REGISTRY.read_text())
        rows = reg if isinstance(reg, list) else list(reg.values()) if isinstance(reg, dict) else []
        for r in rows:
            if not isinstance(r, dict):
                continue
            cmd = r.get("cmd")
            cmd_str = " ".join(str(t) for t in cmd) if isinstance(cmd, list) else str(cmd or "")
            if str(run_dir) not in cmd_str or r.get("status") != "running":
                continue
            log_path = Path(str(r.get("log") or ""))
            if not log_path.is_absolute():
                log_path = _REPO / log_path
            if log_path.is_file():
                run_log.symlink_to(log_path.resolve())
                return f"symlinked run.log -> {log_path} (registry log; run.log was missing)"
    except Exception:
        return None
    return None


def one_tick(run_dir: Path, timeout_s: float) -> int:
    """One short-lived shadow observation. Returns the subprocess rc (nonzero is
    logged, never fatal — the loop keeps observing)."""
    note = ensure_run_log(run_dir)
    if note:
        _log(note)
    cmd = [sys.executable, str(_SHADOW_TOOL), "--run-dir", str(run_dir), "--write"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        tail = (res.stderr or res.stdout or "").strip().splitlines()
        _log(f"tick rc={res.returncode}" + (f" | {tail[-1][:120]}" if tail else ""))
        return res.returncode
    except subprocess.TimeoutExpired:
        _log(f"tick TIMEOUT after {timeout_s:.0f}s (skipped; will retry next tick)")
        return 124
    except Exception as exc:
        _log(f"tick failed ({type(exc).__name__}: {exc})")
        return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--interval-s", type=float, default=900.0,
                    help="seconds between observations (default 900 = 15 min)")
    ap.add_argument("--tick-timeout-s", type=float, default=120.0,
                    help="per-observation subprocess timeout (default 120s)")
    ap.add_argument("--once", action="store_true", help="one observation, then exit")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        _log(f"ERROR: run dir {run_dir} does not exist")
        return 2
    _log(f"observing {run_dir.name} every {args.interval_s:.0f}s (SENSE-only; "
         f"writes ONLY the advisory costate_shadow.jsonl sidecar; self-terminates "
         f"when the trainer exits)")
    if args.once:
        one_tick(run_dir, args.tick_timeout_s)
        return 0
    while True:
        alive = trainer_alive_for(run_dir)
        one_tick(run_dir, args.tick_timeout_s)
        if not alive:
            _log("no live trainer owns this run dir — final observation captured; exiting.")
            return 0
        time.sleep(args.interval_s)


if __name__ == "__main__":
    sys.exit(main())
