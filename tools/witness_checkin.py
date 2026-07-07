#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Canonical one-shot witness-run status check-in (#338 harness wrapper).

READ-ONLY, score-neutral, no gating. Autodiscovers the current level-set
witness run (prefers the run dir owned by a LIVE trainer process, found via
psutil cmdline ``--out-dir``; falls back to the newest
``experiments/results/levelset_n600_witness_*`` dir by artifact mtime) and
reports one HONEST line:

  * alive/dead + pid + RSS GiB of the trainer process
  * best d_seg + epoch (from ``levelset_best.json``: keys d_seg/epoch/path/ts —
    schema verified against the live mod32cap run 2026-07-06)
  * latest-signal freshness (newest of best / ``levelset_resume_state.npz`` /
    ``costate_shadow.jsonl`` mtimes) — an ALIVE pid with no file update in
    >30 min is flagged STALE (never imply progress that is not evidenced)
  * system available-memory headline

Ages are computed from file mtimes vs ``time.time()`` (both epoch seconds on
the same clock) — NOT by parsing timestamp strings, so there is no timezone
parsing to get wrong. ``--json`` emits the full machine-readable dict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
RESULTS_DEFAULT = _REPO / "experiments" / "results"
RUN_GLOB = "levelset_n600_witness_*"
TRAINER_TOKEN = "train_levelset_witness_realized_through_R_mlx"
STALE_AFTER_S_DEFAULT = 30 * 60.0

# Signal files inside a run dir, in reporting order.
BEST_JSON = "levelset_best.json"
RESUME_NPZ = "levelset_resume_state.npz"
COSTATE_JSONL = "costate_shadow.jsonl"


def find_trainer_procs() -> list[dict]:
    """All live trainer processes: [{pid, out_dir|None, rss_bytes}]. Never raises."""
    try:
        import psutil
    except ImportError:
        return []
    found: list[dict] = []
    me = os.getpid()
    for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
        try:
            if proc.pid == me:
                continue
            cmdline = proc.info.get("cmdline") or []
            if not cmdline or "python" not in os.path.basename(cmdline[0]):
                continue  # only python interpreters (not editors/greps of the file)
            if not any(TRAINER_TOKEN in tok for tok in cmdline):
                continue
            out_dir = None
            if "--out-dir" in cmdline:
                idx = cmdline.index("--out-dir")
                if idx + 1 < len(cmdline):
                    out_dir = cmdline[idx + 1]
            found.append(
                {"pid": proc.pid, "out_dir": out_dir, "rss_bytes": proc.memory_info().rss}
            )
        except Exception:
            continue  # racing process exit / permission — skip, read-only tool
    return found


def _signal_mtime(run_dir: Path) -> float:
    """Newest mtime among the run dir's signal files (fallback: dir mtime)."""
    mtimes = []
    for name in (BEST_JSON, RESUME_NPZ, COSTATE_JSONL):
        p = run_dir / name
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    if not mtimes:
        mtimes.append(run_dir.stat().st_mtime)
    return max(mtimes)


def newest_run_dir(results_dir: Path) -> Path | None:
    """Newest run dir matching RUN_GLOB by signal-file mtime; None if none exist."""
    candidates = [d for d in results_dir.glob(RUN_GLOB) if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=_signal_mtime)


def pick_run_dir(procs: list[dict], results_dir: Path) -> tuple[Path | None, dict | None, str]:
    """(run_dir, owning_proc|None, how). Prefers a live trainer's --out-dir."""
    with_dir = []
    for proc in procs:
        out_dir = proc.get("out_dir")
        if out_dir and Path(out_dir).is_dir():
            with_dir.append((Path(out_dir), proc))
    if with_dir:
        run_dir, proc = max(with_dir, key=lambda pair: _signal_mtime(pair[0]))
        return run_dir, proc, "live-trainer --out-dir"
    fallback = newest_run_dir(results_dir)
    # A trainer may be alive with an unresolvable out-dir; keep the pid, note it.
    proc = procs[0] if procs else None
    how = "newest dir by mtime" + (" (pid alive, out-dir unresolved)" if proc else "")
    return fallback, proc, how


def _age_s(path: Path, now: float) -> float | None:
    if not path.exists():
        return None
    return max(0.0, now - path.stat().st_mtime)


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "missing"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _last_jsonl_row(path: Path) -> dict | None:
    try:
        last = None
        with path.open("rb") as fh:
            for raw in fh:
                if raw.strip():
                    last = raw
        return json.loads(last) if last else None
    except Exception:
        return None


def collect_status(run_dir: Path, proc: dict | None, stale_after_s: float) -> dict:
    now = time.time()
    status: dict = {
        "run_dir": str(run_dir),
        "alive": proc is not None,
        "pid": proc["pid"] if proc else None,
        "rss_gib": round(proc["rss_bytes"] / 2**30, 2) if proc else None,
        "warnings": [],
    }
    best_path = run_dir / BEST_JSON
    best = None
    if best_path.exists():
        try:
            best = json.loads(best_path.read_text())
        except Exception:
            status["warnings"].append(f"unreadable {BEST_JSON}")
    status["best"] = best  # keys (verified live): d_seg / epoch / path / ts
    status["best_age_s"] = _age_s(best_path, now)
    status["resume_age_s"] = _age_s(run_dir / RESUME_NPZ, now)
    costate_path = run_dir / COSTATE_JSONL
    status["telemetry_age_s"] = _age_s(costate_path, now)
    row = _last_jsonl_row(costate_path) if costate_path.exists() else None
    status["telemetry_epoch"] = row.get("epoch") if isinstance(row, dict) else None

    ages = [a for a in (status["best_age_s"], status["resume_age_s"], status["telemetry_age_s"]) if a is not None]
    status["freshest_signal_age_s"] = min(ages) if ages else None

    try:
        import psutil

        status["free_mem_gib"] = round(psutil.virtual_memory().available / 2**30, 1)
    except Exception:
        status["free_mem_gib"] = None

    if not status["alive"]:
        status["warnings"].append("DEAD: no live trainer process")
    elif status["freshest_signal_age_s"] is None:
        status["warnings"].append("STALE: alive pid but NO signal files in run dir")
    elif status["freshest_signal_age_s"] > stale_after_s:
        status["warnings"].append(
            f"STALE: alive pid but newest file update {_fmt_age(status['freshest_signal_age_s'])} ago"
        )
    return status


def human_line(status: dict) -> str:
    parts = []
    parts.append("ALIVE" if status["alive"] else "DEAD")
    if status["alive"]:
        parts.append(f"pid={status['pid']} rss={status['rss_gib']}GiB")
    parts.append(f"run={Path(status['run_dir']).name}")
    best = status.get("best")
    if best:
        d_seg = best.get("d_seg")
        d_seg_txt = f"{d_seg:.7f}" if isinstance(d_seg, (int, float)) else repr(d_seg)
        parts.append(
            f"best d_seg={d_seg_txt} @ep{best.get('epoch')} ({_fmt_age(status['best_age_s'])} ago)"
        )
    else:
        parts.append("best=NONE (no levelset_best.json yet)")
    parts.append(f"resume {_fmt_age(status['resume_age_s'])} ago")
    tele = f"telemetry {_fmt_age(status['telemetry_age_s'])} ago"
    if status.get("telemetry_epoch") is not None:
        tele += f" (ep{status['telemetry_epoch']})"
    parts.append(tele)
    if status.get("free_mem_gib") is not None:
        parts.append(f"free {status['free_mem_gib']}GiB")
    line = "[witness-checkin] " + " | ".join(parts)
    for warning in status["warnings"]:
        line += f"  ⚠ {warning}"
    return line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, default=None, help="explicit run dir (skips autodiscovery)")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    ap.add_argument("--stale-after-min", type=float, default=STALE_AFTER_S_DEFAULT / 60.0)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    procs = find_trainer_procs()
    if args.run_dir is not None:
        run_dir = args.run_dir
        proc = next(
            (p for p in procs if p.get("out_dir") and Path(p["out_dir"]).resolve() == run_dir.resolve()),
            None,
        )
        how = "explicit --run-dir"
    else:
        run_dir, proc, how = pick_run_dir(procs, args.results_dir)
    if run_dir is None or not run_dir.is_dir():
        msg = f"[witness-checkin] NO RUN FOUND under {args.results_dir}/{RUN_GLOB}"
        print(json.dumps({"error": msg}) if args.json else msg)
        return 1

    status = collect_status(run_dir, proc, args.stale_after_min * 60.0)
    status["discovery"] = how
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(human_line(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
