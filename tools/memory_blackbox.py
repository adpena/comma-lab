#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Memory BLACK-BOX recorder + dynamic governor daemon — so a future crash yields SIGNAL, not nothing.

WHY (operator P0, 2026-07-02): the machine crashed from system-wide memory exhaustion and we had NO
memory trajectory logged before the runs died — no black box. This is the always-on lightweight
sampler that records the SYSTEM memory trajectory to a durable, rotated JSONL so ``--last-crash`` can
replay exactly what led into the most recent crash/reboot. It DOUBLES AS THE GOVERNOR: each sample it
evaluates memory pressure and, when sustained WARN/CRITICAL, SIGSTOP-pauses the lowest-priority
throttle-eligible job (reversible; never the control plane) and SIGCONT-resumes on recovery.

WHAT EACH SAMPLE CAPTURES (one JSON line, ~2 s cadence; 0.5 s when pressure is elevated):
  wall-clock ts (+ iso + monotonic + kern.boottime for reboot detection); TOTAL physical RAM;
  used / available / free / wired / compressor / swap-used; the macOS memory-PRESSURE level
  (1 normal / 2 warn / 4 critical); 1/5/15 load avg; the adaptive ceiling + training budget +
  baseline; and per-tracked-run RSS (label + pid + group RSS + priority + paused). Negligible CPU.

DURABLE STORE: ``.omx/state/memory_blackbox.jsonl`` (fcntl-locked append; rotated at ~20 MB to
``.omx/state/archive/memory_blackbox_<utc>.jsonl`` — NEVER /tmp). Singleton: an fcntl LOCK_NB on
``.omx/state/.memory_blackbox.singleton.lock`` held for the daemon's life (a 2nd instance exits 0).

AUTO-START: ``ensure_blackbox_running()`` is called by ``tools/spawn_durable_daemon.py`` on the first
training launch (idempotent) so the black box is recording before any heavy job runs.

CLI:
    memory_blackbox.py --daemon [--interval 2 --fast-interval 0.5 --govern]   # the recorder+governor
    memory_blackbox.py --sample-once [--json]                                  # one sample to stdout
    memory_blackbox.py --tail [N] [--json]                                     # last N recorded samples
    memory_blackbox.py --last-crash [--minutes 10]                             # trajectory into the last gap
    memory_blackbox.py --status                                               # is the daemon running?
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

_TOOLS = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import system_memory_governor as gov  # noqa: E402

_STATE = _REPO_ROOT / ".omx" / "state"
_ARCHIVE = _STATE / "archive"
_BLACKBOX = _STATE / "memory_blackbox.jsonl"
_BLACKBOX_LOCK = _STATE / ".memory_blackbox.jsonl.lock"
_SINGLETON_LOCK = _STATE / ".memory_blackbox.singleton.lock"
_ACTION_LOG = _STATE / "memory_blackbox_actions.log"

DEFAULT_INTERVAL_S = 2.0
DEFAULT_FAST_INTERVAL_S = 0.5   # cadence when pressure is elevated
ROTATE_BYTES = 20 * 1024 * 1024
DEFAULT_GAP_SECONDS = 30.0      # a ts jump > this (or a boottime change) is a crash/reboot boundary
_BLACKBOX_LABEL = "memory_blackbox"


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _read_boottime_sec() -> int:
    """kern.boottime seconds (changes on reboot -> the robust reboot signal)."""
    raw = gov._sysctl("kern.boottime")
    if not raw:
        return 0
    m = re.search(r"sec\s*=\s*(\d+)", raw)
    return int(m.group(1)) if m else 0


# ─────────────────────────── one sample ───────────────────────────
def sample_once(*, jobs=None, snapshot=None) -> dict:
    """Gather ONE black-box sample: system snapshot + adaptive ceiling + per-tracked-job RSS. Pure I/O
    (no side effects). ``jobs``/``snapshot`` injectable for testing."""
    snap = snapshot if snapshot is not None else gov.read_system_memory_snapshot()
    tracked = jobs if jobs is not None else gov.list_tracked_jobs()
    tracked_current = gov.sum_tracked_current_gib(tracked)
    ceiling = gov.compute_adaptive_ceiling(
        total_gib=snap.total_gib, used_gib=snap.used_gib, tracked_current_gib=tracked_current)
    level = gov.classify_pressure(snap)
    return {
        "ts": round(time.time(), 3),
        "ts_iso": _utc_now_iso(),
        "mono": round(time.monotonic(), 3),
        "boottime": _read_boottime_sec(),
        "total_gib": round(snap.total_gib, 3),
        "used_gib": round(snap.used_gib, 3),
        "available_gib": round(snap.available_gib, 3),
        "free_gib": round(snap.free_gib, 3),
        "wired_gib": round(snap.wired_gib, 3),
        "compressor_gib": round(snap.compressor_gib, 3),
        "swap_used_gib": round(snap.swap_used_gib, 3),
        "pressure_level": snap.pressure_level,
        "pressure": level,
        "load1": round(snap.load1, 2),
        "load5": round(snap.load5, 2),
        "load15": round(snap.load15, 2),
        # accounting-trust fields (so a crash trajectory shows whether the reading was validated)
        "available_primary_gib": round(snap.available_primary_gib, 3),
        "closure_gib": round(snap.closure_gib, 3),
        "closure_ok": snap.closure_ok,
        "cross_validated": snap.cross_validated,
        "discrepancy_gib": round(snap.discrepancy_gib, 3),
        "fail_safe": snap.fail_safe,
        "safety_margin_gib": round(ceiling.safety_margin_gib, 2),
        "adaptive_ceiling_gib": round(ceiling.adaptive_ceiling_gib, 2),
        "baseline_gib": round(ceiling.baseline_gib, 2),
        "training_budget_gib": round(ceiling.training_budget_gib, 2),
        "system_used_headroom_gib": round(ceiling.adaptive_ceiling_gib - snap.used_gib, 2),
        "tracked_sum_gib": round(tracked_current, 2),
        "tracked": [j.to_json() for j in tracked],
        "sampler_pid": os.getpid(),
    }


# ─────────────────────────── durable append + rotation ───────────────────────────
def _rotate_if_needed_locked(path: Path) -> None:
    """Rotate the live JSONL to the archive when it exceeds ROTATE_BYTES. MUST hold the append lock."""
    try:
        if path.exists() and path.stat().st_size > ROTATE_BYTES:
            _ARCHIVE.mkdir(parents=True, exist_ok=True)
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = _ARCHIVE / f"memory_blackbox_{stamp}.jsonl"
            os.replace(path, dest)
    except OSError:
        pass


def append_sample(sample: dict, *, path: Path | None = None, lock_path: Path | None = None) -> None:
    """fcntl-locked append of one sample line; rotates the file at ROTATE_BYTES first."""
    p = path or _BLACKBOX
    lp = lock_path or _BLACKBOX_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lp), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _rotate_if_needed_locked(p)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _log_action(msg: str) -> None:
    line = f"{_utc_now_iso()} {msg}"
    try:
        _ACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_ACTION_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


# ─────────────────────────── read helpers (tail / last-crash) ───────────────────────────
def _iter_sample_files() -> list[Path]:
    """Live file + all archives, chronological (archives sort by name = utc stamp; live last)."""
    files = sorted(_ARCHIVE.glob("memory_blackbox_*.jsonl")) if _ARCHIVE.exists() else []
    if _BLACKBOX.exists():
        files.append(_BLACKBOX)
    return files


def _iter_samples(files: Sequence[Path] | None = None) -> Iterator[dict]:
    for fp in (files if files is not None else _iter_sample_files()):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "ts" in obj:
                yield obj


def _all_samples_sorted() -> list[dict]:
    return sorted(_iter_samples(), key=lambda s: float(s.get("ts", 0.0)))


def find_last_gap(samples: Sequence[dict], gap_seconds: float = DEFAULT_GAP_SECONDS) -> dict | None:
    """Find the MOST RECENT crash/reboot boundary: the last consecutive pair whose ts gap exceeds
    ``gap_seconds`` OR whose boottime changed. Returns ``{index_before, before, after, dt, reboot}``
    or None if the trajectory is continuous. PURE."""
    last: dict | None = None
    for i in range(len(samples) - 1):
        a, b = samples[i], samples[i + 1]
        dt = float(b.get("ts", 0.0)) - float(a.get("ts", 0.0))
        boot_a, boot_b = int(a.get("boottime", 0)), int(b.get("boottime", 0))
        reboot = boot_a and boot_b and boot_a != boot_b
        if dt > gap_seconds or reboot:
            last = {"index_before": i, "before": a, "after": b, "dt": dt, "reboot": bool(reboot)}
    return last


def last_crash_report(minutes: float = 10.0, gap_seconds: float = DEFAULT_GAP_SECONDS) -> dict:
    """Build the trajectory of the ``minutes`` leading INTO the most recent gap/reboot. PURE-ish read."""
    samples = _all_samples_sorted()
    if not samples:
        return {"status": "no_samples", "note": "black box has no recorded samples yet"}
    gap = find_last_gap(samples, gap_seconds)
    if gap is None:
        # No discontinuity — the sampler has been continuous (no crash captured). Show the tail anyway.
        window_start = float(samples[-1].get("ts", 0.0)) - minutes * 60
        window = [s for s in samples if float(s.get("ts", 0.0)) >= window_start]
        return {
            "status": "continuous_no_gap",
            "note": "no crash/reboot gap detected; sampler has been continuous",
            "window_samples": window,
            "summary": _window_summary(window),
        }
    boundary_ts = float(gap["before"].get("ts", 0.0))
    window_start = boundary_ts - minutes * 60
    window = [s for s in samples if window_start <= float(s.get("ts", 0.0)) <= boundary_ts]
    return {
        "status": "reboot" if gap["reboot"] else "sampler_death_gap",
        "note": (
            f"REBOOT detected (kern.boottime changed) at the boundary"
            if gap["reboot"]
            else f"SAMPLER-DEATH gap of {gap['dt']:.0f}s (crash/hang/kill) — the last sample before "
                 f"silence is the crash edge"
        ),
        "boundary_utc": gap["before"].get("ts_iso"),
        "gap_seconds": round(gap["dt"], 1),
        "window_samples": window,
        "summary": _window_summary(window),
    }


def _window_summary(window: Sequence[dict]) -> dict:
    if not window:
        return {}
    used = [float(s.get("used_gib", 0.0)) for s in window]
    avail = [float(s.get("available_gib", 0.0)) for s in window]
    press = [int(s.get("pressure_level", 0)) for s in window]
    return {
        "n_samples": len(window),
        "peak_used_gib": round(max(used), 2),
        "min_available_gib": round(min(avail), 2),
        "max_pressure_level": max(press) if press else 0,
        "final_used_gib": round(used[-1], 2),
        "final_available_gib": round(avail[-1], 2),
    }


def tail(n: int = 20) -> list[dict]:
    return _all_samples_sorted()[-n:]


# ─────────────────────────── singleton + daemon loop ───────────────────────────
def _acquire_singleton() -> int | None:
    """Acquire the non-blocking singleton lock. Returns the fd (held open for the daemon's life) or
    None if another instance already holds it."""
    _SINGLETON_LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(_SINGLETON_LOCK), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    with contextlib.suppress(OSError):
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {_utc_now_iso()}\n".encode())
    return fd


def _govern_tick(consecutive_warn: int, consecutive_critical: int) -> tuple[int, int, dict | None]:
    """One governor decision from a fresh live read; act (pause/resume) and return the updated
    consecutive counters + an action record (or None)."""
    snap = gov.read_system_memory_snapshot()
    jobs = gov.list_tracked_jobs()
    level = gov.classify_pressure(snap)
    consecutive_warn = consecutive_warn + 1 if level == "warn" else 0
    consecutive_critical = consecutive_critical + 1 if level == "critical" else 0
    action = gov.decide_governor_action(
        level=level, consecutive_warn=consecutive_warn, consecutive_critical=consecutive_critical, jobs=jobs)
    record: dict | None = None
    if action.action == "pause" and action.target is not None:
        ok = gov.pause_job(action.target)
        record = {"action": "pause", "label": action.target.label, "pid": action.target.pid,
                  "rss_gib": round(action.target.current_rss_gib, 1), "ok": ok, "reason": action.reason}
        _log_action(f"GOVERNOR PAUSE label={action.target.label!r} pid={action.target.pid} "
                    f"rss={action.target.current_rss_gib:.1f}GiB ok={ok} :: {action.reason}")
    elif action.action == "resume" and action.resume_targets:
        results = [(j.label, gov.resume_job(j)) for j in action.resume_targets]
        record = {"action": "resume", "results": results, "reason": action.reason}
        _log_action(f"GOVERNOR RESUME {results} :: {action.reason}")
    elif action.action in ("escalate_alert", "alert") and level in ("warn", "critical"):
        record = {"action": action.action, "reason": action.reason}
        _log_action(f"GOVERNOR {action.action.upper()} avail={snap.available_gib:.1f}GiB "
                    f"level={level} :: {action.reason}")
    return consecutive_warn, consecutive_critical, record


def run_daemon(
    *,
    interval: float = DEFAULT_INTERVAL_S,
    fast_interval: float = DEFAULT_FAST_INTERVAL_S,
    govern: bool = True,
    max_iterations: int | None = None,
) -> int:
    """The always-on recorder loop (+ governor). Samples, appends, and — when ``govern`` — evaluates the
    dynamic throttle each tick. Cadence speeds up (``fast_interval``) whenever pressure is elevated."""
    fd = _acquire_singleton()
    if fd is None:
        print("[memory-blackbox] another instance already holds the singleton lock; exiting 0.")
        return 0
    _log_action(f"BLACKBOX start pid={os.getpid()} interval={interval}s fast={fast_interval}s govern={govern}")
    consecutive_warn = consecutive_critical = 0
    it = 0
    try:
        while max_iterations is None or it < max_iterations:
            it += 1
            try:
                sample = sample_once()
                append_sample(sample)
                elevated = sample.get("pressure") != "normal"
                if govern:
                    try:
                        consecutive_warn, consecutive_critical, rec = _govern_tick(
                            consecutive_warn, consecutive_critical)
                        if rec is not None:
                            sample["governor_action"] = rec  # (not re-appended; live-log only)
                    except Exception as exc:  # governor hiccup must NEVER kill the recorder
                        _log_action(f"GOVERNOR ERROR (non-fatal): {exc!r}")
            except Exception as exc:  # a sample hiccup must never kill the black box
                _log_action(f"SAMPLE ERROR (non-fatal): {exc!r}")
                elevated = False
            if max_iterations is not None and it >= max_iterations:
                break
            time.sleep(fast_interval if elevated else interval)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    return 0


# ─────────────────────────── ensure-running (auto-start hook) ───────────────────────────
def is_blackbox_running() -> bool:
    """True iff a memory_blackbox daemon is registered running AND alive (registry + liveness)."""
    reg = _REPO_ROOT / ".omx" / "state" / "durable_daemons.json"
    try:
        rows = json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else []
    except (json.JSONDecodeError, OSError):
        rows = []
    if not isinstance(rows, list):
        return False
    for r in rows:
        if not isinstance(r, dict) or r.get("label") != _BLACKBOX_LABEL or r.get("status") != "running":
            continue
        pid = int(r.get("pid", 0) or 0)
        if pid > 0:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                continue
    return False


def ensure_blackbox_running(*, verbose: bool = True) -> bool:
    """Idempotently ensure the black-box daemon is running (called from spawn_durable_daemon on the
    first training launch). Launches it via spawn_durable_daemon so it is durable + registered +
    liveness-verified. Returns True iff running (already or newly started)."""
    if is_blackbox_running():
        return True
    try:
        import spawn_durable_daemon as sdd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - defensive
        if verbose:
            print(f"[memory-blackbox] WARNING: cannot auto-start (spawn_durable_daemon import: {exc})",
                  file=sys.stderr)
        return False
    _STATE.mkdir(parents=True, exist_ok=True)
    log = _STATE / "memory_blackbox.daemon.log"
    argv = [
        "--log", str(log), "--label", _BLACKBOX_LABEL,
        "--skip-mem-preflight",            # protection infra MUST launch even under pressure
        "--skip-admission-gate",           # the black box is control-plane infra, never admission-gated
        "--verify-s", "2.0",
        "--", sys.executable, str(_TOOLS / "memory_blackbox.py"), "--daemon",
    ]
    rc = sdd.main(argv)
    if verbose:
        if rc == 0:
            print("[memory-blackbox] auto-started the always-on recorder+governor daemon.")
        else:
            print(f"[memory-blackbox] WARNING: auto-start returned rc={rc} (see {log})", file=sys.stderr)
    return rc == 0


# ─────────────────────────── CLI ───────────────────────────
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daemon", action="store_true", help="run the always-on recorder+governor loop")
    ap.add_argument("--sample-once", action="store_true", help="print ONE sample and exit")
    ap.add_argument("--tail", nargs="?", type=int, const=20, default=None, help="print last N samples")
    ap.add_argument("--last-crash", action="store_true", help="print the trajectory into the last gap/reboot")
    ap.add_argument("--minutes", type=float, default=10.0, help="--last-crash lookback window (minutes)")
    ap.add_argument("--gap-seconds", type=float, default=DEFAULT_GAP_SECONDS,
                    help="ts jump (s) that counts as a crash/reboot boundary (default 30)")
    ap.add_argument("--status", action="store_true", help="report whether the black-box daemon is running")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    ap.add_argument("--fast-interval", type=float, default=DEFAULT_FAST_INTERVAL_S)
    ap.add_argument("--no-govern", action="store_true", help="record only; do NOT run the throttle governor")
    ap.add_argument("--json", action="store_true", help="JSON output for --sample-once / --tail")
    args = ap.parse_args(argv)

    if args.sample_once:
        s = sample_once()
        print(json.dumps(s, indent=None if args.json else 2))
        return 0
    if args.tail is not None:
        rows = tail(args.tail)
        if args.json:
            for r in rows:
                print(json.dumps(r))
        else:
            print(json.dumps(rows, indent=2))
        return 0
    if args.last_crash:
        print(json.dumps(last_crash_report(minutes=args.minutes, gap_seconds=args.gap_seconds), indent=2))
        return 0
    if args.status:
        running = is_blackbox_running()
        print(json.dumps({"running": running, "label": _BLACKBOX_LABEL,
                          "jsonl": str(_BLACKBOX), "exists": _BLACKBOX.exists()}))
        return 0
    if args.daemon:
        return run_daemon(interval=args.interval, fast_interval=args.fast_interval, govern=not args.no_govern)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
