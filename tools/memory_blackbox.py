#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Memory BLACK-BOX recorder + dynamic governor daemon — so a future crash yields SIGNAL, not nothing.

WHY (operator P0, 2026-07-02): the machine crashed from system-wide memory exhaustion and we had NO
memory trajectory logged before the runs died — no black box. This is the always-on lightweight
sampler that records the SYSTEM memory trajectory to a durable, rotated JSONL so ``--last-crash`` can
replay exactly what led into the most recent crash/reboot. It DOUBLES AS THE GOVERNOR: each sample it
evaluates memory pressure and, when sustained WARN/CRITICAL, SIGSTOP-pauses the lowest-priority
throttle-eligible job (reversible; never the control plane) and SIGCONT-resumes on recovery.

RECOVERY IS RE-ARMED (ddm_gb1, 2026-08-15). "On recovery" used to mean "when the macOS pressure
level returns to normal" — a STICKY signal, so five live jobs sat SIGSTOPped for 75+ minutes on a
box with 40.5 GiB free. The throttle now resumes on the governor's OWN derived free-GiB reference
(``gov.derived_resume_free_gib``), refuses to PAUSE at all while free memory is above that same
reference, and carries a hard ``max_stop_duration`` escape hatch. Every pid it stops is written to a
persisted ledger and SIGCONT-swept on exit (atexit + SIGTERM/SIGINT); ``--resume-stopped`` is the
sweep for what a SIGKILL leaves behind.

THE ACTUATOR IS DEFAULT-OFF AND DURABLY ARMED (ddm_mb1, 2026-08-16). ddm_gb1 fixed the throttle's
MECHANISM but not its ARMING: ``run_daemon`` still defaulted ``govern=True`` and the auto-start path
passed no opt-out, so the next training launch would have silently restarted the un-adjudicated
SIGSTOP actuator. Now the RECORDER (read-only, score-neutral observability) is always on, while the
THROTTLE actuator runs only when armed by ``TAC_GOV_THROTTLE_ARM=1`` or a truthy
``.omx/state/governor_throttle_arm.flag`` — and its state + reason are LOGGED at every start, so
"off" is a tracked state rather than a forgotten default. The exit/startup SIGCONT sweeps are
UNCONDITIONAL (SIGCONT-only, and a recorder-only daemon must still rescue what an armed predecessor
stranded).

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
    memory_blackbox.py --stopped-ledger                                       # which pids it SIGSTOPped
    memory_blackbox.py --resume-stopped                                       # SIGCONT sweep + clear
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import datetime as _dt
import fcntl
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Iterator, Mapping, Sequence

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
# ── the STOPPED-SET ledger (ddm_gb1 D3) ─────────────────────────────────────────────────────────
# Every pid this daemon SIGSTOPs is written here BEFORE it is signalled, and removed when it is
# resumed. Purpose: a SIGSTOPped victim cannot resume itself, so the daemon's death must never be
# able to strand one. atexit + SIGTERM/SIGINT handlers sweep it; and because it is on DISK, even a
# SIGKILL (which runs no handler) leaves a machine-readable list a successor — or an operator
# running ``memory_blackbox.py --resume-stopped`` — can sweep. MEASURED 2026-08-15: the CONT sweep
# for five stranded jobs had to be reconstructed BY HAND from ps output.
_STOPPED_LEDGER = _STATE / "memory_blackbox_stopped_pids.json"
_STOPPED_LEDGER_LOCK = _STATE / ".memory_blackbox_stopped_pids.lock"
STOPPED_LEDGER_SCHEMA = "memory_blackbox_stopped_pids.v1"

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
def sample_once(*, jobs=None, snapshot=None, floor_smoother=None, process_samples=None) -> dict:
    """Gather ONE black-box sample: system snapshot + adaptive ceiling (tier-scaled DERIVED safety
    floor, BUILD #298 — full decomposition recorded per tick) + per-tracked-job RSS. Pure I/O (no
    side effects). ``jobs``/``snapshot``/``floor_smoother``/``process_samples`` injectable for
    testing; the daemon passes a persistent ``gov.SafetyFloorSmoother`` so the applied floor rises
    instantly but decays slowly (admission verdicts don't flap on an oscillating control plane).

    ddm_gb1 D1: the derived floor's measured leg is now the RSS of the NAMED control-plane
    processes, read from ``process_samples`` (or a live ``ps`` scan when omitted) — NOT
    ``used - tracked``, which is total-used-including-file-cache and produced the permanent
    92-GiB-clamped WARN. The ceiling's ``baseline_gib`` still uses ``used - tracked``; that is the
    quantity the ceiling arithmetic actually wants (see ``gov.non_workload_used_gib``).

    COST: the cp leg costs one extra ``ps`` per tick (``list_tracked_jobs`` keeps its own live scan
    because that scan is also what records the RSS-growth history). ~40 ms per 2 s tick, paid for
    measuring the right object."""
    snap = snapshot if snapshot is not None else gov.read_system_memory_snapshot()
    tracked = jobs if jobs is not None else gov.list_tracked_jobs()
    tracked_current = gov.sum_tracked_current_gib(tracked)
    floor = gov.derive_safety_floor(
        total_gib=snap.total_gib,
        measured_cp_rss_gib=gov.measured_control_plane_rss_gib(samples=process_samples),
        override_gib=gov.safety_floor_env_override_gib(),
        log_fn=lambda m: print(m, file=sys.stderr),
    )
    applied_floor = (float(floor_smoother.update(floor.floor_gib))
                     if floor_smoother is not None else floor.floor_gib)
    ceiling = gov.compute_adaptive_ceiling(
        total_gib=snap.total_gib, used_gib=snap.used_gib, tracked_current_gib=tracked_current,
        safety_margin_gib=applied_floor, floor=floor)
    level = gov.classify_pressure(
        snap, warn_free_gib=gov.derived_warn_free_gib(snap.total_gib),
        critical_free_gib=gov.derived_critical_free_gib(snap.total_gib))
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
        # Post GB-F1 tracked rows are TRUE GiB (converted once at the list_tracked_jobs read
        # boundary); this marker lets mixed-era JSONL consumers (witness_memory_preflight) skip
        # the legacy x0.9537 units correction on new rows.
        "tracked_units": "true_gib",
        # Per-tick derived-floor decomposition (which leg won + all leg values + the applied,
        # possibly-smoothed floor) — max observability into the tier-scaled floor (BUILD #298).
        "safety_floor": {**floor.to_json(), "applied_floor_gib": round(applied_floor, 3),
                         "smoothed": floor_smoother is not None},
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


# ─────────────────────────── the STOPPED-SET ledger (ddm_gb1 D3) ───────────────────────────
def _empty_ledger() -> dict:
    return {"schema": STOPPED_LEDGER_SCHEMA, "daemon_pid": None, "updated_utc": None, "stopped": {}}


def load_stopped_ledger(*, path: Path | None = None) -> dict:
    """Read the stopped-set ledger (never raises; a corrupt/absent file reads as empty)."""
    p = path or _STOPPED_LEDGER
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (json.JSONDecodeError, OSError):
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("stopped"), dict):
        return _empty_ledger()
    data.setdefault("schema", STOPPED_LEDGER_SCHEMA)
    data.setdefault("daemon_pid", None)
    data.setdefault("updated_utc", None)
    return data


def _mutate_stopped_ledger(mutate_fn, *, path: Path | None = None,
                           lock_path: Path | None = None) -> dict:
    """fcntl-locked read-modify-atomic-write of the stopped-set ledger (tmp + os.replace)."""
    p = path or _STOPPED_LEDGER
    lp = lock_path or _STOPPED_LEDGER_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lp), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        data = load_stopped_ledger(path=p)
        data = mutate_fn(data)
        data["schema"] = STOPPED_LEDGER_SCHEMA
        data["daemon_pid"] = os.getpid()
        data["updated_utc"] = _utc_now_iso()
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, p)
        return data
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def record_stopped_pids(pids: Sequence[int], *, label: str, path: Path | None = None,
                        lock_path: Path | None = None, now_ts: float | None = None) -> dict:
    """Record ``pids`` as SIGSTOPped by ``label``. Idempotent: an already-recorded pid KEEPS its
    original ``stopped_ts`` so the escape hatch measures the true hold duration, not the last
    observation."""
    now = time.time() if now_ts is None else float(now_ts)

    def _add(data: dict) -> dict:
        stopped = dict(data.get("stopped") or {})
        for pid in pids:
            key = str(int(pid))
            if key in stopped and stopped[key].get("stopped_ts") is not None:
                continue
            stopped[key] = {"label": label, "stopped_ts": now, "stopped_iso": _utc_now_iso()}
        data["stopped"] = stopped
        return data

    return _mutate_stopped_ledger(_add, path=path, lock_path=lock_path)


def _drop_ledger_keys(keys: Sequence[str], *, path: Path | None = None,
                      lock_path: Path | None = None) -> dict:
    """Drop raw ledger KEYS (used by the sweep, which also drops malformed ones)."""
    drop = set(keys)

    def _remove(data: dict) -> dict:
        data["stopped"] = {k: v for k, v in (data.get("stopped") or {}).items() if k not in drop}
        return data

    return _mutate_stopped_ledger(_remove, path=path, lock_path=lock_path)


def record_resumed(*, pids: Sequence[int] = (), label: str | None = None,
                   path: Path | None = None, lock_path: Path | None = None) -> dict:
    """Drop ``pids`` and/or every entry carrying ``label`` from the ledger (a resumed job is no
    longer stranded, so it must not keep aging toward the escape hatch)."""
    drop = {str(int(p)) for p in pids}

    def _remove(data: dict) -> dict:
        stopped = {
            k: v for k, v in (data.get("stopped") or {}).items()
            if k not in drop and not (label is not None and v.get("label") == label)
        }
        data["stopped"] = stopped
        return data

    return _mutate_stopped_ledger(_remove, path=path, lock_path=lock_path)


def paused_since_ts(*, path: Path | None = None) -> dict[int, float]:
    """``{pid: wall-clock ts when it was SIGSTOPped}`` — the escape hatch's age reference.

    WALL clock, not monotonic, ON PURPOSE: the ledger must survive a daemon restart, and monotonic
    clocks are only comparable within one process. A wall-clock jump (NTP, machine sleep) can only
    make a stopped job look OLDER and therefore resume EARLIER — the safe direction."""
    out: dict[int, float] = {}
    for key, row in (load_stopped_ledger(path=path).get("stopped") or {}).items():
        try:
            out[int(key)] = float(row["stopped_ts"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def resume_all_stopped(*, path: Path | None = None, lock_path: Path | None = None,
                       kill_fn=None, state_fn=None, verify_stopped: bool = True,
                       log: bool = True) -> dict:
    """SIGCONT every pid the ledger says this governor stopped, then clear it. Returns
    ``{"resumed": [...], "skipped_not_stopped": [...], "missing": [...]}``.

    PID-RECYCLING SAFE: with ``verify_stopped`` (default) a pid is only signalled when ``ps`` still
    reports it in state ``T``. A recycled pid running normal work is skipped — SIGCONT to a running
    process is a no-op anyway, but not touching it at all is the honest gate. Every examined pid is
    dropped from the ledger either way, so the sweep is idempotent and cannot loop.

    ONLY EVER SIGCONT. There is no kill-class signal on this path (the control-plane kill-semantics
    gauntlet, #409/#172 lineage)."""
    kill = kill_fn if kill_fn is not None else os.kill
    state = state_fn if state_fn is not None else gov._process_state
    ledger = load_stopped_ledger(path=path)
    resumed: list[int] = []
    skipped: list[int] = []
    missing: list[int] = []
    malformed: list[str] = []
    rows: list[tuple[int, dict]] = []
    for key, row in (ledger.get("stopped") or {}).items():
        try:
            rows.append((int(key), row if isinstance(row, dict) else {}))
        except (TypeError, ValueError):
            # A malformed key can never be signalled, and leaving it would let the ledger grow
            # without bound; it is dropped below with everything else this sweep examined.
            malformed.append(str(key))
    for pid, row in sorted(rows):
        if verify_stopped and not gov.is_paused_state(state(pid)):
            skipped.append(pid)
            continue
        try:
            kill(pid, signal.SIGCONT)
            resumed.append(pid)
        except (ProcessLookupError, PermissionError, OSError):
            missing.append(pid)
        else:
            if log:
                _log_action(f"STOPPED-LEDGER SWEEP: SIGCONT pid={pid} label={row.get('label')!r} "
                            f"(stopped at {row.get('stopped_iso')})")
    if resumed or skipped or missing or malformed:
        _drop_ledger_keys([*(str(p) for p in (*resumed, *skipped, *missing)), *malformed],
                          path=path, lock_path=lock_path)
        if log:
            _log_action(f"STOPPED-LEDGER SWEEP done: resumed={resumed} "
                        f"skipped_not_stopped={skipped} missing={missing} malformed={malformed}")
    return {"resumed": resumed, "skipped_not_stopped": skipped, "missing": missing}


_EXIT_SWEEP_INSTALLED = False


def install_exit_resume_handlers() -> None:
    """Install the atexit + SIGTERM/SIGINT sweep so the daemon can never strand a stopped job.

    A SIGSTOPped process cannot resume itself. The pre-fix daemon had NO exit path that resumed its
    victims, so killing it left them stopped forever (MEASURED 2026-08-15: the CONT sweep for five
    jobs was done by hand). SIGKILL still runs no handler by definition — that is exactly why the
    stopped set is PERSISTED: ``--resume-stopped`` sweeps what a SIGKILL leaves behind. Idempotent;
    a non-main-thread install (no signal handlers available) still gets the atexit leg."""
    global _EXIT_SWEEP_INSTALLED
    if _EXIT_SWEEP_INSTALLED:
        return
    _EXIT_SWEEP_INSTALLED = True

    def _sweep_quietly() -> None:
        with contextlib.suppress(Exception):
            resume_all_stopped()

    atexit.register(_sweep_quietly)

    def _handler(signum, _frame):
        _log_action(f"BLACKBOX received signal {signum} — SIGCONT sweeping the stopped set before exit")
        _sweep_quietly()
        raise SystemExit(128 + int(signum))

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handler)


def _log_action(msg: str) -> None:
    line = f"{_utc_now_iso()} {msg}"
    try:
        _ACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_ACTION_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    # The durable file write above is the record; stdout is a convenience. Guarded because this is
    # reached from the atexit sweep, where stdout may already be closed — and an exception there
    # would abort the SIGCONT sweep that is the whole point of the exit path (ddm_gb1 D3).
    with contextlib.suppress(Exception):
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
            "REBOOT detected (kern.boottime changed) at the boundary"
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


def _adopt_unrecorded_paused(jobs, *, ledger_path: Path | None = None,
                             ledger_lock_path: Path | None = None,
                             now_ts: float | None = None,
                             known: Mapping[int, float] | None = None) -> dict[int, float]:
    """Give every ALREADY-paused job with no ledger entry a stop timestamp NOW; return the updated
    pid -> stop-ts map (so the caller reads the ledger once per tick, not twice).

    Without this the escape hatch is blind to victims it did not stop itself — a job paused by a
    previous (pre-fix, or simply restarted) daemon has no recorded age and would age forever. First
    observation starts the clock: the hatch then bounds the hold at ``max_stop_duration_s`` from
    here, which is late but FINITE. Silent-forever is the failure mode being extincted."""
    since = dict(paused_since_ts(path=ledger_path) if known is None else known)
    now = time.time() if now_ts is None else float(now_ts)
    for j in jobs:
        if j.paused and j.pid not in since:
            with contextlib.suppress(Exception):
                record_stopped_pids([j.pid], label=j.label, path=ledger_path,
                                    lock_path=ledger_lock_path, now_ts=now)
                since[j.pid] = now
    return since


def _stopped_scope_pids(job) -> list[int]:
    """The pid set a pause/resume of ``job`` covers (its full tree), falling back to the registered
    root pid when the tree scan is unavailable or raced to empty."""
    pids: list[int] = []
    with contextlib.suppress(Exception):
        pids = list(gov.job_tree_pids(job))
    if not pids:
        pids = [int(job.pid)]
    return pids


def _govern_tick(consecutive_warn: int, consecutive_critical: int,
                 *, max_stop_duration_s: float | None = None) -> tuple[int, int, dict | None]:
    """One governor decision from a fresh live read; act (pause/resume) and return the updated
    consecutive counters + an action record (or None).

    ddm_gb1 D2/D3: the decision now carries the governor's OWN re-arm references (conservative free
    GiB + the derived resume threshold + the per-pid stop ages from the persisted ledger), and
    every actuation is mirrored into that ledger so nothing this daemon stops can be stranded."""
    snap = gov.read_system_memory_snapshot()
    jobs = gov.list_tracked_jobs()
    # Tier-scaled thresholds (BUILD #298): @128 -> warn 14.8 / critical 8.4 (critical strictly
    # earlier than the legacy 8.0); @8 -> 4.0 / 3.0 (the legacy 15/8 were above everything an
    # 8 GiB box can ever free -> permanent warn).
    level = gov.classify_pressure(
        snap, warn_free_gib=gov.derived_warn_free_gib(snap.total_gib),
        critical_free_gib=gov.derived_critical_free_gib(snap.total_gib))
    consecutive_warn = consecutive_warn + 1 if level == "warn" else 0
    consecutive_critical = consecutive_critical + 1 if level == "critical" else 0
    # ONE ledger read per tick: adopt any un-clocked victim and reuse the resulting map for the
    # decision (a second read here would double the per-tick disk I/O for the same answer).
    since = _adopt_unrecorded_paused(jobs)
    action = gov.decide_governor_action(
        level=level, consecutive_warn=consecutive_warn, consecutive_critical=consecutive_critical,
        jobs=jobs,
        # The re-arm inputs: our own conservative free measurement + our own derived threshold +
        # the persisted stop ages. NEVER the sticky OS pressure level alone.
        available_gib=gov.governing_free_gib(snap),
        resume_free_gib=gov.derived_resume_free_gib(snap.total_gib),
        paused_since_ts=since,
        # Resolved ONCE per daemon (env cannot change mid-process), so a malformed override logs its
        # warning once instead of at the sample rate — the log-spam shape the incident already had.
        max_stop_duration_s=(gov.resolve_max_stop_duration_s()
                             if max_stop_duration_s is None else float(max_stop_duration_s)))
    record: dict | None = None
    if action.action == "pause" and action.target is not None:
        # Record BEFORE signalling: a crash between the two must leave a sweepable ledger, never a
        # stranded pid. Recording a pid we then fail to stop is harmless (the sweep skips any pid
        # that is not in state T).
        scope = _stopped_scope_pids(action.target)
        with contextlib.suppress(Exception):
            record_stopped_pids(scope, label=action.target.label)
        ok = gov.pause_job(action.target)
        record = {"action": "pause", "label": action.target.label, "pid": action.target.pid,
                  "rss_gib": round(action.target.current_rss_gib, 1), "ok": ok,
                  "stopped_pids": scope, "reason": action.reason}
        _log_action(f"GOVERNOR PAUSE label={action.target.label!r} pid={action.target.pid} "
                    f"rss={action.target.current_rss_gib:.1f}GiB ok={ok} pids={scope} "
                    f":: {action.reason}")
    elif action.action == "resume" and action.resume_targets:
        results = [(j.label, gov.resume_job(j)) for j in action.resume_targets]
        for j in action.resume_targets:
            with contextlib.suppress(Exception):
                record_resumed(pids=_stopped_scope_pids(j), label=j.label)
        # ddm_mb1: an ESCAPE-HATCH resume is not routine — it means the throttle held a live
        # measurement past the ceiling, i.e. the re-arm references THEMSELVES failed. Emit it as a
        # TYPED, greppable alarm (CLAUDE.md confound-immune-system L1: turn silent artifacts LOUD),
        # never as another indistinguishable RESUME line. The 2026-08-15 freeze was silent for 75+
        # minutes precisely because nothing in the log said "this is anomalous".
        hatched = gov.ESCAPE_HATCH_REASON_TOKEN in (action.reason or "")
        record = {"action": "resume", "results": results, "reason": action.reason,
                  "escape_hatch": hatched}
        if hatched:
            record["alarm"] = "throttle_escape_hatch"
            _log_action(f"ALARM confound_alarm=throttle_escape_hatch GOVERNOR RESUME {results} "
                        f":: {action.reason}")
        else:
            _log_action(f"GOVERNOR RESUME {results} :: {action.reason}")
    elif action.action in ("escalate_alert", "alert") and level in ("warn", "critical"):
        record = {"action": action.action, "reason": action.reason}
        _log_action(f"GOVERNOR {action.action.upper()} avail={snap.available_gib:.1f}GiB "
                    f"governing_free={gov.governing_free_gib(snap):.1f}GiB "
                    f"level={level} :: {action.reason}")
    return consecutive_warn, consecutive_critical, record


# Guard-band evaluation cadence inside the daemon loop (BUILD #298). ONE source of truth: the
# governor owns it alongside the band machinery it parameterises AND the escape-hatch ceiling
# derived from it (ddm_gb1 D2 — a duplicate literal here would let the two silently decouple).
DEFAULT_BAND_INTERVAL_S = gov.DEFAULT_BAND_INTERVAL_S


def run_daemon(
    *,
    interval: float = DEFAULT_INTERVAL_S,
    fast_interval: float = DEFAULT_FAST_INTERVAL_S,
    govern: bool | None = None,
    max_iterations: int | None = None,
    band: bool = True,
    band_interval_s: float = DEFAULT_BAND_INTERVAL_S,
    band_envelope_frac: float | None = None,
    band_run_dir: Path | None = None,
) -> int:
    """The always-on recorder loop (+ governor + guard bands). Samples, appends, and — when
    ``govern`` — evaluates the dynamic throttle each tick. Cadence speeds up (``fast_interval``)
    whenever pressure is elevated.

    BUILD #298 band wiring: every ``band_interval_s`` (default 30 s; ``band=False`` / ``--no-band``
    opts out) the loop calls the governor's ``band_tick`` IN-PROCESS — exactly the #294 semantics
    (green/yellow/red on ACTUAL RSS vs the frac*RAM envelope; never-kill; defer_to_throttle when
    the system pressure ladder is engaged; red = reversible clean-pause via the canonical
    ``pause_job`` only in the sole-workload case) — with actuation enabled iff ``govern``
    (a ``--no-govern`` recorder emits dry band rows). Band thresholds auto-scale per tier because
    they are fractions of the SAME envelope machinery (0.85*128 -> yellow 92.48 / red 97.92 true
    GiB; 0.55*8 -> 3.74 / 3.96). A persistent ``SafetyFloorSmoother`` keeps the per-tick derived
    floor flap-free (rise-instant / decay <= 0.25 GiB per tick)."""
    fd = _acquire_singleton()
    if fd is None:
        print("[memory-blackbox] another instance already holds the singleton lock; exiting 0.")
        return 0
    # ddm_mb1: the ACTUATOR is default-OFF and durably armed; the RECORDER is always on. ``govern``
    # None = resolve from the arming surface (the auto-start path passes nothing, so it can no
    # longer silently re-enable the throttle); an explicit True/False is a per-invocation override.
    if govern is None:
        arming = gov.throttle_arming()
        govern = arming.armed
    else:
        # An explicit caller override must never be REPORTED as an arming-surface decision — that
        # would make a forced-on test daemon look durably armed in the action log.
        govern = bool(govern)
        arming = gov.Arming(govern, "explicit",
                            f"explicit govern={govern} passed by the caller (overrides the "
                            f"{'armed' if gov.throttle_armed() else 'default-OFF'} arming surface)")
    # ddm_gb1 D3 + ddm_mb1: arm the exit sweep BEFORE the first tick can stop anything, and sweep
    # whatever a PREVIOUS daemon stranded (its SIGKILL ran no handler) so a restart is also a
    # recovery. UNCONDITIONAL — never gated on ``govern``. A recorder-only daemon (now the DEFAULT)
    # must still be able to rescue what a previously-armed governor left in state T; gating this on
    # ``govern`` meant the common post-ddm_mb1 case could see a stranded ledger and walk past it.
    # Both legs are SIGCONT-only, so running them with the actuator disarmed can never hurt.
    install_exit_resume_handlers()
    with contextlib.suppress(Exception):
        stale = resume_all_stopped()
        if stale["resumed"] or stale["skipped_not_stopped"]:
            _log_action(f"BLACKBOX startup sweep of a predecessor's stopped set: {stale}")
    # Resolved ONCE: the env cannot change mid-process, so a malformed override warns once here
    # instead of at the sample rate.
    hatch_s = gov.resolve_max_stop_duration_s()
    _log_action(f"BLACKBOX start pid={os.getpid()} interval={interval}s fast={fast_interval}s "
                f"govern={govern} band={band} band_interval={band_interval_s}s "
                f"max_stop_duration={hatch_s:.0f}s")
    # "Off" is only a TRACKED state if it is surfaced with its reason (CLAUDE.md "'Off' is a
    # tracked queue, never a forgotten default"). One line, every start, either way.
    _log_action(f"THROTTLE ACTUATOR {'ARMED' if govern else 'DISARMED'} "
                f"[source={arming.source}] :: {arming.detail}"
                + ("" if govern else " — recorder still ON; per-job safe_run RSS envelopes carry "
                   "the OOM protection (the MEASURED 2026-08-15 verdict)"))
    consecutive_warn = consecutive_critical = 0
    floor_smoother = gov.SafetyFloorSmoother()
    last_band_mono: float | None = None
    it = 0
    try:
        while max_iterations is None or it < max_iterations:
            it += 1
            try:
                sample = sample_once(floor_smoother=floor_smoother)
                append_sample(sample)
                elevated = sample.get("pressure") != "normal"
                if govern:
                    try:
                        consecutive_warn, consecutive_critical, rec = _govern_tick(
                            consecutive_warn, consecutive_critical,
                            max_stop_duration_s=hatch_s)
                        if rec is not None:
                            sample["governor_action"] = rec  # (not re-appended; live-log only)
                    except Exception as exc:  # governor hiccup must NEVER kill the recorder
                        _log_action(f"GOVERNOR ERROR (non-fatal): {exc!r}")
                if band:
                    now_mono = time.monotonic()
                    if last_band_mono is None or (now_mono - last_band_mono) >= band_interval_s:
                        last_band_mono = now_mono
                        try:
                            d = gov.band_tick(
                                envelope_frac=(band_envelope_frac if band_envelope_frac is not None
                                               else gov.DEFAULT_BAND_ENVELOPE_FRAC),
                                run_dir=band_run_dir, apply=govern)
                            if d.action != "none":
                                _log_action(f"BAND {d.band}/{d.action}: {d.reason}")
                        except Exception as exc:  # band hiccup must NEVER kill the recorder
                            _log_action(f"BAND ERROR (non-fatal): {exc!r}")
            except Exception as exc:  # a sample hiccup must never kill the black box
                _log_action(f"SAMPLE ERROR (non-fatal): {exc!r}")
                elevated = False
            if max_iterations is not None and it >= max_iterations:
                break
            time.sleep(fast_interval if elevated else interval)
    finally:
        # ddm_gb1 D3: a normal loop exit (max_iterations, exception, SystemExit from the signal
        # handler) sweeps too — the atexit leg is the belt, this is the braces. ddm_mb1: also
        # UNCONDITIONAL, for the same reason as the startup sweep — SIGCONT-only, and the ledger it
        # drains may have been written by an armed predecessor rather than by this process.
        with contextlib.suppress(Exception):
            resume_all_stopped()
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
    liveness-verified. Returns True iff running (already or newly started).

    ddm_mb1: the argv below deliberately passes NEITHER ``--govern`` NOR ``--no-govern``, so the
    auto-started daemon defers to the arming surface — RECORDER on, SIGSTOP actuator off unless
    explicitly armed. This is the path that made "the daemon is OFF" untrue in practice: it was
    enforced only by nobody having launched training yet. Adding ``--govern`` here would restore the
    silent re-enable and is refused by ``check_throttle_actuator_defaults_off_until_armed``."""
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
    ap.add_argument("--stopped-ledger", action="store_true",
                    help="print the persisted stopped-set ledger (which pids the governor SIGSTOPped)")
    ap.add_argument("--resume-stopped", action="store_true",
                    help="SIGCONT every pid in the stopped-set ledger and clear it (the SIGKILL "
                         "recovery sweep — a daemon killed with -9 runs no handler)")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    ap.add_argument("--fast-interval", type=float, default=DEFAULT_FAST_INTERVAL_S)
    ap.add_argument("--no-govern", action="store_true", help="record only; do NOT run the throttle governor")
    ap.add_argument("--govern", action="store_true",
                    help="FORCE the SIGSTOP throttle actuator on for this invocation (default: OFF "
                         "unless armed by TAC_GOV_THROTTLE_ARM=1 or a truthy "
                         ".omx/state/governor_throttle_arm.flag). --no-govern wins if both given.")
    # ── guard-band wiring (BUILD #298): in-loop band evaluation, #294 semantics ──
    ap.add_argument("--no-band", action="store_true",
                    help="disable the in-loop guard-band evaluation (default: every --band-interval s)")
    ap.add_argument("--band-interval", type=float, default=DEFAULT_BAND_INTERVAL_S,
                    help="guard-band evaluation cadence in seconds (default 30)")
    ap.add_argument("--band-envelope-frac", type=float, default=None,
                    help="single-workload envelope fraction of total RAM for the bands "
                         "(default: governor DEFAULT_BAND_ENVELOPE_FRAC=0.85; tertiary tier uses 0.55)")
    ap.add_argument("--band-run-dir", type=str, default=None,
                    help="run dir for checkpoint-freshness in band decisions")
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
    if args.stopped_ledger:
        print(json.dumps(load_stopped_ledger(), indent=2))
        return 0
    if args.resume_stopped:
        print(json.dumps(resume_all_stopped(), indent=2))
        return 0
    if args.daemon:
        # Tri-state (ddm_mb1): --no-govern forces OFF, --govern forces ON, neither defers to the
        # arming surface (default OFF). --no-govern wins a contradictory pair: between two operator
        # intents the SAFE one governs, and "off" is the safe one for a SIGSTOP actuator.
        govern_arg = False if args.no_govern else (True if args.govern else None)
        return run_daemon(
            interval=args.interval, fast_interval=args.fast_interval, govern=govern_arg,
            band=not args.no_band, band_interval_s=args.band_interval,
            band_envelope_frac=args.band_envelope_frac,
            band_run_dir=Path(args.band_run_dir) if args.band_run_dir else None)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
