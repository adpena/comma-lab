#!/usr/bin/env python3
"""L1 memory-pressure watchdog: make a silent near-OOM LOUD, and pause before jetsam does (ddm_gov2).

WHAT HAPPENED (MEASURED, ``.omx/state/memory_blackbox.jsonl``, 2026-09-04).  Two ~40 GiB Metal
cells were admitted concurrently by hand-written fire scripts.  The machine did not warn and did
not slow down gracefully -- it collapsed, twice, and macOS jetsam killed background daemons:

    16:45:36.831   compressor  0.66 GiB   swap  2.30   avail 18.83   pressure normal
    16:45:39.981   compressor 19.68 GiB   swap  2.30   avail 33.72   pressure normal   (+6.03 GiB/s)
    16:45:42.714   compressor 46.42 GiB   swap  2.29   avail 29.16   pressure normal   (+9.79 GiB/s)
    16:45:45.105   compressor 63.92 GiB   swap  2.29   avail 23.28   pressure warn
    16:45:56.044   compressor 73.99 GiB   swap 22.03   avail 19.30   pressure CRITICAL  <- jetsam
    17:10:49.046   compressor 76.978 GiB  swap 64.08   avail 18.22   pressure CRITICAL  (peak)
    17:11:05.610   compressor 70.166 GiB  swap 72.00 GiB                                (peak swap)

THREE THINGS THAT DERIVATION SETTLES, and they are not obvious:

1. **The whole collapse takes 16 seconds.**  From the first sample above 16 GiB of compressor to
   CRITICAL is 16.06 s.  A 5 s poll therefore buys about three samples, and a debounce ("fire only
   after two consecutive criticals") would spend most of the runway.  The CRITICAL action here is
   taken on the FIRST critical sample, deliberately.

2. **Availability is a BLIND signal here.**  ``available_gib`` never went below 18.2 GiB during the
   whole event, and the canonical reclaimable basis read 19-33 GiB while daemons were being killed.
   A guard watching free memory would have seen nothing wrong.  Compressor and swap are the signal.

3. **``free`` is NOT usable as a trigger** even though it looks like the leading indicator (it sat
   at 0.02-0.3 GiB for ~40 s before the collapse): over the day's 13,235 samples ``free < 1 GiB``
   was true **28.99%** of the time.  Rejected on its base rate, recorded here so it is not
   re-proposed.  The kept triggers are rare: compressor >= 16 GiB 4.193%, compressor >= 48 GiB
   3.189%, swap >= 4 GiB 4.594%, swap >= 16 GiB 2.199%, pressure >= warn 3.408%, pressure critical
   0.748%.

**THE DEFAULT IS REPORT-ONLY: THIS GUARD ALARMS, IT DOES NOT SIGNAL** (operator decision
2026-09-05, on this tool's own measurements).  In one night SIGSTOP hurt twice -- ng4's trainer
paused three times with only two releases, and mc1's `ceil_block` stranded in state T -- and helped
**zero** times.  Every measured pause was followed by swap GROWTH, never relief, for a reason that
is structural rather than incidental: **a stopped process is the ideal eviction victim** (resident,
dirty, not running), so pausing converts a running job's memory into swap instead of freeing it.
A SIGSTOP relieves CPU and future allocation; it does not return resident bytes.  Under the
never-weaker-state law a guard that destroys work is worse than no guard, so the signalling mode is
now an explicit opt-in (``--act``) and the default is to alarm loudly and NAME THE ACTOR
(``top_rss_growers`` on every row), which is what a human actually needed both times.

Opting in with ``--act`` keeps the full safety apparatus: targeting by RSS growth
(:func:`select_pressure_target`), the bounded pause (:data:`MAX_PAUSE_S`), and the startup
reconciler.  **The reconciler runs in BOTH modes** -- report-only means "never STOP anything", not
"never RESCUE anything", and SIGCONT of a stranded process can only restore work.

    memory_pressure_watchdog run                     # alarm-only (the default)
    memory_pressure_watchdog run --act               # opt in to SIGSTOP/SIGCONT
    memory_pressure_watchdog reconcile               # free anything a dead instance left stopped
    memory_pressure_watchdog once --json
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

ALARM_SCHEMA = "confound_alarm.v1"
SAMPLE_SCHEMA = "memory_pressure_sample.v1"
ALARM_LEDGER = _REPO / ".omx" / "state" / "memory_pressure_alarms.jsonl"

LEVEL_OK = "OK"
LEVEL_WARN = "WARN"
LEVEL_CRITICAL = "CRITICAL"

#: Poll interval.  DERIVED: the 2026-09-04 collapse ran from 16 GiB of compressor to CRITICAL in
#: 16.06 s, so 5 s buys ~3 samples of runway.  Slower polling does not observe the event at all.
DEFAULT_POLL_S = 5.0

#: WARN thresholds.  DERIVED from the ramp: 16 GiB of compressor was first crossed 16.06 s before
#: CRITICAL (base rate 4.193% of the day's samples); swap first moved off its 2.3 GiB floor to
#: 4.883 GiB 9.4 s before CRITICAL (4.594%).
WARN_COMPRESSOR_FRACTION = 16.0 / 128.0
WARN_SWAP_GIB = 4.0
WARN_PRESSURE_LEVEL = 2

#: CRITICAL thresholds.  DERIVED: compressor crossed 48 GiB 10.9 s before jetsam (3.189%); swap
#: crossed 16 GiB 2.6 s before it (2.199%); the OS's own critical level is the ground truth (0.748%).
CRITICAL_COMPRESSOR_FRACTION = 48.0 / 128.0
CRITICAL_SWAP_GIB = 16.0
CRITICAL_PRESSURE_LEVEL = 4

#: A compressor growing this fast has already decided the outcome.  DERIVED: the two measured ramp
#: segments were +6.03 GiB/s and +9.79 GiB/s; nothing else on this box that day approached it.
CRITICAL_COMPRESSOR_GROWTH_GIB_PER_S = 4.0

#: Resume only after the machine has been below WARN for this long.  A shorter window re-admits the
#: allocation into a compressor that has not finished draining (MEASURED: the 16:45 event took
#: ~9 s to fall from 73.99 to 31.74 GiB, and the second event began 25 min later).
CLEAR_HOLD_S = 60.0

#: HARD CEILING on a pause, whatever the clear hold says.  A SIGSTOP without a GUARANTEED SIGCONT
#: is a silent kill of sunk work, and this watchdog proved it on its own second night.
#:
#: WHAT HAPPENED (MEASURED, `.omx/state/memory_pressure_alarms.jsonl` + the memory blackbox,
#: 2026-09-05).  Three CRITICALs, three SIGSTOPs of ng4's trainer (pid 33374), and only TWO
#: SIGCONTs:
#:
#:     SIGSTOP 00:16:45 -> WARN-clear at 00:18:22 (97 s) -> SIGCONT 00:20:49   (244 s paused)
#:     SIGSTOP 00:27:45 -> WARN-clear at 00:28:07 (22 s) -> SIGCONT 00:29:29   (104 s paused)
#:     SIGSTOP 00:34:20 -> NEVER cleared -> stuck in state T until MAIN rescued it at 00:36:35
#:
#: THE MECHANISM, and it is the part that makes this a deadlock rather than bad luck: **stopping a
#: process does not free its memory.**  A stopped process is an ideal eviction victim -- resident,
#: dirty, and not running -- so macOS swapped it out.  Swap crossed the 4 GiB WARN threshold at
#: **00:34:32, twelve seconds AFTER the SIGSTOP**, and stayed above it for 126 samples (4m19s,
#: peaking at 11.29 GiB) with the compressor parked at ~41 GiB.  The pause CREATED the condition
#: that prevented its own release.  No clear hold can ever be met against a metric the pause
#: itself inflates.
#:
#: DERIVED bound: the worst pause that COULD clear took 97 s to go WARN-clear, plus the 60 s hold
#: = 157 s.  180 s leaves 23 s of headroom above that and would have released the third pause at
#: 00:37:20Z instead of never.  The intervention's value is in the first seconds anyway -- every
#: measured burst drained in ~15 s.
MAX_PAUSE_S = 180.0

_PAGE_SIZE_DEFAULT = 16384
_GIB = 1024.0**3

_PRESSURE_LEVELS = {"normal": 1, "warn": 2, "warning": 2, "urgent": 3, "critical": 4}


def utc_text(value: dt.datetime | None = None) -> str:
    return (value or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── sampling ────────────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class PressureSample:
    total_gib: float
    compressor_gib: float
    swap_used_gib: float
    free_gib: float
    pressure_level: int
    pressure: str
    observed_utc: str = dataclasses.field(default_factory=utc_text)
    monotonic: float = dataclasses.field(default_factory=time.monotonic)

    @property
    def compressor_fraction(self) -> float:
        return self.compressor_gib / self.total_gib if self.total_gib > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload.pop("monotonic", None)
        payload["schema"] = SAMPLE_SCHEMA
        payload["compressor_fraction"] = round(self.compressor_fraction, 4)
        return payload


def _run(cmd: Sequence[str], timeout: float = 10.0) -> str:
    try:
        completed = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout or ""


def parse_vm_stat(text: str) -> tuple[float, float]:
    """``(compressor GiB, free GiB)`` from ``vm_stat`` output.

    ``Pages occupied by compressor`` is the RESIDENT cost of compression -- the number that grew to
    76.978 GiB.  ``Pages stored in compressor`` counts the pre-compression pages and is much larger;
    reading the wrong one inflates the alarm by roughly 4.5x.
    """
    page_size = _PAGE_SIZE_DEFAULT
    occupied = free = 0
    for line in text.splitlines():
        if "page size of" in line:
            for token in line.replace("(", " ").replace(")", " ").split():
                if token.isdigit():
                    page_size = int(token)
                    break
        elif line.startswith("Pages occupied by compressor"):
            occupied = int(line.rsplit(":", 1)[1].strip().rstrip("."))
        elif line.startswith("Pages free"):
            free = int(line.rsplit(":", 1)[1].strip().rstrip("."))
    return occupied * page_size / _GIB, free * page_size / _GIB


def parse_swapusage(text: str) -> float:
    """Used swap in GiB from ``sysctl vm.swapusage`` (``total = 2048.00M  used = 1141.31M  ...``)."""
    parts = text.split("used")
    if len(parts) < 2:
        return 0.0
    raw = parts[1].split()[1] if parts[1].split()[0] == "=" else parts[1].split()[0]
    raw = raw.strip().rstrip(",")
    try:
        if raw.endswith("M"):
            return float(raw[:-1]) / 1024.0
        if raw.endswith("G"):
            return float(raw[:-1])
        if raw.endswith("K"):
            return float(raw[:-1]) / (1024.0 * 1024.0)
        return float(raw) / _GIB
    except ValueError:
        return 0.0


def parse_memory_pressure(text: str) -> tuple[int, str]:
    lowered = text.lower()
    for name in ("critical", "urgent", "warn", "normal"):
        if f"system-wide memory pressure: {name}" in lowered or f"memory pressure: {name}" in lowered:
            return _PRESSURE_LEVELS[name], name
    return 1, "normal"


def total_ram_gib() -> float:
    raw = _run(["sysctl", "-n", "hw.memsize"]).strip()
    try:
        return int(raw) / _GIB
    except ValueError:
        return 128.0


def sample_pressure(total_gib: float | None = None) -> PressureSample:
    """One reading of the three signals that describe the collapse."""
    total = total_ram_gib() if total_gib is None else total_gib
    compressor, free = parse_vm_stat(_run(["vm_stat"]))
    swap = parse_swapusage(_run(["sysctl", "vm.swapusage"]))
    level, name = parse_memory_pressure(_run(["memory_pressure", "-Q"]))
    return PressureSample(
        total_gib=total,
        compressor_gib=round(compressor, 4),
        swap_used_gib=round(swap, 4),
        free_gib=round(free, 4),
        pressure_level=level,
        pressure=name,
    )


# ── classification ──────────────────────────────────────────────────────────────────────────────


def classify(sample: PressureSample, previous: PressureSample | None = None) -> tuple[str, list[str]]:
    """``(LEVEL, reasons)`` -- every threshold in the reasons, so the alarm explains itself."""
    reasons: list[str] = []
    level = LEVEL_OK

    growth: float | None = None
    if previous is not None:
        elapsed = sample.monotonic - previous.monotonic
        if elapsed > 0:
            growth = (sample.compressor_gib - previous.compressor_gib) / elapsed

    if sample.pressure_level >= CRITICAL_PRESSURE_LEVEL:
        level = LEVEL_CRITICAL
        reasons.append(f"OS memory pressure is {sample.pressure} (level {sample.pressure_level})")
    if sample.compressor_fraction >= CRITICAL_COMPRESSOR_FRACTION:
        level = LEVEL_CRITICAL
        reasons.append(
            f"compressor {sample.compressor_gib:.2f} GiB >= {CRITICAL_COMPRESSOR_FRACTION * sample.total_gib:.1f} GiB "
            f"({CRITICAL_COMPRESSOR_FRACTION:.3f} of RAM)"
        )
    if sample.swap_used_gib >= CRITICAL_SWAP_GIB:
        level = LEVEL_CRITICAL
        reasons.append(f"swap {sample.swap_used_gib:.2f} GiB >= {CRITICAL_SWAP_GIB:.1f} GiB")
    # RATE ALONE IS NOT A COLLAPSE. A 6 GiB/s ramp from 1 GiB to 5 GiB is a program starting up on
    # a 128 GiB box; the rate rule only means something once the compressor is already carrying
    # real weight. Added after this watchdog SIGSTOPped a live cell three times on rate.
    if (
        growth is not None
        and growth >= CRITICAL_COMPRESSOR_GROWTH_GIB_PER_S
        and sample.compressor_fraction >= WARN_COMPRESSOR_FRACTION
    ):
        level = LEVEL_CRITICAL
        reasons.append(
            f"compressor growing {growth:.2f} GiB/s >= {CRITICAL_COMPRESSOR_GROWTH_GIB_PER_S:.1f} GiB/s "
            f"AND already at {sample.compressor_gib:.2f} GiB "
            f"(>= the {WARN_COMPRESSOR_FRACTION * sample.total_gib:.1f} GiB WARN level; "
            "the measured ramp was 6.03-9.79 GiB/s)"
        )

    if level == LEVEL_CRITICAL:
        return level, reasons

    if sample.pressure_level >= WARN_PRESSURE_LEVEL:
        level = LEVEL_WARN
        reasons.append(f"OS memory pressure is {sample.pressure} (level {sample.pressure_level})")
    if sample.compressor_fraction >= WARN_COMPRESSOR_FRACTION:
        level = LEVEL_WARN
        reasons.append(
            f"compressor {sample.compressor_gib:.2f} GiB >= {WARN_COMPRESSOR_FRACTION * sample.total_gib:.1f} GiB "
            f"({WARN_COMPRESSOR_FRACTION:.3f} of RAM)"
        )
    if sample.swap_used_gib >= WARN_SWAP_GIB:
        level = LEVEL_WARN
        reasons.append(f"swap {sample.swap_used_gib:.2f} GiB >= {WARN_SWAP_GIB:.1f} GiB")
    return level, reasons


# ── the target: the NEWEST governed training cell ───────────────────────────────────────────────


def _create_time(pid: int) -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except Exception:
        return 0.0


def newest_training_cell() -> dict[str, Any] | None:
    """The most recently started governed TRAINING cell.

    SUPERSEDED as the watchdog's target (2026-09-04, see :func:`select_pressure_target`): the
    first anger-data showed the cell is usually NOT the allocator.  Retained as a plain query --
    "which cell started last" is still a real question for diagnostics and the digest.
    """
    try:
        import cell_admission as ca
    except Exception:
        return None
    try:
        cells = [cell for cell in ca.live_cells_from_process_table() if cell.is_cell]
    except Exception:
        return None
    if not cells:
        return None
    ranked = sorted(cells, key=lambda cell: _create_time(cell.trainer_pid or cell.pid))
    target = ranked[-1]
    return {
        "cell_id": target.cell_id,
        "pid": target.pid,
        "trainer_pid": target.trainer_pid,
        "declared_peak_gib": target.declared_peak_gib,
        "stop_pid": target.trainer_pid or target.pid,
        "started_epoch": _create_time(target.trainer_pid or target.pid),
    }


# ── targeting: the ALLOCATOR, not the cell (ddm_gov2, corrected by anger-data 2026-09-04) ───────
#
# THE DEFECT, MEASURED BY THIS WATCHDOG'S OWN FIRST ALARMS.  Eleven WARN rows landed between
# 23:47Z and 23:55Z in three bursts (compressor 20.00 -> 39.83 GiB, swap flat at ~1.01 GiB,
# pressure never left `normal`, no CRITICAL).  The cause was NOT the Metal cell.  It was three
# `ddm_mc1 --stage ceiling` jobs -- launched through the canonical launcher, so GOVERNED, but
# carrying no `run-config`, so ``is_cell=False``.  MEASURED at 23:58Z:
#
#     ceil_planar  pid 81442   10.32 GiB      ng4 cell  pid 33030   0.49 GiB
#     ceil_shift   pid 80930   10.14 GiB
#     ceil_zoom    pid 81089   10.33 GiB      three ceiling jobs = 30.79 GiB, 62.8x the cell
#
# On a CRITICAL the old rule would have SIGSTOPped ng4 -- two hours of sunk work, holding 1.6% of
# the resident footprint -- and left every actual allocator running.  A guard that pauses the
# wrong process is worse than one that pauses nothing: it destroys work AND does not help.
#
# THE RULE.  Rank ALL governed jobs by RSS GROWTH over the recent window, because growth is what
# identifies the thing currently taking memory (a job already sitting at a large steady footprint
# is not the one driving the compressor).  Ties prefer NON-CELLS, and the OLDEST live cell is
# excluded from targeting outright -- it holds the most sunk compute and is never the mistake.
# When the oldest cell is the only live job there is NO target, and the watchdog says so.

#: Polls kept for the growth window. DERIVED: the measured bursts rose and drained inside ~15 s
#: (38.47 -> 22.33 GiB in one 5 s step), so a 6-poll / 30 s window spans a whole burst without
#: averaging two of them together.
GROWTH_WINDOW_POLLS = 6
#: Rows named in a WARN alarm so the CAUSE is legible before anything is CRITICAL.
TOP_GROWERS_REPORTED = 3
#: Growth below this is NOISE, not a signal, and current RSS decides instead.
#:
#: DERIVED by running the pure-growth rule against the live fleet and watching it fail: on a quiet
#: 4 s window every job's growth was within 0.0013 GiB of zero, and the ranking picked the 0.17 GiB
#: dashboard server over a 7.08 GiB ceiling job. Pausing a 0.17 GiB process during a CRITICAL
#: achieves nothing. 0.5 GiB sits ~380x above that measured idle drift and ~20x below the 10.3 GiB
#: allocators, so a real grower still dominates outright and flat jobs fall through to size.
GROWTH_TIE_BAND_GIB = 0.5


@dataclasses.dataclass(frozen=True)
class JobFootprint:
    """One governed job's footprint at one poll."""

    pid: int
    cell_id: str
    is_cell: bool
    rss_gib: float
    started_epoch: float
    stop_pid: int
    declared_peak_gib: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def sample_governed_jobs() -> list[JobFootprint]:
    """Every governed job with its live tree RSS.

    MEASURED cost on this box: 0.168 s for 8 jobs (0.118 s discovery + 0.050 s of tree walks),
    i.e. 3.4% of a 5 s poll -- cheap enough to run every poll, which is what makes a growth
    signal possible at all.
    """
    try:
        import cell_admission as ca
    except Exception:
        return []
    try:
        cells = ca.live_cells_from_process_table()
    except Exception:
        return []
    out: list[JobFootprint] = []
    for cell in cells:
        stop_pid = cell.trainer_pid or cell.pid
        rss = cell.current_rss_gib
        if rss is None:
            rss = ca.process_tree_rss_gib(cell.pid) or 0.0
        out.append(
            JobFootprint(
                pid=cell.pid,
                cell_id=cell.cell_id,
                is_cell=bool(cell.is_cell),
                rss_gib=round(float(rss), 4),
                started_epoch=_create_time(stop_pid),
                stop_pid=stop_pid,
                declared_peak_gib=float(cell.declared_peak_gib),
            )
        )
    return out


def rss_growth(history: Sequence[Sequence[JobFootprint]]) -> dict[int, float]:
    """``{pid: RSS growth in GiB}`` across the retained window.

    A pid first seen mid-window counts its whole footprint as growth -- a job that appeared during
    a pressure burst IS the growth, and treating its arrival as zero would hide the usual cause.
    """
    if not history:
        return {}
    latest = {job.pid: job.rss_gib for job in history[-1]}
    earliest: dict[int, float] = {}
    for snapshot in history:
        for job in snapshot:
            earliest.setdefault(job.pid, job.rss_gib)
    first_seen_index: dict[int, int] = {}
    for index, snapshot in enumerate(history):
        for job in snapshot:
            first_seen_index.setdefault(job.pid, index)
    growth: dict[int, float] = {}
    for pid, current in latest.items():
        base = 0.0 if first_seen_index.get(pid, 0) > 0 else earliest.get(pid, current)
        growth[pid] = round(current - base, 4)
    return growth


def rank_targets(
    jobs: Sequence[JobFootprint],
    growth: Mapping[int, float],
) -> tuple[list[dict[str, Any]], int | None]:
    """``(ranked candidates, excluded oldest-cell pid)`` under the standing targeting rule.

    Order: RSS GROWTH (bucketed by ``GROWTH_TIE_BAND_GIB``) desc -> CURRENT RSS desc -> NON-CELL
    before cell -> NEWEST first.  The oldest live cell is removed from the candidate list entirely.

    The RSS tiebreak is load-bearing and was added after the pure-growth rule was MEASURED failing
    on a quiet machine (see ``GROWTH_TIE_BAND_GIB``): when nothing is growing, the job worth
    pausing is simply the biggest one.
    """
    cells = [job for job in jobs if job.is_cell]
    oldest_cell_pid = min(cells, key=lambda job: job.started_epoch).pid if cells else None
    candidates = [job for job in jobs if job.pid != oldest_cell_pid]
    ranked = sorted(
        candidates,
        key=lambda job: (
            -int(max(float(growth.get(job.pid, 0.0)), 0.0) / GROWTH_TIE_BAND_GIB),
            -job.rss_gib,
            job.is_cell,
            -job.started_epoch,
        ),
    )
    return (
        [{**job.as_dict(), "rss_delta_gib": float(growth.get(job.pid, 0.0))} for job in ranked],
        oldest_cell_pid,
    )


def select_pressure_target(
    history: Sequence[Sequence[JobFootprint]],
) -> dict[str, Any] | None:
    """The job to pause, with the arithmetic that chose it recorded on the row."""
    if not history or not history[-1]:
        return None
    growth = rss_growth(history)
    ranked, oldest_cell_pid = rank_targets(history[-1], growth)
    if not ranked:
        return None
    chosen = ranked[0]
    return {
        **chosen,
        "selection_rule": (
            f"highest RSS growth over the window (bucketed at {GROWTH_TIE_BAND_GIB} GiB); "
            "ties break on current RSS, then non-cell, then newest"
        ),
        "growth_window_polls": len(history),
        "excluded_oldest_cell_pid": oldest_cell_pid,
        "runners_up": ranked[1:TOP_GROWERS_REPORTED],
    }


def top_growers(history: Sequence[Sequence[JobFootprint]], limit: int = TOP_GROWERS_REPORTED) -> list[dict[str, Any]]:
    """The ``limit`` biggest RSS growers right now -- the WARN row's legibility payload."""
    if not history or not history[-1]:
        return []
    growth = rss_growth(history)
    ordered = sorted(
        history[-1],
        key=lambda job: (
            -int(max(float(growth.get(job.pid, 0.0)), 0.0) / GROWTH_TIE_BAND_GIB),
            -job.rss_gib,
        ),
    )
    return [{**job.as_dict(), "rss_delta_gib": float(growth.get(job.pid, 0.0))} for job in ordered[:limit]]


def _signal_tree(pid: int, sig: int) -> dict[str, Any]:
    """Send ``sig`` to ``pid`` and its descendants.  NEVER SIGKILL -- callers pass STOP/CONT only."""
    if sig not in (signal.SIGSTOP, signal.SIGCONT):
        raise ValueError(f"this watchdog only sends SIGSTOP/SIGCONT, never {sig}")
    targets = [pid]
    try:
        import psutil  # type: ignore

        targets.extend(child.pid for child in psutil.Process(pid).children(recursive=True))
    except Exception:
        pass
    delivered, failed = [], []
    # Children first for SIGSTOP (stop the leaves before the parent can spawn more) and parent
    # first for SIGCONT (so a resumed child is not immediately re-stopped by a stopped parent).
    order = list(reversed(targets)) if sig == signal.SIGSTOP else targets
    for target in order:
        try:
            os.kill(target, sig)
            delivered.append(target)
        except OSError as exc:
            failed.append({"pid": target, "error": f"{type(exc).__name__}: {exc}"})
    return {"signal": int(sig), "delivered": delivered, "failed": failed}


def process_state(pid: int) -> str | None:
    """POSIX process state letter (``T`` = stopped), or None when the pid is gone/unreadable."""
    try:
        import psutil  # type: ignore

        status = psutil.Process(pid).status()
        return {"stopped": "T", "running": "R", "sleeping": "S", "zombie": "Z"}.get(status, status[:1].upper())
    except Exception:
        pass
    try:
        completed = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = completed.stdout.strip()
    return text[:1].upper() if text else None


def orphaned_pauses(ledger: Path | None = None) -> list[dict[str, Any]]:
    """Pids whose most recent recorded action was a SIGSTOP with no later SIGCONT.

    This is the ledger read that would have caught the 00:34:20Z pause the moment a fresh instance
    started, instead of leaving a live cell in state T until a human noticed.
    """
    target = ALARM_LEDGER if ledger is None else Path(ledger)
    if not target.is_file():
        return []
    last: dict[int, dict[str, Any]] = {}
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        action = row.get("action")
        if not isinstance(action, Mapping):
            continue
        kind = str(action.get("kind", ""))
        pid = action.get("stop_pid")
        if not isinstance(pid, int) or not kind.startswith(("SIGSTOP", "SIGCONT")):
            continue
        last[pid] = {"kind": kind, "row": row}
    return [
        {"stop_pid": pid, "since_utc": entry["row"].get("observed_utc"), "action": entry["row"].get("action")}
        for pid, entry in sorted(last.items())
        if entry["kind"].startswith("SIGSTOP")
    ]


def reconcile_orphaned_pauses(
    ledger: Path | None = None,
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """At startup, RESUME anything a previous instance stopped and never released.

    A watchdog process is mortal -- it gets TERMed on a landing, it crashes, the box reboots -- and
    every one of those exits can strand a SIGSTOP. On 2026-09-05 the strand was a live training
    cell with 2.7 h of work in it, and only a human noticing recovered it. The ledger already knew;
    nothing read it.
    """
    resumed: list[dict[str, Any]] = []
    for orphan in orphaned_pauses(ledger):
        pid = int(orphan["stop_pid"])
        state = process_state(pid)
        record = {**orphan, "observed_state": state}
        if state is None:
            record["outcome"] = "GONE"
        elif state != "T":
            record["outcome"] = "ALREADY_RUNNING"
        elif dry_run:
            record["outcome"] = "WOULD_SIGCONT"
        else:
            record["outcome"] = "SIGCONT"
            record.update(_signal_tree(pid, signal.SIGCONT))
        resumed.append(record)
        if record["outcome"] in {"SIGCONT", "WOULD_SIGCONT"}:
            append_alarm(
                {
                    "schema": ALARM_SCHEMA,
                    "event": "confound_alarm",
                    "alarm": "orphaned_pause_reconciled",
                    "level": LEVEL_WARN,
                    "observed_utc": utc_text(),
                    "reasons": [
                        f"pid {pid} was SIGSTOPped at {orphan['since_utc']} and never resumed; "
                        "a previous watchdog instance died holding the pause"
                    ],
                    "action": {"kind": f"RECONCILE_{record['outcome']}", "stop_pid": pid, **record},
                    "report_only": bool(dry_run),
                    "score_claim": False,
                },
                ledger,
            )
    return resumed


def source_mtime() -> float:
    """mtime of this tool's own source -- the "fix in git is not the fix in RAM" guard."""
    try:
        return Path(__file__).resolve().stat().st_mtime
    except OSError:
        return 0.0


# ── alarms ──────────────────────────────────────────────────────────────────────────────────────


def append_alarm(row: Mapping[str, Any], ledger: Path | None = None) -> Path:
    target = ALARM_LEDGER if ledger is None else Path(ledger)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return target


def _push(title: str, body: str) -> None:
    """Best-effort desktop notification.  Fail-open: a guard must never die of its own alarm."""
    if os.environ.get("TAC_WATCHDOG_NO_PUSH"):
        return
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification {json.dumps(body)} with title {json.dumps(title)}'],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        pass


def emit_alarm(
    level: str,
    sample: PressureSample,
    reasons: Sequence[str],
    action: Mapping[str, Any] | None,
    *,
    ledger: Path | None = None,
    report_only: bool = False,
    growers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    row = {
        "top_rss_growers": list(growers or []),
        "schema": ALARM_SCHEMA,
        "event": "confound_alarm",
        "alarm": "memory_pressure",
        "level": level,
        "observed_utc": sample.observed_utc,
        "sample": sample.as_dict(),
        "reasons": list(reasons),
        "action": dict(action) if action else None,
        "report_only": bool(report_only),
        "score_claim": False,
        "derivation": "thresholds DERIVED from the 2026-09-04 near-OOM; see module docstring",
    }
    append_alarm(row, ledger)
    print(json.dumps(row, sort_keys=True, default=str), file=sys.stderr, flush=True)
    if level == LEVEL_CRITICAL:
        _push("pact memory CRITICAL", "; ".join(reasons)[:200])
    return row


# ── the loop ────────────────────────────────────────────────────────────────────────────────────


def watch(
    *,
    poll_s: float = DEFAULT_POLL_S,
    duration_s: float | None = None,
    report_only: bool = True,
    ledger: Path | None = None,
    clear_hold_s: float = CLEAR_HOLD_S,
    max_pause_s: float = MAX_PAUSE_S,
    reconcile: bool = True,
    exit_on_source_change: bool = True,
    sampler=sample_pressure,
    job_sampler=sample_governed_jobs,
    sleeper=time.sleep,
    now=time.monotonic,
) -> dict[str, Any]:
    """Poll and alarm.  Signals the biggest RSS grower ONLY when ``report_only`` is False.

    ``report_only`` defaults to **True** -- see the module docstring for why the actuator was
    demoted.  The startup reconciler runs regardless of the mode.
    """
    started = now()
    previous: PressureSample | None = None
    stopped: dict[str, Any] | None = None
    clear_since: float | None = None
    alarms: list[dict[str, Any]] = []
    polls = 0
    stopped_at: float | None = None
    # NOT gated on report_only: freeing a process a dead instance left stopped is a REPAIR. The
    # thing report-only refuses to do is STOP something, not rescue something.
    reconciled = reconcile_orphaned_pauses(ledger) if reconcile else []
    launch_mtime = source_mtime()
    retired_for_source_change = False
    # Rolling per-job footprints: the growth signal that identifies the ALLOCATOR (see
    # select_pressure_target). Sampled EVERY poll, including OK ones -- a window that only fills
    # once pressure is already high has no "before" to measure growth against.
    history: collections.deque[list[JobFootprint]] = collections.deque(maxlen=GROWTH_WINDOW_POLLS)

    while True:
        sample = sampler()
        polls += 1
        history.append(job_sampler())
        level, reasons = classify(sample, previous)
        previous = sample

        # THE BOUNDED PAUSE, checked on EVERY poll BEFORE the level branches. The old code could
        # only resume inside the OK branch, so a pause taken during pressure that the pause itself
        # sustained had no exit at all (MEASURED 2026-09-05: swap crossed WARN 12 s AFTER the
        # SIGSTOP and stayed there; the cell was stuck until a human intervened).
        if stopped is not None and stopped_at is not None and (now() - stopped_at) >= max_pause_s:
            held = now() - stopped_at
            resume = {} if report_only else _signal_tree(int(stopped["stop_pid"]), signal.SIGCONT)
            alarms.append(
                emit_alarm(
                    LEVEL_WARN,
                    sample,
                    [
                        f"MAX PAUSE {max_pause_s:.0f}s reached after {held:.0f}s and the clear hold was "
                        f"NOT met (level {level}); resuming anyway -- a SIGSTOP without a guaranteed "
                        "SIGCONT is a silent kill of sunk work"
                    ],
                    {"kind": "SIGCONT_MAX_PAUSE", **stopped, **resume, "hold_met": False, "paused_s": round(held, 1)},
                    ledger=ledger,
                    report_only=report_only,
                    growers=top_growers(list(history)),
                )
            )
            stopped, stopped_at = None, None

        if level == LEVEL_CRITICAL:
            clear_since = None
            if stopped is None:
                target = select_pressure_target(list(history))
                action: dict[str, Any] | None
                if target is None:
                    action = {
                        "kind": "NO_TARGET",
                        "note": (
                            "no governed job is eligible to pause (the oldest live cell is excluded "
                            "by the targeting rule and nothing else is running)"
                        ),
                    }
                elif report_only:
                    action = {"kind": "WOULD_SIGSTOP", **target}
                else:
                    action = {"kind": "SIGSTOP", **target, **_signal_tree(int(target["stop_pid"]), signal.SIGSTOP)}
                    stopped, stopped_at = target, now()
                alarms.append(
                    emit_alarm(
                        level, sample, reasons, action, ledger=ledger, report_only=report_only,
                        growers=top_growers(list(history)),
                    )
                )
            else:
                alarms.append(
                    emit_alarm(
                        level,
                        sample,
                        reasons,
                        {"kind": "ALREADY_STOPPED", **stopped},
                        ledger=ledger,
                        report_only=report_only,
                        growers=top_growers(list(history)),
                    )
                )
        elif level == LEVEL_WARN:
            clear_since = None
            # A WARN row NAMES the top RSS growers. The first eleven WARN rows this watchdog ever
            # emitted (2026-09-04 23:47-23:55Z) said only "compressor 38.47 GiB" -- true, and it
            # took a separate manual `ps` to learn that three non-cell ceiling jobs held 30.79 GiB
            # while the cell held 0.49. The cause belongs IN the alarm.
            alarms.append(
                emit_alarm(
                    level, sample, reasons, None, ledger=ledger, report_only=report_only,
                    growers=top_growers(list(history)),
                )
            )
        else:
            clear_since = now() if clear_since is None else clear_since
            if stopped is not None and (now() - clear_since) >= clear_hold_s:
                resume = _signal_tree(int(stopped["stop_pid"]), signal.SIGCONT)
                alarms.append(
                    emit_alarm(
                        LEVEL_OK,
                        sample,
                        [f"clear for {clear_hold_s:.0f} s; resuming"],
                        {"kind": "SIGCONT", **stopped, **resume},
                        ledger=ledger,
                        report_only=report_only,
                    )
                )
                stopped, stopped_at = None, None

        if duration_s is not None and (now() - started) >= duration_s:
            break
        # THE FIX IN GIT IS NOT THE FIX IN RAM. A running instance keeps the code it was launched
        # with; on 2026-09-05 a landed targeting fix sat in git while the OLD build kept stopping
        # the wrong process. A stale instance now retires itself (resuming anything it holds on the
        # way out) so a relaunch is the only action a landing needs.
        if exit_on_source_change and launch_mtime and source_mtime() != launch_mtime:
            retired_for_source_change = True
            break
        sleeper(poll_s)

    if stopped is not None and not report_only:
        # Never exit leaving a cell paused: a watchdog that stops a run and dies is worse than none.
        resume = _signal_tree(int(stopped["stop_pid"]), signal.SIGCONT)
        alarms.append(
            emit_alarm(
                LEVEL_OK,
                previous,
                ["watchdog exiting; resuming the paused cell"],
                {"kind": "SIGCONT_ON_EXIT", **stopped, **resume},
                ledger=ledger,
                report_only=report_only,
            )
        )
    return {
        "schema": "memory_pressure_watch_summary.v1",
        "finished_utc": utc_text(),
        "polls": polls,
        "alarms": len(alarms),
        "levels": sorted({row["level"] for row in alarms}),
        "report_only": bool(report_only),
        "poll_s": poll_s,
        "max_pause_s": max_pause_s,
        "reconciled_orphaned_pauses": reconciled,
        "retired_for_source_change": retired_for_source_change,
        "score_claim": False,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────


def _cmd_once(args: argparse.Namespace) -> int:
    sample = sample_pressure()
    level, reasons = classify(sample)
    payload = {
        "schema": "memory_pressure_once.v1",
        "level": level,
        "reasons": reasons,
        "sample": sample.as_dict(),
        "newest_training_cell": newest_training_cell(),
        "thresholds": thresholds_dict(sample.total_gib),
        "default_mode": "report_only (signalling requires --act)",
        "score_claim": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if level == LEVEL_OK else (1 if level == LEVEL_WARN else 2)


def _resolve_report_only(args: argparse.Namespace) -> bool:
    """Alarm-only unless ``--act`` is given; ``--report-only`` always wins if BOTH are passed.

    Fail-safe by construction: no combination of flags can turn signalling on by accident, and the
    old ``--report-only`` argv (which the live instance was launched with) keeps working unchanged.
    """
    if getattr(args, "report_only", False):
        return True
    return not getattr(args, "act", False)


def _cmd_run(args: argparse.Namespace) -> int:
    summary = watch(
        poll_s=args.poll_s,
        duration_s=args.duration_s,
        report_only=_resolve_report_only(args),
        ledger=args.ledger,
        clear_hold_s=args.clear_hold_s,
        max_pause_s=args.max_pause_s,
        reconcile=not args.no_reconcile,
        exit_on_source_change=not args.no_exit_on_source_change,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    """Resume anything a dead watchdog instance left SIGSTOPped. Safe to run any time."""
    resumed = reconcile_orphaned_pauses(args.ledger, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "schema": "memory_pressure_reconcile.v1",
                "checked_utc": utc_text(),
                "orphaned_pauses": resumed,
                "resumed": sum(1 for row in resumed if row["outcome"] == "SIGCONT"),
                "dry_run": bool(args.dry_run),
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def thresholds_dict(total_gib: float) -> dict[str, Any]:
    return {
        "poll_s": DEFAULT_POLL_S,
        "warn_compressor_gib": round(WARN_COMPRESSOR_FRACTION * total_gib, 2),
        "warn_swap_gib": WARN_SWAP_GIB,
        "warn_pressure_level": WARN_PRESSURE_LEVEL,
        "critical_compressor_gib": round(CRITICAL_COMPRESSOR_FRACTION * total_gib, 2),
        "critical_swap_gib": CRITICAL_SWAP_GIB,
        "critical_pressure_level": CRITICAL_PRESSURE_LEVEL,
        "critical_compressor_growth_gib_per_s": CRITICAL_COMPRESSOR_GROWTH_GIB_PER_S,
        "clear_hold_s": CLEAR_HOLD_S,
        "max_pause_s": MAX_PAUSE_S,
        "derived_from": "2026-09-04 near-OOM in .omx/state/memory_blackbox.jsonl (16 s runway)",
        "rejected_trigger": "free < 1 GiB -- 28.99% base rate over 13,235 samples that day",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    once = sub.add_parser("once", help="one reading (rc 0 ok / 1 warn / 2 critical)")
    once.add_argument("--json", action="store_true", help="accepted for symmetry; output is always JSON")
    once.set_defaults(func=_cmd_once)

    run = sub.add_parser("run", help="poll until --duration-s elapses (omit for forever)")
    run.add_argument("--poll-s", type=float, default=DEFAULT_POLL_S)
    run.add_argument("--duration-s", type=float, default=None)
    run.add_argument("--clear-hold-s", type=float, default=CLEAR_HOLD_S)
    run.add_argument(
        "--max-pause-s",
        type=float,
        default=MAX_PAUSE_S,
        help="hard ceiling on any pause; resume even if the clear hold is unmet (see MAX_PAUSE_S)",
    )
    run.add_argument(
        "--no-reconcile",
        action="store_true",
        help="skip the startup sweep for pauses a dead instance left behind (NOT recommended)",
    )
    run.add_argument(
        "--no-exit-on-source-change",
        action="store_true",
        help="keep running after this tool's source changes (the stale-instance hazard)",
    )
    run.add_argument("--ledger", type=Path, default=None)
    run.add_argument(
        "--act",
        action="store_true",
        help=(
            "OPT IN to signalling: SIGSTOP the biggest RSS grower on CRITICAL, bounded by "
            "--max-pause-s. OFF by default -- SIGSTOP hurt twice and helped zero times on "
            "2026-09-05 because pausing converts resident memory into swap instead of freeing it."
        ),
    )
    run.add_argument(
        "--report-only",
        action="store_true",
        help="the DEFAULT (alarm, name the actor, never signal); accepted explicitly and wins over --act",
    )
    run.set_defaults(func=_cmd_run)

    rec = sub.add_parser("reconcile", help="SIGCONT any pid a dead watchdog left stopped")
    rec.add_argument("--ledger", type=Path, default=None)
    rec.add_argument("--dry-run", action="store_true")
    rec.set_defaults(func=_cmd_reconcile)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
