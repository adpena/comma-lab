#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""System-aware dynamic memory GOVERNOR — max out 128 GB RAM safely, never crash the machine.

WHY (the 2026-07-02 machine crash, operator P0): the machine crashed from SYSTEM-WIDE memory
exhaustion. Our OOM protection was PER-PROCESS (``tools/safe_run.py --rss-mb 90000``) and the
launch-preflight (``tools/witness_memory_preflight.py``) projected a SINGLE run's peak — both BLIND to
the system TOTAL. Two (or three) jobs each individually under their per-process cap SUMMED to > 128 GB
(the R1 trainer ~67 GB + a byte-close/bsdtar/inflate job + the ev1 descent-probe + build/measurement
agents) -> macOS jetsam cascade -> hang. This module is the SYSTEM-aware layer that was missing.

THE THREE THINGS THIS ADDS (all SYSTEM-wide, job-type-BLIND — the ceiling is on the SYSTEM):

  1. **Adaptive ceiling** (``compute_adaptive_ceiling``): measure the OS + control-plane BASELINE
     (real vm_stat used MINUS the current RSS of our tracked jobs) and set
     ``training_budget = TOTAL_RAM - baseline - safety_margin`` where ``safety_margin`` is the
     TIER-SCALED DYNAMICAL FLOOR (BUILD #298; constants block + ``derive_safety_floor``):
     ``clamp(max(2, measured_cp + max(1, 0.05T), 0.08T), 2, 0.5T)`` — 10.24+ GiB @128 (never below
     the operator-policy 10 GiB), capped at 4.0 @8 (the old constant 8.0 equalled that ENTIRE box).
     On this 128 GB box baseline+margin ~= 16-20 GiB, so ~108-112 GiB is spendable — MUCH higher
     than the old blind 90 GB per-process cap, AND concurrent runs are bounded by the SAME budget.

  2. **Admission control** (``admission_decision`` — a HARD PREVENT gate): before a launch, REFUSE if
     ``current_system_used + (active jobs' remaining growth to peak) + this run's projected peak >
     adaptive_ceiling``. ``current_system_used`` is the REAL vm_stat used (counts EVERY consumer:
     OS + control-plane + training + byte-close + inflate + bsdtar + probes), so the gate is
     job-type-blind — exactly the SUM-over-128 gap that crashed us. This is a PREVENT gate, not a
     report: the launcher exits nonzero and starts NOTHING (override only via an operator-quoted
     rationale).

  3. **Dynamic throttle** (``decide_governor_action`` + ``select_throttle_target``): the black-box
     daemon (``tools/memory_blackbox.py``) evaluates pressure each sample; under WARN (available <
     tier-derived ``derived_warn_free_gib`` ~14.8 GiB @128 / 4.0 @8, or pressure=warn) it
     SIGSTOP-pauses the LOWEST-priority throttle-eligible job's FULL process tree (#246 fix; growth
     halts, fully reversible via SIGCONT on recovery); under CRITICAL (available < tier-derived
     ~8.4 GiB @128 / 3.0 @8 or pressure=critical) it pauses more aggressively to stay well under
     jetsam. It NEVER SIGSTOPs/kills
     the control plane (Claude / Codex / shells / the black-box / the guard) — the target selector
     reuses ``memory_guard``'s vendored control-plane exclusion gates. Killing (last-resort SIGTERM)
     stays with the existing ``memory_guard.py --watch``; the governor only PAUSES (reversible).

CONTROL-PLANE SAFETY: throttle candidates must pass ALL of ``memory_guard``'s control-plane
exclusions (not a control-plane app, no external control-plane lineage, not on the ssh/tmux/shell
denylist, pgid not protected) AND be their own process-group leader (a detached daemon, so the
pause scope is exactly the daemon subtree, never a shared interactive shell job). If ``memory_guard``
is unavailable, the governor throttles NOTHING and only ALERTs (fail-safe).

DETERMINISM: every decision function (``compute_adaptive_ceiling``, ``admission_decision``,
``project_system_used_after_launch``, ``decide_governor_action``, ``select_throttle_target``,
``classify_pressure``, ``classify_run_priority``) is PURE and unit-tested. The live readers
(``read_system_memory_snapshot``, ``list_tracked_jobs``) are thin I/O wrappers around vm_stat /
sysctl / ps / the durable-daemon registry.

macOS-only (vm_stat / sysctl). Sister of ``tools/memory_guard.py`` (whole-machine shed watchdog),
``tools/safe_run.py`` (per-arm RSS backstop), and ``tools/witness_memory_preflight.py`` (per-run
peak projection — the ``projected_new_gib`` this gate consumes).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

# Soft import of the vendored control-plane-safe machinery. If absent, the governor
# still does adaptive-ceiling + admission (pure math on live vm_stat) but throttles
# NOTHING (fail-safe: never risk the control plane without the exclusion gates).
try:
    import memory_guard as _mg
except Exception:  # pragma: no cover - defensive
    _mg = None

_DURABLE_DAEMON_REGISTRY = _REPO_ROOT / ".omx" / "state" / "durable_daemons.json"

# ── policy constants ────────────────────────────────────────────────────────────────────────
DEFAULT_SAFETY_MARGIN_FLOOR_GIB = 8.0    # LEGACY fixed-mode floor (128 GB-calibrated; kept for
#                                          --safety-floor-mode fixed + the canonical-equation mirror;
#                                          the DEFAULT is now the tier-scaled derived floor below)
DEFAULT_SAFETY_MARGIN_FRAC = 0.08        # static physics leg: 8% of total RAM (10.24 GiB @128 —
#                                          the operator-policy control-plane floor; 0.64 GiB @8)

# ── tier-scaled DYNAMICAL safety floor (BUILD #298; replaces the hardcoded 8 GiB floor) ──────────
# WHY (tertiary edge-sweep spec §2, 2026-07-04): the 8.0 GiB constant was 128 GB-calibrated; on the
# 8 GiB M1 it equals the ENTIRE box -> adaptive_ceiling 0.0, training_budget -5.99, the admission
# gate refuses EVERYTHING, and no CLI/env override existed. The floor is now DERIVED (see
# ``derive_safety_floor``) — every term physical, every number traceable:
#
#   floor = clamp( max( ABS_MIN,
#                       measured_cp_rss + cp_headroom(T),   # DYNAMICAL leg (follows the live
#                                                           #   control plane; None when unmeasured)
#                       0.08 * T ),                         # STATIC physics leg (measurement-free)
#                  ABS_MIN, cap_frac * T )                  # a floor may NEVER eat the box
#
# Term derivations (from the measured rows in n205_memory_behavior_mine_20260704.md +
# tertiary_edge_sweep_spec_20260704.md + operator_memory_policy_sole_workload_* legs):
#   * ABS_MIN 2.0 GiB — jetsam-avoidance minimum on ANY tier: the M1 ran its 3.69 GiB smoke without
#     swap-thrash while conservative available sat ~1.9-2.5 GiB; below ~2 GiB reclaimable macOS
#     pressure escalates. Never reserve less, even on the smallest box.
#   * cp_headroom = max(1.0, 0.05*T) — control-plane SPIKE headroom: @128 -> 6.4 GiB (2-3 concurrent
#     build/review agents at 1-3 GiB each — the 2026-07-02 crash's unaccounted contributors were
#     exactly control-plane-adjacent agents); @8 -> 1.0 GiB (single-agent burst; same order as the
#     tier's measured verdict step +1.26 GiB — the sub-sample-interval transient scale on that box).
#   * measured_cp_rss + cp_headroom — the DYNAMICAL leg: when the control plane is measured to be
#     BIGGER than the static assumption, the floor follows it up (reserving growth room on the same
#     order as its current footprint — agents spawn in bursts proportional to current activity).
#     Deliberately conservative on top of the used-based admission arithmetic; bounded by cap_frac.
#   * 0.08*T static leg — protects when measurement is unavailable; reproduces the operator-policy
#     >=10 GiB control-plane floor on 128 GiB (10.24) and scales to 0.64 @8 / 15.36 @192.
#   * cap_frac 0.5 — a floor that eats more than half the box is the 8-GiB-tier pathology being
#     fixed; structurally impossible after this (applies to EVERY mode, overrides included). On the
#     M1 the cap binds at 4.0 GiB, reproducing the tier's MEASURED safe envelope (~4.0-4.4 GiB,
#     tertiary spec §2; the 3.69 GiB smoke peak fits inside it).
# On this 128 GiB box the derived floor is >= 10.24 GiB in EVERY scenario (static leg is the
# minimum; the measured leg only raises it) — backward-compatible with the >=10 GiB operator policy.
ABS_MIN_SAFETY_FLOOR_GIB = 2.0
DEFAULT_CP_HEADROOM_MIN_GIB = 1.0
DEFAULT_CP_HEADROOM_FRAC = 0.05
DEFAULT_FLOOR_CAP_FRAC = 0.5
SAFETY_FLOOR_ENV = "TAC_GOV_SAFETY_FLOOR_GIB"    # env override (GiB); CLI --safety-floor-gib wins
FLOOR_MODE_DERIVED = "derived"
FLOOR_MODE_FIXED = "fixed"
FLOOR_MODES = (FLOOR_MODE_DERIVED, FLOOR_MODE_FIXED)
# Hysteresis: the floor RISES instantly (protection never lags a control-plane burst) and decays at
# most this many GiB per tick (admission verdicts don't flap on an oscillating control plane).
DEFAULT_FLOOR_MAX_DECAY_GIB_PER_TICK = 0.25

# UNITS (measured, load-bearing — #205 memory mine §1, n205_memory_behavior_mine_20260704.md):
# memory_guard.group_rss_gb returns sum(rss_kb)/1e6 = units of 1e6 KiB ~= 0.95367 TRUE GiB despite
# its ``_gb`` name (cross-validated vs ps: 65.27 units == 62.25 true GiB). Post GB-F1 (throughput
# review 2026-07-04) the conversion is applied EXACTLY ONCE, at the tracked-job READ boundary in
# ``list_tracked_jobs`` — so TrackedJob.current_rss_gib, growth_headroom_gib, the admission
# baseline, the band comparison, and the blackbox rows are ALL true GiB downstream. (Pre-fix the
# admission path mixed units: growth_headroom = peak[true] - current[units] under-counted remaining
# growth by a MEASURED 2.63 GiB — anti-conservative.) The underlying memory_guard rename is still
# owned upstream; until then this constant is the single boundary converter.
TRACKED_RSS_UNITS_TO_GIB = 1e6 * 1024 / (1024.0 ** 3)   # = 0.95367431640625
DEFAULT_WARN_FREE_GIB = 15.0             # available (free+inactive) below this => WARN
DEFAULT_CRITICAL_FREE_GIB = 8.0          # available below this => CRITICAL (well above jetsam)
PRESSURE_NORMAL = 1                       # macOS kern.memorystatus_vm_pressure_level values
PRESSURE_WARN = 2
PRESSURE_CRITICAL = 4
DEFAULT_WARN_CONSECUTIVE = 3              # debounce: sustained WARN polls before a pause
DEFAULT_CRITICAL_CONSECUTIVE = 2         # act faster on CRITICAL (fewer sustained polls)
_FLEET_TOTAL_RAM_FALLBACK_GIB = 128.0    # M5 Max last-resort when sysctl fails

# ── accounting-trust constants (the "why trust this over eyeballing vm_stat" answer) ─────────────
# The accounting is PURE + UNIT-TESTED against fixed captured snapshots, CROSS-VALIDATED against a 2nd
# independent kernel source, and CLOSURE-checked (partition sum ~= physical total). If any check fails
# -> FAIL SAFE (treat memory as scarcer; refuse admission) + log the discrepancy. Never trust a single
# unvalidated number.
CLOSURE_TOL_GIB = 4.0            # |partition_sum - total| above this => parse error => fail safe
                                 # (the legit unaccounted gap is ~1.3 GiB on this box; a wrong page-size
                                 #  parse would blow this out by ~100 GiB, so 4 GiB catches real bugs)
XVALIDATE_TOL_GIB = 3.0          # cross-source disagreement above this => fail safe
FREE_PAGE_XCHECK_TOL_GIB = 2.0   # |vm_stat free - sysctl vm.page_free_count| (in GiB) tolerance
_GIB = 1024.0 ** 3

# ── admission enforcement mode (trust requirement #4: ADVISORY until independent review) ─────────
# The admission gate is DESIGNED as a HARD PREVENT gate, but it ships ADVISORY (logs what it WOULD
# refuse, does NOT block) until an independent adversarial review (reviewer != author) signs off on the
# accounting + control-plane allowlist + fail-safe paths. Flip to ENFORCE by setting the env var (a
# one-line, reviewable flip — the canonical warn-only -> strict-flip pattern).
ADMISSION_ENFORCE_ENV = "TAC_ADMISSION_ENFORCE"
# Durable, project-scoped strict-flip marker (gitignored per-machine state). Presence with a truthy
# first line arms ENFORCE for ALL launches — this session, future sessions, AND subagents (they all
# call admission_enforcing()) — surviving the shell-env reset between Bash calls. Reversible: delete
# the file. The env var remains an equivalent per-invocation override. Absolute path (cwd-independent).
_ADMISSION_ENFORCE_FLAG = _TOOLS.parent / ".omx" / "state" / "admission_enforce.flag"


def admission_enforcing() -> bool:
    """True iff the admission gate is in ENFORCE mode (blocks launches). Default FALSE (ADVISORY —
    logs what it WOULD refuse) until independent review arms enforce via the env var
    ``TAC_ADMISSION_ENFORCE=1`` OR the durable flag file ``.omx/state/admission_enforce.flag``
    (truthy first line). Only ADDS refusals the gate already computes (enforce ⊆ advisory)."""
    _truthy = {"1", "true", "yes", "on"}
    if os.environ.get(ADMISSION_ENFORCE_ENV, "0").strip().lower() in _truthy:
        return True
    try:
        if _ADMISSION_ENFORCE_FLAG.is_file():
            first = (_ADMISSION_ENFORCE_FLAG.read_text().splitlines() or [""])[0].strip().lower()
            return first in _truthy
    except OSError:
        return False
    return False

# BROAD "our heavy jobs" allowlist for THROTTLE candidate discovery (beyond the registry).
# The crash summed HETEROGENEOUS jobs, so the throttle must be able to pause a byte-close / inflate /
# bsdtar / probe too — not only a registered training run. Control-plane exclusion + own-group-leader
# still gate every candidate (so an interactive shell's `tar` is never touched).
OUR_JOBS_PATTERN = (
    r"train_witness_realized_through_R"
    r"|train_levelset_witness"
    r"|train_witness"
    r"|witness_capstone"
    r"|train_substrate_"
    r"|train_renderer"
    r"|launch_split_by_head_basin"
    r"|levelset_byte_close"
    r"|byte_close"
    r"|inflate\.py"
    r"|contest_auth_eval"
    r"|evaluate\.py"
    r"|verdict_mem_microprobe"
    r"|descent_probe"
    r"|scorer_response"
)


# ─────────────────────────── system memory snapshot ───────────────────────────
@dataclass(frozen=True)
class MemoryAccounting:
    """Reconciled, VALIDATED memory accounting (pure; unit-tested against fixed snapshots)."""
    total_gib: float
    available_gib: float          # value to USE for decisions (conservative, fail-safe-adjusted)
    available_primary_gib: float  # (free + inactive) from vm_stat page counts
    used_gib: float               # total - available_gib (THE TRUTH: OS + control-plane + ALL jobs)
    free_gib: float               # strict Pages free
    wired_gib: float
    compressor_gib: float
    closure_gib: float            # |partition_sum - total| (should be ~1-2 GiB sampling noise)
    closure_ok: bool
    cross_validated: bool
    discrepancy_gib: float        # worst cross-source disagreement observed
    fail_safe: bool               # True => a validation check failed => treat memory as scarcer / refuse
    validation_notes: tuple[str, ...]


@dataclass(frozen=True)
class SystemMemorySnapshot:
    total_gib: float
    available_gib: float   # free + inactive (conservative reclaimable; fail-safe-adjusted)
    used_gib: float        # total - available (THE TRUTH: OS + control-plane + ALL our jobs)
    free_gib: float        # strict Pages free
    wired_gib: float
    compressor_gib: float
    swap_used_gib: float
    pressure_level: int    # 1 normal / 2 warn / 4 critical (0 = unknown)
    load1: float
    load5: float
    load15: float
    # ── accounting-trust fields (why this is trustworthy vs eyeballing vm_stat by hand) ──
    available_primary_gib: float = 0.0
    closure_gib: float = 0.0
    closure_ok: bool = True
    cross_validated: bool = True
    discrepancy_gib: float = 0.0
    fail_safe: bool = False
    validation_notes: tuple = ()

    def to_json(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def total_ram_gib() -> float:
    """TOTAL physical RAM (GiB) from ``sysctl hw.memsize``; fallback to os.sysconf / fleet default."""
    raw = _sysctl("hw.memsize")
    if raw and raw.isdigit():
        return int(raw) / (1024.0 ** 3)
    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024.0 ** 3)
    except (ValueError, OSError):
        return _FLEET_TOTAL_RAM_FALLBACK_GIB


def read_memory_pressure_level() -> int:
    """macOS memory-pressure level: 1 normal / 2 warn / 4 critical (0 = unknown/unavailable)."""
    raw = _sysctl("kern.memorystatus_vm_pressure_level")
    if raw and raw.strip().lstrip("-").isdigit():
        return int(raw.strip())
    return 0


def read_swap_used_gib() -> float:
    """Used swap (GiB) parsed from ``sysctl vm.swapusage`` (``used = 512.25M``)."""
    raw = _sysctl("vm.swapusage")
    if not raw:
        return 0.0
    m = re.search(r"used\s*=\s*([0-9.]+)([KMG]?)", raw)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = m.group(2)
    mult = {"K": 1 / 1024.0 / 1024.0, "M": 1 / 1024.0, "G": 1.0, "": 1 / 1024.0}.get(unit, 1 / 1024.0)
    return val * mult  # -> GiB


def page_size_bytes() -> int:
    """Kernel page size from ``sysctl hw.pagesize`` (16384 on M-series). NEVER hardcoded blindly."""
    raw = _sysctl("hw.pagesize")
    if raw and raw.strip().isdigit():
        return int(raw.strip())
    # last-resort: parse the page size vm_stat prints in its header, else the M-series default.
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"page size of (\d+)", out)
        if m:
            return int(m.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    return 16384


def _read_vm_stat_counters() -> dict[str, int]:
    """Parse the vm_stat page counts we need for accounting + closure. Programmatic-field parse (each
    field grabbed by its exact label), NOT an ad-hoc guess — unit-tested via ``reconcile_memory_accounting``."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""

    def _grab(label: str) -> int:
        for line in out.splitlines():
            if line.startswith(label):
                mm = re.search(r"(\d+)", line.split(":", 1)[1])
                return int(mm.group(1)) if mm else 0
        return 0

    return {
        "free": _grab("Pages free"),
        "active": _grab("Pages active"),
        "inactive": _grab("Pages inactive"),
        "speculative": _grab("Pages speculative"),
        "throttled": _grab("Pages throttled"),
        "wired": _grab("Pages wired down"),
        "compressor": _grab("Pages occupied by compressor"),
    }


def _read_sysctl_free_pages() -> int | None:
    raw = _sysctl("vm.page_free_count")
    return int(raw) if raw and raw.strip().isdigit() else None


def _read_memory_pressure_total_bytes() -> int | None:
    """The physical total macOS ``memory_pressure`` reports in its header (independent of hw.memsize).
    Used to cross-validate page_size * page_total == hw.memsize."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=6).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"The system has (\d+)", out)
    return int(m.group(1)) if m else None


def _read_psutil_available_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def reconcile_memory_accounting(
    *,
    page_size: int,
    total_bytes: int,
    vm_free_pages: int,
    vm_active_pages: int,
    vm_inactive_pages: int,
    vm_speculative_pages: int,
    vm_wired_pages: int,
    vm_compressor_pages: int,
    vm_throttled_pages: int = 0,
    sysctl_free_pages: int | None = None,
    mempressure_total_bytes: int | None = None,
    psutil_available_bytes: int | None = None,
    closure_tol_gib: float = CLOSURE_TOL_GIB,
    xvalidate_tol_gib: float = XVALIDATE_TOL_GIB,
    free_page_tol_gib: float = FREE_PAGE_XCHECK_TOL_GIB,
) -> MemoryAccounting:
    """Reconcile + VALIDATE memory accounting from raw kernel counters. PURE — unit-tested against
    fixed captured snapshots so a parse bug (wrong page size / wrong field / CPU-vs-memory confusion)
    fails a test, never ships.

    Validations (each catches a parse bug; each measures the SAME quantity two ways so a legit
    definitional difference never falsely trips):
      * CLOSURE: (free+active+inactive+speculative+throttled+wired+compressor) ~= total (tol).
      * FREE cross-check: vm_stat free pages ~= sysctl vm.page_free_count (tol) — same quantity, 2 srcs.
      * TOTAL cross-check: page_size * memory_pressure page-total ~= hw.memsize (validates page_size).
      * OVERCOUNT cross-check (one-sided): our CONSERVATIVE available must not EXCEED psutil's generous
        available (if psutil present) — a parse overcount would.
    Any failure => ``fail_safe=True`` (decisions treat memory as scarcer / refuse admission)."""
    ps = int(page_size)
    total_gib = total_bytes / _GIB
    free_gib = vm_free_pages * ps / _GIB
    inactive_gib = vm_inactive_pages * ps / _GIB
    wired_gib = vm_wired_pages * ps / _GIB
    compressor_gib = vm_compressor_pages * ps / _GIB
    available_primary_gib = min(free_gib + inactive_gib, total_gib)

    partition_pages = (vm_free_pages + vm_active_pages + vm_inactive_pages + vm_speculative_pages
                       + vm_throttled_pages + vm_wired_pages + vm_compressor_pages)
    partition_gib = partition_pages * ps / _GIB
    closure_gib = abs(total_gib - partition_gib)
    closure_ok = closure_gib <= closure_tol_gib

    notes: list[str] = []
    cross_validated = True
    discrepancy_gib = 0.0

    if not closure_ok:
        cross_validated = False
        notes.append(f"closure FAIL: |total {total_gib:.2f} - partition {partition_gib:.2f}| = "
                     f"{closure_gib:.2f} GiB > {closure_tol_gib} GiB (parse error suspected)")
        discrepancy_gib = max(discrepancy_gib, closure_gib)

    if sysctl_free_pages is not None:
        d = abs(vm_free_pages - sysctl_free_pages) * ps / _GIB
        discrepancy_gib = max(discrepancy_gib, d)
        if d > free_page_tol_gib:
            cross_validated = False
            notes.append(f"free-page cross-check FAIL: vm_stat {vm_free_pages} vs sysctl "
                         f"{sysctl_free_pages} = {d:.2f} GiB > {free_page_tol_gib} GiB")

    if mempressure_total_bytes is not None:
        d = abs(mempressure_total_bytes - total_bytes) / _GIB
        discrepancy_gib = max(discrepancy_gib, d)
        if d > xvalidate_tol_gib:
            cross_validated = False
            notes.append(f"total cross-check FAIL: hw.memsize {total_bytes} vs memory_pressure "
                         f"{mempressure_total_bytes} = {d:.2f} GiB (page_size parse suspected)")

    if psutil_available_bytes is not None:
        psutil_avail_gib = psutil_available_bytes / _GIB
        # one-sided: our CONSERVATIVE available must not exceed psutil's GENEROUS available (+ tol).
        overcount = available_primary_gib - psutil_avail_gib
        if overcount > xvalidate_tol_gib:
            cross_validated = False
            discrepancy_gib = max(discrepancy_gib, overcount)
            notes.append(f"overcount cross-check FAIL: conservative available {available_primary_gib:.2f} "
                         f"> psutil available {psutil_avail_gib:.2f} + {xvalidate_tol_gib} (overcount bug)")

    fail_safe = (not closure_ok) or (not cross_validated)
    # Fail-safe => treat memory as SCARCER: subtract the discrepancy from available so admission refuses.
    available_gib = available_primary_gib
    if fail_safe:
        available_gib = max(0.0, available_primary_gib - discrepancy_gib)
        notes.append(f"FAIL-SAFE: available reduced {available_primary_gib:.2f} -> {available_gib:.2f} GiB")
    used_gib = max(0.0, total_gib - available_gib)

    return MemoryAccounting(
        total_gib=total_gib, available_gib=available_gib, available_primary_gib=available_primary_gib,
        used_gib=used_gib, free_gib=free_gib, wired_gib=wired_gib, compressor_gib=compressor_gib,
        closure_gib=closure_gib, closure_ok=closure_ok, cross_validated=cross_validated,
        discrepancy_gib=discrepancy_gib, fail_safe=fail_safe, validation_notes=tuple(notes),
    )


def read_system_memory_snapshot() -> SystemMemorySnapshot:
    """Live SYSTEM-wide memory snapshot: programmatic kernel counters (vm_stat page counts + sysctl
    page size + hw.memsize) reconciled + VALIDATED (closure + cross-validation) via the pure
    ``reconcile_memory_accounting``, then annotated with pressure / swap / load. ``used_gib`` is the
    TRUTH the admission gate anchors on (job-type-blind, fail-safe-adjusted)."""
    ps = page_size_bytes()
    counters = _read_vm_stat_counters()
    total_bytes = int(round(total_ram_gib() * _GIB))
    acct = reconcile_memory_accounting(
        page_size=ps, total_bytes=total_bytes,
        vm_free_pages=counters["free"], vm_active_pages=counters["active"],
        vm_inactive_pages=counters["inactive"], vm_speculative_pages=counters["speculative"],
        vm_wired_pages=counters["wired"], vm_compressor_pages=counters["compressor"],
        vm_throttled_pages=counters.get("throttled", 0),
        sysctl_free_pages=_read_sysctl_free_pages(),
        mempressure_total_bytes=_read_memory_pressure_total_bytes(),
        psutil_available_bytes=_read_psutil_available_bytes(),
    )
    try:
        l1, l5, l15 = os.getloadavg()
    except (OSError, ValueError):
        l1 = l5 = l15 = 0.0
    if acct.fail_safe:
        _log_fail_safe(acct)
    return SystemMemorySnapshot(
        total_gib=acct.total_gib, available_gib=acct.available_gib, used_gib=acct.used_gib,
        free_gib=acct.free_gib, wired_gib=acct.wired_gib, compressor_gib=acct.compressor_gib,
        swap_used_gib=read_swap_used_gib(), pressure_level=read_memory_pressure_level(),
        load1=l1, load5=l5, load15=l15, available_primary_gib=acct.available_primary_gib,
        closure_gib=acct.closure_gib, closure_ok=acct.closure_ok, cross_validated=acct.cross_validated,
        discrepancy_gib=acct.discrepancy_gib, fail_safe=acct.fail_safe,
        validation_notes=acct.validation_notes,
    )


def _log_fail_safe(acct: MemoryAccounting) -> None:
    """Loudly log an accounting-validation failure (fail-safe path) — never silent."""
    msg = ("[system-governor] ACCOUNTING FAIL-SAFE: " + "; ".join(acct.validation_notes))
    print(msg, file=sys.stderr)
    try:
        log = _REPO_ROOT / ".omx" / "state" / "memory_governor.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            import datetime as _d
            f.write(f"{_d.datetime.now(_d.UTC).isoformat()} {msg}\n")
    except OSError:
        pass


# ─────────────────────────── tier-scaled dynamical safety floor (pure) ───────────────────────────
def cp_headroom_gib(
    total_gib: float,
    *,
    min_gib: float = DEFAULT_CP_HEADROOM_MIN_GIB,
    frac: float = DEFAULT_CP_HEADROOM_FRAC,
) -> float:
    """Control-plane SPIKE headroom = max(1 GiB, 5% of RAM). @128 -> 6.4; @8 -> 1.0. PURE.
    Derivation: see the tier-scaled-floor constants block (agent-burst scale per tier, traceable to
    the #205 mine + tertiary smoke measurements)."""
    return max(float(min_gib), float(frac) * float(total_gib))


def measured_control_plane_rss_gib(*, used_gib: float, tracked_current_gib: float) -> float:
    """The DYNAMICAL floor leg's input: live NON-workload footprint = system used minus the sum of
    our tracked (governed) jobs' current RSS. Both inputs are TRUE GiB post GB-F1 (the units
    conversion happens once, at the ``list_tracked_jobs`` read boundary). Clamped >= 0. PURE."""
    return max(0.0, float(used_gib) - float(tracked_current_gib))


@dataclass(frozen=True)
class SafetyFloorDecomposition:
    """Full decomposition of one derived-floor evaluation — every leg's value + which leg won —
    emitted into the governor/blackbox JSONL rows each tick (max observability)."""
    floor_gib: float                       # the clamped, applied floor
    winning_leg: str                       # abs_min | measured_cp | static_frac | fixed_legacy | override
    total_gib: float
    abs_min_gib: float
    measured_cp_rss_gib: float | None      # true GiB; None = measurement unavailable
    cp_headroom_gib: float
    measured_leg_gib: float | None         # measured_cp_rss + cp_headroom (None when unmeasured)
    static_leg_gib: float                  # DEFAULT_SAFETY_MARGIN_FRAC * total
    cap_gib: float                         # max(abs_min, cap_frac * total) — the never-eat-the-box bound
    raw_floor_gib: float                   # pre-clamp value of the winning leg / override
    clamped: bool                          # True => raw was clamped into [abs_min, cap]
    mode: str                              # derived | fixed
    override_gib: float | None             # explicit CLI/env override (wins when present)

    def to_json(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def derive_safety_floor(
    *,
    total_gib: float,
    measured_cp_rss_gib: float | None = None,
    mode: str = FLOOR_MODE_DERIVED,
    override_gib: float | None = None,
    abs_min_gib: float = ABS_MIN_SAFETY_FLOOR_GIB,
    cp_headroom_min_gib: float = DEFAULT_CP_HEADROOM_MIN_GIB,
    cp_headroom_frac: float = DEFAULT_CP_HEADROOM_FRAC,
    static_frac: float = DEFAULT_SAFETY_MARGIN_FRAC,
    cap_frac: float = DEFAULT_FLOOR_CAP_FRAC,
    legacy_floor_gib: float = DEFAULT_SAFETY_MARGIN_FLOOR_GIB,
    log_fn=None,
) -> SafetyFloorDecomposition:
    """The tier-scaled DYNAMICAL safety floor (BUILD #298). PURE (``log_fn`` optional loud-clamp
    sink; pass e.g. a stderr printer from live callsites).

        floor = clamp( max(ABS_MIN, measured_cp + cp_headroom(T), static_frac*T), ABS_MIN, cap_frac*T )

    Precedence: an explicit ``override_gib`` (CLI/env) WINS, clamped to the same [ABS_MIN, cap]
    range with a loud log when clamped. ``mode='fixed'`` reproduces the legacy max(8, 0.08T)
    formula — still cap-clamped, so even fixed mode can no longer eat an 8 GiB box.
    Derivations for every constant: the tier-scaled-floor constants block above."""
    total = float(total_gib)
    abs_min = float(abs_min_gib)
    cap = max(abs_min, float(cap_frac) * total)
    ch = cp_headroom_gib(total, min_gib=cp_headroom_min_gib, frac=cp_headroom_frac)
    static_leg = float(static_frac) * total
    cp = None if measured_cp_rss_gib is None else max(0.0, float(measured_cp_rss_gib))
    measured_leg = None if cp is None else cp + ch

    if override_gib is not None:
        raw = float(override_gib)
        winning = "override"
    elif mode == FLOOR_MODE_FIXED:
        raw = max(float(legacy_floor_gib), static_leg)
        winning = "fixed_legacy"
    else:
        raw = max(abs_min, static_leg, measured_leg if measured_leg is not None else 0.0)
        if measured_leg is not None and raw == measured_leg:
            winning = "measured_cp"
        elif raw == static_leg:
            winning = "static_frac"
        else:
            winning = "abs_min"

    floor = min(max(raw, abs_min), cap)
    clamped = abs(floor - raw) > 1e-12
    if clamped and log_fn is not None:
        log_fn(f"[system-governor] SAFETY-FLOOR CLAMP: {winning} value {raw:.2f} GiB clamped to "
               f"{floor:.2f} GiB (allowed range [{abs_min:.2f}, {cap:.2f}] on a {total:.0f} GiB box "
               f"— a floor may never eat the box)")
    return SafetyFloorDecomposition(
        floor_gib=floor, winning_leg=winning, total_gib=total, abs_min_gib=abs_min,
        measured_cp_rss_gib=cp, cp_headroom_gib=ch, measured_leg_gib=measured_leg,
        static_leg_gib=static_leg, cap_gib=cap, raw_floor_gib=raw, clamped=clamped,
        mode=(FLOOR_MODE_FIXED if mode == FLOOR_MODE_FIXED else FLOOR_MODE_DERIVED),
        override_gib=(None if override_gib is None else float(override_gib)),
    )


def safety_floor_env_override_gib(env: Mapping[str, str] | None = None) -> float | None:
    """Parse the ``TAC_GOV_SAFETY_FLOOR_GIB`` env override (None when unset/blank/non-numeric —
    a bad value is LOUDLY ignored, never silently misapplied)."""
    raw = (env if env is not None else os.environ).get(SAFETY_FLOOR_ENV, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print(f"[system-governor] WARNING: ignoring non-numeric {SAFETY_FLOOR_ENV}={raw!r}",
              file=sys.stderr)
        return None


def resolve_floor_override_gib(cli_value: float | None,
                               env: Mapping[str, str] | None = None) -> float | None:
    """Override precedence: explicit CLI flag > ``TAC_GOV_SAFETY_FLOOR_GIB`` env > None (derive).
    (Clamping to [ABS_MIN, cap] happens inside ``derive_safety_floor`` with a loud log.)"""
    return cli_value if cli_value is not None else safety_floor_env_override_gib(env)


class SafetyFloorSmoother:
    """Asymmetric hysteresis for the per-tick derived floor: RISES INSTANTLY (protection never lags
    a control-plane burst), decays at most ``max_decay_gib_per_tick`` per update — so admission
    verdicts do not flap when the control-plane RSS oscillates between samples. Deterministic;
    one instance per sampling loop (one-shot CLI reads use the raw floor)."""

    def __init__(self, max_decay_gib_per_tick: float = DEFAULT_FLOOR_MAX_DECAY_GIB_PER_TICK):
        self.max_decay_gib_per_tick = float(max_decay_gib_per_tick)
        self._prev: float | None = None

    def update(self, raw_floor_gib: float) -> float:
        raw = float(raw_floor_gib)
        if self._prev is None or raw >= self._prev:
            self._prev = raw
        else:
            self._prev = max(raw, self._prev - self.max_decay_gib_per_tick)
        return self._prev


def derived_critical_free_gib(total_gib: float) -> float:
    """Tier-scaled CRITICAL available-threshold = ABS_MIN + cp_headroom(T): fire while there is
    still room to absorb one control-plane burst above the jetsam minimum. @128 -> 8.4 GiB
    (strictly MORE protective than the legacy 8.0); @8 -> 3.0 GiB (the legacy constant was above
    everything an 8 GiB box can ever free). PURE."""
    return ABS_MIN_SAFETY_FLOOR_GIB + cp_headroom_gib(total_gib)


def derived_warn_free_gib(total_gib: float) -> float:
    """Tier-scaled WARN available-threshold = critical + one more cp_headroom. @128 -> 14.8 GiB
    (within 1.4% of the legacy 15.0 — warn-pauses fire one 0.2 GiB step later; the jetsam-guarding
    CRITICAL rung above got strictly earlier); @8 -> 4.0 GiB. PURE."""
    return derived_critical_free_gib(total_gib) + cp_headroom_gib(total_gib)


# ─────────────────────────── adaptive ceiling (pure) ───────────────────────────
def compute_safety_margin_gib(
    total_gib: float,
    *,
    floor_gib: float | None = None,
    frac: float | None = None,
    measured_cp_rss_gib: float | None = None,
    override_gib: float | None = None,
    mode: str = FLOOR_MODE_DERIVED,
) -> float:
    """Adaptive safety margin — now the tier-scaled DERIVED floor by default (BUILD #298).

    Backward-compat wrapper around ``derive_safety_floor``: passing the legacy ``floor_gib``/``frac``
    kwargs reproduces the old max(floor, frac*T) formula (cap-clamped — even legacy arithmetic can
    no longer eat an 8 GiB box). With no measured control plane the derived value @128 is 10.24 GiB
    (identical to legacy); @64 -> 5.12; @8 -> 2.0 (vs the legacy 8.0 = the whole box)."""
    if floor_gib is not None or frac is not None:
        legacy = max(
            float(floor_gib if floor_gib is not None else DEFAULT_SAFETY_MARGIN_FLOOR_GIB),
            float(frac if frac is not None else DEFAULT_SAFETY_MARGIN_FRAC) * float(total_gib),
        )
        cap = max(ABS_MIN_SAFETY_FLOOR_GIB, DEFAULT_FLOOR_CAP_FRAC * float(total_gib))
        return min(legacy, cap)
    return derive_safety_floor(
        total_gib=total_gib, measured_cp_rss_gib=measured_cp_rss_gib,
        override_gib=(override_gib if override_gib is not None else safety_floor_env_override_gib()),
        mode=mode,
    ).floor_gib


@dataclass(frozen=True)
class AdaptiveCeiling:
    total_gib: float
    used_gib: float                # current SYSTEM-wide used (truth)
    tracked_current_gib: float     # sum of current RSS of our tracked jobs (TRUE GiB post GB-F1)
    baseline_gib: float            # OS + control-plane = used - tracked_current (clamped >= 0)
    safety_margin_gib: float
    adaptive_ceiling_gib: float    # total - safety_margin (max system-used we tolerate)
    training_budget_gib: float     # ceiling - baseline (total peak all our jobs may collectively use)
    floor_decomposition: dict | None = None   # SafetyFloorDecomposition.to_json() when derived

    def to_json(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def compute_adaptive_ceiling(
    *,
    total_gib: float,
    used_gib: float,
    tracked_current_gib: float,
    safety_margin_gib: float | None = None,
    floor: SafetyFloorDecomposition | None = None,
    floor_override_gib: float | None = None,
    floor_mode: str = FLOOR_MODE_DERIVED,
) -> AdaptiveCeiling:
    """Compute the adaptive ceiling + training budget. PURE (env consulted only on the default
    path, for the ``TAC_GOV_SAFETY_FLOOR_GIB`` override).

    ``training_budget = (total - margin) - baseline`` where ``baseline = used - tracked_current``
    (the OS + control-plane footprint). The margin is now the tier-scaled DERIVED floor by default
    (BUILD #298): ``baseline`` doubles as the derived floor's measured-control-plane input, so the
    floor FOLLOWS the live control plane each tick. Precedence: explicit ``safety_margin_gib``
    (e.g. a smoothed floor from the sampling loop) > pre-computed ``floor`` decomposition >
    derive-from-inputs (+ env override). An explicit margin is still cap-clamped — a floor may
    never eat the box (the 8 GiB-tier pathology this build extincts)."""
    cap = max(ABS_MIN_SAFETY_FLOOR_GIB, DEFAULT_FLOOR_CAP_FRAC * float(total_gib))
    baseline = max(0.0, float(used_gib) - float(tracked_current_gib))
    if safety_margin_gib is not None:
        margin = min(float(safety_margin_gib), cap)
    elif floor is not None:
        margin = min(floor.floor_gib, cap)
    else:
        floor = derive_safety_floor(
            total_gib=total_gib,
            measured_cp_rss_gib=baseline,
            override_gib=(floor_override_gib if floor_override_gib is not None
                          else safety_floor_env_override_gib()),
            mode=floor_mode,
            log_fn=lambda m: print(m, file=sys.stderr),
        )
        margin = floor.floor_gib
    ceiling = float(total_gib) - margin
    budget = ceiling - baseline
    return AdaptiveCeiling(
        total_gib=float(total_gib), used_gib=float(used_gib),
        tracked_current_gib=float(tracked_current_gib), baseline_gib=baseline,
        safety_margin_gib=margin, adaptive_ceiling_gib=ceiling, training_budget_gib=budget,
        floor_decomposition=(floor.to_json() if floor is not None else None),
    )


# ─────────────────────────── admission control (pure, HARD gate) ───────────────────────────
@dataclass(frozen=True)
class AdmissionDecision:
    admit: bool
    projected_new_gib: float
    system_used_gib: float
    active_growth_headroom_gib: float   # sum over active jobs of max(0, projected_peak - current_rss)
    projected_system_used_gib: float    # used + growth_headroom + projected_new
    adaptive_ceiling_gib: float
    headroom_after_gib: float           # ceiling - projected_system_used (negative => refuse)
    reason: str

    def to_json(self) -> dict:
        d = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}
        d["decision"] = "ADMIT" if self.admit else "REFUSE"
        return d


def project_system_used_after_launch(
    *,
    system_used_gib: float,
    active_growth_headroom_gib: float,
    projected_new_gib: float,
) -> float:
    """Projected SYSTEM-wide used RAM if the new job launches and every active job grows to its peak.

    ``system_used_gib`` is the vm_stat TRUTH right now (counts OS + control-plane + ALL our jobs at
    their CURRENT RSS). ``active_growth_headroom_gib`` is the extra each already-running job may still
    allocate to reach its projected peak (``sum max(0, peak - current)``). ``projected_new_gib`` is the
    to-be-launched job's projected peak. NO double count (current RSS is in ``system_used`` once; we
    add only the REMAINING growth). PURE.
    """
    return float(system_used_gib) + max(0.0, float(active_growth_headroom_gib)) + float(projected_new_gib)


def admission_decision(
    *,
    projected_new_gib: float,
    system_used_gib: float,
    active_growth_headroom_gib: float,
    ceiling: AdaptiveCeiling,
    fail_safe: bool = False,
) -> AdmissionDecision:
    """HARD admission gate: ADMIT iff projected system-used <= adaptive ceiling. PURE.

    This is the crash-prevention gate. It REFUSES a 2nd/3rd concurrent job whose peak would push the
    SUM over the ceiling — the exact failure that crashed the machine. ``fail_safe`` (set when the
    accounting failed closure/cross-validation) forces REFUSE regardless of the arithmetic — we never
    admit on a number we could not validate.
    """
    projected = project_system_used_after_launch(
        system_used_gib=system_used_gib,
        active_growth_headroom_gib=active_growth_headroom_gib,
        projected_new_gib=projected_new_gib,
    )
    headroom = ceiling.adaptive_ceiling_gib - projected
    admit = (projected <= ceiling.adaptive_ceiling_gib) and not fail_safe
    if fail_safe and projected <= ceiling.adaptive_ceiling_gib:
        return AdmissionDecision(
            admit=False, projected_new_gib=float(projected_new_gib), system_used_gib=float(system_used_gib),
            active_growth_headroom_gib=float(active_growth_headroom_gib), projected_system_used_gib=projected,
            adaptive_ceiling_gib=ceiling.adaptive_ceiling_gib, headroom_after_gib=headroom,
            reason=("FAIL-SAFE REFUSE: memory accounting failed closure/cross-validation — refusing to "
                    "admit on an unvalidated reading (would otherwise fit under the ceiling)."))
    if admit:
        reason = (f"projected system-used {projected:.1f} GiB <= adaptive ceiling "
                  f"{ceiling.adaptive_ceiling_gib:.1f} GiB (headroom {headroom:.1f} GiB); "
                  f"budget {ceiling.training_budget_gib:.1f} GiB, baseline {ceiling.baseline_gib:.1f} GiB")
    else:
        reason = (f"projected system-used {projected:.1f} GiB EXCEEDS adaptive ceiling "
                  f"{ceiling.adaptive_ceiling_gib:.1f} GiB by {-headroom:.1f} GiB — launching would "
                  f"risk a SYSTEM OOM/jetsam cascade (current used {system_used_gib:.1f} + active-growth "
                  f"{active_growth_headroom_gib:.1f} + new {projected_new_gib:.1f}). REFUSE.")
    return AdmissionDecision(
        admit=admit, projected_new_gib=float(projected_new_gib), system_used_gib=float(system_used_gib),
        active_growth_headroom_gib=float(active_growth_headroom_gib), projected_system_used_gib=projected,
        adaptive_ceiling_gib=ceiling.adaptive_ceiling_gib, headroom_after_gib=headroom, reason=reason,
    )


# ─────────────────────────── run priority (pure) ───────────────────────────
def classify_run_priority(label: str, cmd: str = "") -> int:
    """Priority score for a tracked job (HIGHER = more important = throttle/shed LAST).

    Pointer-movers (#205 sealed capstone / witness) outrank probes/sweeps/measurement. Explicit
    registry ``priority`` fields override this heuristic (handled by the caller)."""
    text = f"{label} {cmd}".lower()
    score = 0
    for kw, pts in (
        ("205", 100), ("sealed", 90), ("store_nothing", 85), ("capstone", 80),
        ("levelset_witness", 60), ("witness", 50), ("train_renderer", 40),
    ):
        if kw in text:
            score = max(score, pts)
    for kw, pts in (
        ("probe", -60), ("descent", -60), ("ev1", -55), ("smoke", -50), ("sweep", -50),
        ("microprobe", -50), ("char", -40), ("drift", -40), ("byte_close", -30),
        ("inflate", -30), ("bsdtar", -35), ("measurement", -45),
    ):
        if kw in text:
            score = min(score if score else 0, score + pts) if score > 0 else score + pts
    return score


# ─────────────────────────── tracked-job model (live) ───────────────────────────
@dataclass(frozen=True)
class TrackedJob:
    label: str
    pid: int
    pgid: int
    cmd: str
    priority: int
    projected_peak_gib: float
    current_rss_gib: float
    paused: bool
    throttle_eligible: bool
    own_group_leader: bool

    @property
    def growth_headroom_gib(self) -> float:
        return max(0.0, self.projected_peak_gib - self.current_rss_gib)

    def to_json(self) -> dict:
        return {
            "label": self.label, "pid": self.pid, "pgid": self.pgid,
            "priority": self.priority, "projected_peak_gib": round(self.projected_peak_gib, 2),
            "current_rss_gib": round(self.current_rss_gib, 2), "paused": self.paused,
            "throttle_eligible": self.throttle_eligible, "own_group_leader": self.own_group_leader,
        }


def _load_registry_rows(path: Path | None = None) -> list[dict]:
    p = path or _DURABLE_DAEMON_REGISTRY
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _running_registry_jobs(rows: Sequence[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == "running"]


def _process_state(pid: int) -> str:
    """Single-process state code from ``ps -o state=`` ('T' = stopped/SIGSTOP-paused)."""
    if pid <= 0:
        return ""
    try:
        out = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def is_paused_state(state: str) -> bool:
    """True iff a ps state code indicates a SIGSTOP-paused process (leading 'T')."""
    return state.strip().upper().startswith("T")


def matches_our_jobs(command: str, pattern: str = OUR_JOBS_PATTERN) -> bool:
    try:
        return re.search(pattern, command) is not None
    except re.error:
        return False


def _throttle_eligible(
    sample,
    *,
    samples: Mapping[int, object],
    self_pid: int,
    self_pgid: int | None,
    protected_pgids: set[int],
    guard_ancestors: set[int],
    owned_pids: frozenset[int],
) -> tuple[bool, bool]:
    """Return (throttle_eligible, own_group_leader) applying memory_guard's control-plane exclusions.

    Eligible iff: NOT the guard/its ancestors; NOT a control-plane app; NO external control-plane
    lineage; NOT on the ssh/tmux/shell denylist; pgid not protected; AND (own-group-leader detached
    daemon). If ``_mg`` is unavailable -> never eligible (fail-safe)."""
    if _mg is None:
        return (False, False)
    pgid = _mg._sample_pgid(sample)
    own_leader = (pgid == sample.pid)
    if sample.pid == self_pid or sample.pid in guard_ancestors:
        return (False, own_leader)
    if _mg.is_host_control_plane_process(sample):
        return (False, own_leader)
    if _mg.has_external_host_control_plane_lineage(
        samples, sample.pid, current_pid=self_pid, owned_pids=owned_pids
    ):
        return (False, own_leader)
    if _mg._matches_extra_protected(sample.command):
        return (False, own_leader)
    if pgid in protected_pgids:
        return (False, own_leader)
    if not own_leader:
        # Only pause detached daemons (own group leader) so the SIGSTOP scope is exactly the daemon
        # subtree, never a shared interactive shell job. (Non-leader children of a build agent are
        # still counted by the SYSTEM-used admission gate; they just aren't pause TARGETS here.)
        return (False, own_leader)
    return (True, own_leader)


def list_tracked_jobs(
    *,
    samples: Mapping[int, object] | None = None,
    registry_rows: Sequence[dict] | None = None,
    self_pid: int | None = None,
    self_pgid: int | None = None,
    our_jobs_pattern: str = OUR_JOBS_PATTERN,
) -> list[TrackedJob]:
    """Build the live tracked-job list = (registry custody jobs) UNION (ps processes matching the
    broad our-jobs pattern), each annotated with priority / projected-peak / current-RSS / paused /
    throttle-eligibility. Live I/O; the PURE selectors below consume the result."""
    if _mg is None:
        return []
    if samples is None:
        samples = _mg.sample_processes()
    if registry_rows is None:
        registry_rows = _load_registry_rows()
    if self_pid is None:
        self_pid = os.getpid()
    if self_pgid is None:
        self_pgid = _mg._safe_getpgrp()

    protected_pgids = _mg.protected_process_group_ids(samples, self_pid=self_pid, self_pgid=self_pgid)
    guard_ancestors = _mg._ancestor_pids(samples, self_pid)
    running = _running_registry_jobs(registry_rows)
    owned_pids = frozenset(int(r.get("pid", 0)) for r in running if int(r.get("pid", 0)) > 0)

    # Registry-recorded projection/priority keyed by pid.
    reg_by_pid: dict[int, dict] = {}
    for r in running:
        try:
            pid = int(r.get("pid", 0))
        except (TypeError, ValueError):
            continue
        if pid > 0:
            reg_by_pid[pid] = r

    # Candidate pids = registry custody UNION ps processes matching the our-jobs pattern.
    candidate_pids: set[int] = set(reg_by_pid)
    for pid, s in samples.items():
        if matches_our_jobs(getattr(s, "command", "")):
            candidate_pids.add(pid)

    jobs: list[TrackedJob] = []
    for pid in candidate_pids:
        s = samples.get(pid)
        if s is None:
            continue
        rec = reg_by_pid.get(pid, {})
        cmd_recorded = rec.get("cmd", "")
        cmd_str = " ".join(cmd_recorded) if isinstance(cmd_recorded, list) else str(cmd_recorded)
        cmd = s.command or cmd_str
        label = str(rec.get("label", "")) or f"pid{pid}"
        eligible, own_leader = _throttle_eligible(
            s, samples=samples, self_pid=self_pid, self_pgid=self_pgid,
            protected_pgids=protected_pgids, guard_ancestors=guard_ancestors, owned_pids=owned_pids,
        )
        # GB-F1 units fix: group_rss_gb returns KiB/1e6 units — convert to TRUE GiB HERE, exactly
        # once, at the read boundary. Everything downstream (growth_headroom, admission baseline,
        # band comparison, blackbox rows) is true GiB. Pre-fix the admission path under-counted
        # remaining growth by a measured 2.63 GiB (anti-conservative mixed-unit arithmetic).
        current_rss = _mg.group_rss_gb(samples, pid) * TRACKED_RSS_UNITS_TO_GIB
        # projected peak: explicit registry field wins; else fall back to current RSS (a floor).
        proj = rec.get("projected_peak_gib")
        try:
            proj_peak = float(proj) if proj is not None else current_rss
        except (TypeError, ValueError):
            proj_peak = current_rss
        proj_peak = max(proj_peak, current_rss)
        prio_field = rec.get("priority")
        try:
            priority = int(prio_field) if prio_field is not None else classify_run_priority(label, cmd)
        except (TypeError, ValueError):
            priority = classify_run_priority(label, cmd)
        paused = is_paused_state(_process_state(pid))
        jobs.append(TrackedJob(
            label=label, pid=pid, pgid=_mg._sample_pgid(s), cmd=cmd, priority=priority,
            projected_peak_gib=proj_peak, current_rss_gib=current_rss, paused=paused,
            throttle_eligible=eligible, own_group_leader=own_leader,
        ))
    return jobs


def sum_active_growth_headroom_gib(jobs: Sequence[TrackedJob]) -> float:
    """Sum of remaining growth-to-peak over all active tracked jobs (admission input)."""
    return float(sum(j.growth_headroom_gib for j in jobs))


def sum_tracked_current_gib(jobs: Sequence[TrackedJob]) -> float:
    return float(sum(j.current_rss_gib for j in jobs))


# ─────────────────────────── throttle target selection (pure) ───────────────────────────
def select_throttle_target(jobs: Sequence[TrackedJob]) -> TrackedJob | None:
    """LOWEST-priority throttle-eligible, not-yet-paused job (tie-break: LARGEST current RSS so the
    pause halts the most growth). Returns None when nothing is safely pausable. PURE."""
    candidates = [j for j in jobs if j.throttle_eligible and not j.paused]
    if not candidates:
        return None
    return min(candidates, key=lambda j: (j.priority, -j.current_rss_gib))


def select_resume_target(jobs: Sequence[TrackedJob]) -> TrackedJob | None:
    """HIGHEST-priority currently-paused job to resume first when pressure clears. PURE."""
    paused = [j for j in jobs if j.paused]
    if not paused:
        return None
    return max(paused, key=lambda j: (j.priority, j.current_rss_gib))


# ─────────────────────────── pressure classification + action (pure) ───────────────────────────
def classify_pressure(
    snapshot: SystemMemorySnapshot,
    *,
    warn_free_gib: float = DEFAULT_WARN_FREE_GIB,
    critical_free_gib: float = DEFAULT_CRITICAL_FREE_GIB,
) -> str:
    """"normal" | "warn" | "critical" from the snapshot (available metric OR the OS pressure level)."""
    lvl = snapshot.pressure_level
    if lvl >= PRESSURE_CRITICAL or snapshot.available_gib < critical_free_gib:
        return "critical"
    if lvl >= PRESSURE_WARN or snapshot.available_gib < warn_free_gib:
        return "warn"
    return "normal"


@dataclass(frozen=True)
class GovernorAction:
    level: str            # normal | warn | critical
    action: str           # none | alert | pause | resume | escalate_alert
    reason: str
    target: TrackedJob | None = None
    resume_targets: tuple = field(default_factory=tuple)

    def to_json(self) -> dict:
        return {
            "level": self.level, "action": self.action, "reason": self.reason,
            "target": self.target.to_json() if self.target else None,
            "resume_targets": [j.to_json() for j in self.resume_targets],
        }


def decide_governor_action(
    *,
    level: str,
    consecutive_warn: int,
    consecutive_critical: int,
    jobs: Sequence[TrackedJob],
    warn_consecutive: int = DEFAULT_WARN_CONSECUTIVE,
    critical_consecutive: int = DEFAULT_CRITICAL_CONSECUTIVE,
) -> GovernorAction:
    """PURE governor policy (debounced, incremental, reversible):

      * NORMAL -> resume any paused job (SIGCONT), highest priority first.
      * WARN sustained (>= warn_consecutive) -> pause ONE lowest-priority throttle-eligible job.
      * CRITICAL sustained (>= critical_consecutive) -> pause ONE (faster debounce); if nothing
        eligible -> escalate_alert (loud; hand off to memory_guard --watch / operator).
      * otherwise -> alert (loud, no action).

    NEVER kills; NEVER touches the control plane (throttle_eligible already excludes it).
    """
    paused = [j for j in jobs if j.paused]
    if level == "normal":
        if paused:
            return GovernorAction(level, "resume", "pressure normal — resume paused job(s)",
                                  resume_targets=tuple(sorted(paused, key=lambda j: (-j.priority, -j.current_rss_gib))))
        return GovernorAction(level, "none", "pressure normal — no paused jobs")
    if level == "critical" and consecutive_critical >= critical_consecutive:
        target = select_throttle_target(jobs)
        if target is not None:
            return GovernorAction(level, "pause",
                                  f"CRITICAL pressure sustained {consecutive_critical} polls — pause "
                                  f"lowest-priority job {target.label!r} (prio {target.priority}, "
                                  f"{target.current_rss_gib:.1f} GiB)", target=target)
        return GovernorAction(level, "escalate_alert",
                              "CRITICAL pressure but NO throttle-eligible job (control plane + "
                              "non-eligible protected) — killing NOTHING; hand off to memory_guard "
                              "--watch / manual intervention")
    if level == "warn" and consecutive_warn >= warn_consecutive:
        target = select_throttle_target(jobs)
        if target is not None:
            return GovernorAction(level, "pause",
                                  f"WARN pressure sustained {consecutive_warn} polls — pause "
                                  f"lowest-priority job {target.label!r} (prio {target.priority}, "
                                  f"{target.current_rss_gib:.1f} GiB)", target=target)
        return GovernorAction(level, "alert",
                              "WARN pressure but NO throttle-eligible job — alert only")
    return GovernorAction(level, "alert",
                          f"{level} pressure (debouncing: warn={consecutive_warn}/{warn_consecutive}, "
                          f"critical={consecutive_critical}/{critical_consecutive})")


# ─────────────────────────── SIGSTOP/SIGCONT actuators (reversible, FULL-TREE) ───────────────────
# #246 FIX (throughput-review F1): the old actuator killpg'd the registered job's pgid only. On the
# canonical launch tree (safe_run wrapper -> bash -> trainer started via start_new_session) the
# memory-bearing trainer sits in its OWN session/process-group, so the wrapper-pgid SIGSTOP reached
# a MEASURED 0.02 GiB of a 54 GiB run — the pause was cosmetic. The actuators now cover the job's
# FULL process TREE (recursive ppid-descendants of the registered root, which crosses session
# boundaries, PLUS the root's own pgid members), SIGSTOP deepest-first / SIGCONT in reverse,
# idempotent (a second pause re-signals stopped pids harmlessly and reports them), and STILL only
# ever pause/resume — no kill-class signal exists on this path. NOTHING outside the registered
# job's own tree can enter the stop set: the set is CONSTRUCTED from the job's descendants/group
# (an unregistered bystander pid is unreachable by construction) and every member is re-checked
# against the control-plane exclusion gates (defense-in-depth).
def _tree_depth(samples: Mapping[int, object], pid: int) -> int:
    """Parent-chain depth of ``pid`` within ``samples`` (cycle-safe). PURE."""
    depth = 0
    seen: set[int] = set()
    current = pid
    while current > 0 and current not in seen:
        seen.add(current)
        s = samples.get(current)
        if s is None or getattr(s, "ppid", 0) <= 0 or s.ppid == current:
            break
        current = s.ppid
        depth += 1
    return depth


def job_tree_pids(job: TrackedJob, *, samples: Mapping[int, object] | None = None) -> list[int]:
    """The FULL signal scope of a registered tracked job, DEEPEST-FIRST: recursive ppid-descendants
    of ``job.pid`` (crosses the start_new_session boundary the wrapper pgid missed — the #246 gap)
    plus, when the job is its own group leader, the members of its own pgid. Control-plane gates
    re-applied per member (the guard itself, its ancestors, control-plane apps, and the extra
    protected denylist can NEVER be in the set). Empty when memory_guard is unavailable (fail-safe:
    no gates -> no actuation)."""
    if _mg is None:
        return []
    if samples is None:
        samples = _mg.sample_processes()
    pids = set(_mg.descendant_pids(samples, job.pid))
    if job.own_group_leader and job.pgid > 0:
        pids |= {p for p, s in samples.items() if _mg._sample_pgid(s) == job.pgid}
    self_pid = os.getpid()
    guard_ancestors = _mg._ancestor_pids(samples, self_pid)
    out: list[int] = []
    for p in pids:
        s = samples.get(p)
        if s is None or p == self_pid or p in guard_ancestors:
            continue
        if _mg.is_host_control_plane_process(s):
            continue
        if _mg._matches_extra_protected(s.command):
            continue
        out.append(p)
    out.sort(key=lambda p: (-_tree_depth(samples, p), p))   # deepest first, deterministic
    return out


def job_tree_rss_gib(job: TrackedJob, *, samples: Mapping[int, object] | None = None) -> float:
    """TRUE-GiB RSS summed over the job's full signal scope (``job_tree_pids``) — the pause scope
    the band row reports. Post-#246 this should ~= the run's actual RSS (the gap field becomes the
    VERIFICATION that the pause reaches the memory)."""
    if _mg is None:
        return 0.0
    if samples is None:
        samples = _mg.sample_processes()
    total_kib = 0
    for p in job_tree_pids(job, samples=samples):
        s = samples.get(p)
        if s is not None:
            total_kib += int(getattr(s, "rss_kb", 0))
    return total_kib / (1024.0 ** 2)


def _signal_job_tree(job: TrackedJob, sig: int, *, samples: Mapping[int, object] | None = None,
                     reverse: bool = False) -> dict:
    """Send ``sig`` (SIGSTOP/SIGCONT ONLY — asserted) to every pid in the job's tree scope.
    Deepest-first by default; ``reverse`` for resume. Per-pid errors tolerated (races). Returns a
    machine-readable record {signalled, missed, order}."""
    assert sig in (signal.SIGSTOP, signal.SIGCONT), "pause/resume path is STOP/CONT only — no kill"
    order = job_tree_pids(job, samples=samples)
    if reverse:
        order = list(reversed(order))
    signalled: list[int] = []
    missed: list[int] = []
    for p in order:
        try:
            os.kill(p, sig)
            signalled.append(p)
        except (ProcessLookupError, PermissionError, OSError):
            missed.append(p)
    return {"signalled": signalled, "missed": missed, "order": order}


def pause_job(job: TrackedJob, *, samples: Mapping[int, object] | None = None) -> bool:
    """SIGSTOP the registered job's FULL process tree, deepest-first (#246 fix — covers the
    detached-session trainer the old pgid-only pause missed). Reversible + idempotent. Falls back
    to the legacy pgid/pid stop when the process sampler is unavailable. Returns ok (>=1 pid
    signalled)."""
    rec = _signal_job_tree(job, signal.SIGSTOP, samples=samples) if _mg is not None else {"signalled": []}
    if rec["signalled"]:
        return True
    # Fail-safe fallback (sampler unavailable or tree scan raced to empty): legacy scope.
    try:
        if job.own_group_leader and job.pgid > 0:
            os.killpg(job.pgid, signal.SIGSTOP)
        else:
            os.kill(job.pid, signal.SIGSTOP)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def resume_job(job: TrackedJob, *, samples: Mapping[int, object] | None = None) -> bool:
    """SIGCONT the registered job's full process tree in REVERSE (shallowest-first) order —
    restores exactly the scope the pause stopped. Same fallback as ``pause_job``. Returns ok."""
    rec = _signal_job_tree(job, signal.SIGCONT, samples=samples, reverse=True) if _mg is not None else {"signalled": []}
    if rec["signalled"]:
        return True
    try:
        if job.own_group_leader and job.pgid > 0:
            os.killpg(job.pgid, signal.SIGCONT)
        else:
            os.kill(job.pid, signal.SIGCONT)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


# ─────────────────────────── GRADUATED GUARD BANDS (BUILD #294 piece C) ───────────────────────────
# Operator memory policy 2026-07-04: bands act on ACTUAL RSS (protective ONLY — the runtime NEVER
# mutates training knobs from RSS, per the deterministic-reproducibility non-negotiable). Relative to
# the single-workload ENVELOPE (safe_frac x total RAM, ~108.8 GiB @ 0.85 x 128):
#   green  < yellow_frac (85%) of envelope  -> normal, no action
#   yellow [85%, 90%)                       -> telemetry alert row + PREP (checkpoint-freshness check)
#   red    >= red_frac (90%)                -> loud operator row + CLEAN CHECKPOINT-FRESHNESS record +
#                                              reversible PAUSE (SIGSTOP via pause_job — the canonical
#                                              pause actuator; NEVER SIGKILL, NEVER the control plane)
# Interaction with the existing pressure throttle (decide_governor_action): the throttle acts on
# SYSTEM availability (<15/<8 GiB free); the bands act on the RUN's actual RSS vs its envelope. When
# system pressure is already warn/critical the band DEFERS to the throttle (backstop BEHIND it, never
# a replacement); the band's clean-pause fires only in the sole-workload case the pressure ladder
# cannot see early (a run busting its envelope while the system still looks calm).
# #246 (governor-review F1) is FIXED as of BUILD #298: pause_job now SIGSTOPs the job's FULL
# process tree (recursive descendants + own pgid, deepest-first — see the actuator section), so the
# detached-session trainer IS halted. The band row's pause_scope field (now job_tree_rss_gib) is
# the standing VERIFICATION: post-fix it should ~= the run's actual RSS; the efficacy-gap note only
# fires if a future launch-tree shape re-opens the gap.
# UNITS: post GB-F1 the tracked units->true-GiB conversion happens ONCE at the list_tracked_jobs
# read boundary (see TRACKED_RSS_UNITS_TO_GIB in the constants block) — TrackedJob.current_rss_gib
# IS true GiB here; the band compares true-vs-true with no local conversion. The blackbox
# ``tracked_sum_gib`` double-count (wrapper group + trainer child counted twice, mine §7) does NOT
# affect the bands: select_band_run picks own-group-leader jobs only, so the run is counted once
# via its descendant-inclusive custody group.

BAND_GREEN = "green"
BAND_YELLOW = "yellow"
BAND_RED = "red"
DEFAULT_BAND_ENVELOPE_FRAC = 0.85       # single-workload envelope (operator policy 2026-07-04)
DEFAULT_BAND_YELLOW_FRAC = 0.85         # of the envelope
DEFAULT_BAND_RED_FRAC = 0.90            # of the envelope
DEFAULT_CKPT_STALE_S = 3600.0           # resume checkpoint older than this = stale (prep alert)
_BAND_LEDGER = _REPO_ROOT / ".omx" / "state" / "memory_governor_bands.jsonl"
_CKPT_GLOBS = ("levelset_resume*.npz", "*_resume_*.npz", "levelset_ckpt_*.npz", "*_ckpt_*.npz")


def band_envelope_gib(total_gib: float, envelope_frac: float = DEFAULT_BAND_ENVELOPE_FRAC) -> float:
    """The single-workload envelope the bands are fractions OF (safe_frac x total RAM)."""
    return float(envelope_frac) * float(total_gib)


def classify_guard_band(
    actual_rss_gib: float,
    envelope_gib: float,
    *,
    yellow_frac: float = DEFAULT_BAND_YELLOW_FRAC,
    red_frac: float = DEFAULT_BAND_RED_FRAC,
) -> str:
    """green | yellow | red from ACTUAL RSS vs the envelope. PURE."""
    if envelope_gib <= 0:
        return BAND_RED  # degenerate envelope => most protective classification
    frac = float(actual_rss_gib) / float(envelope_gib)
    if frac >= red_frac:
        return BAND_RED
    if frac >= yellow_frac:
        return BAND_YELLOW
    return BAND_GREEN


@dataclass(frozen=True)
class GuardBandDecision:
    band: str                 # green | yellow | red
    action: str               # none | alert_prep | already_paused | defer_to_throttle |
    #                           alert_no_target | clean_pause   (NO kill action EXISTS)
    reason: str
    target: TrackedJob | None = None
    actual_rss_gib: float = 0.0
    envelope_gib: float = 0.0
    checkpoint_age_s: float | None = None
    checkpoint_fresh: bool | None = None
    pause_scope_rss_gib: float | None = None   # own-pgid-only RSS the SIGSTOP actually reaches
    efficacy_gap_gib: float | None = None      # run RSS - pause scope (the #246 gap, measured)

    def to_json(self) -> dict:
        return {
            "band": self.band, "action": self.action, "reason": self.reason,
            "target": self.target.to_json() if self.target else None,
            "actual_rss_gib": round(self.actual_rss_gib, 2),
            "envelope_gib": round(self.envelope_gib, 2),
            "checkpoint_age_s": (round(self.checkpoint_age_s, 1)
                                 if self.checkpoint_age_s is not None else None),
            "checkpoint_fresh": self.checkpoint_fresh,
            "pause_scope_rss_gib": (round(self.pause_scope_rss_gib, 2)
                                    if self.pause_scope_rss_gib is not None else None),
            "efficacy_gap_gib": (round(self.efficacy_gap_gib, 2)
                                 if self.efficacy_gap_gib is not None else None),
        }


# The complete, closed action vocabulary. Tests assert no kill-class action can ever be emitted.
GUARD_BAND_ACTIONS = ("none", "alert_prep", "already_paused", "defer_to_throttle",
                      "alert_no_target", "clean_pause")


def decide_guard_band_action(
    *,
    band: str,
    job: TrackedJob | None,
    pressure_class: str,                 # classify_pressure(): normal | warn | critical
    actual_rss_gib: float,
    envelope_gib: float,
    checkpoint_age_s: float | None = None,
    checkpoint_stale_s: float = DEFAULT_CKPT_STALE_S,
    pause_scope_rss_gib: float | None = None,
) -> GuardBandDecision:
    """PURE band policy. Protective only (never mutates training knobs); NEVER kills; NEVER targets
    a non-throttle-eligible (control-plane-protected) process; DEFERS to the pressure throttle when
    the system pressure ladder is already engaged (backstop BEHIND the throttle, not a replacement).
    """
    fresh = (None if checkpoint_age_s is None else bool(checkpoint_age_s <= checkpoint_stale_s))
    ck = ("checkpoint age unknown" if checkpoint_age_s is None else
          f"latest resume checkpoint {checkpoint_age_s:.0f}s old "
          f"({'FRESH' if fresh else f'STALE > {checkpoint_stale_s:.0f}s'})")
    base = {"actual_rss_gib": float(actual_rss_gib), "envelope_gib": float(envelope_gib),
            "checkpoint_age_s": checkpoint_age_s, "checkpoint_fresh": fresh,
            "pause_scope_rss_gib": pause_scope_rss_gib}

    if band == BAND_GREEN:
        return GuardBandDecision(band=band, action="none",
                                 reason=f"green: {actual_rss_gib:.1f} GiB < "
                                        f"{DEFAULT_BAND_YELLOW_FRAC:.0%} of envelope "
                                        f"{envelope_gib:.1f} GiB", target=job, **base)
    if band == BAND_YELLOW:
        return GuardBandDecision(
            band=band, action="alert_prep", target=job,
            reason=f"yellow: {actual_rss_gib:.1f} GiB in [{DEFAULT_BAND_YELLOW_FRAC:.0%},"
                   f"{DEFAULT_BAND_RED_FRAC:.0%}) of envelope {envelope_gib:.1f} GiB — "
                   f"alert + prep; {ck}", **base)

    # RED band.
    if job is None:
        return GuardBandDecision(band=band, action="alert_no_target",
                                 reason=f"red: {actual_rss_gib:.1f} GiB >= "
                                        f"{DEFAULT_BAND_RED_FRAC:.0%} of envelope but NO tracked "
                                        f"training job to pause — alert only (killing NOTHING)",
                                 **base)
    if job.paused:
        return GuardBandDecision(band=band, action="already_paused", target=job,
                                 reason=f"red: job {job.label!r} already paused — no double-actuation",
                                 **base)
    if pressure_class in ("warn", "critical"):
        return GuardBandDecision(
            band=band, action="defer_to_throttle", target=job,
            reason=f"red: system pressure is {pressure_class} — the pressure throttle owns "
                   f"actuation (band stays the backstop BEHIND it, never a replacement); {ck}",
            **base)
    if not job.throttle_eligible:
        return GuardBandDecision(
            band=band, action="alert_no_target", target=job,
            reason=f"red: job {job.label!r} is NOT throttle-eligible (control-plane exclusion "
                   f"gates) — alert only; killing/pausing NOTHING", **base)

    gap = None
    gap_note = ""
    if pause_scope_rss_gib is not None:
        gap = max(0.0, float(actual_rss_gib) - float(pause_scope_rss_gib))
        if float(pause_scope_rss_gib) < 0.5 * float(actual_rss_gib):
            gap_note = (f" | #246 EFFICACY GAP (governor-review F1, REPORTED not fixed here): the "
                        f"SIGSTOP scope (pgid {job.pgid}) holds only {pause_scope_rss_gib:.1f} GiB "
                        f"of the run's {actual_rss_gib:.1f} GiB — the memory-bearing descendant "
                        f"group is NOT halted by this pause")
    return GuardBandDecision(
        band=band, action="clean_pause", target=job, efficacy_gap_gib=gap,
        reason=f"red: {actual_rss_gib:.1f} GiB >= {DEFAULT_BAND_RED_FRAC:.0%} of envelope "
               f"{envelope_gib:.1f} GiB — CLEAN PAUSE (reversible SIGSTOP; NEVER SIGKILL): {ck}"
               + gap_note,
        actual_rss_gib=float(actual_rss_gib), envelope_gib=float(envelope_gib),
        checkpoint_age_s=checkpoint_age_s, checkpoint_fresh=fresh,
        pause_scope_rss_gib=pause_scope_rss_gib)


def checkpoint_age_seconds(run_dir: Path) -> float | None:
    """Age (s) of the NEWEST resume/stage checkpoint in the run dir (I/O helper). None if none."""
    import time as _t

    run_dir = Path(run_dir)
    newest: float | None = None
    for pat in _CKPT_GLOBS:
        for p in run_dir.glob(pat):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if newest is None or m > newest:
                newest = m
    if newest is None:
        return None
    return max(0.0, _t.time() - newest)


def pgid_only_rss_gib(pgid: int) -> float:
    """RSS (GiB) of ONLY the processes whose pgid == the given group — the exact scope a
    killpg(SIGSTOP) reaches (vs group_rss_gb which is descendant-inclusive). Measures the #246 gap."""
    if pgid <= 0:
        return 0.0
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-g", str(pgid)],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total_kib = 0
    for tok in out.stdout.split():
        if tok.isdigit():
            total_kib += int(tok)
    return total_kib / (1024.0 ** 2)


def select_band_run(jobs: Sequence[TrackedJob]) -> TrackedJob | None:
    """The run the bands watch = the LARGEST-RSS tracked job that is its own group leader (the
    registered custody daemon whose descendant-inclusive group RSS covers the whole run). PURE."""
    leaders = [j for j in jobs if j.own_group_leader]
    if not leaders:
        return None
    return max(leaders, key=lambda j: j.current_rss_gib)


def append_band_row(row: dict, ledger_path: Path = _BAND_LEDGER) -> None:
    """fcntl-locked JSONL telemetry append (canonical .omx/state store pattern)."""
    import datetime as _dt
    import fcntl

    row = dict(row)
    row.setdefault("ts", _dt.datetime.now(_dt.UTC).isoformat())
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(ledger_path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as exc:  # telemetry must never take down the governor
        print(f"[system-governor] WARNING: band telemetry append failed: {exc}", file=sys.stderr)


def band_tick(
    *,
    envelope_frac: float = DEFAULT_BAND_ENVELOPE_FRAC,
    yellow_frac: float = DEFAULT_BAND_YELLOW_FRAC,
    red_frac: float = DEFAULT_BAND_RED_FRAC,
    run_dir: Path | None = None,
    apply: bool = False,
    floor_override_gib: float | None = None,
    floor_mode: str = FLOOR_MODE_DERIVED,
) -> GuardBandDecision:
    """ONE live guard-band evaluation (I/O wrapper around the pure policy). Emits the telemetry row
    (including the tick's derived-floor decomposition — max observability); with ``apply`` actually
    SIGSTOP-pauses on clean_pause (reversible; resume via the governor's normal-pressure resume
    path or ``kill -CONT``). The black-box daemon calls this in-process on its band cadence
    (BUILD #298 wiring); ``--band-tick`` (cron/loop) remains the standalone entry point."""
    snap = read_system_memory_snapshot()
    jobs = list_tracked_jobs()
    job = select_band_run(jobs)
    envelope = band_envelope_gib(snap.total_gib, envelope_frac)
    # TrackedJob.current_rss_gib is TRUE GiB post GB-F1 (converted once at the read boundary);
    # the envelope is true GiB (system total) — true-vs-true, no local conversion.
    actual = job.current_rss_gib if job is not None else 0.0
    band = classify_guard_band(actual, envelope, yellow_frac=yellow_frac, red_frac=red_frac)
    # Tier-scaled pressure thresholds (derived from the same headroom physics as the floor).
    pressure = classify_pressure(snap, warn_free_gib=derived_warn_free_gib(snap.total_gib),
                                 critical_free_gib=derived_critical_free_gib(snap.total_gib))
    ck_age = checkpoint_age_seconds(run_dir) if run_dir is not None else None
    # Post-#246 the pause scope is the FULL job tree — this field is the standing verification
    # that the pause reaches the run's memory (scope ~= actual; gap note fires only on regression).
    scope = job_tree_rss_gib(job) if job is not None else None
    decision = decide_guard_band_action(
        band=band, job=job, pressure_class=pressure, actual_rss_gib=actual,
        envelope_gib=envelope, checkpoint_age_s=ck_age, pause_scope_rss_gib=scope)
    applied = False
    if apply and decision.action == "clean_pause" and decision.target is not None:
        applied = pause_job(decision.target)
    floor = derive_safety_floor(
        total_gib=snap.total_gib,
        measured_cp_rss_gib=measured_control_plane_rss_gib(
            used_gib=snap.used_gib, tracked_current_gib=sum_tracked_current_gib(jobs)),
        override_gib=(floor_override_gib if floor_override_gib is not None
                      else safety_floor_env_override_gib()),
        mode=floor_mode, log_fn=lambda m: print(m, file=sys.stderr),
    )
    row = {"kind": "guard_band", "decision": decision.to_json(), "pressure": pressure,
           "system_used_gib": round(snap.used_gib, 2), "applied": applied,
           "safety_floor": floor.to_json(), "tracked_units": "true_gib"}
    append_band_row(row)
    if band == BAND_RED:
        print(f"[system-governor] GUARD-BAND RED: {decision.reason} (applied={applied})",
              file=sys.stderr)
    return decision


# ─────────────────────────── live admission (the HARD gate entry point) ───────────────────────────
@dataclass(frozen=True)
class LiveAdmissionContext:
    decision: AdmissionDecision
    ceiling: AdaptiveCeiling
    snapshot: SystemMemorySnapshot
    active_jobs: tuple

    def to_json(self) -> dict:
        return {
            "decision": self.decision.to_json(),
            "ceiling": self.ceiling.to_json(),
            "snapshot": self.snapshot.to_json(),
            "active_jobs": [j.to_json() for j in self.active_jobs],
        }


def live_admission_decision(
    *,
    projected_new_gib: float,
    snapshot: SystemMemorySnapshot | None = None,
    jobs: Sequence[TrackedJob] | None = None,
    exclude_pid: int | None = None,
    floor_override_gib: float | None = None,
    floor_mode: str = FLOOR_MODE_DERIVED,
) -> LiveAdmissionContext:
    """Read the live system + tracked jobs and evaluate the HARD admission gate for a new job whose
    projected peak is ``projected_new_gib``. ``exclude_pid`` drops a job (e.g. the launcher's own
    to-be-replaced pid) from the active set so it is not double-counted as both active and new.
    The safety floor is the tier-scaled DERIVED floor (BUILD #298); ``floor_override_gib`` /
    ``TAC_GOV_SAFETY_FLOOR_GIB`` / ``floor_mode`` thread the explicit overrides."""
    if snapshot is None:
        snapshot = read_system_memory_snapshot()
    if jobs is None:
        jobs = list_tracked_jobs()
    if exclude_pid is not None:
        jobs = [j for j in jobs if j.pid != exclude_pid]
    ceiling = compute_adaptive_ceiling(
        total_gib=snapshot.total_gib, used_gib=snapshot.used_gib,
        tracked_current_gib=sum_tracked_current_gib(jobs),
        floor_override_gib=floor_override_gib, floor_mode=floor_mode,
    )
    decision = admission_decision(
        projected_new_gib=projected_new_gib, system_used_gib=snapshot.used_gib,
        active_growth_headroom_gib=sum_active_growth_headroom_gib(jobs), ceiling=ceiling,
        fail_safe=snapshot.fail_safe,
    )
    return LiveAdmissionContext(decision=decision, ceiling=ceiling, snapshot=snapshot, active_jobs=tuple(jobs))


# ─────────────────────────── CLI ───────────────────────────
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="System-aware dynamic memory governor (max 128 GB safely).")
    ap.add_argument("--snapshot", action="store_true", help="print the live system memory snapshot (JSON)")
    ap.add_argument("--ceiling", action="store_true",
                    help="print the adaptive ceiling + budget + active tracked jobs (JSON)")
    ap.add_argument("--admit", action="store_true",
                    help="HARD admission gate for a new job (exit 0 ADMIT / 4 REFUSE)")
    ap.add_argument("--projected-gib", type=float, default=25.0,
                    help="projected peak RSS (GiB) of the to-be-launched job (default 25)")
    ap.add_argument("--exclude-pid", type=int, default=None,
                    help="drop this pid from the active set (avoid double-count of a replaced launcher)")
    ap.add_argument("--select-throttle-dry-run", action="store_true",
                    help="print (without pausing) the throttle target the governor WOULD pick now")
    ap.add_argument("--governor-tick", action="store_true",
                    help="evaluate ONE governor decision from a live read (dry unless --apply)")
    ap.add_argument("--apply", action="store_true", help="with --governor-tick: actually pause/resume")
    ap.add_argument("--warn-free-gib", type=float, default=None,
                    help="WARN available threshold (GiB); default: tier-derived "
                         "derived_warn_free_gib(total) (14.8 @128, 4.0 @8)")
    ap.add_argument("--critical-free-gib", type=float, default=None,
                    help="CRITICAL available threshold (GiB); default: tier-derived "
                         "derived_critical_free_gib(total) (8.4 @128, 3.0 @8)")
    # ── tier-scaled dynamical safety floor (BUILD #298) ──
    ap.add_argument("--safety-floor-gib", type=float, default=None,
                    help="EXPLICIT safety-floor override (GiB). Wins over the "
                         f"{SAFETY_FLOOR_ENV} env var; clamped to [ABS_MIN, cap_frac*RAM] with a "
                         "loud log when clamped")
    ap.add_argument("--safety-floor-mode", choices=FLOOR_MODES, default=FLOOR_MODE_DERIVED,
                    help="'derived' (default): tier-scaled dynamical floor; 'fixed': legacy "
                         "max(8, 0.08*T) formula (still cap-clamped — can no longer eat an 8 GiB box)")
    # ── graduated guard bands (BUILD #294 piece C) ──
    ap.add_argument("--band-tick", action="store_true",
                    help="evaluate ONE guard-band decision on the largest tracked run's ACTUAL RSS "
                         "vs the single-workload envelope (green<85%% / yellow 85-90%% / red>=90%%); "
                         "dry unless --apply (red clean_pause = reversible SIGSTOP; NEVER kills)")
    ap.add_argument("--band-envelope-frac", type=float, default=DEFAULT_BAND_ENVELOPE_FRAC)
    ap.add_argument("--band-yellow-frac", type=float, default=DEFAULT_BAND_YELLOW_FRAC)
    ap.add_argument("--band-red-frac", type=float, default=DEFAULT_BAND_RED_FRAC)
    ap.add_argument("--band-run-dir", type=str, default=None,
                    help="run dir for checkpoint-freshness in the band decision")
    args = ap.parse_args(argv)

    # Floor override precedence: CLI flag > env var > none (each clamped inside derive).
    floor_override = resolve_floor_override_gib(args.safety_floor_gib)

    if args.band_tick:
        decision = band_tick(
            envelope_frac=args.band_envelope_frac, yellow_frac=args.band_yellow_frac,
            red_frac=args.band_red_frac,
            run_dir=Path(args.band_run_dir) if args.band_run_dir else None,
            apply=args.apply, floor_override_gib=floor_override, floor_mode=args.safety_floor_mode)
        print(json.dumps(decision.to_json(), indent=2))
        return 0 if decision.band != BAND_RED else 6

    if args.snapshot:
        print(json.dumps(read_system_memory_snapshot().to_json()))
        return 0

    if args.ceiling:
        snap = read_system_memory_snapshot()
        jobs = list_tracked_jobs()
        ceiling = compute_adaptive_ceiling(
            total_gib=snap.total_gib, used_gib=snap.used_gib,
            tracked_current_gib=sum_tracked_current_gib(jobs),
            floor_override_gib=floor_override, floor_mode=args.safety_floor_mode)
        print(json.dumps({
            "snapshot": snap.to_json(), "ceiling": ceiling.to_json(),
            "active_jobs": [j.to_json() for j in jobs],
            "active_growth_headroom_gib": round(sum_active_growth_headroom_gib(jobs), 2),
        }, indent=2))
        return 0

    if args.admit:
        ctx = live_admission_decision(projected_new_gib=args.projected_gib, exclude_pid=args.exclude_pid,
                                      floor_override_gib=floor_override,
                                      floor_mode=args.safety_floor_mode)
        print(json.dumps(ctx.to_json(), indent=2))
        return 0 if ctx.decision.admit else 4

    if args.select_throttle_dry_run:
        jobs = list_tracked_jobs()
        target = select_throttle_target(jobs)
        print(json.dumps({
            "target": target.to_json() if target else None,
            "throttle_candidates": [j.to_json() for j in jobs if j.throttle_eligible and not j.paused],
            "note": "control-plane + non-eligible processes are excluded",
        }, indent=2))
        return 0

    if args.governor_tick:
        snap = read_system_memory_snapshot()
        jobs = list_tracked_jobs()
        warn_free = (args.warn_free_gib if args.warn_free_gib is not None
                     else derived_warn_free_gib(snap.total_gib))
        critical_free = (args.critical_free_gib if args.critical_free_gib is not None
                         else derived_critical_free_gib(snap.total_gib))
        level = classify_pressure(snap, warn_free_gib=warn_free,
                                  critical_free_gib=critical_free)
        cw = DEFAULT_WARN_CONSECUTIVE if level == "warn" else 0
        cc = DEFAULT_CRITICAL_CONSECUTIVE if level == "critical" else 0
        action = decide_governor_action(level=level, consecutive_warn=cw, consecutive_critical=cc, jobs=jobs)
        acted = []
        if args.apply and action.action == "pause" and action.target is not None:
            acted.append(("pause", action.target.label, pause_job(action.target)))
        elif args.apply and action.action == "resume":
            for j in action.resume_targets:
                acted.append(("resume", j.label, resume_job(j)))
        print(json.dumps({"snapshot": snap.to_json(), "action": action.to_json(),
                          "applied": acted}, indent=2))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
