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
     adaptive_ceiling``. ``current_system_used`` is the reclaimable-aware TRUE-COMMITTED used
     (2026-07-16 fix: ``wired + compressor + non-purgeable anonymous`` — file-backed cache /
     speculative / purgeable pages the OS evicts under pressure are NOT counted as committed; the
     legacy ``total - (free+inactive)`` basis counted ~9 GiB of reclaimable cache as pinned and
     false-refused an empirically-green 82 GiB bench on an idle box, while ALSO crediting dirty
     anonymous pages in the inactive queue as free on a loaded box — wrong in both directions).
     It still counts EVERY real consumer (OS + control-plane + training + byte-close + inflate +
     bsdtar + probes), so the gate is job-type-blind — exactly the SUM-over-128 gap that crashed
     us. This is a PREVENT gate, not a report: the launcher exits nonzero and starts NOTHING
     (override only via an operator-quoted rationale).

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
import contextlib
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

from tac.jsonl_store import append_locked_jsonl

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
_RSS_HISTORY_PATH = _REPO_ROOT / ".omx" / "state" / "system_memory_governor_rss_history.json"

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
# SOLE-WORKLOAD burst fraction (2026-07-09, adversarial-review NIT 1). CONCURRENT reserves the FULL
# control-plane footprint as burst headroom (measured_cp leg = cp + ch) DELIBERATELY on top of the
# used-based baseline subtraction (the double-reservation — agents burst proportional to current
# activity, 2026-07-02 crash). For a SOLE workload the used-based arithmetic already subtracts the
# baseline ONCE and no concurrent heavy job is spawning bursts, so reserve only the spike headroom +
# HALF the footprint as proportional burst room (ch + 0.5*cp): kills the double-count but — unlike a
# flat static floor — KEEPS burst protection that SCALES with the control plane. 0.5 is the midpoint
# between concurrent (effectively 1.0) and none (0.0); at the measured ~27 GiB baseline it reproduces
# the operator's documented sole-workload policy (>=10 fail-safe + ~10 margin ~= 20 GiB).
SOLE_WORKLOAD_BURST_FRAC = 0.5
SAFETY_FLOOR_ENV = "TAC_GOV_SAFETY_FLOOR_GIB"    # env override (GiB); CLI --safety-floor-gib wins
# ── OPERATOR CEILING POLICY (operator verbatim 2026-07-21: "You can increase the ceiling to a
# hundred and six gigabytes. We have a hundred and twenty eight gigabyte machine, and your
# experiments are going to be the only thing running on it." — corrected same day: "Sorry. I
# meant you can increase it to one hundred and sixteen gigabytes.") ─────────────────────────────
# On boxes with total RAM >= OPERATOR_CEILING_MIN_TOTAL_GIB, the adaptive ceiling is raised to AT
# LEAST OPERATOR_CEILING_GIB_DEFAULT (the margin is correspondingly capped at total-116, never
# leaving less than ABS_MIN_SAFETY_FLOOR_GIB of headroom). Smaller fleet tiers (8 GiB M1 etc.)
# are UNAFFECTED — the total-RAM guard exists so an absolute 128-GiB-box policy never eats a
# small box. The policy is a RAISE-only override: if the derived ceiling is already higher, it
# stands. Env TAC_GOV_OPERATOR_CEILING_GIB overrides (set 0 to disable). Value provenance:
# operator-provided constant (operator directive 2026-07-21, sole-workload machine), NOT derived;
# memory: operator_ceiling_106gib_sole_workload_20260721.md.
OPERATOR_CEILING_GIB_DEFAULT = 116.0
OPERATOR_CEILING_MIN_TOTAL_GIB = 120.0
OPERATOR_CEILING_ENV = "TAC_GOV_OPERATOR_CEILING_GIB"


def operator_ceiling_gib() -> float:
    """The operator-policy absolute ceiling (GiB); 0.0 disables. Env-overridable."""
    raw = os.environ.get(OPERATOR_CEILING_ENV)
    if raw is None:
        return OPERATOR_CEILING_GIB_DEFAULT
    try:
        return max(0.0, float(raw))
    except ValueError:
        return OPERATOR_CEILING_GIB_DEFAULT


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

# ── guard-band cadence + throttle ESCAPE HATCH (ddm_gb1 D2) ──────────────────────────────────────
# DEFAULT_BAND_INTERVAL_S lives HERE, with the rest of the band machinery it parameterises;
# ``memory_blackbox`` binds the same name to this object so the cadence has ONE source of truth and
# the hatch below cannot silently decouple from the band it is derived from.
DEFAULT_BAND_INTERVAL_S = 30.0
# The absolute ceiling on how long the throttle may hold ANY job SIGSTOPped. DERIVED, not chosen:
# ten consecutive band evaluations with no recovery is not a transient pressure spike — it is a
# throttle that is not working, and a job stopped that long is a FROZEN measurement, not a protected
# one. 10 x 30 s = 300 s, 15x below the MEASURED 2026-08-15 freeze (75+ min) and well above any real
# spike. Env-overridable (a tracked knob, never a forgotten default); a non-positive value would
# DISABLE the hatch and is refused by the parser below, so "0" cannot silently restore the freeze.
MAX_STOP_DURATION_BAND_EVALUATIONS = 10
DEFAULT_MAX_STOP_DURATION_S = MAX_STOP_DURATION_BAND_EVALUATIONS * DEFAULT_BAND_INTERVAL_S
MAX_STOP_DURATION_ENV = "TAC_GOV_MAX_STOP_DURATION_S"
# The token that marks an escape-hatch resume in ``GovernorAction.reason``. ONE source of truth
# (ddm_mb1): ``memory_blackbox`` classifies the hatch by this token to raise its TYPED alarm, so a
# reworded reason string here would otherwise silently stop the alarm from ever firing again — the
# actuation would look like a routine resume, which is exactly the silence the incident ran in.
ESCAPE_HATCH_REASON_TOKEN = "THROTTLE ESCAPE HATCH"


def resolve_max_stop_duration_s(env: Mapping[str, str] | None = None) -> float:
    """The throttle escape-hatch ceiling in seconds: ``TAC_GOV_MAX_STOP_DURATION_S`` when it parses
    to a POSITIVE float, else :data:`DEFAULT_MAX_STOP_DURATION_S`. A blank / non-numeric / <= 0
    value is LOUDLY ignored — the hatch may be lengthened or shortened, never switched off, because
    "off" is exactly the state the 75-minute freeze ran in."""
    raw = (env if env is not None else os.environ).get(MAX_STOP_DURATION_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_STOP_DURATION_S
    try:
        value = float(raw)
    except ValueError:
        print(f"[system-governor] WARNING: ignoring non-numeric {MAX_STOP_DURATION_ENV}={raw!r}",
              file=sys.stderr)
        return DEFAULT_MAX_STOP_DURATION_S
    if value <= 0.0:
        print(f"[system-governor] WARNING: ignoring non-positive {MAX_STOP_DURATION_ENV}={raw!r} "
              f"— the throttle escape hatch may not be disabled.", file=sys.stderr)
        return DEFAULT_MAX_STOP_DURATION_S
    return value

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
# ── reclaimable-aware committed accounting (2026-07-16 admission false-refuse fix) ───────────────
# WHY (operator P0, memory `admission_gate_naive_counts_reclaimable_as_committed_20260716`): the
# legacy decision basis ``used = total - (free + inactive)`` counts RECLAIMABLE memory (file-backed
# cache resident in the ACTIVE queue, speculative read-ahead, purgeable pages) as if it were
# committed. On a genuinely idle 128 GiB box (TRUE committed ~28.7 GiB) it reported used=37.6 and
# REFUSED an empirically-green 82 GiB bench (projected 109 > ceiling 102.9). The OS EVICTS those
# pages under pressure — a new allocation does NOT stack on top of them.
#
# The reclaimable-aware basis uses the kernel's own queue decomposition (vm_stat exposes it
# directly): ``File-backed pages + Anonymous pages == active + inactive + speculative`` (an EXACT
# kernel identity, verified live 2026-07-16: 1865616 + 5543060 == 3703200 + 3704096 + 1380).
#   TRUE committed  = wired + compressor + (anonymous - purgeable)   # needs swap to evict
#   reclaimable avail = free + file_backed + purgeable               # OS evicts w/o swap
# This is CORRECT IN BOTH DIRECTIONS: more generous on a file-cache-heavy idle box (kills the
# false refuse) and MORE CONSERVATIVE on an anon-heavy loaded box (dirty anonymous pages sitting
# in the INACTIVE queue are NOT free — the legacy free+inactive basis wrongly credited them).
# Validation (replaces nothing; ADDS): (1) the queue identity above within tolerance — a broken
# anon/file parse falls back to the legacy conservative basis (never trusted silently); (2) a
# one-sided bound: reclaimable available may never exceed total - wired - compressor (claiming
# wired/compressor as reclaimable is a parse bug). The existing one-sided psutil overcount check
# stays on the CONSERVATIVE free+inactive figure (psutil computes the same quantity — same-value
# two-source validation); the generous reclaimable figure is validated by (1)+(2) instead.
# NOTE kern.memorystatus_level was evaluated and REJECTED as the authority: it reports
# ~(total - wired - compressor)/total (measured 90% while an anon-heavy 63 GiB trainer ran),
# i.e. it treats swap-evictable anonymous memory as available — too generous for a crash gate.
RECLAIM_QUEUE_IDENTITY_TOL_GIB = 2.0   # |anon+file - (active+inactive+speculative)| tolerance
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


_ARM_TRUTHY = frozenset({"1", "true", "yes", "on"})


class Arming(NamedTuple):
    """Resolved state of a durably-armed, default-OFF actuator.

    ``source`` is WHY it reads that way ("env" / "flag" / "default"); ``detail`` is the operator-
    facing sentence a daemon logs at startup. "Off" is only a tracked state if it is SURFACED with
    its reason — a silent default is exactly the orphan-generator CLAUDE.md forbids."""
    armed: bool
    source: str
    detail: str


def resolve_arming(env_name: str, flag_path: Path, *,
                   env: Mapping[str, str] | None = None) -> Arming:
    """ONE resolver for every default-OFF actuator: armed iff ``env_name`` is truthy OR
    ``flag_path`` exists with a truthy first line. PURE apart from the two reads.

    Single source of truth on purpose (ddm_mb1). ``admission_enforcing()`` and
    ``throttle_arming()`` are the two callers; a second hand-rolled copy of this env-or-flag dance
    is how the two arming surfaces would silently diverge in their truthy-parsing or their
    precedence. Malformed / unreadable flag file => NOT armed (fail-safe: the actuator stays off)."""
    src = os.environ if env is None else env
    if (src.get(env_name, "") or "").strip().lower() in _ARM_TRUTHY:
        return Arming(True, "env", f"ARMED by {env_name}")
    try:
        if flag_path.is_file():
            first = (flag_path.read_text().splitlines() or [""])[0].strip().lower()
            if first in _ARM_TRUTHY:
                return Arming(True, "flag", f"ARMED by durable flag {flag_path}")
            return Arming(False, "default",
                          f"NOT armed ({flag_path} present but first line {first!r} is not truthy)")
    except OSError as exc:
        return Arming(False, "default", f"NOT armed ({flag_path} unreadable: {exc})")
    return Arming(False, "default",
                  f"NOT armed (default OFF; set {env_name}=1 or write a truthy {flag_path})")


def admission_enforcing() -> bool:
    """True iff the admission gate is in ENFORCE mode (blocks launches). Default FALSE (ADVISORY —
    logs what it WOULD refuse) until independent review arms enforce via the env var
    ``TAC_ADMISSION_ENFORCE=1`` OR the durable flag file ``.omx/state/admission_enforce.flag``
    (truthy first line). Only ADDS refusals the gate already computes (enforce ⊆ advisory)."""
    return resolve_arming(ADMISSION_ENFORCE_ENV, _ADMISSION_ENFORCE_FLAG).armed


# ── the SIGSTOP THROTTLE actuator arming (ddm_mb1, 2026-08-16) ───────────────────────────────────
# MEASURED INCIDENT 2026-08-15: the throttle SIGSTOPped five live jobs for 75+ minutes on a
# 40.5-GiB-free box. ddm_gb1 fixed the MECHANISM (right object / re-arm / exit-resume). It did NOT
# fix the ARMING: ``run_daemon`` still defaulted ``govern=True`` and the auto-start path
# (``ensure_blackbox_running`` -> ``memory_blackbox.py --daemon``) passed no opt-out, so the next
# training launch would silently restart the actuator that has not yet been re-adjudicated.
#
# THE SPLIT (CLAUDE.md "'Off' is a tracked queue, never a forgotten default"):
#   * the RECORDER is read-only, score-neutral observability -> DEFAULT ON, never gated;
#   * the THROTTLE is an ACTUATOR that SIGSTOPs live measurements -> DEFAULT OFF, durably armed,
#     and its off-state is LOGGED with its reason at every daemon start.
# The incident's own verdict is why off is the right default even post-fix: "per-job safe_run
# envelopes (real RSS caps) were the protection that actually worked — the machine-wide SIGSTOP
# layer was net-negative."
THROTTLE_ARM_ENV = "TAC_GOV_THROTTLE_ARM"
_THROTTLE_ARM_FLAG = _TOOLS.parent / ".omx" / "state" / "governor_throttle_arm.flag"


def throttle_arming(*, env: Mapping[str, str] | None = None) -> Arming:
    """Resolved arming of the SIGSTOP throttle actuator. DEFAULT OFF (see the block above).

    Armed by ``TAC_GOV_THROTTLE_ARM=1`` or a truthy ``.omx/state/governor_throttle_arm.flag``.
    Returns the full :class:`Arming` (not a bare bool) so the daemon can log WHY — an unexplained
    "governor: off" line is how a disabled protection layer becomes a forgotten one."""
    return resolve_arming(THROTTLE_ARM_ENV, _THROTTLE_ARM_FLAG, env=env)


def throttle_armed(*, env: Mapping[str, str] | None = None) -> bool:
    """True iff the SIGSTOP throttle actuator is armed. Convenience over :func:`throttle_arming`."""
    return throttle_arming(env=env).armed

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

# ── unknown-peak growth default (review-fix CRITICAL B: zero-growth assumption was BACKWARDS) ────
# A tracked job with NO recorded projected_peak_gib used to fall back to proj_peak = current_rss —
# i.e. the safety gate assumed an unknown-peak job would grow by ZERO GiB. For an admission gate
# that direction is inverted: unknown MUST mean "assume it can still grow substantially". The
# conservative default is current_rss + UNKNOWN_GROWTH_HEADROOM_GIB, matching the launch paths' own
# unknown-projection fallback (spawn_durable_daemon --projected-gb default 25.0 / the governor CLI
# --projected-gib default 25.0) so a job that slipped past registration is charged the same
# projection it would have been charged at admission time.
# Two DELIBERATE exemptions keep the default from manufacturing phantom refusals:
#   * protection infra (black-box / guard / governor): tiny always-on daemons, deliberately
#     unprojected; +25 GiB each forever would permanently poison every admission decision.
#   * governed DESCENDANTS: a ps-pattern-matched process whose ancestor chain reaches a REGISTERED
#     running job (e.g. the trainer inside a spawn_durable_daemon -> safe_run tree, which sits in
#     its OWN session so it surfaces as a separate ps candidate). Its growth is already projected
#     by the parent's registry row; charging +25 GiB again would double-count (and could
#     false-refuse small jobs while the live #205 runs).
UNKNOWN_GROWTH_HEADROOM_GIB = 25.0
# A governed row is a HEAVY workload iff its RESOLVED projected peak is >= this (GiB). Sub-heavy
# control-plane / telemetry daemons (black-box, dashboards) do NOT reserve heavy-job growth headroom
# (their live rss is already in the vm_stat ``used`` baseline). Kept IDENTICAL to the launcher's
# ``launch_witness_run.HEAVY_MIN_PROJECTED_GIB`` so the admission gate and the launcher's
# ``_governed_active_jobs`` agree on what counts as an active heavy workload.
HEAVY_MIN_PROJECTED_GIB = 4.0
_PROTECTION_INFRA_TOKENS = ("memory_blackbox.py", "memory_guard.py", "system_memory_governor.py")

# ── material-workload RSS floor for UNREGISTERED ps-only matches (2026-07-11 phantom-reservation fix) ──
# WHY (the MEASURED false-positive this extincts): ``OUR_JOBS_PATTERN`` is a BROAD substring regex used
# primarily for THROTTLE candidate discovery. Applied to the ADMISSION growth projection it over-matches:
# ANY process whose argv merely CONTAINS a token (a ``grep``/``ugrep``/``rg`` over the source, an editor
# with the file open, a ``python -c`` mentioning the script, the launch pipeline itself, a short-lived
# byte-close/inflate probe, a sibling build/measurement agent) is counted as a heavy tracked job and
# charged ``current + UNKNOWN_GROWTH_HEADROOM_GIB`` (+25 GiB) of phantom growth. On a nearly-idle machine
# (2026-07-11: 30.7 GiB used / ~97 GiB free / ZERO live training procs) ~8 such incidental matches summed
# to a phantom ~200 GiB projected growth and FALSE-REFUSED an operator-GO'd witness resume; ``--reconcile``
# could not clear it because these are live ps processes, not registry rows. GROUND-TRUTH FIX: an
# UNREGISTERED ps-only match whose CURRENT RSS is below this material floor is charged ZERO growth (its RSS
# is already inside the vm_stat ``used`` baseline the gate anchors on), exactly like the protection-infra /
# governed-descendant exemptions. SAFETY PRESERVED: (1) every genuine heavy launch goes through the
# governed path and REGISTERS (running row with a projected_peak, or a pending reservation) -> recorded
# projection wins -> fully charged, untouched; (2) a materially resident unregistered match retains the
# full current+25 charge unless fresh same-process history exists AND the job is structurally eligible for
# the live runtime throttle; only that throttle-backed intersection may use the bounded measured reserve;
# (3) a sub-floor process by construction cannot itself drive a 128 GiB crash. Value 2.0 GiB ==
# ABS_MIN_SAFETY_FLOOR_GIB (the jetsam-avoidance minimum): below it a lone process is never the crash
# driver. This ONLY relaxes the unknown-peak default for the unregistered-ps case; the registered /
# reservation / infra / descendant paths are bit-identical.
MATERIAL_UNREGISTERED_RSS_FLOOR_GIB = 2.0

# ── measured recent-growth history for MATERIAL unregistered ps-only matches (2026-07-14) ──────
# Admission is a prediction layer. The old flat +25 GiB prediction remains the fail-conservative
# fallback, but it is no longer substituted for a measurement when one exists. The daemon already
# samples process RSS through ``list_tracked_jobs``; that same impure edge persists a bounded rolling
# history, while ``estimate_observed_remaining_growth_gib`` and ``resolve_projected_peak_gib`` remain
# PURE. The bounded projection law is:
#
#   G_remaining(p,t) = clamp(max(G_poll, H * max_i((rss_t-rss_i)/(t-t_i), 0)), 0, G_unknown)
#
# over fresh same-process samples in the rolling window. A flat series at ps' 1-KiB RSS resolution
# has zero slope. Insufficient/stale/clock-skewed/PID-reused history is UNKNOWN and therefore retains
# ``G_unknown = 25 GiB`` exactly. ``H`` is bounded by the history window, so a demonstrably growing
# process can still saturate the old +25 charge. A plateau retains ``G_poll`` (one live-throttle
# poll of reserve at the fastest modeled rate), so lazy materialization never receives literal zero.
#
# SAFETY LAYERS (canonical-equation-style control argument): admission is LAYER 3 prediction;
# ``decide_governor_action`` is the unchanged LAYER 2 live backstop. The measured reserve is passed to
# the pure projection law only when ``_throttle_eligible`` proves the process is an own-group leader the
# daemon can pause; every non-eligible process stays on +25. If a measured slope later rises, every daemon
# tick sees actual available memory and the tier-scaled WARN/CRITICAL floor SIGSTOPs the lowest-priority
# eligible job before jetsam. Therefore this change only removes prediction error for demonstrated,
# throttle-backed plateaus; it does NOT weaken the live throttle, tier-scaled floor, control-plane
# exclusions, recorded projections, pending reservations, or unknown-history fallback.
RSS_HISTORY_SCHEMA = "system_memory_governor_rss_history.v1"
RSS_HISTORY_WINDOW_S = 10.0 * 60.0
RSS_HISTORY_TTL_S = 15.0 * 60.0
RSS_HISTORY_MIN_OBSERVATION_S = 30.0
RSS_GROWTH_PROJECTION_HORIZON_S = RSS_HISTORY_WINDOW_S
RSS_HISTORY_CLOCK_SKEW_TOLERANCE_S = 5.0
RSS_HISTORY_MAX_SAMPLES_PER_PID = 64
RSS_HISTORY_MAX_PIDS = 128
# Canonical runtime pressure sampler cadence (mirrors memory_blackbox.DEFAULT_INTERVAL_S). A material
# plateau keeps one poll's worth of growth at the fastest modeled rate (the old 25-GiB unknown budget
# materializing within the minimum evidence span). This is deliberately NONZERO: lazy graphs/inflate
# can be flat during warmup and burst after the observation window.
RUNTIME_THROTTLE_POLL_INTERVAL_S = 2.0
MATERIAL_PLATEAU_GROWTH_RESERVE_GIB = (
    UNKNOWN_GROWTH_HEADROOM_GIB
    * RUNTIME_THROTTLE_POLL_INTERVAL_S
    / RSS_HISTORY_MIN_OBSERVATION_S
)
# ``ps`` reports RSS in KiB, so one KiB in true GiB is the measurement-resolution plateau epsilon.
RSS_PLATEAU_EPSILON_GIB = 1.0 / (1024.0 * 1024.0)

# ── PENDING admission reservations (review-fix CRITICAL C: launch TOCTOU) ────────────────────────
# Two near-simultaneous governed launches could BOTH pass the admission gate before either wrote
# its registry row (decision read -> Popen -> reservation write). The launchers now write a PENDING
# reservation row {label, projected_peak_gib, reserved_ts, pid=None, status="admitting"} inside the
# registry fcntl lock IMMEDIATELY after an ADMIT decision (see spawn_durable_daemon / safe_run);
# ``list_tracked_jobs`` counts fresh pending rows' projected peaks as growth headroom so the second
# launcher's admission sees the first's reservation. A pending row older than the max age with no
# pid is STALE (a crashed launcher) and is ignored here + swept by the launchers — a phantom
# reservation must never permanently block admission.
PENDING_RESERVATION_STATUS = "admitting"
PENDING_RESERVATION_MAX_AGE_S = 120.0


def is_protection_infra_cmd(cmd: str) -> bool:
    """True iff ``cmd`` is control-plane protection infra (black-box / memory guard / governor) —
    mirrors ``spawn_durable_daemon._is_protection_infra_cmd`` (never admission-gated, tiny)."""
    return any(tok in str(cmd) for tok in _PROTECTION_INFRA_TOKENS)


@dataclass(frozen=True)
class RSSHistorySample:
    """One process-identity-bound RSS observation. PURE value object."""

    pid: int
    rss_gib: float
    ts: float
    process_key: str

    def to_json(self) -> dict:
        return {
            "pid": int(self.pid),
            "rss_gib": float(self.rss_gib),
            "ts": float(self.ts),
            "process_key": str(self.process_key),
        }


def estimate_observed_remaining_growth_gib(
    samples: Sequence[RSSHistorySample],
    *,
    now_ts: float,
    window_s: float = RSS_HISTORY_WINDOW_S,
    min_observation_s: float = RSS_HISTORY_MIN_OBSERVATION_S,
    projection_horizon_s: float = RSS_GROWTH_PROJECTION_HORIZON_S,
    plateau_epsilon_gib: float = RSS_PLATEAU_EPSILON_GIB,
    plateau_growth_reserve_gib: float = MATERIAL_PLATEAU_GROWTH_RESERVE_GIB,
    clock_skew_tolerance_s: float = RSS_HISTORY_CLOCK_SKEW_TOLERANCE_S,
    unknown_growth_headroom_gib: float = UNKNOWN_GROWTH_HEADROOM_GIB,
) -> float | None:
    """Estimate remaining growth from a fresh same-PID/same-identity RSS series. PURE.

    ``None`` means NO admissible history and deliberately selects the legacy +25 GiB fallback.
    Any future timestamp invalidates the whole series; silently using it could manufacture a long
    observation span after a clock reset. The newest sample must be current (within the recency
    tolerance), which prevents a stale plateau from relaxing admission.
    """
    now = float(now_ts)
    window = max(0.0, float(window_s))
    min_span = max(0.0, float(min_observation_s))
    horizon = max(0.0, float(projection_horizon_s))
    cap = max(0.0, float(unknown_growth_headroom_gib))
    epsilon = max(0.0, float(plateau_epsilon_gib))
    plateau_reserve = min(cap, max(0.0, float(plateau_growth_reserve_gib)))
    skew = max(0.0, float(clock_skew_tolerance_s))
    if not math.isfinite(now):
        return None

    valid: list[RSSHistorySample] = []
    identities: set[str] = set()
    pids: set[int] = set()
    for sample in samples:
        try:
            ts = float(sample.ts)
            rss = float(sample.rss_gib)
            pid = int(sample.pid)
            process_key = str(sample.process_key)
        except (AttributeError, TypeError, ValueError):
            return None
        if not (math.isfinite(ts) and math.isfinite(rss)) or rss < 0.0 or pid <= 0 or not process_key:
            return None
        if ts > now:
            return None
        if now - ts <= window:
            valid.append(RSSHistorySample(pid=pid, rss_gib=rss, ts=ts, process_key=process_key))
            pids.add(pid)
            identities.add(process_key)
    if len(valid) < 2 or len(pids) != 1 or len(identities) != 1:
        return None

    valid.sort(key=lambda sample: (sample.ts, sample.rss_gib))
    latest = valid[-1]
    if now - latest.ts > skew:
        return None
    earliest = valid[0]
    if latest.ts - earliest.ts < min_span:
        return None

    rss_values = [sample.rss_gib for sample in valid]
    if max(rss_values) - min(rss_values) <= epsilon:
        return plateau_reserve

    # Conservative recent-trend estimate: take the largest positive endpoint slope from ANY earlier
    # sample, rather than a least-squares average that could cancel a late growth burst.
    max_positive_rate = 0.0
    for sample in valid[:-1]:
        elapsed = latest.ts - sample.ts
        if elapsed <= 0.0:
            continue
        max_positive_rate = max(
            max_positive_rate,
            max(0.0, latest.rss_gib - sample.rss_gib) / elapsed,
        )
    return min(cap, max(plateau_reserve, max_positive_rate * horizon))


def _rss_history_process_key(sample: object) -> str | None:
    """Kernel-start-bound identity guard for PID reuse, content-hashed. PURE.

    A missing start identity is not approximated with command/ppid: two recycled processes can share
    both. ``None`` forces the conservative +25 path until the live sampler supplies ``ps lstart``.
    """
    start_identity = getattr(sample, "start_identity", None)
    if not start_identity:
        return None
    payload = "\0".join(
        (
            str(start_identity),
            str(getattr(sample, "ppid", "")),
            str(getattr(sample, "pgid", "")),
            str(getattr(sample, "command", "")),
        )
    ).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _parse_rss_history_rows(payload: object) -> list[RSSHistorySample]:
    """Parse a history payload fail-closed; any malformed row is discarded. PURE."""
    if not isinstance(payload, dict) or payload.get("schema") != RSS_HISTORY_SCHEMA:
        return []
    raw_rows = payload.get("samples")
    if not isinstance(raw_rows, list):
        return []
    rows: list[RSSHistorySample] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        try:
            parsed = RSSHistorySample(
                pid=int(row["pid"]),
                rss_gib=float(row["rss_gib"]),
                ts=float(row["ts"]),
                process_key=str(row["process_key"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            parsed.pid > 0
            and parsed.rss_gib >= 0.0
            and math.isfinite(parsed.rss_gib)
            and math.isfinite(parsed.ts)
            and parsed.process_key
        ):
            rows.append(parsed)
    return rows


def update_rss_history_and_estimate_growth(
    current_samples: Mapping[int, tuple[float, str]],
    *,
    history_path: Path = _RSS_HISTORY_PATH,
    now_ts: float | None = None,
) -> dict[int, float | None]:
    """fcntl-lock, TTL-sweep, atomically persist RSS samples, and return pure trend estimates.

    This is the IMPURE edge. It never weakens admission on an I/O/parse/lock failure: callers receive
    ``None`` for every PID, which ``resolve_projected_peak_gib`` maps to the legacy +25 GiB charge.
    The separate lock inode stays stable while the JSON data file is atomically replaced.
    """
    # Monotonic time is comparable across governor processes in one boot and cannot jump backward
    # under NTP/wall-clock adjustment. A reboot resets it; the future/stale sweep then drops old rows.
    now = time.monotonic() if now_ts is None else float(now_ts)
    fallback = {int(pid): None for pid in current_samples}
    if not current_samples or not math.isfinite(now):
        return fallback
    path = Path(history_path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    temporary_name: str | None = None
    try:
        import fcntl

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                except (OSError, UnicodeError, json.JSONDecodeError):
                    payload = {}
                prior = _parse_rss_history_rows(payload)

                # TTL sweep first. A stored future timestamp indicates wall-clock rollback/skew; it
                # is not carried forward because it could fake an observation span.
                kept = [
                    row for row in prior
                    if 0.0 <= now - row.ts <= RSS_HISTORY_TTL_S
                ]
                for pid, (rss_gib, process_key) in sorted(current_samples.items()):
                    pid_i = int(pid)
                    rss = max(0.0, float(rss_gib))
                    key = str(process_key)
                    # PID reuse/exec identity change resets this PID's evidence to one sample, which
                    # deliberately yields UNKNOWN/+25 until a new observation window accrues.
                    kept = [row for row in kept if row.pid != pid_i or row.process_key == key]
                    kept.append(RSSHistorySample(pid=pid_i, rss_gib=rss, ts=now, process_key=key))

                by_pid: dict[int, list[RSSHistorySample]] = {}
                for row in sorted(kept, key=lambda item: (item.pid, item.ts, item.rss_gib)):
                    by_pid.setdefault(row.pid, []).append(row)
                # Bound both axes deterministically: newest PIDs by last timestamp, then numeric PID.
                pid_order = sorted(
                    by_pid,
                    key=lambda pid: (-by_pid[pid][-1].ts, pid),
                )[:RSS_HISTORY_MAX_PIDS]
                bounded: list[RSSHistorySample] = []
                for pid in pid_order:
                    bounded.extend(by_pid[pid][-RSS_HISTORY_MAX_SAMPLES_PER_PID:])
                bounded.sort(key=lambda item: (item.pid, item.ts, item.rss_gib))

                out_payload = {
                    "schema": RSS_HISTORY_SCHEMA,
                    "samples": [row.to_json() for row in bounded],
                }
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_name = temporary.name
                    json.dump(out_payload, temporary, sort_keys=True, separators=(",", ":"))
                    temporary.write("\n")
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_name, path)
                temporary_name = None
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"[system-governor] WARNING: RSS-history update failed; retaining +25 GiB fallback: {exc}",
            file=sys.stderr,
        )
        return fallback
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass

    estimates: dict[int, float | None] = {}
    for pid in current_samples:
        series = [row for row in bounded if row.pid == int(pid)]
        estimates[int(pid)] = estimate_observed_remaining_growth_gib(series, now_ts=now)
    return estimates


def resolve_projected_peak_gib(
    recorded_peak,
    current_rss_gib: float,
    *,
    cmd: str = "",
    governed_descendant: bool = False,
    unregistered_ps_only: bool = False,
    observed_remaining_growth_gib: float | None = None,
    material_rss_floor_gib: float = MATERIAL_UNREGISTERED_RSS_FLOOR_GIB,
    unknown_growth_headroom_gib: float = UNKNOWN_GROWTH_HEADROOM_GIB,
) -> float:
    """CONSERVATIVE projected-peak resolution for one tracked job (review-fix CRITICAL B). PURE.

    * A valid recorded projection wins (floored at current RSS — a job can never "un-use" memory).
    * Protection infra + governed descendants: zero-growth default (current RSS) — see the
      exemption rationale on the constants block above.
    * ``unregistered_ps_only`` match below the material RSS floor: zero-growth (current RSS). An
      unregistered ps-pattern match whose argv merely CONTAINS a token but is not materially resident
      (a grep/editor/``python -c``/launch pipeline/short probe) is NOT a heavy job; its RSS is already
      in the vm_stat ``used`` baseline. Charging it +25 GiB manufactures the phantom refusal fixed
      2026-07-11 (see the MATERIAL_UNREGISTERED_RSS_FLOOR_GIB constants block). A genuine heavy job
      registers (recorded projection wins above) OR is materially resident. For a MATERIAL ps-only
      process, a finite ``observed_remaining_growth_gib`` is clamped to [0, +25] and charged; ``None``
      or a malformed/non-finite value retains the full +25 fallback. Safety precondition at the impure
      caller: pass an observation only for an own-group-leader process that ``_throttle_eligible`` says
      Layer 2 can pause; otherwise pass ``None`` and retain +25.
    * Everything else with an unknown peak: ``current_rss + UNKNOWN_GROWTH_HEADROOM_GIB`` — an
      unknown-peak job is assumed to still be able to grow by the same 25 GiB the launch paths
      project by default, NEVER by zero (the old backwards fallback).
    """
    current = max(0.0, float(current_rss_gib))
    if recorded_peak is not None:
        try:
            return max(float(recorded_peak), current)
        except (TypeError, ValueError):
            pass  # malformed recorded value -> treat as unknown (conservative path below)
    if governed_descendant or is_protection_infra_cmd(cmd):
        return current
    if unregistered_ps_only and current < float(material_rss_floor_gib):
        return current  # incidental/transient ps-token match -> zero phantom growth (ground truth)
    if unregistered_ps_only and observed_remaining_growth_gib is not None:
        try:
            observed = float(observed_remaining_growth_gib)
        except (TypeError, ValueError):
            observed = math.nan
        if math.isfinite(observed):
            # Never charge more than the old guard and never turn a negative/noisy trend into credit.
            cap = max(0.0, float(unknown_growth_headroom_gib))
            remaining = min(cap, max(0.0, observed))
            return current + remaining
    return current + float(unknown_growth_headroom_gib)


_RSS_CAP_FLAGS = ("--rss-mb", "--rss-cap-mb")


def declared_rss_cap_gib(cmd: str) -> float | None:
    """The ENFORCED per-process RSS ceiling declared on a ``safe_run`` command line, in TRUE GiB —
    ``--rss-mb N`` / ``--rss-cap-mb N`` (MiB) — or None when the command declares none. PURE.

    WHY THIS IS A PROJECTION AND NOT A GUESS: safe_run KILLS the child at this cap, so it is an
    upper bound the process cannot exceed. A declared, enforced ceiling always beats the
    unknown-peak default.

    ddm_gb1 D5b (MEASURED 2026-08-16, INCIDENT #3). The dashboard's registry row carries no
    ``projected_peak_gib``, so it resolved to ``rss 0.22 + UNKNOWN_GROWTH_HEADROOM_GIB`` = 25.22 —
    above ``HEAVY_MIN_PROJECTED_GIB``, so a 0.22 GiB telemetry daemon was counted as a HEAVY job
    reserving 25 GiB of growth. That is precisely the case ``sum_active_growth_headroom_gib``'s
    docstring says it excludes ("a 0.2-GiB dashboard with a 2.44-GiB recorded projection reserved
    ~2.2 GiB of phantom heavy-growth and REFUSED a real launch", 2026-07-09) — the exclusion held
    only while something recorded that 2.44. 2.44 GiB IS ``--rss-mb 2500``: the number was always in
    the argv. Reading it back makes the #370 control-plane exemption independent of whether a
    launcher remembered to copy it into the registry."""
    tokens = str(cmd).split()
    for i, token in enumerate(tokens):
        flag, _, inline = token.partition("=")
        if flag not in _RSS_CAP_FLAGS:
            continue
        raw = inline if inline else (tokens[i + 1] if i + 1 < len(tokens) else "")
        try:
            mib = float(raw)
        except ValueError:
            continue
        if mib > 0:
            return mib * (1024.0 ** 2) / (1024.0 ** 3)
    return None


def pending_reservation_rows(
    rows: Sequence[dict],
    *,
    now_ts: float | None = None,
    max_age_s: float = PENDING_RESERVATION_MAX_AGE_S,
) -> list[dict]:
    """FRESH pending admission reservations from the registry: status=="admitting", no pid yet,
    ``reserved_ts`` within ``max_age_s``. A row with a missing/unparsable ``reserved_ts`` is
    treated as STALE (never counted) — a malformed row must not permanently block admission (the
    launch-side sweep drops it; the write path always stamps ``reserved_ts``). PURE."""
    now = time.time() if now_ts is None else float(now_ts)
    fresh: list[dict] = []
    for r in rows:
        if not isinstance(r, dict) or r.get("status") != PENDING_RESERVATION_STATUS:
            continue
        if r.get("pid"):
            continue  # the launcher already promoted it (or a malformed row) -> not a reservation
        try:
            age = now - float(r["reserved_ts"])
        except (KeyError, TypeError, ValueError):
            continue  # missing/unparsable timestamp -> stale by definition
        if age <= float(max_age_s):
            # (a negative age — slightly-future ts from clock skew — still counts: conservative)
            fresh.append(r)
    return fresh


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
    # ── reclaimable-aware committed accounting (2026-07-16; see the constants-block rationale) ──
    # ADMISSION decision basis when ``reclaimable_ok``: available that the OS can actually free
    # without swap (free + file_backed + purgeable) and its complement (TRUE committed used =
    # wired + compressor + non-purgeable anonymous + the unaccounted closure gap, conservatively).
    # On fallback (missing counters / identity violation / bound violation) these EQUAL the legacy
    # available_gib / used_gib and ``reclaimable_ok`` is False.
    available_reclaimable_gib: float = 0.0
    used_committed_gib: float = 0.0
    reclaimable_ok: bool = False
    anonymous_gib: float = 0.0
    file_backed_gib: float = 0.0
    purgeable_gib: float = 0.0


@dataclass(frozen=True)
class SystemMemorySnapshot:
    total_gib: float
    available_gib: float   # free + inactive (LEGACY basis; throttle/pressure paths + fallback)
    used_gib: float        # total - available (LEGACY basis; the admission gate anchors on
                           # used_committed_gib below when reclaimable_ok — 2026-07-16 fix)
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
    # ── reclaimable-aware committed accounting (2026-07-16 admission-gate fix) ──
    # The ADMISSION decision basis when ``reclaimable_ok`` (defaults keep hand-built snapshots on
    # the legacy basis): see MemoryAccounting + the RECLAIM_QUEUE_IDENTITY_TOL_GIB constants block.
    available_reclaimable_gib: float = 0.0
    used_committed_gib: float = 0.0
    reclaimable_ok: bool = False
    anonymous_gib: float = 0.0
    file_backed_gib: float = 0.0
    purgeable_gib: float = 0.0

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
        # Reclaimable-aware committed accounting inputs (2026-07-16 fix). On kernels whose vm_stat
        # lacks these labels _grab returns 0 and the accounting FALLS BACK to the legacy basis.
        "purgeable": _grab("Pages purgeable"),
        "file_backed": _grab("File-backed pages"),
        "anonymous": _grab("Anonymous pages"),
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
    vm_purgeable_pages: int = 0,
    vm_file_backed_pages: int = 0,
    vm_anonymous_pages: int = 0,
    sysctl_free_pages: int | None = None,
    mempressure_total_bytes: int | None = None,
    psutil_available_bytes: int | None = None,
    closure_tol_gib: float = CLOSURE_TOL_GIB,
    xvalidate_tol_gib: float = XVALIDATE_TOL_GIB,
    free_page_tol_gib: float = FREE_PAGE_XCHECK_TOL_GIB,
    reclaim_identity_tol_gib: float = RECLAIM_QUEUE_IDENTITY_TOL_GIB,
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

    # ── reclaimable-aware committed accounting (2026-07-16 admission false-refuse fix) ──────────
    # See the RECLAIM_QUEUE_IDENTITY_TOL_GIB constants block for the full derivation + validation
    # rationale. Fallback (counters missing / identity violated / bound violated) = the legacy
    # conservative basis above, LOUDLY noted — the generous figure is never trusted unvalidated.
    active_gib = vm_active_pages * ps / _GIB
    speculative_gib = vm_speculative_pages * ps / _GIB
    anonymous_gib = vm_anonymous_pages * ps / _GIB
    file_backed_gib = vm_file_backed_pages * ps / _GIB
    purgeable_gib = vm_purgeable_pages * ps / _GIB
    reclaimable_ok = False
    available_reclaimable_gib = available_gib
    used_committed_gib = used_gib
    if vm_anonymous_pages > 0 or vm_file_backed_pages > 0:
        queue_gib = active_gib + inactive_gib + speculative_gib
        identity_dev_gib = abs((anonymous_gib + file_backed_gib) - queue_gib)
        committed_gib = wired_gib + compressor_gib + max(0.0, anonymous_gib - purgeable_gib)
        # Conservative min of the two derivations: the direct reclaimable sum vs total-minus-
        # committed (they differ by the unaccounted closure gap, which we charge as USED).
        reclaim_raw = min(free_gib + file_backed_gib + purgeable_gib,
                          max(0.0, total_gib - committed_gib), total_gib)
        nonwired_bound_gib = total_gib - wired_gib - compressor_gib
        if identity_dev_gib > reclaim_identity_tol_gib:
            notes.append(
                f"reclaimable accounting DISABLED: queue identity |anon {anonymous_gib:.2f} + file "
                f"{file_backed_gib:.2f} - queues {queue_gib:.2f}| = {identity_dev_gib:.2f} GiB > "
                f"{reclaim_identity_tol_gib} GiB (anon/file parse suspected) — legacy free+inactive basis")
        elif reclaim_raw > nonwired_bound_gib + xvalidate_tol_gib:
            notes.append(
                f"reclaimable accounting DISABLED: reclaimable {reclaim_raw:.2f} GiB exceeds "
                f"total - wired - compressor = {nonwired_bound_gib:.2f} GiB + {xvalidate_tol_gib} "
                f"(overcount bug) — legacy free+inactive basis")
        else:
            reclaimable_ok = True
            if fail_safe:
                reclaim_raw = max(0.0, reclaim_raw - discrepancy_gib)
            available_reclaimable_gib = reclaim_raw
            used_committed_gib = max(0.0, total_gib - reclaim_raw)
    # else: counters absent (older kernel / hand-built snapshot) — benign fallback to the legacy
    # basis, signalled by ``reclaimable_ok=False`` (notes are reserved for validation FAILURES so a
    # clean legacy snapshot still reconciles with zero notes).

    return MemoryAccounting(
        total_gib=total_gib, available_gib=available_gib, available_primary_gib=available_primary_gib,
        used_gib=used_gib, free_gib=free_gib, wired_gib=wired_gib, compressor_gib=compressor_gib,
        closure_gib=closure_gib, closure_ok=closure_ok, cross_validated=cross_validated,
        discrepancy_gib=discrepancy_gib, fail_safe=fail_safe, validation_notes=tuple(notes),
        available_reclaimable_gib=available_reclaimable_gib, used_committed_gib=used_committed_gib,
        reclaimable_ok=reclaimable_ok, anonymous_gib=anonymous_gib, file_backed_gib=file_backed_gib,
        purgeable_gib=purgeable_gib,
    )


def read_system_memory_snapshot() -> SystemMemorySnapshot:
    """Live SYSTEM-wide memory snapshot: programmatic kernel counters (vm_stat page counts + sysctl
    page size + hw.memsize) reconciled + VALIDATED (closure + cross-validation) via the pure
    ``reconcile_memory_accounting``, then annotated with pressure / swap / load. The admission gate
    anchors on ``used_committed_gib`` (reclaimable-aware TRUE committed; job-type-blind,
    fail-safe-adjusted) when ``reclaimable_ok``, else the legacy ``used_gib``; the throttle /
    pressure paths keep the legacy ``available_gib`` basis."""
    ps = page_size_bytes()
    counters = _read_vm_stat_counters()
    total_bytes = int(round(total_ram_gib() * _GIB))
    acct = reconcile_memory_accounting(
        page_size=ps, total_bytes=total_bytes,
        vm_free_pages=counters["free"], vm_active_pages=counters["active"],
        vm_inactive_pages=counters["inactive"], vm_speculative_pages=counters["speculative"],
        vm_wired_pages=counters["wired"], vm_compressor_pages=counters["compressor"],
        vm_throttled_pages=counters.get("throttled", 0),
        vm_purgeable_pages=counters.get("purgeable", 0),
        vm_file_backed_pages=counters.get("file_backed", 0),
        vm_anonymous_pages=counters.get("anonymous", 0),
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
        available_reclaimable_gib=acct.available_reclaimable_gib,
        used_committed_gib=acct.used_committed_gib, reclaimable_ok=acct.reclaimable_ok,
        anonymous_gib=acct.anonymous_gib, file_backed_gib=acct.file_backed_gib,
        purgeable_gib=acct.purgeable_gib,
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


def non_workload_used_gib(*, used_gib: float, tracked_current_gib: float) -> float:
    """The adaptive ceiling's BASELINE: system used minus the sum of our tracked (governed) jobs'
    current RSS — i.e. OS + file cache + control plane + anything untracked. Both inputs are TRUE
    GiB post GB-F1 (the units conversion happens once, at the ``list_tracked_jobs`` read boundary).
    Clamped >= 0. PURE.

    This is NOT the control-plane RSS (see ``measured_control_plane_rss_gib``). It was called that
    until 2026-08-15 and the name lied about the object it measured — the ddm_gb1 incident."""
    return max(0.0, float(used_gib) - float(tracked_current_gib))


def measured_control_plane_rss_gib(*, samples: Mapping[int, object] | None = None) -> float | None:
    """TRUE-GiB RSS summed over the NAMED control-plane processes ONLY — enumerated from the live
    process table by the same identity gates the throttle uses to refuse touching them
    (``memory_guard.is_host_control_plane_process`` = Claude/Codex apps + helpers + their launcher
    lineage, ``_matches_extra_protected`` = the ssh/tmux/shell denylist, ``is_protection_infra_cmd``
    = black-box/guard/governor). Each pid counted once. ``samples`` injectable (PURE over an
    injected table); a live scan otherwise.

    Returns ``None`` when the measurement is UNAVAILABLE — ``memory_guard`` missing, or a LIVE scan
    that came back empty (a running box always has processes, so an empty live table is a FAILED
    scan, never "zero control plane"). ``None`` makes ``derive_safety_floor`` fall back to its
    STATIC policy leg (0.08*T): a floor derived from a measurement we do not have is worse than the
    policy floor.

    MEASURED INCIDENT (2026-08-15, ddm_gb1; memory ``governor-stuck-throttle-froze-three-live-
    measurements``). This function used to return ``used_gib - tracked_current_gib`` — the box's
    TOTAL used memory INCLUDING FILE CACHE. Live receipt from
    ``.omx/state/memory_blackbox.daemon.log``: "SAFETY-FLOOR CLAMP: measured_cp value 92.00 GiB
    clamped to 64.00 GiB" repeating at ~1.5 Hz on a machine with 40.5 GiB free. A floor input of 92
    on a 128 GiB box declares "less than half the box is free" permanently. The old arithmetic is a
    real and useful quantity — the ceiling BASELINE — and now lives under its honest name
    ``non_workload_used_gib``. WRONG-OBJECT MEASUREMENT is the confound genus here (sister:
    ``check_no_raw_virtual_memory_safety_basis``); a name that lies about its object is how it
    survived review for two months."""
    if _mg is None:
        return None
    live = samples is None
    table = _mg.sample_processes() if live else samples
    if live and not table:
        return None
    total_kib = 0
    for sample in table.values():
        cmd = str(getattr(sample, "command", "") or "")
        if (
            _mg.is_host_control_plane_process(sample)
            or _mg._matches_extra_protected(cmd)
            or is_protection_infra_cmd(cmd)
        ):
            total_kib += int(getattr(sample, "rss_kb", 0) or 0)
    return total_kib / (1024.0 ** 2)


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
    sole_workload: bool = False,
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
    Derivations for every constant: the tier-scaled-floor constants block above.

    ``sole_workload`` (2026-07-09) drops the DYNAMICAL ``measured_cp`` leg and falls back to the
    STATIC policy leg (0.08*T = the operator-policy >=10 GiB control-plane floor, line-119 comment).
    WHY: the dynamical leg reserves free headroom on the ORDER OF the current control-plane RSS
    (line-115 rationale: "agents spawn in bursts proportional to current activity") — DELIBERATELY
    conservative ON TOP OF the used-based admission arithmetic that ALREADY counts that same RSS.
    That double-reservation is CORRECT when concurrent heavy jobs / agent bursts are in flight (the
    2026-07-02 crash class), but for a genuine SOLE workload it contradicts the operator's 2026-07-04
    sole-workload policy ("no artificial ceiling; >=10 fail-safe + ~10 margin"). Under sole-workload
    the LAUNCH-time floor relaxes to the static policy leg; the RUNTIME guard-bands (blackbox
    WARN/CRITICAL + throttle) remain the real-time protection against any control-plane burst DURING
    the run, so this is safe. Concurrency is measured by the caller (any other admitted HEAVY tracked
    job, projected_peak >= HEAVY_MIN_PROJECTED_GIB) — mirrors the launcher's sole-vs-concurrent
    safe_frac logic so the two gates agree."""
    total = float(total_gib)
    abs_min = float(abs_min_gib)
    cap = max(abs_min, float(cap_frac) * total)
    ch = cp_headroom_gib(total, min_gib=cp_headroom_min_gib, frac=cp_headroom_frac)
    static_leg = float(static_frac) * total
    cp = None if measured_cp_rss_gib is None else max(0.0, float(measured_cp_rss_gib))
    # DYNAMICAL leg. CONCURRENT: reserve the FULL control-plane footprint + spike headroom (cp + ch),
    # DELIBERATELY on top of the used-based baseline subtraction (2026-07-02 crash protection). SOLE:
    # the used-based arithmetic already subtracts the baseline once and no concurrent heavy job is
    # bursting, so reserve only the spike headroom + a NON-DOUBLED proportional burst fraction
    # (ch + SOLE_WORKLOAD_BURST_FRAC*cp) — kills the double-count while KEEPING control-plane-scaled
    # burst protection (adversarial-review NIT 1, 2026-07-09). See SOLE_WORKLOAD_BURST_FRAC.
    if cp is None:
        measured_leg = None
    elif sole_workload:
        measured_leg = ch + SOLE_WORKLOAD_BURST_FRAC * cp
    else:
        measured_leg = cp + ch

    if override_gib is not None:
        raw = float(override_gib)
        winning = "override"
    elif mode == FLOOR_MODE_FIXED:
        raw = max(float(legacy_floor_gib), static_leg)
        winning = "fixed_legacy"
    else:
        raw = max(abs_min, static_leg, measured_leg if measured_leg is not None else 0.0)
        if measured_leg is not None and raw == measured_leg:
            winning = "sole_workload_burst_reserve" if sole_workload else "measured_cp"
        elif raw == static_leg:
            winning = "static_frac_sole_workload" if sole_workload else "static_frac"
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


def derived_resume_free_gib(total_gib: float) -> float:
    """Tier-scaled RESUME/NO-PAUSE threshold = warn + one more cp_headroom. @128 -> 21.2 GiB;
    @8 -> 5.0 GiB. PURE.

    THE GOVERNOR'S OWN re-arm reference (ddm_gb1 D2). The throttle must never depend SOLELY on the
    macOS ``kern.memorystatus_vm_pressure_level`` to come back: that signal is STICKY at warn, so a
    resume gated only on it never fires — the spike-guard median-freeze genus (#304), a guard whose
    reference never re-arms. MEASURED 2026-08-15: level=warn at 40.4 GiB available for 75+ minutes
    while five jobs sat SIGSTOPped.

    One cp_headroom above WARN is deliberate HYSTERESIS: pause fires below this line, resume at or
    above it, so the [warn, resume] band is a dead zone and the throttle cannot flap between the two
    actuations on a sticky OS level. Same headroom physics as every other rung — derived, not a new
    constant."""
    return derived_warn_free_gib(total_gib) + cp_headroom_gib(total_gib)


def governing_free_gib(snapshot: SystemMemorySnapshot) -> float:
    """The CONSERVATIVE free-memory basis for throttle ACTUATION: reclaimable-aware available when
    the accounting validated, else the legacy free+inactive. PURE.

    macOS ``available`` = free + inactive counts DIRTY ANON parked in the inactive queue as
    available (CLAUDE.md "raw virtual memory safety basis" class), so it OVER-reports. Using the
    smaller reclaimable number here is protective in both directions of the ddm_gb1 D2 cure: it
    makes the no-pause veto FIRE LESS and the resume FIRE LATER than the optimistic number would."""
    if getattr(snapshot, "reclaimable_ok", False):
        return float(snapshot.available_reclaimable_gib)
    return float(snapshot.available_gib)


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
    sole_workload: bool = False,
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
            sole_workload=sole_workload,
            log_fn=lambda m: print(m, file=sys.stderr),
        )
        margin = floor.floor_gib
    ceiling = float(total_gib) - margin
    # Operator ceiling policy (2026-07-21, sole-workload 128 GiB box): RAISE-only, guarded to
    # big boxes so small fleet tiers keep their derived-floor protection unchanged.
    _oc = operator_ceiling_gib()
    if _oc > 0.0 and float(total_gib) >= OPERATOR_CEILING_MIN_TOTAL_GIB and ceiling < _oc:
        ceiling = min(_oc, float(total_gib) - ABS_MIN_SAFETY_FLOOR_GIB)
        margin = float(total_gib) - ceiling
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
    # A ps-only candidate whose ancestor chain reaches a REGISTERED running job. Its subtree RSS is
    # ALREADY inside the parent's group RSS (2026-07-16 fix: counting both inflated tracked_current
    # by the full trainer RSS — measured live: wrapper 63.56 + trainer 63.56 = 127.17 GiB tracked on
    # a 71 GiB-used box, clamping baseline to 0 and UNDER-deriving the safety floor).
    governed_descendant: bool = False

    @property
    def growth_headroom_gib(self) -> float:
        return max(0.0, self.projected_peak_gib - self.current_rss_gib)

    def to_json(self) -> dict:
        return {
            "label": self.label, "pid": self.pid, "pgid": self.pgid,
            "priority": self.priority, "projected_peak_gib": round(self.projected_peak_gib, 2),
            "current_rss_gib": round(self.current_rss_gib, 2), "paused": self.paused,
            "throttle_eligible": self.throttle_eligible, "own_group_leader": self.own_group_leader,
            "governed_descendant": self.governed_descendant,
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
    rss_history_path: Path | None = None,
    rss_history_now_ts: float | None = None,
    layer2_backstop_armed: bool | None = None,
) -> list[TrackedJob]:
    """Build the live tracked-job list = (registry custody jobs) UNION (ps processes matching the
    broad our-jobs pattern), each annotated with priority / projected-peak / current-RSS / paused /
    throttle-eligibility. Live I/O; the PURE selectors below consume the result.

    A real live scan records material unregistered RSS history at ``_RSS_HISTORY_PATH``. Tests and
    callers that inject ``samples`` remain hermetic unless they explicitly provide a history path.

    ``layer2_backstop_armed`` (ddm_mb1) defaults to the live throttle arming. It licenses the
    measured-growth admission relaxation below: the relaxation exists ONLY because Layer 2 can pause
    the job it relaxes, so a DISARMED throttle must withdraw it. See the callsite comment.
    """
    if _mg is None:
        return []
    # Resolved ONCE per scan (not per candidate): one flag read, and a mid-scan flip cannot make two
    # candidates in the same tick disagree about whether a backstop exists.
    layer2_armed = (throttle_armed() if layer2_backstop_armed is None
                    else bool(layer2_backstop_armed))
    live_process_scan = samples is None
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

    # Read-boundary conversion + measured-growth sampling happen ONCE per candidate per scan. The
    # history edge is intentionally outside the pure resolver. Only MATERIAL unregistered ps-only
    # candidates need trend evidence; all other branches are bit-identical and never consult it.
    current_rss_by_pid: dict[int, float] = {}
    material_unregistered_samples: dict[int, tuple[float, str]] = {}
    for pid in sorted(candidate_pids):
        sample = samples.get(pid)
        if sample is None:
            continue
        current_rss = _mg.group_rss_gb(samples, pid) * TRACKED_RSS_UNITS_TO_GIB
        current_rss_by_pid[pid] = current_rss
        if pid not in reg_by_pid and current_rss >= MATERIAL_UNREGISTERED_RSS_FLOOR_GIB:
            process_key = _rss_history_process_key(sample)
            if process_key is not None:
                material_unregistered_samples[pid] = (current_rss, process_key)
    effective_history_path = (
        _RSS_HISTORY_PATH if live_process_scan and rss_history_path is None else rss_history_path
    )
    if effective_history_path is not None and material_unregistered_samples:
        observed_growth_by_pid = update_rss_history_and_estimate_growth(
            material_unregistered_samples,
            history_path=effective_history_path,
            now_ts=rss_history_now_ts,
        )
    else:
        observed_growth_by_pid = dict.fromkeys(material_unregistered_samples)

    jobs: list[TrackedJob] = []
    for pid in sorted(candidate_pids):
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
        current_rss = current_rss_by_pid[pid]
        # Projected peak (review-fix CRITICAL B): explicit registry field wins; an UNKNOWN peak
        # gets the CONSERVATIVE +UNKNOWN_GROWTH_HEADROOM_GIB default (never the old zero-growth
        # fallback), except protection infra + governed descendants (see resolve_projected_peak_gib
        # + the constants-block rationale). A ps-only candidate whose ancestor chain reaches a
        # REGISTERED running pid is a governed descendant (its growth is projected by the parent
        # row — e.g. the trainer inside a daemon->safe_run tree sits in its own session).
        # ddm_gb1 D5a (MEASURED 2026-08-16, INCIDENT #3): a candidate whose ancestor chain reaches
        # ANOTHER CANDIDATE — registered or ps-only — is a descendant of that candidate's launch
        # tree. Its RSS is ALREADY inside the ancestor's descendant-inclusive ``group_rss_gb``, so
        # charging it an independent +25 triple-counts one job. Pre-fix this recognised REGISTRY-
        # owned ancestors only, and the mp2 tree's root (a ps-only ``launch_detached`` wrapper) was
        # not registered: pids 39740 <- 39748 <- 25923, one intact ppid chain, MEASURED group RSS
        # 7.30 GiB, charged 6.56 + 25.00 + 25.00 = 56.56 GiB and REFUSED a READY_TO_FIRE launch on a
        # box using 37.9 of 128 GiB. pid 0/1 are excluded: the init chain is every process's
        # ancestor and could never be a launch-tree root.
        governed_desc = False
        if not rec:
            try:
                ancestors = _mg._ancestor_pids(samples, pid) - {pid, 0, 1}
                governed_desc = bool(ancestors & (owned_pids | candidate_pids))
            except Exception:
                governed_desc = False
        # A registered row with no declared projection falls back to its ENFORCED safe_run RSS cap
        # before the unknown-peak default (ddm_gb1 D5b — the #370 control-plane regression).
        recorded_peak = rec.get("projected_peak_gib")
        if recorded_peak is None and rec:
            # Registry argv first (the authoritative launch record), live ps argv second. Explicit
            # None-checks, not `or`: a cap is a float and `or` would swallow a legitimate 0.0.
            recorded_peak = declared_rss_cap_gib(cmd_str)
            if recorded_peak is None:
                recorded_peak = declared_rss_cap_gib(cmd)
        proj_peak = resolve_projected_peak_gib(
            recorded_peak, current_rss, cmd=cmd,
            governed_descendant=governed_desc,
            # An UNREGISTERED (ps-only) candidate below the material RSS floor is an incidental token
            # match (grep/editor/launcher/short probe), NOT a heavy job — charge it zero phantom growth
            # (2026-07-11 phantom-reservation fix). A registered row (rec present) is never relaxed.
            unregistered_ps_only=(not rec),
            # MATERIAL unregistered jobs use a bounded fresh observed slope. Missing/stale/invalid
            # history is None and therefore retains the old +25 GiB conservative fallback.
            # CRITICAL safety precondition: only throttle-eligible detached group leaders may use a
            # relaxed measurement. A non-leader/unpausable material process has no Layer-2 backstop
            # and therefore stays on +25 even if its history appears flat.
            # ddm_mb1 SECOND LEG of the same precondition: structural eligibility is necessary but
            # NOT sufficient — a throttle-eligible job still has no Layer-2 backstop when the
            # actuator is DISARMED (now the default). So the relaxation is withdrawn wholesale in
            # that state and every material unregistered job falls back to the conservative +25.
            # Direction is deliberately the safe one: disarming the throttle makes admission
            # STRICTER, never wider. Without this, ddm_mb1's default-OFF change would have silently
            # widened Layer-3 admission by removing the very backstop that licensed it.
            # KNOWN RESIDUAL (stated, not hidden): "armed" means the flag/env says the actuator MAY
            # run, not that a daemon is running right now. A liveness check would have to import
            # memory_blackbox, which imports THIS module — a cycle. This is strictly better than the
            # pre-ddm_mb1 behaviour, which granted the relaxation with no throttle condition at all;
            # closing the gap needs a registry-side liveness read and is left as owed work.
            observed_remaining_growth_gib=(observed_growth_by_pid.get(pid)
                                           if (eligible and layer2_armed) else None),
        )
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
            governed_descendant=governed_desc,
        ))

    # (review-fix CRITICAL C, read side) Count FRESH pending admission reservations as tracked
    # jobs with zero current RSS and their reserved projected peak as growth headroom — so a second
    # launcher's admission decision sees a first launcher's just-admitted-but-not-yet-spawned job.
    # Synthetic rows are NEVER throttle-eligible (pid 0 — nothing to pause). A transient overlap
    # (reservation still pending while the freshly-spawned process already shows up via the ps
    # pattern) double-counts CONSERVATIVELY for the ~100 ms between Popen and row promotion.
    for r in pending_reservation_rows(registry_rows):
        p_label = str(r.get("label", "")) or "pending_reservation"
        p_cmd = r.get("cmd", "")
        p_cmd_str = " ".join(p_cmd) if isinstance(p_cmd, list) else str(p_cmd)
        p_proj = r.get("projected_peak_gib")
        try:
            p_peak = float(p_proj) if p_proj is not None else UNKNOWN_GROWTH_HEADROOM_GIB
        except (TypeError, ValueError):
            p_peak = UNKNOWN_GROWTH_HEADROOM_GIB
        jobs.append(TrackedJob(
            label=p_label, pid=0, pgid=0, cmd=p_cmd_str,
            priority=classify_run_priority(p_label, p_cmd_str),
            projected_peak_gib=max(0.0, p_peak), current_rss_gib=0.0, paused=False,
            throttle_eligible=False, own_group_leader=False,
        ))
    return jobs


def sum_active_growth_headroom_gib(
    jobs: Sequence[TrackedJob],
    *,
    heavy_min_projected_gib: float = HEAVY_MIN_PROJECTED_GIB,
) -> float:
    """Sum of remaining growth-to-peak over active HEAVY tracked jobs (admission input).

    Only jobs whose RESOLVED projected peak is >= ``heavy_min_projected_gib`` reserve growth
    headroom. A sub-heavy CONTROL-PLANE / telemetry daemon (the memory black-box, a dashboard
    HTTP server) has its CURRENT rss already counted in the vm_stat ``used`` baseline; reserving
    its projected "growth-to-peak" as HEAVY-job headroom on top is a double-count that pins the
    box (measured 2026-07-09: a 0.2-GiB dashboard with a 2.44-GiB recorded projection reserved
    ~2.2 GiB of phantom heavy-growth and REFUSED a real launch). This mirrors the launcher's
    already-reviewed ``_governed_active_jobs`` HEAVY definition (``launch_witness_run.HEAVY_MIN_
    PROJECTED_GIB``) so the two gates agree instead of the operator hand-reconciling them.
    Conservative by construction: a heavy training job (projection >> 4 GiB) or an unknown-peak
    job matching the heavy pattern (resolved to the 25-GiB default) is ALWAYS counted; only
    genuinely-small control-plane daemons are excluded, and their live RSS still sits in ``used``."""
    return float(sum(
        j.growth_headroom_gib for j in jobs
        if float(j.projected_peak_gib) >= float(heavy_min_projected_gib)
    ))


def sum_tracked_current_gib(jobs: Sequence[TrackedJob]) -> float:
    """Sum of tracked jobs' CURRENT group RSS, excluding governed DESCENDANTS — a descendant's
    subtree RSS is already inside its registered parent's group RSS, so counting both double-counts
    the workload, clamps ``baseline = used - tracked`` to 0, and UNDER-derives the safety floor
    (anti-conservative; measured live 2026-07-16: 127.17 GiB tracked on a 71 GiB-used box)."""
    return float(sum(j.current_rss_gib for j in jobs if not j.governed_descendant))


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


def _resume_order(jobs: Sequence[TrackedJob]) -> tuple:
    """Paused jobs in RESUME order: highest priority first, largest RSS as tie-break. PURE."""
    return tuple(sorted(jobs, key=lambda j: (-j.priority, -j.current_rss_gib)))


def decide_governor_action(
    *,
    level: str,
    consecutive_warn: int,
    consecutive_critical: int,
    jobs: Sequence[TrackedJob],
    warn_consecutive: int = DEFAULT_WARN_CONSECUTIVE,
    critical_consecutive: int = DEFAULT_CRITICAL_CONSECUTIVE,
    available_gib: float | None = None,
    resume_free_gib: float | None = None,
    paused_since_ts: Mapping[int, float] | None = None,
    now_ts: float | None = None,
    max_stop_duration_s: float | None = None,
) -> GovernorAction:
    """PURE governor policy (debounced, incremental, reversible, RE-ARMING):

      * ESCAPE HATCH — any job SIGSTOPped for >= ``max_stop_duration_s`` resumes UNCONDITIONALLY,
        whatever the pressure says. Outranks every other rung.
      * DERIVED-FREE RESUME — paused jobs resume when the pressure classifies NORMAL **or** when
        our OWN measured free memory reaches ``resume_free_gib``, even while the OS level still
        reads warn/critical.
      * DERIVED-FREE NO-PAUSE VETO — no pause fires while free >= ``resume_free_gib``, so a sticky
        OS level cannot drive actuation in EITHER direction and the two rungs cannot flap.
      * WARN sustained (>= warn_consecutive) -> pause ONE lowest-priority throttle-eligible job.
      * CRITICAL sustained (>= critical_consecutive) -> pause ONE (faster debounce); if nothing
        eligible -> escalate_alert (loud; hand off to memory_guard --watch / operator).
      * otherwise -> alert (loud, no action).

    NEVER kills; NEVER touches the control plane (throttle_eligible already excludes it).

    ddm_gb1 D2 (MEASURED INCIDENT 2026-08-15). The pre-fix policy resumed on ``level == "normal"``
    ALONE, and ``classify_pressure`` returns warn whenever the macOS pressure level reads >= 2 —
    a STICKY signal. Receipt: "GOVERNOR ALERT avail=40.4GiB level=warn" while five jobs (the mp2
    differential eval, the wc1 decode, the wd3 W0 warm train, the dashboard, three safe_run
    wrappers) sat stopped for 75+ minutes on a 40.5-GiB-free box. The guard's reference never
    re-armed — the spike-guard median-freeze genus (#304). ``resume_free_gib`` / ``available_gib``
    give the policy its OWN reference (see ``derived_resume_free_gib`` for the hysteresis
    derivation), and the escape hatch bounds the damage of ANY future reference that gets stuck.

    BACKWARD-COMPATIBLE: the three re-arm inputs default to ``None`` and their rungs are simply
    skipped when a caller does not supply them, so an old callsite behaves exactly as before. Every
    LIVE callsite supplies them (enforced by ``check_throttle_rearms_and_admission_reconciles``).
    """
    paused = [j for j in jobs if j.paused]

    # ── RUNG 0: escape hatch. A job held longer than the ceiling is a frozen measurement. ──
    if paused and paused_since_ts is not None:
        ceiling = (float(max_stop_duration_s) if max_stop_duration_s is not None
                   else DEFAULT_MAX_STOP_DURATION_S)
        now = time.time() if now_ts is None else float(now_ts)
        overdue = []
        for j in paused:
            since = paused_since_ts.get(j.pid)
            if since is None:
                continue
            held = now - float(since)
            if held >= ceiling:
                overdue.append((j, held))
        if overdue:
            longest = max(held for _, held in overdue)
            return GovernorAction(
                level, "resume",
                f"{ESCAPE_HATCH_REASON_TOKEN}: {len(overdue)} job(s) SIGSTOPped >= {ceiling:.0f}s "
                f"(longest {longest:.0f}s) — a stop this long is a FROZEN measurement, not "
                f"protection; resuming regardless of pressure level {level!r}",
                resume_targets=_resume_order([j for j, _ in overdue]))

    # ── RUNG 1: resume on the governor's OWN derived free-GiB reference (never the sticky OS level
    # alone). ``level == "normal"`` remains sufficient; it is no longer NECESSARY. ──
    derived_free_clear = (
        available_gib is not None and resume_free_gib is not None
        and float(available_gib) >= float(resume_free_gib)
    )
    if level == "normal" or derived_free_clear:
        if paused:
            why = ("pressure normal" if level == "normal" else
                   f"DERIVED free {float(available_gib):.1f}GiB >= resume threshold "
                   f"{float(resume_free_gib):.1f}GiB (OS pressure level still reads {level!r} — "
                   f"sticky; our own measurement governs)")
            return GovernorAction(level, "resume", f"{why} — resume paused job(s)",
                                  resume_targets=_resume_order(paused))
        if level == "normal":
            return GovernorAction(level, "none", "pressure normal — no paused jobs")
        return GovernorAction(
            level, "alert",
            f"OS pressure level reads {level!r} but DERIVED free {float(available_gib):.1f}GiB >= "
            f"{float(resume_free_gib):.1f}GiB — NO pause (a sticky OS level may not actuate the "
            f"throttle on its own); alerting only")
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
    """fcntl-locked JSONL telemetry append via the canonical .omx/state helper
    (tac.jsonl_store.append_locked_jsonl; see
    .omx/research/fcntl_lock_canonicalization_plan_20260710.md Batch 1). The
    ``OSError`` guard is preserved verbatim — telemetry must never take down the governor."""
    import datetime as _dt

    row = dict(row)
    row.setdefault("ts", _dt.datetime.now(_dt.UTC).isoformat())
    try:
        append_locked_jsonl(ledger_path, row)
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
        # ddm_gb1 D1: NAMED control-plane processes only (was ``used - tracked`` = total-used incl.
        # file cache — the 92-GiB clamp spam on a 40.5-GiB-free box).
        measured_cp_rss_gib=measured_control_plane_rss_gib(),
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
    # SOLE-WORKLOAD when NO OTHER admitted HEAVY tracked job is in flight (projected_peak >=
    # HEAVY_MIN) — the new job is the only heavy workload. Sub-heavy control-plane/telemetry
    # daemons (dashboard, blackbox) do not count as concurrency (same HEAVY_MIN the launcher's
    # _governed_active_jobs + sum_active_growth_headroom_gib use). Relaxes the LAUNCH-time floor
    # to the operator's sole-workload policy; the runtime guard-bands still protect bursts.
    sole_workload = not any(
        float(j.projected_peak_gib) >= HEAVY_MIN_PROJECTED_GIB for j in jobs
    )
    # ADMISSION decision basis (2026-07-16 fix): the reclaimable-aware TRUE-committed used, so the
    # gate is equivalent to ``projected_new + active_growth <= reclaimable_available - floor``.
    # Fallback to the legacy free+inactive basis when the reclaimable figures were not validated
    # (missing counters / identity or bound violation / hand-built snapshot) — never trust the
    # generous figure unvalidated. The throttle/watchdog paths (classify_pressure on
    # ``available_gib``) deliberately keep the legacy conservative basis — this fix is scoped to
    # ADMISSION accounting only.
    used_for_admission = (snapshot.used_committed_gib if snapshot.reclaimable_ok
                          else snapshot.used_gib)
    ceiling = compute_adaptive_ceiling(
        total_gib=snapshot.total_gib, used_gib=used_for_admission,
        tracked_current_gib=sum_tracked_current_gib(jobs),
        floor_override_gib=floor_override_gib, floor_mode=floor_mode,
        sole_workload=sole_workload,
    )
    decision = admission_decision(
        projected_new_gib=projected_new_gib, system_used_gib=used_for_admission,
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
        # Same ADMISSION basis as live_admission_decision (reclaimable-aware committed used).
        ceiling = compute_adaptive_ceiling(
            total_gib=snap.total_gib,
            used_gib=(snap.used_committed_gib if snap.reclaimable_ok else snap.used_gib),
            tracked_current_gib=sum_tracked_current_gib(jobs),
            floor_override_gib=floor_override, floor_mode=args.safety_floor_mode)
        print(json.dumps({
            "snapshot": snap.to_json(), "ceiling": ceiling.to_json(),
            "active_jobs": [j.to_json() for j in jobs],
            "active_growth_headroom_gib": round(sum_active_growth_headroom_gib(jobs), 2),
        }, indent=2))
        return 0

    if args.admit:
        # ddm_gb1 D4 (sister of the safe_run fix): converge the registry this decision counts to
        # GROUND TRUTH first. A dead ``running`` row charges UNKNOWN_GROWTH_HEADROOM_GIB (25 GiB)
        # of phantom active growth; MEASURED 2026-08-15, three dead rows = 100 GiB of phantom
        # growth that refused a real launch twice. Fail-OPEN: the reconcile is a self-clean, the
        # admission decision below is the real (fail-CLOSED) gate.
        try:
            import spawn_durable_daemon as _sdd  # tools/ is on sys.path (same dir)
            _sdd.reconcile_dead_daemons(verbose=False)
        except Exception:
            pass
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
        # ddm_gb1 D2: the CLI tick actuates the same throttle as the daemon, so it carries the same
        # re-arm inputs (derived-free resume reference + escape hatch). A dry CLI tick that decided
        # on the sticky OS level alone would report a different action than the live governor.
        try:
            import memory_blackbox as _mbb  # tools/ is on sys.path (same dir)
            _paused_since = _mbb.paused_since_ts()
        except Exception:
            _mbb = None
            _paused_since = {}
        action = decide_governor_action(
            level=level, consecutive_warn=cw, consecutive_critical=cc, jobs=jobs,
            available_gib=governing_free_gib(snap),
            resume_free_gib=derived_resume_free_gib(snap.total_gib),
            paused_since_ts=_paused_since,
            max_stop_duration_s=resolve_max_stop_duration_s())
        acted = []
        if args.apply and action.action == "pause" and action.target is not None:
            # ddm_mb1: MIRROR the daemon's ledger discipline at this second SIGSTOP entry point.
            # Pre-fix, a `--apply` pause here SIGSTOPped a job and recorded NOTHING, so the stop was
            # invisible to the exit sweep, to `memory_blackbox.py --resume-stopped`, and to the
            # escape hatch (which ages pids by their ledger timestamp). That is the same "no
            # exit-resume" defect the 2026-08-15 incident named, just reached through the CLI
            # instead of the daemon — and it strands a job precisely when no daemon is running to
            # adopt it. Record BEFORE signalling: a crash between the two must leave a sweepable
            # ledger, never a stranded pid (recording a pid we then fail to stop is harmless — the
            # sweep skips anything not in state T).
            # INTERACTION, stated so it does not surprise: once recorded, this pause is sweepable,
            # so a later black-box daemon start will SIGCONT it. That is the SAFE direction and it
            # is intended — this CLI is one-shot, so its pause is stranded the moment the process
            # exits. A resumed job at worst uses memory; a stopped one is a frozen measurement.
            if _mbb is not None:
                with contextlib.suppress(Exception):
                    _mbb.record_stopped_pids(_mbb._stopped_scope_pids(action.target),
                                             label=action.target.label)
            acted.append(("pause", action.target.label, pause_job(action.target)))
        elif args.apply and action.action == "resume":
            for j in action.resume_targets:
                acted.append(("resume", j.label, resume_job(j)))
                if _mbb is not None:
                    with contextlib.suppress(Exception):
                        _mbb.record_resumed(pids=_mbb._stopped_scope_pids(j), label=j.label)
        print(json.dumps({"snapshot": snap.to_json(), "action": action.to_json(),
                          "applied": acted}, indent=2))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
