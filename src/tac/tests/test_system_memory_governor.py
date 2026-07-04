# SPDX-License-Identifier: MIT
"""Tests for tools/system_memory_governor.py — the SYSTEM-aware memory governor (P0 crash protection).

These tests make the guard TRUSTWORTHY (despite eyeballing vm_stat/ps by hand being error-prone):
  * the ACCOUNTING is reconciled from FIXED captured kernel counters and asserted to EXACT GiB, so a
    parse bug (wrong page size / wrong field) fails a test, never ships;
  * a wrong-page-size / free-mismatch / total-mismatch / overcount FAILS SAFE (refuses admission);
  * the ADMISSION math REFUSES the exact SUM-over-RAM multi-job overflow that crashed the machine;
  * the THROTTLE target selection is pure + control-plane-safe by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import pytest  # noqa: E402
import system_memory_governor as gov  # noqa: E402


@pytest.fixture(autouse=True)
def _no_floor_env(monkeypatch):
    """Keep every test hermetic vs a leaked TAC_GOV_SAFETY_FLOOR_GIB (the derived-floor default
    path consults the env override; BUILD #298)."""
    monkeypatch.delenv(gov.SAFETY_FLOOR_ENV, raising=False)


# EXACT captured live counters (2026-07-02; page size 16384, hw.memsize 137438953472 = 128.0 GiB).
_SNAP = dict(
    page_size=16384, total_bytes=137438953472,
    vm_free_pages=6572823, vm_active_pages=728620, vm_inactive_pages=467151,
    vm_speculative_pages=279768, vm_wired_pages=253178, vm_compressor_pages=0, vm_throttled_pages=0,
    sysctl_free_pages=6572785, mempressure_total_bytes=137438953472,
    psutil_available_bytes=119916593152,
)


def _job(label, *, pid=1000, priority=0, peak=60.0, rss=30.0, paused=False,
         eligible=True, leader=True):
    return gov.TrackedJob(
        label=label, pid=pid, pgid=pid, cmd=f"python {label}.py", priority=priority,
        projected_peak_gib=peak, current_rss_gib=rss, paused=paused,
        throttle_eligible=eligible, own_group_leader=leader)


# ── accounting (unit-tested against fixed snapshots to EXACT GiB) ──────────────────────────────
def test_reconcile_exact_captured_snapshot():
    a = gov.reconcile_memory_accounting(**_SNAP)
    assert abs(a.total_gib - 128.0) < 1e-6
    assert abs(a.available_primary_gib - 107.42) < 0.05      # (free+inactive) exact
    assert abs(a.used_gib - 20.58) < 0.05                    # total - available
    assert abs(a.free_gib - 100.29) < 0.05
    assert abs(a.wired_gib - 3.86) < 0.05
    assert abs(a.closure_gib - 1.33) < 0.10                  # legit unaccounted gap ~1.3 GiB
    assert a.closure_ok and a.cross_validated and not a.fail_safe
    assert a.validation_notes == ()


def test_reconcile_wrong_page_size_fails_safe():
    """A wrong page-size parse (4096 vs 16384) blows closure out by ~96 GiB -> FAIL SAFE."""
    bad = dict(_SNAP)
    bad["page_size"] = 4096
    a = gov.reconcile_memory_accounting(**bad)
    assert not a.closure_ok
    assert a.fail_safe
    assert a.closure_gib > 50.0
    assert a.available_gib < a.available_primary_gib  # fail-safe reduced available


def test_reconcile_free_page_mismatch_fails_safe():
    bad = dict(_SNAP)
    bad["sysctl_free_pages"] = 6572823 - 500_000  # ~7.6 GiB fewer free pages than vm_stat
    a = gov.reconcile_memory_accounting(**bad)
    assert not a.cross_validated
    assert a.fail_safe


def test_reconcile_total_mismatch_fails_safe():
    bad = dict(_SNAP)
    bad["mempressure_total_bytes"] = 137438953472 - 10 * gov._GIB  # 10 GiB total disagreement
    a = gov.reconcile_memory_accounting(**bad)
    assert not a.cross_validated
    assert a.fail_safe


def test_reconcile_overcount_vs_psutil_fails_safe():
    """If our CONSERVATIVE available exceeds psutil's GENEROUS available, a parse overcount is caught."""
    bad = dict(_SNAP)
    bad["psutil_available_bytes"] = int((107.42 - 6.0) * gov._GIB)  # psutil << our conservative reading
    a = gov.reconcile_memory_accounting(**bad)
    assert not a.cross_validated
    assert a.fail_safe


def test_reconcile_no_optional_sources_still_validates_via_closure():
    a = gov.reconcile_memory_accounting(
        page_size=16384, total_bytes=137438953472,
        vm_free_pages=6572823, vm_active_pages=728620, vm_inactive_pages=467151,
        vm_speculative_pages=279768, vm_wired_pages=253178, vm_compressor_pages=0)
    assert a.closure_ok and a.cross_validated and not a.fail_safe


def test_reconcile_psutil_more_generous_is_not_a_failure():
    """psutil.available (incl. speculative/purgeable) is legitimately > our conservative reading —
    that must NOT trip the (one-sided) overcount check."""
    a = gov.reconcile_memory_accounting(**_SNAP)  # psutil 111.68 > primary 107.42
    assert a.cross_validated and not a.fail_safe


# ── adaptive ceiling (BUILD #298: the margin is the tier-scaled DERIVED floor) ──────────────────
def test_safety_margin_is_tier_scaled_derived():
    """The hardcoded 8 GiB floor is GONE: with no measured control plane the static 0.08*T leg
    (>= ABS_MIN 2.0) applies — @128 unchanged vs legacy (10.24); @64 now 5.12 (was 8.0); @8 now
    2.0 (the old floor equalled the ENTIRE box -> ceiling 0.0, budget -5.99, refuse-everything)."""
    assert gov.compute_safety_margin_gib(128.0) == 128.0 * 0.08   # 10.24 — backward-compatible
    assert gov.compute_safety_margin_gib(64.0) == 64.0 * 0.08     # 5.12 (legacy 8.0 hardcode gone)
    assert gov.compute_safety_margin_gib(8.0) == 2.0              # ABS_MIN (was 8.0 = whole box)
    # legacy kwargs reproduce the old max(floor, frac*T) — but cap-clamped (never eats the box):
    assert gov.compute_safety_margin_gib(64.0, floor_gib=8.0, frac=0.08) == 8.0
    assert gov.compute_safety_margin_gib(8.0, floor_gib=8.0, frac=0.08) == 4.0   # capped 0.5*8


def test_adaptive_ceiling_math_maxes_out_128():
    # Explicit legacy margin: identical arithmetic to pre-#298 (backward-compat surface).
    c = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=24.0, tracked_current_gib=8.0,
                                     safety_margin_gib=10.24)
    assert c.baseline_gib == 16.0                    # used - tracked = OS+control-plane
    assert abs(c.adaptive_ceiling_gib - 117.76) < 1e-6  # total - 10.24 margin
    assert abs(c.training_budget_gib - 101.76) < 1e-6   # ceiling - baseline (>> old blind 90 GB cap)


def test_adaptive_ceiling_default_is_derived_and_follows_control_plane():
    """Default path: the floor FOLLOWS the measured control plane (baseline doubles as the
    dynamical leg's input): cp=16 -> floor 16+6.4=22.4 -> ceiling 105.6, budget 89.6; a realistic
    live scenario (used 70, tracked 62 true GiB -> cp 8) -> floor 14.4 -> ceiling 113.6."""
    d = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=24.0, tracked_current_gib=8.0)
    assert abs(d.safety_margin_gib - 22.4) < 1e-9
    assert abs(d.adaptive_ceiling_gib - 105.6) < 1e-9
    assert abs(d.training_budget_gib - 89.6) < 1e-9
    assert d.floor_decomposition is not None
    assert d.floor_decomposition["winning_leg"] == "measured_cp"
    live = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=70.0, tracked_current_gib=62.0)
    assert abs(live.safety_margin_gib - 14.4) < 1e-9
    assert abs(live.adaptive_ceiling_gib - 113.6) < 1e-9
    assert live.safety_margin_gib >= 10.0   # >= the operator-policy 10 GiB on the 128 box, always


def test_adaptive_ceiling_derived_floor_at_least_10_on_128_for_any_cp():
    """Backward-compat invariant (operator memory policy): on a 128 GiB box the derived floor is
    >= 10 GiB in EVERY scenario (static leg 10.24 is the minimum; the measured leg only raises)."""
    for cp in (0.0, 1.0, 3.84, 8.0, 16.0, 40.0, 100.0):
        d = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=cp, tracked_current_gib=0.0)
        assert d.safety_margin_gib >= 10.0, f"cp={cp} floor={d.safety_margin_gib}"


# ── admission (the crash math) ──────────────────────────────────────────────────────────────────
def test_project_system_used_no_double_count():
    # used already includes active jobs' current RSS; we add only their REMAINING growth + the new job.
    p = gov.project_system_used_after_launch(
        system_used_gib=90.0, active_growth_headroom_gib=10.0, projected_new_gib=20.0)
    assert p == 120.0


def test_admission_refuses_sum_over_ceiling_the_crash():
    """R1 (~67 GB) already running + a 2nd job whose peak pushes the SUM > ceiling -> REFUSE."""
    c = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=70.0, tracked_current_gib=67.0)
    d = gov.admission_decision(projected_new_gib=60.0, system_used_gib=70.0,
                               active_growth_headroom_gib=0.0, ceiling=c)
    assert not d.admit
    assert d.projected_system_used_gib == 130.0
    assert d.headroom_after_gib < 0


def test_admission_admits_when_it_fits():
    c = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=20.0, tracked_current_gib=0.0)
    d = gov.admission_decision(projected_new_gib=60.0, system_used_gib=20.0,
                               active_growth_headroom_gib=0.0, ceiling=c)
    assert d.admit
    assert d.headroom_after_gib > 0


def test_admission_fail_safe_forces_refuse_even_when_it_would_fit():
    c = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=20.0, tracked_current_gib=0.0)
    d = gov.admission_decision(projected_new_gib=10.0, system_used_gib=20.0,
                               active_growth_headroom_gib=0.0, ceiling=c, fail_safe=True)
    assert not d.admit
    assert "FAIL-SAFE" in d.reason


# ── priority + throttle selection ───────────────────────────────────────────────────────────────
def test_priority_pointer_movers_outrank_probes():
    assert gov.classify_run_priority("levelset_witness_sealed_205") >= 90
    assert gov.classify_run_priority("levelset_n600_witness") >= 50
    assert gov.classify_run_priority("ev1_descent_probe") < 0
    assert gov.classify_run_priority("byte_close_bsdtar") < 0
    # the pointer-mover strictly outranks the probe (throttle/shed the probe first)
    assert gov.classify_run_priority("sealed_205") > gov.classify_run_priority("descent_probe")


def test_select_throttle_target_lowest_priority_first():
    jobs = [_job("sealed_205", pid=1, priority=100), _job("probe", pid=2, priority=-60),
            _job("witness", pid=3, priority=50)]
    t = gov.select_throttle_target(jobs)
    assert t is not None and t.label == "probe"


def test_select_throttle_target_tiebreak_largest_rss():
    jobs = [_job("a", pid=1, priority=0, rss=10.0), _job("b", pid=2, priority=0, rss=40.0)]
    t = gov.select_throttle_target(jobs)
    assert t.label == "b"  # same priority -> pause the biggest to halt the most growth


def test_select_throttle_target_skips_ineligible_and_paused():
    jobs = [_job("cp", pid=1, priority=-100, eligible=False),  # control-plane: never
            _job("paused", pid=2, priority=-90, paused=True),  # already paused
            _job("ok", pid=3, priority=10)]
    t = gov.select_throttle_target(jobs)
    assert t is not None and t.label == "ok"


def test_select_throttle_target_none_when_nothing_eligible():
    jobs = [_job("cp", pid=1, eligible=False), _job("p", pid=2, paused=True)]
    assert gov.select_throttle_target(jobs) is None


def test_select_resume_target_highest_priority_first():
    jobs = [_job("lo", pid=1, priority=-50, paused=True), _job("hi", pid=2, priority=80, paused=True)]
    assert gov.select_resume_target(jobs).label == "hi"


# ── pressure classification + governor action ───────────────────────────────────────────────────
def _snapshot(*, available, pressure_level=1):
    return gov.SystemMemorySnapshot(
        total_gib=128.0, available_gib=available, used_gib=128.0 - available, free_gib=available,
        wired_gib=0.0, compressor_gib=0.0, swap_used_gib=0.0, pressure_level=pressure_level,
        load1=0.0, load5=0.0, load15=0.0)


def test_classify_pressure_thresholds():
    assert gov.classify_pressure(_snapshot(available=50.0)) == "normal"
    assert gov.classify_pressure(_snapshot(available=12.0)) == "warn"     # < 15
    assert gov.classify_pressure(_snapshot(available=5.0)) == "critical"  # < 8
    # OS pressure level overrides even when available looks fine
    assert gov.classify_pressure(_snapshot(available=50.0, pressure_level=4)) == "critical"
    assert gov.classify_pressure(_snapshot(available=50.0, pressure_level=2)) == "warn"


def test_decide_action_warn_pauses_after_debounce():
    jobs = [_job("probe", priority=-60)]
    a = gov.decide_governor_action(level="warn", consecutive_warn=3, consecutive_critical=0, jobs=jobs)
    assert a.action == "pause" and a.target.label == "probe"


def test_decide_action_warn_debounces_to_alert():
    jobs = [_job("probe", priority=-60)]
    a = gov.decide_governor_action(level="warn", consecutive_warn=1, consecutive_critical=0, jobs=jobs)
    assert a.action == "alert"


def test_decide_action_critical_pauses_fast():
    jobs = [_job("probe", priority=-60)]
    a = gov.decide_governor_action(level="critical", consecutive_warn=0, consecutive_critical=2, jobs=jobs)
    assert a.action == "pause"


def test_decide_action_critical_escalates_when_nothing_eligible():
    jobs = [_job("cp", eligible=False)]
    a = gov.decide_governor_action(level="critical", consecutive_warn=0, consecutive_critical=3, jobs=jobs)
    assert a.action == "escalate_alert"  # loud; never kills the control plane


def test_decide_action_normal_resumes_paused():
    jobs = [_job("probe", priority=-60, paused=True)]
    a = gov.decide_governor_action(level="normal", consecutive_warn=0, consecutive_critical=0, jobs=jobs)
    assert a.action == "resume" and a.resume_targets and a.resume_targets[0].label == "probe"


def test_decide_action_normal_no_paused_is_none():
    a = gov.decide_governor_action(level="normal", consecutive_warn=0, consecutive_critical=0,
                                   jobs=[_job("x")])
    assert a.action == "none"


# ── enforce-mode + misc ─────────────────────────────────────────────────────────────────────────
def test_admission_enforcing_defaults_advisory(monkeypatch):
    monkeypatch.delenv(gov.ADMISSION_ENFORCE_ENV, raising=False)
    assert gov.admission_enforcing() is False
    monkeypatch.setenv(gov.ADMISSION_ENFORCE_ENV, "1")
    assert gov.admission_enforcing() is True
    monkeypatch.setenv(gov.ADMISSION_ENFORCE_ENV, "0")
    assert gov.admission_enforcing() is False


def test_is_paused_state():
    assert gov.is_paused_state("T")
    assert gov.is_paused_state("T+")
    assert not gov.is_paused_state("R")
    assert not gov.is_paused_state("S")


def test_matches_our_jobs():
    assert gov.matches_our_jobs(".venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py")
    assert gov.matches_our_jobs("python tools/levelset_byte_close_and_eval.py")
    assert not gov.matches_our_jobs("node /Applications/Claude.app/cli.js")


# ── enforce arming: env var OR durable flag file (survives shell-env reset between Bash calls) ────
def test_admission_enforcing_default_advisory(monkeypatch, tmp_path):
    """Default is ADVISORY: no env, no flag file -> False (logs WOULD-REFUSE, does not block)."""
    monkeypatch.delenv(gov.ADMISSION_ENFORCE_ENV, raising=False)
    monkeypatch.setattr(gov, "_ADMISSION_ENFORCE_FLAG", tmp_path / "absent.flag")
    assert gov.admission_enforcing() is False


def test_admission_enforcing_env_arms(monkeypatch, tmp_path):
    monkeypatch.setattr(gov, "_ADMISSION_ENFORCE_FLAG", tmp_path / "absent.flag")
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv(gov.ADMISSION_ENFORCE_ENV, truthy)
        assert gov.admission_enforcing() is True
    for falsy in ("0", "no", "off", ""):
        monkeypatch.setenv(gov.ADMISSION_ENFORCE_ENV, falsy)
        assert gov.admission_enforcing() is False


def test_admission_enforcing_durable_flag_file_arms(monkeypatch, tmp_path):
    """The durable file arms enforce even with NO env var (survives the shell-env reset)."""
    monkeypatch.delenv(gov.ADMISSION_ENFORCE_ENV, raising=False)
    flag = tmp_path / "admission_enforce.flag"
    monkeypatch.setattr(gov, "_ADMISSION_ENFORCE_FLAG", flag)
    # truthy first line (comment on a later line is ignored) -> armed
    flag.write_text("1\n# armed after independent review\n")
    assert gov.admission_enforcing() is True
    # explicit disarm content -> advisory; empty -> advisory
    flag.write_text("0\n")
    assert gov.admission_enforcing() is False
    flag.write_text("")
    assert gov.admission_enforcing() is False


def test_admission_enforcing_flag_read_error_fails_advisory(monkeypatch, tmp_path):
    """An unreadable flag path must not crash the launcher (fail to ADVISORY, not raise)."""
    monkeypatch.delenv(gov.ADMISSION_ENFORCE_ENV, raising=False)
    # a directory at the flag path -> read_text raises OSError -> caught -> False
    d = tmp_path / "flag_is_a_dir.flag"
    d.mkdir()
    monkeypatch.setattr(gov, "_ADMISSION_ENFORCE_FLAG", d)
    assert gov.admission_enforcing() is False
