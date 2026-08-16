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

import json
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


def test_adaptive_ceiling_default_is_derived_and_follows_control_plane(monkeypatch):
    """Default path: the floor FOLLOWS the measured control plane (baseline doubles as the
    dynamical leg's input): cp=16 -> floor 16+6.4=22.4 -> ceiling 105.6, budget 89.6; a realistic
    live scenario (used 70, tracked 62 true GiB -> cp 8) -> floor 14.4 -> ceiling 113.6.
    Operator-ceiling policy (2026-07-21, RAISE-only) disabled here: this test verifies the
    DERIVED-floor path specifically."""
    monkeypatch.setenv(gov.OPERATOR_CEILING_ENV, "0")
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


# ── sole-workload floor relaxation (2026-07-09; adversarial-review-hardened NIT 1 + NIT 3) ─────────
def test_sole_workload_drops_the_double_count_but_keeps_proportional_burst_reserve():
    """SOLE workload: floor = ch + 0.5*cp (non-doubled proportional burst reserve), NOT the concurrent
    cp+ch. cp=16 -> 6.4 + 8 = 14.4; and it is ALWAYS strictly below the concurrent floor (kills the
    double-count) yet ALWAYS scales with cp (keeps burst protection — NIT 1)."""
    for cp in (8.0, 16.0, 27.0, 40.0):
        conc = gov.derive_safety_floor(total_gib=128.0, measured_cp_rss_gib=cp)
        sole = gov.derive_safety_floor(total_gib=128.0, measured_cp_rss_gib=cp, sole_workload=True)
        assert sole.floor_gib <= conc.floor_gib, f"cp={cp}: sole must not exceed concurrent"
        if cp >= 8.0:  # above where the burst leg overtakes the 10.24 static leg
            assert sole.winning_leg == "sole_workload_burst_reserve"
            assert abs(sole.floor_gib - (gov.cp_headroom_gib(128.0) + 0.5 * cp)) < 1e-9


def test_sole_workload_reproduces_operator_policy_at_measured_baseline(monkeypatch):
    # Operator-ceiling policy (2026-07-21, RAISE-only) disabled: this test asserts exact
    # DERIVED-floor ceiling arithmetic on a 128 GiB scenario.
    monkeypatch.setenv("TAC_GOV_OPERATOR_CEILING_GIB", "0")
    """At the ~27 GiB live control-plane baseline the sole reserve lands ~20 GiB = the operator's
    documented >=10 fail-safe + ~10 margin policy — and the run ADMITS (budget > a 71.5 GiB run)."""
    d = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=27.4, tracked_current_gib=0.2,
                                     sole_workload=True)
    assert 19.0 <= d.safety_margin_gib <= 21.0, d.safety_margin_gib
    assert d.training_budget_gib > 71.5, d.training_budget_gib
    assert d.floor_decomposition["winning_leg"] == "sole_workload_burst_reserve"


def test_sole_workload_default_false_is_bit_identical_to_298():
    """Regression lock: sole_workload defaults False and reproduces the #298 concurrent floor EXACTLY
    (the reviewer's bit-identical-concurrent requirement)."""
    for cp in (0.0, 8.0, 16.0, 40.0):
        default = gov.derive_safety_floor(total_gib=128.0, measured_cp_rss_gib=cp)
        explicit = gov.derive_safety_floor(total_gib=128.0, measured_cp_rss_gib=cp, sole_workload=False)
        assert default.floor_gib == explicit.floor_gib
        assert default.winning_leg == explicit.winning_leg


def test_sole_workload_still_honors_abs_min_and_cap():
    """Clamps still bind under sole-workload: never below abs_min, never above cap_frac*T."""
    tiny = gov.derive_safety_floor(total_gib=8.0, measured_cp_rss_gib=0.0, sole_workload=True)
    assert tiny.floor_gib >= gov.ABS_MIN_SAFETY_FLOOR_GIB
    assert tiny.floor_gib <= gov.DEFAULT_FLOOR_CAP_FRAC * 8.0
    huge = gov.derive_safety_floor(total_gib=128.0, measured_cp_rss_gib=200.0, sole_workload=True)
    assert huge.floor_gib <= gov.DEFAULT_FLOOR_CAP_FRAC * 128.0


def test_live_admission_sole_workload_flag_from_heavy_jobs():
    """live_admission_decision derives sole_workload from HEAVY tracked jobs only: a sub-heavy
    control-plane daemon (proj < 4) leaves it sole; a heavy job (proj >= 4) makes it concurrent."""
    snap = gov.read_system_memory_snapshot()
    dash = gov.TrackedJob(label="dash", pid=111111, pgid=111111, cmd="dashboard", priority=5,
                          projected_peak_gib=2.44, current_rss_gib=0.2, paused=False,
                          throttle_eligible=False, own_group_leader=True)
    heavy = gov.TrackedJob(label="train", pid=222222, pgid=222222, cmd="train", priority=5,
                           projected_peak_gib=50.0, current_rss_gib=20.0, paused=False,
                           throttle_eligible=True, own_group_leader=True)
    sole = gov.live_admission_decision(projected_new_gib=10.0, snapshot=snap, jobs=[dash])
    conc = gov.live_admission_decision(projected_new_gib=10.0, snapshot=snap, jobs=[heavy])
    assert sole.ceiling.floor_decomposition["winning_leg"] in (
        "sole_workload_burst_reserve", "static_frac_sole_workload")
    assert conc.ceiling.floor_decomposition["winning_leg"] in ("measured_cp", "static_frac", "abs_min")


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
def test_admission_enforcing_defaults_advisory(monkeypatch, tmp_path):
    # hermetic vs the DURABLE enforce flag (this machine has the real flag ARMED; the env-var
    # semantics under test must be isolated from it — same pattern as the sister flag tests below).
    monkeypatch.setattr(gov, "_ADMISSION_ENFORCE_FLAG", tmp_path / "absent.flag")
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


# ── conservative unknown-peak growth default (review-fix CRITICAL B) ─────────────────────────────
def test_resolve_projected_peak_recorded_projection_wins():
    # a valid recorded projection is authoritative (floored at current RSS)
    assert gov.resolve_projected_peak_gib(72.0, 60.0) == 72.0
    assert gov.resolve_projected_peak_gib(50.0, 60.0) == 60.0   # never below current RSS


def test_resolve_projected_peak_unknown_defaults_conservative_not_zero_growth():
    """The OLD backwards fallback assumed an unknown-peak job grows by ZERO; the fix charges it
    current + UNKNOWN_GROWTH_HEADROOM_GIB (matching the launch paths' 25 GiB default)."""
    assert gov.UNKNOWN_GROWTH_HEADROOM_GIB == 25.0
    assert gov.resolve_projected_peak_gib(None, 60.0) == 85.0
    assert gov.resolve_projected_peak_gib(None, 0.0) == 25.0
    # malformed recorded value -> treated as unknown -> conservative
    assert gov.resolve_projected_peak_gib("not-a-number", 10.0) == 35.0


def test_resolve_projected_peak_protection_infra_zero_growth():
    """Protection infra (blackbox/guard/governor) is deliberately unprojected + tiny: a permanent
    +25 GiB phantom would poison every admission decision."""
    for tok in ("memory_blackbox.py", "memory_guard.py", "system_memory_governor.py"):
        assert gov.resolve_projected_peak_gib(None, 0.5, cmd=f"python tools/{tok} --daemon") == 0.5
    # ...but a recorded projection on infra still wins
    assert gov.resolve_projected_peak_gib(2.0, 0.5, cmd="python tools/memory_blackbox.py") == 2.0


def test_resolve_projected_peak_governed_descendant_zero_growth():
    """The trainer INSIDE a registered daemon->safe_run tree (own session -> separate ps candidate)
    is already projected by its parent's registry row — charging +25 GiB again would double-count
    and could false-refuse small jobs while the live #205 runs."""
    assert gov.resolve_projected_peak_gib(None, 60.0, governed_descendant=True) == 60.0


def test_unknown_growth_headroom_matches_launch_path_default():
    """The conservative default MUST match the admission gate's own unknown-projection fallback
    (spawn_durable_daemon --projected-gb default 25.0 / governor CLI --projected-gib default 25.0)
    so an unregistered job is charged the same projection it would have declared at launch."""
    assert gov.UNKNOWN_GROWTH_HEADROOM_GIB == 25.0


# ── 2026-07-11 phantom-reservation fix: unregistered ps-only sub-floor match => ZERO growth ────────
def test_resolve_unregistered_ps_only_below_floor_zero_growth():
    """An UNREGISTERED ps-only token match (a grep/editor/launcher/short probe whose argv merely
    CONTAINS a pattern token) below the material RSS floor is charged ZERO growth — its RSS is
    already in the vm_stat `used` baseline. This is the phantom-refusal fix (2026-07-11)."""
    assert gov.MATERIAL_UNREGISTERED_RSS_FLOOR_GIB == 2.0
    # tiny incidental match -> current (zero growth), NOT current+25.
    assert gov.resolve_projected_peak_gib(None, 0.0002, unregistered_ps_only=True) == 0.0002
    assert gov.resolve_projected_peak_gib(None, 1.99, unregistered_ps_only=True) == 1.99


def test_resolve_unregistered_ps_only_at_or_above_floor_still_charged():
    """SAFETY PRESERVED: a materially-resident unregistered match (>= floor) STILL gets current+25 —
    the SUM-over-RAM crash is driven by multi-GiB resident jobs, which stay fully counted."""
    assert gov.resolve_projected_peak_gib(None, 2.0, unregistered_ps_only=True) == 27.0
    assert gov.resolve_projected_peak_gib(None, 30.0, unregistered_ps_only=True) == 55.0


def test_resolve_default_and_registered_paths_unchanged_by_fix():
    """The relaxation is SCOPED to unregistered_ps_only=True. Default False (registered row without a
    projection) is bit-identical to the pre-fix behavior; a recorded projection always wins."""
    assert gov.resolve_projected_peak_gib(None, 0.0002) == 0.0002 + 25.0   # default False unchanged
    assert gov.resolve_projected_peak_gib(None, 0.0) == 25.0
    # a recorded projection wins even for a ps-only candidate (registered heavy job at low RSS).
    assert gov.resolve_projected_peak_gib(60.0, 0.0002, unregistered_ps_only=True) == 60.0


def _ps(pid, command, *, rss_kb=200, ppid=1, pgid=None, start_identity=None):
    return gov._mg.ProcessSample(pid=pid, ppid=ppid, rss_kb=rss_kb, command=command,
                                 pgid=pgid if pgid is not None else pid,
                                 start_identity=start_identity)


@pytest.mark.skipif(gov._mg is None, reason="memory_guard unavailable (fail-safe: no tracked jobs)")
def test_incidental_ps_token_match_charges_zero_phantom_growth():
    """THE reproduced false-positive: a nearly-idle machine, ZERO registered jobs, but ~8 incidental
    processes whose argv contains a pattern token (grep/editor/launcher/probe). Pre-fix each was
    charged +25 GiB => ~200 GiB phantom projected growth that FALSE-REFUSED a legit launch. Post-fix
    each sub-floor unregistered match contributes ZERO growth."""
    incidental = {
        901: _ps(901, "ugrep -R byte_close src/", rss_kb=192),
        902: _ps(902, "python -c 'import inflate.py'", rss_kb=8000),
        903: _ps(903, "vim experiments/train_witness_realized_through_R_mlx.py", rss_kb=40000),
        904: _ps(904, "bash launch_split_by_head_basin.sh", rss_kb=3000),
        905: _ps(905, "tail -f logs/descent_probe.log", rss_kb=1500),
    }
    jobs = gov.list_tracked_jobs(samples=incidental, registry_rows=[], self_pid=1)
    # every incidental match is tracked (throttle-discoverable) but charged ZERO heavy growth.
    assert len(jobs) == len(incidental)
    assert gov.sum_active_growth_headroom_gib(jobs) == 0.0
    for j in jobs:
        assert j.growth_headroom_gib == 0.0


@pytest.mark.skipif(gov._mg is None, reason="memory_guard unavailable (fail-safe: no tracked jobs)")
def test_materially_resident_unregistered_job_still_charged():
    """SAFETY: an unregistered ps-match that IS materially resident (~30 GiB) still reserves
    current+25 growth headroom — the crash-relevant case is never relaxed."""
    big = {700: _ps(700, "python levelset_byte_close_and_eval.py", rss_kb=31_457_280)}  # ~30 GiB
    jobs = gov.list_tracked_jobs(samples=big, registry_rows=[], self_pid=1)
    assert len(jobs) == 1
    assert jobs[0].current_rss_gib > 25.0
    assert gov.sum_active_growth_headroom_gib(jobs) == gov.UNKNOWN_GROWTH_HEADROOM_GIB  # +25 charged


@pytest.mark.skipif(gov._mg is None, reason="memory_guard unavailable (fail-safe: no tracked jobs)")
def test_registered_heavy_job_low_rss_still_fully_charged():
    """The fix must NOT relax a REGISTERED heavy job: a running row with projected_peak=60 at low
    current RSS still reserves its full remaining growth (recorded projection wins)."""
    reg = [{"label": "n600", "pid": 800, "pgid": 800, "status": "running",
            "cmd": ["python", "train_levelset_witness_realized_through_R_mlx.py"],
            "projected_peak_gib": 60.0}]
    samples = {800: _ps(800, "python train_levelset_witness_realized_through_R_mlx.py", rss_kb=1_000_000)}
    jobs = gov.list_tracked_jobs(samples=samples, registry_rows=reg, self_pid=1)
    j = next(x for x in jobs if x.pid == 800)
    assert j.projected_peak_gib == 60.0
    assert gov.sum_active_growth_headroom_gib(jobs) > 55.0  # full remaining growth reserved


# ── 2026-07-14 material-unregistered MEASURED-growth fix ───────────────────────────────────────
def _history(pid, rss_rows, *, process_key="proc-start-a"):
    return [
        gov.RSSHistorySample(pid=pid, rss_gib=rss, ts=ts, process_key=process_key)
        for ts, rss in rss_rows
    ]


def test_observed_growth_stable_material_keeps_nonzero_runtime_poll_reserve():
    observed = gov.estimate_observed_remaining_growth_gib(
        _history(71, [(100.0, 4.4), (160.0, 4.4)]),
        now_ts=160.0,
    )
    assert observed == pytest.approx(gov.MATERIAL_PLATEAU_GROWTH_RESERVE_GIB)
    assert 0.0 < observed < gov.UNKNOWN_GROWTH_HEADROOM_GIB
    assert gov.resolve_projected_peak_gib(
        None,
        4.4,
        unregistered_ps_only=True,
        observed_remaining_growth_gib=observed,
    ) == pytest.approx(4.4 + gov.MATERIAL_PLATEAU_GROWTH_RESERVE_GIB)


def test_observed_growth_real_rising_trend_saturates_old_charge():
    observed = gov.estimate_observed_remaining_growth_gib(
        _history(72, [(100.0, 5.1), (160.0, 8.1)]),
        now_ts=160.0,
    )
    assert observed == gov.UNKNOWN_GROWTH_HEADROOM_GIB
    assert gov.resolve_projected_peak_gib(
        None,
        8.1,
        unregistered_ps_only=True,
        observed_remaining_growth_gib=observed,
    ) == pytest.approx(33.1)


def test_observed_growth_requires_two_well_separated_current_samples():
    assert gov.estimate_observed_remaining_growth_gib(
        _history(73, [(100.0, 4.4)]), now_ts=100.0
    ) is None
    assert gov.estimate_observed_remaining_growth_gib(
        _history(73, [(100.0, 4.4), (101.0, 4.4)]), now_ts=101.0
    ) is None


@pytest.mark.parametrize("bad_observation", [None, float("nan"), float("inf"), "bad"])
def test_observed_growth_no_or_invalid_history_falls_back_to_plus_25(bad_observation):
    assert gov.resolve_projected_peak_gib(
        None,
        4.4,
        unregistered_ps_only=True,
        observed_remaining_growth_gib=bad_observation,
    ) == pytest.approx(29.4)


def test_observed_growth_cannot_change_recorded_descendant_infra_or_below_floor_paths():
    # Every result is the exact pre-fix branch result even with an adversarial zero observation.
    assert gov.resolve_projected_peak_gib(
        9.0, 4.4, unregistered_ps_only=True, observed_remaining_growth_gib=0.0
    ) == 9.0
    assert gov.resolve_projected_peak_gib(
        None, 4.4, governed_descendant=True, unregistered_ps_only=True,
        observed_remaining_growth_gib=0.0,
    ) == 4.4
    assert gov.resolve_projected_peak_gib(
        None, 4.4, cmd="python tools/memory_blackbox.py", unregistered_ps_only=True,
        observed_remaining_growth_gib=0.0,
    ) == 4.4
    assert gov.resolve_projected_peak_gib(
        None, 1.99, unregistered_ps_only=True, observed_remaining_growth_gib=25.0
    ) == 1.99


@pytest.mark.skipif(gov._mg is None, reason="memory_guard unavailable (fail-safe: no tracked jobs)")
def test_fixed_snapshot_plus50_false_refuse_is_gone_but_real_growth_still_refuses(tmp_path):
    """MEASURED apparatus replay of the incident: used~50 + two material stable ps-only jobs whose
    current 4.4/5.1 GiB is already in used + new 6. Old +25 each => 106 > 72 REFUSE. Fresh measured
    plateaus keep only the runtime-poll reserves => ~59.3 < 72 ADMIT. A real rising fixture still
    reaches the old +25 and refuses.

    ddm_mb1: every ``list_tracked_jobs`` call here passes ``layer2_backstop_armed=True`` EXPLICITLY.
    The measured-growth relaxation is licensed by Layer 2 (the SIGSTOP throttle) being able to pause
    the job it relaxes, and that actuator is now default-OFF, so the relaxation is withdrawn while
    it is disarmed. Stating the precondition here isolates the variable this test is actually about
    (history freshness) instead of letting the assertion swing on whatever arming state the host
    machine happens to carry.
    """
    history_path = tmp_path / "rss_history.json"
    ceiling = gov.compute_adaptive_ceiling(
        total_gib=80.0,
        used_gib=50.0,
        tracked_current_gib=9.5,
        safety_margin_gib=8.0,
    )
    assert ceiling.adaptive_ceiling_gib == 72.0

    stable = {
        710: _ps(
            710,
            "python tools/levelset_byte_close_and_eval.py --click-polish",
            rss_kb=round(4.4 * 1024**2),
            start_identity="Tue Jul 14 00:00:01 2026",
        ),
        720: _ps(
            720,
            "python tools/verdict_mem_microprobe.py --n 600",
            rss_kb=round(5.1 * 1024**2),
            start_identity="Tue Jul 14 00:00:02 2026",
        ),
    }
    first = gov.list_tracked_jobs(
        samples=stable, registry_rows=[], self_pid=1,
        rss_history_path=history_path, rss_history_now_ts=100.0,
        layer2_backstop_armed=True,
    )
    old_headroom = gov.sum_active_growth_headroom_gib(first)
    old_decision = gov.admission_decision(
        projected_new_gib=6.0,
        system_used_gib=50.0,
        active_growth_headroom_gib=old_headroom,
        ceiling=ceiling,
    )
    assert old_headroom == 50.0
    assert old_decision.projected_system_used_gib == 106.0
    assert not old_decision.admit

    plateau = gov.list_tracked_jobs(
        samples=stable, registry_rows=[], self_pid=1,
        rss_history_path=history_path, rss_history_now_ts=160.0,
        layer2_backstop_armed=True,
    )
    measured_headroom = gov.sum_active_growth_headroom_gib(plateau)
    measured_decision = gov.admission_decision(
        projected_new_gib=6.0,
        system_used_gib=50.0,
        active_growth_headroom_gib=measured_headroom,
        ceiling=ceiling,
    )
    assert measured_headroom == pytest.approx(2.0 * gov.MATERIAL_PLATEAU_GROWTH_RESERVE_GIB)
    assert measured_decision.projected_system_used_gib == pytest.approx(
        56.0 + 2.0 * gov.MATERIAL_PLATEAU_GROWTH_RESERVE_GIB
    )
    assert measured_decision.admit

    growing = dict(stable)
    growing[720] = _ps(
        720,
        "python tools/verdict_mem_microprobe.py --n 600",
        rss_kb=round(8.1 * 1024**2),
        start_identity="Tue Jul 14 00:00:02 2026",
    )
    rising = gov.list_tracked_jobs(
        samples=growing, registry_rows=[], self_pid=1,
        rss_history_path=history_path, rss_history_now_ts=220.0,
        layer2_backstop_armed=True,
    )
    rising_headroom = gov.sum_active_growth_headroom_gib(rising)
    assert rising_headroom >= gov.UNKNOWN_GROWTH_HEADROOM_GIB
    rising_decision = gov.admission_decision(
        projected_new_gib=6.0,
        system_used_gib=50.0,
        active_growth_headroom_gib=rising_headroom,
        ceiling=ceiling,
    )
    assert rising_decision.projected_system_used_gib > 72.0
    assert not rising_decision.admit


@pytest.mark.skipif(gov._mg is None, reason="memory_guard unavailable (fail-safe: no tracked jobs)")
def test_measured_relaxation_requires_start_identity_and_throttle_eligibility(tmp_path):
    path = tmp_path / "rss_history.json"
    no_start = {
        730: _ps(730, "python tools/verdict_mem_microprobe.py", rss_kb=5 * 1024**2),
    }
    for now in (100.0, 160.0):
        jobs = gov.list_tracked_jobs(
            samples=no_start, registry_rows=[], self_pid=1,
            rss_history_path=path, rss_history_now_ts=now,
        )
    assert gov.sum_active_growth_headroom_gib(jobs) == 25.0
    assert not path.exists(), "missing kernel start identity must not create relaxable evidence"

    nonleader = {
        740: _ps(
            740,
            "python tools/verdict_mem_microprobe.py",
            rss_kb=5 * 1024**2,
            pgid=700,
            start_identity="Tue Jul 14 00:00:04 2026",
        ),
    }
    for now in (200.0, 260.0):
        jobs = gov.list_tracked_jobs(
            samples=nonleader, registry_rows=[], self_pid=1,
            rss_history_path=path, rss_history_now_ts=now,
        )
    assert jobs[0].throttle_eligible is False
    assert gov.sum_active_growth_headroom_gib(jobs) == 25.0


def test_history_store_is_pid_reuse_safe_ttl_swept_bounded_and_atomic(tmp_path):
    path = tmp_path / "rss_history.json"
    assert gov.update_rss_history_and_estimate_growth(
        {800: (4.4, "start-a")}, history_path=path, now_ts=100.0
    )[800] is None
    assert gov.update_rss_history_and_estimate_growth(
        {800: (4.4, "start-a")}, history_path=path, now_ts=160.0
    )[800] == pytest.approx(gov.MATERIAL_PLATEAU_GROWTH_RESERVE_GIB)
    # Same PID, different kernel start identity: prior evidence is dropped, never mixed to a plateau.
    assert gov.update_rss_history_and_estimate_growth(
        {800: (4.4, "start-b")}, history_path=path, now_ts=220.0
    )[800] is None

    seeded = {
        "schema": gov.RSS_HISTORY_SCHEMA,
        "samples": [
            {"pid": pid, "rss_gib": 3.0, "ts": 300.0 + i, "process_key": f"start-{pid}"}
            for pid in range(1, gov.RSS_HISTORY_MAX_PIDS + 20)
            for i in range(gov.RSS_HISTORY_MAX_SAMPLES_PER_PID + 5)
        ] + [
            {"pid": 9999, "rss_gib": 3.0, "ts": -10_000.0, "process_key": "stale"},
        ],
    }
    path.write_text(json.dumps(seeded), encoding="utf-8")
    gov.update_rss_history_and_estimate_growth(
        {900: (4.4, "start-900")}, history_path=path, now_ts=400.0
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["samples"]
    assert len({row["pid"] for row in rows}) <= gov.RSS_HISTORY_MAX_PIDS
    counts: dict[int, int] = {}
    for row in rows:
        counts[row["pid"]] = counts.get(row["pid"], 0) + 1
    assert max(counts.values()) <= gov.RSS_HISTORY_MAX_SAMPLES_PER_PID
    assert all(row["pid"] != 9999 for row in rows)
    assert path.with_suffix(path.suffix + ".lock").exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_history_corrupt_or_unwritable_state_never_relaxes(tmp_path):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    first = gov.update_rss_history_and_estimate_growth(
        {901: (4.4, "start-901")}, history_path=corrupt, now_ts=100.0
    )
    assert first[901] is None
    assert json.loads(corrupt.read_text(encoding="utf-8"))["schema"] == gov.RSS_HISTORY_SCHEMA

    unwritable_shape = tmp_path / "is-a-directory"
    unwritable_shape.mkdir()
    failed = gov.update_rss_history_and_estimate_growth(
        {902: (4.4, "start-902")}, history_path=unwritable_shape, now_ts=100.0
    )
    assert failed[902] is None
    assert gov.resolve_projected_peak_gib(
        None, 4.4, unregistered_ps_only=True,
        observed_remaining_growth_gib=failed[902],
    ) == pytest.approx(29.4)


# ── PENDING admission reservations (review-fix CRITICAL C, read side) ────────────────────────────
def _pending_row(label="pending_a", proj=50.0, age_s=1.0, now=1_000_000.0, pid=None):
    return {"label": label, "pid": pid, "status": gov.PENDING_RESERVATION_STATUS,
            "projected_peak_gib": proj, "reserved_ts": now - age_s, "cmd": []}


def test_pending_reservation_rows_fresh_vs_stale():
    now = 1_000_000.0
    rows = [
        _pending_row("fresh", age_s=5.0, now=now),
        _pending_row("stale", age_s=999.0, now=now),
        {"label": "no_ts", "pid": None, "status": gov.PENDING_RESERVATION_STATUS},  # malformed
        _pending_row("promoted", age_s=5.0, now=now, pid=1234),  # pid set -> not a reservation
        {"label": "running", "pid": 42, "status": "running"},
    ]
    fresh = gov.pending_reservation_rows(rows, now_ts=now)
    assert [r["label"] for r in fresh] == ["fresh"]


@pytest.mark.skipif(gov._mg is None, reason="memory_guard unavailable (fail-safe: no tracked jobs)")
def test_pending_reservation_counted_as_growth_headroom():
    """A fresh pending reservation surfaces as a zero-RSS tracked job whose growth headroom equals
    its reserved projection — so a SECOND launcher's admission arithmetic sees the first's
    just-admitted job (the TOCTOU close, read side)."""
    rows = [_pending_row("job_a", proj=50.0, age_s=1.0, now=gov.time.time())]
    jobs = gov.list_tracked_jobs(samples={}, registry_rows=rows)
    pend = [j for j in jobs if j.label == "job_a"]
    assert len(pend) == 1
    assert pend[0].current_rss_gib == 0.0
    assert pend[0].projected_peak_gib == 50.0
    assert pend[0].growth_headroom_gib == 50.0
    assert pend[0].throttle_eligible is False       # pid 0 — nothing to pause
    assert gov.sum_active_growth_headroom_gib(jobs) >= 50.0
    # and the admission math flips on it: fits without the reservation, refuses with it.
    ceiling = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=20.0, tracked_current_gib=0.0)
    without = gov.admission_decision(projected_new_gib=60.0, system_used_gib=20.0,
                                     active_growth_headroom_gib=0.0, ceiling=ceiling)
    with_res = gov.admission_decision(projected_new_gib=60.0, system_used_gib=20.0,
                                      active_growth_headroom_gib=50.0, ceiling=ceiling)
    assert without.admit and not with_res.admit


@pytest.mark.skipif(gov._mg is None, reason="memory_guard unavailable (fail-safe: no tracked jobs)")
def test_stale_pending_reservation_not_counted():
    rows = [_pending_row("crashed_launcher", proj=50.0, age_s=9999.0, now=gov.time.time())]
    jobs = gov.list_tracked_jobs(samples={}, registry_rows=rows)
    assert [j for j in jobs if j.label == "crashed_launcher"] == []


# ── reclaimable-aware committed accounting (2026-07-16 admission false-refuse fix) ──────────────
# Memory anchor: admission_gate_naive_counts_reclaimable_as_committed_20260716. The legacy basis
# ``used = total - (free+inactive)`` counted ~9 GiB of reclaimable file cache as committed on an
# IDLE box and refused an empirically-green 82 GiB bench (ran GREEN twice, no jetsam). The fix
# gates on TRUE committed = wired + compressor + non-purgeable anonymous.

# Captured 2026-07-16 IDLE state (the false-refuse night; 128 GiB M5 Max, Terminal+Tailscale only):
# free 76.8 / active 21.1 / inactive 13.8 / speculative 7.4 / wired 5.7 / compressed 1.9 GiB,
# file-backed 21.5 / purgeable 0.5; anonymous = active+inactive+spec - file (EXACT kernel identity).
_IDLE_20260716 = dict(
    page_size=16384, total_bytes=137438953472,
    vm_free_pages=5033165, vm_active_pages=1382810, vm_inactive_pages=904397,
    vm_speculative_pages=484966, vm_wired_pages=373555, vm_compressor_pages=124518,
    vm_throttled_pages=0,
    vm_purgeable_pages=32768, vm_file_backed_pages=1409024,
    vm_anonymous_pages=1382810 + 904397 + 484966 - 1409024,   # = 1363149 (identity-exact)
    sysctl_free_pages=5033165, mempressure_total_bytes=137438953472,
    psutil_available_bytes=(5033165 + 904397) * 16384,        # psutil == free+inactive on macOS
)

# Live-captured ANON-HEAVY state (2026-07-16, 63.6 GiB trainer running): the committed basis must
# be MORE conservative than legacy here (dirty anon in the INACTIVE queue is NOT reclaimable).
_ANON_HEAVY_20260716 = dict(
    page_size=16384, total_bytes=137438953472,
    vm_free_pages=132865, vm_active_pages=3703200, vm_inactive_pages=3704096,
    vm_speculative_pages=1380, vm_wired_pages=646616, vm_compressor_pages=115875,
    vm_throttled_pages=0,
    vm_purgeable_pages=16888, vm_file_backed_pages=1865616, vm_anonymous_pages=5543060,
    sysctl_free_pages=132865, mempressure_total_bytes=137438953472,
    psutil_available_bytes=(132865 + 3704096) * 16384,
)


def _snap_from_acct(a, *, pressure=1, swap=0.0):
    """Mirror read_system_memory_snapshot's acct->snapshot copy for hermetic fixed-counter tests."""
    return gov.SystemMemorySnapshot(
        total_gib=a.total_gib, available_gib=a.available_gib, used_gib=a.used_gib,
        free_gib=a.free_gib, wired_gib=a.wired_gib, compressor_gib=a.compressor_gib,
        swap_used_gib=swap, pressure_level=pressure, load1=0.0, load5=0.0, load15=0.0,
        available_primary_gib=a.available_primary_gib, closure_gib=a.closure_gib,
        closure_ok=a.closure_ok, cross_validated=a.cross_validated,
        discrepancy_gib=a.discrepancy_gib, fail_safe=a.fail_safe,
        validation_notes=a.validation_notes,
        available_reclaimable_gib=a.available_reclaimable_gib,
        used_committed_gib=a.used_committed_gib, reclaimable_ok=a.reclaimable_ok,
        anonymous_gib=a.anonymous_gib, file_backed_gib=a.file_backed_gib,
        purgeable_gib=a.purgeable_gib)


def test_reclaimable_accounting_idle_snapshot_exact():
    """(proof a, accounting layer) The idle 2026-07-16 counters reconcile to TRUE committed
    ~27.9 GiB / reclaimable available ~98.8 GiB — vs the legacy basis's used 37.4 that counted
    ~9 GiB of reclaimable cache as pinned."""
    a = gov.reconcile_memory_accounting(**_IDLE_20260716)
    assert a.closure_ok and a.cross_validated and not a.fail_safe
    assert a.reclaimable_ok
    assert abs(a.available_primary_gib - 90.6) < 0.05     # legacy free+inactive
    assert abs(a.used_gib - 37.4) < 0.05                  # legacy used (the naive over-refuse basis)
    assert abs(a.used_committed_gib - 29.2) < 0.10        # TRUE committed + unaccounted gap
    assert abs(a.available_reclaimable_gib - 98.8) < 0.10
    # the reclaimable estimate never exceeds physical reality (non-wired, non-compressor RAM):
    assert a.available_reclaimable_gib <= a.total_gib - a.wired_gib - a.compressor_gib


def test_reclaimable_idle_admits_the_82gib_bench_that_ran_green(monkeypatch):
    """(proof a) The EXACT false-refuse scenario: projected_new 71.54 GiB (the c2 delta-bench that
    ran GREEN twice + live tonight with no jetsam) on the idle box ADMITS under the committed
    basis — and the same snapshot REFUSES under the legacy basis (pinning the bug we fixed).
    Operator-ceiling policy disabled: the legacy-contrast leg is a HISTORICAL replay of the
    2026-07-16 false-refuse night, which predates the 2026-07-21 operator ceiling."""
    monkeypatch.setenv(gov.OPERATOR_CEILING_ENV, "0")
    a = gov.reconcile_memory_accounting(**_IDLE_20260716)
    ctx = gov.live_admission_decision(projected_new_gib=71.54, snapshot=_snap_from_acct(a), jobs=[])
    assert ctx.decision.admit, ctx.decision.reason
    assert ctx.decision.headroom_after_gib > 0
    # regression contrast: the legacy used-basis arithmetic reproduces tonight's refusal
    # (projected ~108.9 vs ceiling ~102.9 — the log's "EXCEEDS adaptive ceiling by ~6 GiB").
    legacy_ceiling = gov.compute_adaptive_ceiling(
        total_gib=a.total_gib, used_gib=a.used_gib, tracked_current_gib=0.0, sole_workload=True)
    legacy = gov.admission_decision(projected_new_gib=71.54, system_used_gib=a.used_gib,
                                    active_growth_headroom_gib=0.0, ceiling=legacy_ceiling)
    assert not legacy.admit
    assert 4.0 < -legacy.headroom_after_gib < 8.0   # ~6 GiB over, as logged on the false-refuse night


def test_reclaimable_genuinely_full_still_refuses_82gib():
    """(proof b — the NON-NEGOTIABLE safety direction) TRUE committed ~99 GiB: an 82 GiB job MUST
    refuse under the committed basis (physically impossible: 99+82 > 128)."""
    full = dict(
        page_size=16384, total_bytes=137438953472,
        vm_free_pages=262144, vm_active_pages=3932160, vm_inactive_pages=2621440,
        vm_speculative_pages=589824, vm_wired_pages=655360, vm_compressor_pages=327680,
        vm_throttled_pages=0,
        vm_purgeable_pages=65536, vm_file_backed_pages=1572864,
        vm_anonymous_pages=3932160 + 2621440 + 589824 - 1572864,   # 85 GiB anon
        sysctl_free_pages=262144, mempressure_total_bytes=137438953472,
        psutil_available_bytes=(262144 + 2621440) * 16384,
    )
    a = gov.reconcile_memory_accounting(**full)
    assert a.reclaimable_ok and not a.fail_safe
    assert abs(a.used_committed_gib - 99.0) < 0.05
    assert abs(a.available_reclaimable_gib - 29.0) < 0.05
    ctx = gov.live_admission_decision(projected_new_gib=82.0, snapshot=_snap_from_acct(a), jobs=[])
    assert not ctx.decision.admit
    assert ctx.decision.projected_system_used_gib > a.total_gib   # physically impossible request
    # even the most permissive legal floor cannot admit it:
    floor_min = gov.ABS_MIN_SAFETY_FLOOR_GIB
    assert a.used_committed_gib + 82.0 > a.total_gib - floor_min


def test_reclaimable_fail_safe_snapshot_still_refuses():
    """(proof c) An accounting-validation failure (wrong page size -> closure blowout) forces
    REFUSE through the live admission path even when the arithmetic would fit."""
    # (c1) catastrophic parse failure (wrong page size): available slashed -> arithmetic refuse.
    bad = dict(_IDLE_20260716)
    bad["page_size"] = 4096
    a = gov.reconcile_memory_accounting(**bad)
    assert a.fail_safe
    ctx = gov.live_admission_decision(projected_new_gib=1.0, snapshot=_snap_from_acct(a), jobs=[])
    assert not ctx.decision.admit
    # (c2) modest cross-check failure where the arithmetic WOULD fit: the fail_safe flag alone
    # forces the refusal (the explicit FAIL-SAFE branch).
    bad2 = dict(_IDLE_20260716)
    bad2["sysctl_free_pages"] = bad2["vm_free_pages"] - 500_000
    a2 = gov.reconcile_memory_accounting(**bad2)
    assert a2.fail_safe
    ctx2 = gov.live_admission_decision(projected_new_gib=1.0, snapshot=_snap_from_acct(a2), jobs=[])
    assert not ctx2.decision.admit
    assert "FAIL-SAFE" in ctx2.decision.reason


def test_reclaimable_anon_heavy_is_MORE_conservative_than_legacy():
    """No blanket over-admit: on the live anon-heavy capture (63.6 GiB trainer resident) the
    committed basis reports MORE used than legacy — dirty anonymous pages sitting in the INACTIVE
    queue were being credited as free by free+inactive; they are not (they need swap to evict)."""
    a = gov.reconcile_memory_accounting(**_ANON_HEAVY_20260716)
    assert a.reclaimable_ok and not a.fail_safe
    assert a.used_committed_gib > a.used_gib + 20.0     # measured: ~97.2 vs ~69.5
    assert a.available_reclaimable_gib < a.available_gib
    assert a.available_reclaimable_gib <= a.total_gib - a.wired_gib - a.compressor_gib


def test_reclaimable_fallback_when_counters_missing_is_legacy_bit_identical():
    """Old kernels / hand-built snapshots without Anonymous/File-backed counters keep the legacy
    basis EXACTLY (reclaimable_ok False; committed fields mirror the legacy values)."""
    a = gov.reconcile_memory_accounting(**_SNAP)
    assert not a.reclaimable_ok
    assert a.used_committed_gib == a.used_gib
    assert a.available_reclaimable_gib == a.available_gib
    assert a.closure_ok and not a.fail_safe   # and it is NOT a validation failure


def test_reclaimable_queue_identity_violation_falls_back_loudly():
    """A broken anon/file parse (identity |anon+file - queues| > tol) DISABLES the generous basis
    (falls back to legacy, loud note) without poisoning the validated legacy accounting."""
    bad = dict(_IDLE_20260716)
    bad["vm_anonymous_pages"] = bad["vm_anonymous_pages"] + 655360   # +10 GiB parse corruption
    a = gov.reconcile_memory_accounting(**bad)
    assert not a.reclaimable_ok
    assert a.used_committed_gib == a.used_gib
    assert any("reclaimable accounting DISABLED" in n for n in a.validation_notes)
    assert not a.fail_safe   # legacy basis remains validated -> gate stays usable, conservatively


def test_reclaimable_fail_safe_reduces_generous_available_too():
    """When a cross-check fails, the discrepancy reduction applies to the reclaimable figure as
    well — fail-safe never leaves the generous number un-penalized."""
    bad = dict(_IDLE_20260716)
    bad["sysctl_free_pages"] = bad["vm_free_pages"] - 500_000   # ~7.6 GiB free-page disagreement
    a = gov.reconcile_memory_accounting(**bad)
    assert a.fail_safe
    good = gov.reconcile_memory_accounting(**_IDLE_20260716)
    assert a.available_reclaimable_gib < good.available_reclaimable_gib - 5.0


def test_live_admission_uses_committed_basis_only_when_validated(monkeypatch):
    # Operator-ceiling policy (2026-07-21, RAISE-only) disabled: this test asserts exact
    # DERIVED-floor ceiling arithmetic on a 128 GiB scenario.
    monkeypatch.setenv("TAC_GOV_OPERATOR_CEILING_GIB", "0")
    """The switch point: two snapshots identical except reclaimable_ok flip the 71.54 GiB verdict
    (validated committed basis ADMITS; unvalidated falls back to the legacy REFUSE)."""
    a = gov.reconcile_memory_accounting(**_IDLE_20260716)
    snap_ok = _snap_from_acct(a)
    snap_fallback = gov.SystemMemorySnapshot(**{**{
        f: getattr(snap_ok, f) for f in snap_ok.__dataclass_fields__}, "reclaimable_ok": False})
    admit = gov.live_admission_decision(projected_new_gib=71.54, snapshot=snap_ok, jobs=[])
    refuse = gov.live_admission_decision(projected_new_gib=71.54, snapshot=snap_fallback, jobs=[])
    assert admit.decision.admit
    assert not refuse.decision.admit
    assert admit.decision.system_used_gib < refuse.decision.system_used_gib


def test_default_snapshot_fields_keep_legacy_behavior():
    """Hand-built snapshots (older tests / callers) default reclaimable_ok=False -> the admission
    path is bit-identical to the legacy basis (never consumes the 0.0 committed defaults)."""
    snap = gov.SystemMemorySnapshot(
        total_gib=128.0, available_gib=90.0, used_gib=38.0, free_gib=70.0, wired_gib=6.0,
        compressor_gib=2.0, swap_used_gib=0.0, pressure_level=1, load1=0.0, load5=0.0, load15=0.0)
    assert snap.reclaimable_ok is False
    ctx = gov.live_admission_decision(projected_new_gib=10.0, snapshot=snap, jobs=[])
    assert ctx.decision.system_used_gib == 38.0   # legacy used, NOT the 0.0 default


def test_governed_descendant_excluded_from_tracked_current_sum():
    """(phantom-baseline fix) A governed descendant's subtree RSS is already inside its registered
    parent's group RSS: counting both clamped baseline to 0 and UNDER-derived the floor (measured
    live 2026-07-16: wrapper 63.56 + trainer 63.56 = 127.17 GiB tracked on a 71 GiB-used box)."""
    parent = _job("saferun_wrapper", pid=83774, peak=71.54, rss=63.56)
    child = gov.TrackedJob(
        label="pid83775", pid=83775, pgid=83775, cmd="python train_levelset_witness.py",
        priority=50, projected_peak_gib=63.56, current_rss_gib=63.56, paused=False,
        throttle_eligible=False, own_group_leader=True, governed_descendant=True)
    assert gov.sum_tracked_current_gib([parent, child]) == pytest.approx(63.56)
    assert gov.sum_tracked_current_gib([parent]) == pytest.approx(63.56)
    assert child.to_json()["governed_descendant"] is True
    assert parent.governed_descendant is False   # default: registered rows are never descendants


# ── OPERATOR CEILING POLICY (operator directive 2026-07-21, corrected same day: 116 GiB on the 128 GiB sole-workload
# box) — RAISE-only, big-box-guarded, env-overridable ────────────────────────────────────────────
class TestOperatorCeilingPolicy:
    def test_128gib_box_ceiling_raised_to_116(self, monkeypatch):
        monkeypatch.delenv(gov.OPERATOR_CEILING_ENV, raising=False)
        ac = gov.compute_adaptive_ceiling(
            total_gib=128.0, used_gib=41.5, tracked_current_gib=0.0)
        assert ac.adaptive_ceiling_gib >= 116.0
        assert ac.safety_margin_gib == pytest.approx(128.0 - ac.adaptive_ceiling_gib)
        # budget stays physically consistent: ceiling - baseline
        assert ac.training_budget_gib == pytest.approx(ac.adaptive_ceiling_gib - ac.baseline_gib)

    def test_raise_only_never_lowers_a_higher_ceiling(self, monkeypatch):
        monkeypatch.delenv(gov.OPERATOR_CEILING_ENV, raising=False)
        # tiny explicit margin -> derived ceiling 126 > 106; the policy must NOT pull it down
        ac = gov.compute_adaptive_ceiling(
            total_gib=128.0, used_gib=10.0, tracked_current_gib=0.0, safety_margin_gib=2.0)
        assert ac.adaptive_ceiling_gib == pytest.approx(126.0)

    def test_small_box_unaffected(self, monkeypatch):
        monkeypatch.delenv(gov.OPERATOR_CEILING_ENV, raising=False)
        ac = gov.compute_adaptive_ceiling(
            total_gib=8.0, used_gib=4.0, tracked_current_gib=0.0)
        # an 8 GiB M1 must never inherit a 106 GiB absolute ceiling
        assert ac.adaptive_ceiling_gib < 8.0

    def test_env_zero_disables_policy(self, monkeypatch):
        monkeypatch.setenv(gov.OPERATOR_CEILING_ENV, "0")
        assert gov.operator_ceiling_gib() == 0.0
        ac_off = gov.compute_adaptive_ceiling(
            total_gib=128.0, used_gib=41.5, tracked_current_gib=0.0, safety_margin_gib=27.1)
        assert ac_off.adaptive_ceiling_gib == pytest.approx(128.0 - 27.1)

    def test_env_override_value_and_garbage(self, monkeypatch):
        monkeypatch.setenv(gov.OPERATOR_CEILING_ENV, "110")
        assert gov.operator_ceiling_gib() == pytest.approx(110.0)
        monkeypatch.setenv(gov.OPERATOR_CEILING_ENV, "not-a-number")
        assert gov.operator_ceiling_gib() == pytest.approx(gov.OPERATOR_CEILING_GIB_DEFAULT)

    def test_ceiling_never_eats_abs_min_headroom(self, monkeypatch):
        monkeypatch.setenv(gov.OPERATOR_CEILING_ENV, "127.5")
        ac = gov.compute_adaptive_ceiling(
            total_gib=128.0, used_gib=41.5, tracked_current_gib=0.0)
        assert ac.adaptive_ceiling_gib <= 128.0 - gov.ABS_MIN_SAFETY_FLOOR_GIB

    def test_preflight_intrinsic_leg_honors_operator_ceiling(self, monkeypatch):
        monkeypatch.delenv(gov.OPERATOR_CEILING_ENV, raising=False)
        import witness_memory_preflight as wmp
        proj = wmp.project_peak_rss_gib(
            num_pairs=600, verdict_batch=16, micro_batch_pairs=2, total_ram_gib=128.0)
        assert proj.safe_ceiling_gib == pytest.approx(116.0, abs=0.1)
        proj8 = wmp.project_peak_rss_gib(
            num_pairs=24, self_orient=False, verdict_batch=8, total_ram_gib=8.0)
        assert proj8.safe_ceiling_gib == pytest.approx(0.70 * 8.0, abs=0.1)
