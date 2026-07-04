# SPDX-License-Identifier: MIT
"""Tests for BUILD #298: the tier-scaled DYNAMICAL safety floor + GB-F1 admission-units fix +
#246 full-tree pause/resume + the blackbox continuous band wiring.

THE FORMULA (tools/system_memory_governor.py::derive_safety_floor):

    floor = clamp( max( ABS_MIN(2.0),
                        measured_cp_rss + cp_headroom(T),   # dynamical leg; cp_headroom = max(1, 0.05*T)
                        0.08 * T ),                         # static physics leg
                   ABS_MIN, 0.5 * T )                       # cap: a floor may never eat the box

EXACT ASSERTION NUMBERS (derived from the formula; the matrix tests assert these literally):

    T (GiB):          8      16     32     64     128     192
    unavailable:      2.00   2.00   2.56   5.12   10.24   15.36     (static leg; ABS_MIN below 25 GiB)
    idle    (cp=2):   3.00   3.00   3.60   5.20   10.24   15.36
    loaded  (cp=6):   4.00*  7.00   7.60   9.20   12.40   15.60     (* cap-clamped from raw 7.00)

    usable envelope (T - floor), loaded row:  4.0   9.0   24.4   54.8   115.6   176.4
    -> @8 GiB loaded: 4.0 GiB usable (> 2 GiB required; == the tertiary tier's MEASURED safe
       envelope ~4.0-4.4 GiB; the pre-fix constant floor 8.0 gave ceiling 0.0 / budget -5.99).
    -> @128 GiB: floor >= 10.24 in EVERY scenario (operator-policy >= 10 GiB, backward-compatible).

    Tier-scaled pressure thresholds: (warn, critical) = (ABS_MIN+2*ch, ABS_MIN+ch):
       @128 -> (14.8, 8.4) [legacy (15, 8): critical strictly MORE protective; warn within 1.4%]
       @8   -> (4.0, 3.0)  [legacy 15/8 were above anything an 8 GiB box can free]

    Band thresholds (frac x RAM envelope machinery, auto-scaling):
       128 @ 0.85 -> envelope 108.80, yellow 92.48, red 97.92 (true GiB)
       8   @ 0.55 -> envelope   4.40, yellow  3.74, red  3.96
"""
from __future__ import annotations

import inspect
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import memory_blackbox as mbb  # noqa: E402
import memory_guard as mg  # noqa: E402
import system_memory_governor as gov  # noqa: E402

TIERS = (8.0, 16.0, 32.0, 64.0, 128.0, 192.0)
# scenario -> (measured_cp_rss_gib, expected floors per tier)
MATRIX = {
    None: (2.00, 2.00, 2.56, 5.12, 10.24, 15.36),
    2.0: (3.00, 3.00, 3.60, 5.20, 10.24, 15.36),
    6.0: (4.00, 7.00, 7.60, 9.20, 12.40, 15.60),
}


@pytest.fixture(autouse=True)
def _no_floor_env(monkeypatch):
    monkeypatch.delenv(gov.SAFETY_FLOOR_ENV, raising=False)


def _floor(total, cp, **kw):
    return gov.derive_safety_floor(total_gib=total, measured_cp_rss_gib=cp, **kw)


# ── 1. the tier matrix (exact values, monotonicity, bounds) ─────────────────────────────────────
def test_floor_matrix_exact_values():
    for cp, expected in MATRIX.items():
        for total, want in zip(TIERS, expected, strict=True):
            d = _floor(total, cp)
            assert abs(d.floor_gib - want) < 1e-9, f"T={total} cp={cp}: {d.floor_gib} != {want}"


def test_floor_monotone_in_ram_per_scenario():
    for cp in MATRIX:
        floors = [_floor(t, cp).floor_gib for t in TIERS]
        assert floors == sorted(floors), f"cp={cp}: not monotone in RAM: {floors}"


def test_floor_never_exceeds_cap_and_at_least_abs_min():
    for cp in (None, 0.0, 2.0, 6.0, 30.0, 500.0):
        for total in TIERS:
            d = _floor(total, cp)
            assert d.floor_gib <= 0.5 * total + 1e-12
            assert d.floor_gib >= gov.ABS_MIN_SAFETY_FLOOR_GIB - 1e-12


def test_128_floor_at_least_10_every_scenario():
    """Operator memory policy backward-compat: >= 10 GiB on the 128 box, ALWAYS."""
    for cp in (None, 0.0, 2.0, 6.0, 16.0, 60.0):
        assert _floor(128.0, cp).floor_gib >= 10.0


def test_8gb_loaded_leaves_usable_envelope_over_2gib():
    """The finding being fixed: the constant 8.0 floor == the whole 8 GiB box. Derived: the cap
    binds at 4.0 -> usable envelope T - floor = 4.0 GiB (> 2 GiB; == the tertiary tier's measured
    ~4.0-4.4 GiB safe envelope)."""
    d = _floor(8.0, 6.0)
    assert d.floor_gib == 4.0 and d.clamped and d.raw_floor_gib == 7.0
    assert 8.0 - d.floor_gib == 4.0 > 2.0


def test_8gb_default_no_longer_refuses_everything():
    """Tertiary idle replay (used 6.06, no tracked jobs): pre-fix adaptive_ceiling 0.0 and
    training_budget -5.99 refused every launch. Post-fix the ceiling is 4.0 GiB and a small
    edge job admits once the box is not idle-saturated."""
    c = gov.compute_adaptive_ceiling(total_gib=8.0, used_gib=6.06, tracked_current_gib=0.0)
    assert c.adaptive_ceiling_gib == 4.0            # 8 - cap-clamped floor 4.0 (was 0.0)
    assert c.safety_margin_gib == 4.0
    # under load the OS yields (tertiary smoke: baseline ~3 under pressure) -> a 0.9 GiB job fits:
    c2 = gov.compute_adaptive_ceiling(total_gib=8.0, used_gib=3.0, tracked_current_gib=0.0)
    d = gov.admission_decision(projected_new_gib=0.9, system_used_gib=3.0,
                               active_growth_headroom_gib=0.0, ceiling=c2)
    assert d.admit


def test_winning_leg_labels():
    assert _floor(8.0, None).winning_leg == "abs_min"
    assert _floor(128.0, None).winning_leg == "static_frac"
    assert _floor(128.0, 8.0).winning_leg == "measured_cp"
    assert _floor(128.0, cp=None, override_gib=12.0).winning_leg == "override"
    assert _floor(64.0, None, mode=gov.FLOOR_MODE_FIXED).winning_leg == "fixed_legacy"


def test_decomposition_carries_all_legs():
    d = _floor(128.0, 8.0)
    j = d.to_json()
    assert j["abs_min_gib"] == 2.0
    assert abs(j["cp_headroom_gib"] - 6.4) < 1e-9
    assert abs(j["measured_leg_gib"] - 14.4) < 1e-9
    assert abs(j["static_leg_gib"] - 10.24) < 1e-9
    assert j["cap_gib"] == 64.0
    assert j["winning_leg"] == "measured_cp" and j["clamped"] is False
    dm = _floor(128.0, None)
    assert dm.measured_leg_gib is None and dm.measured_cp_rss_gib is None


def test_fixed_mode_is_legacy_formula_but_capped():
    assert _floor(64.0, None, mode=gov.FLOOR_MODE_FIXED).floor_gib == 8.0     # max(8, 5.12)
    assert _floor(128.0, None, mode=gov.FLOOR_MODE_FIXED).floor_gib == 10.24  # max(8, 10.24)
    d8 = _floor(8.0, None, mode=gov.FLOOR_MODE_FIXED)
    assert d8.floor_gib == 4.0 and d8.clamped   # even fixed mode can no longer eat an 8 GiB box


# ── 2. overrides: CLI/env precedence, clamping, loud log ────────────────────────────────────────
def test_env_override_parse(monkeypatch):
    assert gov.safety_floor_env_override_gib({}) is None
    assert gov.safety_floor_env_override_gib({gov.SAFETY_FLOOR_ENV: ""}) is None
    assert gov.safety_floor_env_override_gib({gov.SAFETY_FLOOR_ENV: "12.5"}) == 12.5
    assert gov.safety_floor_env_override_gib({gov.SAFETY_FLOOR_ENV: "not-a-number"}) is None


def test_override_precedence_cli_beats_env():
    env = {gov.SAFETY_FLOOR_ENV: "5.0"}
    assert gov.resolve_floor_override_gib(12.0, env) == 12.0   # explicit CLI wins
    assert gov.resolve_floor_override_gib(None, env) == 5.0    # env next
    assert gov.resolve_floor_override_gib(None, {}) is None    # derive


def test_override_wins_and_is_clamped_with_loud_log():
    logs: list[str] = []
    d = _floor(128.0, 8.0, override_gib=200.0, log_fn=logs.append)
    assert d.floor_gib == 64.0 and d.clamped and d.winning_leg == "override"
    assert logs and "CLAMP" in logs[0] and "200.00" in logs[0]
    logs.clear()
    d2 = _floor(128.0, 8.0, override_gib=0.5, log_fn=logs.append)
    assert d2.floor_gib == 2.0 and d2.clamped         # clamped UP to ABS_MIN
    assert logs and "CLAMP" in logs[0]
    d3 = _floor(128.0, 8.0, override_gib=20.0, log_fn=logs.append)
    assert d3.floor_gib == 20.0 and not d3.clamped    # in-range override applies verbatim


def test_env_override_flows_through_adaptive_ceiling(monkeypatch):
    monkeypatch.setenv(gov.SAFETY_FLOOR_ENV, "20.0")
    c = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=24.0, tracked_current_gib=8.0)
    assert c.safety_margin_gib == 20.0
    assert c.floor_decomposition["winning_leg"] == "override"


# ── 3. smoothing / hysteresis (admission verdicts do not flap) ─────────────────────────────────
def test_smoother_rises_instantly_decays_slowly():
    s = gov.SafetyFloorSmoother(max_decay_gib_per_tick=0.25)
    assert s.update(10.4) == 10.4
    assert s.update(18.4) == 18.4              # rise is instant (protection never lags)
    assert s.update(10.4) == pytest.approx(18.15)   # decay bounded at 0.25/tick
    assert s.update(10.4) == pytest.approx(17.90)
    assert s.update(19.0) == 19.0              # rises again instantly


def test_flap_oscillating_control_plane_raw_flaps_smoothed_stable():
    """cp oscillates 4<->12 GiB @128 (raw floor 10.4<->18.4 -> raw ceiling 117.6<->109.6).
    A 95 GiB launch on used=20 projects to 115: RAW verdicts flap ADMIT/REFUSE tick-to-tick;
    SMOOTHED verdicts are stable (REFUSE) from the first high-cp tick onward."""
    cps = [4.0, 12.0] * 10
    smoother = gov.SafetyFloorSmoother(max_decay_gib_per_tick=0.25)

    def verdict(margin):
        c = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=20.0,
                                         tracked_current_gib=0.0, safety_margin_gib=margin)
        return gov.admission_decision(projected_new_gib=95.0, system_used_gib=20.0,
                                      active_growth_headroom_gib=0.0, ceiling=c).admit

    raw_verdicts, smooth_verdicts = [], []
    for cp in cps:
        raw = _floor(128.0, cp).floor_gib
        raw_verdicts.append(verdict(raw))
        smooth_verdicts.append(verdict(smoother.update(raw)))
    assert True in raw_verdicts[1:] and False in raw_verdicts[1:]   # raw FLAPS
    assert all(v is False for v in smooth_verdicts[1:])             # smoothed STABLE


# ── 4. tier-scaled pressure thresholds + band-threshold auto-scaling ───────────────────────────
def test_derived_pressure_thresholds_numbers():
    assert gov.derived_critical_free_gib(128.0) == pytest.approx(8.4)   # legacy 8.0: stricter
    assert gov.derived_warn_free_gib(128.0) == pytest.approx(14.8)      # legacy 15.0: within 1.4%
    assert gov.derived_critical_free_gib(128.0) >= 8.0
    assert gov.derived_critical_free_gib(8.0) == pytest.approx(3.0)
    assert gov.derived_warn_free_gib(8.0) == pytest.approx(4.0)
    # monotone in RAM
    crit = [gov.derived_critical_free_gib(t) for t in TIERS]
    assert crit == sorted(crit)


def test_band_thresholds_reproduce_tier_numbers():
    env128 = gov.band_envelope_gib(128.0, 0.85)
    assert env128 == pytest.approx(108.80)
    assert 0.85 * env128 == pytest.approx(92.48)   # yellow onset, true GiB
    assert 0.90 * env128 == pytest.approx(97.92)   # red onset
    env8 = gov.band_envelope_gib(8.0, 0.55)
    assert env8 == pytest.approx(4.40)
    assert 0.85 * env8 == pytest.approx(3.74)
    assert 0.90 * env8 == pytest.approx(3.96)
    # classification agrees at the computed thresholds on both tiers
    for env in (env128, env8):
        assert gov.classify_guard_band(0.849 * env, env) == gov.BAND_GREEN
        assert gov.classify_guard_band(0.85 * env, env) == gov.BAND_YELLOW
        assert gov.classify_guard_band(0.90 * env, env) == gov.BAND_RED


def test_band_defer_to_throttle_precedence_preserved_at_edge_tier():
    """8 GiB tier: avail 2.5 < derived critical 3.0 -> pressure 'critical' -> a red band DEFERS
    to the throttle (backstop BEHIND it) — #294 semantics preserved under tier scaling."""
    snap = gov.SystemMemorySnapshot(
        total_gib=8.0, available_gib=2.5, used_gib=5.5, free_gib=2.5, wired_gib=1.0,
        compressor_gib=1.0, swap_used_gib=1.0, pressure_level=1, load1=0, load5=0, load15=0)
    level = gov.classify_pressure(snap, warn_free_gib=gov.derived_warn_free_gib(8.0),
                                  critical_free_gib=gov.derived_critical_free_gib(8.0))
    assert level == "critical"
    job = gov.TrackedJob(label="edge", pid=9, pgid=9, cmd="python train_levelset_witness.py",
                         priority=50, projected_peak_gib=3.7, current_rss_gib=3.6, paused=False,
                         throttle_eligible=True, own_group_leader=True)
    d = gov.decide_guard_band_action(band=gov.BAND_RED, job=job, pressure_class=level,
                                     actual_rss_gib=4.2, envelope_gib=4.4)
    assert d.action == "defer_to_throttle"


# ── 5. GB-F1: admission-path units are TRUE GiB (converted once, at read) ──────────────────────
def test_admission_growth_headroom_true_gib_matches_review_numbers():
    """The review's measured case: projected 67.61 true / tracked 56.69 units = 54.06 true.
    Pre-fix growth = 67.61 - 56.69 = 10.92 (under-count 2.63 GiB, anti-conservative);
    post-fix TrackedJob carries TRUE GiB -> growth = 13.55."""
    job = gov.TrackedJob(label="levelset_witness_run", pid=1, pgid=1, cmd="python x.py",
                         priority=60, projected_peak_gib=67.61, current_rss_gib=54.06,
                         paused=False, throttle_eligible=True, own_group_leader=True)
    assert gov.sum_active_growth_headroom_gib([job]) == pytest.approx(13.55)
    # band + admission consume the SAME true-GiB field and agree on the same sample:
    assert gov.select_band_run([job]).current_rss_gib == pytest.approx(54.06)
    assert gov.measured_control_plane_rss_gib(
        used_gib=70.0, tracked_current_gib=gov.sum_tracked_current_gib([job])
    ) == pytest.approx(15.94)


def test_units_conversion_applied_exactly_once_at_read_boundary():
    """Structural: list_tracked_jobs is the ONLY conversion site; the band tick no longer
    converts locally (double-conversion would under-read by x0.9537^2)."""
    read_src = inspect.getsource(gov.list_tracked_jobs)
    assert read_src.count("TRACKED_RSS_UNITS_TO_GIB") == 1
    assert "group_rss_gb" in read_src
    band_src = inspect.getsource(gov.band_tick)
    assert "TRACKED_RSS_UNITS_TO_GIB" not in band_src
    # no other half-converted group_rss_gb consumer left in the governor module:
    module_src = inspect.getsource(gov)
    assert module_src.count("_mg.group_rss_gb(") == 1


def test_tracked_units_constant_value_unchanged():
    assert abs(gov.TRACKED_RSS_UNITS_TO_GIB - 0.95367431640625) < 1e-12


# ── 6. #246: full-tree pause/resume (mock process tree; NEVER touches a live pid) ──────────────
def _tree_samples():
    """wrapper(100, own pgid) -> bash(101, same pgid) -> trainer(102, OWN session/pgid — the
    start_new_session boundary the old killpg missed) -> trainer child(103). Plus an unregistered
    bystander (200) and a control-plane app (300) + a control-plane descendant (104)."""
    rows = [
        mg.ProcessSample(pid=100, ppid=1, rss_kb=10_000, command="python tools/safe_run.py --rss-mb 90000", pgid=100),
        mg.ProcessSample(pid=101, ppid=100, rss_kb=5_000, command="bash launch.sh", pgid=100),
        mg.ProcessSample(pid=102, ppid=101, rss_kb=56_000_000, command="python experiments/train_levelset_witness_realized_through_R_mlx.py", pgid=102),
        mg.ProcessSample(pid=103, ppid=102, rss_kb=1_000, command="python worker.py", pgid=102),
        mg.ProcessSample(pid=104, ppid=100, rss_kb=2_000, command="claude", pgid=100),  # control plane INSIDE tree
        mg.ProcessSample(pid=200, ppid=1, rss_kb=99_000_000, command="python unregistered_bystander.py", pgid=200),
        mg.ProcessSample(pid=300, ppid=1, rss_kb=3_000, command="claude", pgid=300),
    ]
    return {s.pid: s for s in rows}


def _tree_job():
    return gov.TrackedJob(label="levelset_witness_run", pid=100, pgid=100,
                          cmd="python tools/safe_run.py", priority=60, projected_peak_gib=67.61,
                          current_rss_gib=53.4, paused=False, throttle_eligible=True,
                          own_group_leader=True)


def test_job_tree_pids_covers_detached_session_deepest_first():
    """The stop set crosses the start_new_session boundary (trainer 102 + child 103 — the pre-fix
    killpg missed them entirely). The intermediate bash (101) is EXCLUDED by the shell denylist
    (defense-in-depth: shells are never signal targets; a parent bash simply blocks in wait()
    while its stopped child is paused — zero memory consequence)."""
    order = gov.job_tree_pids(_tree_job(), samples=_tree_samples())
    assert set(order) == {100, 102, 103}          # crosses the session boundary (102, 103)
    assert order == [103, 102, 100]               # deepest first (children before parents)


def test_job_tree_never_contains_unregistered_or_control_plane_pids():
    order = gov.job_tree_pids(_tree_job(), samples=_tree_samples())
    assert 200 not in order    # unregistered bystander: unreachable by construction
    assert 300 not in order    # control-plane app outside the tree
    assert 104 not in order    # control-plane app INSIDE the tree: gate re-applied per member
    assert 101 not in order    # shell denylist member (never a signal target)


def test_job_tree_rss_covers_the_run():
    """Post-#246 the pause scope ~= the run's actual RSS (the trainer's 56 GB is IN scope —
    the measured pre-fix scope was 0.02 GiB of a 54 GiB run)."""
    rss = gov.job_tree_rss_gib(_tree_job(), samples=_tree_samples())
    assert rss == pytest.approx(56_011_000 / 1024.0 ** 2)   # 53.42 true GiB (wrapper+trainer+child)
    assert rss > 50.0


def test_pause_job_sigstops_full_tree_deepest_first_and_is_idempotent(monkeypatch):
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))
    job, samples = _tree_job(), _tree_samples()
    assert gov.pause_job(job, samples=samples) is True
    assert [p for p, _ in sent] == [103, 102, 100]
    assert all(sig == signal.SIGSTOP for _, sig in sent)
    sent.clear()
    assert gov.pause_job(job, samples=samples) is True   # idempotent: same set, no error
    assert [p for p, _ in sent] == [103, 102, 100]


def test_resume_job_sigconts_reverse_order_exactly_the_stopped_set(monkeypatch):
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert gov.resume_job(_tree_job(), samples=_tree_samples()) is True
    assert [p for p, _ in sent] == [100, 102, 103]   # reverse of the pause order
    assert all(sig == signal.SIGCONT for _, sig in sent)


def test_signal_job_tree_refuses_kill_class_signals():
    with pytest.raises(AssertionError):
        gov._signal_job_tree(_tree_job(), signal.SIGKILL, samples=_tree_samples())
    src = (inspect.getsource(gov._signal_job_tree) + inspect.getsource(gov.pause_job)
           + inspect.getsource(gov.resume_job) + inspect.getsource(gov.job_tree_pids))
    assert "signal.SIGKILL" not in src and "signal.SIGTERM" not in src


def test_pause_job_race_tolerates_vanished_pid(monkeypatch):
    def _kill(pid, sig):
        if pid == 103:
            raise ProcessLookupError
    monkeypatch.setattr(os, "kill", _kill)
    assert gov.pause_job(_tree_job(), samples=_tree_samples()) is True   # others still signalled


def test_pause_job_fallback_when_guard_unavailable(monkeypatch):
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(gov, "_mg", None)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killed.append((pgid, sig)))
    assert gov.job_tree_pids(_tree_job()) == []          # fail-safe: no gates -> no tree
    assert gov.pause_job(_tree_job()) is True            # legacy pgid fallback still protects
    assert killed == [(100, signal.SIGSTOP)]


# ── 7. blackbox continuous band wiring (BUILD #298 part 2) ─────────────────────────────────────
@pytest.fixture()
def _daemon_harness(monkeypatch, tmp_path):
    """Hermetic run_daemon: singleton lock in tmp, no real sampling/append/log I/O."""
    monkeypatch.setattr(mbb, "_SINGLETON_LOCK", tmp_path / ".singleton.lock")
    monkeypatch.setattr(mbb, "_log_action", lambda msg: None)
    monkeypatch.setattr(mbb, "sample_once", lambda **kw: {"pressure": "normal"})
    monkeypatch.setattr(mbb, "append_sample", lambda s, **kw: None)
    calls: list[dict] = []

    def fake_band_tick(**kw):
        calls.append(kw)
        return SimpleNamespace(action="none", band="green", reason="test")

    monkeypatch.setattr(gov, "band_tick", fake_band_tick)
    return calls


def test_daemon_band_called_on_cadence(_daemon_harness):
    mbb.run_daemon(max_iterations=3, interval=0.0, fast_interval=0.0, govern=False,
                   band=True, band_interval_s=0.0)
    assert len(_daemon_harness) == 3    # every tick at cadence 0


def test_daemon_band_respects_interval(_daemon_harness):
    mbb.run_daemon(max_iterations=3, interval=0.0, fast_interval=0.0, govern=False,
                   band=True, band_interval_s=3600.0)
    assert len(_daemon_harness) == 1    # first tick fires, then waits out the hour


def test_daemon_band_opt_out(_daemon_harness):
    mbb.run_daemon(max_iterations=3, interval=0.0, fast_interval=0.0, govern=False, band=False)
    assert _daemon_harness == []


def test_daemon_band_apply_follows_govern(_daemon_harness, monkeypatch):
    monkeypatch.setattr(mbb, "_govern_tick", lambda w, c: (0, 0, None))
    mbb.run_daemon(max_iterations=1, interval=0.0, fast_interval=0.0, govern=True,
                   band=True, band_interval_s=0.0)
    assert _daemon_harness[-1]["apply"] is True
    mbb.run_daemon(max_iterations=1, interval=0.0, fast_interval=0.0, govern=False,
                   band=True, band_interval_s=0.0)
    assert _daemon_harness[-1]["apply"] is False   # --no-govern recorder emits DRY band rows


def test_daemon_band_error_is_nonfatal(monkeypatch, tmp_path):
    monkeypatch.setattr(mbb, "_SINGLETON_LOCK", tmp_path / ".singleton.lock")
    monkeypatch.setattr(mbb, "_log_action", lambda msg: None)
    monkeypatch.setattr(mbb, "sample_once", lambda **kw: {"pressure": "normal"})
    monkeypatch.setattr(mbb, "append_sample", lambda s, **kw: None)

    def boom(**kw):
        raise RuntimeError("band exploded")

    monkeypatch.setattr(gov, "band_tick", boom)
    assert mbb.run_daemon(max_iterations=2, interval=0.0, fast_interval=0.0, govern=False,
                          band=True, band_interval_s=0.0) == 0


def test_daemon_band_path_has_no_kill(monkeypatch):
    src = inspect.getsource(mbb.run_daemon)
    assert "SIGKILL" not in src and "SIGTERM" not in src and "kill(" not in src
    assert "band_tick" in src   # routes through the canonical #294 band evaluator only


def test_daemon_smoother_wired_into_samples(monkeypatch, tmp_path):
    """The daemon passes a persistent SafetyFloorSmoother into sample_once each tick."""
    monkeypatch.setattr(mbb, "_SINGLETON_LOCK", tmp_path / ".singleton.lock")
    monkeypatch.setattr(mbb, "_log_action", lambda msg: None)
    monkeypatch.setattr(mbb, "append_sample", lambda s, **kw: None)
    seen: list[object] = []

    def fake_sample_once(*, floor_smoother=None, **kw):
        seen.append(floor_smoother)
        return {"pressure": "normal"}

    monkeypatch.setattr(mbb, "sample_once", fake_sample_once)
    mbb.run_daemon(max_iterations=2, interval=0.0, fast_interval=0.0, govern=False, band=False)
    assert len(seen) == 2
    assert all(isinstance(s, gov.SafetyFloorSmoother) for s in seen)
    assert seen[0] is seen[1]   # ONE persistent smoother across ticks (hysteresis has memory)
