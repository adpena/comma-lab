# SPDX-License-Identifier: MIT
"""ddm_gb1 (#1073) — the memory_blackbox governor two-landing fix, one test per defect.

MEASURED INCIDENT (2026-08-15, memory ``governor-stuck-throttle-froze-three-live-measurements``):
``tools/memory_blackbox.py --daemon`` SIGSTOPped five live jobs and never resumed them, on a box
with 40.5 GiB free; and separately, three dead ``running`` rows in the durable-daemon registry
charged 100.0 GiB of phantom active growth and REFUSED a real relaunch twice.

Four defects, four test sections:

* **D1** ``measured_control_plane_rss_gib`` measured the WRONG OBJECT (total used incl. file cache).
* **D2** the throttle had no re-arm: resume gated solely on the sticky macOS pressure level.
* **D3** daemon death stranded its SIGSTOPped victims forever (no exit sweep, no ledger).
* **D4** ``safe_run``'s admission path did not reconcile the registry it counts.

HERMETIC: every fixture is a synthetic process table / snapshot / tmp ledger. The ONE test that
signals a real process spawns its OWN child, which stops ITSELF; the test only ever sends SIGCONT
(and a SIGKILL cleanup) to that child. No pre-existing process is signalled anywhere.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import memory_blackbox as mbb  # noqa: E402
import memory_guard as mg  # noqa: E402
import system_memory_governor as gov  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────────────────────────
def _proc(pid: int, command: str, rss_gib: float, ppid: int = 1) -> mg.ProcessSample:
    return mg.ProcessSample(pid=pid, ppid=ppid, rss_kb=int(rss_gib * 1024 * 1024),
                            command=command, pgid=pid)


def _incident_process_table() -> dict[int, mg.ProcessSample]:
    """The 2026-08-15 shape: a huge workload + a modest control plane on a 128 GiB box."""
    return {
        7445: _proc(7445, "claude --dangerously-skip-permissions", 4.0),
        8997: _proc(8997, "/usr/bin/python3 tools/memory_blackbox.py --daemon", 0.2),
        3001: _proc(3001, "/Applications/Codex.app/Contents/MacOS/Codex", 1.0),
        4242: _proc(4242, "python experiments/train_levelset_witness_realized_through_R_mlx.py", 60.0),
        4243: _proc(4243, "python tools/mp2_differential_eval.py", 26.8),
    }


def _snap(*, available: float, pressure_level: int, total: float = 128.0,
          reclaimable: float | None = None) -> gov.SystemMemorySnapshot:
    return gov.SystemMemorySnapshot(
        total_gib=total, available_gib=available, used_gib=total - available, free_gib=available,
        wired_gib=4.0, compressor_gib=1.0, swap_used_gib=0.0, pressure_level=pressure_level,
        load1=3.0, load5=2.0, load15=1.0, available_primary_gib=available, closure_gib=0.0,
        closure_ok=True, cross_validated=True, discrepancy_gib=0.0, fail_safe=False,
        available_reclaimable_gib=(available if reclaimable is None else reclaimable),
        used_committed_gib=total - available, reclaimable_ok=reclaimable is not None)


def _job(label="mp2_eval", pid=4243, *, paused=False, priority=-45, rss=26.8):
    return gov.TrackedJob(label=label, pid=pid, pgid=pid, cmd=f"python {label}.py",
                          priority=priority, projected_peak_gib=40.0, current_rss_gib=rss,
                          paused=paused, throttle_eligible=True, own_group_leader=True)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# D1 — the cp measurement counts NAMED CONTROL-PLANE PROCESSES, never total-used.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_d1_cp_rss_counts_only_named_control_plane_processes():
    """The incident table sums to 92.0 GiB — the EXACT number in the receipt "SAFETY-FLOOR CLAMP:
    measured_cp value 92.00 GiB clamped to 64.00 GiB" — of which 86.8 is OUR workload. The cp
    measurement must return the 5.2 GiB control plane, never the 92."""
    table = _incident_process_table()
    cp = gov.measured_control_plane_rss_gib(samples=table)
    assert cp == pytest.approx(4.0 + 1.0 + 0.2, abs=1e-6), (
        "cp RSS must sum ONLY claude + Codex + protection infra"
    )
    # ...and the workload it must NOT count is 86.8 GiB of the same table.
    total_rss = sum(s.rss_kb for s in table.values()) / (1024.0 ** 2)
    assert total_rss == pytest.approx(92.0, abs=1e-3), "the fixture reproduces the receipt exactly"
    assert cp < total_rss / 10.0


def test_d1_non_workload_used_preserves_the_ceiling_baseline_arithmetic():
    """The OLD arithmetic is a REAL quantity — the ceiling's baseline — and keeps working under its
    honest name. This is the exact case the pre-rename test asserted (70 used, 54.06 tracked)."""
    job = _job(label="levelset_witness_run", pid=1, rss=54.06)
    assert gov.non_workload_used_gib(
        used_gib=70.0, tracked_current_gib=gov.sum_tracked_current_gib([job])
    ) == pytest.approx(15.94)


def test_d1_wrong_object_measurement_is_what_clamped_the_floor():
    """REGRESSION on the measured receipt: 'SAFETY-FLOOR CLAMP: measured_cp value 92.00 GiB clamped
    to 64.00 GiB'. The old object clamps against the never-eat-the-box cap; the new one does not."""
    table = _incident_process_table()
    old = gov.non_workload_used_gib(used_gib=92.0, tracked_current_gib=0.0)
    new = gov.measured_control_plane_rss_gib(samples=table)
    old_floor = gov.derive_safety_floor(total_gib=128.0, measured_cp_rss_gib=old)
    new_floor = gov.derive_safety_floor(total_gib=128.0, measured_cp_rss_gib=new)
    assert old_floor.clamped and old_floor.floor_gib == pytest.approx(64.0)
    assert not new_floor.clamped
    assert new_floor.floor_gib == pytest.approx(5.2 + gov.cp_headroom_gib(128.0))
    assert new_floor.floor_gib < old_floor.floor_gib / 5.0


def test_d1_unavailable_measurement_returns_none_not_zero(monkeypatch):
    """Fail-safe: no process table => None => derive_safety_floor falls back to the STATIC policy
    leg. Returning 0.0 would silently DROP the control-plane reservation."""
    monkeypatch.setattr(gov, "_mg", None)
    assert gov.measured_control_plane_rss_gib() is None
    floor = gov.derive_safety_floor(total_gib=128.0, measured_cp_rss_gib=None)
    assert floor.measured_leg_gib is None
    assert floor.winning_leg == "static_frac"
    assert floor.floor_gib == pytest.approx(gov.DEFAULT_SAFETY_MARGIN_FRAC * 128.0)


def test_d1_empty_live_scan_is_a_failed_scan_not_zero_control_plane(monkeypatch):
    monkeypatch.setattr(gov._mg, "sample_processes", lambda: {})
    assert gov.measured_control_plane_rss_gib() is None
    # An explicitly INJECTED empty table is a caller's declared fixture, so it measures 0.0.
    assert gov.measured_control_plane_rss_gib(samples={}) == pytest.approx(0.0)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# D5 — INCIDENT #3 (2026-08-16): one launch tree must be charged ONCE, and a declared cap is a
# projection. Both are the D1 family: growth accounting that measures the wrong object.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def _mp2_tree() -> dict[int, mg.ProcessSample]:
    """The MEASURED incident-#3 tree (ps captured live 2026-08-16 before the job exited):

        39740 launch_detached wrapper   ppid=1      pgid=39740  rss 0.01 GiB
        39748 contest_auth_eval         ppid=39740  pgid=39740  rss 0.07 GiB   (own session)
        25923 evaluate.py               ppid=39748  pgid=39740  rss 6.86 GiB   (own session)

    ONE logical job. ``group_rss_gb(39740)`` = 7.30 GiB — it already contains all three."""
    return {
        1: mg.ProcessSample(pid=1, ppid=0, rss_kb=20_000, command="/sbin/launchd", pgid=1),
        39740: mg.ProcessSample(pid=39740, ppid=1, rss_kb=int(0.01 * 1024 * 1024),
                                command="python tools/launch_detached_byte_close.py --mp2",
                                pgid=39740),
        39748: mg.ProcessSample(pid=39748, ppid=39740, rss_kb=int(0.07 * 1024 * 1024),
                                command="python experiments/contest_auth_eval.py --device cpu",
                                pgid=39740),
        25923: mg.ProcessSample(pid=25923, ppid=39748, rss_kb=int(6.86 * 1024 * 1024),
                                command="python upstream/evaluate.py --n 600", pgid=39740),
    }


def _tracked(samples, registry_rows, monkeypatch):
    monkeypatch.setattr(gov, "_process_state", lambda pid: "")
    return gov.list_tracked_jobs(samples=samples, registry_rows=registry_rows,
                                 self_pid=999_999, self_pgid=999_999)


def test_d5a_one_launch_tree_is_charged_once_not_once_per_session(monkeypatch):
    """MEASURED refusal: 6.56 + 25.00 + 25.00 = 56.56 GiB charged against a 7.30 GiB tree, on a box
    using 37.9 of 128. The two descendants must now charge ZERO — their RSS is already inside the
    root's descendant-inclusive group RSS, so charging them again triple-counts one job."""
    jobs = {j.pid: j for j in _tracked(_mp2_tree(), [], monkeypatch)}
    assert set(jobs) == {39740, 39748, 25923}, "all three still VISIBLE — only the charge changes"
    assert jobs[39740].governed_descendant is False, "the tree ROOT carries the charge"
    assert jobs[39748].governed_descendant is True
    assert jobs[25923].governed_descendant is True
    assert jobs[39748].growth_headroom_gib == pytest.approx(0.0)
    assert jobs[25923].growth_headroom_gib == pytest.approx(0.0)
    charged = gov.sum_active_growth_headroom_gib(list(jobs.values()))
    assert charged == pytest.approx(gov.UNKNOWN_GROWTH_HEADROOM_GIB, abs=1e-6), (
        "exactly ONE unknown-peak charge for the tree, not three"
    )
    assert charged < 56.56 / 2


def test_d5a_tree_current_rss_is_counted_once(monkeypatch):
    """The same triple-count corrupts ``tracked_current`` (the ceiling BASELINE input): three rows
    of ~7 GiB each for one 7.30 GiB tree."""
    samples = _mp2_tree()
    jobs = _tracked(samples, [], monkeypatch)
    tree_rss = mg.group_rss_gb(samples, 39740) * gov.TRACKED_RSS_UNITS_TO_GIB
    assert gov.sum_tracked_current_gib(jobs) == pytest.approx(tree_rss, abs=1e-6)
    naive = sum(j.current_rss_gib for j in jobs)
    assert naive > 2.5 * gov.sum_tracked_current_gib(jobs), "the pre-fix over-count was ~3x"


def test_d5a_a_registered_root_still_wins_with_its_declared_projection(monkeypatch):
    """The DECLARED path keeps working: a registered wrapper's projection is the tree's charge."""
    rows = [{"label": "mp2_eval", "pid": 39740, "pgid": 39740, "status": "running",
             "projected_peak_gib": 9.0, "cmd": ["python", "tools/launch_detached_byte_close.py"]}]
    jobs = {j.pid: j for j in _tracked(_mp2_tree(), rows, monkeypatch)}
    assert jobs[39740].projected_peak_gib == pytest.approx(9.0)
    assert gov.sum_active_growth_headroom_gib(list(jobs.values())) == pytest.approx(
        9.0 - jobs[39740].current_rss_gib)


def test_d5a_an_independent_job_is_still_charged_independently(monkeypatch):
    """FALSE-NEGATIVE guard: aggregation must not swallow a genuinely separate launch tree."""
    samples = _mp2_tree()
    samples[50000] = mg.ProcessSample(
        pid=50000, ppid=1, rss_kb=int(8.0 * 1024 * 1024),
        command="python experiments/train_levelset_witness_realized_through_R_mlx.py", pgid=50000)
    jobs = {j.pid: j for j in _tracked(samples, [], monkeypatch)}
    assert jobs[50000].governed_descendant is False
    assert gov.sum_active_growth_headroom_gib(list(jobs.values())) == pytest.approx(
        2 * gov.UNKNOWN_GROWTH_HEADROOM_GIB, abs=1e-6), "two trees, two charges"


def test_d5a_init_is_never_a_launch_tree_root(monkeypatch):
    """pid 1 is every process's ancestor. If it counted, ONE candidate under launchd would mark
    every other candidate a descendant and the whole gate would go blind."""
    samples = _mp2_tree()
    del samples[39748], samples[25923]
    samples[50000] = mg.ProcessSample(
        pid=50000, ppid=1, rss_kb=int(8.0 * 1024 * 1024),
        command="python experiments/contest_auth_eval.py", pgid=50000)
    jobs = {j.pid: j for j in _tracked(samples, [], monkeypatch)}
    assert jobs[39740].governed_descendant is False
    assert jobs[50000].governed_descendant is False


def test_d5b_declared_safe_run_cap_is_the_projection_for_the_dashboard():
    """MEASURED: the LIVE dashboard registry row (pid 25204) carries no ``projected_peak_gib`` but
    launches through ``safe_run.py --rss-mb 2500``. Pre-fix it resolved to 0.22 + 25 = 25.22 and was
    counted HEAVY; the cap says 2.44, which is below HEAVY_MIN_PROJECTED_GIB, so it is excluded —
    the #370 control-plane intent, restored from the argv instead of from a launcher's memory."""
    cmd = ("python tools/safe_run.py --rss-mb 2500 --skip-admission-gate --label "
           "levelset_dash_server -- python tools/dashboard_server.py --port 8790")
    cap = gov.declared_rss_cap_gib(cmd)
    assert cap == pytest.approx(2500 * 1024 ** 2 / 1024 ** 3)
    assert cap == pytest.approx(2.44, abs=0.01)
    assert cap < gov.HEAVY_MIN_PROJECTED_GIB
    proj = gov.resolve_projected_peak_gib(cap, 0.22, cmd=cmd)
    assert proj == pytest.approx(cap)
    assert gov.sum_active_growth_headroom_gib([
        gov.TrackedJob(label="dash", pid=25204, pgid=25204, cmd=cmd, priority=0,
                       projected_peak_gib=proj, current_rss_gib=0.22, paused=False,
                       throttle_eligible=True, own_group_leader=True)]) == pytest.approx(0.0)


def test_d5b_cap_parsing_forms_and_refusals():
    assert gov.declared_rss_cap_gib("safe_run.py --rss-cap-mb 4096 -- python x.py") == pytest.approx(4.0)
    assert gov.declared_rss_cap_gib("safe_run.py --rss-mb=1024 -- python x.py") == pytest.approx(1.0)
    for none_case in ("python x.py", "safe_run.py --rss-mb", "safe_run.py --rss-mb junk",
                      "safe_run.py --rss-mb 0", "safe_run.py --rss-mb -5"):
        assert gov.declared_rss_cap_gib(none_case) is None, none_case


def test_d5b_an_explicit_projection_still_outranks_the_cap(monkeypatch):
    """Precedence: a real recorded projection wins; the cap is only the fallback for an UNKNOWN
    peak. A cap is an upper bound, not a forecast."""
    rows = [{"label": "dash", "pid": 39740, "pgid": 39740, "status": "running",
             "projected_peak_gib": 11.0,
             "cmd": ["python", "tools/safe_run.py", "--rss-mb", "2500", "--", "python", "d.py"]}]
    jobs = {j.pid: j for j in _tracked(_mp2_tree(), rows, monkeypatch)}
    assert jobs[39740].projected_peak_gib == pytest.approx(11.0)


def test_d5_incident3_refusal_arithmetic_is_reproduced_and_then_cured(monkeypatch):
    """END TO END on the measured incident: dashboard + mp2 tree. Pre-fix charge 100.0 GiB refused
    a READY_TO_FIRE 20 GiB launch under a 116 GiB ceiling on a 37.9-GiB-used box; post-fix the same
    scene charges one tree's worth."""
    samples = _mp2_tree()
    samples[25204] = mg.ProcessSample(
        pid=25204, ppid=1, rss_kb=int(0.22 * 1024 * 1024), pgid=25204,
        command=("python tools/safe_run.py --rss-mb 2500 --label levelset_dash_server -- "
                 "python tools/dashboard_server.py --port 8790"))
    rows = [{"label": "levelset_dash_server", "pid": 25204, "pgid": 25204, "status": "running",
             "cmd": ["python", "tools/safe_run.py", "--rss-mb", "2500", "--", "python",
                     "tools/dashboard_server.py"]}]
    jobs = _tracked(samples, rows, monkeypatch)
    charged = gov.sum_active_growth_headroom_gib(jobs)
    assert charged == pytest.approx(gov.UNKNOWN_GROWTH_HEADROOM_GIB, abs=1e-6)
    assert charged < 100.0 / 3, "the measured 100.0 GiB phantom charge is gone"
    # ...and the launch the incident refused is admitted under the same ceiling.
    assert 37.9 + 20.0 + charged < gov.OPERATOR_CEILING_GIB_DEFAULT


# ════════════════════════════════════════════════════════════════════════════════════════════════
# D2 — the throttle re-arms on the governor's OWN reference + a max-stop escape hatch.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def test_d2_thresholds_are_ordered_and_derived_at_every_tier():
    for total in (8.0, 32.0, 128.0):
        crit = gov.derived_critical_free_gib(total)
        warn = gov.derived_warn_free_gib(total)
        resume = gov.derived_resume_free_gib(total)
        assert crit < warn < resume, f"tier {total}: rungs must be strictly ordered"
        assert resume - warn == pytest.approx(gov.cp_headroom_gib(total))
    assert gov.derived_resume_free_gib(128.0) == pytest.approx(21.2)


def test_d2_sticky_os_warn_with_high_free_RESUMES_the_frozen_jobs():
    """THE INCIDENT, exactly: OS pressure_level=2 (sticky warn) at 40.4 GiB available with jobs
    SIGSTOPped. Pre-fix this returned no resume forever."""
    snap = _snap(available=40.4, pressure_level=gov.PRESSURE_WARN)
    level = gov.classify_pressure(snap, warn_free_gib=gov.derived_warn_free_gib(128.0),
                                  critical_free_gib=gov.derived_critical_free_gib(128.0))
    assert level == "warn", "the sticky OS level still classifies warn — that part is unchanged"
    action = gov.decide_governor_action(
        level=level, consecutive_warn=99, consecutive_critical=0,
        jobs=[_job(paused=True), _job(label="wd3_train", pid=92077, paused=True, priority=60)],
        available_gib=gov.governing_free_gib(snap),
        resume_free_gib=gov.derived_resume_free_gib(snap.total_gib))
    assert action.action == "resume"
    assert {j.label for j in action.resume_targets} == {"mp2_eval", "wd3_train"}
    assert action.resume_targets[0].label == "wd3_train", "highest priority resumes first"
    assert "sticky" in action.reason


def test_d2_sticky_os_warn_with_high_free_does_NOT_pause_the_anti_flap_veto():
    """Without this veto the two rungs alternate at the sample rate: resume (free high) -> pause
    (level warn) -> resume ... A flapping SIGSTOP is not protection either."""
    snap = _snap(available=40.4, pressure_level=gov.PRESSURE_WARN)
    action = gov.decide_governor_action(
        level="warn", consecutive_warn=99, consecutive_critical=0, jobs=[_job(paused=False)],
        available_gib=gov.governing_free_gib(snap),
        resume_free_gib=gov.derived_resume_free_gib(snap.total_gib))
    assert action.action == "alert"
    assert action.target is None
    assert "NO pause" in action.reason


def test_d2_genuine_pressure_still_pauses():
    """The cure must not disarm the guard: below the resume threshold the pause rung is intact."""
    snap = _snap(available=6.0, pressure_level=gov.PRESSURE_CRITICAL)
    action = gov.decide_governor_action(
        level="critical", consecutive_warn=0, consecutive_critical=5, jobs=[_job(paused=False)],
        available_gib=gov.governing_free_gib(snap),
        resume_free_gib=gov.derived_resume_free_gib(snap.total_gib))
    assert action.action == "pause" and action.target is not None
    assert action.target.label == "mp2_eval"


def test_d2_hysteresis_band_keeps_a_paused_job_paused():
    """Between warn (14.8) and resume (21.2) nothing actuates: no resume, no pause. That dead zone
    is what makes the two rungs non-flapping."""
    snap = _snap(available=18.0, pressure_level=gov.PRESSURE_WARN)
    action = gov.decide_governor_action(
        level="warn", consecutive_warn=99, consecutive_critical=0, jobs=[_job(paused=True)],
        available_gib=gov.governing_free_gib(snap),
        resume_free_gib=gov.derived_resume_free_gib(snap.total_gib))
    assert action.action != "resume"


def test_d2_escape_hatch_resumes_even_under_critical_pressure():
    """The hatch outranks every rung — including CRITICAL. A job held past the ceiling is a frozen
    measurement whatever the pressure says, and the guard must admit it is not working."""
    now = 10_000.0
    jobs = [_job(paused=True), _job(label="wc1_decode", pid=39748, paused=True)]
    action = gov.decide_governor_action(
        level="critical", consecutive_warn=0, consecutive_critical=9, jobs=jobs,
        available_gib=2.0, resume_free_gib=gov.derived_resume_free_gib(128.0),
        paused_since_ts={4243: now - 301.0, 39748: now - 10.0},
        now_ts=now, max_stop_duration_s=300.0)
    assert action.action == "resume"
    assert [j.label for j in action.resume_targets] == ["mp2_eval"], "only the OVERDUE job"
    assert "ESCAPE HATCH" in action.reason


def test_d2_escape_hatch_bounds_the_measured_75_minute_freeze():
    """The measured hold was 75+ minutes. At the derived ceiling the hatch fires 15x earlier."""
    assert gov.DEFAULT_MAX_STOP_DURATION_S == pytest.approx(300.0)
    assert gov.DEFAULT_MAX_STOP_DURATION_S < 75 * 60 / 10


def test_d2_escape_hatch_ceiling_may_be_retuned_but_never_disabled(monkeypatch, capsys):
    monkeypatch.setenv(gov.MAX_STOP_DURATION_ENV, "120")
    assert gov.resolve_max_stop_duration_s() == pytest.approx(120.0)
    for bad in ("0", "-5", "nonsense"):
        monkeypatch.setenv(gov.MAX_STOP_DURATION_ENV, bad)
        assert gov.resolve_max_stop_duration_s() == pytest.approx(gov.DEFAULT_MAX_STOP_DURATION_S)
    assert "WARNING" in capsys.readouterr().err
    monkeypatch.delenv(gov.MAX_STOP_DURATION_ENV, raising=False)
    assert gov.resolve_max_stop_duration_s() == pytest.approx(gov.DEFAULT_MAX_STOP_DURATION_S)


def test_d2_governing_free_uses_the_conservative_reclaimable_basis():
    """available = free + inactive over-reports on macOS. The actuation basis takes the smaller
    reclaimable number, which makes the veto fire LESS and the resume fire LATER."""
    snap = _snap(available=40.0, pressure_level=gov.PRESSURE_WARN, reclaimable=12.0)
    assert gov.governing_free_gib(snap) == pytest.approx(12.0)
    legacy = _snap(available=40.0, pressure_level=gov.PRESSURE_WARN)
    assert gov.governing_free_gib(legacy) == pytest.approx(40.0)
    # With the conservative basis the resume does NOT fire at 12 < 21.2 — protection preserved.
    action = gov.decide_governor_action(
        level="warn", consecutive_warn=99, consecutive_critical=0, jobs=[_job(paused=True)],
        available_gib=gov.governing_free_gib(snap),
        resume_free_gib=gov.derived_resume_free_gib(128.0))
    assert action.action != "resume"


def test_d2_old_callsites_without_rearm_inputs_behave_exactly_as_before():
    """Backward compatibility: the re-arm rungs are skipped when a caller supplies nothing."""
    paused = gov.decide_governor_action(level="normal", consecutive_warn=0,
                                        consecutive_critical=0, jobs=[_job(paused=True)])
    assert paused.action == "resume"
    idle = gov.decide_governor_action(level="normal", consecutive_warn=0,
                                      consecutive_critical=0, jobs=[_job(paused=False)])
    assert idle.action == "none"
    warn = gov.decide_governor_action(level="warn", consecutive_warn=5,
                                      consecutive_critical=0, jobs=[_job(paused=False)])
    assert warn.action == "pause"


# ════════════════════════════════════════════════════════════════════════════════════════════════
# D3 — the persisted stopped-set ledger + the exit sweep.
# ════════════════════════════════════════════════════════════════════════════════════════════════
@pytest.fixture()
def ledger(tmp_path):
    return {"path": tmp_path / "stopped.json", "lock_path": tmp_path / "stopped.lock"}


def test_d3_ledger_round_trip_and_idempotent_stop_timestamp(ledger):
    mbb.record_stopped_pids([100, 101], label="mp2_eval", now_ts=1000.0, **ledger)
    assert mbb.paused_since_ts(path=ledger["path"]) == {100: 1000.0, 101: 1000.0}
    # A re-record (the daemon re-observing the same paused job) must NOT reset the clock, or the
    # escape hatch would never mature.
    mbb.record_stopped_pids([100], label="mp2_eval", now_ts=9999.0, **ledger)
    assert mbb.paused_since_ts(path=ledger["path"])[100] == 1000.0
    mbb.record_resumed(pids=[100], **ledger)
    assert set(mbb.paused_since_ts(path=ledger["path"])) == {101}
    mbb.record_resumed(label="mp2_eval", **ledger)
    assert mbb.paused_since_ts(path=ledger["path"]) == {}


def test_d3_corrupt_or_absent_ledger_reads_empty_never_raises(tmp_path):
    assert mbb.load_stopped_ledger(path=tmp_path / "nope.json")["stopped"] == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert mbb.load_stopped_ledger(path=bad)["stopped"] == {}
    assert mbb.paused_since_ts(path=bad) == {}


def test_d3_sweep_conts_stopped_pids_and_skips_recycled_ones(ledger):
    """PID recycling safety: only a pid still in state ``T`` is signalled. Every examined pid is
    dropped either way, so the sweep is idempotent."""
    mbb.record_stopped_pids([200, 201, 202], label="wc1", now_ts=1.0, **ledger)
    sent: list[tuple[int, int]] = []
    states = {200: "T", 201: "S", 202: "T+"}   # 201 was recycled into live work

    def fake_kill(pid, sig):
        sent.append((pid, sig))

    out = mbb.resume_all_stopped(kill_fn=fake_kill, state_fn=lambda p: states.get(p, ""),
                                 log=False, **ledger)
    assert out["resumed"] == [200, 202]
    assert out["skipped_not_stopped"] == [201]
    assert sent == [(200, signal.SIGCONT), (202, signal.SIGCONT)]
    assert all(sig == signal.SIGCONT for _, sig in sent), "STOP/CONT path is CONT-only here"
    assert mbb.paused_since_ts(path=ledger["path"]) == {}
    # idempotent: a second sweep signals nothing.
    sent.clear()
    assert mbb.resume_all_stopped(kill_fn=fake_kill, state_fn=lambda p: "T",
                                  log=False, **ledger)["resumed"] == []
    assert sent == []


def test_d3_sweep_drops_malformed_keys_so_the_ledger_cannot_grow_without_bound(ledger):
    """A key that is not a pid can never be signalled. Leaving it would make the ledger a
    monotonically growing file that every tick re-reads."""
    mbb.record_stopped_pids([400], label="ok", now_ts=1.0, **ledger)
    data = mbb.load_stopped_ledger(path=ledger["path"])
    data["stopped"]["not-a-pid"] = {"label": "junk", "stopped_ts": 1.0}
    ledger["path"].write_text(__import__("json").dumps(data), encoding="utf-8")
    out = mbb.resume_all_stopped(kill_fn=lambda p, s: None, state_fn=lambda p: "T",
                                 log=False, **ledger)
    assert out["resumed"] == [400]
    assert mbb.load_stopped_ledger(path=ledger["path"])["stopped"] == {}


def test_d3_sweep_tolerates_a_vanished_pid(ledger):
    mbb.record_stopped_pids([300], label="gone", now_ts=1.0, **ledger)

    def boom(pid, sig):
        raise ProcessLookupError(pid)

    out = mbb.resume_all_stopped(kill_fn=boom, state_fn=lambda p: "T", log=False, **ledger)
    assert out["missing"] == [300] and out["resumed"] == []
    assert mbb.paused_since_ts(path=ledger["path"]) == {}


def test_d3_sweep_really_resumes_a_stopped_process(ledger):
    """The one non-synthetic test. It spawns its OWN child, the child stops ITSELF, and the sweep
    SIGCONTs it. No pre-existing process is signalled; the child is killed in ``finally``."""
    code = "import os, signal, time\nos.kill(os.getpid(), signal.SIGSTOP)\ntime.sleep(30)\n"
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if gov.is_paused_state(gov._process_state(proc.pid)):
                break
            time.sleep(0.05)
        assert gov.is_paused_state(gov._process_state(proc.pid)), "child did not self-stop"

        mbb.record_stopped_pids([proc.pid], label="own_test_child", **ledger)
        out = mbb.resume_all_stopped(log=False, **ledger)
        assert out["resumed"] == [proc.pid]

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if not gov.is_paused_state(gov._process_state(proc.pid)):
                break
            time.sleep(0.05)
        assert not gov.is_paused_state(gov._process_state(proc.pid)), "SIGCONT did not take"
        assert mbb.paused_since_ts(path=ledger["path"]) == {}
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_d3_exit_handlers_are_installed_for_term_and_int(monkeypatch):
    """A daemon death must not strand its victims. SIGKILL runs no handler by definition — which is
    exactly why the stopped set is on DISK and ``--resume-stopped`` exists."""
    monkeypatch.setattr(mbb, "_EXIT_SWEEP_INSTALLED", False)
    registered: list = []
    monkeypatch.setattr(mbb.atexit, "register", lambda fn: registered.append(fn))
    installed: dict[int, object] = {}
    monkeypatch.setattr(mbb.signal, "signal", lambda s, h: installed.setdefault(s, h))
    mbb.install_exit_resume_handlers()
    assert len(registered) == 1
    assert set(installed) == {signal.SIGTERM, signal.SIGINT}
    # Idempotent: a second install adds nothing.
    mbb.install_exit_resume_handlers()
    assert len(registered) == 1


def test_d3_signal_handler_sweeps_then_exits(monkeypatch):
    monkeypatch.setattr(mbb, "_EXIT_SWEEP_INSTALLED", False)
    monkeypatch.setattr(mbb.atexit, "register", lambda fn: None)
    handlers: dict[int, object] = {}
    monkeypatch.setattr(mbb.signal, "signal", lambda s, h: handlers.setdefault(s, h))
    swept: list[int] = []
    monkeypatch.setattr(mbb, "resume_all_stopped", lambda **kw: swept.append(1) or {})
    monkeypatch.setattr(mbb, "_log_action", lambda msg: None)
    mbb.install_exit_resume_handlers()
    with pytest.raises(SystemExit) as exc:
        handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert exc.value.code == 128 + int(signal.SIGTERM)
    assert swept == [1], "the handler must SIGCONT the stopped set BEFORE exiting"


def test_d3_pre_existing_paused_jobs_get_their_clock_started(ledger):
    """A victim of a PREVIOUS (pre-fix or restarted) daemon has no recorded age. First observation
    starts the clock so the hatch bounds the hold at something FINITE."""
    jobs = [_job(paused=True), _job(label="already_known", pid=555, paused=True),
            _job(label="running_fine", pid=777, paused=False)]
    mbb.record_stopped_pids([555], label="already_known", now_ts=50.0, path=ledger["path"],
                            lock_path=ledger["lock_path"])
    returned = mbb._adopt_unrecorded_paused(
        jobs, ledger_path=ledger["path"], ledger_lock_path=ledger["lock_path"], now_ts=2000.0)
    since = mbb.paused_since_ts(path=ledger["path"])
    assert returned == since, "the returned map must equal the ledger — one read per tick"
    assert since[4243] == 2000.0, "the unrecorded victim's clock starts now"
    assert since[555] == 50.0, "an already-recorded victim keeps its true age"
    assert 777 not in since, "a running job is never recorded as stopped"


def test_d3_stopped_scope_falls_back_to_the_root_pid(monkeypatch):
    monkeypatch.setattr(gov, "job_tree_pids", lambda job, **kw: [])
    assert mbb._stopped_scope_pids(_job(pid=4243)) == [4243]
    monkeypatch.setattr(gov, "job_tree_pids", lambda job, **kw: [4243, 4244])
    assert mbb._stopped_scope_pids(_job(pid=4243)) == [4243, 4244]

    def boom(job, **kw):
        raise RuntimeError("ps raced")

    monkeypatch.setattr(gov, "job_tree_pids", boom)
    assert mbb._stopped_scope_pids(_job(pid=4243)) == [4243]


def test_d3_cli_exposes_the_ledger_and_the_recovery_sweep(monkeypatch, capsys):
    monkeypatch.setattr(mbb, "load_stopped_ledger", lambda **kw: {"stopped": {"1": {}}})
    assert mbb.main(["--stopped-ledger"]) == 0
    assert '"stopped"' in capsys.readouterr().out
    monkeypatch.setattr(mbb, "resume_all_stopped", lambda **kw: {"resumed": [1]})
    assert mbb.main(["--resume-stopped"]) == 0
    assert '"resumed"' in capsys.readouterr().out


# ════════════════════════════════════════════════════════════════════════════════════════════════
# D4 — safe_run's admission path reconciles the registry BEFORE it projects.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def _fake_admission_modules(monkeypatch, order: list[str], *, admit: bool, enforcing: bool = True):
    fake_gov = types.ModuleType("system_memory_governor")

    class _D:
        def __init__(self):
            self.admit = admit
            self.reason = "active-growth 100.0 GiB (4 jobs) — projected 181.6 > ceiling 116.0"

    class _Ctx:
        decision = _D()

    def _decide(**kwargs):
        order.append("live_admission_decision")
        return _Ctx()

    fake_gov.live_admission_decision = _decide
    fake_gov.admission_enforcing = lambda: enforcing
    fake_sdd = types.ModuleType("spawn_durable_daemon")

    def _reconcile(*, verbose=True):
        order.append("reconcile_dead_daemons")
        return 3

    fake_sdd.reconcile_dead_daemons = _reconcile
    monkeypatch.setitem(sys.modules, "system_memory_governor", fake_gov)
    monkeypatch.setitem(sys.modules, "spawn_durable_daemon", fake_sdd)


def test_d4_safe_run_reconciles_before_the_sum_over_ram_decision(monkeypatch):
    import argparse

    import safe_run

    order: list[str] = []
    _fake_admission_modules(monkeypatch, order, admit=True)
    ns = argparse.Namespace(skip_admission_gate=False, projected_gib=81.6, rss_mb=90000,
                            admission_override_rationale=None)
    assert safe_run._system_admission_gate(ns, ["python", "x.py"]) is None
    assert order == ["reconcile_dead_daemons", "live_admission_decision"], (
        "the registry must be converged to ground truth BEFORE it is counted"
    )


def test_d4_reconcile_leg_is_fail_open_and_the_decision_stays_fail_closed(monkeypatch):
    """A reconcile hiccup must never crash a launch; a genuine refusal must still refuse."""
    import argparse

    import safe_run

    order: list[str] = []
    _fake_admission_modules(monkeypatch, order, admit=False)

    def boom(*, verbose=True):
        order.append("reconcile_raised")
        raise RuntimeError("registry locked by a sibling")

    sys.modules["spawn_durable_daemon"].reconcile_dead_daemons = boom
    ns = argparse.Namespace(skip_admission_gate=False, projected_gib=81.6, rss_mb=90000,
                            admission_override_rationale=None)
    assert safe_run._system_admission_gate(ns, ["python", "x.py"]) == 5
    assert order == ["reconcile_raised", "live_admission_decision"]


def test_d4_skip_admission_gate_still_short_circuits(monkeypatch):
    import argparse

    import safe_run

    order: list[str] = []
    _fake_admission_modules(monkeypatch, order, admit=False)
    ns = argparse.Namespace(skip_admission_gate=True, projected_gib=81.6, rss_mb=90000,
                            admission_override_rationale=None)
    assert safe_run._system_admission_gate(ns, ["python", "x.py"]) is None
    assert order == []


def test_d4_phantom_growth_arithmetic_is_the_measured_100_gib():
    """The refusal receipt's number, re-derived from the constant: 4 unknown-peak rows x 25 GiB."""
    assert gov.UNKNOWN_GROWTH_HEADROOM_GIB == pytest.approx(25.0)
    assert 4 * gov.UNKNOWN_GROWTH_HEADROOM_GIB == pytest.approx(100.0)
    assert 3 * gov.UNKNOWN_GROWTH_HEADROOM_GIB == pytest.approx(75.0), (
        "three of the four rows were DEAD — reconciling them is the whole 75 GiB"
    )


# ════════════════════════════════════════════════════════════════════════════════════════════════
# SECOND LANDING — the gate refuses both anti-patterns and reads CLEAN on the cure.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


_PREFIX_THROTTLE = (
    "def decide(level, jobs):\n"
    "    paused = [j for j in jobs if j.paused]\n"
    '    if level == "normal" and paused:\n'
    "        return resume_targets(paused)\n"
    "    return None\n"
    "def resume_targets(p):\n"
    "    return tuple(p)\n"
)
_CURED_THROTTLE = (
    "def decide(level, jobs, available_gib, resume_free_gib, max_stop_duration_s):\n"
    "    paused = [j for j in jobs if j.paused]\n"
    '    if level == "normal" or available_gib >= resume_free_gib:\n'
    "        return resume_targets(paused)\n"
    "    return None\n"
    "def resume_targets(p):\n"
    "    return tuple(p)\n"
)
_PREFIX_ADMISSION = (
    "import system_memory_governor as gov\n"
    "def gate(p):\n"
    "    ctx = gov.live_admission_decision(projected_new_gib=p)\n"
    "    return 5 if not ctx.decision.admit else None\n"
)
_CURED_ADMISSION = (
    "import system_memory_governor as gov\n"
    "import spawn_durable_daemon as sdd\n"
    "def gate(p):\n"
    "    sdd.reconcile_dead_daemons(verbose=False)\n"
    "    ctx = gov.live_admission_decision(projected_new_gib=p)\n"
    "    return 5 if not ctx.decision.admit else None\n"
)


def _gate():
    from tac.confound_gates import check_throttle_rearms_and_admission_reconciles

    return check_throttle_rearms_and_admission_reconciles


def test_gate_fires_on_both_prefix_anti_patterns(tmp_path):
    _write(tmp_path, "tools/throttle.py", _PREFIX_THROTTLE)
    _write(tmp_path, "tools/admit.py", _PREFIX_ADMISSION)
    found = _gate()(repo_root=tmp_path, strict=False, verbose=False)
    assert len(found) == 2
    assert any("throttle.py" in v and "resume_free_gib" in v for v in found)
    assert any("admit.py" in v and "reconcile_dead_daemons" in v for v in found)


def test_gate_zeroes_on_the_cure(tmp_path):
    """DETECTOR-ZEROES-ON-THE-CURE: apply exactly the fixes this landing applied and the gauge must
    read clean. A gate that still fires after its own cure is a permanently-red gate."""
    _write(tmp_path, "tools/throttle.py", _CURED_THROTTLE)
    _write(tmp_path, "tools/admit.py", _CURED_ADMISSION)
    assert _gate()(repo_root=tmp_path, strict=False, verbose=False) == []


def test_gate_reads_clean_on_the_real_fixed_repo():
    """The live surface, post-fix. Pre-fix this measured 5 (3 throttle functions + 2 admission
    modules); the pre-fix count is recorded in the ddm_gb1 verdict memo."""
    assert _gate()(repo_root=_REPO, strict=False, verbose=False) == []


def test_gate_honours_both_waivers(tmp_path):
    _write(tmp_path, "tools/throttle.py",
           _PREFIX_THROTTLE.replace("def decide(level, jobs):",
                                    "def decide(level, jobs):  # THROTTLE_REARM_OK: dry-run "
                                    "reporter, never actuates"))
    _write(tmp_path, "tools/admit.py",
           _PREFIX_ADMISSION.replace("ctx = gov.live_admission_decision(projected_new_gib=p)",
                                     "ctx = gov.live_admission_decision(projected_new_gib=p)  "
                                     "# ADMISSION_RECONCILE_OK: caller reconciles under the lock"))
    assert _gate()(repo_root=tmp_path, strict=False, verbose=False) == []


def test_gate_rejects_placeholder_waivers(tmp_path):
    _write(tmp_path, "tools/throttle.py",
           _PREFIX_THROTTLE.replace("def decide(level, jobs):",
                                    "def decide(level, jobs):  # THROTTLE_REARM_OK: <rationale>"))
    assert len(_gate()(repo_root=tmp_path, strict=False, verbose=False)) == 1


def test_gate_ignores_checkpoint_resume_and_prose(tmp_path):
    """FALSE-POSITIVE guard. `resume` is everywhere in this repo (checkpoint resume); a detector
    that matched the bare word would be permanently red and therefore ignored."""
    _write(tmp_path, "tools/trainer.py",
           "def train(args):\n"
           '    """Resume from disk when pressure is normal."""\n'
           "    if args.resume and args.checkpoint.exists():\n"
           "        state = load(args.checkpoint)\n"
           '        print("resumed", state["epoch"])\n')
    _write(tmp_path, "tools/commented.py",
           "def decide(level, jobs):\n"
           "    # resume_targets and resume_free_gib and max_stop_duration in a COMMENT\n"
           '    if level == "normal":\n'
           "        return None\n")
    assert _gate()(repo_root=tmp_path, strict=False, verbose=False) == []


def test_gate_prose_alone_cannot_satisfy_the_cure(tmp_path):
    """A docstring or comment that NARRATES resume_free_gib / max_stop_duration must not count as
    wiring it. Only executable text votes."""
    _write(tmp_path, "tools/throttle.py",
           "def decide(level, jobs):\n"
           '    """We resume on resume_free_gib with a max_stop_duration_s escape hatch."""\n'
           "    # resume_free_gib max_stop_duration_s\n"
           "    paused = [j for j in jobs if j.paused]\n"
           '    if level == "normal" and paused:\n'
           "        return resume_targets(paused)\n"
           "    return None\n")
    found = _gate()(repo_root=tmp_path, strict=False, verbose=False)
    assert len(found) == 1 and "throttle.py" in found[0]


def test_gate_is_registered_with_executed_positive_controls():
    from tac import confound_gates as cg

    names = [fn.__name__ for fn in cg.CONFOUND_GATES]
    assert "check_throttle_rearms_and_admission_reconciles" in names
    mine = [c for c in cg.POSITIVE_CONTROLS
            if c.gate == "check_throttle_rearms_and_admission_reconciles"]
    assert len(mine) == 2, "one control per leg — a single control leaves the other gutable"
    # The #831 class guard EXECUTES them; if it passes, both controls still fire.
    assert cg.check_refusal_gates_have_live_positive_control(strict=False, verbose=False) == []
    cov = cg.positive_control_coverage()
    assert "check_throttle_rearms_and_admission_reconciles" in cov["covered_gates"]
    assert cov["covered"] >= cg.MIN_POSITIVE_CONTROL_COVERAGE
    assert len(cov["uncovered_gates"]) <= cg.MAX_UNCOVERED_REFUSE_GATES


def test_gate_declares_its_denominator(capsys):
    _gate()(repo_root=_REPO, strict=False, verbose=True)
    out = capsys.readouterr().out
    assert "throttle-resume function(s)" in out and "in-scope source file(s)" in out, (
        "a narrowed scope must never print a clean OK over an empty scan"
    )


def test_governor_and_blackbox_live_callsites_pass_the_rearm_inputs():
    """STRUCTURAL: the three live throttle callsites carry the re-arm inputs. This is what the gate
    checks, asserted here directly so a gate regression cannot hide a callsite regression."""
    import inspect

    for src in (inspect.getsource(mbb._govern_tick), inspect.getsource(gov.main)):
        assert "resume_free_gib" in src and "max_stop_duration" in src
    decide = inspect.getsource(gov.decide_governor_action)
    assert "ESCAPE HATCH" in decide and "resume_free_gib" in decide
