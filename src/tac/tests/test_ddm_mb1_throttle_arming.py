# SPDX-License-Identifier: MIT
"""ddm_mb1 (2026-08-16) — the SIGSTOP throttle ACTUATOR is default-OFF and durably armed.

WHY THIS EXISTS. On 2026-08-15 ``tools/memory_blackbox.py --daemon`` SIGSTOPped three live
measurements (the mp2 differential eval, the wc1 decode, the wd3 warm train) plus the dashboard and
three safe_run wrappers, and never resumed them, on a box with 40.5 GiB free. ddm_gb1 cured the
MECHANISM — wrong-object cp measurement, no re-arm, no exit-resume. It did NOT cure the ARMING:

  * ``run_daemon`` still defaulted ``govern=True``; and
  * ``ensure_blackbox_running`` built ``[... "memory_blackbox.py", "--daemon"]`` with no opt-out.

So "the daemon is OFF pending adjudication" was enforced only by nobody having launched training
yet. The next ``spawn_durable_daemon`` launch would have silently restarted the un-adjudicated
SIGSTOP actuator. These tests pin the cure on all four surfaces it touches:

  A. the shared arming resolver (ONE helper, no parallel twin with ``admission_enforcing``);
  B. ``run_daemon`` / CLI tri-state (None = defer to arming, explicit = per-invocation override);
  C. the exit/startup SIGCONT sweeps, now UNCONDITIONAL — a recorder-only daemon (the new default)
     must still rescue what an armed predecessor stranded. Includes the REAL-PROCESS positive
     control: a live child stops itself, and a DISARMED daemon resumes it;
  D. the escape hatch's TYPED alarm, and the Layer-2 backstop coupling in admission.
"""
from __future__ import annotations

import inspect
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import memory_blackbox as mbb  # noqa: E402
import system_memory_governor as gov  # noqa: E402

_NOWHERE = Path("/nonexistent/ddm_mb1/governor_throttle_arm.flag")


# ── A. the shared arming resolver ───────────────────────────────────────────────────────────────
def test_arming_defaults_off_and_says_how_to_arm():
    """Default OFF, and the detail SURFACES the reason. An unexplained 'off' is how a disabled
    protection layer becomes a forgotten one (CLAUDE.md: "'Off' is a tracked queue")."""
    a = gov.resolve_arming("TAC_MB1_UNSET", _NOWHERE, env={})
    assert a.armed is False
    assert a.source == "default"
    assert "TAC_MB1_UNSET" in a.detail and "OFF" in a.detail


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
def test_arming_env_truthy_values_arm(raw):
    a = gov.resolve_arming("TAC_MB1_ARM", _NOWHERE, env={"TAC_MB1_ARM": raw})
    assert a.armed is True and a.source == "env"


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "maybe", "2"])
def test_arming_env_non_truthy_values_do_not_arm(raw):
    """Anything not explicitly truthy leaves the actuator OFF — fail-safe, never fail-armed."""
    assert gov.resolve_arming("TAC_MB1_ARM", _NOWHERE, env={"TAC_MB1_ARM": raw}).armed is False


def test_arming_durable_flag_file_arms_and_survives_env_reset(tmp_path):
    """The flag is the DURABLE leg: the shell env resets between Bash calls, subagents get a fresh
    environ, but the file persists — so an operator arming decision is not lost on the next call."""
    flag = tmp_path / "governor_throttle_arm.flag"
    flag.write_text("1\n")
    a = gov.resolve_arming("TAC_MB1_ARM", flag, env={})
    assert a.armed is True and a.source == "flag" and str(flag) in a.detail


def test_arming_flag_present_but_not_truthy_explains_itself(tmp_path):
    """A present-but-off flag must not read like an absent one: the operator wrote something, and
    the log has to say why it did not take."""
    flag = tmp_path / "governor_throttle_arm.flag"
    flag.write_text("off\n")
    a = gov.resolve_arming("TAC_MB1_ARM", flag, env={})
    assert a.armed is False
    assert "present but first line" in a.detail and "off" in a.detail


def test_arming_unreadable_flag_fails_safe(tmp_path):
    """An unreadable flag (a DIRECTORY at the flag path) must resolve to NOT armed, never raise and
    never arm. The actuator's failure direction is off."""
    flag = tmp_path / "governor_throttle_arm.flag"
    flag.mkdir()
    a = gov.resolve_arming("TAC_MB1_ARM", flag, env={})
    assert a.armed is False


def test_admission_enforcing_reuses_the_shared_resolver_no_parallel_twin():
    """ONE arming helper, two callers. A hand-rolled second copy of the env-or-flag dance is how the
    two surfaces would silently diverge on truthy-parsing or precedence."""
    src = inspect.getsource(gov.admission_enforcing)
    assert "resolve_arming(" in src
    assert '"1", "true"' not in src, "admission_enforcing re-implemented the truthy set"
    assert "resolve_arming(" in inspect.getsource(gov.throttle_arming)


def test_throttle_armed_agrees_with_throttle_arming(monkeypatch):
    monkeypatch.setenv(gov.THROTTLE_ARM_ENV, "1")
    assert gov.throttle_armed() is True and gov.throttle_arming().armed is True
    monkeypatch.setenv(gov.THROTTLE_ARM_ENV, "0")
    # (The durable flag may still arm it on a real machine; compare the two surfaces, not a literal.)
    assert gov.throttle_armed() == gov.throttle_arming().armed


# ── B. run_daemon / CLI tri-state ───────────────────────────────────────────────────────────────
@pytest.fixture
def hermetic_daemon(tmp_path, monkeypatch):
    """Point every durable path at tmp and neuter the process-global exit hooks, so a daemon tick in
    this test can touch nothing real."""
    monkeypatch.setattr(mbb, "_STOPPED_LEDGER", tmp_path / "stopped.json")
    monkeypatch.setattr(mbb, "_STOPPED_LEDGER_LOCK", tmp_path / "stopped.lock")
    monkeypatch.setattr(mbb, "_ACTION_LOG", tmp_path / "actions.log")
    monkeypatch.setattr(mbb, "_BLACKBOX", tmp_path / "bb.jsonl")
    monkeypatch.setattr(mbb, "_BLACKBOX_LOCK", tmp_path / "bb.jsonl.lock")
    monkeypatch.setattr(mbb, "_SINGLETON_LOCK", tmp_path / "singleton.lock")
    monkeypatch.setattr(mbb, "_ARCHIVE", tmp_path / "archive")
    # Exit hooks: capture instead of installing into the real interpreter.
    monkeypatch.setattr(mbb, "_EXIT_SWEEP_INSTALLED", False)
    registered: list = []
    installed: dict = {}
    monkeypatch.setattr(mbb.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(mbb.signal, "signal", lambda s, h: installed.setdefault(s, h))
    return {"registered": registered, "installed": installed, "tmp": tmp_path}


def _arm(monkeypatch, armed: bool) -> None:
    monkeypatch.setattr(
        gov, "throttle_arming",
        lambda **kw: gov.Arming(armed, "flag" if armed else "default", "test-injected"))
    monkeypatch.setattr(gov, "throttle_armed", lambda **kw: armed)


def _ticks(monkeypatch) -> list:
    seen: list = []

    def _fake_tick(cw, cc, **kw):
        seen.append((cw, cc))
        return cw, cc, None

    monkeypatch.setattr(mbb, "_govern_tick", _fake_tick)
    return seen


def test_run_daemon_disarmed_never_actuates(hermetic_daemon, monkeypatch):
    """THE REGRESSION. Pre-fix this daemon governed by default; the auto-start relied on that."""
    _arm(monkeypatch, False)
    seen = _ticks(monkeypatch)
    assert mbb.run_daemon(max_iterations=1, interval=0.0, fast_interval=0.0, band=False) == 0
    assert seen == [], "a DISARMED daemon must never run the SIGSTOP throttle"


def test_run_daemon_armed_actuates(hermetic_daemon, monkeypatch):
    """Armed is still fully functional — this is a tracked default, not a deletion."""
    _arm(monkeypatch, True)
    seen = _ticks(monkeypatch)
    assert mbb.run_daemon(max_iterations=1, interval=0.0, fast_interval=0.0, band=False) == 0
    assert seen == [(0, 0)]


def test_explicit_govern_overrides_the_arming_surface(hermetic_daemon, monkeypatch):
    """An explicit caller argument is a per-invocation override in BOTH directions."""
    _arm(monkeypatch, False)
    seen = _ticks(monkeypatch)
    mbb.run_daemon(max_iterations=1, interval=0.0, fast_interval=0.0, band=False, govern=True)
    assert seen == [(0, 0)], "explicit govern=True must beat a disarmed surface"

    _arm(monkeypatch, True)
    seen2 = _ticks(monkeypatch)
    mbb.run_daemon(max_iterations=1, interval=0.0, fast_interval=0.0, band=False, govern=False)
    assert seen2 == [], "explicit govern=False must beat an armed surface"


def test_actuator_state_is_logged_with_its_reason(hermetic_daemon, monkeypatch):
    """'Off' is only a TRACKED state if every start says so, and says why."""
    _arm(monkeypatch, False)
    _ticks(monkeypatch)
    mbb.run_daemon(max_iterations=1, interval=0.0, fast_interval=0.0, band=False)
    log = (hermetic_daemon["tmp"] / "actions.log").read_text()
    assert "THROTTLE ACTUATOR DISARMED" in log
    assert "safe_run" in log, "the disarmed line must name what carries the protection instead"


def test_explicit_override_is_not_reported_as_an_arming_decision(hermetic_daemon, monkeypatch):
    """A forced-on test daemon must not look DURABLY ARMED in the action log."""
    _arm(monkeypatch, False)
    _ticks(monkeypatch)
    mbb.run_daemon(max_iterations=1, interval=0.0, fast_interval=0.0, band=False, govern=True)
    log = (hermetic_daemon["tmp"] / "actions.log").read_text()
    assert "source=explicit" in log


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--daemon"], None),                      # defer to the arming surface (default OFF)
        (["--daemon", "--govern"], True),          # explicit force on
        (["--daemon", "--no-govern"], False),      # explicit force off
        (["--daemon", "--govern", "--no-govern"], False),  # contradiction -> the SAFE one wins
    ],
)
def test_cli_govern_tri_state(monkeypatch, argv, expected):
    captured: dict = {}
    monkeypatch.setattr(mbb, "run_daemon", lambda **kw: captured.update(kw) or 0)
    assert mbb.main(argv) == 0
    assert captured["govern"] is expected


def test_autostart_argv_does_not_force_the_actuator():
    """The exact path that made 'the daemon is OFF' untrue in practice."""
    src = inspect.getsource(mbb.ensure_blackbox_running)
    assert '"--daemon"' in src
    assert '"--govern"' not in src, "auto-start must not FORCE the SIGSTOP actuator on"
    assert '"--no-govern"' not in src, "auto-start must still honour an operator-armed flag"


# ── C. exit/startup sweeps are UNCONDITIONAL ────────────────────────────────────────────────────
def test_exit_handlers_installed_even_when_disarmed(hermetic_daemon, monkeypatch):
    """Pre-ddm_mb1 these lived behind ``if govern:``. Once the actuator became default-OFF, that
    gating meant the COMMON case could see a stranded ledger and walk past it."""
    _arm(monkeypatch, False)
    _ticks(monkeypatch)
    mbb.run_daemon(max_iterations=1, interval=0.0, fast_interval=0.0, band=False)
    assert len(hermetic_daemon["registered"]) == 1, "atexit sweep must install when disarmed"
    assert set(hermetic_daemon["installed"]) == {signal.SIGTERM, signal.SIGINT}


def test_run_daemon_sweeps_unconditionally_source_check():
    """Structural backstop: neither sweep may drift back behind the govern gate."""
    src = inspect.getsource(mbb.run_daemon)
    assert "if govern:\n        install_exit_resume_handlers()" not in src


def test_positive_control_disarmed_daemon_resumes_a_real_stranded_child(hermetic_daemon, monkeypatch):
    """POSITIVE CONTROL — the whole incident, in miniature, with a REAL process.

    A live child SIGSTOPs ITSELF (nothing pre-existing is ever signalled), the stopped-set ledger
    records it exactly as an armed predecessor would have, and then a RECORDER-ONLY (disarmed)
    daemon starts. It must SIGCONT the child. This is the stop -> condition-clears -> resume loop
    end to end; if the sweep is ever re-gated on ``govern`` this test goes red instead of a real job
    silently sitting in state 'T'. The child is killed in ``finally``.
    """
    _arm(monkeypatch, False)
    _ticks(monkeypatch)
    code = "import os, signal, time\nos.kill(os.getpid(), signal.SIGSTOP)\ntime.sleep(30)\n"
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if gov.is_paused_state(gov._process_state(proc.pid)):
                break
            time.sleep(0.05)
        assert gov.is_paused_state(gov._process_state(proc.pid)), "child did not self-stop"

        # Exactly what an armed predecessor would have written before it was SIGKILLed.
        mbb.record_stopped_pids([proc.pid], label="ddm_mb1_stranded_child")
        assert proc.pid in mbb.paused_since_ts()

        mbb.run_daemon(max_iterations=1, interval=0.0, fast_interval=0.0, band=False)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if not gov.is_paused_state(gov._process_state(proc.pid)):
                break
            time.sleep(0.05)
        assert not gov.is_paused_state(gov._process_state(proc.pid)), (
            "a recorder-only daemon must still SIGCONT what an armed predecessor stranded")
        assert mbb.paused_since_ts() == {}, "the ledger must be drained after the sweep"
    finally:
        proc.kill()
        proc.wait(timeout=10)


# ── D. the escape hatch's TYPED alarm + the Layer-2 backstop coupling ───────────────────────────
def _snapshot(available_gib: float) -> gov.SystemMemorySnapshot:
    return gov.SystemMemorySnapshot(
        total_gib=128.0, available_gib=available_gib, used_gib=128.0 - available_gib,
        free_gib=available_gib, wired_gib=8.0, compressor_gib=4.0, swap_used_gib=0.0,
        pressure_level=1, load1=1.0, load5=1.0, load15=1.0)


def _paused_job() -> gov.TrackedJob:
    return gov.TrackedJob(
        label="mp2_eval", pid=4242, pgid=4242, cmd="python tools/levelset_byte_close_and_eval.py",
        priority=50, projected_peak_gib=8.0, current_rss_gib=6.0, paused=True,
        throttle_eligible=True, own_group_leader=True)


def _drive_govern_tick(monkeypatch, *, available_gib: float, stopped_age_s: float):
    """Drive the REAL policy (no faked GovernorAction) with an injected snapshot + paused job."""
    job = _paused_job()
    monkeypatch.setattr(gov, "read_system_memory_snapshot", lambda: _snapshot(available_gib))
    monkeypatch.setattr(gov, "list_tracked_jobs", lambda **kw: [job])
    monkeypatch.setattr(gov, "resume_job", lambda j, **kw: True)
    monkeypatch.setattr(mbb, "_adopt_unrecorded_paused",
                        lambda jobs, **kw: {job.pid: time.time() - stopped_age_s})
    monkeypatch.setattr(mbb, "_stopped_scope_pids", lambda j: [j.pid])
    monkeypatch.setattr(mbb, "record_resumed", lambda **kw: None)
    logged: list[str] = []
    monkeypatch.setattr(mbb, "_log_action", lambda msg: logged.append(msg))
    _, _, rec = mbb._govern_tick(0, 0, max_stop_duration_s=300.0)
    return rec, logged


def test_escape_hatch_resume_emits_a_typed_alarm(monkeypatch):
    """A stop held past the ceiling means the re-arm references THEMSELVES failed. That is not a
    routine RESUME line — the 2026-08-15 freeze was silent for 75+ minutes precisely because
    nothing in the log said 'this is anomalous'."""
    rec, logged = _drive_govern_tick(monkeypatch, available_gib=5.0, stopped_age_s=9999.0)
    assert rec is not None and rec["action"] == "resume"
    assert rec["escape_hatch"] is True
    assert rec["alarm"] == "throttle_escape_hatch"
    assert any("ALARM confound_alarm=throttle_escape_hatch" in m for m in logged)


def test_ordinary_resume_does_not_emit_the_alarm(monkeypatch):
    """The alarm must stay meaningful: a normal recovery resume is NOT an alarm."""
    rec, logged = _drive_govern_tick(monkeypatch, available_gib=60.0, stopped_age_s=1.0)
    assert rec is not None and rec["action"] == "resume"
    assert rec["escape_hatch"] is False and "alarm" not in rec
    assert not any("ALARM confound_alarm" in m for m in logged)


@pytest.mark.skipif(gov._mg is None, reason="memory_guard unavailable (fail-safe: no tracked jobs)")
def test_disarmed_throttle_withdraws_the_admission_relaxation(tmp_path):
    """THE COUPLING ddm_mb1 had to repair, not paper over.

    Layer-3 admission may charge a relaxed measured-growth number ONLY for the class Layer 2 can
    pause (``list_tracked_jobs`` comment, pre-existing). Structural eligibility is necessary but not
    sufficient: a throttle-eligible job has no Layer-2 backstop when the ACTUATOR IS DISARMED. So
    disarming must make admission STRICTER (fall back to the conservative +25 GiB), never wider.
    Without this leg, ddm_mb1's default-OFF change would have silently widened admission by removing
    the very backstop that licensed the relaxation.
    """
    import memory_guard as mg

    history = tmp_path / "rss_history.json"
    sample = mg.ProcessSample(
        pid=710, ppid=1, pgid=710, rss_kb=round(4.4 * 1024**2),
        command="python tools/verdict_mem_microprobe.py --n 600",
        # Stable process identity: the RSS history keys on it, so without it the two observations
        # below are not recognised as the SAME process and no slope is ever formed.
        start_identity="Tue Jul 14 00:00:01 2026")
    kw = {"samples": {710: sample}, "registry_rows": [], "self_pid": 1,
          "rss_history_path": history}
    # Two observations 60 s apart => a fresh, flat (plateau) slope the relaxation can use.
    gov.list_tracked_jobs(**kw, rss_history_now_ts=100.0, layer2_backstop_armed=True)
    armed = gov.list_tracked_jobs(**kw, rss_history_now_ts=160.0, layer2_backstop_armed=True)
    disarmed = gov.list_tracked_jobs(**kw, rss_history_now_ts=160.0, layer2_backstop_armed=False)

    armed_headroom = gov.sum_active_growth_headroom_gib(armed)
    disarmed_headroom = gov.sum_active_growth_headroom_gib(disarmed)
    assert armed_headroom == pytest.approx(gov.MATERIAL_PLATEAU_GROWTH_RESERVE_GIB)
    assert disarmed_headroom == pytest.approx(gov.UNKNOWN_GROWTH_HEADROOM_GIB)
    assert disarmed_headroom > armed_headroom, "disarming must be STRICTER, never wider"


def test_governor_cli_apply_records_its_pause_in_the_stopped_ledger():
    """THE SECOND SIGSTOP ENTRY POINT (found in ddm_mb1 review pass 2).

    ``system_memory_governor.py --governor-tick --apply`` actuates the same throttle as the daemon.
    Pre-fix it called ``pause_job`` and recorded NOTHING, so a CLI-initiated stop was invisible to
    the exit sweep, to ``memory_blackbox.py --resume-stopped``, and to the escape hatch (which ages
    pids by their ledger timestamp). That is the incident's own "no exit-resume" defect reached
    through the CLI — and it strands a job exactly when no daemon is running to adopt it.
    """
    src = inspect.getsource(gov.main)
    record_at = src.find("record_stopped_pids")
    signal_at = src.find("pause_job(action.target)")
    assert record_at != -1, "the CLI --apply pause must record to the stopped-set ledger"
    assert signal_at != -1
    assert record_at < signal_at, (
        "record BEFORE signalling — a crash between the two must leave a SWEEPABLE ledger, never a "
        "stranded pid")
    assert "record_resumed" in src, "the CLI --apply resume must drain the ledger"


def test_escape_hatch_token_has_one_source_of_truth():
    """The alarm classifies the hatch by this token. If the governor reworded its reason string
    independently, the TYPED alarm would silently stop firing and an escape-hatch resume would look
    routine again — the exact silence the incident ran in."""
    # Assert the SHARED IDENTIFIER is referenced on both sides, not the literal value: a hardcoded
    # copy of the string on either side is exactly the silent-decouple this constant removes.
    assert "ESCAPE_HATCH_REASON_TOKEN" in inspect.getsource(gov.decide_governor_action)
    assert "ESCAPE_HATCH_REASON_TOKEN" in inspect.getsource(mbb._govern_tick)
    assert gov.ESCAPE_HATCH_REASON_TOKEN == "THROTTLE ESCAPE HATCH"


def test_list_tracked_jobs_defaults_to_the_live_arming_surface():
    src = inspect.getsource(gov.list_tracked_jobs)
    assert "throttle_armed()" in src
    assert "if (eligible and layer2_armed) else None" in src


# ── E. the gate (Leg C) refuses re-introduction ─────────────────────────────────────────────────
def _gate():
    from tac.confound_gates import check_throttle_rearms_and_admission_reconciles
    return check_throttle_rearms_and_admission_reconciles


def _run_gate_on(tmp_path: Path, relative: str, text: str) -> list[str]:
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return _gate()(repo_root=tmp_path, strict=False, verbose=False)


def test_gate_c1_flags_a_spawn_that_forces_the_actuator_on(tmp_path):
    found = _run_gate_on(tmp_path, "tools/spawner.py",
                         'def go():\n'
                         '    argv = ["python", "tools/memory_blackbox.py", "--daemon", "--govern"]\n'
                         '    return spawn(argv)\n')
    assert any("spawner.py" in f and "--govern" in f for f in found)


def test_gate_c1_clean_without_the_force_flag(tmp_path):
    found = _run_gate_on(tmp_path, "tools/spawner.py",
                         'def go():\n'
                         '    argv = ["python", "tools/memory_blackbox.py", "--daemon"]\n'
                         '    return spawn(argv)\n')
    assert not any("spawner.py" in f for f in found)


def test_gate_c2_flags_hardcoded_govern_true(tmp_path):
    found = _run_gate_on(tmp_path, "tools/d.py",
                         "def run_daemon(*, govern=True):\n"
                         "    if govern:\n"
                         "        pause_job(t)\n")
    assert any("d.py" in f and "hardcoded default" in f for f in found)


def test_gate_c2_flags_unresolved_none_default(tmp_path):
    """None without a resolve is not a tracked default-OFF — it is an actuator nobody can arm."""
    found = _run_gate_on(tmp_path, "tools/d.py",
                         "def run_daemon(*, govern=None):\n"
                         "    if govern:\n"
                         "        pause_job(t)\n")
    assert any("d.py" in f and "never resolves it" in f for f in found)


def test_gate_c2_clean_when_none_resolves_through_the_arming_surface(tmp_path):
    found = _run_gate_on(tmp_path, "tools/d.py",
                         "def run_daemon(*, govern=None):\n"
                         "    if govern is None:\n"
                         "        govern = throttle_arming().armed\n"
                         "    if govern:\n"
                         "        pause_job(t)\n")
    assert not any("d.py" in f for f in found)


def test_gate_c2_clean_for_explicit_false_default(tmp_path):
    found = _run_gate_on(tmp_path, "tools/d.py",
                         "def run_daemon(*, govern=False):\n"
                         "    if govern:\n"
                         "        pause_job(t)\n")
    assert not any("d.py" in f for f in found)


def test_gate_c_waiver_is_respected(tmp_path):
    found = _run_gate_on(tmp_path, "tools/d.py",
                         "def run_daemon(*, govern=True):\n"
                         "    # THROTTLE_ARM_OK: hermetic replay harness, never spawns a real daemon\n"
                         "    if govern:\n"
                         "        pause_job(t)\n")
    assert not any("d.py" in f for f in found)


def test_gate_is_clean_on_the_live_repo():
    """Live count 0 — the landing that adds the leg also closes it (strict-flip atomicity)."""
    assert _gate()(strict=False, verbose=False) == []
