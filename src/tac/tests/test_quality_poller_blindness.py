"""Controls for the #1064 quality-poller BLINDNESS cure (both directions).

The defect ddm_lw2 measured: ``_pid_alive`` collapsed three states into one
bool, so a MISSING/garbage ``pid_file`` (the poller is blind) read identically
to a child that exited (genuine death, owned by the liveness watcher).  The
poller then returned 0 silently and wrote no receipt --- ``ddm_ra2c_rank4``'s
quality poller watched nothing for an entire run and looked like health.

The bar these controls must clear is the one the PRIOR #1064 fix failed: that
fix had passing tests and a confident docstring and was INERT.  So every test
below asserts an OBSERVABLE effect (return code, alert file written, reason
string), never merely that a function exists.

Both directions are pinned:
  * blindness (missing pidfile / absent log) past grace  -> rc=1 + alert
  * GENUINE death (readable pidfile, dead pid)           -> rc=0 + NO alert
The second is the anti-overfire control: curing blindness must not convert
ordinary child exit into a quality alert, which the liveness watcher owns.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:  # the poller is a tools/ script, not a package
    sys.path.insert(0, str(TOOLS))

run_quality_poller = pytest.importorskip("run_quality_poller")


def _dead_pid() -> int:
    """A pid that is readable-but-gone: fork a child and reap it."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def _config(tmp_path: Path, *, pid_file: Path, log_path: Path, grace: float = 0.0):
    """Build the config through the CANONICAL loader, never by hand.

    Hand-typing this dict is the same genus the cure addresses: an invented
    config drifts from the real contract (my first draft omitted
    ``phase_field`` and every poll raised KeyError).  ``load_config`` is the
    one surface that knows the true key set, so the fixture goes through it.
    """
    raw = {
        "schema": run_quality_poller.CONFIG_SCHEMA,
        "log_path": str(log_path),
        "pid_file": str(pid_file),
        "telemetry_path": str(tmp_path / "telemetry.jsonl"),
        "alert_path": str(tmp_path / "alert.json"),
        "poll_s": 0.01,
        "eval_period_s": 1.0,
        "stale_periods": 1000.0,
        "startup_grace_s": grace,
        "json_marker": "EVAL",
        "fields": {"epoch": "epoch", "value": "value", "phase": "phase", "finite": []},
        "bar_value": 1.0,
        "bar_start_epoch": 0,
        "phase_knee": {"epoch": 10**9, "continuous_phase": "cont"},
        "best_not_latest": {"phase": "cont", "lag_epochs": 10**9},
        # One condition must be enabled for the loader to accept the config;
        # stale_telemetry cannot fire here because stale_periods is huge.
        "alert_conditions": {"stale_telemetry": True},
    }
    cfg_path = tmp_path / "quality_config.json"
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")
    return run_quality_poller.load_config(cfg_path)


# --- tri-state pid read: the collapse that WAS the defect ---------------------


def test_pid_state_separates_unreadable_from_dead(tmp_path: Path) -> None:
    """The three states must be distinguishable; the bool could not tell them apart."""
    missing = tmp_path / "absent.pid"
    garbage = tmp_path / "garbage.pid"
    garbage.write_text("not-a-pid", encoding="utf-8")
    zero = tmp_path / "zero.pid"
    zero.write_text("0", encoding="utf-8")
    dead = tmp_path / "dead.pid"
    dead.write_text(str(_dead_pid()), encoding="utf-8")
    live = tmp_path / "live.pid"
    live.write_text(str(os.getpid()), encoding="utf-8")

    assert run_quality_poller._pid_state(missing) == run_quality_poller.PID_UNREADABLE
    assert run_quality_poller._pid_state(garbage) == run_quality_poller.PID_UNREADABLE
    assert run_quality_poller._pid_state(zero) == run_quality_poller.PID_UNREADABLE
    assert run_quality_poller._pid_state(dead) == run_quality_poller.PID_DEAD
    assert run_quality_poller._pid_state(live) == run_quality_poller.PID_ALIVE

    # The legacy bool still behaves, and still cannot separate the two failures.
    assert run_quality_poller._pid_alive(live) is True
    assert run_quality_poller._pid_alive(dead) is False
    assert run_quality_poller._pid_alive(missing) is False


# --- direction 1: blindness is LOUD -------------------------------------------


def test_missing_pidfile_past_grace_alerts_rc1(tmp_path: Path) -> None:
    """rank4's exact fault: pid_file drifted to a path that does not exist."""
    log = tmp_path / "run.log"
    log.write_text("", encoding="utf-8")
    cfg = _config(tmp_path, pid_file=tmp_path / "gone.pid", log_path=log)

    rc = run_quality_poller.run(cfg, once=True)

    assert rc == 1, "a blind poller must not return success"
    payload = json.loads(cfg["alert_path"].read_text(encoding="utf-8"))
    assert payload["reason"] == "watcher_blind_pid_file_unreadable"
    assert payload["pid_file"] == str(cfg["pid_file"])


def test_absent_log_past_grace_alerts_rc1(tmp_path: Path) -> None:
    """rank4's second drift: log_path aimed at a stdout.log that never existed."""
    pid = tmp_path / "live.pid"
    pid.write_text(str(os.getpid()), encoding="utf-8")
    cfg = _config(tmp_path, pid_file=pid, log_path=tmp_path / "never_written.log")

    rc = run_quality_poller.run(cfg, once=True)

    assert rc == 1
    payload = json.loads(cfg["alert_path"].read_text(encoding="utf-8"))
    assert payload["reason"] == "watcher_blind_log_absent"


# --- direction 2: the anti-overfire controls ----------------------------------


def test_genuine_child_death_stays_rc0_with_no_alert(tmp_path: Path) -> None:
    """Process death is the LIVENESS watcher's job -- curing blindness must not steal it."""
    pid = tmp_path / "dead.pid"
    pid.write_text(str(_dead_pid()), encoding="utf-8")
    log = tmp_path / "run.log"
    log.write_text("", encoding="utf-8")
    cfg = _config(tmp_path, pid_file=pid, log_path=log)

    rc = run_quality_poller.run(cfg, once=False)

    assert rc == 0, "a cleanly-exited child is not a quality alert"
    assert not cfg["alert_path"].exists(), "no alert may be written for ordinary death"


def test_healthy_child_and_log_produce_no_alert(tmp_path: Path) -> None:
    pid = tmp_path / "live.pid"
    pid.write_text(str(os.getpid()), encoding="utf-8")
    log = tmp_path / "run.log"
    log.write_text("", encoding="utf-8")
    cfg = _config(tmp_path, pid_file=pid, log_path=log)

    rc = run_quality_poller.run(cfg, once=True)

    assert rc == 0
    assert not cfg["alert_path"].exists()


def test_within_startup_grace_blindness_is_silent(tmp_path: Path) -> None:
    """A launcher legitimately has not written pidfile/log yet -- do not fire early."""
    cfg = _config(
        tmp_path,
        pid_file=tmp_path / "not_yet.pid",
        log_path=tmp_path / "not_yet.log",
        grace=600.0,
    )

    assert (
        run_quality_poller.blindness_alert(
            cfg, now=1000.0, started_at=999.0
        )
        is None
    ), "blindness inside the grace window must stay silent"

    assert (
        run_quality_poller.blindness_alert(cfg, now=2000.0, started_at=999.0)
        is not None
    ), "blindness must become loud once the grace has elapsed"


def test_blindness_is_not_gated_by_quality_conditions(tmp_path: Path) -> None:
    """An operator may disable a quality condition; 'am I watching anything' is not tunable."""
    cfg = _config(tmp_path, pid_file=tmp_path / "gone.pid", log_path=tmp_path / "x.log")
    cfg["conditions"] = {}  # every quality condition off

    alert = run_quality_poller.blindness_alert(cfg, now=10.0, started_at=0.0)

    assert alert is not None
    assert alert["reason"] == "watcher_blind_pid_file_unreadable"
