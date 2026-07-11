"""Guard tests for tools/costate_observer_loop.ensure_run_log (the 2026-07-10
observer-gap self-match bug).

Bug: when the trainer logs to daemon.log (not run.log) and the observer's OWN
durable-daemon registry row is present, an un-guarded scan self-matched — its cmd
carries ``--run-dir <abs>`` and its log is observer.log — so run.log got symlinked
to observer.log (no ``loss_terms`` telemetry -> empirical costates degenerate to
n=0). Compounded by the trainer row storing a RELATIVE cmd path that an
absolute-substring test misses, leaving the observer row as the only (wrong) match.

Fix: skip observer rows, match on run-dir basename too, never self-symlink to
observer.log.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_CO_PATH = Path(__file__).resolve().parents[3] / "tools" / "costate_observer_loop.py"
_spec = importlib.util.spec_from_file_location("tac_costate_observer_loop_under_test", _CO_PATH)
co = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["tac_costate_observer_loop_under_test"] = co
_spec.loader.exec_module(co)


def _setup(tmp_path, monkeypatch, rows):
    reg = tmp_path / "durable_daemons.json"
    reg.write_text(json.dumps(rows))
    monkeypatch.setattr(co, "_REGISTRY", reg)
    monkeypatch.setattr(co, "_REPO", tmp_path)
    return reg


def test_picks_trainer_daemon_log_not_observer_self_log(tmp_path, monkeypatch):
    run = tmp_path / "experiments" / "results" / "levelset_v752_baseline_X"
    run.mkdir(parents=True)
    daemon_log = run / "daemon.log"
    daemon_log.write_text('{"stage": "loss_terms", "ep": 50}\n')
    observer_log = run / "observer.log"
    observer_log.write_text("[costate-observer] observing...\n")
    # Trainer row: RELATIVE cmd path (absolute-substring test would miss it) + daemon.log.
    # Observer row FIRST in registry order: absolute --run-dir cmd + observer.log (the trap).
    _setup(tmp_path, monkeypatch, [
        {"label": "costate_obs_levelset_v752_baseline_X", "status": "running",
         "cmd": [sys.executable, "tools/costate_observer_loop.py", "--run-dir", str(run.resolve())],
         "log": str(observer_log)},
        {"label": "v752_baseline_X", "status": "running",
         "cmd": ["bash", "experiments/results/levelset_v752_baseline_X/launch.sh"],
         "log": str(daemon_log)},
    ])
    note = co.ensure_run_log(run.resolve())
    link = run.resolve() / "run.log"
    assert link.is_symlink(), "run.log should have been created"
    assert link.resolve() == daemon_log.resolve(), (
        f"run.log must point at the trainer daemon.log, not observer.log; got {link.resolve()}"
    )
    assert note and "daemon.log" in note


def test_noop_when_run_log_already_present(tmp_path, monkeypatch):
    run = tmp_path / "experiments" / "results" / "r"
    run.mkdir(parents=True)
    (run / "run.log").write_text("existing\n")
    _setup(tmp_path, monkeypatch, [])
    assert co.ensure_run_log(run.resolve()) is None


def test_no_match_when_only_observer_row_exists(tmp_path, monkeypatch):
    # If the ONLY running row is the observer itself, we must NOT create a symlink
    # (there is no trainer log to point at) rather than self-symlink to observer.log.
    run = tmp_path / "experiments" / "results" / "r2"
    run.mkdir(parents=True)
    (run / "observer.log").write_text("obs\n")
    _setup(tmp_path, monkeypatch, [
        {"label": "costate_obs_r2", "status": "running",
         "cmd": [sys.executable, "tools/costate_observer_loop.py", "--run-dir", str(run.resolve())],
         "log": str(run / "observer.log")},
    ])
    assert co.ensure_run_log(run.resolve()) is None
    assert not (run.resolve() / "run.log").exists()
