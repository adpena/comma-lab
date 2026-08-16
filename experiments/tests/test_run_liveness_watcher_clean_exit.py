#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tests for the #1064 clean-exit fix in tools/run_liveness_watcher.py.

Bug class: a child that exits rc=0 with a clean done receipt used to raise a
``child_dead`` alert.  The fix adds an optional ``success_receipts`` config
family checked only at child death: a clean receipt means silent success, a
dirty or absent receipt still alerts.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "run_liveness_watcher", _REPO / "tools" / "run_liveness_watcher.py"
)
watcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(watcher)


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _base_config(tmp_path: Path, **overrides) -> dict:
    heartbeat = tmp_path / "heartbeat.log"
    heartbeat.write_text("x\n", encoding="utf-8")
    raw = {
        "schema": watcher.CONFIG_SCHEMA,
        "pid_file": str(tmp_path / "run.pid"),
        "alert_path": str(tmp_path / "alert.json"),
        "poll_s": 1,
        "warmup_s": 0,
        "artifact_checks": [
            {"label": "hb", "path": str(heartbeat), "max_age_s": 3600, "grace_s": 0}
        ],
    }
    raw.update(overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    return watcher.load_config(config_path)


class TestSummaryIsSuccess:
    def test_rc_zero_is_success(self):
        assert watcher._summary_is_success({"rc": 0})

    def test_rc_nonzero_is_failure(self):
        assert not watcher._summary_is_success({"rc": 1})

    def test_exit_zero_is_success(self):
        assert watcher._summary_is_success({"exit": 0})

    def test_status_completed_is_success(self):
        assert watcher._summary_is_success({"status": "completed"})

    def test_status_failed_is_failure(self):
        assert not watcher._summary_is_success({"status": "failed"})

    def test_status_killed_beats_rc_zero(self):
        assert not watcher._summary_is_success({"rc": 0, "status": "killed"})

    def test_empty_summary_is_not_success(self):
        assert not watcher._summary_is_success({})

    def test_bool_rc_is_not_numeric_evidence(self):
        # True == 1 in Python; a boolean must not be read as an exit code.
        assert not watcher._summary_is_success({"rc": True})


class TestConfigParsing:
    def test_success_receipts_absent_defaults_empty(self, tmp_path):
        config = _base_config(tmp_path)
        assert config["success_receipts"] == []
        assert config["success_settle_s"] == 30.0

    def test_success_receipts_parsed(self, tmp_path):
        config = _base_config(
            tmp_path,
            success_receipts=[{"label": "done", "path": str(tmp_path / "done.json")}],
            success_settle_s=0,
        )
        assert len(config["success_receipts"]) == 1
        assert config["success_receipts"][0]["label"] == "done"
        assert config["success_settle_s"] == 0.0

    def test_invalid_success_receipt_row_refused(self, tmp_path):
        with pytest.raises(watcher.ConfigError):
            _base_config(tmp_path, success_receipts=["not-an-object"])


class TestEvaluateChildDeath:
    def _write_pid(self, config, pid: int) -> None:
        config["pid_file"].write_text(f"{pid}\n", encoding="utf-8")

    def test_clean_receipt_yields_clean_exit(self, tmp_path):
        done = tmp_path / "done.json"
        done.write_text(json.dumps({"rc": 0, "status": "completed"}), encoding="utf-8")
        config = _base_config(
            tmp_path,
            success_receipts=[{"label": "done", "path": str(done)}],
            success_settle_s=0,
        )
        self._write_pid(config, _dead_pid())
        alert = watcher.evaluate(config, started_at=time.time(), now=time.time())
        assert alert is not None
        assert alert["reason"] == "clean_exit"
        assert alert["success_receipt"] == "done"

    def test_failure_receipt_still_alerts_child_dead(self, tmp_path):
        done = tmp_path / "done.json"
        done.write_text(json.dumps({"rc": 1}), encoding="utf-8")
        config = _base_config(
            tmp_path,
            success_receipts=[{"label": "done", "path": str(done)}],
            success_settle_s=0,
        )
        self._write_pid(config, _dead_pid())
        alert = watcher.evaluate(config, started_at=time.time(), now=time.time())
        assert alert is not None
        assert alert["reason"] == "child_dead"
        assert alert["success_receipt_summaries"]["done"]["rc"] == 1

    def test_absent_receipt_still_alerts_child_dead(self, tmp_path):
        config = _base_config(
            tmp_path,
            success_receipts=[{"label": "done", "path": str(tmp_path / "missing.json")}],
            success_settle_s=0,
        )
        self._write_pid(config, _dead_pid())
        alert = watcher.evaluate(config, started_at=time.time(), now=time.time())
        assert alert is not None
        assert alert["reason"] == "child_dead"

    def test_legacy_config_behavior_unchanged(self, tmp_path):
        config = _base_config(tmp_path)
        self._write_pid(config, _dead_pid())
        alert = watcher.evaluate(config, started_at=time.time(), now=time.time())
        assert alert is not None
        assert alert["reason"] == "child_dead"


class TestRunAdjudication:
    def test_clean_exit_returns_zero_without_alert_file(self, tmp_path, capsys):
        done = tmp_path / "done.json"
        done.write_text(json.dumps({"rc": 0}), encoding="utf-8")
        config = _base_config(
            tmp_path,
            success_receipts=[{"label": "done", "path": str(done)}],
            success_settle_s=0,
        )
        config["pid_file"].write_text(f"{_dead_pid()}\n", encoding="utf-8")
        rc = watcher.run(config, once=True)
        assert rc == 0
        assert not config["alert_path"].exists()
        assert "clean_exit" in capsys.readouterr().out

    def test_dead_child_without_receipt_alerts_after_zero_settle(self, tmp_path):
        config = _base_config(
            tmp_path,
            success_receipts=[{"label": "done", "path": str(tmp_path / "missing.json")}],
            success_settle_s=0,
        )
        config["pid_file"].write_text(f"{_dead_pid()}\n", encoding="utf-8")
        rc = watcher.run(config, once=False)
        assert rc == 1
        body = json.loads(config["alert_path"].read_text(encoding="utf-8"))
        assert body["reason"] == "child_dead"

    def test_receipt_written_during_settle_window_rescues(self, tmp_path):
        done = tmp_path / "done.json"
        config = _base_config(
            tmp_path,
            poll_s=1,
            success_receipts=[{"label": "done", "path": str(done)}],
            success_settle_s=30,
        )
        config["pid_file"].write_text(f"{_dead_pid()}\n", encoding="utf-8")
        # Simulate the launcher writing the receipt shortly after child exit.
        import threading

        threading.Timer(
            0.5, lambda: done.write_text(json.dumps({"rc": 0}), encoding="utf-8")
        ).start()
        rc = watcher.run(config, once=False)
        assert rc == 0
        assert not config["alert_path"].exists()


class TestZombiePidIsDead:
    """A zombie stays in the process table until its parent reaps it, so
    kill(pid, 0) reports it alive forever — the rescue-poller incident of
    2026-08-15 (a SIGSTOPped parent could not reap its exited child and the
    pid-liveness check looped for the full timeout). A zombie can never run
    again: liveness must treat stat ``Z`` as dead."""

    def test_zombie_counts_as_dead(self, monkeypatch):
        def fake_run(argv, **kwargs):
            class R:
                returncode = 0
                stdout = "Z+\n"
            return R()

        monkeypatch.setattr(watcher.subprocess, "run", fake_run)
        assert watcher._pid_alive(12345) is False

    def test_running_stat_counts_as_alive(self, monkeypatch):
        def fake_run(argv, **kwargs):
            class R:
                returncode = 0
                stdout = "S+\n"
            return R()

        monkeypatch.setattr(watcher.subprocess, "run", fake_run)
        # PID 1 (launchd) always exists; kill(1, 0) raises PermissionError,
        # which the alive check treats as alive without consulting ps — use a
        # real child of ours instead so the kill(pid, 0) leg also exercises.
        proc = subprocess.Popen(["sleep", "5"])
        try:
            assert watcher._pid_alive(proc.pid) is True
        finally:
            proc.kill()
            proc.wait()

    def test_ps_failure_falls_back_to_alive(self, monkeypatch):
        def fake_run(argv, **kwargs):
            raise OSError("ps unavailable")

        monkeypatch.setattr(watcher.subprocess, "run", fake_run)
        proc = subprocess.Popen(["sleep", "5"])
        try:
            assert watcher._pid_alive(proc.pid) is True
        finally:
            proc.kill()
            proc.wait()


class TestLauncherSuccessReceiptInjection:
    """The launcher must close the config-orphan half of #1064: a liveness
    config without success_receipts gets one derived from the wrapped
    command's safe_run --status-receipt, written as an EFFECTIVE copy (the
    caller's file is never mutated)."""

    def _load_launcher(self):
        spec = importlib.util.spec_from_file_location(
            "launch_detached_process", _REPO / "tools" / "launch_detached_process.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _bare_config(self, tmp_path: Path, **extra) -> Path:
        raw = {
            "schema": "pact.run_liveness_watcher.config.v1",
            "pid_file": str(tmp_path / "run.pid"),
            "alert_path": str(tmp_path / "alert.json"),
            "poll_s": 1,
        }
        raw.update(extra)
        path = tmp_path / "liveness.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def test_injects_receipt_from_status_receipt_flag(self, tmp_path):
        launcher = self._load_launcher()
        config_path = self._bare_config(tmp_path)
        cmd = ["python", "tools/safe_run.py", "--status-receipt", str(tmp_path / "s.json"), "--", "python", "train.py"]
        effective = launcher._augment_liveness_success_receipts(config_path, cmd, tmp_path)
        assert effective != config_path
        augmented = json.loads(effective.read_text(encoding="utf-8"))
        assert augmented["success_receipts"] == [
            {"label": "safe_run_status", "path": str(tmp_path / "s.json")}
        ]
        assert augmented["success_settle_s"] == 90
        # The caller's file is untouched.
        original = json.loads(config_path.read_text(encoding="utf-8"))
        assert "success_receipts" not in original

    def test_declared_receipts_pass_through(self, tmp_path):
        launcher = self._load_launcher()
        config_path = self._bare_config(
            tmp_path,
            success_receipts=[{"label": "mine", "path": str(tmp_path / "mine.json")}],
        )
        cmd = ["python", "tools/safe_run.py", "--status-receipt", str(tmp_path / "s.json")]
        assert launcher._augment_liveness_success_receipts(config_path, cmd, tmp_path) == config_path

    def test_no_status_receipt_passes_through(self, tmp_path):
        launcher = self._load_launcher()
        config_path = self._bare_config(tmp_path)
        cmd = ["python", "some_tool.py", "--flag", "value"]
        assert launcher._augment_liveness_success_receipts(config_path, cmd, tmp_path) == config_path

    def test_augmented_config_loads_and_adjudicates_clean_exit(self, tmp_path):
        launcher = self._load_launcher()
        heartbeat = tmp_path / "heartbeat.log"
        heartbeat.write_text("x\n", encoding="utf-8")
        config_path = self._bare_config(
            tmp_path,
            artifact_checks=[
                {"label": "hb", "path": str(heartbeat), "max_age_s": 3600, "grace_s": 0}
            ],
        )
        receipt = tmp_path / "safe_run_status.json"
        cmd = ["python", "tools/safe_run.py", "--status-receipt", str(receipt), "--", "python", "x.py"]
        effective = launcher._augment_liveness_success_receipts(config_path, cmd, tmp_path)
        config = watcher.load_config(effective)
        (tmp_path / "run.pid").write_text(str(_dead_pid()), encoding="utf-8")
        receipt.write_text(json.dumps({"exit": 0, "status": "ok"}), encoding="utf-8")
        now = time.time()
        alert = watcher.evaluate(config, started_at=now - 10, now=now)
        assert alert is not None and alert["reason"] == "clean_exit"
