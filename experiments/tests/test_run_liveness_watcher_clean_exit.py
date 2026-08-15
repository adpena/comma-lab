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
