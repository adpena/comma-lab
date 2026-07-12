# SPDX-License-Identifier: MIT
"""Tests for tools/modal_dispatch.py — the detached Modal dispatch wrapper."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "modal_dispatch.py"


def _load():
    spec = importlib.util.spec_from_file_location("modal_dispatch", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_worker(monkeypatch, tmp_path, fake_cmd: list[str]):
    """Run the internal _run worker synchronously against a fake command."""
    mod = _load()
    monkeypatch.setattr(mod, "STATE_DIR", tmp_path)
    import argparse

    ns = argparse.Namespace(dispatch_id="unit_test", argv=["--", *fake_cmd])
    rc = mod.cmd_run(ns)
    assert rc == 0
    state = json.loads((tmp_path / "unit_test.json").read_text())
    return mod, state


def test_worker_extracts_app_and_call_id_and_marks_done(monkeypatch, tmp_path):
    fake = [
        sys.executable,
        "-c",
        "print('creating objects ap-ABCdef123456'); "
        "print('dispatch fc-01ABCDEF23456789'); "
        "print('LAUNCHER EXITED rc=0')",
    ]
    _mod, state = _run_worker(monkeypatch, tmp_path, fake)
    assert state["app_id"] == "ap-ABCdef123456"
    assert state["call_id"] == "fc-01ABCDEF23456789"
    assert state["state"] == "done"
    assert state["launcher_rc"] == 0


def test_worker_marks_refused_when_no_call_id(monkeypatch, tmp_path):
    fake = [sys.executable, "-c", "print('REFUSED: some money-safety guard fired'); raise SystemExit(1)"]
    _mod, state = _run_worker(monkeypatch, tmp_path, fake)
    assert state["state"] == "refused"
    assert state["call_id"] is None
    assert state["launcher_rc"] == 1


def test_worker_failed_when_nonzero_but_no_ids(monkeypatch, tmp_path):
    fake = [sys.executable, "-c", "print('some output'); raise SystemExit(3)"]
    _mod, state = _run_worker(monkeypatch, tmp_path, fake)
    assert state["state"] == "failed"
    assert state["launcher_rc"] == 3


def test_worker_failed_but_call_id_still_captured(monkeypatch, tmp_path):
    # rc!=0 AFTER a call_id spawned => flagged failed for review, id still harvestable.
    fake = [
        sys.executable,
        "-c",
        "print('dispatch fc-01KEEPME2345678'); raise SystemExit(2)",
    ]
    _mod, state = _run_worker(monkeypatch, tmp_path, fake)
    assert state["call_id"] == "fc-01KEEPME2345678"
    assert state["state"] == "failed"


def test_status_reads_durable_state(monkeypatch, tmp_path, capsys):
    fake = [
        sys.executable,
        "-c",
        "print('ap-STATUS1234567 fc-01STATUS23456789'); print('LAUNCHER EXITED rc=0')",
    ]
    mod, _state = _run_worker(monkeypatch, tmp_path, fake)
    import argparse

    ns = argparse.Namespace(
        dispatch_id="unit_test", label=None, all=False, limit=5, live=False, json=True
    )
    assert mod.cmd_status(ns) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["app_id"] == "ap-STATUS1234567"
    assert out[0]["call_id"] == "fc-01STATUS23456789"
    assert out[0]["state"] == "done"


def test_cli_help_smoke():
    r = subprocess.run(
        [sys.executable, str(TOOL), "--help"], capture_output=True, text=True, cwd=str(REPO)
    )
    assert r.returncode == 0
    assert "fire" in r.stdout and "status" in r.stdout and "stop" in r.stdout
