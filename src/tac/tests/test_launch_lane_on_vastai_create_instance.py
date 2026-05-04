from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "scripts" / "launch_lane_on_vastai.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("launch_lane_on_vastai_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vast_create_api_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()

    def fake_run(_cmd, timeout=60, capture=True):
        return 0, '{"error": true, "status_code": 400, "msg": "Your account lacks credit"}', ""

    monkeypatch.setattr(launcher, "run", fake_run)
    with pytest.raises(RuntimeError, match="API error"):
        launcher.create_instance(123, "unit-test", disk_gb=80)


def test_vast_create_empty_output_reports_unknown_state(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher()

    def fake_run(_cmd, timeout=60, capture=True):
        return 0, "", ""

    monkeypatch.setattr(launcher, "run", fake_run)
    with pytest.raises(RuntimeError, match="empty stdout/stderr"):
        launcher.create_instance(123, "unit-test", disk_gb=80)
