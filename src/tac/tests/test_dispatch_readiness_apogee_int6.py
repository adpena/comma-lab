from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "dispatch_readiness_apogee_int6.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("dispatch_readiness_apogee_int6_test", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apogee_int6_readiness_fails_closed_on_missing_distortion_model() -> None:
    module = _load_tool()

    ok, detail = module._check_predicted_band_below_frontier()

    assert ok is False
    assert "lacks a valid distortion model" in detail
    assert "cannot authorize dispatch" in detail


def test_apogee_int6_cli_json_cannot_become_dispatch_ready(monkeypatch, capsys) -> None:
    module = _load_tool()
    monkeypatch.setattr(module, "_check_archive_integrity", lambda: (True, "archive ok"))
    monkeypatch.setattr(module, "_check_sanity_ladder", lambda: (True, "sanity ok"))
    monkeypatch.setattr(module, "_check_preflight_clean", lambda: (True, "preflight ok"))
    monkeypatch.setattr(module, "_check_lane_registry_consistent", lambda: (True, "registry ok"))

    rc = module.main(["--json"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert payload["all_ok"] is False
    assert any(
        check["name"] == "predicted_band_below_frontier" and check["ok"] is False
        for check in payload["checks"]
    )


def test_apogee_intn_inflate_sh_is_executable_for_contest_runtime() -> None:
    inflate_sh = REPO / "submissions" / "apogee_intN" / "inflate.sh"

    assert inflate_sh.is_file()
    assert os.access(inflate_sh, os.X_OK)
