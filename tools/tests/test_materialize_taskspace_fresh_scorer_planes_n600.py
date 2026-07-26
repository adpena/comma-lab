from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "materialize_taskspace_fresh_scorer_planes_n600.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("fresh_plane_cli", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_preflight_emits_bound_receipt(monkeypatch, tmp_path: Path, capsys) -> None:
    tool = _load_tool()
    receipt_path = tmp_path / "preflight.json"
    receipt_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        tool,
        "run_preflight",
        lambda _config: (
            receipt_path,
            {"preflight_sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(
        tool,
        "file_identity",
        lambda path: {"path": str(path), "bytes": 2, "sha256": "b" * 64},
    )
    assert tool.main([str(tmp_path / "config.json"), "--preflight-only"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["kind"] == "preflight"
    assert row["sealed_self_sha256"] == "a" * 64
    assert row["pointer_moved"] is False
    assert row["score_claim"] is False


def test_cli_materialize_is_explicit(monkeypatch, tmp_path: Path, capsys) -> None:
    tool = _load_tool()
    receipt_path = tmp_path / "aggregate.json"
    receipt_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        tool,
        "materialize",
        lambda _config: (
            receipt_path,
            {"aggregate_receipt_sha256": "c" * 64},
        ),
    )
    monkeypatch.setattr(
        tool,
        "file_identity",
        lambda path: {"path": str(path), "bytes": 2, "sha256": "d" * 64},
    )
    monkeypatch.setattr(tool, "assert_governed_admission", lambda *_a, **_kw: True)
    assert tool.main([str(tmp_path / "config.json"), "--materialize"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["kind"] == "aggregate"
    assert row["sealed_self_sha256"] == "c" * 64
    assert row["candidate_claim"] is False


def test_cli_refuses_implicit_action(tmp_path: Path) -> None:
    tool = _load_tool()
    with pytest.raises(SystemExit):
        tool.main([str(tmp_path / "config.json")])
