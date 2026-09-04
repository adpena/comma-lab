# SPDX-License-Identifier: MIT
"""Tests for experiments/ddm_reseal_pins_inside_sealed_tree.py — path re-root only, content drift refused."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ddm_reseal_pins_inside_sealed_tree as tool


def _cfg(tmp_path: Path, pins: dict) -> Path:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"cell_id": "x", "source_pins": pins, "other": 1}))
    return p


def test_reroot_changes_only_paths(tmp_path, monkeypatch):
    old = {"a": {"path": "/work/a", "sha256": "s1", "bytes": 1}, "b": {"path": "/work/b", "sha256": "s2", "bytes": 2}}
    live = {"a": {"path": "/sealed/a", "sha256": "s1", "bytes": 1}, "b": {"path": "/work/b", "sha256": "s2", "bytes": 2}}
    monkeypatch.setattr(tool, "verify_inputs_inside", lambda _t: live)
    out, rc = tmp_path / "out.json", tmp_path / "r.json"
    receipt = tool.reroot(_cfg(tmp_path, old), tmp_path, out, rc)
    got = json.loads(out.read_text())
    assert got["source_pins"]["a"]["path"] == "/sealed/a"
    assert got["source_pins"]["b"]["path"] == "/work/b"
    assert got["other"] == 1 and got["cell_id"] == "x"
    assert receipt["paths_rerooted"] == ["a"] and receipt["pins_total"] == 2
    assert json.loads(rc.read_text())["schema"] == "sealed_config_pin_reroot.v1"


def test_content_drift_is_refused(tmp_path, monkeypatch):
    old = {"a": {"path": "/work/a", "sha256": "s1", "bytes": 1}}
    live = {"a": {"path": "/sealed/a", "sha256": "DIFFERENT", "bytes": 1}}
    monkeypatch.setattr(tool, "verify_inputs_inside", lambda _t: live)
    with pytest.raises(tool.ResealError, match="CONTENT drift"):
        tool.reroot(_cfg(tmp_path, old), tmp_path, tmp_path / "o.json", tmp_path / "r.json")
    assert not (tmp_path / "o.json").exists()


def test_key_set_mismatch_is_refused(tmp_path, monkeypatch):
    old = {"a": {"path": "/work/a", "sha256": "s1", "bytes": 1}}
    live = {"a": {"path": "/sealed/a", "sha256": "s1", "bytes": 1}, "zz": {"path": "/sealed/zz", "sha256": "s9", "bytes": 9}}
    monkeypatch.setattr(tool, "verify_inputs_inside", lambda _t: live)
    with pytest.raises(tool.ResealError, match="pin key sets differ"):
        tool.reroot(_cfg(tmp_path, old), tmp_path, tmp_path / "o.json", tmp_path / "r.json")
