"""Tests for the Lightning supply-chain scan wrapper."""
# ruff: noqa: I001
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "scan_lightning_supply_chain.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scan_lightning_supply_chain_under_test",
        str(SCRIPT),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scan_wrapper_emits_json_and_fails_strict_on_bad_pin(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "requirements.txt").write_text("lightning==2.6.3\n")
    out = tmp_path / "scan.json"

    rc = mod.main([
        "--repo-root",
        str(tmp_path),
        "--site-packages",
        str(tmp_path / "missing-site-packages"),
        "--json-out",
        str(out),
        "--strict",
        "--quiet",
    ])

    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["status"] == "FAIL"
    assert payload["violation_count"] >= 1
    assert payload["package_versions"]["lightning"] is None


def test_scan_wrapper_passes_for_clean_minimal_repo(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='clean'\n")

    rc = mod.main([
        "--repo-root",
        str(tmp_path),
        "--site-packages",
        str(tmp_path / "missing-site-packages"),
        "--strict",
        "--quiet",
    ])

    assert rc == 0
