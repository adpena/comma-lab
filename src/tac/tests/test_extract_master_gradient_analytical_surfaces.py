# SPDX-License-Identifier: MIT
"""Tests for the analytical-surface coverage manifest in extract_master_gradient.

Per Cable D D2 (task #887/#890) lane
`lane_cable_d_master_gradient_extension_batch_20260519`. Verifies the manifest
schema and CLI subcommand surface.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "extract_master_gradient.py"


@pytest.fixture(scope="module")
def extract_module():
    spec = importlib.util.spec_from_file_location(
        "extract_master_gradient_test_module", TOOL_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extract_master_gradient_test_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_list_analytical_surfaces_returns_canonical_schema(extract_module):
    manifest = extract_module.list_analytical_surfaces()
    assert manifest["schema"] == "master_gradient_analytical_surface_manifest_v1"
    assert manifest["score_claim_allowed"] is False
    assert manifest["promotion_eligible"] is False


def test_manifest_has_surfaces_list(extract_module):
    manifest = extract_module.list_analytical_surfaces()
    assert "surfaces" in manifest
    assert isinstance(manifest["surfaces"], list)
    assert len(manifest["surfaces"]) >= 10  # at least 10 surfaces tracked


def test_each_surface_has_required_fields(extract_module):
    manifest = extract_module.list_analytical_surfaces()
    for entry in manifest["surfaces"]:
        assert "surface_id" in entry
        assert "module_path" in entry
        assert "coverage" in entry
        assert "wire_in_hook" in entry
        assert "notes" in entry
        assert entry["coverage"] in {"active", "indirect", "decorator", "pending"}


def test_coverage_fractions_sum_to_one(extract_module):
    manifest = extract_module.list_analytical_surfaces()
    active = manifest["coverage_fraction_active"]
    pending = manifest["coverage_fraction_pending"]
    assert 0.0 <= active <= 1.0
    assert 0.0 <= pending <= 1.0
    assert abs(active + pending - 1.0) < 1e-6


def test_canonical_consumer_surface_is_active(extract_module):
    """The master_gradient_consumers module itself MUST be marked active."""
    manifest = extract_module.list_analytical_surfaces()
    by_id = {s["surface_id"]: s for s in manifest["surfaces"]}
    assert "tac.master_gradient_consumers" in by_id
    assert by_id["tac.master_gradient_consumers"]["coverage"] == "active"


def test_cathedral_autopilot_loop_tracked(extract_module):
    """The cathedral autopilot loop tool MUST be tracked as an analytical surface."""
    manifest = extract_module.list_analytical_surfaces()
    by_id = {s["surface_id"]: s for s in manifest["surfaces"]}
    assert "tools/cathedral_autopilot_autonomous_loop.py" in by_id


def test_cli_list_analytical_surfaces_subcommand_emits_json():
    """End-to-end CLI dispatch via subprocess."""
    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "list-analytical-surfaces"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    payload = json.loads(result.stdout)
    assert payload["schema"] == "master_gradient_analytical_surface_manifest_v1"
    assert "surfaces" in payload
