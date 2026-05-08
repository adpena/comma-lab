"""Tests for tools/cathedral_autopilot_meta_lagrangian_bridge.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "cathedral_autopilot_meta_lagrangian_bridge.py"


def _load_bridge():
    spec = importlib.util.spec_from_file_location(
        "cathedral_autopilot_meta_lagrangian_bridge_under_test",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rank_axis_metadata_defaults_missing_to_dual_with_flag() -> None:
    bridge = _load_bridge()
    rank_axis, missing = bridge._rank_axis_metadata(
        {"name": "legacy_row_without_rank_axis"},
        {},
    )
    assert rank_axis == "dual"
    assert missing is True


def test_rank_axis_metadata_inherits_plan_operator_axis() -> None:
    bridge = _load_bridge()
    rank_axis, missing = bridge._rank_axis_metadata(
        {"name": "row_without_rank_axis"},
        {"operator_state": {"rank_axis": "cpu"}},
    )
    assert rank_axis == "cpu"
    assert missing is False


def test_rank_axis_metadata_row_overrides_plan_axis() -> None:
    bridge = _load_bridge()
    rank_axis, missing = bridge._rank_axis_metadata(
        {"name": "row_with_axis", "rank_axis": "cuda"},
        {"operator_state": {"rank_axis": "dual"}},
    )
    assert rank_axis == "cuda"
    assert missing is False


def test_rank_axis_metadata_rejects_invalid_axis() -> None:
    bridge = _load_bridge()
    with pytest.raises(ValueError, match="invalid rank_axis"):
        bridge._rank_axis_metadata(
            {"name": "bad_row", "rank_axis": "gpu"},
            {},
        )


def test_current_score_axis_metadata_defaults_missing_to_unspecified() -> None:
    bridge = _load_bridge()
    axis, missing = bridge._current_score_axis_metadata(
        {"name": "legacy_row_without_current_axis"},
        {},
    )
    assert axis == "unspecified"
    assert missing is True


def test_current_score_axis_metadata_inherits_plan_operator_axis() -> None:
    bridge = _load_bridge()
    axis, missing = bridge._current_score_axis_metadata(
        {"name": "row_without_current_axis"},
        {"operator_state": {"current_score_axis": "cpu"}},
    )
    assert axis == "cpu"
    assert missing is False


def test_current_score_axis_metadata_rejects_invalid_axis() -> None:
    bridge = _load_bridge()
    with pytest.raises(ValueError, match="invalid current_score_axis"):
        bridge._current_score_axis_metadata(
            {"name": "bad_current_axis", "current_score_axis": "mps"},
            {},
        )
