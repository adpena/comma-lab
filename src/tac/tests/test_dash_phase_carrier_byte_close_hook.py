# SPDX-License-Identifier: MIT
"""Tests for the #425 dash-phase carrier byte-close hook (build_dash_phase_carrier_section)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "levelset_byte_close_and_eval", _REPO / "tools" / "levelset_byte_close_and_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("levelset_byte_close_and_eval", mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


@pytest.fixture()
def tiny_gt_cache(tmp_path: Path) -> Path:
    H, W, P = 96, 128, 4
    ls = np.zeros((P, H, W), dtype=np.int64)
    for p in range(P):
        ls[p, 40:43, 40 + p : 48 + p] = 1  # one translating lane dash
    poses = np.zeros((P, 6), dtype=np.float64)
    cp = tmp_path / "gt_tiny.npz"
    np.savez(cp, lstars=ls, gt_poses=poses, margins=np.ones((P, H, W), np.float32))
    return cp


def test_builder_requires_cache(tool):
    with pytest.raises(ValueError, match="dash-phase-carrier requires --gt-cache"):
        tool.build_dash_phase_carrier_section(None, 4, {})


def test_builder_missing_file_raises(tool, tmp_path):
    with pytest.raises(FileNotFoundError):
        tool.build_dash_phase_carrier_section(str(tmp_path / "nope.npz"), 4, {})


def test_builder_missing_keys_raises(tool, tmp_path):
    cp = tmp_path / "bad.npz"
    np.savez(cp, lstars=np.zeros((2, 8, 8), np.int64))
    with pytest.raises(ValueError, match="lacks"):
        tool.build_dash_phase_carrier_section(str(cp), 2, {})


def test_builder_end_to_end_report(tool, tiny_gt_cache):
    section, report = tool.build_dash_phase_carrier_section(str(tiny_gt_cache), 4, {})
    assert report["active"] is True
    assert report["section_bytes"] == len(section) > 0
    assert report["reconstruction_bit_identical"] is True
    assert report["n_births"] == 1 and report["n_matched"] == 3
    assert report["counted_rate_term_contribution"] > 0.0
    # NO-FAKE staging: bytes measured, d_seg explicitly not claimed
    assert report["recovered_d_seg"] is None
    assert "OWED" in report["recovered_d_seg_status"]
    assert "COUNTED (archive.zip)" in report["rule_118_boundary"]


def test_builder_cfg_include_xi_false(tool, tiny_gt_cache):
    _, rep_yes = tool.build_dash_phase_carrier_section(str(tiny_gt_cache), 4, {})
    _, rep_no = tool.build_dash_phase_carrier_section(str(tiny_gt_cache), 4, {"include_xi": False})
    assert rep_no["xi_bytes_in_section"] == 0
    assert rep_yes["xi_bytes_in_section"] == 4 * 6 * 2
    assert rep_no["section_bytes"] < rep_yes["section_bytes"]


def test_cli_flags_exist_and_default_off(tool):
    import subprocess

    out = subprocess.run(
        [sys.executable, str(_REPO / "tools" / "levelset_byte_close_and_eval.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert "--dash-phase-carrier" in out.stdout
    assert "--dash-phase-no-xi" in out.stdout
