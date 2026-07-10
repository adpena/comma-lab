"""Test the clause-A geometric-section derivability audit wired into the byte-close report.

The audit is score-neutral + read-only + fail-open. We test the section helper directly (importing the
byte-close CLI module) against a synthetic gt-cache npz with an ``lstars`` key.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
_TOOL = _REPO / "tools" / "levelset_byte_close_and_eval.py"


def _load_tool():
    if str(_REPO / "src") not in sys.path:
        sys.path.insert(0, str(_REPO / "src"))
    spec = importlib.util.spec_from_file_location("_lbce_dedup_test", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_lstars(n=2, h=48, w=64) -> np.ndarray:
    """A tiny 5-class argmax field with a road/undrivable/lane/movable/mycar layout (self-detectable)."""
    a = np.full((n, h, w), 2, dtype=np.int64)      # top = undrivable(2)
    a[:, h // 2:, :] = 0                            # lower = road(0)
    a[:, -8:, :] = 4                                # bottom = mycar(4)
    a[:, h // 2, ::4] = 1                           # a lane row (1)
    a[:, h // 2 - 4: h // 2, w // 2: w // 2 + 6] = 3   # a movable blob (3)
    return a


def test_dedup_section_none_without_gt_cache():
    mod = _load_tool()
    assert mod.dedup_audit_section(None) is None
    assert mod.dedup_audit_section("/nonexistent/path.npz") is None


def test_dedup_section_returns_pairwise_table(tmp_path):
    mod = _load_tool()
    npz = tmp_path / "gt.npz"
    np.savez(npz, lstars=_synthetic_lstars())
    out = mod.dedup_audit_section(str(npz), max_frames=2)
    assert out is not None
    assert out["n_frames_audited"] == 2
    assert out["source_gt_cache"] == str(npz)
    assert isinstance(out["pairs"], list) and len(out["pairs"]) >= 1
    for p in out["pairs"]:
        assert {"row_a", "row_b", "shared_px_total", "overlap_bytes_double_counted"} <= set(p)
    assert "clause-A" in out["note"]


def test_dedup_section_fail_open_on_missing_lstars_key(tmp_path):
    mod = _load_tool()
    npz = tmp_path / "no_lstars.npz"
    np.savez(npz, something_else=np.zeros((2, 4, 4)))
    assert mod.dedup_audit_section(str(npz)) is None  # no lstars -> None, never raises
