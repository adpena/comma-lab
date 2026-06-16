# SPDX-License-Identifier: MIT
"""NO-FAKE test for the disambiguator's last-evaluated-d_seg helper (the round-2 CRITICAL
fix). The sustained-win verdict needs the d_seg of the FINAL evaluated epoch; the prior code
read a non-existent ``summary['last_eval']`` key so it was always None → GO unreachable. The
helper reads the durable trajectory JSONL instead. Tests assert REAL parsing behavior.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

# Load the launcher module by path (it lives in experiments/, not an installed package).
_SPEC = importlib.util.spec_from_file_location(
    "_oomph_launcher",
    Path(__file__).resolve().parents[4] / "experiments" / "launch_oomph_finetune_disambiguator.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)
_last_evaluated_dseg = _mod._last_evaluated_dseg


def _write_traj(out_dir: Path, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "torch_vehicle_trajectory.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )


def test_returns_none_when_no_trajectory(tmp_path):
    assert _last_evaluated_dseg(tmp_path / "absent") is None


def test_returns_last_evaluated_dseg_not_min(tmp_path):
    """Picks the LAST non-null d_seg row — NOT the minimum (that's the whole point: a
    transient min must be distinguishable from a sustained final value)."""
    _write_traj(tmp_path, [
        {"global_epoch": 10, "evaluated": True, "d_seg": 0.0050},
        {"global_epoch": 20, "evaluated": True, "d_seg": 0.0031},  # the MIN (transient)
        {"global_epoch": 30, "evaluated": False, "d_seg": None},   # non-eval row (carried)
        {"global_epoch": 40, "evaluated": True, "d_seg": 0.0044},  # the LAST eval
    ])
    # last (0.0044), NOT min (0.0031).
    assert abs(_last_evaluated_dseg(tmp_path) - 0.0044) < 1e-9


def test_skips_trailing_null_and_malformed_rows(tmp_path):
    p = tmp_path / "r"
    p.mkdir()
    (p / "torch_vehicle_trajectory.jsonl").write_text(
        json.dumps({"global_epoch": 10, "d_seg": 0.0040}) + "\n"
        + json.dumps({"global_epoch": 20, "d_seg": None}) + "\n"
        + "{ this is not valid json\n"          # malformed trailing line
        + json.dumps({"global_epoch": 30, "d_seg": None}) + "\n"
    )
    # the last NON-null, well-formed d_seg is the ep10 row (0.0040).
    assert abs(_last_evaluated_dseg(p) - 0.0040) < 1e-9


def test_returns_none_when_all_rows_null(tmp_path):
    _write_traj(tmp_path, [
        {"global_epoch": 1, "evaluated": False, "d_seg": None},
        {"global_epoch": 2, "evaluated": False, "d_seg": None},
    ])
    assert _last_evaluated_dseg(tmp_path) is None
