# SPDX-License-Identifier: MIT
"""Sanity tests for ``tools/check_substrate_dykstra_feasibility.py``.

Per the HIGH-RISK substrate cargo-cult unwind audit 2026-05-16 D3
operator-approved decision: every L1+ substrate design memo MUST run
this helper against its predicted ΔS band; the gate emits a
FEASIBLE / INFEASIBLE / INDETERMINATE verdict consumed by the autopilot
ranker + design memo §predicted-band sections.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    """Load the CLI module without executing main()."""
    target = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "tools"
        / "check_substrate_dykstra_feasibility.py"
    )
    module_name = "check_substrate_dykstra_feasibility_test"
    spec = importlib.util.spec_from_file_location(module_name, target)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass typing introspection needs the module visible in sys.modules
    # at exec_module time.
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_helper_module_loads():
    mod = _load_module()
    assert hasattr(mod, "check_substrate_dykstra_feasibility")
    assert hasattr(mod, "DykstraFeasibilityVerdict")
    assert hasattr(mod, "CONTEST_RATE_DENOM_BYTES")
    assert mod.CONTEST_RATE_DENOM_BYTES == 37_545_489
    assert mod.CONTEST_SEG_MULTIPLIER == 100.0
    assert mod.CONTEST_SCORE_FORMULA.startswith("100*seg_dist")


def test_feasible_band_inside_polytope():
    mod = _load_module()
    # Archive size 200,000 B -> rate_contribution = 25 * 200000 / 37545489
    # ~= 0.1331. With S=0.001 and P=0.011 the feasible upper bound is
    # 0.1331 + 100*0.001 + sqrt(0.11) ~= 0.565. A band [0.15, 0.20] sits
    # comfortably inside the polytope.
    verdict = mod.check_substrate_dykstra_feasibility(
        substrate_id="test_feasible",
        predicted_band_lo=0.15,
        predicted_band_hi=0.20,
        archive_size_bytes=200_000,
    )
    assert verdict.verdict == "FEASIBLE"
    assert verdict.archive_size_bytes == 200_000
    assert verdict.blocker_axis is None
    assert verdict.feasibility_band_lo <= verdict.feasibility_band_hi
    assert verdict.dykstra_iteration_count >= 1
    assert verdict.score_formula == mod.CONTEST_SCORE_FORMULA
    assert verdict.contest_seg_multiplier == 100.0
    assert verdict.constraint_set_ids == mod.BASE_CONSTRAINT_SET_IDS
    assert verdict.feasibility_scope == mod.FEASIBILITY_SCOPE
    assert verdict.move_level_constraint_proof is False
    assert verdict.score_claim is False


def test_infeasible_band_below_rate_contribution_flags_rate_blocker():
    mod = _load_module()
    # Archive size 10,000,000 B -> rate contribution ~= 25 * 10M / 37.5M ~= 6.66.
    # A claimed band of [0.10, 0.18] is mathematically impossible because
    # even with zero distortion the rate alone exceeds 6.0.
    verdict = mod.check_substrate_dykstra_feasibility(
        substrate_id="test_rate_infeasible",
        predicted_band_lo=0.10,
        predicted_band_hi=0.18,
        archive_size_bytes=10_000_000,
    )
    assert verdict.verdict == "INFEASIBLE"
    assert verdict.blocker_axis == "rate"
    assert "rate" in verdict.feasibility_rationale.lower() or "blocker" in verdict.feasibility_rationale.lower()


def test_infeasible_band_above_polytope_flags_pose_blocker():
    mod = _load_module()
    # Tight budgets — predicted band claims a score well above what the
    # polytope can reach with seg=0.001 and pose=0.011 at a 100k archive.
    verdict = mod.check_substrate_dykstra_feasibility(
        substrate_id="test_pose_infeasible",
        predicted_band_lo=2.5,
        predicted_band_hi=3.0,
        archive_size_bytes=100_000,
    )
    assert verdict.verdict == "INFEASIBLE"
    assert verdict.blocker_axis in {"pose", "seg"}


def test_band_outside_rate_polytope_when_band_far_below_rate():
    mod = _load_module()
    # Archive 10M B -> rate ~6.66; band [0.1, 0.2] entirely below.
    verdict = mod.check_substrate_dykstra_feasibility(
        substrate_id="test_far_below",
        predicted_band_lo=0.1,
        predicted_band_hi=0.2,
        archive_size_bytes=10_000_000,
    )
    assert verdict.verdict == "INFEASIBLE"
    assert verdict.rate_contribution > 0.2


def test_contest_seg_multiplier_is_used_in_feasible_upper_bound():
    mod = _load_module()
    # If the helper forgot the contest 100x seg multiplier, this band would
    # be outside the feasible projection when pose_budget=0.0. With the real
    # contest formula, 100*0.001 makes the upper bound roughly 0.1.
    verdict = mod.check_substrate_dykstra_feasibility(
        substrate_id="test_seg_multiplier",
        predicted_band_lo=0.099,
        predicted_band_hi=0.101,
        archive_size_bytes=1,
        pose_budget=0.0,
    )
    assert verdict.verdict == "FEASIBLE"
    assert verdict.feasibility_band_hi == pytest.approx(
        verdict.rate_contribution + 0.1
    )


def test_invalid_inputs_raise_value_error():
    mod = _load_module()
    # Negative archive bytes
    try:
        mod.check_substrate_dykstra_feasibility(
            substrate_id="x",
            predicted_band_lo=0.1,
            predicted_band_hi=0.2,
            archive_size_bytes=-5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on negative archive_size_bytes")
    # Inverted band
    try:
        mod.check_substrate_dykstra_feasibility(
            substrate_id="x",
            predicted_band_lo=0.3,
            predicted_band_hi=0.1,
            archive_size_bytes=100_000,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on inverted band")
    # Empty substrate id
    try:
        mod.check_substrate_dykstra_feasibility(
            substrate_id="",
            predicted_band_lo=0.1,
            predicted_band_hi=0.2,
            archive_size_bytes=100_000,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty substrate_id")
    # Non-finite band
    try:
        mod.check_substrate_dykstra_feasibility(
            substrate_id="x",
            predicted_band_lo=float("nan"),
            predicted_band_hi=0.2,
            archive_size_bytes=100_000,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on NaN band")


def test_cli_writes_json_and_returns_zero_on_feasible(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tool = repo_root / "tools" / "check_substrate_dykstra_feasibility.py"
    out_path = tmp_path / "verdict.json"
    proc = subprocess.run(
        [
            sys.executable, str(tool),
            "--substrate-id", "test_cli_feasible",
            "--predicted-band-lo", "0.15",
            "--predicted-band-hi", "0.20",
            "--archive-size-bytes", "200000",
            "--tt5l-five-move-polytope",
            "--output-json", str(out_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["verdict"] == "FEASIBLE"
    assert data["substrate_id"] == "test_cli_feasible"
    assert data["archive_size_bytes"] == 200_000
    assert data["score_formula"].startswith("100*seg_dist")
    assert "tt5l_predictive_coding_hierarchy" in data["constraint_set_ids"]
    assert data["constraint_set_count"] == 8
    assert data["feasibility_scope"] == "score_axis_sanity_only"
    assert data["move_level_constraint_proof"] is False
    assert "not a move-level feasibility proof" in data["projection_limitations"]
    assert data["score_claim"] is False


def test_cli_returns_nonzero_on_infeasible(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tool = repo_root / "tools" / "check_substrate_dykstra_feasibility.py"
    proc = subprocess.run(
        [
            sys.executable, str(tool),
            "--substrate-id", "test_cli_infeasible",
            "--predicted-band-lo", "0.1",
            "--predicted-band-hi", "0.2",
            "--archive-size-bytes", "10000000",
        ],
        capture_output=True, text=True, timeout=30,
    )
    # rc=1 for INFEASIBLE per CLI contract; rc=2 for ValueError.
    assert proc.returncode == 1, (proc.returncode, proc.stderr)
    data = json.loads(proc.stdout)
    assert data["verdict"] == "INFEASIBLE"
    assert data["blocker_axis"] == "rate"


def test_cli_returns_nonzero_on_indeterminate_without_explicit_allowance(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tool = repo_root / "tools" / "check_substrate_dykstra_feasibility.py"
    out_path = tmp_path / "verdict.json"
    base_args = [
        sys.executable,
        str(tool),
        "--substrate-id",
        "test_cli_indeterminate",
        "--predicted-band-lo",
        "0.1",
        "--predicted-band-hi",
        "0.2",
        "--archive-size-bytes",
        "0",
        "--pose-budget",
        "0",
        "--output-json",
        str(out_path),
    ]
    proc = subprocess.run(
        base_args,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["verdict"] == "INDETERMINATE"

    allowed = subprocess.run(
        [*base_args, "--allow-indeterminate-exit-zero"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert allowed.returncode == 0, allowed.stderr
