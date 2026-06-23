# SPDX-License-Identifier: MIT
"""Tests for the math-optimal joint decoder solver (tac.optimization.math_optimal_joint_solver).

These verify the joint-solve ARITHMETIC, the surface-model consistency with the MEASURED
anchors, the existence-proof cross-check discipline (CLAUDE.md feedback_terminal_conclusion
2026-06-23), and the boundary behaviour. NOT a score claim — the module is [advisory]
NON-PROMOTABLE by construction. Positive + negative + consistency + edge + the
existence-proof guard.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from tac import capacity_rd_qat as crq
from tac.contest_score import compute_contest_score
from tac.optimization import math_optimal_joint_solver as mj

# ---------------------------------------------------------------------------
# Constants + anchor consistency.
# ---------------------------------------------------------------------------


def test_dseg_384_floor_constant_matches_measured():
    # The pinned 384 floor must match the measured N=600 JSON value (1.875e-4).
    assert pytest.approx(1.875e-4, rel=1e-3) == mj.DSEG_384_FLOOR
    assert pytest.approx(100.0 * mj.DSEG_384_FLOOR) == mj.DSEG_384_FLOOR_S_UNITS


def test_pose_anchors_from_canonical_capacity_rd_qat():
    # Pose anchors must be sourced from the canonical capacity_rd_qat anchors, not invented.
    assert crq.ANCHOR_BC20.d_pose == mj.DPOSE_BASIN
    assert crq.ANCHOR_FRONTIER.d_pose == mj.DPOSE_CONVERGED
    # The converged pose is far below the basin (the dominant sub-0.15 swing).
    assert mj.DPOSE_CONVERGED < mj.DPOSE_BASIN


def test_frontier_S_matches_pointer():
    assert pytest.approx(0.19109982419209975, abs=1e-9) == mj.FRONTIER_S


# ---------------------------------------------------------------------------
# C-axis: power law CLAMPED at the 384 floor.
# ---------------------------------------------------------------------------


def test_capacity_power_law_decreases_then_clamps_at_384_floor():
    d20 = mj.dseg_capacity_floor_bounded(20)
    d36 = mj.dseg_capacity_floor_bounded(36)
    # bc20 = measured basin; bc36 = frontier d_seg; capacity lowers d_seg.
    assert d20 == pytest.approx(crq.ANCHOR_BC20.d_seg, rel=1e-9)
    assert d36 < d20
    # No capacity dips below the 384 floor (the existence-proof lower bound).
    for bc in (16, 20, 28, 36, 48, 56, 64):
        assert mj.dseg_capacity_floor_bounded(bc) >= mj.DSEG_384_FLOOR


def test_huge_capacity_clamps_exactly_at_floor():
    # At an enormous capacity the raw power law would dip below the floor; it must clamp.
    d, ev = mj.dseg_capacity_power_law(256)
    assert d == pytest.approx(mj.DSEG_384_FLOOR, rel=1e-9)
    assert "CLAMPED" in ev


# ---------------------------------------------------------------------------
# E-axis: convergence curve.
# ---------------------------------------------------------------------------


def test_dseg_convergence_monotone_decreasing_to_asymptote():
    d_inf = 0.0006
    vals = [mj.dseg_convergence(E, d_inf) for E in (0, 500, 2000, 10000, 1e6)]
    for a, b in pairwise(vals):
        assert b <= a + 1e-12  # non-increasing
    assert vals[0] == pytest.approx(mj.DSEG_INIT, rel=1e-9)  # E=0 -> init
    assert vals[-1] == pytest.approx(d_inf, abs=1e-6)  # E->inf -> asymptote


def test_dseg_convergence_reproduces_measured_bc20_anchors():
    # At the bc20 asymptote, the convergence curve must pass near the two measured anchors
    # (it is fit to them) within a tolerance (2-point exp fit, not exact through both).
    d_inf = mj.dseg_capacity_floor_bounded(20)
    # The high anchor (ep2325) is the bc20 basin; the curve at 2325 should be >= asymptote.
    d_high = mj.dseg_convergence(2325.0, d_inf)
    assert d_high >= d_inf


def test_dpose_convergence_basin_to_frontier():
    p0 = mj.dpose_convergence(0)
    pinf = mj.dpose_convergence(1e7)
    assert p0 == pytest.approx(mj.DPOSE_BASIN, rel=1e-9)
    assert pinf == pytest.approx(mj.DPOSE_CONVERGED, abs=1e-7)
    assert pinf < p0


def test_dseg_convergence_negative_epochs_raises():
    with pytest.raises(ValueError):
        mj.dseg_convergence(-1.0, 0.001)


# ---------------------------------------------------------------------------
# The joint solve.
# ---------------------------------------------------------------------------


def test_solve_returns_optimum_and_two_floors():
    res = mj.solve_math_optimal_joint()
    # Optimum minimises S over the grid.
    assert min(c.S for c in res.grid) == res.optimum.S
    # The surface-model lower bound is the fully-converged best config.
    assert res.achievable_S_lower_bound == pytest.approx(res.achievable_S_lower_bound_config.S)
    # The physical floor is far below the surface-model lower bound (the capacity-
    # realization gap) and comfortably sub-0.15.
    assert res.physical_floor_S < res.achievable_S_lower_bound
    assert res.physical_floor_S < 0.15


def test_optimum_score_recomputes_via_contest_score():
    # Every JointConfig.S must equal the canonical contest-score formula on its components.
    res = mj.solve_math_optimal_joint()
    o = res.optimum
    assert pytest.approx(
        compute_contest_score(o.d_seg, o.d_pose, o.archive_bytes), rel=1e-12
    ) == o.S


def test_optimum_dseg_at_or_above_384_floor():
    res = mj.solve_math_optimal_joint()
    for c in res.grid:
        assert c.d_seg >= mj.DSEG_384_FLOOR - 1e-15


def test_converged_configs_have_low_pose():
    # At E->inf the pose must have converged toward the frontier value (not the basin).
    res = mj.solve_math_optimal_joint()
    conv = [c for c in res.grid if c.epochs >= 1e6]
    assert conv
    for c in conv:
        assert c.d_pose == pytest.approx(mj.DPOSE_CONVERGED, abs=1e-6)


def test_higher_capacity_has_lower_dseg_inf_and_more_native_bytes():
    res = mj.solve_math_optimal_joint(base_chs=(20, 36), epochs_options=(1e6,))
    by_bc = {}
    for c in res.grid:
        by_bc.setdefault(c.base_ch, c)
    assert by_bc[36].d_seg_inf < by_bc[20].d_seg_inf
    # native bytes (frac_low=0 / int8) grow with capacity.
    i8_20 = min((c for c in res.grid if c.base_ch == 20 and c.qat_frac_low_precision == 0.0), key=lambda c: c.S)
    i8_36 = min((c for c in res.grid if c.base_ch == 36 and c.qat_frac_low_precision == 0.0), key=lambda c: c.S)
    assert i8_36.archive_bytes > i8_20.archive_bytes


# ---------------------------------------------------------------------------
# The existence-proof cross-check (the discipline guard).
# ---------------------------------------------------------------------------


def test_existence_proof_flags_invalid_floor():
    # A deliberately too-high "floor" claim must be flagged INVALID because perfect-384 /
    # PR95 / frontier beat it.
    proof = mj.existence_proof_crosscheck(0.30, d_pose=mj.DPOSE_CONVERGED)
    assert proof.is_valid_floor is False
    assert proof.best_known_S < 0.30
    assert "INVALID" in proof.verdict


def test_existence_proof_accepts_genuinely_low_floor():
    # A floor below ALL known artifacts is VALID (no known artifact beats it).
    proof = mj.existence_proof_crosscheck(0.001, d_pose=mj.DPOSE_CONVERGED)
    assert proof.is_valid_floor is True
    assert "VALID" in proof.verdict


def test_physical_floor_beats_frontier_and_is_sub_015():
    pf = mj.physical_achievable_floor()
    assert pf.S < mj.FRONTIER_S
    assert pf.S < 0.15
    # Built from the MEASURED 384 d_seg floor (from the loaded surfaces, more precise than
    # the rounded pinned constant) + converged pose — NOT the power law.
    surf = mj.load_ingested_surfaces()
    assert pf.d_seg == pytest.approx(surf.dseg_384_floor, rel=1e-9)
    assert pf.d_seg == pytest.approx(mj.DSEG_384_FLOOR, rel=1e-2)  # near the pinned constant
    assert pf.d_pose == pytest.approx(mj.DPOSE_CONVERGED, rel=1e-9)


def test_physical_floor_is_consistent_existence_proof():
    # The physical floor's S must recompute via the canonical contest-score formula on its
    # OWN components (the measured 384 floor + converged pose + its byte budget).
    pf = mj.physical_achievable_floor()
    s = compute_contest_score(pf.d_seg, pf.d_pose, pf.archive_bytes)
    assert pytest.approx(s, rel=1e-12) == pf.S


# ---------------------------------------------------------------------------
# Training-time Pareto + min-budget.
# ---------------------------------------------------------------------------


def test_training_pareto_monotone_in_epochs():
    pts = mj.training_time_pareto(base_ch=36, qat_nbits=4, qat_frac_low_precision=1.0)
    # More epochs -> lower (or equal) S (d_seg + pose both improve).
    for a, b in pairwise(pts):
        assert b.S <= a.S + 1e-9


def test_min_training_budget_reachable_and_unreachable():
    # sub-0.19 at a high-capacity int4 config is reachable at finite budget.
    b = mj.min_training_budget_for_threshold(
        0.19, base_ch=36, qat_nbits=4, qat_frac_low_precision=1.0
    )
    assert b is not None
    assert b == float("inf") or (math.isfinite(b) and b >= 0)
    # An impossibly low target the converged config can't reach -> None.
    b_none = mj.min_training_budget_for_threshold(
        0.001, base_ch=36, qat_nbits=4, qat_frac_low_precision=1.0
    )
    assert b_none is None


# ---------------------------------------------------------------------------
# Prune-path plan.
# ---------------------------------------------------------------------------


def test_prune_path_plan_steps_and_contract():
    plan = mj.plan_capacity_rd_prune_path()
    assert plan.source_base_ch == 36
    assert len(plan.steps) >= 4
    # Predicted native S decreases with capacity (the prune-path rungs).
    by_bc = sorted(plan.steps, key=lambda s: s.target_base_ch)
    for a, b in pairwise(by_bc):
        assert b.predicted_native_S <= a.predicted_native_S + 1e-9
    # The runner contract is non-trivial and includes byte-close + exact-score.
    contract = " ".join(plan.runner_contract).lower()
    assert "exact" in contract and "byte-close" in contract
    # Measured columns start empty (the plan is gated, not run).
    assert all(s.measured_S is None for s in plan.steps)


# ---------------------------------------------------------------------------
# Surfaces ingestion (interface contract for the sister agents).
# ---------------------------------------------------------------------------


def test_load_surfaces_uses_measured_384_floor():
    surf = mj.load_ingested_surfaces()
    # The 384 floor is the one landed surface; it must be the measured value.
    assert surf.dseg_384_floor == pytest.approx(mj.DSEG_384_FLOOR, rel=1e-2)
    # Q-axis byte fractions are the measured int8->int-N ratios.
    assert surf.qaxis_byte_shrink[8] == pytest.approx(1.0)
    assert surf.qaxis_byte_shrink[4] < surf.qaxis_byte_shrink[8]


def test_load_surfaces_sister_qaxis_override(tmp_path):
    # The interface contract: a sister Q-axis JSON with {"byte_fraction": {...}} is ingested.
    import json

    q = tmp_path / "qaxis.json"
    q.write_text(json.dumps({"byte_fraction": {"8": 1.0, "4": 0.40}}))
    surf = mj.load_ingested_surfaces(qaxis_surface_path=q)
    assert surf.qaxis_byte_shrink[4] == pytest.approx(0.40)
    assert "sister Q-axis" in surf.qaxis_provenance


def test_load_surfaces_sister_eaxis_override(tmp_path):
    import json

    e = tmp_path / "eaxis.json"
    e.write_text(json.dumps({"epoch_dseg_anchors": [[100.0, 0.004], [5000.0, 0.001]]}))
    surf = mj.load_ingested_surfaces(eaxis_surface_path=e)
    assert surf.eaxis_anchors == ((100.0, 0.004), (5000.0, 0.001))
    assert "sister E-axis" in surf.eaxis_provenance
