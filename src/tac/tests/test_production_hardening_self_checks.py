"""Tests for production-hardening self-check methods landed in defense-in-depth pass.

Per CLAUDE.md "production hardening + polish + defense-in-depth" landing
2026-05-09: every long-lived solver primitive now exposes an explicit self-check
method so a tripwire can fire BEFORE corrupt state propagates.

Surfaces covered:
- ``tac.unified_action.Action.assert_invariants()``
- ``tac.continual_learning.ContinualLearningPosterior.is_consistent()``

Memory: feedback_production_hardening_polish_defense_in_depth_landed_20260509.md.
"""
from __future__ import annotations

import math

import pytest

from tac.continual_learning import (
    ContinualLearningPosterior,
    PerTrackPosterior,
    SourceRhoPosterior,
)
from tac.unified_action import Action, DualVariables


# ── Action.assert_invariants ─────────────────────────────────────────────


def test_action_assert_invariants_clean_state_passes():
    """A well-formed Action with at least one baseline track passes."""
    action = Action(
        L_seg=lambda theta: theta.sum(),
        duals=DualVariables(),
    )
    action.assert_invariants()


def test_action_assert_invariants_no_baseline_raises():
    """An Action with only refinement tracks (no baseline) raises."""
    action = Action(
        L_t11=lambda theta: theta.sum(),
        duals=DualVariables(lambda_t11=1.0),
    )
    with pytest.raises(ValueError, match="no baseline track wired"):
        action.assert_invariants()


def test_action_assert_invariants_negative_baseline_dual_raises():
    """A baseline-track dual that went negative raises."""
    action = Action(
        L_seg=lambda theta: theta.sum(),
        duals=DualVariables(lambda_seg=-1.0),
    )
    with pytest.raises(ValueError, match="lambda_seg.*< 0"):
        action.assert_invariants()


def test_action_assert_invariants_nan_dual_raises():
    """A NaN dual (from a bad dual_update callable) raises."""
    action = Action(
        L_seg=lambda theta: theta.sum(),
        duals=DualVariables(lambda_t7=math.nan),
    )
    with pytest.raises(ValueError, match="not finite"):
        action.assert_invariants()


def test_action_assert_invariants_inf_dual_raises():
    """An Inf dual raises."""
    action = Action(
        L_seg=lambda theta: theta.sum(),
        duals=DualVariables(lambda_t11=math.inf),
    )
    with pytest.raises(ValueError, match="not finite"):
        action.assert_invariants()


def test_action_assert_invariants_signed_refinement_dual_ok():
    """Refinement track duals MAY be signed (council-approved sign flip)."""
    action = Action(
        L_seg=lambda theta: theta.sum(),
        L_t11=lambda theta: theta.sum(),
        duals=DualVariables(lambda_t11=-0.5),
    )
    # No error — refinement duals can be signed
    action.assert_invariants()


# ── ContinualLearningPosterior.is_consistent ─────────────────────────────


def test_posterior_is_consistent_empty_state_passes():
    """A fresh empty posterior passes."""
    p = ContinualLearningPosterior()
    ok, problems = p.is_consistent()
    assert ok, f"empty posterior failed self-check: {problems}"


def test_posterior_is_consistent_after_normal_update_passes():
    """A posterior after a normal Welford update passes."""
    p = ContinualLearningPosterior()
    p.track_correction_posteriors["hnerv_cluster"] = PerTrackPosterior(
        track_kind="seg"
    )
    p.track_correction_posteriors["hnerv_cluster"].update(1.17)
    p.track_correction_posteriors["hnerv_cluster"].update(1.15)
    p.source_rho_posteriors["hnerv_cluster"] = SourceRhoPosterior(
        architecture_class="hnerv"
    )
    p.source_rho_posteriors["hnerv_cluster"].update(0.5)
    ok, problems = p.is_consistent()
    assert ok, f"normal-update posterior failed self-check: {problems}"


def test_posterior_is_consistent_schema_mismatch_fails():
    """A posterior with stale schema fails."""
    p = ContinualLearningPosterior(schema="stale-schema-v0")
    ok, problems = p.is_consistent()
    assert not ok
    assert any("schema mismatch" in pr for pr in problems)


def test_posterior_is_consistent_negative_refused_count_fails():
    """A posterior with negative `refused_anchor_count` fails."""
    p = ContinualLearningPosterior(refused_anchor_count=-1)
    ok, problems = p.is_consistent()
    assert not ok
    assert any("refused_anchor_count" in pr for pr in problems)


def test_posterior_is_consistent_history_count_mismatch_fails():
    """`accepted_anchor_count` must match `accepted_anchor_history` len ±1."""
    p = ContinualLearningPosterior(
        accepted_anchor_count=10, accepted_anchor_history=[]
    )
    ok, problems = p.is_consistent()
    assert not ok
    assert any("accepted_anchor_count" in pr for pr in problems)


def test_posterior_is_consistent_nan_mean_correction_fails():
    """A NaN `mean_correction` (from a bad observation) fails."""
    p = ContinualLearningPosterior()
    p.track_correction_posteriors["bad"] = PerTrackPosterior(
        track_kind="seg", mean_correction=math.nan
    )
    ok, problems = p.is_consistent()
    assert not ok
    assert any("mean_correction" in pr and "not finite" in pr for pr in problems)


def test_posterior_is_consistent_negative_sum_squared_dev_fails():
    """Welford accumulator should never go negative."""
    p = ContinualLearningPosterior()
    p.track_correction_posteriors["bad"] = PerTrackPosterior(
        track_kind="seg", sum_squared_dev=-1.0
    )
    ok, problems = p.is_consistent()
    assert not ok
    assert any("sum_squared_dev" in pr and "negative" in pr for pr in problems)


def test_posterior_is_consistent_rho_outside_unit_interval_fails():
    """A `mean_rho` outside (-1, 1) is an invalid correlation."""
    p = ContinualLearningPosterior()
    p.source_rho_posteriors["bad"] = SourceRhoPosterior(
        architecture_class="hnerv", n_observations=5, mean_rho=1.5
    )
    ok, problems = p.is_consistent()
    assert not ok
    assert any("mean_rho" in pr and "outside" in pr for pr in problems)


def test_posterior_is_consistent_returns_problem_list_not_bool():
    """The contract returns (ok, problems_list) — not just a bool."""
    p = ContinualLearningPosterior(refused_anchor_count=-1, schema="bad")
    ok, problems = p.is_consistent()
    assert ok is False
    assert isinstance(problems, list)
    assert len(problems) >= 2  # both schema and refused_count
