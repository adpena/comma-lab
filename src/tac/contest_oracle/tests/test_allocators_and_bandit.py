# SPDX-License-Identifier: MIT
"""Tests for cell_allocator, pixel_budget_allocator, bandit_per_pair, per_pair_decomposition."""
from __future__ import annotations

import pytest

from tac.contest_oracle.bandit_per_pair import (
    BanditError,
    BetaBernoulliPosterior,
    PerPairBanditPlan,
    thompson_sample_per_pair_with_posterior,
    update_beta_bernoulli_posterior,
)
from tac.contest_oracle.cell_allocator import (
    CellAllocation,
    CellAllocatorError,
    SparseWaterFillingAllocation,
    sparse_water_fill,
)
from tac.contest_oracle.per_pair_decomposition import (
    CANONICAL_PER_PAIR_PLAN_AVAILABLE,
    thompson_sample_per_pair_assignment,
)
from tac.contest_oracle.pixel_budget_allocator import (
    PixelBudgetRecommendation,
    recommend_internal_resolution,
)


# ---------------------------------------------------------------------------
# pixel_budget_allocator
# ---------------------------------------------------------------------------
def test_recommend_internal_resolution_canonical_2x():
    """At budget=4.0 -> internal_multiple=2 (768x1024 = canonical sweet spot)."""
    rec = recommend_internal_resolution(compute_budget_relative=4.0)
    assert isinstance(rec, PixelBudgetRecommendation)
    assert rec.recommended_internal_multiple == 2
    assert rec.recommended_internal_height == 768
    assert rec.recommended_internal_width == 1024
    assert rec.scorer_response_match_quality == "IDEAL"


def test_recommend_internal_resolution_tight_budget_1x():
    """At budget=1.0 -> only 1x feasible -> UNDERSAMPLED."""
    rec = recommend_internal_resolution(compute_budget_relative=1.0)
    assert rec.recommended_internal_multiple == 1
    assert rec.scorer_response_match_quality == "UNDERSAMPLED"


def test_recommend_internal_resolution_huge_budget_saturates():
    """At budget=64+ -> internal_multiple=8 -> SATURATED."""
    rec = recommend_internal_resolution(compute_budget_relative=64.0)
    assert rec.recommended_internal_multiple == 8
    assert rec.scorer_response_match_quality == "SATURATED"


def test_recommend_internal_resolution_rejects_budget_lt_1():
    with pytest.raises(ValueError):
        recommend_internal_resolution(compute_budget_relative=0.5)


# ---------------------------------------------------------------------------
# cell_allocator
# ---------------------------------------------------------------------------
def test_sparse_water_fill_basic():
    """Top-3 cells with sensitivity 10, 5, 1; budget 1600 bits."""
    cells = [(0, 10.0), (100, 5.0), (200, 1.0), (300, 0.0), (400, -1.0)]
    res = sparse_water_fill(
        per_cell_sensitivity=cells,
        total_bits_budget=1600,
        min_bits_per_nonzero_cell=1,
    )
    assert isinstance(res, SparseWaterFillingAllocation)
    # Only positive cells eligible
    assert res.num_nonzero_cells == 3
    # Proportional: 10:5:1 -> 1000, 500, 100
    allocs = {a.cell_index: a.bits_allocated for a in res.allocations}
    assert allocs[0] >= allocs[100] >= allocs[200]
    # Total close to budget (may differ by int rounding floor)
    assert abs(res.total_bits_allocated - 1600) <= 3


def test_sparse_water_fill_empty_positive():
    """All-zero sensitivity -> no allocations."""
    res = sparse_water_fill(
        per_cell_sensitivity=[(0, 0.0), (1, 0.0)],
        total_bits_budget=1000,
    )
    assert res.num_nonzero_cells == 0
    assert res.total_bits_allocated == 0


def test_sparse_water_fill_negative_budget_rejected():
    with pytest.raises(CellAllocatorError):
        sparse_water_fill(per_cell_sensitivity=[(0, 1.0)], total_bits_budget=-1)


def test_sparse_water_fill_bad_min_bits_rejected():
    with pytest.raises(CellAllocatorError):
        sparse_water_fill(per_cell_sensitivity=[(0, 1.0)], total_bits_budget=1000, min_bits_per_nonzero_cell=0)


def test_sparse_water_fill_tight_budget_top1_only():
    """Budget too small for proportional top-K -> top-1 with min_bits."""
    cells = [(0, 10.0), (100, 5.0), (200, 1.0)]
    res = sparse_water_fill(
        per_cell_sensitivity=cells,
        total_bits_budget=8,
        min_bits_per_nonzero_cell=8,
    )
    # Top-1 gets at least min_bits
    assert res.num_nonzero_cells == 1


def test_sparse_water_fill_zero_budget():
    """Zero budget -> zero allocations (graceful)."""
    res = sparse_water_fill(
        per_cell_sensitivity=[(0, 10.0)],
        total_bits_budget=0,
    )
    assert res.num_nonzero_cells == 0


def test_sparse_water_fill_sparsity_ratio_tiny_for_realistic_case():
    """3 cells out of 588M -> ratio ~ 5e-9 (very sparse, canonical regime)."""
    cells = [(i, float(10 - i)) for i in range(3)]
    res = sparse_water_fill(
        per_cell_sensitivity=cells, total_bits_budget=300
    )
    assert res.sparsity_ratio < 1e-7
    assert res.total_cells_in_space == 589_824_000


# ---------------------------------------------------------------------------
# per_pair_decomposition (simple synthetic Thompson sampling)
# ---------------------------------------------------------------------------
def test_thompson_sample_per_pair_assignment_basic():
    plan = thompson_sample_per_pair_assignment(
        substrate_arms=("pr101", "hnerv", "nerv"),
        config_arms=("conf_a", "conf_b"),
        codec_arms=("brotli", "lzma"),
        num_pairs=10,
        random_seed=42,
    )
    assert plan.num_pairs == 10
    assert len(plan.per_pair_assignments) == 10
    for s, c, k in plan.per_pair_assignments:
        assert s in ("pr101", "hnerv", "nerv")
        assert c in ("conf_a", "conf_b")
        assert k in ("brotli", "lzma")


def test_thompson_sample_per_pair_default_num_pairs_600():
    plan = thompson_sample_per_pair_assignment(
        substrate_arms=("a",), config_arms=("b",), codec_arms=("c",),
    )
    assert plan.num_pairs == 600


def test_thompson_sample_per_pair_rejects_empty_arms():
    with pytest.raises(ValueError):
        thompson_sample_per_pair_assignment(
            substrate_arms=(), config_arms=("c",), codec_arms=("d",),
        )


def test_thompson_sample_per_pair_is_deterministic_with_seed():
    p1 = thompson_sample_per_pair_assignment(
        substrate_arms=("a", "b"), config_arms=("c",), codec_arms=("d",),
        num_pairs=5, random_seed=42,
    )
    p2 = thompson_sample_per_pair_assignment(
        substrate_arms=("a", "b"), config_arms=("c",), codec_arms=("d",),
        num_pairs=5, random_seed=42,
    )
    assert p1.per_pair_assignments == p2.per_pair_assignments


def test_canonical_per_pair_plan_available_flag_set():
    """The sister canonical helper should be importable."""
    assert CANONICAL_PER_PAIR_PLAN_AVAILABLE is True


# ---------------------------------------------------------------------------
# bandit_per_pair (Beta-Bernoulli + Thompson)
# ---------------------------------------------------------------------------
def test_beta_bernoulli_update_success():
    p0 = BetaBernoulliPosterior(arm_id="a", alpha=1.0, beta=1.0, num_observations=0)
    p1 = update_beta_bernoulli_posterior(posterior=p0, observed_success=True)
    assert p1.alpha == 2.0
    assert p1.beta == 1.0
    assert p1.num_observations == 1


def test_beta_bernoulli_update_failure():
    p0 = BetaBernoulliPosterior(arm_id="a", alpha=1.0, beta=1.0, num_observations=0)
    p1 = update_beta_bernoulli_posterior(posterior=p0, observed_success=False)
    assert p1.alpha == 1.0
    assert p1.beta == 2.0


def test_thompson_sample_with_posterior_basic():
    posteriors = [
        BetaBernoulliPosterior(arm_id="a", alpha=10.0, beta=2.0, num_observations=10),
        BetaBernoulliPosterior(arm_id="b", alpha=2.0, beta=10.0, num_observations=10),
    ]
    plan = thompson_sample_per_pair_with_posterior(
        posteriors=posteriors, num_pairs=100, random_seed=42,
    )
    assert isinstance(plan, PerPairBanditPlan)
    assert plan.num_pairs == 100
    # "a" has high posterior mean (~0.83); "b" has low (~0.17); plan should favor "a"
    a_count = sum(1 for asg in plan.assignments if asg.assigned_arm == "a")
    assert a_count > 80  # Strong preference for higher-mean arm


def test_thompson_sample_with_posterior_rejects_empty():
    with pytest.raises(BanditError):
        thompson_sample_per_pair_with_posterior(posteriors=(), num_pairs=10)


def test_thompson_sample_with_posterior_rejects_zero_pairs():
    with pytest.raises(BanditError):
        thompson_sample_per_pair_with_posterior(
            posteriors=[BetaBernoulliPosterior(arm_id="a", alpha=1, beta=1, num_observations=0)],
            num_pairs=0,
        )


def test_thompson_sample_with_posterior_is_deterministic():
    posteriors = [
        BetaBernoulliPosterior(arm_id="a", alpha=2.0, beta=2.0, num_observations=2),
        BetaBernoulliPosterior(arm_id="b", alpha=3.0, beta=3.0, num_observations=4),
    ]
    p1 = thompson_sample_per_pair_with_posterior(
        posteriors=posteriors, num_pairs=20, random_seed=12345,
    )
    p2 = thompson_sample_per_pair_with_posterior(
        posteriors=posteriors, num_pairs=20, random_seed=12345,
    )
    for a1, a2 in zip(p1.assignments, p2.assignments):
        assert a1.assigned_arm == a2.assigned_arm
        assert a1.sampled_reward == a2.sampled_reward
