# SPDX-License-Identifier: MIT
"""Synthetic positive-control tests for the Amari cross-term diagnostic.

The controls prove the tool MEASURES what it claims: the canary must stay QUIET on a
dual-orthogonal (additive) allocation and FIRE on an interacting one, with the cross-term
matching the KNOWN coupling / inner product exactly.
"""
from __future__ import annotations

import math

from tac.losses.variable_level_waterfill_allocator import solve_waterfill_allocation
from tac.optimization.pythagorean_crossterm import (
    additivity_canary,
    bilinear_synthetic_oracle,
    diagnose_waterfill_additivity,
    measure_pairwise_crossterms,
    quadratic_support_oracle,
    rd_table_additive_oracle,
)

# --------------------------------------------------------------------------------------
# Control (a): dual-orthogonal pair -> cross ~= 0, canary QUIET.
# --------------------------------------------------------------------------------------


def test_orthogonal_pair_has_zero_crosstern_and_quiet_canary() -> None:
    # Two independent components: solo distortions, NO coupling.
    oracle = bilinear_synthetic_oracle({"A": 0.30, "B": 0.20}, couplings={})
    report = measure_pairwise_crossterms(["A", "B"], oracle)

    pair = report.pair("A", "B")
    assert pair is not None
    assert pair.delta_a == 0.30
    assert pair.delta_b == 0.20
    assert pair.delta_ab == 0.50  # exactly additive
    assert abs(pair.cross) <= 1e-12
    assert pair.regime == "orthogonal"
    assert report.additivity_error_fraction == 0.0

    fired, msg = additivity_canary(report)
    assert fired is False
    assert "QUIET" in msg


def test_orthogonal_support_vectors_zero_cross() -> None:
    # cross(a,b) = 2<v_a,v_b>; orthogonal support vectors -> cross exactly 0.
    oracle = quadratic_support_oracle({"A": (1.0, 0.0), "B": (0.0, 1.0)})
    report = measure_pairwise_crossterms(["A", "B"], oracle)
    pair = report.pair("A", "B")
    assert pair is not None
    assert math.isclose(pair.cross, 0.0, abs_tol=1e-12)
    assert pair.regime == "orthogonal"


# --------------------------------------------------------------------------------------
# Control (b): interacting pair -> cross != 0 with correct sign/magnitude, canary FIRES.
# --------------------------------------------------------------------------------------


def test_bilinear_coupling_recovers_exact_crosstern_superadditive() -> None:
    # A positive coupling => joint distortion EXCEEDS the sum => superadditive (cross>0).
    oracle = bilinear_synthetic_oracle(
        {"A": 0.30, "B": 0.20}, couplings={frozenset(("A", "B")): 0.07}
    )
    report = measure_pairwise_crossterms(["A", "B"], oracle)
    pair = report.pair("A", "B")
    assert pair is not None
    assert math.isclose(pair.cross, 0.07, abs_tol=1e-12)  # cross == the known coupling
    assert pair.regime == "superadditive"
    assert pair.additive_prediction == 0.50
    assert math.isclose(pair.delta_ab, 0.57, abs_tol=1e-12)

    fired, msg = additivity_canary(report, rel_tol=0.05)
    assert fired is True
    assert "FIRED" in msg
    # error fraction = |0.07| / (0.30 + 0.20)
    assert math.isclose(report.additivity_error_fraction, 0.07 / 0.50, abs_tol=1e-12)


def test_negative_coupling_is_subadditive() -> None:
    oracle = bilinear_synthetic_oracle(
        {"A": 0.30, "B": 0.20}, couplings={frozenset(("A", "B")): -0.06}
    )
    report = measure_pairwise_crossterms(["A", "B"], oracle)
    pair = report.pair("A", "B")
    assert pair is not None
    assert math.isclose(pair.cross, -0.06, abs_tol=1e-12)
    assert pair.regime == "subadditive"
    fired, _ = additivity_canary(report, rel_tol=0.05)
    assert fired is True


def test_aligned_support_vectors_superadditive_equals_two_inner_product() -> None:
    # v_a=(1,1), v_b=(2,0) -> 2<v_a,v_b> = 2*(1*2 + 1*0) = 4.0
    oracle = quadratic_support_oracle({"A": (1.0, 1.0), "B": (2.0, 0.0)})
    report = measure_pairwise_crossterms(["A", "B"], oracle)
    pair = report.pair("A", "B")
    assert pair is not None
    assert math.isclose(pair.cross, 4.0, abs_tol=1e-12)
    assert pair.regime == "superadditive"


# --------------------------------------------------------------------------------------
# Mixed multi-component: the canary localises the worst interactor.
# --------------------------------------------------------------------------------------


def test_worst_pair_localisation_in_mixed_allocation() -> None:
    oracle = bilinear_synthetic_oracle(
        {"A": 0.10, "B": 0.10, "C": 0.10, "D": 0.10},
        couplings={
            frozenset(("A", "B")): 0.001,  # weak
            frozenset(("C", "D")): 0.050,  # strong -> should be worst
        },
    )
    report = measure_pairwise_crossterms(["A", "B", "C", "D"], oracle, worst_k=2)
    assert len(report.pairs) == 6
    worst = report.worst_pairs[0]
    assert frozenset((worst.a, worst.b)) == frozenset(("C", "D"))
    assert math.isclose(worst.cross, 0.050, abs_tol=1e-12)
    assert report.n_interacting_pairs == 2
    # oracle calls: baseline(1) + singles(4) + pairs(6) = 11 distinct sets.
    assert report.n_oracle_calls == 11


# --------------------------------------------------------------------------------------
# NULL control: the waterfill's OWN rd_table oracle is additive-by-construction -> cross~=0.
# --------------------------------------------------------------------------------------


def test_rd_table_additive_oracle_is_pythagorean_exact() -> None:
    # Two tensors, each with a small RD curve (127 baseline + one coarser level).
    rd_table = {
        "t0": {127: (0.0, 0.0), 64: (500.0, 0.004)},
        "t1": {127: (0.0, 0.0), 64: (400.0, 0.003)},
    }
    levels = {"t0": 64, "t1": 64}
    oracle = rd_table_additive_oracle(rd_table, levels)
    report = measure_pairwise_crossterms(["t0", "t1"], oracle)
    pair = report.pair("t0", "t1")
    assert pair is not None
    # additive-by-construction => zero cross, quiet canary => the dual-orthogonal NULL.
    assert abs(pair.cross) <= 1e-12
    assert report.additivity_error_fraction == 0.0
    fired, _ = additivity_canary(report)
    assert fired is False


def test_diagnose_waterfill_additivity_null_residual_matches_solver() -> None:
    # Build a realized allocation from the real #157 solver, then check the additive
    # rd_table oracle reproduces the solver's total_dist_cost with ZERO residual.
    rd_table = {
        "t0": {127: (0.0, 0.0), 64: (800.0, 0.0020)},
        "t1": {127: (0.0, 0.0), 64: (700.0, 0.0018)},
        "t2": {127: (0.0, 0.0), 64: (50.0, 0.0100)},  # expensive per byte -> likely 127
    }
    alloc = solve_waterfill_allocation(rd_table)
    oracle = rd_table_additive_oracle(rd_table, alloc.levels)
    diag = diagnose_waterfill_additivity(rd_table, alloc.levels, oracle)
    # additive oracle == the solver's own model => residual ~ 0, trustworthy.
    assert math.isclose(diag.residual, 0.0, abs_tol=1e-12)
    assert math.isclose(
        diag.waterfill_additive_dist_cost, diag.true_joint_delta, abs_tol=1e-12
    )
    assert diag.trustworthy is True


def test_diagnose_waterfill_additivity_fires_on_interacting_oracle() -> None:
    # Same allocation, but a REAL-style oracle that couples the two coarsened tensors:
    # residual != 0 and the canary FIRES (localising the interaction).
    rd_table = {
        "t0": {127: (0.0, 0.0), 64: (800.0, 0.0020)},
        "t1": {127: (0.0, 0.0), 64: (700.0, 0.0018)},
    }
    alloc = solve_waterfill_allocation(rd_table)
    coarsened = [t for t, lv in alloc.levels.items() if lv < 127]
    # only meaningful if both coarsened; construct a coupling oracle over them.
    linear = {t: rd_table[t][alloc.levels[t]][1] for t in coarsened}
    couplings = (
        {frozenset(coarsened[:2]): 0.005} if len(coarsened) >= 2 else {}
    )
    oracle = bilinear_synthetic_oracle(linear, couplings=couplings)
    diag = diagnose_waterfill_additivity(rd_table, alloc.levels, oracle)
    if len(coarsened) >= 2:
        assert math.isclose(diag.residual, 0.005, abs_tol=1e-12)
        assert diag.canary_fired is True
        assert not diag.trustworthy


def test_expensive_oracle_is_cached_once_per_set() -> None:
    calls: list[frozenset] = []

    def counting_oracle(active: frozenset) -> float:
        calls.append(active)
        return math.fsum({"A": 1.0, "B": 2.0, "C": 3.0}.get(c, 0.0) for c in active)

    report = measure_pairwise_crossterms(["A", "B", "C"], counting_oracle)
    # distinct sets = baseline + 3 singles + 3 pairs = 7; no set evaluated twice.
    assert len(calls) == 7
    assert report.n_oracle_calls == 7
    assert len(set(calls)) == len(calls)
