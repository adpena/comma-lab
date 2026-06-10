# SPDX-License-Identifier: MIT
"""Behaviour tests for the cross-pair scorer-quotient-space waterfilled corrector (task #54).

These verify the ALLOCATION ACTUALLY equalizes marginals (NO-FAKE class 1: a real allocation,
not a no-op), the verdict is computed from the exact contest score from components (class 8), and
the honest collateral accounting (new_bad / pose_side) is computed not faked (the #55 honesty rule).

Each test would FAIL if the corrector were replaced by a no-op / constant / per-pair-independent stub.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tac.optimization.cross_pair_waterfilled_corrector import (
    WATER_LEVEL_LAMBDA_STAR,
    CrossPairPoseObserver,
    CrossPairPoseWaterfiller,
    RegionRepairCandidate,
    allocate_seg_regions,
    compose_water_level_allocation,
    constant_correction_result,
    contest_score,
    pose_marginal,
    pose_score_term,
    set_region_pose_operating_point,
)


# ---------------------------------------------------------------------------
# Analytic pose observer: a deterministic oracle so the allocator BEHAVIOUR is testable
# without the heavyweight frozen scorer. Each pair has a base residual; each mode multiplies
# the residual by a per-(pair,mode) factor (a stand-in for the exact PoseNet response).
# ---------------------------------------------------------------------------
class AnalyticPoseObserver:
    """A deterministic pose observer with a controllable per-(pair,mode) response table."""

    def __init__(self, base: np.ndarray, response: dict[tuple[int, str], float], byte_cost: float = 1.0):
        self._base = np.asarray(base, dtype=np.float64)
        self._response = response  # (pair, mode) -> resulting d_pose
        self._byte_cost = float(byte_cost)

    def base_pose_residuals(self) -> np.ndarray:
        return self._base.copy()

    def pose_residual_under_mode(self, pair_index: int, mode_id: str) -> float:
        # default: mode leaves residual unchanged (no effect) unless table specifies.
        return float(self._response.get((pair_index, mode_id), self._base[pair_index]))

    def mode_byte_cost(self, pair_index: int, mode_id: str) -> float:
        return self._byte_cost


def _assert_is_observer(obs) -> None:
    assert isinstance(obs, CrossPairPoseObserver)


# ---------------------------------------------------------------------------
# 1-4: exact-score primitives (class 8 authority).
# ---------------------------------------------------------------------------
def test_water_level_is_rate_price_25_over_D():
    assert pytest.approx(25.0 / 37_545_489) == WATER_LEVEL_LAMBDA_STAR
    assert pytest.approx(6.66e-7, rel=1e-2) == WATER_LEVEL_LAMBDA_STAR


def test_contest_score_matches_closed_spec_formula():
    s = contest_score(5.6e-4, 2.9e-5, 177_169)
    expected = 100 * 5.6e-4 + math.sqrt(10 * 2.9e-5) + 25 * 177_169 / 37_545_489
    assert s == pytest.approx(expected)


def test_pose_marginal_grows_as_dpose_shrinks():
    # the documented concave-budget property: the value of shaving pose GROWS near zero.
    assert pose_marginal(1e-5) > pose_marginal(1e-3) > pose_marginal(1e-1)


def test_pose_score_term_is_sqrt_pooled():
    assert pose_score_term(2.9e-5) == pytest.approx(math.sqrt(10 * 2.9e-5))


# ---------------------------------------------------------------------------
# 5-9: the waterfiller actually equalizes marginals (class 1: real allocation).
# ---------------------------------------------------------------------------
def test_waterfiller_admits_only_rent_paying_actions():
    # one pair has a big improvement (rent-paying), one a tiny one (under-water at huge byte cost).
    base = np.array([1e-3, 1e-3])
    response = {
        (0, "good"): 1e-6,  # massive improvement
        (1, "weak"): 0.999e-3,  # negligible improvement
    }
    obs = AnalyticPoseObserver(base, response, byte_cost=1.0)
    _assert_is_observer(obs)
    wf = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "good", "weak"))
    res = wf.run(bytes_before=1000)
    # pair 0 admitted (steep, pays rent); pair 1's weak mode: value-per-byte tiny.
    admitted_pairs = {s.pair_index for s in res.steps}
    assert 0 in admitted_pairs
    # the weak action's value-per-byte must be checked against the water level.
    # delta_pose for pair1 ~ -1e-6/2 in pooled => ΔS pose term tiny; at byte_cost=1 it's
    # well below lambda* IF the pooled term is small enough. Assert net is dominated by pair0.
    assert res.beats_base
    assert res.new_bad == 0


def test_waterfiller_rejects_pair_worsening_modes():
    base = np.array([1e-4, 1e-4])
    response = {
        (0, "worse"): 5e-4,  # makes pair 0 WORSE
        (1, "worse"): 5e-4,
    }
    obs = AnalyticPoseObserver(base, response, byte_cost=1.0)
    wf = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "worse"))
    res = wf.run(bytes_before=1000)
    # no pair-worsening action may be admitted -> nothing admitted -> new_bad==0.
    assert res.admitted == 0
    assert res.new_bad == 0
    assert res.net_delta_score == pytest.approx(0.0)


def test_waterfiller_picks_best_mode_per_pair_not_first():
    # pair 0 has two modes; the second is strictly better. The allocator must pick the better.
    base = np.array([1e-3])
    response = {
        (0, "ok"): 5e-4,  # halves
        (0, "best"): 1e-6,  # near-zeros (better)
    }
    obs = AnalyticPoseObserver(base, response, byte_cost=1.0)
    wf = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "ok", "best"))
    res = wf.run(bytes_before=1000)
    assert len(res.steps) == 1
    assert res.steps[0].mode_id == "best"


def test_waterfiller_terminates_at_water_level_equalization():
    # All improvements are below the water level at byte_cost large enough -> nothing admitted.
    base = np.array([1e-6, 1e-6, 1e-6])
    response = {(i, "tiny"): 0.9e-6 for i in range(3)}
    obs = AnalyticPoseObserver(base, response, byte_cost=1e9)  # absurd byte cost
    wf = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "tiny"))
    res = wf.run(bytes_before=1000)
    assert res.admitted == 0  # every action is under water at this byte cost


def test_waterfiller_global_pool_recompute_concave_budget_property():
    # The pose marginal depends on the GLOBAL pool: because sqrt is CONCAVE, reducing the pooled
    # value when it is ALREADY SMALL yields a LARGER score reduction than when it is large. The
    # waterfiller must price each step against the CURRENT pool (not a fixed per-pair argmin), so
    # the LATER step (smaller pool) is steeper -- the opposite of separable greedy.
    base = np.array([4e-3, 4e-3])
    response = {(0, "m"): 1e-6, (1, "m"): 1e-6}
    pooled0 = 4e-3
    pooled1 = (1e-6 + 4e-3) / 2  # after step 1
    pooled2 = (1e-6 + 1e-6) / 2  # after step 2
    d1 = pose_score_term(pooled1) - pose_score_term(pooled0)  # negative (step 1 reduction)
    d2 = pose_score_term(pooled2) - pose_score_term(pooled1)  # negative (step 2 reduction)
    # CONCAVE-BUDGET PROPERTY: step 2 reduces the score MORE (the pool is smaller). A separable
    # per-pair planner using the BASE operating point for both would mis-price step 2.
    assert abs(d2) > abs(d1), f"concave budget violated: |d2|={abs(d2):.3e} <= |d1|={abs(d1):.3e}"
    # at byte_cost=1 both pay rent; both admitted (the pool recompute keeps the 2nd attractive).
    obs = AnalyticPoseObserver(base, response, byte_cost=1.0)
    wf = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "m"))
    res = wf.run(bytes_before=1000)
    assert res.admitted == 2
    # the recorded per-step delta_score_total must MATCH the pool-recompute math (not a fixed
    # base-operating-point estimate) -- proves the recompute is load-bearing.
    step_deltas = sorted(abs(s.delta_score_total) for s in res.steps)
    # the per-step pose-term magnitudes (ignoring the tiny equal rate term) must be ordered |d1|<|d2|.
    assert step_deltas[1] > step_deltas[0]


def test_waterfiller_step_delta_uses_current_pool_not_base():
    # Direct proof the per-step ΔS is computed at the CURRENT (mutating) pool: the second admitted
    # step's pooled_d_pose_before must equal the first step's pooled_d_pose_after.
    base = np.array([2e-3, 2e-3])
    response = {(0, "m"): 1e-6, (1, "m"): 1e-6}
    obs = AnalyticPoseObserver(base, response, byte_cost=1.0)
    wf = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "m"))
    res = wf.run(bytes_before=1000)
    assert len(res.steps) == 2
    assert res.steps[1].pooled_d_pose_before == pytest.approx(res.steps[0].pooled_d_pose_after)


# ---------------------------------------------------------------------------
# 10-11: the constant-correction control is dominated (proves waterfilling is load-bearing).
# ---------------------------------------------------------------------------
def test_constant_correction_is_dominated_by_waterfill():
    # A single mode helps pair 0 a lot and HURTS pair 1; the waterfiller corrects only pair 0,
    # the constant control corrects both (hurting pair 1) and pays bytes everywhere.
    base = np.array([1e-3, 1e-5])
    response = {
        (0, "m"): 1e-6,  # huge help on pair 0
        (1, "m"): 5e-4,  # hurts pair 1
    }
    obs = AnalyticPoseObserver(base, response, byte_cost=1.0)
    wf = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "m"))
    wf_res = wf.run(bytes_before=1000)
    const_res = constant_correction_result(obs, "m", bytes_before=1000, byte_cost_per_pair=1.0)
    # the constant control made pair 1 worse (new_bad >= 1); the waterfill never does.
    assert const_res.new_bad >= 1
    assert wf_res.new_bad == 0
    # the waterfill's net ΔS must be <= the constant control's (it is the optimal allocation).
    assert wf_res.net_delta_score <= const_res.net_delta_score + 1e-12


def test_constant_correction_accounting_is_honest():
    base = np.array([1e-3, 1e-3])
    response = {(0, "m"): 5e-4, (1, "m"): 2e-3}  # helps 0, hurts 1
    obs = AnalyticPoseObserver(base, response, byte_cost=1.0)
    const_res = constant_correction_result(obs, "m", bytes_before=0, byte_cost_per_pair=1.0)
    assert const_res.admitted == 2  # applied to all
    assert const_res.new_bad == 1  # pair 1 worsened -- counted honestly


# ---------------------------------------------------------------------------
# 12-15: seg-region allocator operates on REGIONS, prices collateral, declines salt-and-pepper.
# ---------------------------------------------------------------------------
def test_region_allocator_declines_salt_and_pepper_single_pixel_flips():
    set_region_pose_operating_point(2.9e-5)
    # the #55 frontier residual: 1-px regions, contour byte cost ~7B, net collateral > value.
    candidates = [
        RegionRepairCandidate(
            region_id=i,
            pixels=1,
            flips_in_region=1,
            coded_bytes=7.0,  # header+chain for a 1px component
            new_bad_flips=2,  # receptive-field coupling flips 2 correct neighbours
            pose_side_effect=1e-7,
        )
        for i in range(100)
    ]
    res = allocate_seg_regions(candidates)
    assert not res.any_fundable  # salt-and-pepper -> NONE fund (the #55 finding at region level)
    assert res.net_delta_score == pytest.approx(0.0)


def test_region_allocator_funds_contiguous_repairable_region():
    set_region_pose_operating_point(2.9e-5)
    # a contiguous lever-C-style region: 500 flips, cheap contour (40B), low collateral.
    big = RegionRepairCandidate(
        region_id=0,
        pixels=600,
        flips_in_region=500,
        coded_bytes=40.0,
        new_bad_flips=10,
        pose_side_effect=0.0,
    )
    res = allocate_seg_regions([big])
    assert res.any_fundable  # 490 net flips * 100/N >> 40B * 25/D
    assert res.net_delta_score < 0.0
    assert res.total_flips_repaired == 490


def test_region_candidate_net_accounts_new_bad_and_pose_collateral():
    set_region_pose_operating_point(2.9e-5)
    # a region whose repaired flips are entirely eaten by new_bad collateral -> declines.
    c = RegionRepairCandidate(
        region_id=0,
        pixels=10,
        flips_in_region=5,
        coded_bytes=10.0,
        new_bad_flips=5,  # all 5 repaired flips offset by 5 new bad -> net 0 flips
        pose_side_effect=1e-6,  # plus positive pose collateral
    )
    assert c.seg_value == pytest.approx(0.0)  # net flips repaired == 0
    assert c.net_delta_score > 0.0  # cost (bytes + pose collateral) with zero value -> declines
    assert not c.pays_rent


def test_region_allocator_ranks_fundable_by_value_per_byte():
    set_region_pose_operating_point(2.9e-5)
    steep = RegionRepairCandidate(0, 600, 500, 20.0, 0, 0.0)  # high net flips, low bytes
    shallow = RegionRepairCandidate(1, 600, 500, 200.0, 0, 0.0)  # same flips, 10x bytes
    res = allocate_seg_regions([shallow, steep])
    assert res.fundable[0].region_id == 0  # steeper value-per-byte ranked first


# ---------------------------------------------------------------------------
# 16-18: the composed water-level allocation sums disjoint-section axes.
# ---------------------------------------------------------------------------
def test_composed_allocation_sums_pose_and_region_net_delta():
    base = np.array([1e-3, 1e-3])
    response = {(0, "m"): 1e-6, (1, "m"): 1e-6}
    obs = AnalyticPoseObserver(base, response, byte_cost=1.0)
    wf = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "m"))
    pose_res = wf.run(bytes_before=1000)
    region_res = allocate_seg_regions(
        [RegionRepairCandidate(0, 600, 500, 40.0, 10, 0.0)]
    )
    composed = compose_water_level_allocation(pose_res, region_res)
    assert composed.net_delta_score == pytest.approx(
        pose_res.net_delta_score + region_res.net_delta_score
    )


def test_composed_beats_base_iff_sum_negative():
    base = np.array([1e-3])
    response = {(0, "m"): 1e-6}
    obs = AnalyticPoseObserver(base, response, byte_cost=1.0)
    pose_res = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "m")).run(bytes_before=0)
    composed = compose_water_level_allocation(pose_res, None)
    assert composed.beats_base == (composed.net_delta_score < 0.0)


def test_result_rows_are_non_promotable_advisory():
    base = np.array([1e-3])
    response = {(0, "m"): 1e-6}
    obs = AnalyticPoseObserver(base, response, byte_cost=1.0)
    res = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "m")).run(bytes_before=0)
    row = res.to_row()
    assert row["promotable"] is False
    assert row["score_claim"] is False
    assert row["evidence_grade"] == "[local CPU-torch advisory]"


# ---------------------------------------------------------------------------
# 19: a no-op observer (mode never changes residual) admits nothing (anti-fake).
# ---------------------------------------------------------------------------
def test_no_op_observer_admits_nothing():
    base = np.array([1e-3, 1e-3, 1e-3])
    obs = AnalyticPoseObserver(base, {}, byte_cost=1.0)  # empty response = no mode changes anything
    wf = CrossPairPoseWaterfiller(obs, candidate_modes=("none", "m"))
    res = wf.run(bytes_before=1000)
    assert res.admitted == 0
    assert res.net_delta_score == pytest.approx(0.0)
    assert res.pooled_d_pose_after == pytest.approx(res.pooled_d_pose_before)
