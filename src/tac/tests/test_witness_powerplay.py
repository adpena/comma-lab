#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tests for tac.witness_dsl.powerplay — the POWERPLAY campaign primitives.

Certifies: (a) our contest S IS the POWERPLAY cost (powerplay_cost == compute_contest_score
+ correct decomposition); (b) the CorrectnessDemonstration is the executable axis-9 and
FAILS CLOSED on the exact surrogate grades that let #205 through (ancestor/proxy/MPS/
training-side/predicted) + on the OOM (peak RSS over ceiling) + on regression; (c)
variant_ii_accept implements c*_pred - c_new > eps; (d) simplest_unsolvable_rank orders by
improvement-per-(description+validation)-cost, byte-free facet-orthogonal first.
"""
import math

import pytest

from tac.contest_score import compute_contest_score
from tac.witness_dsl.powerplay import (
    CorrectnessDemonstration,
    DemonstrationLevel,
    EvidenceGrade,
    LeverCandidate,
    ScoredQuantity,
    powerplay_cost,
    simplest_unsolvable_rank,
    variant_ii_accept,
)


# --- 1. S IS the POWERPLAY cost -------------------------------------------------
def test_powerplay_cost_total_equals_canonical_score():
    c = powerplay_cost(0.001, 1e-4, 80_000)
    assert c.S == compute_contest_score(0.001, 1e-4, 80_000)


def test_powerplay_cost_decomposition_sums_to_total():
    c = powerplay_cost(0.003, 2e-4, 82_000)
    assert math.isclose(c.description_bits_term + c.task_deficit_term, c.S, rel_tol=1e-12)
    # L(s) is the rate term (bytes>0 -> positive); deficit is 100*d_seg + sqrt(10*d_pose).
    assert c.description_bits_term > 0
    assert math.isclose(c.task_deficit_term, 100 * 0.003 + math.sqrt(10 * 2e-4), rel_tol=1e-9)


# --- 3. Variant-II acceptance ---------------------------------------------------
def test_variant_ii_accept_improvement():
    ok, margin = variant_ii_accept(0.180, 0.191, eps=1e-5)
    assert ok is True
    assert math.isclose(margin, 0.011, rel_tol=1e-9)


def test_variant_ii_accept_regression():
    ok, margin = variant_ii_accept(0.195, 0.191, eps=1e-5)
    assert ok is False
    assert margin < 0


def test_variant_ii_accept_within_eps_is_rejected():
    ok, margin = variant_ii_accept(0.191, 0.191 + 5e-6, eps=1e-5)  # margin 5e-6 < eps
    assert ok is False


# --- 2. Correctness Demonstration: accept path ----------------------------------
def _sq(name, value, grade):
    return ScoredQuantity(name=name, value=value, grade=grade)


def _seal_demo(grade, *, peak_rss_mb=40_000.0, ceiling=90_000.0, predecessor_S=None):
    return CorrectnessDemonstration(
        label="t", level=DemonstrationLevel.LOCAL_SEAL,
        d_seg=_sq("d_seg", 0.003, grade),
        d_pose=_sq("d_pose", 2e-4, grade),
        archive_bytes=_sq("archive_bytes", 82_000, grade),
        predecessor_S=predecessor_S, peak_rss_mb=peak_rss_mb, rss_ceiling_mb=ceiling,
    )


def test_local_seal_accepts_advisory_through_decode():
    d = _seal_demo(EvidenceGrade.MACOS_CPU_ADVISORY_THROUGH_DECODE)
    assert d.validate() == []
    assert d.accepted is True


def test_local_seal_accepts_contest_authority():
    assert _seal_demo(EvidenceGrade.CONTEST_CPU).accepted
    assert _seal_demo(EvidenceGrade.CONTEST_CUDA).accepted


# --- 2b. Correctness Demonstration: FAIL-CLOSED on the #205 surrogate grades -----
@pytest.mark.parametrize("bad", [
    EvidenceGrade.ANCESTOR,      # the exact #205 pose bug: a borrowed-vehicle number
    EvidenceGrade.PROXY,
    EvidenceGrade.MPS,
    EvidenceGrade.TRAINING_SIDE,
    EvidenceGrade.PREDICTED,
])
def test_surrogate_grade_always_rejected(bad):
    d = _seal_demo(bad)
    viols = d.validate()
    assert d.accepted is False
    # one violation per scored quantity (d_seg, d_pose, archive_bytes)
    assert sum("SURROGATE" in v for v in viols) == 3


def test_promotion_refuses_advisory_but_seal_accepts_it():
    adv = EvidenceGrade.MACOS_CPU_ADVISORY_THROUGH_DECODE
    promo = CorrectnessDemonstration(
        label="p", level=DemonstrationLevel.PROMOTION,
        d_seg=_sq("d_seg", 0.003, adv), d_pose=_sq("d_pose", 2e-4, adv),
        archive_bytes=_sq("archive_bytes", 82_000, adv),
        predecessor_S=1.0,  # trivially improved so only the grade is under test
    )
    assert promo.accepted is False  # advisory is not contest authority
    assert _seal_demo(adv).accepted is True  # but a local seal accepts it


# --- 2c. Runnability (axis-9 (a)) -----------------------------------------------
def test_local_seal_requires_measured_rss():
    d = _seal_demo(EvidenceGrade.CONTEST_CPU, peak_rss_mb=None)
    assert any("runnability UNPROVEN" in v for v in d.validate())


def test_rss_over_ceiling_is_the_oom_class():
    d = _seal_demo(EvidenceGrade.CONTEST_CPU, peak_rss_mb=95_000.0, ceiling=90_000.0)
    assert any("runnability FAILS" in v and "OOM" in v for v in d.validate())


def test_rss_under_ceiling_ok():
    d = _seal_demo(EvidenceGrade.CONTEST_CPU, peak_rss_mb=55_000.0, ceiling=90_000.0)
    assert d.accepted


# --- 2d. No-regression (Variant-II) --------------------------------------------
def test_regression_rejected():
    d = _seal_demo(EvidenceGrade.CONTEST_CPU)
    worse_pred = d.new_S - 0.01  # predecessor is BETTER than new -> regression
    d2 = _seal_demo(EvidenceGrade.CONTEST_CPU, predecessor_S=worse_pred)
    assert any("REGRESSION" in v for v in d2.validate())


def test_improvement_over_predecessor_accepted():
    d = _seal_demo(EvidenceGrade.CONTEST_CPU)
    better_than_pred = d.new_S + 0.01  # predecessor is WORSE -> new improves
    d2 = _seal_demo(EvidenceGrade.CONTEST_CPU, predecessor_S=better_than_pred)
    assert d2.accepted


def test_first_solver_no_predecessor_no_regression_check():
    d = _seal_demo(EvidenceGrade.CONTEST_CPU, predecessor_S=None)
    assert d.accepted


def test_new_S_matches_canonical():
    d = _seal_demo(EvidenceGrade.CONTEST_CPU)
    assert d.new_S == compute_contest_score(0.003, 2e-4, 82_000)


# --- 2e. The #205 scenario end-to-end (multiple violations at once) --------------
def test_the_205_failure_is_rejected():
    # ancestor d_pose grade + peak RSS over ceiling = exactly what the #205 SEAL let through.
    d = CorrectnessDemonstration(
        label="#205", level=DemonstrationLevel.LOCAL_SEAL,
        d_seg=_sq("d_seg", 0.5, EvidenceGrade.TRAINING_SIDE),
        d_pose=_sq("d_pose", 3.4e-5, EvidenceGrade.ANCESTOR),  # the borrowed 3.4e-5
        archive_bytes=_sq("archive_bytes", 82_000, EvidenceGrade.TRAINING_SIDE),
        peak_rss_mb=90_300.0, rss_ceiling_mb=90_000.0,  # the OOM
    )
    viols = d.validate()
    assert d.accepted is False
    assert any("SURROGATE" in v for v in viols)
    assert any("OOM" in v for v in viols)


# --- 4. simplest_unsolvable_rank ------------------------------------------------
def test_rank_orders_by_improvement_per_cost():
    cands = [
        LeverCandidate("big_gain_costly", expected_delta_S=-0.02, description_bits=30_000, validation_cost=4.0),
        LeverCandidate("small_gain_free", expected_delta_S=-0.005, description_bits=0.0, validation_cost=1.0),
        LeverCandidate("no_gain", expected_delta_S=+0.001, description_bits=0.0, validation_cost=1.0),
    ]
    ranked = simplest_unsolvable_rank(cands, bit_weight=1e-4)
    labels = [c.label for c, _ in ranked]
    # non-improving is last; both improvers score > 0
    assert labels[-1] == "no_gain"
    assert all(s > 0 for c, s in ranked if c.label != "no_gain")


def test_byte_free_facet_orthogonal_lever_beats_same_gain_costly():
    cands = [
        LeverCandidate("free_orthogonal", expected_delta_S=-0.01, description_bits=0.0, validation_cost=1.0),
        LeverCandidate("costly_coupled", expected_delta_S=-0.01, description_bits=20_000, validation_cost=5.0),
    ]
    ranked = simplest_unsolvable_rank(cands, bit_weight=1e-4)
    assert ranked[0][0].label == "free_orthogonal"
    assert ranked[0][1] > ranked[1][1]


def test_rank_empty():
    assert simplest_unsolvable_rank([]) == []


def test_rank_all_non_improving_score_zero():
    cands = [LeverCandidate("a", 0.0), LeverCandidate("b", +0.1)]
    ranked = simplest_unsolvable_rank(cands)
    assert all(s == 0.0 for _, s in ranked)
