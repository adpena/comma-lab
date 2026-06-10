# SPDX-License-Identifier: MIT
"""Behavior tests for the unified scorer-quotient candidate schema + firewall.

Every test verifies BEHAVIOR (the firewall actually blocks, the ladder actually
gates, the score is actually recomputed) — replacing the logic with a constant
or no-op fails these.
"""
from __future__ import annotations

import math

import pytest

from tac.optimization.scorer_quotient_candidate_row import (
    RATE_DENOM,
    LegalFrameFeasibilityTrace,
    SchemaError,
    ScorerQuotientCandidateRow,
    promotable,
    rank_candidates,
    recompute_score,
)


def _row(**kw):
    base = {
        "lever_id": "#69",
        "candidate_kind": "requant",
        "base_archive_sha256": "deadbeef",
        "bytes_before": 177169,
        "bytes_after": 177169,
        "d_seg_before": 5.6e-4,
        "d_seg_after": 5.6e-4,
        "d_pose_before": 2.9e-5,
        "d_pose_after": 2.9e-5,
        "authority_tier": "contest_cpu",
        "metric_family": "exact_evaluate",
        "decision": "continue",
    }
    base.update(kw)
    return ScorerQuotientCandidateRow(**base)


def test_recompute_score_matches_formula():
    s = recompute_score(5.6e-4, 2.9e-5, 177169)
    expected = 100 * 5.6e-4 + math.sqrt(10 * 2.9e-5) + 25 * 177169 / RATE_DENOM
    assert s == pytest.approx(expected)
    # frontier sanity: ~0.191
    assert 0.190 < s < 0.192


def test_score_is_recomputed_not_trusted():
    # The row computes score_after from components; a smaller archive lowers it.
    r = _row(bytes_after=130000)
    assert r.score_after < r.score_before
    assert r.delta_score_total == pytest.approx(r.score_after - r.score_before)
    # archive_bytes_delta is derived
    assert r.archive_bytes_delta == 130000 - 177169


def test_firewall_blocks_advisory_even_with_negative_delta():
    # A local-CPU-advisory row that lowers S is NOT pointer-eligible.
    r = _row(bytes_after=120000, authority_tier="exact_cpu_advisory",
             metric_family="exact_pair_scorer")
    assert r.delta_score_total < 0
    assert r.pointer_update_eligible is False


def test_firewall_blocks_mlx_proxy():
    r = _row(bytes_after=100000, authority_tier="telemetry_proxy",
             metric_family="scorer_proxy")
    assert r.delta_score_total < 0
    assert r.pointer_update_eligible is False


def test_firewall_promotes_contest_exact_negative_delta():
    r = _row(bytes_after=130000, authority_tier="contest_cpu",
             metric_family="exact_evaluate")
    assert r.delta_score_total < 0
    assert r.pointer_update_eligible is True


def test_firewall_blocks_contest_with_positive_delta():
    # A contest row that RAISES the score must not promote.
    r = _row(bytes_after=200000, authority_tier="contest_cuda",
             metric_family="exact_evaluate")
    assert r.delta_score_total > 0
    assert r.pointer_update_eligible is False


def test_firewall_requires_exact_evaluate_metric():
    # contest_cpu authority but pair-scorer metric (not full evaluate) -> blocked.
    r = _row(bytes_after=130000, authority_tier="contest_cpu",
             metric_family="exact_pair_scorer")
    assert r.pointer_update_eligible is False


def test_net_repaired_flips_is_honest():
    r = _row(repaired_flips=421, new_bad_flips=786)
    assert r.net_repaired_flips == 421 - 786  # the #55 lesson: net is negative
    assert r.net_repaired_flips < 0


def test_net_repaired_none_when_unmeasured():
    r = _row()
    assert r.net_repaired_flips is None


def test_unknown_candidate_kind_rejected():
    with pytest.raises(SchemaError):
        _row(candidate_kind="magic")


def test_unknown_authority_rejected():
    with pytest.raises(SchemaError):
        _row(authority_tier="mps")  # MPS is never an authority


def test_unknown_metric_rejected():
    with pytest.raises(SchemaError):
        _row(metric_family="vibes")


def test_negative_components_rejected():
    with pytest.raises(SchemaError):
        _row(d_seg_after=-1.0)


def test_rank_candidates_sorts_by_delta_score():
    a = _row(bytes_after=170000)   # small save
    b = _row(bytes_after=120000)   # big save
    c = _row(bytes_after=180000)   # loss
    ranked = rank_candidates([a, b, c])
    assert ranked[0] is b and ranked[-1] is c


def test_promotable_filters_to_eligible_only():
    good = _row(bytes_after=120000, authority_tier="contest_cpu", metric_family="exact_evaluate")
    advisory = _row(bytes_after=100000, authority_tier="exact_cpu_advisory", metric_family="exact_pair_scorer")
    rows = promotable([advisory, good])
    assert rows == [good]


def test_feasibility_ladder_scorer_effect_requires_exact():
    with pytest.raises(SchemaError):
        LegalFrameFeasibilityTrace(
            base_candidate="palette", basis_used="resize_null",
            constraints_projected=("margin", "pose"), projection_residual=0.01,
            margin_constraints_satisfied=True, pose_tube_surrogate_satisfied=True,
            exact_d_seg_after=None, exact_d_pose_after=None, bytes_estimate=None,
            decision="scorer_effect",
        )


def test_feasibility_ladder_byte_real_requires_bytes():
    with pytest.raises(SchemaError):
        LegalFrameFeasibilityTrace(
            base_candidate="palette", basis_used="resize_null",
            constraints_projected=("margin", "pose"), projection_residual=0.01,
            margin_constraints_satisfied=True, pose_tube_surrogate_satisfied=True,
            exact_d_seg_after=5e-4, exact_d_pose_after=3e-5, bytes_estimate=None,
            decision="byte_real",
        )


def test_feasibility_projection_only_allowed_without_exact():
    t = LegalFrameFeasibilityTrace(
        base_candidate="palette", basis_used="resize_null",
        constraints_projected=("margin",), projection_residual=0.5,
        margin_constraints_satisfied=False, pose_tube_surrogate_satisfied=False,
        exact_d_seg_after=None, exact_d_pose_after=None, bytes_estimate=None,
        decision="projection_only",
    )
    assert t.is_exact_candidate is False


def test_feasibility_exact_candidate_full_evidence():
    t = LegalFrameFeasibilityTrace(
        base_candidate="lowres_gt", basis_used="resize_null+lowrank",
        constraints_projected=("margin", "pose", "cheap"), projection_residual=1e-4,
        margin_constraints_satisfied=True, pose_tube_surrogate_satisfied=True,
        exact_d_seg_after=5e-4, exact_d_pose_after=3e-5, bytes_estimate=120000,
        decision="exact_candidate",
    )
    assert t.is_exact_candidate is True
