# SPDX-License-Identifier: MIT
"""Tests for ``tac.fec6_selector_discovery_sensitivity_weighted`` (Ext 3 of
fec6 stacking wave).

Design memo: ``.omx/research/fec6_plus_sensitivity_weighted_discovery_design_20260517.md``
Lane: ``lane_fec6_stacking_wave_5_grammar_extensions_20260517``

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287
evidence-tag discipline: all score claims in these tests are
``[predicted, theoretical]``; no contest-CPU / contest-CUDA claims.
"""
from __future__ import annotations

import pytest

from tac.fec6_selector_discovery_sensitivity_weighted import (
    compute_weighted_per_pair_scores,
    pick_weighted_per_pair_modes,
    reweight_per_pair_candidate_table,
)
from tac.sensitivity_map.axis_weights import (
    PR106_R2_FRONTIER_AXIS_WEIGHTS,
    AxisWeights,
)


@pytest.fixture
def baseline_axis_weights() -> AxisWeights:
    """Unweighted baseline; recovers the canonical per-pair scoring."""
    return AxisWeights(
        pose=1.0,
        seg=1.0,
        rate=1.0,
        mixed=1.0,
        operating_point_tag="baseline_unweighted_test",
        basis="uniform 1.0 per axis; test fixture",
    )


@pytest.fixture
def pose_dominant_pair() -> list[dict[str, float]]:
    """3 candidate modes; mode 0 wins on pose, mode 2 wins on seg, mode 1 wins on uniform sum."""
    return [
        {"delta_d_pose": 0.001, "delta_d_seg": 0.020},  # pose-best, seg-worst
        {"delta_d_pose": 0.010, "delta_d_seg": 0.010},  # uniform-sum-best
        {"delta_d_pose": 0.020, "delta_d_seg": 0.001},  # pose-worst, seg-best
    ]


def test_compute_weighted_per_pair_scores_baseline_recovers_uniform_sum(
    baseline_axis_weights, pose_dominant_pair
):
    """With uniform 1.0 weights, the score is plain sum of (Δd_pose + Δd_seg)."""
    scores = compute_weighted_per_pair_scores(
        pair_id=0,
        candidate_modes=pose_dominant_pair,
        axis_weights=baseline_axis_weights,
    )
    assert scores == [0.021, 0.020, 0.021]
    # argmin is mode 1 (uniform-sum-best)
    assert min(range(3), key=lambda k: scores[k]) == 1


def test_compute_weighted_per_pair_scores_pr106_r2_picks_pose_dominant_mode(
    pose_dominant_pair,
):
    """With PR106 r2 frontier weights (pose=2.71, seg=1.00), mode 0 wins."""
    scores = compute_weighted_per_pair_scores(
        pair_id=0,
        candidate_modes=pose_dominant_pair,
        axis_weights=PR106_R2_FRONTIER_AXIS_WEIGHTS,
    )
    # mode 0: 2.71 * 0.001 + 1.0 * 0.020 = 0.00271 + 0.020 = 0.02271
    # mode 1: 2.71 * 0.010 + 1.0 * 0.010 = 0.0271 + 0.010  = 0.0371
    # mode 2: 2.71 * 0.020 + 1.0 * 0.001 = 0.0542 + 0.001  = 0.0552
    assert scores[0] == pytest.approx(0.02271)
    assert scores[1] == pytest.approx(0.0371)
    assert scores[2] == pytest.approx(0.0552)
    # argmin is mode 0 (pose-best); divergence from unweighted argmin=mode 1
    assert min(range(3), key=lambda k: scores[k]) == 0


def test_compute_weighted_per_pair_scores_empty_modes_rejected():
    with pytest.raises(ValueError, match="candidate_modes is empty"):
        compute_weighted_per_pair_scores(
            pair_id=0,
            candidate_modes=[],
            axis_weights=PR106_R2_FRONTIER_AXIS_WEIGHTS,
        )


def test_compute_weighted_per_pair_scores_missing_key_rejected():
    with pytest.raises(KeyError, match="missing required key"):
        compute_weighted_per_pair_scores(
            pair_id=42,
            candidate_modes=[{"delta_d_pose": 0.001}],  # missing delta_d_seg
            axis_weights=PR106_R2_FRONTIER_AXIS_WEIGHTS,
        )


def test_pick_weighted_per_pair_modes_baseline_picks_uniform_sum_argmin(
    baseline_axis_weights, pose_dominant_pair
):
    per_pair = [pose_dominant_pair, pose_dominant_pair]  # 2 pairs same modes
    chosen = pick_weighted_per_pair_modes(
        per_pair_candidate_modes=per_pair,
        axis_weights=baseline_axis_weights,
    )
    assert chosen == [1, 1]  # mode 1 is uniform-sum-best for both


def test_pick_weighted_per_pair_modes_pr106_r2_picks_pose_dominant(
    pose_dominant_pair,
):
    per_pair = [pose_dominant_pair, pose_dominant_pair]
    chosen = pick_weighted_per_pair_modes(
        per_pair_candidate_modes=per_pair,
        axis_weights=PR106_R2_FRONTIER_AXIS_WEIGHTS,
    )
    assert chosen == [0, 0]  # mode 0 is pose-best under PR106 r2 weighting


def test_pick_weighted_per_pair_modes_empty_rejected():
    with pytest.raises(ValueError, match="per_pair_candidate_modes is empty"):
        pick_weighted_per_pair_modes(
            per_pair_candidate_modes=[],
            axis_weights=PR106_R2_FRONTIER_AXIS_WEIGHTS,
        )


def test_reweight_per_pair_candidate_table_emits_full_manifest(pose_dominant_pair):
    per_pair = [pose_dominant_pair] * 5  # 5 identical pairs
    result = reweight_per_pair_candidate_table(
        per_pair_candidate_modes=per_pair,
        axis_weights=PR106_R2_FRONTIER_AXIS_WEIGHTS,
    )
    assert set(result.keys()) == {
        "chosen_per_pair_indices",
        "per_pair_weighted_scores",
        "axis_weights_evidence_tag",
        "per_pair_unweighted_indices",
        "per_pair_divergence_count",
        "n_pairs",
        "score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_provider_dispatch",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "evidence_tag",
    }
    assert result["n_pairs"] == 5
    assert result["chosen_per_pair_indices"] == [0, 0, 0, 0, 0]  # pose-dominant
    assert result["per_pair_unweighted_indices"] == [1, 1, 1, 1, 1]  # uniform-sum
    assert result["per_pair_divergence_count"] == 5  # all 5 pairs diverge
    assert result["score_claim"] is False  # diagnostic-only
    assert result["promotion_eligible"] is False
    assert result["rank_or_kill_eligible"] is False
    assert result["ready_for_provider_dispatch"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["dispatch_attempted"] is False
    assert result["axis_weights_evidence_tag"].startswith("[axis_weights v1;")
    assert "pr106_r2_frontier" in result["axis_weights_evidence_tag"].lower()


def test_reweight_per_pair_candidate_table_unweighted_baseline_zero_divergence(
    baseline_axis_weights, pose_dominant_pair
):
    """Unweighted baseline shows zero divergence (sanity check)."""
    per_pair = [pose_dominant_pair] * 3
    result = reweight_per_pair_candidate_table(
        per_pair_candidate_modes=per_pair,
        axis_weights=baseline_axis_weights,
    )
    assert result["per_pair_divergence_count"] == 0
    assert result["chosen_per_pair_indices"] == result["per_pair_unweighted_indices"]


def test_pr106_r2_frontier_axis_weights_canonical_values():
    """Regression guard: the canonical PR106 r2 frontier weights are pinned."""
    assert PR106_R2_FRONTIER_AXIS_WEIGHTS.pose == pytest.approx(2.71)
    assert PR106_R2_FRONTIER_AXIS_WEIGHTS.seg == pytest.approx(1.00)
    assert PR106_R2_FRONTIER_AXIS_WEIGHTS.operating_point_tag == "pr106_r2_frontier"


def test_evidence_tag_propagates_per_apples_to_apples_discipline(pose_dominant_pair):
    """Per CLAUDE.md 'Apples-to-apples evidence discipline', the operating
    point tag MUST appear in the artifact's evidence tag."""
    per_pair = [pose_dominant_pair]
    result = reweight_per_pair_candidate_table(
        per_pair_candidate_modes=per_pair,
        axis_weights=PR106_R2_FRONTIER_AXIS_WEIGHTS,
    )
    tag = result["axis_weights_evidence_tag"]
    assert "operating_point=pr106_r2_frontier" in tag
    assert "basis=" in tag
    assert tag.startswith("[axis_weights v1;")
    assert tag.endswith("]")


@pytest.mark.parametrize(
    ("bad_key", "bad_value"),
    [
        ("delta_d_pose", float("nan")),
        ("delta_d_pose", float("inf")),
        ("delta_d_seg", float("-inf")),
    ],
)
def test_nonfinite_candidate_delta_rejected_before_argmin(bad_key, bad_value):
    """NaN/inf deltas must not silently win or lose an argmin decision."""
    candidate = {"delta_d_pose": 0.001, "delta_d_seg": 0.001}
    candidate[bad_key] = bad_value
    with pytest.raises(ValueError, match=f"non-finite {bad_key}"):
        compute_weighted_per_pair_scores(
            pair_id=17,
            candidate_modes=[candidate],
            axis_weights=PR106_R2_FRONTIER_AXIS_WEIGHTS,
        )


def test_per_pair_independence_holds(baseline_axis_weights):
    """Sanity: per-pair independence — picking for pair 0 doesn't affect pair 1's pick."""
    # Two pairs with different optimal modes
    pair_a = [
        {"delta_d_pose": 0.001, "delta_d_seg": 0.001},  # all-zero-best
        {"delta_d_pose": 0.999, "delta_d_seg": 0.999},
    ]
    pair_b = [
        {"delta_d_pose": 0.999, "delta_d_seg": 0.999},
        {"delta_d_pose": 0.001, "delta_d_seg": 0.001},  # mode 1 best
    ]
    chosen = pick_weighted_per_pair_modes(
        per_pair_candidate_modes=[pair_a, pair_b],
        axis_weights=baseline_axis_weights,
    )
    assert chosen == [0, 1]
