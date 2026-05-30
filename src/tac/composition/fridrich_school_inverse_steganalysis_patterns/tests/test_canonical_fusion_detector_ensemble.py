# SPDX-License-Identifier: MIT
"""Tests for canonical fusion detector ensemble (Yousfi autostego)."""

from __future__ import annotations

import math

import pytest

from tac.composition.fridrich_school_inverse_steganalysis_patterns import (
    CONTEST_FUSION_WEIGHTS_CANONICAL,
    FusionConfig,
    FusionDetectorError,
    FusionStrategy,
    compute_canonical_contest_fusion_weights,
    compute_fusion_score,
)


def test_canonical_contest_fusion_weights_match_claude_md() -> None:
    """Per CLAUDE.md 'Exact scorer architectures' verbatim:
    S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37545489.
    """
    weights = compute_canonical_contest_fusion_weights()
    assert weights["d_seg"] == 100.0
    assert weights["d_pose_sqrt_10x"] == 1.0
    assert weights["rate_25_over_denom"] == 1.0
    assert weights == dict(CONTEST_FUSION_WEIGHTS_CANONICAL)


def test_linear_weighted_sum_canonical_matches_dot_product() -> None:
    """LINEAR_WEIGHTED_SUM = sum_i w_i * score_i (canonical Yousfi fusion)."""
    detector_scores = {"d_seg": 0.001, "d_pose_sqrt_10x": 0.02, "rate_25_over_denom": 0.10}
    cfg = FusionConfig()  # canonical defaults: contest weights + LINEAR_WEIGHTED_SUM
    score = compute_fusion_score(detector_scores, cfg)
    # 100 * 0.001 + 1 * 0.02 + 1 * 0.10 = 0.22
    assert score == pytest.approx(0.22)


def test_geometric_mean_canonical() -> None:
    """GEOMETRIC_MEAN = exp(sum_i w_i * log(score_i))."""
    detector_scores = {"d_seg": 0.5, "d_pose_sqrt_10x": 0.5, "rate_25_over_denom": 0.5}
    cfg = FusionConfig(
        strategy=FusionStrategy.GEOMETRIC_MEAN,
        detector_weights={"d_seg": 1.0, "d_pose_sqrt_10x": 1.0, "rate_25_over_denom": 1.0},
    )
    score = compute_fusion_score(detector_scores, cfg)
    # exp(3 * log(0.5)) = 0.125
    assert score == pytest.approx(0.125)


def test_max_detector_wins_canonical() -> None:
    """MAX_DETECTOR_WINS = max(w_i * score_i)."""
    detector_scores = {"d_seg": 0.001, "d_pose_sqrt_10x": 0.02, "rate_25_over_denom": 0.10}
    cfg = FusionConfig(strategy=FusionStrategy.MAX_DETECTOR_WINS)
    score = compute_fusion_score(detector_scores, cfg)
    # 100 * 0.001 = 0.1, 1 * 0.02 = 0.02, 1 * 0.10 = 0.10
    # max = 0.10 (rate_25_over_denom wins under canonical weights)
    assert score == pytest.approx(0.10)


def test_learned_mlp_2_layer_bounded() -> None:
    """LEARNED_MLP_2_LAYER produces bounded output via tanh nonlinearity."""
    detector_scores = {"d_seg": 0.001, "d_pose_sqrt_10x": 0.02, "rate_25_over_denom": 0.10}
    cfg = FusionConfig(strategy=FusionStrategy.LEARNED_MLP_2_LAYER)
    score = compute_fusion_score(detector_scores, cfg)
    # Must be in (-1, 1) due to tanh.
    assert -1.0 < score < 1.0


def test_strategies_substantively_distinct() -> None:
    """Slot EEE: each canonical fusion strategy produces DIFFERENT output
    on the same input."""
    detector_scores = {"d_seg": 0.5, "d_pose_sqrt_10x": 0.3, "rate_25_over_denom": 0.7}
    outputs = {}
    for strategy in FusionStrategy:
        cfg = FusionConfig(strategy=strategy)
        if strategy == FusionStrategy.GEOMETRIC_MEAN:
            # Geometric mean requires positive scores; our example is positive.
            outputs[strategy] = compute_fusion_score(detector_scores, cfg)
        else:
            outputs[strategy] = compute_fusion_score(detector_scores, cfg)
    distinct = {round(v, 4) for v in outputs.values()}
    assert len(distinct) == len(outputs), (
        f"All canonical strategies must produce distinct outputs: {outputs}"
    )


def test_softmax_normalize_changes_output() -> None:
    """Softmax normalization changes the weighted sum result."""
    detector_scores = {"d_seg": 0.001, "d_pose_sqrt_10x": 0.02, "rate_25_over_denom": 0.10}
    cfg_no_softmax = FusionConfig(softmax_normalize=False)
    cfg_softmax = FusionConfig(softmax_normalize=True)
    score_no_softmax = compute_fusion_score(detector_scores, cfg_no_softmax)
    score_softmax = compute_fusion_score(detector_scores, cfg_softmax)
    # With softmax normalization, weights sum to 1; output is in much smaller range.
    assert abs(score_softmax - score_no_softmax) > 1e-3


def test_geometric_mean_requires_positive_scores() -> None:
    """GEOMETRIC_MEAN raises on non-positive scores."""
    detector_scores = {"d_seg": 0.0, "d_pose_sqrt_10x": 0.5, "rate_25_over_denom": 0.5}
    cfg = FusionConfig(
        strategy=FusionStrategy.GEOMETRIC_MEAN,
        detector_weights={"d_seg": 1.0, "d_pose_sqrt_10x": 1.0, "rate_25_over_denom": 1.0},
    )
    with pytest.raises(FusionDetectorError, match="strictly positive"):
        compute_fusion_score(detector_scores, cfg)


def test_detector_name_mismatch_raises() -> None:
    """detector_scores key not in config.detector_weights raises."""
    detector_scores = {"unknown_detector": 0.5}
    cfg = FusionConfig()
    with pytest.raises(FusionDetectorError, match="not in config.detector_weights"):
        compute_fusion_score(detector_scores, cfg)


def test_empty_scores_raise() -> None:
    """Empty detector_scores raises."""
    cfg = FusionConfig()
    with pytest.raises(FusionDetectorError, match="empty"):
        compute_fusion_score({}, cfg)


def test_config_invalid_strategy_raises() -> None:
    """Wrong type for strategy raises."""
    with pytest.raises(FusionDetectorError, match="must be FusionStrategy"):
        FusionConfig(strategy="bogus")  # type: ignore[arg-type]


def test_config_invalid_weight_type_raises() -> None:
    """Non-numeric weight raises."""
    with pytest.raises(FusionDetectorError, match="must be numeric"):
        FusionConfig(detector_weights={"d_seg": "high"})  # type: ignore[dict-item]


def test_config_empty_weights_raises() -> None:
    """Empty detector_weights raises."""
    with pytest.raises(FusionDetectorError, match="empty"):
        FusionConfig(detector_weights={})


def test_all_4_canonical_strategies_present() -> None:
    """4 canonical fusion strategies."""
    expected = {
        "linear_weighted_sum",
        "geometric_mean",
        "max_detector_wins",
        "learned_mlp_2_layer",
    }
    assert {s.value for s in FusionStrategy} == expected
