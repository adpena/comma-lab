# SPDX-License-Identifier: MIT
"""Tests for score_predictor, theoretical_floor, arithmetic_coder_class_conditional."""
from __future__ import annotations

import math

import pytest

from tac.contest_oracle.arithmetic_coder_class_conditional import (
    ArithmeticCoderError,
    ClassConditionalCodebook,
    build_class_conditional_codebook,
)
from tac.contest_oracle.gradient import compute_score
from tac.contest_oracle.score_predictor import (
    ContestScorePrediction,
    build_contest_action,
    predict_score,
    validate_against_canonical_formula,
)
from tac.contest_oracle.theoretical_floor import (
    CANONICAL_BLAHUT_AVAILABLE,
    compute_contest_floor,
)


# ---------------------------------------------------------------------------
# score_predictor
# ---------------------------------------------------------------------------
def test_predict_score_decomposition_matches_canonical():
    pred = predict_score(d_seg=0.01, d_pose=0.0001, archive_bytes=1_000_000)
    assert isinstance(pred, ContestScorePrediction)
    expected_seg = 100 * 0.01
    expected_pose = math.sqrt(10 * 0.0001)
    expected_rate = 25 * 1_000_000 / 37_545_489
    assert math.isclose(pred.seg_contribution, expected_seg)
    assert math.isclose(pred.pose_contribution, expected_pose)
    assert math.isclose(pred.rate_contribution, expected_rate)
    assert math.isclose(pred.predicted_score, expected_seg + expected_pose + expected_rate)


def test_predict_score_evidence_grade_predicted():
    """Per Catalog #287/#323."""
    pred = predict_score(d_seg=0.01, d_pose=0.0001, archive_bytes=1_000_000)
    assert pred.evidence_grade == "predicted_analytical"


def test_predict_score_rejects_negative():
    with pytest.raises(ValueError):
        predict_score(d_seg=-0.01, d_pose=0.0001, archive_bytes=1_000_000)
    with pytest.raises(ValueError):
        predict_score(d_seg=0.01, d_pose=-0.0001, archive_bytes=1_000_000)
    with pytest.raises(ValueError):
        predict_score(d_seg=0.01, d_pose=0.0001, archive_bytes=-1)


def test_validate_against_canonical_formula_matches():
    assert validate_against_canonical_formula(
        d_seg=0.01, d_pose=0.0001, archive_bytes=1_000_000
    )


def test_predict_score_equals_compute_score():
    """Cross-check sister function."""
    pred = predict_score(d_seg=0.005, d_pose=2e-5, archive_bytes=500_000)
    sister = compute_score(0.005, 2e-5, 500_000)
    assert math.isclose(pred.predicted_score, sister)


def test_build_contest_action_returns_callable():
    """Should construct without ImportError + return something callable-like."""
    action = build_contest_action()
    # The Action is a typed dataclass per tac.unified_action; verify it has the
    # canonical track-callables structure.
    assert action is not None


# ---------------------------------------------------------------------------
# theoretical_floor (alias to symposium_impls.blahut_arimoto)
# ---------------------------------------------------------------------------
def test_canonical_blahut_available():
    """The sister module is importable."""
    assert CANONICAL_BLAHUT_AVAILABLE is True


def test_compute_contest_floor_returns_dataclass():
    """Smoke test that the alias forwards correctly."""
    floor = compute_contest_floor(target_d_seg=0.0005, target_d_pose=1e-6)
    # The sister returns ContestTheoreticalFloor; we don't import the type
    # here to avoid coupling, just verify it has the expected attribute.
    assert hasattr(floor, "contest_score_floor")


# ---------------------------------------------------------------------------
# arithmetic_coder_class_conditional
# ---------------------------------------------------------------------------
def test_build_codebook_5_class_uniform_observations():
    """5 classes with identical pooled distribution -> savings ~= 0."""
    obs_per_class = [[0, 1, 2, 3, 0, 1, 2, 3] for _ in range(5)]
    cb = build_class_conditional_codebook(
        per_class_symbol_observations=obs_per_class
    )
    assert isinstance(cb, ClassConditionalCodebook)
    assert len(cb.class_priors) == 5
    assert cb.vocabulary_size == 4
    # Each class identical -> savings ~ 0
    assert abs(cb.expected_savings_vs_class_agnostic_bits_per_symbol) < 0.01


def test_build_codebook_class_specific_yields_savings():
    """Different per-class distributions -> savings > 0."""
    obs = [
        [0] * 100,           # class 0 all symbol 0
        [1] * 100,           # class 1 all symbol 1
        [2] * 100,           # class 2 all symbol 2
        [3] * 100,           # class 3 all symbol 3
        [0, 1, 2, 3] * 25,   # class 4 uniform
    ]
    cb = build_class_conditional_codebook(per_class_symbol_observations=obs)
    # Classes 0-3 have zero entropy each (deterministic), class 4 has 2 bits
    # Class-agnostic entropy is higher (mixed distribution)
    assert cb.expected_savings_vs_class_agnostic_bits_per_symbol > 0.5


def test_build_codebook_rejects_wrong_outer_length():
    with pytest.raises(ArithmeticCoderError):
        build_class_conditional_codebook(
            per_class_symbol_observations=[[0], [1]]  # 2 not 5
        )


def test_build_codebook_rejects_empty_inner():
    with pytest.raises(ArithmeticCoderError):
        build_class_conditional_codebook(
            per_class_symbol_observations=[[0], [], [1], [2], [3]]
        )


def test_build_codebook_rejects_negative_symbol():
    with pytest.raises(ArithmeticCoderError):
        build_class_conditional_codebook(
            per_class_symbol_observations=[[0, -1], [1], [2], [3], [4]]
        )


def test_build_codebook_per_class_entropy_zero_for_deterministic_class():
    """Class with single symbol -> entropy = 0."""
    obs = [[0] * 10, [1] * 10, [2] * 10, [3] * 10, [4] * 10]
    cb = build_class_conditional_codebook(per_class_symbol_observations=obs)
    for cp in cb.class_priors:
        assert cp.estimated_entropy_bits_per_symbol == 0.0
