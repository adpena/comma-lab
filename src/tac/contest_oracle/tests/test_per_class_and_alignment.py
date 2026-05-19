# SPDX-License-Identifier: MIT
"""Tests for ``per_class_lagrangian`` + ``substrate_alignment``."""
from __future__ import annotations

import math

import pytest

from tac.contest_oracle.per_class_lagrangian import (
    DEFAULT_EFFECTIVE_NUMBER_BETA,
    PerClassLagrangianError,
    PerClassLambdaSeg,
    apply_per_class_lambda_to_seg_loss,
    compute_per_class_lambda_seg,
)
from tac.contest_oracle.substrate_alignment import (
    PR101_GOLD_ALIGNED_FACETS,
    AlignmentFacet,
    SubstrateAlignmentScore,
    pr101_gold_reference,
    score_substrate_alignment,
)


# ---------------------------------------------------------------------------
# per_class_lagrangian
# ---------------------------------------------------------------------------
def test_compute_per_class_lambda_returns_normalized_5tuple():
    freqs = [0.5, 0.2, 0.15, 0.1, 0.05]
    res = compute_per_class_lambda_seg(class_frequencies=freqs)
    assert isinstance(res, PerClassLambdaSeg)
    assert len(res.lambda_per_class_inverse_frequency) == 5
    # Normalized to mean 1
    mean_inv = sum(res.lambda_per_class_inverse_frequency) / 5
    assert math.isclose(mean_inv, 1.0, rel_tol=1e-6)


def test_rare_classes_get_higher_lambda():
    """5% class should get higher lambda than 50% class."""
    freqs = [0.5, 0.2, 0.15, 0.1, 0.05]
    res = compute_per_class_lambda_seg(class_frequencies=freqs)
    # last (rarest, 0.05) should have higher lambda than first (most common, 0.5)
    assert res.lambda_per_class_inverse_frequency[-1] > res.lambda_per_class_inverse_frequency[0]


def test_compute_per_class_rejects_wrong_length():
    with pytest.raises(PerClassLagrangianError):
        compute_per_class_lambda_seg(class_frequencies=[0.5, 0.5])  # 2 not 5


def test_compute_per_class_rejects_negative():
    with pytest.raises(PerClassLagrangianError):
        compute_per_class_lambda_seg(class_frequencies=[-0.1, 0.3, 0.3, 0.3, 0.1])


def test_compute_per_class_rejects_zero_total():
    with pytest.raises(PerClassLagrangianError):
        compute_per_class_lambda_seg(class_frequencies=[0.0] * 5)


def test_compute_per_class_handles_uniform():
    """Uniform freqs -> all lambda = 1."""
    res = compute_per_class_lambda_seg(class_frequencies=[0.2] * 5)
    for lam in res.lambda_per_class_inverse_frequency:
        assert math.isclose(lam, 1.0, rel_tol=1e-6)


def test_compute_per_class_rejects_bad_beta():
    with pytest.raises(PerClassLagrangianError):
        compute_per_class_lambda_seg(class_frequencies=[0.2] * 5, beta=1.5)
    with pytest.raises(PerClassLagrangianError):
        compute_per_class_lambda_seg(class_frequencies=[0.2] * 5, beta=-0.1)


def test_apply_per_class_lambda_to_seg_loss():
    """L = sum_c lambda_c * d_seg_c."""
    lambdas = (1.0, 2.0, 3.0, 4.0, 5.0)
    d_segs = (0.1, 0.1, 0.1, 0.1, 0.1)
    loss = apply_per_class_lambda_to_seg_loss(
        per_class_argmax_disagreement=d_segs,
        lambda_per_class=lambdas,
    )
    assert math.isclose(loss, 0.1 * (1 + 2 + 3 + 4 + 5), rel_tol=1e-9)


def test_apply_per_class_lambda_rejects_length_mismatch():
    with pytest.raises(PerClassLagrangianError):
        apply_per_class_lambda_to_seg_loss(
            per_class_argmax_disagreement=(0.1, 0.1, 0.1),
            lambda_per_class=(1.0, 1.0, 1.0, 1.0, 1.0),
        )


def test_apply_per_class_lambda_rejects_negative():
    with pytest.raises(PerClassLagrangianError):
        apply_per_class_lambda_to_seg_loss(
            per_class_argmax_disagreement=(-0.1, 0.1, 0.1, 0.1, 0.1),
            lambda_per_class=(1.0, 1.0, 1.0, 1.0, 1.0),
        )


def test_default_effective_number_beta_canonical():
    """Cui et al 2019 canonical."""
    assert DEFAULT_EFFECTIVE_NUMBER_BETA == 0.9999


# ---------------------------------------------------------------------------
# substrate_alignment
# ---------------------------------------------------------------------------
def test_pr101_gold_alignment_score():
    """PR101 GOLD reference: 7-of-8 facets = 0.875 = FULLY_ALIGNED."""
    score = pr101_gold_reference()
    assert isinstance(score, SubstrateAlignmentScore)
    assert math.isclose(score.alignment_score, 7.0 / 8.0)
    assert score.verdict == "FULLY_ALIGNED"
    assert AlignmentFacet.FULL_RENDERER_RGB_OUT in score.aligned_facets


def test_score_substrate_alignment_all_facets_satisfied():
    score = score_substrate_alignment(
        substrate_id="hypothetical_perfect",
        aligned_facets=tuple(AlignmentFacet),
    )
    assert score.alignment_score == 1.0
    assert score.verdict == "FULLY_ALIGNED"
    assert len(score.misaligned_facets) == 0


def test_score_substrate_alignment_no_facets_satisfied():
    score = score_substrate_alignment(
        substrate_id="pure_research_substrate",
        aligned_facets=(),
    )
    assert score.alignment_score == 0.0
    assert score.verdict == "MIS_ALIGNED"


def test_score_substrate_alignment_mostly_aligned():
    """5-of-8 aligned -> MOSTLY_ALIGNED."""
    facets = (
        AlignmentFacet.FULL_RENDERER_RGB_OUT,
        AlignmentFacet.PER_PAIR_STRUCTURE,
        AlignmentFacet.PER_CLASS_STRUCTURE,
        AlignmentFacet.SPATIAL_RESOLUTION_MATCH,
        AlignmentFacet.POSE_AXIS_AWARE,
    )
    score = score_substrate_alignment(substrate_id="hypothetical", aligned_facets=facets)
    assert score.alignment_score == 5.0 / 8.0
    assert score.verdict == "MOSTLY_ALIGNED"


def test_score_substrate_alignment_partially_aligned():
    """3-of-8 aligned -> PARTIALLY_ALIGNED."""
    facets = (
        AlignmentFacet.FULL_RENDERER_RGB_OUT,
        AlignmentFacet.PER_PAIR_STRUCTURE,
        AlignmentFacet.PER_CLASS_STRUCTURE,
    )
    score = score_substrate_alignment(substrate_id="hypothetical", aligned_facets=facets)
    assert score.alignment_score == 3.0 / 8.0
    assert score.verdict == "PARTIALLY_ALIGNED"


def test_score_substrate_alignment_rejects_empty_id():
    with pytest.raises(ValueError):
        score_substrate_alignment(substrate_id="", aligned_facets=())
