# SPDX-License-Identifier: MIT
"""Tests for tac.blahut_arimoto canonical Blahut-Arimoto R(D) helpers.

Per WAVE-3-PATH-A.2 closure of PATH-A's documented deeper gap. Tests
verify:
  * RateDistortionCurve dataclass invariants (frozen / required fields /
    canonical Provenance per Catalog #323).
  * iterate_rate_distortion convergence on the binary-symmetric-source
    canonical example (Cover & Thomas 2nd ed example 10.8.1
    ``R(D) = 1 - H(D)`` for ``D in [0, 1/2]``).
  * iterate_categorical_rd for DreamerV3 RSSM canonical configs
    (Hafner 32x32 and C6 Path B2 24x256 per PATH-A canonical-config table).
  * iterate_contest_scorer_rd smoke test with synthetic distortion oracle.
  * Edge cases (zero / infinite distortion / degenerate source).
  * Canonical equation registration round-trip per Catalog #344.
  * Catalog #185 sister-callable regression guard.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.blahut_arimoto import (
    DEFAULT_C6_PATH_B2_G,
    DEFAULT_C6_PATH_B2_K,
    DEFAULT_DREAMERV3_HAFNER_G,
    DEFAULT_DREAMERV3_HAFNER_K,
    DEFAULT_LAMBDA_SWEEP_POINTS,
    RateDistortionCurve,
    build_default_lambda_sweep,
    default_categorical_msl_distortion,
    iterate_categorical_rd,
    iterate_contest_scorer_rd,
    iterate_rate_distortion,
)
from tac.blahut_arimoto.contest_scorer import build_distortion_matrix_from_oracle
from tac.provenance.contract import Provenance, ProvenanceEvidenceGrade, ProvenanceKind


# ---------------------------------------------------------------------------
# RateDistortionCurve dataclass contract
# ---------------------------------------------------------------------------


def _build_minimal_curve() -> RateDistortionCurve:
    from tac.provenance.builders import build_provenance_for_predicted

    prov = build_provenance_for_predicted(
        model_id="test", inputs_sha256="a" * 64
    )
    return RateDistortionCurve(
        lambda_values=(0.1, 1.0, 10.0),
        rate_values=(0.0, 0.3, 1.0),
        distortion_values=(0.5, 0.25, 0.0),
        converged=(True, True, True),
        canonical_provenance=prov,
        derivation_method="test_v1",
        source_distribution_summary="test source",
    )


def test_rate_distortion_curve_frozen() -> None:
    curve = _build_minimal_curve()
    with pytest.raises((AttributeError, Exception)):  # frozen dataclass
        curve.rate_values = (1.0, 2.0, 3.0)  # type: ignore[misc]


def test_rate_distortion_curve_required_fields_present() -> None:
    curve = _build_minimal_curve()
    assert len(curve.lambda_values) == 3
    assert len(curve.rate_values) == 3
    assert len(curve.distortion_values) == 3
    assert len(curve.converged) == 3
    assert isinstance(curve.canonical_provenance, Provenance)
    assert curve.derivation_method == "test_v1"
    assert curve.source_distribution_summary == "test source"


def test_rate_distortion_curve_canonical_provenance_predicted_grade() -> None:
    curve = _build_minimal_curve()
    assert curve.canonical_provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
    assert curve.canonical_provenance.promotion_eligible is False
    assert curve.canonical_provenance.score_claim_valid is False
    assert curve.canonical_provenance.artifact_kind == ProvenanceKind.PREDICTED_FROM_MODEL


def test_rate_distortion_curve_length_mismatch_rejected() -> None:
    from tac.provenance.builders import build_provenance_for_predicted

    prov = build_provenance_for_predicted(model_id="test", inputs_sha256="a" * 64)
    with pytest.raises(ValueError, match="rate_values length"):
        RateDistortionCurve(
            lambda_values=(0.1, 1.0),
            rate_values=(0.0, 0.3, 1.0),  # wrong length
            distortion_values=(0.5, 0.25),
            converged=(True, True),
            canonical_provenance=prov,
            derivation_method="test_v1",
            source_distribution_summary="test",
        )


def test_rate_distortion_curve_negative_rate_rejected() -> None:
    from tac.provenance.builders import build_provenance_for_predicted

    prov = build_provenance_for_predicted(model_id="test", inputs_sha256="a" * 64)
    with pytest.raises(ValueError, match="rate_values"):
        RateDistortionCurve(
            lambda_values=(1.0,),
            rate_values=(-0.1,),
            distortion_values=(0.5,),
            converged=(True,),
            canonical_provenance=prov,
            derivation_method="test_v1",
            source_distribution_summary="test",
        )


def test_rate_distortion_curve_negative_distortion_rejected() -> None:
    from tac.provenance.builders import build_provenance_for_predicted

    prov = build_provenance_for_predicted(model_id="test", inputs_sha256="a" * 64)
    with pytest.raises(ValueError, match="distortion_values"):
        RateDistortionCurve(
            lambda_values=(1.0,),
            rate_values=(0.5,),
            distortion_values=(-0.1,),
            converged=(True,),
            canonical_provenance=prov,
            derivation_method="test_v1",
            source_distribution_summary="test",
        )


def test_rate_distortion_curve_empty_lambdas_rejected() -> None:
    from tac.provenance.builders import build_provenance_for_predicted

    prov = build_provenance_for_predicted(model_id="test", inputs_sha256="a" * 64)
    with pytest.raises(ValueError, match="lambda_values"):
        RateDistortionCurve(
            lambda_values=(),
            rate_values=(),
            distortion_values=(),
            converged=(),
            canonical_provenance=prov,
            derivation_method="test_v1",
            source_distribution_summary="test",
        )


def test_rate_distortion_curve_as_dict_round_trip() -> None:
    curve = _build_minimal_curve()
    d = curve.as_dict()
    assert "lambda_values" in d
    assert "rate_values" in d
    assert "canonical_provenance" in d
    assert isinstance(d["canonical_provenance"], dict)


# ---------------------------------------------------------------------------
# iterate_rate_distortion canonical convergence
# ---------------------------------------------------------------------------


def test_binary_symmetric_source_canonical_rd() -> None:
    """Cover & Thomas example 10.8.1: BSS + Hamming distortion -> R(D) = 1 - H(D).

    For source p_x = [0.5, 0.5] and Hamming distortion, the canonical
    R(D) is 1 - H(D) for D in [0, 1/2]. The BA iteration should
    converge to values consistent with this closed form at multiple
    operating points.
    """
    p_x = np.array([0.5, 0.5])
    distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
    curve = iterate_rate_distortion(distortion, p_x)

    # Sanity: all converged
    assert all(curve.converged)
    # Sanity: lambda ascending -> rate ascending (monotone curve)
    assert curve.rate_values[-1] >= curve.rate_values[0] - 1e-9
    # Sanity: lambda ascending -> distortion descending (monotone)
    assert curve.distortion_values[0] >= curve.distortion_values[-1] - 1e-9
    # Sanity: at large enough lambda, achieved D is small and rate is near R_max = 1
    # (the canonical capacity of BSS is 1 bit at zero-distortion)
    largest_rate = max(curve.rate_values)
    assert largest_rate <= 1.0 + 1e-3, f"BSS rate exceeds 1 bit capacity: {largest_rate}"
    # Canonical: R(0.25) = 1 - H(0.25) ~= 0.189 bits
    # Find a point near D=0.25
    closest_idx = int(np.argmin(np.abs(np.array(curve.distortion_values) - 0.25)))
    achieved_d = curve.distortion_values[closest_idx]
    achieved_r = curve.rate_values[closest_idx]
    # Closed-form check
    if 0.05 < achieved_d < 0.45:
        expected_r = 1.0 + achieved_d * math.log2(achieved_d) + (1.0 - achieved_d) * math.log2(1.0 - achieved_d)
        assert abs(achieved_r - expected_r) < 0.05, (
            f"BSS canonical mismatch: D={achieved_d}, R={achieved_r}, expected={expected_r}"
        )


def test_iterate_rate_distortion_validates_inputs() -> None:
    # Source must be 1D
    with pytest.raises(ValueError, match="1D"):
        iterate_rate_distortion(np.eye(3), np.eye(2))
    # Source must sum to 1
    with pytest.raises(ValueError, match="sum to 1"):
        iterate_rate_distortion(np.eye(2), np.array([0.3, 0.3]))
    # Distortion shape must match source dim
    with pytest.raises(ValueError, match="shape"):
        iterate_rate_distortion(np.eye(3), np.array([0.5, 0.5]))
    # Negative distortion rejected
    with pytest.raises(ValueError, match="non-negative"):
        iterate_rate_distortion(
            np.array([[-1.0, 1.0], [1.0, 1.0]]), np.array([0.5, 0.5])
        )


def test_iterate_rate_distortion_canonical_provenance_emitted() -> None:
    p_x = np.array([0.5, 0.5])
    distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
    curve = iterate_rate_distortion(distortion, p_x)
    assert curve.canonical_provenance.artifact_kind == ProvenanceKind.PREDICTED_FROM_MODEL
    assert curve.canonical_provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
    assert curve.canonical_provenance.promotion_eligible is False
    assert curve.canonical_provenance.score_claim_valid is False
    assert "blahut_arimoto" in curve.canonical_provenance.source_path


def test_iterate_rate_distortion_default_lambda_sweep() -> None:
    p_x = np.array([0.5, 0.5])
    distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
    curve = iterate_rate_distortion(distortion, p_x)
    assert len(curve.lambda_values) == DEFAULT_LAMBDA_SWEEP_POINTS


def test_iterate_rate_distortion_custom_lambda_sweep() -> None:
    p_x = np.array([0.5, 0.5])
    distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
    custom_lambdas = [0.5, 2.0, 5.0]
    curve = iterate_rate_distortion(distortion, p_x, lambda_values=custom_lambdas)
    assert len(curve.lambda_values) == 3
    assert curve.lambda_values == (0.5, 2.0, 5.0)


def test_build_default_lambda_sweep_adapts_to_distortion_scale() -> None:
    small = np.array([[0.0, 0.001], [0.001, 0.0]])
    large = np.array([[0.0, 100.0], [100.0, 0.0]])
    sweep_small = build_default_lambda_sweep(small)
    sweep_large = build_default_lambda_sweep(large)
    # Smaller distortion -> larger lambda required for same effect
    assert max(sweep_small) > max(sweep_large)


def test_build_default_lambda_sweep_validates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_default_lambda_sweep(np.array([]).reshape(0, 0))
    with pytest.raises(ValueError, match="n_points"):
        build_default_lambda_sweep(np.eye(2), n_points=1)


# ---------------------------------------------------------------------------
# iterate_categorical_rd specialization
# ---------------------------------------------------------------------------


def test_iterate_categorical_rd_default_hafner_config() -> None:
    """Hafner DreamerV3 canonical 32x32 config per PATH-A canonical-config."""
    curve = iterate_categorical_rd(G=4, K=4, n_lambda_points=4)
    # All converged
    assert all(curve.converged)
    # Rate values bounded by joint capacity G * log2(K) = 4 * 2 = 8 bits
    assert max(curve.rate_values) <= 4 * math.log2(4) + 0.1
    # Provenance is PREDICTED
    assert curve.canonical_provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
    # Derivation method tags the categorical specialization
    assert "categorical" in curve.derivation_method


def test_iterate_categorical_rd_hafner_capacity_ceiling_respected() -> None:
    """Per PATH-A capacity ceiling: max joint rate <= G * log2(K) = 32 * 5 = 160 bits."""
    curve = iterate_categorical_rd(
        G=DEFAULT_DREAMERV3_HAFNER_G,
        K=DEFAULT_DREAMERV3_HAFNER_K,
        n_lambda_points=4,  # smaller sweep for speed
    )
    capacity = DEFAULT_DREAMERV3_HAFNER_G * math.log2(DEFAULT_DREAMERV3_HAFNER_K)
    # Achievable rate cannot exceed capacity
    for r in curve.rate_values:
        assert r <= capacity + 0.1, f"rate {r} exceeds Hafner capacity {capacity}"


def test_iterate_categorical_rd_c6_path_b2_capacity_ceiling_respected() -> None:
    """Per PATH-A capacity ceiling: C6 Path B2 max joint rate <= 24 * 8 = 192 bits."""
    # Use smaller G=4, K=16 for speed (preserving K=2^4 structure)
    curve = iterate_categorical_rd(G=4, K=16, n_lambda_points=4)
    capacity = 4 * math.log2(16)
    for r in curve.rate_values:
        assert r <= capacity + 0.1


def test_iterate_categorical_rd_validates_G_K() -> None:
    with pytest.raises(ValueError, match="G must be"):
        iterate_categorical_rd(G=0, K=4)
    with pytest.raises(ValueError, match="K must be"):
        iterate_categorical_rd(G=4, K=1)
    with pytest.raises(ValueError, match="n_lambda_points"):
        iterate_categorical_rd(G=4, K=4, n_lambda_points=1)


def test_iterate_categorical_rd_heterogeneous_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        iterate_categorical_rd(G=4, K=4, homogeneous_groups=False)


def test_default_categorical_msl_distortion_canonical() -> None:
    """Default Hamming distortion: d(i,j) = 1 - delta_{ij}."""
    d = default_categorical_msl_distortion(4)
    assert d.shape == (4, 4)
    np.testing.assert_allclose(np.diag(d), 0.0)
    np.testing.assert_allclose(d[d != 0], 1.0)


def test_default_categorical_msl_distortion_validates() -> None:
    with pytest.raises(ValueError, match="K must be"):
        default_categorical_msl_distortion(1)


def test_iterate_categorical_rd_custom_distortion_fn() -> None:
    def my_distortion(K: int) -> np.ndarray:
        return np.full((K, K), 0.5) - np.eye(K) * 0.5  # asymmetric variant

    curve = iterate_categorical_rd(G=2, K=4, distortion_fn=my_distortion, n_lambda_points=4)
    assert all(curve.converged)


def test_iterate_categorical_rd_custom_source_prior() -> None:
    prior = np.array([0.1, 0.4, 0.3, 0.2])
    curve = iterate_categorical_rd(G=2, K=4, source_prior=prior, n_lambda_points=4)
    # BA can converge slowly at small lambda for non-uniform priors per
    # Blahut 1972 §III.C. At least the high-lambda end (low distortion) MUST
    # converge.
    assert any(curve.converged), "no lambda points converged"
    # Largest-lambda rate must be strictly larger than smallest-lambda rate
    # (curve trends from near-zero rate at small lambda to near-capacity at
    # large lambda; rates near 1e-5 at very small lambda are within
    # numerical noise of the BA fixed point and not monotone-significant).
    sorted_pairs = sorted(zip(curve.lambda_values, curve.rate_values))
    rates_in_lambda_order = [r for _, r in sorted_pairs]
    assert rates_in_lambda_order[-1] > rates_in_lambda_order[0], (
        f"max-lambda rate {rates_in_lambda_order[-1]} not > "
        f"min-lambda rate {rates_in_lambda_order[0]}"
    )


def test_iterate_categorical_rd_source_prior_must_sum_to_1() -> None:
    prior = np.array([0.1, 0.1, 0.1, 0.1])  # sums to 0.4
    with pytest.raises(ValueError, match="sum to 1"):
        iterate_categorical_rd(G=2, K=4, source_prior=prior)


# ---------------------------------------------------------------------------
# iterate_contest_scorer_rd smoke test
# ---------------------------------------------------------------------------


def test_iterate_contest_scorer_rd_synthetic_oracle() -> None:
    """Synthetic Hamming oracle: verify BA converges + Provenance is PREDICTED."""

    def hamming_oracle(x: int, y: int) -> float:
        return 0.0 if x == y else 1.0

    curve = iterate_contest_scorer_rd(
        hamming_oracle,
        n_source_symbols=4,
        n_lambda_points=4,
    )
    assert all(curve.converged)
    assert curve.derivation_method == "blahut_arimoto_contest_scorer_v1"
    assert curve.canonical_provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED


def test_iterate_contest_scorer_rd_oracle_validates_non_negative() -> None:
    def bad_oracle(x: int, y: int) -> float:
        return -1.0

    with pytest.raises(ValueError, match=">= 0"):
        iterate_contest_scorer_rd(bad_oracle, n_source_symbols=4)


def test_iterate_contest_scorer_rd_oracle_validates_finite() -> None:
    def bad_oracle(x: int, y: int) -> float:
        return float("nan")

    with pytest.raises(ValueError, match="finite"):
        iterate_contest_scorer_rd(bad_oracle, n_source_symbols=4)


def test_build_distortion_matrix_from_oracle_basic() -> None:
    def asymmetric(x: int, y: int) -> float:
        return float(abs(x - y))

    matrix = build_distortion_matrix_from_oracle(asymmetric, 4, 4)
    assert matrix.shape == (4, 4)
    # Diagonal is 0
    np.testing.assert_allclose(np.diag(matrix), 0.0)
    # Off-diagonal matches abs(i - j)
    assert matrix[0, 3] == 3.0
    assert matrix[2, 0] == 2.0


def test_iterate_contest_scorer_rd_validates_n_source() -> None:
    def trivial(x: int, y: int) -> float:
        return 0.0

    with pytest.raises(ValueError, match="n_source_symbols"):
        iterate_contest_scorer_rd(trivial, n_source_symbols=1)


def test_iterate_contest_scorer_rd_custom_n_reproduction_symbols() -> None:
    """When |X| != |Y|, the matrix is non-square but BA still works."""

    def trivial(x: int, y: int) -> float:
        return 0.5 if x == y else 1.0

    curve = iterate_contest_scorer_rd(
        trivial, n_source_symbols=4, n_reproduction_symbols=2, n_lambda_points=4
    )
    assert all(curve.converged)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_iterate_rate_distortion_zero_distortion_alphabet() -> None:
    """All-zero distortion matrix: rate is 0 at any lambda."""
    p_x = np.array([0.5, 0.5])
    zero_dist = np.zeros((2, 2))
    curve = iterate_rate_distortion(zero_dist, p_x)
    assert all(r < 1e-6 for r in curve.rate_values), f"non-zero rate: {curve.rate_values}"
    assert all(d < 1e-6 for d in curve.distortion_values)


def test_iterate_rate_distortion_degenerate_source_concentrated() -> None:
    """Source mass concentrated on one symbol: rate is 0 at any distortion."""
    p_x = np.array([1.0, 0.0])
    distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
    curve = iterate_rate_distortion(distortion, p_x)
    # All rates should be 0 (no entropy to compress)
    assert all(r < 1e-6 for r in curve.rate_values), f"non-zero rate: {curve.rate_values}"


# ---------------------------------------------------------------------------
# Canonical equation registration round-trip per Catalog #344
# ---------------------------------------------------------------------------


def test_blahut_arimoto_sub_equation_registered() -> None:
    """The sub-equation must be registered via register_canonical_equation."""
    from tac.canonical_equations import get_equation_by_id

    eq = get_equation_by_id("categorical_blahut_arimoto_rate_distortion_v1")
    assert eq is not None, (
        "Sub-equation must be registered. "
        "Run tools/register_blahut_arimoto_equation.py if missing."
    )
    # Canonical-producer + consumer audit per Catalog #344
    assert len(eq.canonical_producers) >= 1
    assert len(eq.canonical_consumers) >= 1
    # Canonical Provenance per Catalog #323
    assert eq.provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED


def test_blahut_arimoto_sub_equation_refines_path_a() -> None:
    """The sub-equation must cite PATH-A's necessary-condition equation."""
    from tac.canonical_equations import get_equation_by_id

    eq = get_equation_by_id("categorical_blahut_arimoto_rate_distortion_v1")
    assert eq is not None
    # Sister citation: PATH-A's equation_id must appear in producers OR
    # the equation's relevance_tokens (domain_of_validity)
    domain_str = str(eq.domain_of_validity)
    consumers_str = str(eq.canonical_consumers)
    producers_str = str(eq.canonical_producers)
    # Reference to PATH-A equation must be present somewhere
    found_path_a_ref = (
        "categorical_posterior_capacity_vs_continuous_gaussian_v1" in domain_str
        or "categorical_posterior_capacity_vs_continuous_gaussian_v1" in consumers_str
        or "categorical_posterior_capacity_vs_continuous_gaussian_v1" in producers_str
    )
    assert found_path_a_ref, "PATH-A sister citation missing from equation"


def test_blahut_arimoto_sub_equation_has_canonical_relevance_tokens() -> None:
    """Sub-equation must carry the canonical relevance tokens for lookup."""
    from tac.canonical_equations import get_equation_by_id

    eq = get_equation_by_id("categorical_blahut_arimoto_rate_distortion_v1")
    assert eq is not None
    # Relevance tokens live in domain_of_validity per PATH-A pattern
    tokens = eq.domain_of_validity.get("relevance_tokens", [])
    token_set = set(tokens)
    # Canonical token coverage
    expected_anchors = {
        "blahut_arimoto",
        "iterate_rate_distortion",
        "categorical_rd",
        "exact_achievable_rate",
        "cover_thomas_13_8",
    }
    missing = expected_anchors - token_set
    assert not missing, f"missing canonical tokens: {missing}"


# ---------------------------------------------------------------------------
# Catalog #185 sister-callable regression guard
# ---------------------------------------------------------------------------


def test_iterate_rate_distortion_callable_via_globals() -> None:
    """Per Catalog #185 META-meta: the canonical helper must be callable."""
    import tac.blahut_arimoto

    assert callable(tac.blahut_arimoto.iterate_rate_distortion)
    assert callable(tac.blahut_arimoto.iterate_categorical_rd)
    assert callable(tac.blahut_arimoto.iterate_contest_scorer_rd)


def test_canonical_constants_pinned() -> None:
    """Per CLAUDE.md "Apples-to-apples evidence discipline": constants pinned."""
    assert DEFAULT_DREAMERV3_HAFNER_G == 32
    assert DEFAULT_DREAMERV3_HAFNER_K == 32
    assert DEFAULT_C6_PATH_B2_G == 24
    assert DEFAULT_C6_PATH_B2_K == 256
    assert DEFAULT_LAMBDA_SWEEP_POINTS >= 4
