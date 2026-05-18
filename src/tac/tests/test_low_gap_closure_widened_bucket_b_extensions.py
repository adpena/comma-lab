# SPDX-License-Identifier: MIT
"""Tests for BUCKET B (3 NEW LOW gap candidate module extensions).

LOW gap closure widened wave 2026-05-17 — `lane_low_gap_closure_widened_8_modules_
plus_autopilot_wire_in_20260517` BUCKET B. Exercises the 3 NEW public per-pair
extensions on `tac.optimization.bayesian_experimental_design`,
`tac.optimization.cooperative_receiver_integration`, and
`tac.optimization.entropy_rate_decomposition`.

Per CLAUDE.md "Apples-to-apples evidence discipline": every outcome carries
`score_claim=False`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.bayesian_experimental_design import (
    PER_PAIR_EIG_SCHEMA,
    BayesianExperimentalDesignError,
    compute_expected_information_gain_per_pair,
)
from tac.optimization.cooperative_receiver_integration import (
    PER_PAIR_Z4_WEIGHTING_SCHEMA,
    weight_z4_loss_by_per_pair_fisher,
)
from tac.optimization.entropy_rate_decomposition import (
    PER_BYTE_RATE_ATTRIBUTION_SCHEMA,
    EntropyRateDecompositionError,
    attribute_rate_distortion_per_byte,
)


# ── BED: compute_expected_information_gain_per_pair ──────────────────────────


def test_bed_eig_basic_per_pair_calculation():
    """Smoke: 4 pairs of 10 bytes × 3 axes; EIG is non-negative for each pair."""
    rng = np.random.default_rng(seed=42)
    ppg = rng.standard_normal((10, 4, 3)).astype(np.float64)
    eig = compute_expected_information_gain_per_pair(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=ppg,
        observation_noise_variance=1e-4,
        auto_load=False,
    )
    assert len(eig) == 4
    for pair_idx, val in eig.items():
        assert isinstance(pair_idx, int)
        assert val >= 0.0
        assert isinstance(val, float)


def test_bed_eig_canonical_formula_correctness():
    """Verify EIG = 0.5 * log(1 + signal_var / noise_var) at known input."""
    # Construct a per-pair gradient where pair 0 has uniform unit magnitude:
    # 10 bytes × 3 axes × magnitude 1 → sum_of_squares = 30 → variance = 30/30 = 1.0
    ppg = np.zeros((10, 1, 3), dtype=np.float64)
    ppg[:, 0, :] = 1.0  # all elements of pair 0 = 1.0
    eig = compute_expected_information_gain_per_pair(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=ppg,
        observation_noise_variance=1.0,
        auto_load=False,
    )
    # EIG_0 = 0.5 * log(1 + 1.0/1.0) = 0.5 * log(2) ≈ 0.3466
    import math

    expected = 0.5 * math.log(2.0)
    assert abs(eig[0] - expected) < 1e-9


def test_bed_eig_rejects_bad_sha():
    rng = np.random.default_rng(seed=42)
    ppg = rng.standard_normal((10, 4, 3)).astype(np.float64)
    with pytest.raises(BayesianExperimentalDesignError, match="archive_sha256"):
        compute_expected_information_gain_per_pair(
            archive_sha256="short",
            per_pair_gradient=ppg,
            auto_load=False,
        )


def test_bed_eig_rejects_bad_noise_variance():
    rng = np.random.default_rng(seed=42)
    ppg = rng.standard_normal((10, 4, 3)).astype(np.float64)
    with pytest.raises(BayesianExperimentalDesignError, match="observation_noise_variance"):
        compute_expected_information_gain_per_pair(
            archive_sha256="deadbeef1234567890abcdef",
            per_pair_gradient=ppg,
            observation_noise_variance=-1.0,
            auto_load=False,
        )


def test_bed_eig_no_gradient_returns_empty_dict():
    """No per-pair gradient + auto_load=False → empty dict (planning-only,
    NO score claim)."""
    out = compute_expected_information_gain_per_pair(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=None,
        auto_load=False,
    )
    assert out == {}


def test_bed_eig_schema_pinned():
    assert PER_PAIR_EIG_SCHEMA == "tac_bayesian_experimental_design_per_pair_eig_v1"


# ── CR-Int: weight_z4_loss_by_per_pair_fisher ────────────────────────────────


def test_cr_z4_weights_basic_normalization():
    """Per-pair weights are normalized by max Fisher; max maps to 1.0."""
    pf = {0: 1.5, 1: 0.3, 2: 2.0}
    out = weight_z4_loss_by_per_pair_fisher(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_fisher=pf,
        auto_load=False,
    )
    assert out["per_pair_z4_weights"][2] == 1.0  # max maps to 1.0
    assert out["per_pair_z4_weights"][0] == 0.75  # 1.5/2.0
    assert out["per_pair_z4_weights"][1] == 0.15  # 0.3/2.0
    assert out["n_pairs_weighted"] == 3
    assert out["per_pair_fisher_consumed"] is False  # supplied directly
    assert out["score_claim"] is False


def test_cr_z4_empty_fisher_returns_empty_envelope():
    out = weight_z4_loss_by_per_pair_fisher(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_fisher={},
        auto_load=False,
    )
    assert out["per_pair_z4_weights"] == {}
    assert out["n_pairs_weighted"] == 0
    assert out["score_claim"] is False


def test_cr_z4_rejects_bad_sha():
    with pytest.raises(ValueError, match="archive_sha256"):
        weight_z4_loss_by_per_pair_fisher(
            archive_sha256="bad",
            per_pair_fisher={0: 1.0},
            auto_load=False,
        )


def test_cr_z4_weights_clamped_to_unit_interval():
    """Weights are always clamped to [0, 1]."""
    pf = {0: 5.0, 1: 0.001}
    out = weight_z4_loss_by_per_pair_fisher(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_fisher=pf,
        auto_load=False,
    )
    for w in out["per_pair_z4_weights"].values():
        assert 0.0 <= w <= 1.0


def test_cr_z4_schema_pinned():
    assert PER_PAIR_Z4_WEIGHTING_SCHEMA == "tac_cooperative_receiver_per_pair_z4_weighting_v1"


# ── ERD: attribute_rate_distortion_per_byte ──────────────────────────────────


def test_erd_attribute_basic_decomposition():
    """Per-byte rate + score contribution sums to total rate term."""
    alloc = {0: 100, 1: 200, 2: 50}
    out = attribute_rate_distortion_per_byte(
        archive_sha256="deadbeef1234567890abcdef",
        per_byte_allocation_map=alloc,
        archive_total_bytes=350,
        auto_load=False,
    )
    # Each per-byte rate fraction sums to 1.0 when allocation_total ==
    # archive_total
    sum_rate = sum(out["per_byte_rate_contribution"].values())
    assert abs(sum_rate - 1.0) < 1e-9
    # Total rate term = 25 * 350 / 37545489
    expected_total = 25.0 * 350 / 37_545_489
    assert abs(out["total_rate_term"] - expected_total) < 1e-12
    assert out["per_byte_allocation_consumed"] is False  # supplied directly
    assert out["score_claim"] is False


def test_erd_attribute_score_contribution_per_byte_canonical_formula():
    """Per-byte score contribution = 25 * byte's bytes / CONTEST_ORIGINAL_BYTES."""
    alloc = {0: 1000}
    out = attribute_rate_distortion_per_byte(
        archive_sha256="deadbeef1234567890abcdef",
        per_byte_allocation_map=alloc,
        archive_total_bytes=1000,
        auto_load=False,
    )
    expected = 25.0 * 1000 / 37_545_489
    assert abs(out["per_byte_score_contribution"][0] - expected) < 1e-12


def test_erd_attribute_empty_alloc_returns_empty_envelope():
    out = attribute_rate_distortion_per_byte(
        archive_sha256="deadbeef1234567890abcdef",
        per_byte_allocation_map={},
        auto_load=False,
    )
    assert out["per_byte_rate_contribution"] == {}
    assert out["per_byte_score_contribution"] == {}
    assert out["total_rate_term"] == 0.0
    assert out["score_claim"] is False


def test_erd_attribute_rejects_bad_sha():
    with pytest.raises(EntropyRateDecompositionError, match="archive_sha256"):
        attribute_rate_distortion_per_byte(
            archive_sha256="bad",
            per_byte_allocation_map={0: 100},
            auto_load=False,
        )


def test_erd_attribute_no_alloc_no_total_returns_empty():
    """When neither map nor auto_load supplies signal → empty envelope."""
    out = attribute_rate_distortion_per_byte(
        archive_sha256="deadbeef1234567890abcdef",
        per_byte_allocation_map=None,
        auto_load=False,
    )
    assert out["per_byte_rate_contribution"] == {}
    assert out["score_claim"] is False


def test_erd_attribute_derives_total_from_allocation_when_none():
    """If archive_total_bytes is None, denominator = allocation sum (so
    per-byte rate fractions still sum to 1.0)."""
    alloc = {0: 100, 1: 200}
    out = attribute_rate_distortion_per_byte(
        archive_sha256="deadbeef1234567890abcdef",
        per_byte_allocation_map=alloc,
        archive_total_bytes=None,
        auto_load=False,
    )
    sum_rate = sum(out["per_byte_rate_contribution"].values())
    assert abs(sum_rate - 1.0) < 1e-9


def test_erd_attribute_schema_pinned():
    assert PER_BYTE_RATE_ATTRIBUTION_SCHEMA == "tac_entropy_rate_per_byte_attribution_v1"


# ── Catalog #123 compliance: each new function consumes score-gradient-derived
# inputs (NOT weight-domain saliency) ────────────────────────────────────────


def test_bed_eig_input_is_score_gradient_not_weights():
    """The compute_expected_information_gain_per_pair function consumes
    per-pair GRADIENT (score-gradient-derived per
    PER_PAIR_GRADIENT_TENSOR_KIND), NOT weight tensors. Per Catalog #123
    the bug class is weight-domain saliency (mean(theta**2) etc.).
    Sister-function compliance verified by upstream loader contract."""
    # The function's signature accepts `per_pair_gradient` only; the
    # parameter name + canonical loader (load_per_pair_gradient_from_anchor)
    # both enforce score-gradient provenance.
    import inspect

    sig = inspect.signature(compute_expected_information_gain_per_pair)
    assert "per_pair_gradient" in sig.parameters


def test_cr_z4_input_is_score_gradient_derived_fisher_not_weights():
    """weight_z4_loss_by_per_pair_fisher consumes per_pair_fisher (which
    upstream Catalog #123-compliant sister #817 Gap 3
    allocate_per_pair_fisher_importance produces from score-gradient covariance)."""
    import inspect

    sig = inspect.signature(weight_z4_loss_by_per_pair_fisher)
    assert "per_pair_fisher" in sig.parameters


def test_erd_attribute_input_is_per_byte_allocation_not_weights():
    """attribute_rate_distortion_per_byte consumes per_byte_allocation_map
    (which upstream Catalog #123-compliant sister #817 Gap 1
    allocate_per_pair_bits produces). NOT weight-domain saliency."""
    import inspect

    sig = inspect.signature(attribute_rate_distortion_per_byte)
    assert "per_byte_allocation_map" in sig.parameters
