# SPDX-License-Identifier: MIT
"""Synthetic tests for Lane MDL/Bayesian codec-comparison framework.

All tests run on synthetic / hand-computed numbers; no GPU, no real archives.
Real-archive integration measurement is Level 2 (out of scope for this scaffold).

References
----------
- Module: src/tac/mdl_bayesian_codec.py
- Design: .omx/research/council_lane_mdl_bayesian_design_20260430.md
"""
from __future__ import annotations

import math

import pytest

from tac.mdl_bayesian_codec import (
    MDL_FRAMEWORK_VERSION,
    MDLCodecResult,
    OccamCheck,
    bayes_factor_log2,
    bayesian_model_average,
    derive_codec_prior_log2,
    laplace_log_evidence,
    mdl_total_bits,
    rank_codecs,
)


# ── primitives ─────────────────────────────────────────────────────────


def test_mdl_total_bits_simple_sum():
    """L_total = L(M) + L(D|M)."""
    assert mdl_total_bits(model_bits=1000, residual_bits=5000) == 6000


def test_mdl_total_bits_rejects_negative_inputs():
    with pytest.raises(ValueError, match="model_bits"):
        mdl_total_bits(model_bits=-1, residual_bits=100)
    with pytest.raises(ValueError, match="residual_bits"):
        mdl_total_bits(model_bits=100, residual_bits=-1)


def test_bayes_factor_log2_positive_means_a_wins():
    """Smaller L_total → larger p(D|M) → positive log2 BF.

    If L_A = 1000 and L_B = 2000, then BF_{A,B} = 2^1000 (A is hugely preferred).
    log2 BF = -L_A + L_B = -1000 + 2000 = +1000.
    """
    assert bayes_factor_log2(l_total_a_bits=1000, l_total_b_bits=2000) == 1000.0


def test_bayes_factor_log2_symmetric():
    """log2 BF_{A,B} = -log2 BF_{B,A}."""
    bf_ab = bayes_factor_log2(l_total_a_bits=500, l_total_b_bits=750)
    bf_ba = bayes_factor_log2(l_total_a_bits=750, l_total_b_bits=500)
    assert math.isclose(bf_ab, -bf_ba)


# ── Laplace approximation ─────────────────────────────────────────────


def test_laplace_log_evidence_zero_params_zero_correction():
    """k=0 → Occam term = -0.5 log |H| only."""
    val = laplace_log_evidence(
        log_likelihood_max_nats=-100.0,
        log_prior_max_nats=-1.0,
        hessian_logdet=2.0,
        n_params=0,
    )
    # Expected: -100 + (-1) + 0 - 0.5*2 = -102
    assert math.isclose(val, -102.0, abs_tol=1e-9)


def test_laplace_log_evidence_with_params():
    """k>0 adds (k/2) log(2π) to the Occam correction."""
    val = laplace_log_evidence(
        log_likelihood_max_nats=0.0,
        log_prior_max_nats=0.0,
        hessian_logdet=0.0,
        n_params=10,
    )
    # Expected: 0 + 0 + (10/2)*log(2π) - 0 = 5*log(2π)
    expected = 5.0 * math.log(2.0 * math.pi)
    assert math.isclose(val, expected, abs_tol=1e-9)


def test_laplace_rejects_negative_n_params():
    with pytest.raises(ValueError, match="n_params"):
        laplace_log_evidence(
            log_likelihood_max_nats=0.0,
            log_prior_max_nats=0.0,
            hessian_logdet=0.0,
            n_params=-1,
        )


# ── MDLCodecResult ─────────────────────────────────────────────────────


def test_codec_result_total_bits_two_part_mdl():
    r = MDLCodecResult(
        codec_name="lane_balle_hyperprior",
        model_bits=5000,
        residual_bits=20000,
        n_data_symbols=1000,
    )
    assert r.total_bits() == 25000


def test_codec_result_per_symbol_bits():
    r = MDLCodecResult(
        codec_name="lane_sh_static",
        model_bits=400,
        residual_bits=3600,
        n_data_symbols=1000,
    )
    assert math.isclose(r.per_symbol_bits(), 3.6)


def test_codec_result_per_symbol_bits_rejects_zero_symbols():
    r = MDLCodecResult(
        codec_name="bad",
        model_bits=0,
        residual_bits=0,
        n_data_symbols=0,
    )
    with pytest.raises(ValueError, match="n_data_symbols"):
        r.per_symbol_bits()


def test_codec_result_laplace_requires_both_params_and_hessian():
    r = MDLCodecResult(
        codec_name="bad",
        model_bits=100,
        residual_bits=100,
        n_data_symbols=10,
    )
    with pytest.raises(ValueError, match="laplace_log_evidence requires"):
        r.laplace_log_evidence()
    r2 = MDLCodecResult(
        codec_name="bad2",
        model_bits=100,
        residual_bits=100,
        n_data_symbols=10,
        n_codec_params=5,
    )
    with pytest.raises(ValueError, match="hessian_logdet"):
        r2.laplace_log_evidence()


# ── ranking ────────────────────────────────────────────────────────────


def test_rank_codecs_smallest_l_total_wins():
    """Smaller L_total → higher posterior weight → rank 1."""
    results = [
        MDLCodecResult(
            codec_name="big_codec", model_bits=10000, residual_bits=10000,
            n_data_symbols=100,
        ),
        MDLCodecResult(
            codec_name="small_codec", model_bits=500, residual_bits=5000,
            n_data_symbols=100,
        ),
        MDLCodecResult(
            codec_name="medium_codec", model_bits=1000, residual_bits=8000,
            n_data_symbols=100,
        ),
    ]
    ranking = rank_codecs(results, temperature=1.0)
    # Ranks are 1-indexed
    assert ranking[0][0] == "small_codec"
    assert ranking[0][2] == 1
    assert ranking[1][0] == "medium_codec"
    assert ranking[2][0] == "big_codec"
    # Posterior weights sum to 1
    total_weight = sum(w for _, w, _ in ranking)
    assert math.isclose(total_weight, 1.0, abs_tol=1e-9)


def test_rank_codecs_with_priors():
    """Negative log_prior_log2_p_model penalizes a codec a-priori."""
    # Two codecs with equal L_total, but second has -log2 prior penalty
    results = [
        MDLCodecResult(
            codec_name="canonical",
            model_bits=100, residual_bits=100,
            n_data_symbols=100,
            log_prior_log2_p_model=0.0,
        ),
        MDLCodecResult(
            codec_name="experimental",
            model_bits=100, residual_bits=100,
            n_data_symbols=100,
            log_prior_log2_p_model=-10.0,  # heavy a-priori penalty
        ),
    ]
    ranking = rank_codecs(results, temperature=1.0)
    assert ranking[0][0] == "canonical"
    # Posterior weight of canonical should be much greater
    canonical_weight = next(w for n, w, _ in ranking if n == "canonical")
    experimental_weight = next(w for n, w, _ in ranking if n == "experimental")
    assert canonical_weight > 100 * experimental_weight


def test_rank_codecs_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        rank_codecs([], temperature=1.0)


def test_rank_codecs_rejects_nonpositive_temperature():
    r = MDLCodecResult(
        codec_name="x", model_bits=1, residual_bits=1, n_data_symbols=1,
    )
    with pytest.raises(ValueError, match="temperature"):
        rank_codecs([r], temperature=0)
    with pytest.raises(ValueError, match="temperature"):
        rank_codecs([r], temperature=-1)


# ── Bayesian model averaging ──────────────────────────────────────────


def test_bayesian_model_average_simple():
    """0.7 * 100 + 0.3 * 200 = 130."""
    val = bayesian_model_average(weights=[0.7, 0.3], predictions=[100.0, 200.0])
    assert math.isclose(val, 130.0)


def test_bayesian_model_average_rejects_unnormalized_weights():
    with pytest.raises(ValueError, match="sum to 1"):
        bayesian_model_average(weights=[0.5, 0.4], predictions=[1.0, 2.0])


def test_bayesian_model_average_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        bayesian_model_average(weights=[0.5, 0.5], predictions=[1.0, 2.0, 3.0])


def test_bayesian_model_average_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        bayesian_model_average(weights=[1.5, -0.5], predictions=[1.0, 2.0])


# ── Occam check ────────────────────────────────────────────────────────


def test_occam_check_passes_thrifty_codec():
    """Codec with model bits within ceiling passes."""
    check = OccamCheck(achievable_savings_bytes=1000)
    codec = MDLCodecResult(
        codec_name="thrifty", model_bits=4000, residual_bits=20000,
        n_data_symbols=100,
    )
    # 4000 bits = 500 bytes < 1000 byte ceiling
    passed, reason = check.evaluate(codec)
    assert passed
    assert "OK" in reason


def test_occam_check_rejects_bloated_codec():
    """Codec with model bits exceeding ceiling fails."""
    check = OccamCheck(achievable_savings_bytes=500)
    codec = MDLCodecResult(
        codec_name="bloated",
        model_bits=10000,  # 1250 bytes > 500 byte ceiling
        residual_bits=20000,
        n_data_symbols=100,
    )
    passed, reason = check.evaluate(codec)
    assert not passed
    assert "REFUSE TO SHIP" in reason


def test_occam_check_safety_margin_loosens():
    """safety_margin=0.5 → ceiling doubles."""
    check_strict = OccamCheck(achievable_savings_bytes=500, safety_margin=1.0)
    check_loose = OccamCheck(achievable_savings_bytes=500, safety_margin=0.5)
    codec = MDLCodecResult(
        codec_name="boundary",
        model_bits=6000,  # 750 bytes — between 500 and 1000
        residual_bits=20000,
        n_data_symbols=100,
    )
    assert not check_strict.evaluate(codec)[0]
    assert check_loose.evaluate(codec)[0]


# ── prior derivation ──────────────────────────────────────────────────


def test_derive_codec_prior_uniform_at_baseline():
    """Fresh codec, no review, no validation → 0.0 (uniform prior)."""
    val = derive_codec_prior_log2(
        codec_lineage_depth=0,
        has_contest_cuda_validation=False,
        has_3_clean_review=False,
    )
    assert val == 0.0


def test_derive_codec_prior_full_credit():
    """Lane G v3 lineage + contest-CUDA + 3-clean → +3.0 log2 bits."""
    val = derive_codec_prior_log2(
        codec_lineage_depth=3,
        has_contest_cuda_validation=True,
        has_3_clean_review=True,
    )
    # Expected: min(0.5*2, 1.0) + 1.0 + 1.0 = 3.0
    assert math.isclose(val, 3.0)


def test_derive_codec_prior_lineage_capped_at_one():
    """Lineage bonus caps at +1.0 even for very deep codecs."""
    val = derive_codec_prior_log2(
        codec_lineage_depth=10,
        has_contest_cuda_validation=False,
        has_3_clean_review=False,
    )
    assert val == 1.0


def test_derive_codec_prior_rejects_negative_depth():
    with pytest.raises(ValueError, match="lineage_depth"):
        derive_codec_prior_log2(
            codec_lineage_depth=-1,
            has_contest_cuda_validation=False,
            has_3_clean_review=False,
        )


# ── version sentinel ──────────────────────────────────────────────────


def test_framework_version_pinned():
    assert MDL_FRAMEWORK_VERSION == 1
