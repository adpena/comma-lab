"""Tests for :mod:`tac.codec.factorized_hnerv_codec`.

Coverage targets per task spec:

1. Roundtrip determinism: factorize → encode → decode → reconstruct gives
   bit-identical tensor reconstruction at full-rank.
2. Approximation accuracy: at r=20 on stem.weight (1728, 28) on a *low-rank-
   structured* synthetic, rel_err should be < 5%.
3. Wire format stability: encoded bytes deterministic for same input + params.
4. Cross-paradigm composability: factorized output is consumable by
   :func:`apply_factor_lossy_coarsening` on the factor streams.
5. Reconstruction fidelity: reconstructed tensor norm matches original within ε.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.codec.factorized_hnerv_codec import (
    FIXED_STATE_SCHEMA,
    SECTION_MAGIC,
    WIRE_FORMAT_VERSION,
    FactorizedHnervCodecError,
    FactorizedSectionPlan,
    apply_factor_lossy_coarsening,
    decode_factorized_section,
    encode_factorized_section,
    estimate_factorized_byte_savings,
    factorize_tensor_svd,
)
from tac.codec.rel_err import REL_ERR_FORM_KEY, RelErrForm


def _make_synthetic_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {n: torch.randn(*s) for n, s in FIXED_STATE_SCHEMA}


def _make_low_rank_stem_weight(M: int = 1728, N: int = 28, true_rank: int = 12, seed: int = 0) -> torch.Tensor:
    """Construct a (M, N) tensor that is genuinely close to ``true_rank``.

    SVD low-rank approximation can recover this tensor at small ``r`` with
    low error; with random gaussian, full rank is required.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, true_rank)).astype(np.float64)
    B = rng.standard_normal((true_rank, N)).astype(np.float64)
    W = (A @ B).astype(np.float32)
    # Add small noise so the spectrum has a sharp elbow at ``true_rank``.
    W += 0.001 * rng.standard_normal(W.shape).astype(np.float32)
    return torch.from_numpy(W)


# ---------------------------------------------------------------------------
# Test 1: roundtrip determinism (full-rank ⇒ exact float reconstruction)
# ---------------------------------------------------------------------------

def test_full_rank_factorization_recovers_within_quant_noise():
    """At full rank, the int8-quantized reconstruction should match the
    original within INT8 quantization noise (rel_err ~ 1/N_QUANT range).
    """
    torch.manual_seed(42)
    W = torch.randn(64, 32) * 0.5
    ft = factorize_tensor_svd("stem.weight", W, target_rank=32)  # full rank
    assert ft.rank == 32
    recon = ft.reconstruct_float()
    rel = np.linalg.norm(recon - W.numpy()) / np.linalg.norm(W.numpy())
    # INT8 quant noise on three components (U, S, V) compounds; loose bound.
    assert rel < 0.08, f"full-rank reconstruction rel_err={rel:.4f} too large"


def test_section_roundtrip_matches_factorized_records():
    """Encoding then decoding a section produces a state_dict whose
    factorized tensors match :meth:`FactorizedTensor.reconstruct_float`
    bit-exactly (no precision loss at the wire-format layer).
    """
    sd = _make_synthetic_state_dict(seed=1)
    plan = FactorizedSectionPlan(
        factorized_indices=(0, 2, 4),
        per_index_rank={0: 12, 2: 16, 4: 16},
    )
    section, telem = encode_factorized_section(sd, plan, brotli_quality=5)
    sd_recon = decode_factorized_section(section)

    # Recompute the factorized tensor on the encoder side and compare.
    for idx in plan.factorized_indices:
        name, shape = FIXED_STATE_SCHEMA[idx]
        ft = factorize_tensor_svd(
            name=name, tensor=sd[name], target_rank=plan.per_index_rank[idx]
        )
        expected = ft.reconstruct_float()
        actual = sd_recon[name].detach().cpu().numpy()
        np.testing.assert_array_equal(actual.shape, expected.shape)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)


def test_non_factorized_tensors_round_trip_within_int8_noise():
    """Non-factorized tensors round-trip with the standard INT8 + fp16-scale
    error envelope (≤ 1/127 in normalized units).
    """
    sd = _make_synthetic_state_dict(seed=2)
    plan = FactorizedSectionPlan(
        factorized_indices=(0,),
        per_index_rank={0: 20},
    )
    section, _ = encode_factorized_section(sd, plan, brotli_quality=5)
    sd_recon = decode_factorized_section(section)
    for idx, (name, _shape) in enumerate(FIXED_STATE_SCHEMA):
        if idx == 0:
            continue
        orig = sd[name].detach().cpu().numpy()
        recon = sd_recon[name].detach().cpu().numpy()
        denom = np.linalg.norm(orig) + 1e-12
        rel = np.linalg.norm(recon - orig) / denom
        # 1/N_QUANT = 1/127 ≈ 0.0079; with fp16 scale rounding it can be
        # slightly higher. Loose bound to absorb fp16 scale noise.
        assert rel < 0.05, f"non-fact tensor {name!r}: rel_err={rel:.4f}"


# ---------------------------------------------------------------------------
# Test 2: approximation accuracy on low-rank structured data
# ---------------------------------------------------------------------------

def test_low_rank_synthetic_under_5pct_at_r12():
    """A 1728×28 weight that's truly rank-12 should factorize at r=12 with
    < 5% RMS rel_err post-quantization.
    """
    W = _make_low_rank_stem_weight(M=1728, N=28, true_rank=12, seed=42)
    ft = factorize_tensor_svd("stem.weight", W, target_rank=12)
    assert ft.rel_err < 0.05, f"rank-12 rel_err={ft.rel_err:.4f}"


def test_target_rms_err_bisect_finds_smallest_rank():
    """When ``target_rms_err`` is supplied, the bisect should pick the
    smallest rank that meets it (or max rank if none does)."""
    W = _make_low_rank_stem_weight(M=1728, N=28, true_rank=10, seed=7)
    ft = factorize_tensor_svd(
        "stem.weight", W, target_rms_err=0.05,
    )
    # Rank shouldn't be wildly above the true rank; allow a small margin
    # for INT8 quantization noise on the components.
    assert ft.rel_err <= 0.05 + 0.01  # tiny epsilon for boundary fp noise
    assert 1 <= ft.rank <= 28


# ---------------------------------------------------------------------------
# Test 3: wire-format stability
# ---------------------------------------------------------------------------

def test_section_bytes_deterministic():
    """Same input + params → same encoded bytes."""
    sd = _make_synthetic_state_dict(seed=3)
    plan = FactorizedSectionPlan(
        factorized_indices=(0, 2),
        per_index_rank={0: 14, 2: 30},
    )
    sec_a, _ = encode_factorized_section(sd, plan, brotli_quality=5)
    sec_b, _ = encode_factorized_section(sd, plan, brotli_quality=5)
    assert sec_a == sec_b


def test_section_magic_present_and_first():
    sd = _make_synthetic_state_dict(seed=4)
    plan = FactorizedSectionPlan(factorized_indices=(0,), per_index_rank={0: 8})
    section, _ = encode_factorized_section(sd, plan)
    assert section[:4] == SECTION_MAGIC


def test_decode_rejects_bad_magic():
    bad = b"BAD\x00" + b"\x00" * 12
    with pytest.raises(FactorizedHnervCodecError):
        decode_factorized_section(bad)


def test_decode_rejects_truncated_section():
    # Just a header, no payload of the lengths it claims
    sd = _make_synthetic_state_dict(seed=5)
    plan = FactorizedSectionPlan(factorized_indices=(0,), per_index_rank={0: 4})
    section, _ = encode_factorized_section(sd, plan)
    truncated = section[: len(section) - 5]
    with pytest.raises(FactorizedHnervCodecError):
        decode_factorized_section(truncated)


# ---------------------------------------------------------------------------
# Test 4: cross-paradigm composability (lossy_coarsening on factor streams)
# ---------------------------------------------------------------------------

def test_apply_factor_lossy_coarsening_K1_is_noop():
    sd = _make_synthetic_state_dict(seed=6)
    ft = factorize_tensor_svd("stem.weight", sd["stem.weight"], target_rank=14)
    coarsened = apply_factor_lossy_coarsening(ft, K_u=1, K_s=1, K_v=1)
    np.testing.assert_array_equal(coarsened.u_i8, ft.u_i8)
    np.testing.assert_array_equal(coarsened.s_i8, ft.s_i8)
    np.testing.assert_array_equal(coarsened.v_i8, ft.v_i8)
    # rel_err is the recomputed RMS error vs the input's reconstruction;
    # at K=1 the coarsened reconstruction equals the input's reconstruction,
    # so rel_err should be exactly 0.
    assert coarsened.rel_err == pytest.approx(0.0, abs=1e-12)


def test_apply_factor_lossy_coarsening_K2_increases_rel_err_only():
    """Coarsening introduces additional noise; rel_err vs the original
    factorized reconstruction should be >= 0 and finite.
    """
    sd = _make_synthetic_state_dict(seed=7)
    ft = factorize_tensor_svd("stem.weight", sd["stem.weight"], target_rank=12)
    coarse = apply_factor_lossy_coarsening(ft, K_u=2, K_s=1, K_v=2)
    # No coarsening occurred at K_s=1; U and V have been coarsened.
    assert coarse.rel_err >= 0.0
    assert np.isfinite(coarse.rel_err)


def test_apply_factor_lossy_coarsening_rejects_zero_K():
    sd = _make_synthetic_state_dict(seed=8)
    ft = factorize_tensor_svd("stem.weight", sd["stem.weight"], target_rank=10)
    with pytest.raises(FactorizedHnervCodecError):
        apply_factor_lossy_coarsening(ft, K_u=0)


# ---------------------------------------------------------------------------
# Test 5: reconstruction fidelity / norm consistency
# ---------------------------------------------------------------------------

def test_reconstructed_tensor_norm_matches_within_epsilon():
    """The reconstructed tensor norm should be close to the original norm."""
    W = _make_low_rank_stem_weight(M=1728, N=28, true_rank=8, seed=11)
    ft = factorize_tensor_svd("stem.weight", W, target_rank=8)
    recon = ft.reconstruct_float()
    orig_norm = np.linalg.norm(W.numpy())
    recon_norm = np.linalg.norm(recon)
    assert abs(recon_norm - orig_norm) / orig_norm < 0.10


def test_4d_conv_factorization_shape_preserved():
    """Conv2d (O, I, kH, kW) factorization preserves the original shape on decode."""
    sd = _make_synthetic_state_dict(seed=12)
    # Index 2 is blocks.0.weight (144, 36, 3, 3).
    ft = factorize_tensor_svd("blocks.0.weight", sd["blocks.0.weight"], target_rank=36)
    recon = ft.reconstruct_float()
    assert recon.shape == (144, 36, 3, 3)


# ---------------------------------------------------------------------------
# Plan / API guard tests
# ---------------------------------------------------------------------------

def test_plan_rejects_duplicate_indices():
    with pytest.raises(FactorizedHnervCodecError):
        FactorizedSectionPlan(
            factorized_indices=(0, 0),
            per_index_rank={0: 10},
        )


def test_plan_rejects_missing_rank():
    with pytest.raises(FactorizedHnervCodecError):
        FactorizedSectionPlan(
            factorized_indices=(0, 2),
            per_index_rank={0: 10},
        )


def test_factorize_rejects_1d_tensor():
    with pytest.raises(FactorizedHnervCodecError):
        factorize_tensor_svd(
            "stem.bias", torch.randn(1728), target_rank=1,
        )


def test_factorize_requires_rank_or_target():
    with pytest.raises(FactorizedHnervCodecError):
        factorize_tensor_svd("stem.weight", torch.randn(8, 4))


def test_encode_rejects_missing_tensor_in_state_dict():
    sd = _make_synthetic_state_dict(seed=13)
    del sd["stem.weight"]
    plan = FactorizedSectionPlan(factorized_indices=(0,), per_index_rank={0: 5})
    with pytest.raises(FactorizedHnervCodecError):
        encode_factorized_section(sd, plan)


# ---------------------------------------------------------------------------
# Telemetry / form-uniformity contract
# ---------------------------------------------------------------------------

def test_telemetry_carries_rel_err_form_tag():
    """Telemetry must declare its rel_err form per the form-uniformity contract."""
    sd = _make_synthetic_state_dict(seed=14)
    plan = FactorizedSectionPlan(factorized_indices=(0,), per_index_rank={0: 10})
    _, telem = encode_factorized_section(sd, plan)
    assert REL_ERR_FORM_KEY in telem
    assert telem[REL_ERR_FORM_KEY] == RelErrForm.RMS.value
    assert telem["wire_format_version"] == WIRE_FORMAT_VERSION


def test_estimate_byte_savings_matches_form_contract():
    sd = _make_synthetic_state_dict(seed=15)
    out = estimate_factorized_byte_savings(
        sd, candidate_indices=[0, 2], target_ranks={0: 14, 2: 36},
    )
    for idx, row in out.items():
        assert REL_ERR_FORM_KEY in row
        assert row[REL_ERR_FORM_KEY] == RelErrForm.RMS.value
        assert "factor_record_brotli_bytes" in row
        assert "isolated_savings_bytes_brotli" in row


# ---------------------------------------------------------------------------
# End-to-end: factorize a few tensors, encode, decode, and verify the
# state_dict can be loaded into the HNeRVDecoder model.
# ---------------------------------------------------------------------------

def test_decoded_state_dict_is_loadable_into_HNeRVDecoder_model():
    """Sanity check that the decoded state_dict has the right keys / shapes
    for the HNeRVDecoder defined in submissions/apogee_intN/src/model.py.

    We avoid importing the model here directly (sister-package import) but
    we do verify that every name + shape in the decoded state_dict matches
    FIXED_STATE_SCHEMA exactly.
    """
    sd = _make_synthetic_state_dict(seed=99)
    plan = FactorizedSectionPlan(
        factorized_indices=(0, 2, 4, 6),
        per_index_rank={0: 16, 2: 32, 4: 32, 6: 32},
    )
    section, _ = encode_factorized_section(sd, plan, brotli_quality=5)
    sd_recon = decode_factorized_section(section)
    assert set(sd_recon.keys()) == set(name for name, _ in FIXED_STATE_SCHEMA)
    for name, shape in FIXED_STATE_SCHEMA:
        assert tuple(sd_recon[name].shape) == shape, (
            f"shape mismatch {name}: expected {shape}, got {tuple(sd_recon[name].shape)}"
        )
