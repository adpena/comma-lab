# SPDX-License-Identifier: MIT
"""Tests for SOAR Decoupled Scale Storage (DSS) opt-in.

Source: Bao et al. 2026 arXiv:2605.12245v1 §4.2 + Algorithm 1.
Lane: lane_soar_cjso_dss_opt_in_dev_20260513.

Backward-compat invariant: with ``decoupled_scale=False`` (default), ALL outputs
must be byte/tensor-identical to the pre-DSS code path.

DSS empirically improves reconstruction MSE on score-AGNOSTIC random weights
because the encoder-side scale (fp32) is allowed to differ from the
decoder-side scale (fp16) — the local search finds the (Δᵍ, Δᵈ) pair that
best matches the codebook's discretization to the actual weight distribution
after fp16 projection of the stored scale.

WARNING per CLAUDE.md Catalog #123: DSS is FALSIFIED on score-gradient-
trained substrates. The MSE-improvement tests below run on score-AGNOSTIC
random data exclusively.
"""
from __future__ import annotations

import torch

from tac.fp4_quantize import (
    DEFAULT_CODEBOOK,
    DEFAULT_DSS_ENCODER_GRID,
    DEFAULT_DSS_STORAGE_GRID,
    _quantize_block,
    _quantize_block_dss,
    _reconstruction_mse_at_scale_pair,
    dequantize_fp4,
    quantize_fp4,
    quantize_state_dict_fp4_dss,
)

# ── _reconstruction_mse_at_scale_pair (DSS objective) ────────────────────


def test_dss_mse_at_pair_zero_scales_returns_inf():
    """Pathological scales should be rejected (return +inf)."""
    block = torch.tensor([0.1, -0.2, 0.3])
    mse = _reconstruction_mse_at_scale_pair(block, DEFAULT_CODEBOOK, 0.0, 1.0)
    assert mse == float("inf")
    mse = _reconstruction_mse_at_scale_pair(block, DEFAULT_CODEBOOK, 1.0, -1.0)
    assert mse == float("inf")


def test_dss_mse_at_pair_zero_block_zero_mse():
    """All-zero block: reconstruction is exact regardless of scale."""
    block = torch.zeros(16)
    mse = _reconstruction_mse_at_scale_pair(block, DEFAULT_CODEBOOK, 1.0, 1.0)
    assert mse == 0.0


def test_dss_mse_at_pair_matches_codebook():
    """When weights equal a codebook value at scale=1.0, MSE should be near zero."""
    # Use codebook value 1.0 * scale → reconstruction recovers exactly.
    cb = DEFAULT_CODEBOOK
    block = torch.tensor([1.0, -1.0, 2.0, -2.0])
    mse = _reconstruction_mse_at_scale_pair(block, cb, 1.0, 1.0)
    # Codebook contains 1.0 and 2.0 → argmin index hits them, MSE ≈ 0.
    assert mse < 1e-10


# ── _quantize_block_dss core ────────────────────────────────────────────


def test_quantize_block_dss_zero_block_returns_indices_zero():
    """All-zero block: zero-indices, scale 1.0 (deterministic)."""
    block = torch.zeros(16)
    indices, signs, scale = _quantize_block_dss(
        block, DEFAULT_CODEBOOK,
        robust_scale=False,
        storage_grid=DEFAULT_DSS_STORAGE_GRID,
        encoder_grid=DEFAULT_DSS_ENCODER_GRID,
    )
    assert torch.all(indices == 0)
    assert scale.item() == 1.0


def test_quantize_block_dss_returns_correct_dtypes():
    """Contract: indices uint8, signs int8, scale scalar tensor."""
    torch.manual_seed(0)
    block = torch.randn(32) * 0.1
    indices, signs, scale = _quantize_block_dss(
        block, DEFAULT_CODEBOOK,
        robust_scale=False,
        storage_grid=DEFAULT_DSS_STORAGE_GRID,
        encoder_grid=DEFAULT_DSS_ENCODER_GRID,
    )
    assert indices.dtype == torch.uint8
    assert signs.dtype == torch.int8
    assert indices.shape == (32,)
    assert signs.shape == (32,)


def test_quantize_block_dss_indices_within_codebook_range():
    """All indices must be in [0, len(codebook)-1] = [0, 7]."""
    torch.manual_seed(1)
    block = torch.randn(32) * 0.5
    indices, signs, scale = _quantize_block_dss(
        block, DEFAULT_CODEBOOK,
        robust_scale=False,
        storage_grid=DEFAULT_DSS_STORAGE_GRID,
        encoder_grid=DEFAULT_DSS_ENCODER_GRID,
    )
    assert indices.max().item() <= 7
    assert indices.min().item() >= 0


def test_quantize_block_dss_signs_in_pm_one():
    """Signs must be in {-1, +1} (zero maps to +1 per the encoder convention)."""
    torch.manual_seed(2)
    block = torch.randn(32) * 0.1
    block[5] = 0.0  # explicit zero
    _, signs, _ = _quantize_block_dss(
        block, DEFAULT_CODEBOOK,
        robust_scale=False,
        storage_grid=DEFAULT_DSS_STORAGE_GRID,
        encoder_grid=DEFAULT_DSS_ENCODER_GRID,
    )
    assert set(signs.tolist()).issubset({-1, 1})


def test_quantize_block_dss_robust_scale_compatibility():
    """DSS composes with robust_scale=True (percentile-based init)."""
    torch.manual_seed(3)
    block = torch.randn(32) * 0.1
    block[0] = 5.0  # outlier
    # Should not raise.
    indices, signs, scale = _quantize_block_dss(
        block, DEFAULT_CODEBOOK,
        robust_scale=True,
        storage_grid=DEFAULT_DSS_STORAGE_GRID,
        encoder_grid=DEFAULT_DSS_ENCODER_GRID,
    )
    assert torch.isfinite(scale).item()


# ── quantize_fp4 opt-in flag ────────────────────────────────────────────


def test_quantize_fp4_default_off_byte_identical():
    """Backward-compat: decoupled_scale=False matches default everywhere."""
    torch.manual_seed(4)
    sd = {"w": torch.randn(64, 32) * 0.1}
    p_default = quantize_fp4(sd)
    p_off = quantize_fp4(sd, decoupled_scale=False)
    for k in p_default:
        if torch.is_tensor(p_default[k]):
            assert torch.equal(p_default[k], p_off[k]), f"mismatch on {k}"
        elif isinstance(p_default[k], (list, int)):
            assert p_default[k] == p_off[k]


def test_quantize_fp4_default_off_no_dss_metadata():
    """When DSS is off, no scale_optimizer provenance is stamped."""
    sd = {"w": torch.randn(32, 16) * 0.1}
    p = quantize_fp4(sd, decoupled_scale=False)
    assert "__scale_optimizer__" not in p


def test_quantize_fp4_dss_stamps_provenance():
    """DSS-on archive carries scale_optimizer provenance."""
    sd = {"w": torch.randn(32, 16) * 0.1}
    p = quantize_fp4(sd, decoupled_scale=True)
    assert p["__scale_optimizer__"] == "dss_soar_v1"
    assert p["__scale_optimizer_paper__"] == "arXiv:2605.12245v1"


def test_quantize_fp4_dss_byte_format_unchanged():
    """DSS does not change packed/scales tensor shapes — same archive bytes
    structure, just different scale VALUES."""
    torch.manual_seed(5)
    sd = {"w": torch.randn(64, 32) * 0.1}
    p_default = quantize_fp4(sd, decoupled_scale=False)
    p_dss = quantize_fp4(sd, decoupled_scale=True)
    assert p_default["w.packed"].shape == p_dss["w.packed"].shape
    assert p_default["w.packed"].dtype == p_dss["w.packed"].dtype
    assert p_default["w.scales"].shape == p_dss["w.scales"].shape
    assert p_default["w.scales"].dtype == p_dss["w.scales"].dtype  # both fp16
    assert p_default["w.numel"] == p_dss["w.numel"]


def test_quantize_fp4_dss_roundtrip_finite():
    """DSS roundtrip produces finite output of the correct shape."""
    torch.manual_seed(6)
    sd = {"w": torch.randn(64, 32) * 0.1}
    p = quantize_state_dict_fp4_dss(sd)
    dq = dequantize_fp4(p)
    assert dq["w"].shape == sd["w"].shape
    assert torch.isfinite(dq["w"]).all()


def test_quantize_fp4_dss_improves_mse_on_random_weights():
    """Headline empirical claim: DSS reduces reconstruction MSE on random
    score-AGNOSTIC weights at the same byte cost.

    SOAR §4.2 + Figure 4 predicts ~0.5-5% MSE drop. We require at least a
    non-regression (DSS MSE <= max-rule MSE within numerical tolerance).
    """
    torch.manual_seed(7)
    sd = {"w": torch.randn(128, 64) * 0.1}
    p_default = quantize_fp4(sd, decoupled_scale=False)
    p_dss = quantize_fp4(sd, decoupled_scale=True)
    dq_default = dequantize_fp4(p_default)
    dq_dss = dequantize_fp4(p_dss)
    mse_default = (dq_default["w"] - sd["w"]).pow(2).mean().item()
    mse_dss = (dq_dss["w"] - sd["w"]).pow(2).mean().item()
    # Allow a small numerical wobble (fp16 projection); require non-regression.
    assert mse_dss <= mse_default * 1.001, (
        f"DSS regressed: default={mse_default:.6g}  DSS={mse_dss:.6g}"
    )


def test_quantize_fp4_dss_skips_ndim_lt_2_tensors():
    """ndim < 2 buffers (e.g., 1-D bias) are passed through unmodified."""
    sd = {
        "w_2d": torch.randn(8, 4) * 0.1,
        "bias_1d": torch.randn(4) * 0.05,
    }
    p = quantize_state_dict_fp4_dss(sd)
    # 2-D tensor was quantized.
    assert "w_2d.packed" in p
    # 1-D tensor was passed through.
    assert torch.allclose(p["bias_1d"], sd["bias_1d"])


def test_quantize_fp4_dss_non_floating_passthrough():
    """Non-float tensors are passed through verbatim under DSS too."""
    sd = {
        "w": torch.randn(8, 4) * 0.1,
        "indices": torch.arange(10, dtype=torch.int64),
    }
    p = quantize_state_dict_fp4_dss(sd)
    assert torch.equal(p["indices"], sd["indices"])


# ── quantize_state_dict_fp4_dss wrapper ─────────────────────────────────


def test_quantize_state_dict_fp4_dss_wrapper_equivalent_to_flag():
    """Wrapper matches calling quantize_fp4 with decoupled_scale=True."""
    torch.manual_seed(8)
    sd = {"w": torch.randn(32, 16) * 0.1}
    p_wrapper = quantize_state_dict_fp4_dss(sd)
    p_flag = quantize_fp4(sd, decoupled_scale=True)
    # Compare tensor values (fp16 storage may quantize identical scales).
    for k in p_wrapper:
        if torch.is_tensor(p_wrapper[k]):
            assert torch.equal(p_wrapper[k], p_flag[k]), f"mismatch on {k}"


def test_quantize_state_dict_fp4_dss_default_grids_match_module_constants():
    """SOAR Algorithm 1 default K_d=5 storage candidates, K_g=3 encoder."""
    assert len(DEFAULT_DSS_STORAGE_GRID) == 5
    assert len(DEFAULT_DSS_ENCODER_GRID) == 3
    # Grid must be centered on 1.0 (the baseline scale).
    assert 1.0 in DEFAULT_DSS_STORAGE_GRID
    assert 1.0 in DEFAULT_DSS_ENCODER_GRID


# ── DSS grid customization ──────────────────────────────────────────────


def test_quantize_fp4_dss_custom_grid_singleton_equals_default():
    """If the storage AND encoder grid are singletons of 1.0 (i.e. no search),
    DSS produces the SAME scale as the max-rule baseline (modulo fp16
    projection of the same value).
    """
    torch.manual_seed(9)
    sd = {"w": torch.randn(32, 16) * 0.1}
    p_default = quantize_fp4(sd, decoupled_scale=False)
    p_dss_singleton = quantize_fp4(
        sd, decoupled_scale=True,
        dss_storage_grid=(1.0,),
        dss_encoder_grid=(1.0,),
    )
    # The fp16-projected scales should match.
    s_default = p_default["w.scales"]
    s_dss = p_dss_singleton["w.scales"]
    # fp16 projection in both paths → exact equality.
    assert torch.allclose(s_default.float(), s_dss.float(), atol=1e-4)


def test_quantize_fp4_dss_larger_grid_at_least_as_good():
    """Larger search grid never regresses MSE (monotonicity of the search)."""
    torch.manual_seed(10)
    sd = {"w": torch.randn(64, 32) * 0.2}
    # Small grid
    p_small = quantize_fp4(
        sd, decoupled_scale=True,
        dss_storage_grid=(1.0,),
        dss_encoder_grid=(1.0,),
    )
    # Default (larger) grid
    p_large = quantize_fp4(sd, decoupled_scale=True)
    dq_small = dequantize_fp4(p_small)
    dq_large = dequantize_fp4(p_large)
    mse_small = (dq_small["w"] - sd["w"]).pow(2).mean().item()
    mse_large = (dq_large["w"] - sd["w"]).pow(2).mean().item()
    assert mse_large <= mse_small * 1.001, (
        f"larger grid regressed: small={mse_small:.6g}  large={mse_large:.6g}"
    )


def test_quantize_block_dss_via_quantize_block_decoupled_scale_flag():
    """Calling _quantize_block with decoupled_scale=True dispatches to DSS."""
    torch.manual_seed(11)
    block = torch.randn(32) * 0.1
    out_dss = _quantize_block(block, DEFAULT_CODEBOOK, decoupled_scale=True)
    out_direct = _quantize_block_dss(
        block, DEFAULT_CODEBOOK,
        robust_scale=False,
        storage_grid=DEFAULT_DSS_STORAGE_GRID,
        encoder_grid=DEFAULT_DSS_ENCODER_GRID,
    )
    assert torch.equal(out_dss[0], out_direct[0])
    assert torch.equal(out_dss[1], out_direct[1])
    assert torch.equal(out_dss[2], out_direct[2])
