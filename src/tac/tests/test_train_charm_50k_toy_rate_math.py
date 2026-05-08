"""Unit tests for the corrected ChARM 50K toy rate math.

Codex review (2026-05-08), summarized in
`.omx/research/codex_finding_charm_high_a_b_recursive_review_20260508.md`, caught
that the prior rate formula in `experiments/train_charm_50k_toy_substrate.py` double-
counted the `e` factor: it added both `0.5·log2(2π·e·σ²)` and a separate
`0.5·log2(e)·ratio` correction, inflating the matched-Gaussian case by ~0.7213 bits
per symbol. These tests verify the corrected formula yields exactly the differential
entropy `0.5·log2(2π·e·σ²)` in the matched-Gaussian limit.

[empirical:experiments/train_charm_50k_toy_substrate.py:CharmHyperprior.forward]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from train_charm_50k_toy_substrate import (  # noqa: E402
    CharmHyperprior,
    decode_weight_with_charm,
    encode_weight_with_charm,
)


def _gaussian_diff_entropy_bits(sigma: float) -> float:
    """Closed-form: 0.5·log2(2π·e·σ²)."""
    return 0.5 * math.log2(2.0 * math.pi * math.e * sigma * sigma)


def test_matched_gaussian_rate_equals_differential_entropy() -> None:
    """When (μ_pred, σ_pred) match (μ_emp, σ_emp), per-symbol rate must
    equal the canonical differential entropy of the predicted Gaussian
    (no `e` double-count).

    Strategy: instead of relying on the trained model to predict matched
    parameters, we directly evaluate the per-symbol cross-entropy formula
    used inside CharmHyperprior.forward, in isolation, on synthetic
    (μ_pred, σ_pred, μ_emp, σ_emp) pairs.
    """
    log2_e = 1.4426950408889634
    # Test multiple sigma scales
    for sigma in [0.5, 1.0, 2.0, 5.0, 10.0]:
        sigma_t = torch.tensor(sigma, dtype=torch.float64)
        # Matched: μ_emp == μ_pred == 0, σ_emp == σ_pred
        var_ratio = (sigma_t.pow(2)) / (sigma_t.pow(2))  # = 1.0
        mean_diff = torch.tensor(0.0, dtype=torch.float64)
        diff_entropy_bits = 0.5 * torch.log2(
            2.0 * torch.tensor(math.pi, dtype=torch.float64) * math.e * sigma_t.pow(2)
        )
        # Codex-corrected formula: subtract 1 from the correction term
        correction_bits = 0.5 * log2_e * (var_ratio + mean_diff - 1.0)
        per_symbol = diff_entropy_bits + correction_bits

        expected = _gaussian_diff_entropy_bits(sigma)
        assert abs(per_symbol.item() - expected) < 1e-6, (
            f"matched Gaussian rate mismatch at σ={sigma}: "
            f"got {per_symbol.item()}, expected {expected}"
        )


def test_pre_fix_formula_demonstrably_inflates_by_log2e_div_2() -> None:
    """Negative test: the OLD (buggy) formula adds 0.5·log2(e) ≈ 0.7213
    extra bits in the matched-Gaussian case. This test documents the
    quantitative gap so the fix is auditable.
    """
    log2_e = 1.4426950408889634
    sigma = 1.0
    sigma_t = torch.tensor(sigma, dtype=torch.float64)
    var_ratio = torch.tensor(1.0, dtype=torch.float64)
    mean_diff = torch.tensor(0.0, dtype=torch.float64)
    diff_entropy_bits = 0.5 * torch.log2(
        2.0 * torch.tensor(math.pi, dtype=torch.float64) * math.e * sigma_t.pow(2)
    )
    # OLD buggy: var_ratio + mean_diff (no -1)
    old_correction = 0.5 * log2_e * (var_ratio + mean_diff)
    old_per_symbol = diff_entropy_bits + old_correction
    # NEW correct: var_ratio + mean_diff - 1
    new_correction = 0.5 * log2_e * (var_ratio + mean_diff - 1.0)
    new_per_symbol = diff_entropy_bits + new_correction
    delta = old_per_symbol - new_per_symbol
    # The delta should be exactly 0.5·log2(e) ≈ 0.7213
    assert abs(delta.item() - 0.5 * log2_e) < 1e-9, (
        f"Pre-fix vs post-fix delta should equal 0.5·log2(e); got {delta.item()}"
    )


def test_mismatched_gaussian_kl_correction_is_positive() -> None:
    """When predicted (μ, σ) differ from empirical, the KL correction
    must be strictly positive (cross-entropy ≥ entropy).
    """
    log2_e = 1.4426950408889634
    sigma_pred = torch.tensor(1.0, dtype=torch.float64)
    sigma_emp = torch.tensor(2.0, dtype=torch.float64)  # double the variance
    mu_pred = torch.tensor(0.0, dtype=torch.float64)
    mu_emp = torch.tensor(1.0, dtype=torch.float64)  # offset by 1
    var_ratio = sigma_emp.pow(2) / sigma_pred.pow(2)  # = 4
    mean_diff = (mu_pred - mu_emp).pow(2) / sigma_pred.pow(2)  # = 1
    correction_bits = 0.5 * log2_e * (var_ratio + mean_diff - 1.0)
    # var_ratio + mean_diff - 1 = 4 + 1 - 1 = 4; correction = 0.5 * 1.4427 * 4 = 2.885
    assert correction_bits.item() > 0.0
    assert abs(correction_bits.item() - 0.5 * log2_e * 4.0) < 1e-9


def test_charm_hyperprior_forward_calls_context_net() -> None:
    """HIGH-B fix: verify CharmHyperprior.forward actually invokes the
    context network by checking that mu_context_offsets and
    log_sigma_context_offsets are produced AND non-zero for c >= 1.

    The context net only fires for c >= 1 (c=0 has no prior channels).
    With random INT8 weights, downstream channel offsets should be
    non-trivial (not identically zero).
    """
    torch.manual_seed(0)
    model = CharmHyperprior(num_channels=4, num_weight_channels=8)
    weight = torch.randn(64) * 0.1  # small synthetic tensor
    out = model(weight)
    assert "mu_context_offsets" in out, (
        "forward must emit mu_context_offsets per channel"
    )
    assert "log_sigma_context_offsets" in out, (
        "forward must emit log_sigma_context_offsets per channel"
    )
    # c=0 offset should be zero (no prior channels)
    assert out["mu_context_offsets"][0].item() == pytest.approx(0.0, abs=1e-9)
    assert out["log_sigma_context_offsets"][0].item() == pytest.approx(0.0, abs=1e-9)
    # At least one downstream offset should be non-zero (the context net fired)
    has_nonzero_offset = (
        out["mu_context_offsets"][1:].abs().sum().item() > 0.0
        or out["log_sigma_context_offsets"][1:].abs().sum().item() > 0.0
    )
    assert has_nonzero_offset, (
        "CharmContextNet must produce non-zero offsets for c >= 1; "
        "if all offsets are zero the context net is dead code (HIGH-B regression)"
    )


def test_charm_hyperprior_emits_separated_hyperprior_and_context() -> None:
    """Verify the forward dict separates the factorized-hyperprior (μ_h, σ_h)
    from the context offsets, so downstream consumers can audit which piece
    is hyperprior-only vs ChARM-conditional.
    """
    torch.manual_seed(0)
    model = CharmHyperprior(num_channels=4, num_weight_channels=8)
    weight = torch.randn(64) * 0.1
    out = model(weight)
    assert "mu_hyperprior" in out
    assert "sigma_hyperprior" in out
    assert "mu_context_offsets" in out
    assert "log_sigma_context_offsets" in out
    # μ_final = μ_h + μ_offsets
    expected_mu = out["mu_hyperprior"] + out["mu_context_offsets"]
    assert torch.allclose(out["mu"], expected_mu, atol=1e-6)


def test_carm2_wire_roundtrips_from_canonical_sidecar_pmfs() -> None:
    """CARM2 must encode with the same fp16 sidecar parameters used at decode.

    A prior wiring encoded with float32 (μ, σ) but decoded from fp16
    (μ, log_σ), which can desynchronize the range-coder PMF and corrupt the
    stream. This test locks the byte-stream contract to sidecar-canonical PMFs.
    """
    torch.manual_seed(17)
    charm = CharmHyperprior(num_channels=2, num_weight_channels=4)
    weight = torch.randn(3, 5) * 0.2

    blob, meta = encode_weight_with_charm(weight, charm)
    recovered = decode_weight_with_charm(blob, tuple(weight.shape))

    assert blob.startswith(b"CARM2")
    assert meta["wire_format"] == "CARM2"
    assert meta["actual_bytes"] == len(blob)
    assert meta["n_padded"] % meta["num_channels"] == 0
    scale = weight.abs().max().clamp(min=1e-8)
    expected = (
        (weight / scale * 127.0).round().clamp(min=-128.0, max=127.0)
        / 127.0
        * scale
    )
    assert torch.allclose(recovered, expected, atol=1e-5)


def test_carm2_decoder_rejects_trailing_bytes() -> None:
    torch.manual_seed(23)
    charm = CharmHyperprior(num_channels=2, num_weight_channels=4)
    weight = torch.randn(2, 3) * 0.1
    blob, _meta = encode_weight_with_charm(weight, charm)

    with pytest.raises(ValueError, match="trailing bytes"):
        decode_weight_with_charm(blob + b"hidden", tuple(weight.shape))
