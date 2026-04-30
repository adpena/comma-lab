"""Tests for Lane 20 β-variant — sensitivity-conditioned Ballé hyperprior."""
from __future__ import annotations

import pytest
import torch

from tac.balle_sensitivity_weighted import (
    SensitivityConditionedHyperDecoder,
    SensitivityWeightedBalleError,
    aggregate_pixel_sensitivity_to_blocks,
    apply_multiplicative_sensitivity_to_sigma,
)


def test_aggregate_mean_block_sensitivity() -> None:
    s = torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0, 1.0])
    out = aggregate_pixel_sensitivity_to_blocks(
        pixel_sensitivity=s,
        block_size=2,
        aggregate="mean",
    )
    expected = torch.tensor([2.0, 3.0, 3.0])
    assert torch.allclose(out, expected)


def test_aggregate_max_block_sensitivity() -> None:
    s = torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0, 1.0])
    out = aggregate_pixel_sensitivity_to_blocks(
        pixel_sensitivity=s,
        block_size=2,
        aggregate="max",
    )
    expected = torch.tensor([3.0, 4.0, 5.0])
    assert torch.allclose(out, expected)


def test_aggregate_pads_when_not_divisible() -> None:
    s = torch.tensor([1.0, 2.0, 3.0])  # 3 elements, block_size=2 → 2 blocks
    out = aggregate_pixel_sensitivity_to_blocks(
        pixel_sensitivity=s,
        block_size=2,
        aggregate="mean",
    )
    assert out.shape == (2,)
    # Last block: [3, 0] → mean = 1.5
    assert out[1].item() == pytest.approx(1.5)


def test_aggregate_rejects_bad_inputs() -> None:
    with pytest.raises(SensitivityWeightedBalleError, match="must be 1-D"):
        aggregate_pixel_sensitivity_to_blocks(
            pixel_sensitivity=torch.zeros(2, 2),
            block_size=2,
        )
    with pytest.raises(SensitivityWeightedBalleError, match="positive int"):
        aggregate_pixel_sensitivity_to_blocks(
            pixel_sensitivity=torch.zeros(4),
            block_size=0,
        )
    with pytest.raises(SensitivityWeightedBalleError, match="aggregate"):
        aggregate_pixel_sensitivity_to_blocks(
            pixel_sensitivity=torch.zeros(4),
            block_size=2,
            aggregate="bogus",  # type: ignore[arg-type]
        )
    with pytest.raises(SensitivityWeightedBalleError, match="non-negative"):
        aggregate_pixel_sensitivity_to_blocks(
            pixel_sensitivity=torch.tensor([-1.0, 1.0, 2.0]),
            block_size=1,
        )


def test_hyper_decoder_init_and_forward_shape() -> None:
    torch.manual_seed(2026)
    dec = SensitivityConditionedHyperDecoder(
        z_dim=4,
        hidden_dim=8,
        sigma_min=0.1,
        sigma_max=4.0,
    )
    z = torch.randn(7, 4)
    sens = torch.linspace(0.0, 1.0, 7)
    sigma = dec(z, sens)
    assert sigma.shape == (7,)
    assert (sigma >= 0.1).all()
    assert (sigma <= 4.0).all()


def test_hyper_decoder_rejects_bad_z_shape() -> None:
    dec = SensitivityConditionedHyperDecoder(
        z_dim=4,
        hidden_dim=8,
        sigma_min=0.1,
        sigma_max=4.0,
    )
    with pytest.raises(SensitivityWeightedBalleError, match="z shape"):
        dec(torch.randn(7, 8), torch.zeros(7))


def test_hyper_decoder_rejects_bad_sensitivity_shape() -> None:
    dec = SensitivityConditionedHyperDecoder(
        z_dim=4,
        hidden_dim=8,
        sigma_min=0.1,
        sigma_max=4.0,
    )
    z = torch.randn(7, 4)
    with pytest.raises(SensitivityWeightedBalleError, match="sensitivity shape"):
        dec(z, torch.zeros(8))


def test_hyper_decoder_init_validates_args() -> None:
    with pytest.raises(SensitivityWeightedBalleError, match="z_dim and hidden_dim"):
        SensitivityConditionedHyperDecoder(
            z_dim=0,
            hidden_dim=8,
            sigma_min=0.1,
            sigma_max=4.0,
        )
    with pytest.raises(SensitivityWeightedBalleError, match="sigma"):
        SensitivityConditionedHyperDecoder(
            z_dim=4,
            hidden_dim=8,
            sigma_min=4.0,
            sigma_max=0.1,
        )


def test_multiplicative_sigma_tightens_on_high_sensitivity() -> None:
    sigma_baseline = torch.tensor([1.0, 1.0, 1.0])
    sens = torch.tensor([0.0, 0.5, 1.0])
    sigma_beta = apply_multiplicative_sensitivity_to_sigma(
        sigma_baseline=sigma_baseline,
        per_block_sensitivity=sens,
        alpha=1.0,
        sigma_min=0.05,
        sigma_max=4.0,
    )
    # max-normalized sens = [0, 0.5, 1]. σ_β = 1 / (1 + 1*sens_norm) = [1, 0.667, 0.5]
    assert sigma_beta[0].item() == pytest.approx(1.0)
    assert sigma_beta[1].item() == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert sigma_beta[2].item() == pytest.approx(0.5, rel=1e-3)


def test_multiplicative_sigma_clamps_to_bounds() -> None:
    sigma_baseline = torch.tensor([10.0, 0.001])
    sens = torch.tensor([0.0, 1.0])
    sigma_beta = apply_multiplicative_sensitivity_to_sigma(
        sigma_baseline=sigma_baseline,
        per_block_sensitivity=sens,
        alpha=1.0,
        sigma_min=0.1,
        sigma_max=4.0,
    )
    assert sigma_beta[0].item() == pytest.approx(4.0)  # clamped to max
    assert sigma_beta[1].item() == pytest.approx(0.1)  # clamped to min


def test_multiplicative_rejects_bad_alpha_and_sigma() -> None:
    sigma_baseline = torch.tensor([1.0])
    sens = torch.tensor([1.0])
    with pytest.raises(SensitivityWeightedBalleError, match="alpha"):
        apply_multiplicative_sensitivity_to_sigma(
            sigma_baseline=sigma_baseline,
            per_block_sensitivity=sens,
            alpha=0.0,
            sigma_min=0.1,
            sigma_max=4.0,
        )
    with pytest.raises(SensitivityWeightedBalleError, match="sigma"):
        apply_multiplicative_sensitivity_to_sigma(
            sigma_baseline=sigma_baseline,
            per_block_sensitivity=sens,
            alpha=1.0,
            sigma_min=4.0,
            sigma_max=0.1,
        )


def test_multiplicative_rejects_bad_block_count() -> None:
    sigma_baseline = torch.tensor([1.0, 2.0, 3.0])
    sens = torch.tensor([1.0, 2.0])  # mismatch
    with pytest.raises(SensitivityWeightedBalleError, match="sensitivity shape"):
        apply_multiplicative_sensitivity_to_sigma(
            sigma_baseline=sigma_baseline,
            per_block_sensitivity=sens,
            alpha=1.0,
            sigma_min=0.1,
            sigma_max=4.0,
        )
