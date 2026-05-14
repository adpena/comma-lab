# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.fridrich import segnet_uncertainty_map, variance_weighted_noise
from tac.losses import segnet_uncertainty_weighted_loss, uniward_quant_noise_loss


def _half_textured(batch: int = 2, height: int = 32, width: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    image = torch.empty(batch, 3, height, width)
    image[:, :, :, : width // 2] = 127.5
    image[:, :, :, width // 2 :] = torch.rand(batch, 3, height, width // 2) * 255.0
    return image


def test_variance_weighted_noise_concentrates_in_textured_regions() -> None:
    torch.manual_seed(42)
    image = _half_textured()
    sq_accum = torch.zeros_like(image)
    for _ in range(80):
        sq_accum += variance_weighted_noise(image, base_std=2.0, mode="variance").pow(2)
    std = (sq_accum / 80).sqrt()
    flat = std[:, :, :, : image.shape[-1] // 2].mean().item()
    textured = std[:, :, :, image.shape[-1] // 2 :].mean().item()
    assert textured > flat * 5.0


def test_variance_weighted_noise_supports_inverse_and_wavelet_modes() -> None:
    torch.manual_seed(43)
    image = _half_textured()
    inverse = variance_weighted_noise(image, base_std=1.0, mode="inverse_variance")
    wavelet = variance_weighted_noise(image, base_std=1.0, mode="wavelet_db4")
    assert inverse.shape == image.shape
    assert wavelet.shape == image.shape
    assert inverse.dtype == image.dtype
    assert wavelet.dtype == image.dtype
    with pytest.raises(ValueError, match="mode"):
        variance_weighted_noise(image, base_std=1.0, mode="bogus")


def test_uniward_quant_noise_loss_is_wired_and_differentiable() -> None:
    torch.manual_seed(44)
    target = torch.rand(2, 16, 16, 3) * 255.0
    reconstructed = (target + torch.randn_like(target) * 4.0).requires_grad_(True)
    loss = uniward_quant_noise_loss(
        reconstructed,
        target,
        base_std=1.5,
        mode="wavelet_db4",
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert reconstructed.grad is not None
    assert torch.isfinite(reconstructed.grad).all()
    assert reconstructed.grad.abs().sum() > 0


class _MockSegNet(nn.Module):
    def __init__(self, input_size: tuple[int, int] = (32, 24)) -> None:
        super().__init__()
        self.input_size = input_size
        self.conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        self.eval()

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, -1, ...]
        return F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_segnet_uncertainty_map_shape_range_and_stop_grad() -> None:
    segnet = _MockSegNet(input_size=(24, 16))
    image = (torch.rand(2, 3, 48, 32) * 255.0).requires_grad_(True)
    entropy = segnet_uncertainty_map(segnet, image)
    assert entropy.shape == (2, 1, 24, 16)
    assert entropy.requires_grad is False
    assert entropy.min().item() >= 0.0
    assert entropy.max().item() <= math.log(5.0) + 1e-3


def test_uncertainty_weighted_loss_runs_backward_without_scorer_grads() -> None:
    segnet = _MockSegNet(input_size=(24, 16))
    target = torch.rand(2, 48, 32, 3) * 255.0
    reconstructed = (target + torch.randn_like(target)).requires_grad_(True)
    loss = segnet_uncertainty_weighted_loss(reconstructed, target, segnet)
    loss.backward()
    assert torch.isfinite(loss)
    assert reconstructed.grad is not None
    assert torch.isfinite(reconstructed.grad).all()
    for param in segnet.parameters():
        assert param.grad is None
