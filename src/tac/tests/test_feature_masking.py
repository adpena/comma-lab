# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch
import torch.nn as nn

from tac.feature_masking import FeatureMasker
from tac.loss_t2_xpred import x_prediction_loss


def test_mask_token_is_learnable_parameter() -> None:
    module = FeatureMasker(channels=4)

    assert module.mask_token in list(module.parameters())


def test_mask_only_applied_in_final_fraction() -> None:
    module = FeatureMasker(
        channels=3,
        p=1.0,
        mask_ratio=1.0,
        apply_in_final_fraction=0.4,
    )
    x = torch.randn(2, 3, 5, 7)

    out = module(x, training_progress=0.5)

    assert torch.equal(out, x)


def test_mask_ratio_correct() -> None:
    module = FeatureMasker(
        channels=2,
        p=1.0,
        mask_ratio=0.15,
        apply_in_final_fraction=1.0,
    )
    x = torch.ones(8, 2, 64, 64)
    with torch.no_grad():
        module.mask_token.fill_(0.0)

    fractions = []
    for _ in range(100):
        out = module(x, training_progress=1.0)
        masked = (out[:, 0] == 0.0)
        fractions.append(masked.float().mean())

    empirical = torch.stack(fractions).mean().item()
    assert abs(empirical - 0.15) < 0.03


def test_mask_token_replaces_masked_positions() -> None:
    module = FeatureMasker(
        channels=3,
        p=1.0,
        mask_ratio=0.5,
        apply_in_final_fraction=1.0,
    )
    x = torch.full((2, 3, 8, 8), 9.0)
    token = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)
    with torch.no_grad():
        module.mask_token.copy_(token)

    out = module(x, training_progress=1.0)
    masked_positions = out[:, 0] == 1.0

    assert masked_positions.any()
    expected = module.mask_token.expand_as(out)
    assert torch.equal(out[masked_positions.unsqueeze(1).expand_as(out)], expected[masked_positions.unsqueeze(1).expand_as(out)])


def test_differentiable_through_mask_token() -> None:
    module = FeatureMasker(
        channels=3,
        p=1.0,
        mask_ratio=1.0,
        apply_in_final_fraction=1.0,
    )
    x = torch.randn(2, 3, 4, 4, requires_grad=True)

    out = module(x, training_progress=1.0)
    loss = out.mean()
    loss.backward()

    assert module.mask_token.grad is not None
    assert module.mask_token.grad.abs().sum().item() > 0


def test_compose_with_t2_xpred() -> None:
    class TinyRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Conv2d(3, 4, 3, padding=1)
            self.masker = FeatureMasker(
                channels=4,
                p=1.0,
                mask_ratio=1.0,
                apply_in_final_fraction=1.0,
            )
            self.decoder = nn.Conv2d(4, 3, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.encoder(x)
            h = self.masker(h, training_progress=1.0)
            return self.decoder(h)

    torch.manual_seed(7)
    model = TinyRenderer()
    x = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 8, 8)

    pred = model(x)
    loss = x_prediction_loss(pred, target, sigma=1.0, weighting="v")
    assert torch.isfinite(loss)
    loss.backward()

    assert model.masker.mask_token.grad is not None
    assert model.masker.mask_token.grad.abs().sum().item() > 0
