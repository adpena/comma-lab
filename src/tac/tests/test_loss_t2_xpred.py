# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.loss_t2_xpred import x_prediction_loss


def test_x_prediction_equivalent_to_mse_at_sigma_inf() -> None:
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(8, 3, 16, 16, generator=g)
    target = torch.randn(8, 3, 16, 16, generator=g)

    loss = x_prediction_loss(pred, target, sigma=1e6, weighting="v")

    assert torch.allclose(loss, F.mse_loss(pred, target), atol=1e-4, rtol=1e-4)


def test_v_loss_weighting_correct() -> None:
    pred = torch.tensor([1.0, 3.0, -2.0])
    target = torch.tensor([0.0, 1.0, 2.0])
    mse = F.mse_loss(pred, target)

    loss = x_prediction_loss(pred, target, sigma=2.0, weighting="v")

    assert torch.allclose(loss, mse * 1.25)


def test_v_loss_differentiable() -> None:
    pred = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    target = torch.randn(2, 3, dtype=torch.double, requires_grad=True)

    def fn(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_prediction_loss(p, t, sigma=1.5, weighting="v")

    assert torch.autograd.gradcheck(fn, (pred, target))


def test_v_loss_lower_loss_when_pred_correct_at_low_sigma() -> None:
    target = torch.randn(4, 3, 8, 8)

    loss = x_prediction_loss(target, target, sigma=0.01, weighting="v")

    assert loss.item() == 0.0


def test_smoke_train_50_batches() -> None:
    torch.manual_seed(123)
    model = nn.Linear(6, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(64, 6)
    true_w = torch.randn(6, 4)
    target = x @ true_w

    first = None
    last = None
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = x_prediction_loss(pred, target, sigma=1.0, weighting="v")
        assert torch.isfinite(loss)
        if first is None:
            first = loss.item()
        last = loss.item()
        loss.backward()
        opt.step()

    assert first is not None and last is not None
    assert last < first
