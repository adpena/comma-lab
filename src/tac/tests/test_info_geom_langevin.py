from __future__ import annotations

import math

import pytest
import torch

from tac.optimization.iglt import IGLTOptimizer
from tac.optimization.info_geom_langevin import (
    InfoGeomLangevinConfig,
    InfoGeomLangevinOptimizer,
    build_info_geom_langevin_optimizer,
    fisher_diagonal_ema,
    precondition_gradient,
)


def test_config_temperature_schedule_and_builder_are_deterministic() -> None:
    cfg = InfoGeomLangevinConfig(
        lr=1e-3,
        T_init=0.2,
        T_final=0.0,
        n_steps=10,
        warmup_steps=0,
        noise_seed=17,
    )
    assert math.isclose(cfg.temperature_at(0), 0.2)
    assert math.isclose(cfg.temperature_at(10), 0.0)

    p1 = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    p2 = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    opt1 = build_info_geom_langevin_optimizer([p1], cfg)
    opt2 = build_info_geom_langevin_optimizer([p2], cfg)

    assert isinstance(opt1, IGLTOptimizer)
    assert InfoGeomLangevinOptimizer is IGLTOptimizer
    for _ in range(4):
        p1.grad = torch.tensor([0.5, -0.25])
        p2.grad = torch.tensor([0.5, -0.25])
        opt1.step()
        opt2.step()
    assert torch.allclose(p1, p2, atol=0.0, rtol=0.0)


def test_fisher_diagonal_ema_is_checkable_and_non_mutating() -> None:
    previous = torch.tensor([2.0, 8.0])
    gradient = torch.tensor([3.0, 4.0])

    updated = fisher_diagonal_ema(previous, gradient, decay=0.25)

    assert torch.allclose(updated, torch.tensor([7.25, 14.0]))
    assert torch.allclose(previous, torch.tensor([2.0, 8.0]))


def test_precondition_gradient_supports_inverse_sqrt_and_inverse() -> None:
    gradient = torch.tensor([2.0, 4.0])
    fisher = torch.tensor([4.0, 16.0])

    inv_sqrt = precondition_gradient(gradient, fisher, eps=1e-6)
    inv = precondition_gradient(gradient, fisher, eps=1e-6, power="inverse")

    assert torch.allclose(inv_sqrt, torch.tensor([1.0, 1.0]), atol=1e-5)
    assert torch.allclose(inv, torch.tensor([0.5, 0.25]), atol=1e-5)


def test_info_geom_langevin_rejects_invalid_configs_and_tensors() -> None:
    with pytest.raises(ValueError, match="T_init"):
        InfoGeomLangevinConfig(T_init=0.0, T_final=1.0).validate()
    with pytest.raises(ValueError, match="lr must be finite"):
        InfoGeomLangevinConfig(lr=float("nan")).validate()
    with pytest.raises(ValueError, match="temperatures must be finite"):
        InfoGeomLangevinConfig(T_init=float("inf")).validate()
    with pytest.raises(ValueError, match="weight_decay"):
        InfoGeomLangevinConfig(weight_decay=float("nan")).validate()
    with pytest.raises(ValueError, match="fisher_eps"):
        InfoGeomLangevinConfig(fisher_eps=float("inf")).validate()
    with pytest.raises(ValueError, match="schedule"):
        InfoGeomLangevinConfig(schedule="bad").validate()
    with pytest.raises(ValueError, match="decay"):
        fisher_diagonal_ema(None, torch.ones(2), decay=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        precondition_gradient(torch.ones(1), torch.tensor([-1.0]))
