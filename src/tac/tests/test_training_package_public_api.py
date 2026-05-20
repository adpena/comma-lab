# SPDX-License-Identifier: MIT
"""Regression coverage for the ``tac.training`` package public API."""

from __future__ import annotations

import importlib

import torch
from torch import nn

import tac.training as training_pkg
from tac.training import EMA, SWA, KalmanWeightFilter, TrainConfig, Trainer


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))


def test_training_package_exports_legacy_public_api() -> None:
    assert EMA is training_pkg.EMA
    assert TrainConfig is training_pkg.TrainConfig
    assert Trainer is training_pkg.Trainer
    assert SWA is training_pkg.SWA
    assert KalmanWeightFilter is training_pkg.KalmanWeightFilter


def test_ema_tracks_float_weights_and_copies_integer_buffers() -> None:
    model = _ToyModel()
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
    ema = EMA(model, decay=0.5)

    with torch.no_grad():
        model.linear.weight.fill_(3.0)
        model.step_count.fill_(7)
    ema.update(model)

    assert torch.allclose(
        ema.shadow["linear.weight"],
        torch.full_like(ema.shadow["linear.weight"], 2.0),
    )
    assert ema.shadow["step_count"].item() == 7


def test_ema_decay_formula_remains_available_from_package() -> None:
    assert round(EMA.decay_from_total_steps(1666), 4) == 0.997


def test_training_package_submodules_remain_importable() -> None:
    mod = importlib.import_module("tac.training.score_weighted_reconstruction_loss")
    assert mod.__name__ == "tac.training.score_weighted_reconstruction_loss"
