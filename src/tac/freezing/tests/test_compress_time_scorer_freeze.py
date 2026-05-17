# SPDX-License-Identifier: MIT
"""Tests for ``tac.freezing.compress_time_scorer_freeze``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.freezing.compress_time_scorer_freeze import (
    FreezeReport,
    ScorerNotFrozenError,
    ensure_compress_time_scorer_freeze,
    freeze_module_parameters,
)


def _make_module(in_features: int = 8, out_features: int = 4) -> nn.Module:
    """Return a small Linear+ReLU+BN module used across tests."""
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.BatchNorm1d(out_features),
    )


def test_freeze_module_parameters_sets_requires_grad_false():
    """All parameters become non-trainable after a freeze call."""
    module = _make_module()
    assert any(p.requires_grad for p in module.parameters())
    report = freeze_module_parameters(module, name="m")
    assert not any(p.requires_grad for p in module.parameters())
    assert report.trainable_after == 0


def test_freeze_module_parameters_sets_eval_mode():
    """Module is placed in eval mode after the freeze call."""
    module = _make_module()
    module.train()
    assert module.training is True
    freeze_module_parameters(module, name="m")
    assert module.training is False


def test_freeze_module_parameters_returns_typed_report():
    """The return value is a :class:`FreezeReport` with the expected fields."""
    module = _make_module()
    module.train()
    report = freeze_module_parameters(module, name="scorer-x")
    assert isinstance(report, FreezeReport)
    assert report.name == "scorer-x"
    assert report.parameter_count > 0
    assert report.trainable_before > 0
    assert report.trainable_after == 0
    assert report.training_before is True
    assert report.training_after is False


def test_freeze_module_parameters_is_idempotent():
    """A second call on an already-frozen module is a safe no-op."""
    module = _make_module()
    freeze_module_parameters(module, name="m")
    # No exception; report still describes the now-frozen state.
    report2 = freeze_module_parameters(module, name="m")
    assert report2.trainable_before == 0
    assert report2.trainable_after == 0
    assert report2.training_before is False
    assert report2.training_after is False


def test_freeze_module_parameters_counts_parameters_correctly():
    """``parameter_count`` matches ``sum(p.numel() for p in module.parameters())``."""
    module = _make_module(in_features=16, out_features=8)
    expected = sum(p.numel() for p in module.parameters())
    report = freeze_module_parameters(module, name="m")
    assert report.parameter_count == expected


def test_ensure_compress_time_scorer_freeze_accepts_already_frozen():
    """If both scorers are already frozen, the call succeeds + returns reports."""
    seg = _make_module()
    pose = _make_module()
    freeze_module_parameters(seg, name="seg")
    freeze_module_parameters(pose, name="pose")
    reports = ensure_compress_time_scorer_freeze(seg, pose, names=("segnet", "posenet"))
    assert len(reports) == 2
    assert all(isinstance(r, FreezeReport) for r in reports)
    assert reports[0].name == "segnet"
    assert reports[1].name == "posenet"


def test_ensure_compress_time_scorer_freeze_rejects_trainable():
    """Trainable scorer raises :class:`ScorerNotFrozenError`."""
    seg = _make_module()  # Still trainable.
    freeze_module_parameters(_make_module(), name="dummy")
    with pytest.raises(ScorerNotFrozenError):
        ensure_compress_time_scorer_freeze(seg, names=("segnet",))


def test_ensure_compress_time_scorer_freeze_rejects_train_mode():
    """Frozen-params-but-training-mode still raises (CLAUDE.md eval-mode contract)."""
    seg = _make_module()
    for p in seg.parameters():
        p.requires_grad_(False)
    seg.train()  # Forgot to switch eval.
    with pytest.raises(ScorerNotFrozenError):
        ensure_compress_time_scorer_freeze(seg, names=("segnet",))


def test_ensure_compress_time_scorer_freeze_names_length_mismatch():
    """Length mismatch on names raises ``ValueError``."""
    seg = _make_module()
    pose = _make_module()
    freeze_module_parameters(seg, name="seg")
    freeze_module_parameters(pose, name="pose")
    with pytest.raises(ValueError, match="names length"):
        ensure_compress_time_scorer_freeze(seg, pose, names=("only_one",))


def test_ensure_compress_time_scorer_freeze_default_names():
    """Default name set ``scorer_0``, ``scorer_1``, ... when ``names=None``."""
    a = _make_module()
    b = _make_module()
    freeze_module_parameters(a, name="a")
    freeze_module_parameters(b, name="b")
    reports = ensure_compress_time_scorer_freeze(a, b)
    assert reports[0].name == "scorer_0"
    assert reports[1].name == "scorer_1"


def test_frozen_module_forward_pass_still_works():
    """Frozen module's forward pass produces correct output."""
    module = nn.Linear(8, 4)
    freeze_module_parameters(module, name="linear")
    x = torch.randn(3, 8)
    y = module(x)
    assert y.shape == (3, 4)
    # Forward pass requires_grad=False on output because all params are frozen.
    assert y.requires_grad is False


def test_frozen_module_gradient_does_not_flow_to_parameters():
    """Backprop through a frozen module's params yields no gradient."""
    module = nn.Linear(8, 4)
    freeze_module_parameters(module, name="linear")
    x = torch.randn(3, 8, requires_grad=True)
    y = module(x)
    loss = y.sum()
    loss.backward()
    for p in module.parameters():
        assert p.grad is None
    # x's gradient DOES flow (the module is frozen but the input is not).
    assert x.grad is not None
    assert x.grad.shape == x.shape
