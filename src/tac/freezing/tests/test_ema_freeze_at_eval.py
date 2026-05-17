# SPDX-License-Identifier: MIT
"""Tests for ``tac.freezing.ema_freeze_at_eval``."""

from __future__ import annotations

import torch
import torch.nn as nn

from tac.freezing.ema_freeze_at_eval import (
    EMAEvalSnapshot,
    ema_freeze_at_eval_snapshot_restore,
)


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1))


def _capture_state(m: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def test_snapshot_returns_typed_object():
    """Context manager yields :class:`EMAEvalSnapshot`."""
    m = _make_model()
    ema_state = _capture_state(m)
    with ema_freeze_at_eval_snapshot_restore(m, ema_state) as snap:
        assert isinstance(snap, EMAEvalSnapshot)
        assert snap.tensor_count == len(ema_state)


def test_eval_mode_during_context():
    """Model is in eval mode while the EMA weights are applied."""
    m = _make_model()
    m.train()
    ema_state = _capture_state(m)
    with ema_freeze_at_eval_snapshot_restore(m, ema_state):
        assert m.training is False


def test_training_mode_restored_after_context():
    """Training mode is restored on context exit."""
    m = _make_model()
    m.train()
    ema_state = _capture_state(m)
    with ema_freeze_at_eval_snapshot_restore(m, ema_state):
        pass
    assert m.training is True


def test_live_weights_restored_after_context():
    """Live weights are restored on context exit even if EMA weights differ."""
    m = _make_model()
    live = _capture_state(m)
    # Construct EMA weights that differ from live.
    ema_state = {k: v + 0.5 for k, v in live.items()}
    with ema_freeze_at_eval_snapshot_restore(m, ema_state):
        # Inside the context, model's weights == EMA.
        for k, v in m.state_dict().items():
            assert torch.allclose(v, ema_state[k], atol=1e-6)
    # After context, restored to live.
    for k, v in m.state_dict().items():
        assert torch.allclose(v, live[k], atol=1e-6)


def test_exception_inside_context_still_restores():
    """Even if an exception fires inside the context, live weights are restored."""
    m = _make_model()
    live = _capture_state(m)
    ema_state = {k: v + 0.5 for k, v in live.items()}
    try:
        with ema_freeze_at_eval_snapshot_restore(m, ema_state):
            raise RuntimeError("synthetic eval-time error")
    except RuntimeError:
        pass
    # Live weights restored despite exception.
    for k, v in m.state_dict().items():
        assert torch.allclose(v, live[k], atol=1e-6)


def test_training_mode_restored_after_exception():
    """Training mode is restored to ``train()`` even if an exception fires."""
    m = _make_model()
    m.train()
    live = _capture_state(m)
    ema_state = {k: v + 0.1 for k, v in live.items()}
    try:
        with ema_freeze_at_eval_snapshot_restore(m, ema_state):
            raise RuntimeError("synthetic eval-time error")
    except RuntimeError:
        pass
    assert m.training is True


def test_eval_mode_propagates_eval_after_eval_start():
    """If model was in eval mode before context, it remains eval after."""
    m = _make_model()
    m.eval()
    ema = _capture_state(m)
    with ema_freeze_at_eval_snapshot_restore(m, ema):
        assert m.training is False
    assert m.training is False


def test_strict_false_allows_partial_state_dict():
    """``strict=False`` allows partial EMA state dicts (sister-bolt-on use case)."""
    m = _make_model()
    full = _capture_state(m)
    # EMA state covers only one parameter (the rest stay at live).
    one_key = next(iter(full.keys()))
    partial = {one_key: full[one_key]}
    # Should not raise even though the EMA state is partial.
    with ema_freeze_at_eval_snapshot_restore(m, partial) as snap:
        assert snap.tensor_count == 1
