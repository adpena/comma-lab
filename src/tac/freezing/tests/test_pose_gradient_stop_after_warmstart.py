# SPDX-License-Identifier: MIT
"""Tests for ``tac.freezing.pose_gradient_stop_after_warmstart``."""

from __future__ import annotations

import pytest
import torch.nn as nn

from tac.freezing.pose_gradient_stop_after_warmstart import (
    GradientStopReport,
    apply_pose_gradient_stop_after_warmstart,
)


def _make_module() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))


def test_below_threshold_is_noop():
    """Calling below the warmstart threshold does not freeze the module."""
    m = _make_module()
    assert any(p.requires_grad for p in m.parameters())
    report = apply_pose_gradient_stop_after_warmstart(
        m, current_epoch=2, warmstart_epochs=10
    )
    assert isinstance(report, GradientStopReport)
    assert report.stopped is False
    assert report.current_epoch == 2
    assert report.warmstart_epochs == 10
    # Module is still trainable.
    assert any(p.requires_grad for p in m.parameters())


def test_at_threshold_freezes():
    """Calling with ``current_epoch == warmstart_epochs`` fires the freeze."""
    m = _make_module()
    report = apply_pose_gradient_stop_after_warmstart(
        m, current_epoch=10, warmstart_epochs=10
    )
    assert report.stopped is True
    assert report.trainable_after == 0
    assert not any(p.requires_grad for p in m.parameters())


def test_above_threshold_freezes():
    """Calling well above threshold also fires the freeze."""
    m = _make_module()
    report = apply_pose_gradient_stop_after_warmstart(
        m, current_epoch=50, warmstart_epochs=10
    )
    assert report.stopped is True
    assert report.trainable_after == 0


def test_threshold_zero_freezes_immediately():
    """``warmstart_epochs=0`` makes the helper a one-shot freeze."""
    m = _make_module()
    report = apply_pose_gradient_stop_after_warmstart(
        m, current_epoch=0, warmstart_epochs=0
    )
    assert report.stopped is True
    assert report.trainable_after == 0


def test_negative_epoch_raises():
    """Negative epoch values are rejected."""
    m = _make_module()
    with pytest.raises(ValueError):
        apply_pose_gradient_stop_after_warmstart(
            m, current_epoch=-1, warmstart_epochs=5
        )


def test_negative_warmstart_raises():
    """Negative warmstart threshold values are rejected."""
    m = _make_module()
    with pytest.raises(ValueError):
        apply_pose_gradient_stop_after_warmstart(
            m, current_epoch=0, warmstart_epochs=-1
        )


def test_loop_pattern_fires_exactly_once_at_threshold():
    """Simulating an epoch loop: the helper fires exactly at the threshold epoch."""
    m = _make_module()
    transitions: list[int] = []
    was_trainable = True
    for epoch in range(20):
        report = apply_pose_gradient_stop_after_warmstart(
            m, current_epoch=epoch, warmstart_epochs=7
        )
        now_trainable = any(p.requires_grad for p in m.parameters())
        if was_trainable and not now_trainable:
            transitions.append(epoch)
        was_trainable = now_trainable
        assert report.stopped == (epoch >= 7)
    assert transitions == [7]


def test_idempotent_after_freeze():
    """After the initial freeze, subsequent calls remain a no-op-by-state."""
    m = _make_module()
    apply_pose_gradient_stop_after_warmstart(
        m, current_epoch=10, warmstart_epochs=5
    )
    assert not any(p.requires_grad for p in m.parameters())
    # Call again at a later epoch: still no trainable params; report.stopped True.
    report = apply_pose_gradient_stop_after_warmstart(
        m, current_epoch=20, warmstart_epochs=5
    )
    assert report.stopped is True
    assert report.trainable_after == 0


def test_report_carries_name():
    """The optional ``name`` field is propagated to the report."""
    m = _make_module()
    report = apply_pose_gradient_stop_after_warmstart(
        m, current_epoch=5, warmstart_epochs=5, name="posenet"
    )
    assert report.name == "posenet"


def test_below_threshold_report_records_state():
    """Below-threshold report still records trainable parameter count."""
    m = _make_module()
    expected_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    report = apply_pose_gradient_stop_after_warmstart(
        m, current_epoch=0, warmstart_epochs=10
    )
    assert report.trainable_after == expected_trainable
    assert report.trainable_after > 0
