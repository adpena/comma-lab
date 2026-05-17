# SPDX-License-Identifier: MIT
"""Demo wiring test: PoseNet gradient-stop-after-warmstart in ``sane_hnerv``-style trainer.

Demonstrates how :func:`tac.freezing.pose_gradient_stop_after_warmstart.apply_pose_gradient_stop_after_warmstart`
slots into the canonical ``experiments/train_substrate_sane_hnerv.py`` pattern.

The sane_hnerv trainer already freezes its scorers at load time per the
canonical Balle/PR95-paradigm pattern. This demo exercises the SECOND-PHASE
freezing exploit: PoseNet receives gradient via the score-aware loss for the
first N warmstart epochs (so the renderer specializes against the live pose
response surface), then the gradient-stop fires + PoseNet is frozen for the
remainder of training (so the renderer fine-tunes against the now-fixed pose
target). This is the canonical Quantizr 5-stage QAT pattern transition between
joint and final stages, applied to a NeRV-family substrate.

The demo is hermetic: synthetic stand-ins for PoseNet + a tiny renderer; the
loop simulates ``args.epochs`` iterations + ``args.pose_warmstart_epochs``
threshold; we verify (a) PoseNet has gradient during warmstart, (b) PoseNet
gradient stops at the threshold, (c) the renderer continues training, (d) the
typed report is suitable for serialization into ``modal_metadata.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import torch
import torch.nn as nn

from tac.freezing.pose_gradient_stop_after_warmstart import (
    GradientStopReport,
    apply_pose_gradient_stop_after_warmstart,
)


def _make_synthetic_posenet() -> nn.Module:
    """Stand-in PoseNet: 12-channel YUV6 input + MLP head."""
    return nn.Sequential(
        nn.Conv2d(12, 8, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 12),
    )


def _make_synthetic_renderer() -> nn.Module:
    """Stand-in renderer: small linear → 3-channel output."""
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )


def test_warmstart_then_freeze_transition_epoch():
    """The gradient-stop fires at exactly epoch ``warmstart_epochs``."""
    posenet = _make_synthetic_posenet()
    warmstart = 3
    transitions: list[int] = []
    was_trainable = True
    for epoch in range(8):
        report = apply_pose_gradient_stop_after_warmstart(
            posenet, current_epoch=epoch, warmstart_epochs=warmstart
        )
        now_trainable = any(p.requires_grad for p in posenet.parameters())
        if was_trainable and not now_trainable:
            transitions.append(epoch)
        was_trainable = now_trainable
        assert isinstance(report, GradientStopReport)
        assert report.stopped == (epoch >= warmstart)
    assert transitions == [warmstart]


def test_warmstart_phase_posenet_has_gradient():
    """During warmstart, PoseNet gradient flows through to its parameters."""
    posenet = _make_synthetic_posenet()
    # Epoch 0, below warmstart threshold of 5: posenet still trainable.
    apply_pose_gradient_stop_after_warmstart(
        posenet, current_epoch=0, warmstart_epochs=5
    )
    x = torch.randn(2, 12, 8, 8)
    y = posenet(x).sum()
    y.backward()
    for p in posenet.parameters():
        assert p.grad is not None


def test_post_warmstart_posenet_has_no_gradient():
    """After threshold, gradients flow through PoseNet but not into it."""
    posenet = _make_synthetic_posenet()
    apply_pose_gradient_stop_after_warmstart(
        posenet, current_epoch=5, warmstart_epochs=5
    )
    x = torch.randn(2, 12, 8, 8, requires_grad=True)
    y = posenet(x).sum()
    y.backward()
    assert x.grad is not None
    for p in posenet.parameters():
        assert p.grad is None


def test_renderer_continues_training_after_pose_freeze():
    """Renderer's gradient is unaffected by the pose-gradient-stop."""
    posenet = _make_synthetic_posenet()
    renderer = _make_synthetic_renderer()
    apply_pose_gradient_stop_after_warmstart(
        posenet, current_epoch=10, warmstart_epochs=5
    )
    # Simulate a compound loss: renderer output → posenet (frozen) → loss.
    x = torch.randn(2, 8)
    rendered = renderer(x)
    # Synthetic projection to PoseNet's 12-channel input shape.
    rendered_image = rendered.unsqueeze(-1).unsqueeze(-1).expand(2, 3, 8, 8)
    # Pad channels to 12 (synthetic; just to satisfy the shape).
    rendered_image_12ch = torch.cat([rendered_image, rendered_image.repeat(1, 3, 1, 1)], dim=1)[
        :, :12, :, :
    ]
    pose_out = posenet(rendered_image_12ch).sum()
    pose_out.backward()
    # Renderer params receive gradient (the pose forward is still consumed even
    # when its own params are frozen — the gradient flows through to renderer).
    for p in renderer.parameters():
        assert p.grad is not None
    # PoseNet params receive no gradient.
    for p in posenet.parameters():
        assert p.grad is None


def test_provenance_dict_serializable_for_modal_metadata():
    """``GradientStopReport`` round-trips through JSON serialization."""
    posenet = _make_synthetic_posenet()
    report = apply_pose_gradient_stop_after_warmstart(
        posenet, current_epoch=5, warmstart_epochs=5, name="posenet"
    )
    payload = json.dumps(asdict(report))
    assert "posenet" in payload
    assert '"stopped": true' in payload
    # Verify round-trip.
    restored = json.loads(payload)
    assert restored["name"] == "posenet"
    assert restored["stopped"] is True
    assert restored["current_epoch"] == 5
    assert restored["warmstart_epochs"] == 5


def test_simulated_training_loop_invariant_holds():
    """Full simulated loop: report fires once at threshold; idempotent thereafter."""
    posenet = _make_synthetic_posenet()
    warmstart = 4
    epochs = 12
    stopped_count = 0
    fired_epoch: int | None = None
    was_trainable = True
    for epoch in range(epochs):
        report = apply_pose_gradient_stop_after_warmstart(
            posenet, current_epoch=epoch, warmstart_epochs=warmstart
        )
        if report.stopped:
            stopped_count += 1
        now_trainable = any(p.requires_grad for p in posenet.parameters())
        if was_trainable and not now_trainable and fired_epoch is None:
            fired_epoch = epoch
        was_trainable = now_trainable
    # Every epoch >= warmstart reported stopped=True.
    assert stopped_count == epochs - warmstart
    # State transitioned exactly once, at warmstart.
    assert fired_epoch == warmstart


def test_threshold_above_epochs_is_pure_noop():
    """Setting threshold > epochs is a permanent no-op (PoseNet stays trainable)."""
    posenet = _make_synthetic_posenet()
    for epoch in range(10):
        report = apply_pose_gradient_stop_after_warmstart(
            posenet, current_epoch=epoch, warmstart_epochs=1000
        )
        assert report.stopped is False
    assert any(p.requires_grad for p in posenet.parameters())
