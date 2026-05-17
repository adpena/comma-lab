# SPDX-License-Identifier: MIT
"""Demo wiring test: SegNet-freeze integration in ``balle_renderer``-style trainer.

Demonstrates how :func:`tac.freezing.compress_time_scorer_freeze.ensure_compress_time_scorer_freeze`
slots into the canonical ``experiments/train_substrate_balle_renderer.py`` pattern.

The Balle renderer trainer ALREADY freezes its scorers inline at line 852:

    for p in list(posenet.parameters()) + list(segnet.parameters()):
        p.requires_grad_(False)
    posenet.eval()
    segnet.eval()

This test exercises the canonical helper as a drop-in replacement that
additionally returns typed :class:`FreezeReport` objects (for provenance JSON)
AND raises :class:`ScorerNotFrozenError` if any scorer parameter is
inadvertently re-enabled mid-training (defense-in-depth per CLAUDE.md
"Strict scorer rule").

The demo is hermetic: we synthesize tiny SegNet + PoseNet stand-in modules
that match the same nn.Module interface contract. Real-scorer integration
is exercised by the trainer's own integration tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tac.freezing.compress_time_scorer_freeze import (
    FreezeReport,
    ScorerNotFrozenError,
    ensure_compress_time_scorer_freeze,
)


def _make_synthetic_segnet() -> nn.Module:
    """Stand-in SegNet: U-Net-shape conv stack at toy scale."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 5, kernel_size=1),  # 5-class SegNet output convention
    )


def _make_synthetic_posenet() -> nn.Module:
    """Stand-in PoseNet: 12-channel YUV6 input + MLP head."""
    return nn.Sequential(
        nn.Conv2d(12, 8, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 12),  # 12-d pose output (first 6 used by scorer)
    )


def test_canonical_helper_replaces_inline_freeze_pattern():
    """The canonical helper produces the same end state as the inline pattern.

    Mirrors :file:`experiments/train_substrate_balle_renderer.py:851-855`.
    """
    segnet = _make_synthetic_segnet()
    posenet = _make_synthetic_posenet()
    # Inline pattern (legacy):
    for p in list(posenet.parameters()) + list(segnet.parameters()):
        p.requires_grad_(False)
    posenet.eval()
    segnet.eval()
    # Now the canonical helper as a defense-in-depth gate.
    reports = ensure_compress_time_scorer_freeze(
        segnet, posenet, names=("segnet", "posenet")
    )
    assert len(reports) == 2
    assert all(isinstance(r, FreezeReport) for r in reports)
    assert all(r.trainable_after == 0 for r in reports)


def test_canonical_helper_catches_inadvertent_unfreeze():
    """If a trainer bug reenables scorer grad, the helper fails closed."""
    segnet = _make_synthetic_segnet()
    posenet = _make_synthetic_posenet()
    for p in list(posenet.parameters()) + list(segnet.parameters()):
        p.requires_grad_(False)
    posenet.eval()
    segnet.eval()
    # Synthetic bug: a parameter group accidentally re-enabled.
    next(segnet.parameters()).requires_grad_(True)
    try:
        ensure_compress_time_scorer_freeze(
            segnet, posenet, names=("segnet", "posenet")
        )
    except ScorerNotFrozenError as exc:
        assert "segnet" in str(exc)
    else:
        raise AssertionError("expected ScorerNotFrozenError")


def test_archival_provenance_dict_serializable():
    """The report's typed fields can be serialized as part of provenance JSON.

    Mirrors what the Balle renderer trainer does at archive-build time
    (writes per-stage event dict into ``modal_metadata.json``).
    """
    segnet = _make_synthetic_segnet()
    posenet = _make_synthetic_posenet()
    for p in list(posenet.parameters()) + list(segnet.parameters()):
        p.requires_grad_(False)
    posenet.eval()
    segnet.eval()
    reports = ensure_compress_time_scorer_freeze(
        segnet, posenet, names=("segnet", "posenet")
    )
    import json
    from dataclasses import asdict

    serializable = {r.name: asdict(r) for r in reports}
    payload = json.dumps(serializable)
    assert "segnet" in payload
    assert "posenet" in payload


def test_forward_pass_post_freeze_yields_no_param_gradient():
    """Demo: after freeze, scorer forward in the training loop has no param grad.

    This is the contract every PR95-paradigm substrate trainer relies on:
    the scorer is part of the forward graph (gradients flow THROUGH it to the
    renderer's parameters via the score-aware loss) but NEVER into the scorer's
    own weights.
    """
    segnet = _make_synthetic_segnet()
    for p in segnet.parameters():
        p.requires_grad_(False)
    segnet.eval()
    # A trainable input simulating the renderer's output.
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    y = segnet(x)
    loss = y.sum()
    loss.backward()
    # Gradient flows to x (the trainable renderer output).
    assert x.grad is not None
    # No gradient on scorer parameters.
    for p in segnet.parameters():
        assert p.grad is None
