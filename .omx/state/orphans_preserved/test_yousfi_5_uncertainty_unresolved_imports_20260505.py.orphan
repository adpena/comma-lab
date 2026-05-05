"""Property tests for Yousfi trick #5 (ScanNet-style spatial uncertainty maps).

Covers:
  - segnet_uncertainty_map shape, range, dtype, and gradient barrier
  - Higher entropy in synthetically-ambiguous image regions vs synthetically
    confident regions
  - segnet_uncertainty_weighted_loss runs and propagates gradients to the
    reconstructed input (and not to the segnet parameters)
  - --use-uncertainty-loss CLI flag is parseable in train_distill.py and
    surfaces as a DistillConfig field

These tests use a tiny mock SegNet (single Conv2d to 5 channels) so they
run in <1s on CPU. They DO NOT depend on the real upstream SegNet weights.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.fridrich import segnet_uncertainty_map
from tac.losses import segnet_uncertainty_weighted_loss


# ── Mock SegNet ──────────────────────────────────────────────────────────


class _MockSegNet(nn.Module):
    """Mock that mirrors the upstream SegNet contract.

    - `preprocess_input((B, T, C, H, W))` slices last frame, resizes to a
      fixed input size, returns (B, 3, H_in, W_in).
    - `forward((B, 3, H_in, W_in))` returns (B, 5, H_in, W_in) logits.

    The conv is sufficient to produce non-trivial spatially-varying logits
    for entropy testing.
    """

    def __init__(self, input_size: tuple[int, int] = (96, 64)) -> None:
        super().__init__()
        self.input_size = input_size  # (H, W)
        # Small conv so logits depend on image content.
        self.conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        # Eval mode: matches frozen-scorer usage in production.
        self.eval()

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W) → take last frame → resize to input_size.
        x = x[:, -1, ...]  # (B, 3, H, W)
        return F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── Test 1: shape / range / dtype ────────────────────────────────────────


def test_segnet_uncertainty_map_shape() -> None:
    """Entropy map has shape (B, 1, H_seg, W_seg) on the SegNet input grid."""
    segnet = _MockSegNet(input_size=(48, 32))
    image = torch.rand(2, 3, 96, 64) * 255.0  # any input resolution

    entropy = segnet_uncertainty_map(segnet, image)

    assert entropy.shape == (2, 1, 48, 32), \
        f"expected (2, 1, 48, 32), got {tuple(entropy.shape)}"


def test_segnet_uncertainty_map_range() -> None:
    """Entropy is in [0, log(5)] (the max entropy over 5 classes)."""
    segnet = _MockSegNet(input_size=(32, 32))
    image = torch.rand(1, 3, 32, 32) * 255.0

    entropy = segnet_uncertainty_map(segnet, image)
    max_entropy = math.log(5.0)

    assert (entropy >= 0).all(), f"entropy went negative: min={entropy.min().item()}"
    # Allow a tiny numerical slack for the +eps log term.
    assert (entropy <= max_entropy + 1e-3).all(), \
        f"entropy exceeded log(5)≈{max_entropy:.4f}: max={entropy.max().item()}"


def test_segnet_uncertainty_map_no_grad_through_segnet() -> None:
    """Gradients do NOT flow through SegNet (it's a stop-grad weight)."""
    segnet = _MockSegNet(input_size=(32, 32))
    image = torch.rand(1, 3, 32, 32, requires_grad=True) * 255.0

    entropy = segnet_uncertainty_map(segnet, image)
    # The map itself has no grad_fn — it's wrapped in torch.no_grad.
    assert entropy.requires_grad is False, \
        "entropy map must be a stop-grad tensor (used as loss weight)"


def test_segnet_uncertainty_map_validates_input() -> None:
    """3D input or wrong channel count → ValueError, not silent garbage."""
    segnet = _MockSegNet()
    with pytest.raises(ValueError, match="must be"):
        segnet_uncertainty_map(segnet, torch.rand(3, 32, 32))  # missing batch
    with pytest.raises(ValueError, match="3 channels"):
        segnet_uncertainty_map(segnet, torch.rand(1, 4, 32, 32))


# ── Test 2: higher entropy in low-confidence regions ─────────────────────


def test_uncertainty_higher_in_low_confidence_regions() -> None:
    """Synthetically-confident pixels have lower entropy than ambiguous ones.

    Build a SegNet whose logits are a linear function of input intensity:
    a saturated bright region pushes one class strongly, a mid-gray region
    leaves all classes near 0 → high entropy.
    """
    # Build a deterministic mock where logits[c] = α_c * mean(input). Then
    # at a saturated pixel, logits diverge → low entropy. At a near-zero
    # pixel, logits ≈ 0 → uniform softmax → log(5) entropy.
    class _DeterministicSegNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_size = (32, 32)
            # Per-class scale: spread the logits widely so saturated pixels
            # have a clear winner.
            self.scale = nn.Parameter(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
                                       requires_grad=False)

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            x = x[:, -1, ...]
            return F.interpolate(x, size=self.input_size,
                                 mode="bilinear", align_corners=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Per-pixel: logits[b, c, h, w] = scale[c] * (luma[b, h, w] / 127.5 - 1)
            B, _, H, W = x.shape
            luma = x.mean(dim=1, keepdim=True) / 127.5 - 1.0  # roughly [-1, 1]
            return luma * self.scale.view(1, 5, 1, 1)

    segnet = _DeterministicSegNet().eval()

    # Two images: one near-mid-gray (logits ≈ 0 → max entropy), one saturated
    # white (logits diverge → low entropy).
    mid_gray = torch.full((1, 3, 32, 32), 127.5)
    saturated = torch.full((1, 3, 32, 32), 255.0)

    h_gray = segnet_uncertainty_map(segnet, mid_gray).mean().item()
    h_sat = segnet_uncertainty_map(segnet, saturated).mean().item()

    assert h_gray > h_sat, \
        f"expected mid-gray entropy ({h_gray:.4f}) > saturated entropy ({h_sat:.4f})"
    # Mid-gray should be near log(5) ≈ 1.609.
    assert abs(h_gray - math.log(5.0)) < 0.05, \
        f"mid-gray entropy {h_gray:.4f} should be near log(5)={math.log(5.0):.4f}"


# ── Test 3: weighted loss runs backward ──────────────────────────────────


def test_uncertainty_weighted_loss_runs_backward() -> None:
    """Loss is finite, scalar, and gradients reach `reconstructed`."""
    segnet = _MockSegNet(input_size=(48, 32))
    target = torch.rand(2, 3, 96, 64) * 255.0
    reconstructed = (target + torch.randn_like(target) * 5.0).requires_grad_(True)

    loss = segnet_uncertainty_weighted_loss(reconstructed, target, segnet)

    assert loss.ndim == 0, f"loss must be scalar; got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss).item(), f"loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"loss should be positive; got {loss.item()}"

    loss.backward()
    assert reconstructed.grad is not None, "no gradient flowed to `reconstructed`"
    assert torch.isfinite(reconstructed.grad).all(), "non-finite gradient on reconstructed"
    # And segnet weights MUST NOT receive gradient (frozen scorer assumption).
    for p in segnet.parameters():
        assert p.grad is None, \
            f"SegNet parameter received a gradient — should be stop-grad, got {p.grad}"


def test_uncertainty_weighted_loss_zero_at_perfect_reconstruction() -> None:
    """Perfect reconstruction → zero L1 loss regardless of weighting."""
    segnet = _MockSegNet(input_size=(32, 32))
    target = torch.rand(1, 3, 64, 32) * 255.0
    loss = segnet_uncertainty_weighted_loss(target.clone(), target, segnet)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_uncertainty_weighted_loss_accepts_hwc() -> None:
    """The loss auto-detects (B, H, W, 3) HWC inputs (matches train_distill)."""
    segnet = _MockSegNet(input_size=(48, 32))
    target_hwc = torch.rand(1, 96, 64, 3) * 255.0
    recon_hwc = (target_hwc + torch.randn_like(target_hwc) * 3.0).requires_grad_(True)

    loss = segnet_uncertainty_weighted_loss(recon_hwc, target_hwc, segnet)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert recon_hwc.grad is not None


# ── Test 4: CLI flag propagates through train_distill.py ─────────────────


def test_cli_flag_propagates_through_train_distill(monkeypatch) -> None:
    """`--use-uncertainty-loss` parses, lands on DistillConfig, and is passed through.

    We import train_distill.parse_args and feed a minimal argv; we then
    construct a DistillConfig with the new fields and verify they round-trip.
    This catches the "added flag but never plumbed it to DistillConfig" class
    of bug that has burned us before.
    """
    train_distill = pytest.importorskip("experiments.train_distill",
                                        reason="train_distill module not importable")

    # Minimal argv exercising every required positional arg of parse_args.
    # We monkeypatch sys.argv and call parse_args directly.
    argv = [
        "train_distill.py",
        "--tto-frames", "/tmp/_doesnotmatter.pt",
        "--checkpoint", "/tmp/_doesnotmatter.pt",
        "--upstream", "/tmp/upstream",
        "--use-uncertainty-loss",
        "--uncertainty-loss-weight", "0.07",
        "--uncertainty-loss-floor", "0.25",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    args = train_distill.parse_args()
    assert args.use_uncertainty_loss is True
    assert args.uncertainty_loss_weight == pytest.approx(0.07)
    assert args.uncertainty_loss_floor == pytest.approx(0.25)

    # And the DistillConfig dataclass must accept these fields without
    # raising — i.e. the field names match between argparse and the dataclass.
    cfg = train_distill.DistillConfig(
        tto_frames_path=Path("/tmp/x.pt"),
        checkpoint_path=Path("/tmp/y.pt"),
        upstream_dir=Path("/tmp/u"),
        use_uncertainty_loss=True,
        uncertainty_loss_weight=0.07,
        uncertainty_loss_floor=0.25,
    )
    assert cfg.use_uncertainty_loss is True
    assert cfg.uncertainty_loss_weight == pytest.approx(0.07)
    assert cfg.uncertainty_loss_floor == pytest.approx(0.25)


def test_pipeline_config_accepts_uncertainty_fields() -> None:
    """PipelineConfig (experiments/pipeline.py) exposes the new fields too."""
    pipeline = pytest.importorskip("experiments.pipeline",
                                   reason="pipeline module not importable")
    from dataclasses import fields
    field_names = {f.name for f in fields(pipeline.PipelineConfig)}
    assert "use_uncertainty_loss" in field_names
    assert "uncertainty_loss_weight" in field_names
    assert "uncertainty_loss_floor" in field_names


def test_profiles_carry_uncertainty_fields() -> None:
    """WILDE / SHIRAZ / DEN profiles include the Yousfi #5 fields."""
    from tac.profiles import DEN, GREEN, SHIRAZ, WILDE
    for name, prof in [("WILDE", WILDE), ("SHIRAZ", SHIRAZ), ("DEN", DEN), ("GREEN", GREEN)]:
        assert "use_uncertainty_loss" in prof, f"{name} missing use_uncertainty_loss"
        assert "uncertainty_loss_weight" in prof, f"{name} missing uncertainty_loss_weight"
        assert "uncertainty_loss_floor" in prof, f"{name} missing uncertainty_loss_floor"
