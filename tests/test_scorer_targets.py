"""Tests for PoseNet target extraction and supervised TTO.

Tests the round-trip: extract -> save -> load -> verify, and the
supervised TTO loss computation with mock models.
"""
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from tac.scorer_targets import load_posenet_targets, save_posenet_targets


class MockPoseNet(nn.Module):
    """Minimal mock PoseNet that returns deterministic outputs."""

    def preprocess_input(self, x):
        # x: (B, T, C, H, W) -> just downsample and concat
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        x_small = nn.functional.interpolate(x_flat, size=(16, 16), mode="bilinear")
        return x_small.reshape(B, T * C, 16, 16)

    def forward(self, x):
        # Return deterministic pose outputs based on input mean
        B = x.shape[0]
        feat = x.mean(dim=(1, 2, 3))  # (B,)
        pose = feat.unsqueeze(1).expand(B, 12) * 0.001
        return {"pose": pose}


class MockPostFilter(nn.Module):
    """Minimal mock postfilter."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1, bias=True)
        nn.init.eye_(self.conv.weight.view(3, 3))
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return (x + self.conv(x) * 0.01).clamp(0, 255)


def test_save_load_roundtrip():
    """Test that save -> load preserves targets within float16 precision."""
    n_pairs = 600
    targets = torch.randn(n_pairs, 6)
    targets_dict = {
        "targets": targets,
        "n_pairs": n_pairs,
        "n_frames": 1200,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "posenet_targets.bin"
        size = save_posenet_targets(targets_dict, path)

        assert path.exists()
        assert size > 0
        # 600 pairs * 6 floats * 2 bytes = 7200 raw; compressed should be smaller
        assert size < 15000, f"File too large: {size} bytes"

        loaded = load_posenet_targets(path)
        assert loaded is not None
        assert loaded["n_pairs"] == n_pairs
        assert loaded["n_frames"] == 1200
        assert loaded["targets"].shape == (n_pairs, 6)

        # float16 round-trip: check within tolerance
        torch.testing.assert_close(
            loaded["targets"], targets,
            atol=0.01, rtol=0.01,
        )


def test_load_missing_file():
    """Test graceful fallback when file doesn't exist."""
    result = load_posenet_targets("/nonexistent/path/targets.bin")
    assert result is None


def test_save_load_small():
    """Test with minimal data."""
    targets_dict = {
        "targets": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        "n_pairs": 1,
        "n_frames": 2,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "targets.bin"
        save_posenet_targets(targets_dict, path)
        loaded = load_posenet_targets(path)
        assert loaded is not None
        assert loaded["n_pairs"] == 1
        assert loaded["targets"].shape == (1, 6)


def test_posenet_target_loss():
    """Test the supervised TTO loss computation with mock models."""
    from tac.tto import posenet_target_loss

    model = MockPostFilter()
    posenet = MockPoseNet()

    # Create fake frames: 4 frames -> 2 pairs
    frames = torch.randn(4, 3, 32, 32) * 50 + 128
    frames = frames.clamp(0, 255)

    # Create fake targets
    targets = torch.randn(2, 6)

    loss = posenet_target_loss(model, frames, posenet, targets, pair_start=0)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.requires_grad, "Loss should be differentiable"
    assert loss.item() >= 0, "MSE loss should be non-negative"

    # Verify backward works
    loss.backward()
    grad_found = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            grad_found = True
            break
    assert grad_found, "Gradients should flow through the model"


def test_supervised_tto_runs():
    """Test that supervised_tto runs without error and returns a model."""
    from tac.tto import supervised_tto

    model = MockPostFilter()
    posenet = MockPoseNet()

    frames = torch.randn(8, 3, 32, 32) * 50 + 128
    frames = frames.clamp(0, 255)
    targets = torch.randn(4, 6)  # 4 pairs

    result = supervised_tto(
        model, frames, posenet, targets,
        n_steps=3, lr=1e-3,
        param_mode="all", grad_clip=1.0,
        time_budget_seconds=30.0,
        batch_size=4, verbose=False,
    )

    assert isinstance(result, nn.Module)
    # Model should still produce valid output
    with torch.no_grad():
        out = result(frames[:1])
    assert out.shape == (1, 3, 32, 32)


def test_file_size_realistic():
    """Verify file size stays within budget for realistic data."""
    # 600 pairs of actual PoseNet-scale outputs
    targets = torch.randn(600, 6) * 0.1  # typical PoseNet output scale
    targets_dict = {
        "targets": targets,
        "n_pairs": 600,
        "n_frames": 1200,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "targets.bin"
        size = save_posenet_targets(targets_dict, path)
        # Must be < 10KB to be negligible rate impact
        assert size < 10000, f"File size {size} exceeds 10KB budget"
        print(f"Realistic file size: {size} bytes")
