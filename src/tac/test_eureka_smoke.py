#!/usr/bin/env python
"""Smoke tests for Eureka #5-9 implementations.

Runs without real scorer models -- uses mock scorers with correct interfaces.
Tests:
  - scorer_as_compressor (CPU Eureka #7)
  - coupled_trajectory_optimize (GPU Eureka #5)
  - alternating_projections_optimize (GPU Eureka #7)
  - newton_step_optimize (Newton step obsession)
  - extract_batchnorm_statistics + batchnorm_style_loss (GPU Eureka #6)
  - compute_quantization_directions + apply_quantization_directions (CPU Eureka #9)
  - deblock_frames (CPU Eureka #8)

Usage:
    python -m tac.test_eureka_smoke
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Mock scorer models ----

class MockPoseNet(nn.Module):
    """Minimal PoseNet mock with preprocess_input and BatchNorm."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 6)
        # Set running stats for BN testing
        self.bn.running_mean = torch.randn(16) * 0.1
        self.bn.running_var = torch.ones(16) * 0.5

    def preprocess_input(self, pair_btchw):
        """Mock preprocessing: (B, T=2, C, H, W) -> (B, 6, H', W')."""
        B, T, C, H, W = pair_btchw.shape
        # Resize to small fixed size
        t0 = F.interpolate(pair_btchw[:, 0], size=(8, 8), mode="bilinear", align_corners=False)
        t1 = F.interpolate(pair_btchw[:, 1], size=(8, 8), mode="bilinear", align_corners=False)
        return torch.cat([t0, t1], dim=1)  # (B, 6, 8, 8)

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        h = h.mean(dim=(2, 3))  # global avg pool
        pose = self.fc(h)
        return {"pose": pose}


class MockSegNet(nn.Module):
    """Minimal SegNet mock with preprocess_input."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(3, num_classes, 3, padding=1)

    def preprocess_input(self, pair_btchw):
        """Mock preprocessing: uses last frame only."""
        if pair_btchw.dim() == 5:
            return F.interpolate(pair_btchw[:, -1], size=(12, 16),
                                 mode="bilinear", align_corners=False)
        return F.interpolate(pair_btchw, size=(12, 16),
                             mode="bilinear", align_corners=False)

    def forward(self, x):
        return self.conv(x)  # (B, num_classes, H, W)


def _make_test_data(n_frames=4, H=24, W=32, device="cpu"):
    """Create test frames and masks."""
    masks = torch.randint(0, 5, (n_frames, H, W), device=device)
    frames = torch.rand(n_frames, H, W, 3, device=device) * 255.0
    frames_chw = frames.permute(0, 3, 1, 2)  # (N, 3, H, W)
    expected_pose = torch.randn(n_frames - 1, 6, device=device) * 0.01
    return frames, frames_chw, masks, expected_pose


def test_scorer_as_compressor():
    """CPU Eureka #7: scorer outputs as compressed representation."""
    from tac.constrained_gen import scorer_as_compressor

    posenet = MockPoseNet().eval()
    segnet = MockSegNet().eval()
    frames, _, _, _ = _make_test_data(n_frames=4)

    result = scorer_as_compressor(frames, posenet, segnet, device="cpu", topk=2, batch_size=2)

    assert "posenet_targets" in result
    assert "segnet_masks" in result
    assert "segnet_logits_topk" in result
    assert "segnet_logits_topk_idx" in result

    assert result["posenet_targets"].shape == (3, 6), f"Expected (3, 6), got {result['posenet_targets'].shape}"
    assert result["posenet_targets"].dtype == torch.float16
    assert result["segnet_masks"].dtype == torch.uint8
    assert result["segnet_logits_topk"].dtype == torch.float16
    assert result["segnet_logits_topk_idx"].dtype == torch.uint8

    N = 4
    assert result["segnet_masks"].shape[0] == N
    assert result["segnet_logits_topk"].shape[0] == N
    assert result["segnet_logits_topk"].shape[1] == 2  # topk=2

    print("  PASS: scorer_as_compressor")


def test_coupled_trajectory_optimize():
    """GPU Eureka #5: coupled 4D-Var trajectory optimization."""
    from tac.constrained_gen import coupled_trajectory_optimize

    posenet = MockPoseNet().eval()
    segnet = MockSegNet().eval()
    _, _, masks, expected_pose = _make_test_data(n_frames=4)

    result = coupled_trajectory_optimize(
        masks, expected_pose, posenet, segnet,
        num_steps=5, lr=0.1, device="cpu", log_every=0,
    )

    assert result.shape == (4, 24, 32, 3), f"Expected (4, 24, 32, 3), got {result.shape}"
    assert result.min() >= 0.0 and result.max() <= 255.0
    # Check it's quantized to integer-compatible values
    assert torch.allclose(result, result.round())

    print("  PASS: coupled_trajectory_optimize")


def test_alternating_projections():
    """GPU Eureka #7: Dykstra's alternating projections."""
    from tac.constrained_gen import alternating_projections_optimize

    posenet = MockPoseNet().eval()
    segnet = MockSegNet().eval()
    _, _, masks, expected_pose = _make_test_data(n_frames=4)

    result = alternating_projections_optimize(
        masks, expected_pose, posenet, segnet,
        num_outer_iterations=3, num_inner_steps=2,
        lr=0.1, device="cpu", log_every=0,
    )

    assert result.shape == (4, 24, 32, 3), f"Expected (4, 24, 32, 3), got {result.shape}"
    assert result.min() >= 0.0 and result.max() <= 255.0
    assert torch.allclose(result, result.round())

    print("  PASS: alternating_projections_optimize")


def test_newton_step():
    """Newton/L-BFGS optimization."""
    from tac.constrained_gen import newton_step_optimize

    posenet = MockPoseNet().eval()
    segnet = MockSegNet().eval()
    _, _, masks, expected_pose = _make_test_data(n_frames=4)

    result = newton_step_optimize(
        frames=None, posenet=posenet, segnet=segnet,
        masks=masks, expected_pose=expected_pose,
        num_newton_steps=2, max_iter_per_step=3,
        lr=1.0, history_size=3,
        device="cpu", log_every=0,
    )

    assert result.shape == (4, 24, 32, 3), f"Expected (4, 24, 32, 3), got {result.shape}"
    assert result.min() >= 0.0 and result.max() <= 255.0

    print("  PASS: newton_step_optimize")


def test_batchnorm_statistics():
    """GPU Eureka #6: BatchNorm statistics extraction and style loss."""
    from tac.scorer_exploits import extract_batchnorm_statistics, batchnorm_style_loss

    posenet = MockPoseNet().eval()

    # Test extraction
    stats = extract_batchnorm_statistics(posenet)
    assert len(stats) > 0, "Expected at least one BN layer"
    for s in stats:
        assert "running_mean" in s
        assert "running_var" in s
        assert "name" in s
        assert "num_features" in s
        assert isinstance(s["running_mean"], torch.Tensor)
        assert isinstance(s["running_var"], torch.Tensor)

    print(f"  Extracted BN stats from {len(stats)} layers")

    # Test style loss
    frames = torch.rand(2, 6, 8, 8) * 255.0
    frames = frames.detach().requires_grad_(True)
    loss = batchnorm_style_loss(frames, posenet, stats)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert loss.requires_grad, "Loss should be differentiable"

    # Test backward
    loss.backward()
    assert frames.grad is not None, "Gradient should flow to frames"
    assert frames.grad.shape == frames.shape

    # Test empty stats
    frames2 = torch.rand(2, 6, 8, 8) * 255.0
    frames2 = frames2.detach().requires_grad_(True)
    empty_loss = batchnorm_style_loss(frames2, posenet, [])
    assert empty_loss.item() == 0.0, "Empty stats should give zero loss"

    print("  PASS: extract_batchnorm_statistics + batchnorm_style_loss")


def test_quantization_directions():
    """CPU Eureka #9: Jacobian-directed quantization."""
    from tac.precompute_corrections import (
        compute_quantization_directions,
        apply_quantization_directions,
    )

    posenet = MockPoseNet().eval()
    segnet = MockSegNet().eval()
    frames_chw = torch.rand(4, 3, 24, 32) * 255.0

    # Test compute
    directions = compute_quantization_directions(
        frames_chw, posenet, segnet,
        device="cpu", batch_size=2, verbose=False,
    )

    assert directions.shape == (4, 3, 24, 32), f"Expected (4, 3, 24, 32), got {directions.shape}"
    assert directions.dtype == torch.int8
    assert set(directions.unique().tolist()).issubset({-1, 0, 1}), \
        f"Unexpected values: {directions.unique().tolist()}"

    # Test apply (BCHW)
    frames_float = torch.rand(4, 3, 24, 32) * 254.0 + 0.5  # fractional values
    quantized = apply_quantization_directions(frames_float, directions, hwc_input=False)
    assert quantized.shape == frames_float.shape
    assert quantized.min() >= 0.0 and quantized.max() <= 255.0
    # Every value should be an integer (floor/ceil/round)
    assert torch.allclose(quantized, quantized.round()), "Output should be integer-valued"

    # Test apply (HWC)
    frames_hwc = frames_float.permute(0, 2, 3, 1).contiguous()
    quantized_hwc = apply_quantization_directions(frames_hwc, directions, hwc_input=True)
    assert quantized_hwc.shape == frames_hwc.shape
    assert torch.allclose(quantized_hwc, quantized_hwc.round())

    print("  PASS: compute_quantization_directions + apply_quantization_directions")


def test_deblock_frames():
    """CPU Eureka #8: Non-local means deblocking."""
    try:
        import cv2
    except ImportError:
        print("  SKIP: deblock_frames (cv2 not installed)")
        return

    # Import from the submission path
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent / "submissions" / "robust_current"))
    try:
        from inflate_postfilter import deblock_frames, deblock_tensor
    except ImportError:
        # Try direct import
        print("  SKIP: deblock_frames (import failed)")
        return

    # Test numpy interface
    frames_np = np.random.randint(0, 256, (3, 24, 32, 3), dtype=np.uint8)
    deblocked_np = deblock_frames(frames_np, h=10, template_window=7, search_window=21)
    assert deblocked_np.shape == frames_np.shape
    assert deblocked_np.dtype == np.uint8

    # Test single frame
    single = frames_np[0]
    deblocked_single = deblock_frames(single, h=10)
    assert deblocked_single.shape == single.shape

    # Test tensor interface
    frames_t = torch.rand(3, 3, 24, 32) * 255.0
    deblocked_t = deblock_tensor(frames_t, h=10)
    assert deblocked_t.shape == frames_t.shape

    print("  PASS: deblock_frames + deblock_tensor")


def test_profiles_exist():
    """Check all new profiles are registered."""
    from tac.profiles import PROFILES

    required = [
        "coupled_trajectory_smoke",
        "alternating_projections_smoke",
        "newton_step_smoke",
        "shannon_compressor_smoke",
    ]
    for name in required:
        assert name in PROFILES, f"Profile '{name}' not found in PROFILES"

    print(f"  PASS: all {len(required)} new profiles registered")


def main():
    print("=" * 60)
    print("Eureka #5-9 smoke tests")
    print("=" * 60)

    tests = [
        test_scorer_as_compressor,
        test_coupled_trajectory_optimize,
        test_alternating_projections,
        test_newton_step,
        test_batchnorm_statistics,
        test_quantization_directions,
        test_deblock_frames,
        test_profiles_exist,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test_fn.__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
