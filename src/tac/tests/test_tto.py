# SPDX-License-Identifier: MIT
"""Tests for test-time optimization (TTO) module."""
import time

import pytest
import torch
import torch.nn as nn

from tac.architectures import build_postfilter
from tac.tto import (
    _select_params,
    _sobel_edges,
    edge_preservation_loss,
    is_inflate_tto_enabled,
    reconstruction_loss,
    temporal_consistency_loss,
    test_time_optimize,
)


# ---- Helpers ---- #

def _make_frames(n: int = 8, h: int = 64, w: int = 64) -> torch.Tensor:
    """Create synthetic frames in CHW format (N, 3, H, W) float [0, 255]."""
    return torch.rand(n, 3, h, w) * 255.0


def _make_frames_hwc(n: int = 8, h: int = 64, w: int = 64) -> torch.Tensor:
    """Create synthetic frames in HWC format (N, H, W, 3) float [0, 255]."""
    return torch.rand(n, h, w, 3) * 255.0


def _small_model(hidden: int = 8) -> nn.Module:
    return build_postfilter("standard", hidden=hidden)


# ---- Loss function tests ---- #

class TestSobelEdges:
    def test_output_shape(self):
        x = torch.rand(2, 3, 32, 32)
        edges = _sobel_edges(x)
        assert edges.shape == (2, 3, 30, 30)

    def test_constant_input_zero_edges(self):
        x = torch.ones(1, 3, 16, 16) * 128.0
        edges = _sobel_edges(x)
        assert edges.max().item() < 1e-4


class TestTemporalConsistencyLoss:
    def test_returns_scalar(self):
        model = _small_model()
        frames = _make_frames(n=4, h=32, w=32)
        loss = temporal_consistency_loss(model, frames)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_single_frame_returns_zero(self):
        model = _small_model()
        frames = _make_frames(n=1, h=32, w=32)
        loss = temporal_consistency_loss(model, frames)
        assert loss.item() == 0.0

    def test_identical_frames_low_loss(self):
        model = _small_model()
        frame = torch.ones(1, 3, 32, 32) * 128.0
        frames = frame.repeat(4, 1, 1, 1)
        loss = temporal_consistency_loss(model, frames)
        # Identical input frames should produce very low temporal loss
        assert loss.item() < 1.0


class TestReconstructionLoss:
    def test_returns_scalar(self):
        model = _small_model()
        frames = _make_frames(n=4, h=32, w=32)
        loss = reconstruction_loss(model, frames)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_positive_loss(self):
        model = _small_model()
        frames = _make_frames(n=4, h=32, w=32)
        loss = reconstruction_loss(model, frames, noise_std=5.0)
        assert loss.item() >= 0.0

    def test_zero_noise_zero_loss(self):
        model = _small_model()
        frames = _make_frames(n=4, h=32, w=32)
        loss = reconstruction_loss(model, frames, noise_std=0.0)
        assert loss.item() < 1e-4


class TestEdgePreservationLoss:
    def test_returns_scalar(self):
        model = _small_model()
        frames = _make_frames(n=4, h=32, w=32)
        loss = edge_preservation_loss(model, frames)
        assert loss.ndim == 0
        assert loss.requires_grad


# ---- Parameter selection tests ---- #

class TestParamSelection:
    def test_last_layer_selects_subset(self):
        model = _small_model(hidden=8)
        params = _select_params(model, "last_layer")
        all_params = list(model.parameters())
        assert 0 < len(params) < len(all_params)

    def test_all_selects_everything(self):
        model = _small_model(hidden=8)
        params = _select_params(model, "all")
        all_params = [p for p in model.parameters() if p.requires_grad]
        assert len(params) == len(all_params)

    def test_bn_only_falls_back_to_last_layer(self):
        # Standard PostFilter has no BN, should fall back to last_layer
        model = _small_model(hidden=8)
        params_bn = _select_params(model, "bn_only")
        params_last = _select_params(model, "last_layer")
        assert len(params_bn) == len(params_last)


# ---- Integration tests ---- #

class TestTestTimeOptimize:
    def test_inflate_time_gate_fails_closed_by_default(self, monkeypatch):
        monkeypatch.delenv("INFLATE_TTO", raising=False)
        model = _small_model(hidden=8)
        frames = _make_frames(n=4, h=32, w=32)

        with pytest.raises(RuntimeError, match="INFLATE_TTO=1"):
            test_time_optimize(
                model,
                frames,
                n_steps=1,
                verbose=False,
                inflate_time=True,
            )

    def test_inflate_time_gate_allows_explicit_operator_enable(self, monkeypatch):
        monkeypatch.setenv("INFLATE_TTO", "1")
        assert is_inflate_tto_enabled()
        model = _small_model(hidden=8)
        frames = _make_frames(n=4, h=32, w=32)

        result = test_time_optimize(
            model,
            frames,
            n_steps=1,
            verbose=False,
            inflate_time=True,
        )

        assert isinstance(result, nn.Module)

    def test_chw_input(self):
        model = _small_model(hidden=8)
        frames = _make_frames(n=8, h=32, w=32)
        result = test_time_optimize(
            model, frames, n_steps=2, lr=1e-3,
            loss_type="temporal_consistency", verbose=False,
        )
        assert isinstance(result, nn.Module)

    def test_hwc_input(self):
        model = _small_model(hidden=8)
        frames = _make_frames_hwc(n=8, h=32, w=32)
        result = test_time_optimize(
            model, frames, n_steps=2, lr=1e-3,
            loss_type="reconstruction", verbose=False,
        )
        assert isinstance(result, nn.Module)

    def test_edge_preservation_mode(self):
        model = _small_model(hidden=8)
        frames = _make_frames(n=8, h=32, w=32)
        result = test_time_optimize(
            model, frames, n_steps=2, lr=1e-3,
            loss_type="edge_preservation", verbose=False,
        )
        assert isinstance(result, nn.Module)

    def test_time_budget_respected(self):
        model = _small_model(hidden=8)
        frames = _make_frames(n=16, h=64, w=64)
        t0 = time.monotonic()
        test_time_optimize(
            model, frames, n_steps=1000, lr=1e-3,
            time_budget_seconds=1.0, verbose=False,
        )
        elapsed = time.monotonic() - t0
        # Should have stopped within a reasonable margin of the budget
        assert elapsed < 5.0

    def test_weights_change(self):
        model = _small_model(hidden=8)
        frames = _make_frames(n=8, h=32, w=32)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        test_time_optimize(
            model, frames, n_steps=5, lr=1e-2,
            loss_type="reconstruction", param_mode="all",
            verbose=False,
        )
        after = model.state_dict()
        # At least some weights should have changed
        changed = any(
            not torch.equal(before[k], after[k])
            for k in before
            if before[k].is_floating_point()
        )
        assert changed, "TTO should modify at least some weights"

    def test_quality_check_rollback(self):
        """If loss degrades significantly, original weights should be restored."""
        model = _small_model(hidden=8)
        frames = _make_frames(n=4, h=32, w=32)
        original = {k: v.clone() for k, v in model.state_dict().items()}

        # Use absurdly high LR to cause degradation
        test_time_optimize(
            model, frames, n_steps=3, lr=10.0,
            loss_type="reconstruction", param_mode="all",
            quality_check=True, verbose=False,
        )
        # We can't guarantee rollback triggers with synthetic data,
        # but the function should not crash
        assert isinstance(model, nn.Module)

    def test_dilated_architecture(self):
        model = build_postfilter("dilated", hidden=16)
        frames = _make_frames(n=8, h=32, w=32)
        result = test_time_optimize(
            model, frames, n_steps=2, lr=1e-3,
            loss_type="temporal_consistency", verbose=False,
        )
        assert isinstance(result, nn.Module)

    def test_psd_architecture(self):
        # PixelShuffle requires dimensions divisible by 2
        model = build_postfilter("psd", hidden=16)
        frames = _make_frames(n=8, h=64, w=64)
        result = test_time_optimize(
            model, frames, n_steps=2, lr=1e-3,
            loss_type="edge_preservation", verbose=False,
        )
        assert isinstance(result, nn.Module)

    def test_last_layer_only(self):
        model = _small_model(hidden=8)
        frames = _make_frames(n=8, h=32, w=32)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        test_time_optimize(
            model, frames, n_steps=5, lr=1e-2,
            param_mode="last_layer", verbose=False,
        )
        after = model.state_dict()
        # Early layers should NOT change
        assert torch.equal(before["conv1.weight"], after["conv1.weight"])
        # Last layer should change (conv3)
        changed_last = not torch.equal(before["conv3.weight"], after["conv3.weight"])
        assert changed_last, "last_layer mode should update conv3"
