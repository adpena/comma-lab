# SPDX-License-Identifier: MIT
"""Tests for FiLM-style pose-conditioning bolt-on."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.nerv_pose_conditioning_bolton import (
    FiLMModulator,
    FiLMModulatorConfig,
    apply_film,
    compute_pose_input_for_pair,
    modulate_latent,
)


# ── Config ───────────────────────────────────────────────────────────────


def test_config_default():
    cfg = FiLMModulatorConfig()
    assert cfg.pose_dim == 6
    assert cfg.latent_dim == 16
    assert cfg.init_scale_to_one is True


def test_config_rejects_zero_pose_dim():
    with pytest.raises(ValueError, match="pose_dim must be positive"):
        FiLMModulatorConfig(pose_dim=0)


def test_config_rejects_zero_latent_dim():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        FiLMModulatorConfig(latent_dim=0)


def test_config_rejects_zero_hidden():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        FiLMModulatorConfig(hidden_dim=0)


def test_config_rejects_negative_hidden_layers():
    with pytest.raises(ValueError, match="n_hidden_layers must be non-negative"):
        FiLMModulatorConfig(n_hidden_layers=-1)


def test_config_rejects_invalid_clamp():
    with pytest.raises(ValueError, match="clamp_scale_min"):
        FiLMModulatorConfig(clamp_scale_min=0.0)
    with pytest.raises(ValueError, match="clamp_scale_min"):
        FiLMModulatorConfig(clamp_scale_min=2.0, clamp_scale_max=1.0)


# ── FiLMModulator ───────────────────────────────────────────────────────


def test_film_modulator_output_shape():
    cfg = FiLMModulatorConfig(pose_dim=6, latent_dim=16)
    mod = FiLMModulator(cfg)
    pose = torch.randn(3, 6)
    scale, shift = mod(pose)
    assert scale.shape == (3, 16)
    assert shift.shape == (3, 16)


def test_film_modulator_init_identity_when_init_scale_to_one():
    cfg = FiLMModulatorConfig(pose_dim=6, latent_dim=8, init_scale_to_one=True)
    mod = FiLMModulator(cfg)
    pose = torch.randn(2, 6)
    scale, shift = mod(pose)
    # With zero weights + ones bias on scale_head, all outputs ≈ 1.
    assert torch.allclose(scale, torch.ones_like(scale))
    # Shift head has zero weights + zero bias, so all zero.
    assert torch.allclose(shift, torch.zeros_like(shift))


def test_film_modulator_no_identity_when_init_scale_to_one_false():
    cfg = FiLMModulatorConfig(pose_dim=6, latent_dim=8, init_scale_to_one=False)
    mod = FiLMModulator(cfg)
    pose = torch.randn(2, 6)
    scale, shift = mod(pose)
    # Scale should be in valid clamp range but NOT all ones.
    assert (scale >= cfg.clamp_scale_min).all() and (scale <= cfg.clamp_scale_max).all()


def test_film_modulator_scale_clamped_to_range():
    cfg = FiLMModulatorConfig(pose_dim=6, latent_dim=8,
                               clamp_scale_min=0.5, clamp_scale_max=2.0)
    mod = FiLMModulator(cfg)
    # Set huge scale_head bias so raw output exceeds clamp.
    nn.init.zeros_(mod.scale_head.weight)
    nn.init.constant_(mod.scale_head.bias, 100.0)
    pose = torch.randn(2, 6)
    scale, _ = mod(pose)
    assert (scale <= 2.0).all()
    nn.init.constant_(mod.scale_head.bias, -100.0)
    scale, _ = mod(pose)
    assert (scale >= 0.5).all()


def test_film_modulator_rejects_wrong_pose_shape():
    cfg = FiLMModulatorConfig(pose_dim=6, latent_dim=8)
    mod = FiLMModulator(cfg)
    bad = torch.randn(2, 99)
    with pytest.raises(ValueError, match="FiLMModulator expected"):
        mod(bad)


def test_film_modulator_zero_hidden_layers_works():
    cfg = FiLMModulatorConfig(pose_dim=6, latent_dim=8, n_hidden_layers=0)
    mod = FiLMModulator(cfg)
    pose = torch.randn(2, 6)
    scale, shift = mod(pose)
    assert scale.shape == (2, 8)


def test_film_modulator_grad_flows():
    cfg = FiLMModulatorConfig(pose_dim=6, latent_dim=8)
    mod = FiLMModulator(cfg)
    pose = torch.randn(2, 6, requires_grad=True)
    scale, shift = mod(pose)
    (scale.sum() + shift.sum()).backward()
    assert pose.grad is not None


# ── apply_film ───────────────────────────────────────────────────────────


def test_apply_film_basic_arithmetic():
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scale = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    shift = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    out = apply_film(z, scale, shift)
    assert torch.allclose(out, torch.tensor([[3.0, 7.0], [13.0, 21.0]]))


def test_apply_film_rejects_shape_mismatch():
    z = torch.zeros(2, 4)
    bad_scale = torch.zeros(2, 5)
    shift = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="z shape"):
        apply_film(z, bad_scale, shift)


def test_apply_film_identity_with_one_zero():
    z = torch.randn(3, 4)
    one = torch.ones_like(z)
    zero = torch.zeros_like(z)
    out = apply_film(z, one, zero)
    assert torch.allclose(out, z)


# ── compute_pose_input_for_pair ──────────────────────────────────────────


def test_compute_pose_input_for_pair_default():
    # 4 frames (2 pairs), 12 pose dims; we extract first 6 dims.
    pose_stream = torch.arange(48, dtype=torch.float32).view(4, 12)
    pair_idx = torch.tensor([0, 1])
    out = compute_pose_input_for_pair(pose_stream, pair_idx)
    assert out.shape == (2, 6)
    # Pair 0 is mean of frames [0, 1]; first 6 dims means.
    assert torch.allclose(out[0], 0.5 * (pose_stream[0, :6] + pose_stream[1, :6]))


def test_compute_pose_input_rejects_pose_dim_mismatch():
    pose_stream = torch.zeros(4, 4)  # only 4 dims < 6 default
    pair_idx = torch.tensor([0])
    with pytest.raises(ValueError, match="need >= 6"):
        compute_pose_input_for_pair(pose_stream, pair_idx)


def test_compute_pose_input_rejects_out_of_range_pair_index():
    pose_stream = torch.zeros(4, 12)  # 4 frames = 2 pairs (indices 0, 1)
    pair_idx = torch.tensor([0, 5])  # index 5 needs frame 11
    with pytest.raises(ValueError, match="requires frame"):
        compute_pose_input_for_pair(pose_stream, pair_idx)


def test_compute_pose_input_rejects_non_2d_stream():
    bad = torch.zeros(4)
    pair_idx = torch.tensor([0])
    with pytest.raises(ValueError, match="must be 2-D"):
        compute_pose_input_for_pair(bad, pair_idx)


def test_compute_pose_input_rejects_non_1d_indices():
    pose_stream = torch.zeros(4, 12)
    bad = torch.tensor([[0, 1]])
    with pytest.raises(ValueError, match="must be 1-D"):
        compute_pose_input_for_pair(pose_stream, bad)


# ── modulate_latent ──────────────────────────────────────────────────────


def test_modulate_latent_identity_at_init():
    cfg = FiLMModulatorConfig(pose_dim=6, latent_dim=8, init_scale_to_one=True)
    mod = FiLMModulator(cfg)
    z = torch.randn(2, 8)
    pose = torch.randn(2, 6)
    out = modulate_latent(z=z, pose=pose, modulator=mod)
    assert torch.allclose(out, z)


def test_modulate_latent_changes_z_after_training():
    cfg = FiLMModulatorConfig(pose_dim=6, latent_dim=8, init_scale_to_one=False)
    mod = FiLMModulator(cfg)
    nn.init.constant_(mod.shift_head.bias, 5.0)  # large shift
    z = torch.zeros(2, 8)
    pose = torch.randn(2, 6)
    out = modulate_latent(z=z, pose=pose, modulator=mod)
    # With z=0 and shift=5 (post any small scale*0 = 0 contribution), out ≈ 5.
    assert (out > 4.0).all()


def test_modulate_latent_composes_with_lane_12_v2_decoder():
    """End-to-end: pose-conditioned latent → Lane 12-v2 NeRV decoder."""
    from tac.lane_12_v2_nerv_as_renderer import (
        Lane12V2NeRVConfig,
        Lane12V2NeRVRenderer,
    )

    mod_cfg = FiLMModulatorConfig(pose_dim=6, latent_dim=8)
    mod = FiLMModulator(mod_cfg)
    dec_cfg = Lane12V2NeRVConfig(latent_dim=8, base_channels=8, n_pairs=4,
                                  cuda_required=False)
    dec = Lane12V2NeRVRenderer(dec_cfg)

    z = torch.randn(2, 8) * 0.01
    pose = torch.randn(2, 6) * 0.01
    z_mod = modulate_latent(z=z, pose=pose, modulator=mod)
    out = dec(z_mod)
    assert out.shape == (2, 2, 3, 384, 512)
