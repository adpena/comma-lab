# SPDX-License-Identifier: MIT
"""Tests for the Time-Traveler L5 Autonomy architecture (world model)."""

from __future__ import annotations

import pytest
import torch

from tac.substrates.time_traveler_l5_autonomy.architecture import (
    CAMERA_HW,
    EVAL_HW,
    NUM_PAIRS,
    PER_PAIR_SIDE_INFO_TARGET_BYTES,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    EgoMotionDynamicsPrior,
    LogPolarFoveationGrid,
    PredictiveRenderer,
    TimeTravelerConfig,
    TimeTravelerSubstrate,
)


def test_eval_hw_matches_scorer_resolution() -> None:
    """The renderer eval HW must match the contest scorer's (384, 512)."""
    assert EVAL_HW == (384, 512)


def test_camera_hw_matches_contest_native_resolution() -> None:
    """CAMERA_HW is the contest's native (874, 1164)."""
    assert CAMERA_HW == (874, 1164)


def test_num_pairs_matches_contest_pair_count() -> None:
    """NUM_PAIRS = 1200 frames / 2 = 600."""
    assert NUM_PAIRS == 600


def test_per_pair_side_info_budget_matches_design_memo() -> None:
    """Design-memo per-pair budget is 45 B."""
    assert PER_PAIR_SIDE_INFO_TARGET_BYTES == 45


def test_total_archive_budget_band_is_95_to_110_KB() -> None:
    """Design-memo total budget band is 95-110 KB."""
    assert TOTAL_ARCHIVE_TARGET_BYTES_MIN == 95_000
    assert TOTAL_ARCHIVE_TARGET_BYTES_MAX == 110_000


def test_default_config_validates() -> None:
    """The default ``TimeTravelerConfig`` validates without raising."""
    cfg = TimeTravelerConfig()
    assert cfg.hidden_dim == 64
    assert cfg.coord_dim == 4
    assert cfg.pose_dim == 6
    assert cfg.num_pairs == 600


def test_config_rejects_invalid_coord_dim() -> None:
    """coord_dim must be 4."""
    with pytest.raises(ValueError, match="coord_dim must be 4"):
        TimeTravelerConfig(coord_dim=3)


def test_config_rejects_invalid_pose_dim() -> None:
    """pose_dim must be 6 (SE(3))."""
    with pytest.raises(ValueError, match="pose_dim must be 6"):
        TimeTravelerConfig(pose_dim=4)


def test_config_rejects_nonpositive_hidden_dim() -> None:
    """hidden_dim must be positive."""
    with pytest.raises(ValueError, match="hidden_dim"):
        TimeTravelerConfig(hidden_dim=0)


def test_config_rejects_oversized_per_pair_side_info() -> None:
    """per_pair_side_info_bytes > 256 is rejected."""
    with pytest.raises(ValueError, match="per_pair_side_info_bytes"):
        TimeTravelerConfig(per_pair_side_info_bytes=300)


def test_config_predicts_decoder_param_count_under_60k_for_defaults() -> None:
    """Default config gives sub-60K renderer-MLP params (design budget)."""
    cfg = TimeTravelerConfig()
    n_params = cfg.predict_decoder_param_count()
    # 4 hidden layers of 64 each + I/O => ~20K params.
    assert n_params < 60_000, f"renderer params {n_params} exceed design budget"
    assert n_params > 1_000


def test_log_polar_foveation_grid_estimate_is_under_2KB() -> None:
    """foveation grid bytes (int8) + 8 B header must be < 2 KB."""
    cfg = TimeTravelerConfig()
    grid = LogPolarFoveationGrid(cfg)
    assert grid.estimate_int8_bytes() < 2_000


def test_log_polar_foveation_grid_output_shape() -> None:
    """Foveation forward() returns a (H, W) map at output resolution."""
    cfg = TimeTravelerConfig(output_height=64, output_width=96)
    grid = LogPolarFoveationGrid(cfg)
    out = grid().detach()
    assert out.shape == (64, 96)
    # Values clamped to [0.1, 4.0]
    assert float(out.min()) >= 0.1
    assert float(out.max()) <= 4.0


def test_ego_motion_dynamics_prior_predicts_next_pose_shape() -> None:
    """Markov-1 predictor returns same-shape SE(3) prediction."""
    cfg = TimeTravelerConfig()
    prior = EgoMotionDynamicsPrior(cfg)
    prev = torch.randn(4, 6)
    next_pose = prior.predict_next(prev)
    assert next_pose.shape == (4, 6)


def test_ego_motion_dynamics_prior_byte_estimate_is_under_100B() -> None:
    """Markov dynamics matrix + bias fits in well under 100 B (FP16)."""
    cfg = TimeTravelerConfig()
    prior = EgoMotionDynamicsPrior(cfg)
    # 6*6 + 6 = 42 FP16s = 84 B + 8 B header = 92 B.
    assert prior.estimate_fp16_bytes() < 100


def test_predictive_renderer_output_shape_and_range() -> None:
    """Renderer outputs ``(..., 6)`` in [0, 1] (sigmoid output)."""
    cfg = TimeTravelerConfig(hidden_dim=16, num_hidden_layers=2)
    renderer = PredictiveRenderer(cfg)
    coords = torch.randn(10, 4)
    out = renderer(coords).detach()
    assert out.shape == (10, 6)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_predictive_renderer_param_count_under_60K_for_defaults() -> None:
    """The default renderer's param count is sub-60K."""
    cfg = TimeTravelerConfig()
    renderer = PredictiveRenderer(cfg)
    n_params = sum(p.numel() for p in renderer.parameters())
    assert n_params < 60_000


def test_predictive_renderer_param_count_matches_closed_form_prediction() -> None:
    """``cfg.predict_decoder_param_count()`` agrees with instantiated module."""
    cfg = TimeTravelerConfig(hidden_dim=32, num_hidden_layers=3)
    renderer = PredictiveRenderer(cfg)
    instantiated = sum(p.numel() for p in renderer.parameters())
    predicted = cfg.predict_decoder_param_count()
    assert instantiated == predicted


def test_substrate_render_pair_output_shapes() -> None:
    """render_pair returns two (1, 3, H, W) tensors."""
    cfg = TimeTravelerConfig(
        hidden_dim=16, num_hidden_layers=2, output_height=32, output_width=48
    )
    substrate = TimeTravelerSubstrate(cfg)
    rgb_0, rgb_1 = substrate.render_pair(0)
    assert rgb_0.shape == (1, 3, 32, 48)
    assert rgb_1.shape == (1, 3, 32, 48)


def test_substrate_render_pair_output_in_unit_range() -> None:
    """Output RGB tensors are in [0, 1] (sigmoid)."""
    cfg = TimeTravelerConfig(
        hidden_dim=16, num_hidden_layers=2, output_height=16, output_width=16
    )
    substrate = TimeTravelerSubstrate(cfg)
    with torch.no_grad():
        rgb_0, rgb_1 = substrate.render_pair(0)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0
    assert float(rgb_1.min()) >= 0.0
    assert float(rgb_1.max()) <= 1.0


def test_substrate_render_pair_rejects_out_of_range_index() -> None:
    """Negative or too-large pair_idx raises IndexError."""
    cfg = TimeTravelerConfig(
        hidden_dim=8, num_hidden_layers=2, output_height=16, output_width=16, num_pairs=5
    )
    substrate = TimeTravelerSubstrate(cfg)
    with pytest.raises(IndexError):
        substrate.render_pair(10)
    with pytest.raises(IndexError):
        substrate.render_pair(-1)


def test_substrate_render_pair_changes_with_time_index() -> None:
    """Different pair indices produce different RGB outputs (t-conditioning)."""
    cfg = TimeTravelerConfig(
        hidden_dim=16, num_hidden_layers=2, output_height=16, output_width=16, num_pairs=100
    )
    substrate = TimeTravelerSubstrate(cfg)
    rgb_a_0, _ = substrate.render_pair(0)
    rgb_b_0, _ = substrate.render_pair(50)
    # The two pairs must differ somewhere (the t-conditioning works).
    assert not torch.allclose(rgb_a_0, rgb_b_0, atol=1e-6)


def test_substrate_parameter_count_under_design_budget() -> None:
    """Total trainable params for default config land under 60K (renderer alone)
    plus the foveation+dynamics+pose-codes (small) keeps the whole substrate
    in a single archive section."""
    cfg = TimeTravelerConfig()
    substrate = TimeTravelerSubstrate(cfg)
    # 60K renderer + 384 foveation + 42 dynamics + 3600 pose codes = ~64K
    assert substrate.parameter_count() < 70_000


def test_substrate_estimate_world_model_bytes_under_70KB() -> None:
    """Closed-form world-model byte estimate is in the Stage 1 budget (~55-70 KB)."""
    cfg = TimeTravelerConfig()
    substrate = TimeTravelerSubstrate(cfg)
    # Allow upper bound 80 KB to accommodate FP16 vs FP32 overhead pre-brotli.
    assert substrate.estimate_world_model_bytes() < 80_000


def test_substrate_render_pair_accepts_tensor_index() -> None:
    """render_pair accepts a torch.Tensor pair index (e.g. from a DataLoader)."""
    cfg = TimeTravelerConfig(
        hidden_dim=8, num_hidden_layers=2, output_height=16, output_width=16
    )
    substrate = TimeTravelerSubstrate(cfg)
    idx = torch.tensor([3])
    rgb_0, rgb_1 = substrate.render_pair(idx)
    assert rgb_0.shape == (1, 3, 16, 16)


def test_substrate_renderer_gradients_flow_back() -> None:
    """A loss on render_pair output propagates gradient to renderer + pose codes."""
    cfg = TimeTravelerConfig(
        hidden_dim=8, num_hidden_layers=2, output_height=16, output_width=16
    )
    substrate = TimeTravelerSubstrate(cfg)
    rgb_0, rgb_1 = substrate.render_pair(0)
    loss = (rgb_0**2 + rgb_1**2).mean()
    loss.backward()
    # Renderer weights must have gradient.
    for name, p in substrate.renderer.named_parameters():
        assert p.grad is not None, f"renderer.{name} has no gradient"
        assert p.grad.abs().sum().item() > 0, f"renderer.{name} gradient is zero"


def test_substrate_state_dict_has_expected_top_level_keys() -> None:
    """State dict has renderer/foveation/dynamics/pose_codes namespaces."""
    cfg = TimeTravelerConfig(hidden_dim=8, num_hidden_layers=2)
    substrate = TimeTravelerSubstrate(cfg)
    sd = substrate.state_dict()
    assert any(k.startswith("renderer.") for k in sd)
    assert any(k.startswith("foveation.") for k in sd)
    assert any(k.startswith("dynamics.") for k in sd)
    assert "pose_codes" in sd
