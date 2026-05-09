"""Tests for T22 — temporal-consistency regularizer.

Per coherence council 2026-05-09: T22 closes the temporal gap left by
T7+T8+T11+T20 (all per-frame signals). Cross-frame smoothness on
rendered RGB suppresses temporal flicker per Horn-Schunck 1981 brightness-
constancy / Anandan 1989 warp-residual.

Tests verify:

* Identical-frame input → loss zero (no flicker, no penalty).
* Motion-justified delta → small/zero loss when flow is correct.
* Boundary handling — all 3 grid_sample padding modes accepted.
* Gradient finite at edges and at large flow magnitudes.
* Batched-vs-loop equivalence — vectorized form matches per-pair loop.
* No-flow fallback collapses to pure adjacent-frame smoothness.
* Validation gates — lambda / boundary / flow shape / T < 2 fail-loud.
* Shape contract — accepts both (B, T, C, H, W) and (T, C, H, W) shorthand.
* Lambda multiplier scales loss linearly.
* warp_with_flow_grid_sample matches identity at zero flow.
* Config wrapper preserves config.lambda_weight + boundary_handling.
"""
from __future__ import annotations

import math

import pytest
import torch

from tac.temporal_consistency_regularizer import (
    DEFAULT_BOUNDARY_HANDLING,
    DEFAULT_LAMBDA_WEIGHT,
    VALID_BOUNDARY_HANDLING,
    VALID_FLOW_SOURCES,
    TemporalConsistencyConfig,
    apply_temporal_consistency,
    identity_warp_residual,
    temporal_consistency_loss,
    warp_with_flow_grid_sample,
)


# ---------------------------------------------------------------------------
# Convergence — identical frames → loss zero
# ---------------------------------------------------------------------------


def test_identical_frames_no_flow_loss_zero() -> None:
    torch.manual_seed(0)
    frame = torch.randn(1, 3, 16, 16)
    frames = frame.unsqueeze(1).expand(-1, 4, -1, -1, -1).contiguous()
    loss = temporal_consistency_loss(frames, flow=None)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_identical_frames_zero_flow_loss_zero() -> None:
    torch.manual_seed(1)
    frame = torch.randn(1, 3, 16, 16)
    frames = frame.unsqueeze(1).expand(-1, 4, -1, -1, -1).contiguous()
    flow = torch.zeros(1, 3, 2, 16, 16)
    loss = temporal_consistency_loss(frames, flow=flow)
    # Zero flow + identical frames → zero residual.
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_strictly_positive_when_frames_differ() -> None:
    torch.manual_seed(2)
    frames = torch.randn(1, 4, 3, 16, 16)
    loss = temporal_consistency_loss(frames, flow=None)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# Motion-justified delta — flow that matches the actual displacement → low loss
# ---------------------------------------------------------------------------


def test_translated_frames_with_correct_flow_low_loss() -> None:
    """Build two frames where the second is the first translated by (dx, dy);
    a flow encoding that exact translation should produce ~zero residual
    (modulo bilinear-interpolation truncation)."""
    torch.manual_seed(3)
    H, W = 32, 32
    img = torch.randn(1, 3, H, W)
    # Translate by 1 pixel in +x direction via a hand-crafted shifted frame.
    img_shifted = torch.roll(img, shifts=1, dims=-1)
    frames = torch.stack([img, img_shifted], dim=1)  # (1, 2, 3, H, W)
    # Build a flow that maps frame 0 → frame 1: each pixel needs to look up
    # the source pixel that's 1 px to the LEFT (because the OUTPUT shifted
    # right by 1 px). In normalized coords, dx = -2/(W-1).
    flow = torch.zeros(1, 1, 2, H, W)
    flow[:, :, 0, :, :] = -2.0 / (W - 1)  # x-displacement (look left)
    loss_with_flow = temporal_consistency_loss(frames, flow=flow)
    loss_no_flow = temporal_consistency_loss(frames, flow=None)
    # Flow-corrected residual should be MUCH smaller than no-flow residual.
    # Allow some slack for bilinear interp at the rolled boundary.
    assert loss_with_flow.item() < loss_no_flow.item() * 0.5


def test_no_motion_flow_equals_no_flow() -> None:
    """A zero-flow tensor must produce the same loss as flow=None."""
    torch.manual_seed(4)
    frames = torch.randn(2, 3, 3, 16, 16)
    flow_zero = torch.zeros(2, 2, 2, 16, 16)
    loss_zero = temporal_consistency_loss(frames, flow=flow_zero)
    loss_none = temporal_consistency_loss(frames, flow=None)
    assert loss_zero.item() == pytest.approx(loss_none.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Boundary handling
# ---------------------------------------------------------------------------


def test_all_boundary_modes_accepted() -> None:
    torch.manual_seed(5)
    frames = torch.randn(1, 2, 3, 16, 16)
    flow = torch.randn(1, 1, 2, 16, 16) * 0.05
    for boundary in VALID_BOUNDARY_HANDLING:
        loss = temporal_consistency_loss(
            frames, flow=flow, boundary_handling=boundary
        )
        assert math.isfinite(loss.item())


def test_invalid_boundary_mode_raises() -> None:
    frames = torch.randn(1, 2, 3, 8, 8)
    with pytest.raises(ValueError, match="boundary_handling"):
        temporal_consistency_loss(frames, boundary_handling="invalid")


def test_boundary_mode_must_not_be_int() -> None:
    frames = torch.randn(1, 2, 3, 8, 8)
    with pytest.raises(ValueError, match="boundary_handling"):
        temporal_consistency_loss(frames, boundary_handling=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Gradient flow + finite at extremes
# ---------------------------------------------------------------------------


def test_gradient_flows_through_rendered_frames() -> None:
    torch.manual_seed(6)
    frames = torch.randn(1, 3, 3, 16, 16, requires_grad=True)
    loss = temporal_consistency_loss(frames)
    loss.backward()
    assert frames.grad is not None
    assert torch.isfinite(frames.grad).all()
    # Gradient must be non-zero somewhere when the loss is non-zero.
    assert frames.grad.abs().sum().item() > 0.0


def test_gradient_flows_through_flow_when_provided() -> None:
    torch.manual_seed(7)
    frames = torch.randn(1, 2, 3, 16, 16)
    # Make flow a leaf tensor so grad accumulates onto it directly.
    flow = (torch.randn(1, 1, 2, 16, 16) * 0.1).detach().requires_grad_(True)
    loss = temporal_consistency_loss(frames, flow=flow)
    loss.backward()
    assert flow.grad is not None
    assert torch.isfinite(flow.grad).all()


def test_finite_loss_at_large_flow_magnitudes() -> None:
    torch.manual_seed(8)
    frames = torch.randn(1, 2, 3, 16, 16)
    # Large flow that pushes most pixels off the grid; padding handles edges.
    flow = torch.full((1, 1, 2, 16, 16), 5.0)  # way past [-1, 1]
    loss = temporal_consistency_loss(frames, flow=flow)
    assert math.isfinite(loss.item())


def test_finite_loss_with_zero_frames() -> None:
    frames = torch.zeros(1, 3, 3, 16, 16)
    loss = temporal_consistency_loss(frames)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Batched-vs-loop equivalence
# ---------------------------------------------------------------------------


def test_batched_no_flow_matches_per_batch_loop() -> None:
    torch.manual_seed(9)
    B = 4
    frames = torch.randn(B, 3, 3, 16, 16)
    batched = temporal_consistency_loss(frames).item()
    per_b = []
    for b in range(B):
        per_b.append(temporal_consistency_loss(frames[b : b + 1]).item())
    assert batched == pytest.approx(sum(per_b) / B, rel=1e-5)


def test_batched_with_flow_matches_per_batch_loop() -> None:
    torch.manual_seed(10)
    B = 3
    frames = torch.randn(B, 3, 3, 12, 12)
    flow = torch.randn(B, 2, 2, 12, 12) * 0.05
    batched = temporal_consistency_loss(frames, flow=flow).item()
    per_b = [
        temporal_consistency_loss(frames[b : b + 1], flow=flow[b : b + 1]).item()
        for b in range(B)
    ]
    assert batched == pytest.approx(sum(per_b) / B, rel=1e-5)


# ---------------------------------------------------------------------------
# No-flow fallback = pure adjacent-frame smoothness
# ---------------------------------------------------------------------------


def test_no_flow_fallback_equals_adjacent_diff_squared_mean() -> None:
    torch.manual_seed(11)
    frames = torch.randn(1, 4, 3, 16, 16)
    expected = (frames[:, 1:] - frames[:, :-1]).pow(2).mean().item()
    actual = temporal_consistency_loss(frames, flow=None, lambda_weight=1.0).item()
    assert actual == pytest.approx(expected, rel=1e-5)


def test_identity_warp_residual_helper_matches_no_flow_path() -> None:
    torch.manual_seed(12)
    frames = torch.randn(1, 3, 3, 12, 12)
    via_helper = identity_warp_residual(frames).item()
    via_main = temporal_consistency_loss(frames, flow=None).item()
    assert via_helper == pytest.approx(via_main, abs=1e-7)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


def test_lambda_must_be_non_negative() -> None:
    frames = torch.randn(1, 2, 3, 8, 8)
    with pytest.raises(ValueError, match="lambda_weight"):
        temporal_consistency_loss(frames, lambda_weight=-0.1)
    with pytest.raises(ValueError, match="lambda_weight"):
        temporal_consistency_loss(frames, lambda_weight=float("nan"))
    with pytest.raises(ValueError, match="lambda_weight"):
        temporal_consistency_loss(frames, lambda_weight=float("inf"))


def test_lambda_must_not_be_bool() -> None:
    frames = torch.randn(1, 2, 3, 8, 8)
    with pytest.raises(ValueError, match="lambda_weight"):
        temporal_consistency_loss(frames, lambda_weight=True)  # type: ignore[arg-type]


def test_lambda_zero_returns_zero_regardless_of_input() -> None:
    """λ=0 disables the regularizer entirely."""
    torch.manual_seed(13)
    frames = torch.randn(1, 4, 3, 16, 16)
    loss = temporal_consistency_loss(frames, lambda_weight=0.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_t_less_than_2_raises() -> None:
    frames = torch.randn(1, 1, 3, 8, 8)
    with pytest.raises(ValueError, match="T >= 2"):
        temporal_consistency_loss(frames)


def test_4d_input_accepted_as_single_batch() -> None:
    torch.manual_seed(14)
    frames = torch.randn(3, 3, 16, 16)  # (T, C, H, W)
    loss = temporal_consistency_loss(frames)
    assert math.isfinite(loss.item())


def test_3d_input_rejected() -> None:
    frames = torch.zeros(3, 16, 16)
    with pytest.raises(ValueError, match="\\(B, T, C, H, W\\)"):
        temporal_consistency_loss(frames)


def test_empty_input_raises() -> None:
    frames = torch.zeros(0, 2, 3, 8, 8)
    with pytest.raises(ValueError, match="non-empty"):
        temporal_consistency_loss(frames)


def test_flow_shape_mismatch_batch_raises() -> None:
    frames = torch.randn(2, 2, 3, 8, 8)
    flow = torch.randn(3, 1, 2, 8, 8)
    with pytest.raises(ValueError, match="flow batch"):
        temporal_consistency_loss(frames, flow=flow)


def test_flow_shape_mismatch_time_raises() -> None:
    frames = torch.randn(1, 4, 3, 8, 8)
    flow = torch.randn(1, 2, 2, 8, 8)  # wrong: expects T-1 = 3
    with pytest.raises(ValueError, match="time dim"):
        temporal_consistency_loss(frames, flow=flow)


def test_flow_shape_mismatch_channels_raises() -> None:
    frames = torch.randn(1, 2, 3, 8, 8)
    flow = torch.randn(1, 1, 3, 8, 8)  # wrong: must be 2
    with pytest.raises(ValueError, match="channel dim"):
        temporal_consistency_loss(frames, flow=flow)


def test_flow_shape_mismatch_spatial_raises() -> None:
    frames = torch.randn(1, 2, 3, 8, 8)
    flow = torch.randn(1, 1, 2, 16, 16)
    with pytest.raises(ValueError, match="spatial"):
        temporal_consistency_loss(frames, flow=flow)


def test_flow_4d_accepted_as_single_batch() -> None:
    torch.manual_seed(15)
    frames = torch.randn(2, 3, 8, 8)  # (T, C, H, W)
    flow = torch.randn(1, 2, 8, 8) * 0.05  # (T-1, 2, H, W)
    loss = temporal_consistency_loss(frames, flow=flow)
    assert math.isfinite(loss.item())


def test_flow_3d_rejected() -> None:
    frames = torch.randn(1, 2, 3, 8, 8)
    flow = torch.zeros(2, 8, 8)  # missing T-1 dim
    with pytest.raises(ValueError, match="\\(B, T-1, 2, H, W\\)"):
        temporal_consistency_loss(frames, flow=flow)


# ---------------------------------------------------------------------------
# Lambda multiplier linearity
# ---------------------------------------------------------------------------


def test_lambda_multiplier_scales_loss_linearly() -> None:
    torch.manual_seed(16)
    frames = torch.randn(1, 3, 3, 12, 12)
    loss_l1 = temporal_consistency_loss(frames, lambda_weight=1.0).item()
    loss_l5 = temporal_consistency_loss(frames, lambda_weight=5.0).item()
    assert loss_l5 == pytest.approx(5.0 * loss_l1, rel=1e-5)


# ---------------------------------------------------------------------------
# warp_with_flow_grid_sample primitive
# ---------------------------------------------------------------------------


def test_warp_with_zero_flow_is_identity() -> None:
    torch.manual_seed(17)
    image = torch.randn(2, 3, 16, 16)
    flow = torch.zeros(2, 2, 16, 16)
    warped = warp_with_flow_grid_sample(image, flow)
    assert torch.allclose(warped, image, atol=1e-5)


def test_warp_with_flow_grid_sample_validates_image_shape() -> None:
    image_bad = torch.zeros(3, 16, 16)
    flow = torch.zeros(1, 2, 16, 16)
    with pytest.raises(ValueError, match="\\(B, C, H, W\\)"):
        warp_with_flow_grid_sample(image_bad, flow)


def test_warp_with_flow_grid_sample_validates_flow_shape() -> None:
    image = torch.zeros(1, 3, 16, 16)
    flow_bad = torch.zeros(1, 3, 16, 16)  # wrong channel count
    with pytest.raises(ValueError, match="channel dim"):
        warp_with_flow_grid_sample(image, flow_bad)


def test_warp_with_flow_grid_sample_validates_batch_match() -> None:
    image = torch.zeros(2, 3, 16, 16)
    flow = torch.zeros(3, 2, 16, 16)
    with pytest.raises(ValueError, match="batch"):
        warp_with_flow_grid_sample(image, flow)


def test_warp_with_flow_grid_sample_validates_spatial_match() -> None:
    image = torch.zeros(1, 3, 16, 16)
    flow = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match="spatial"):
        warp_with_flow_grid_sample(image, flow)


# ---------------------------------------------------------------------------
# Config validation gates + frozen invariant
# ---------------------------------------------------------------------------


def test_config_default_values_match_module_defaults() -> None:
    config = TemporalConsistencyConfig()
    assert config.lambda_weight == DEFAULT_LAMBDA_WEIGHT == 0.1
    assert config.boundary_handling == DEFAULT_BOUNDARY_HANDLING == "border"
    assert config.flow_source == "identity"


def test_config_validates_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_weight"):
        TemporalConsistencyConfig(lambda_weight=-0.1)
    with pytest.raises(ValueError, match="lambda_weight"):
        TemporalConsistencyConfig(lambda_weight=float("inf"))


def test_config_validates_flow_source() -> None:
    with pytest.raises(ValueError, match="flow_source"):
        TemporalConsistencyConfig(flow_source="bogus")


def test_config_validates_boundary_handling() -> None:
    with pytest.raises(ValueError, match="boundary_handling"):
        TemporalConsistencyConfig(boundary_handling="bogus")


def test_config_is_frozen() -> None:
    config = TemporalConsistencyConfig()
    with pytest.raises(Exception):  # FrozenInstanceError or TypeError
        config.lambda_weight = 0.5  # type: ignore[misc]


def test_valid_flow_sources_constant_exposes_three() -> None:
    assert set(VALID_FLOW_SOURCES) == {"ego_motion", "estimated", "identity"}


# ---------------------------------------------------------------------------
# apply_temporal_consistency wrapper
# ---------------------------------------------------------------------------


def test_apply_temporal_consistency_default_config() -> None:
    torch.manual_seed(18)
    frames = torch.randn(1, 3, 3, 16, 16)
    config = TemporalConsistencyConfig()
    loss_via_wrapper = apply_temporal_consistency(frames, None, config)
    loss_direct = temporal_consistency_loss(frames, flow=None)
    assert loss_via_wrapper.item() == pytest.approx(loss_direct.item(), abs=1e-7)


def test_apply_temporal_consistency_propagates_lambda() -> None:
    torch.manual_seed(19)
    frames = torch.randn(1, 3, 3, 16, 16)
    config_l1 = TemporalConsistencyConfig(lambda_weight=1.0)
    config_l3 = TemporalConsistencyConfig(lambda_weight=3.0)
    l1 = apply_temporal_consistency(frames, None, config_l1).item()
    l3 = apply_temporal_consistency(frames, None, config_l3).item()
    assert l3 == pytest.approx(3.0 * l1, rel=1e-5)


def test_apply_temporal_consistency_propagates_boundary_handling() -> None:
    torch.manual_seed(20)
    frames = torch.randn(1, 2, 3, 16, 16)
    flow = torch.full((1, 1, 2, 16, 16), 0.5)  # large flow → boundary matters
    config_zeros = TemporalConsistencyConfig(boundary_handling="zeros")
    config_border = TemporalConsistencyConfig(boundary_handling="border")
    l_zeros = apply_temporal_consistency(frames, flow, config_zeros).item()
    l_border = apply_temporal_consistency(frames, flow, config_border).item()
    # The two boundary modes should give different residuals at this flow.
    assert l_zeros != pytest.approx(l_border, abs=1e-6)


def test_apply_temporal_consistency_rejects_non_config() -> None:
    frames = torch.zeros(1, 2, 3, 8, 8)
    with pytest.raises(TypeError, match="TemporalConsistencyConfig"):
        apply_temporal_consistency(frames, None, {"lambda_weight": 0.1})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Renderer-shape contract — works on canonical (B=1, T=600, C=3, H=384, W=512)
# size class via a downsampled stand-in (don't allocate 384×512×600 in tests)
# ---------------------------------------------------------------------------


def test_works_on_renderer_shape_class_downsampled() -> None:
    """Smoke: canonical renderer output is (B, T, 3, 384, 512). Use a much
    smaller stand-in so the test stays fast — the math is independent of
    spatial size."""
    torch.manual_seed(21)
    frames = torch.randn(1, 8, 3, 24, 32)  # downsampled stand-in
    loss = temporal_consistency_loss(frames)
    assert math.isfinite(loss.item())
    assert loss.item() > 0.0
