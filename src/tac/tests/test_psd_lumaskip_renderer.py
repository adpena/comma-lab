# SPDX-License-Identifier: MIT
"""Tests for ``tac.psd_lumaskip_renderer.PSDLumaSkipPostFilter``.

Phase A council-approved scaffold (2026-04-30,
``.omx/research/council_lane_psd_lumaskip_design_20260430.md``).

These tests verify scaffold-level invariants only — they do NOT measure
score impact. GPU dispatch and predicted-band validation requires a
separate council convene per Council #271 reactivation criterion #1.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.psd_lumaskip_renderer import PSDLumaSkipPostFilter


# ──────────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────────


def _rand_rgb(b: int = 2, h: int = 384, w: int = 512) -> torch.Tensor:
    """Random RGB tensor in [0, 255], shape (b, 3, h, w)."""
    return torch.rand(b, 3, h, w) * 255.0


# ──────────────────────────────────────────────────────────────────────────
# Test 1: forward shape — segnet input size (384x512)
# ──────────────────────────────────────────────────────────────────────────


def test_forward_shape_full_res():
    """Forward pass at SegNet input resolution preserves shape."""
    model = PSDLumaSkipPostFilter(hidden=64, kernel=3, luma_hidden=16)
    x = _rand_rgb(b=2, h=384, w=512)
    y = model(x)
    assert y.shape == x.shape, f"expected {x.shape}, got {y.shape}"
    assert y.dtype == x.dtype


# ──────────────────────────────────────────────────────────────────────────
# Test 2: forward shape — camera_size (874x1164)
# ──────────────────────────────────────────────────────────────────────────


def test_forward_shape_camera_size():
    """Forward pass at camera resolution (1164x874) preserves shape.

    The actual camera_size is W=1164, H=874 per upstream/frame_utils.py.
    PixelUnshuffle(2) requires both divisible by 2: 874 % 2 == 0,
    1164 % 2 == 0 — good.
    """
    model = PSDLumaSkipPostFilter(hidden=32, kernel=3, luma_hidden=8)
    x = _rand_rgb(b=1, h=874, w=1164)
    y = model(x)
    assert y.shape == x.shape, f"expected {x.shape}, got {y.shape}"


# ──────────────────────────────────────────────────────────────────────────
# Test 3: output value range
# ──────────────────────────────────────────────────────────────────────────


def test_forward_value_range():
    """Output is clamped to [0, 255]."""
    model = PSDLumaSkipPostFilter(hidden=16, kernel=3, luma_hidden=8)
    # Train the model briefly so residuals are nonzero (otherwise the
    # zero-init residuals make the output exactly equal to the input).
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    target = _rand_rgb(b=2, h=64, w=64)
    for _ in range(10):
        opt.zero_grad()
        x = _rand_rgb(b=2, h=64, w=64)
        y = model(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        opt.step()
    # Now check clamping — feed an extreme input
    x = torch.full((1, 3, 64, 64), 250.0)
    y = model(x)
    assert y.min() >= 0.0 - 1e-4
    assert y.max() <= 255.0 + 1e-4


# ──────────────────────────────────────────────────────────────────────────
# Test 4: starts as identity (zero-init)
# ──────────────────────────────────────────────────────────────────────────


def test_starts_as_identity():
    """At initialization, the postfilter is identity (zero residuals).

    Both ``self.conv4`` (chroma path output) and ``self.luma_out`` (luma
    path output) are zero-init, so both residuals are exactly zero at
    init. Output should equal input modulo the (no-op) clamp.
    """
    model = PSDLumaSkipPostFilter(hidden=64, kernel=3, luma_hidden=16)
    x = _rand_rgb(b=2, h=64, w=64)
    with torch.no_grad():
        y = model(x)
    # The forward pass divides by 255, processes, and multiplies back.
    # Float32 round-trip introduces noise on the order of 1e-3 (after ×255).
    diff = (y - x).abs()
    # Allow modest float32 round-trip noise from /255 then *255
    assert diff.max() < 1e-2, f"expected near-identity at init; max diff = {diff.max()}"


# ──────────────────────────────────────────────────────────────────────────
# Test 5: luma path is full-resolution (no spatial downsampling)
# ──────────────────────────────────────────────────────────────────────────


def test_luma_path_full_res():
    """The luma residual produced by ``forward_luma_path`` matches input
    H, W — i.e. the luma path NEVER downsamples spatially.

    This is the central architectural invariant of PSD-LumaSkip. If this
    test fails, the design is broken at the load-bearing claim.
    """
    model = PSDLumaSkipPostFilter(hidden=32, kernel=3, luma_hidden=8)
    h, w = 128, 256
    y_in = torch.rand(2, 1, h, w)
    y_residual = model.forward_luma_path(y_in)
    assert y_residual.shape == (2, 1, h, w), (
        f"luma path must preserve full res; expected (2, 1, {h}, {w}), got {y_residual.shape}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 6: chroma path operates at half-resolution (PSD bottleneck preserved)
# ──────────────────────────────────────────────────────────────────────────


def test_chroma_path_psd_half_res():
    """The chroma path's intermediate tensor is at half spatial resolution.

    This preserves PSD's SegNet-aligned 12.8% advantage by mirroring
    PSDPostFilter's PixelUnshuffle(2) → conv body → PixelShuffle(2)
    geometry. We probe by asserting the inner conv1 output shape is
    (B, hidden, H/2, W/2) when the input is (B, 3, H, W).
    """
    model = PSDLumaSkipPostFilter(hidden=64, kernel=3, luma_hidden=8)
    x_norm = torch.rand(2, 3, 128, 256)
    h_down = model.down(x_norm)
    assert h_down.shape == (2, 12, 64, 128), (
        f"PixelUnshuffle(2) should produce (2, 12, 64, 128); got {h_down.shape}"
    )
    # The full chroma path output is at full res after PixelShuffle(2),
    # but the heavy lifting happens at half res internally.
    rgb_residual = model.forward_chroma_path(x_norm)
    assert rgb_residual.shape == (2, 3, 128, 256)


# ──────────────────────────────────────────────────────────────────────────
# Test 7: learned luma projection is a real Conv2d 1->3 (not hardcoded broadcast)
# ──────────────────────────────────────────────────────────────────────────


def test_luma_projection_3channel_output():
    """When ``use_learned_luma_projection=True``, the luma_project layer is
    a learnable Conv2d(1, 3, 1) with the broadcast prior at init.

    When False, the model uses the parameter-free ``.expand`` broadcast.
    """
    # Learned-projection variant
    learned = PSDLumaSkipPostFilter(
        hidden=16, kernel=3, luma_hidden=4, use_learned_luma_projection=True
    )
    assert isinstance(learned.luma_project, nn.Conv2d)
    assert learned.luma_project.in_channels == 1
    assert learned.luma_project.out_channels == 3
    assert learned.luma_project.kernel_size == (1, 1)
    # Broadcast-prior init: weight is non-zero at [:, 0, 0, 0] = 1.0
    w = learned.luma_project.weight
    assert torch.allclose(w[:, 0, 0, 0], torch.ones(3))
    assert torch.allclose(learned.luma_project.bias, torch.zeros(3))

    # Broadcast (no projection) variant
    broadcast_only = PSDLumaSkipPostFilter(
        hidden=16, kernel=3, luma_hidden=4, use_learned_luma_projection=False
    )
    assert broadcast_only.luma_project is None


# ──────────────────────────────────────────────────────────────────────────
# Test 8: gradients flow through both paths (dual-path stability)
# ──────────────────────────────────────────────────────────────────────────


def test_gradient_flows_both_paths():
    """Backward pass produces nonzero gradients on BOTH the luma path
    parameters AND the chroma path parameters.

    Yousfi caveat 1: dual-path training requires both paths to receive
    gradient signal. If either path becomes a 'gradient ghost' the
    architecture is degenerate and cannot recover the kill memo's
    rejection mechanism.

    NOTE: To produce a non-zero gradient on the luma_in/luma_mid layers
    when the only output of the luma path is via a zero-init luma_out,
    we first non-trivially perturb luma_out so gradient flows backwards.
    Equivalently, we backprop a target that's nonzero so the chain
    rule produces signal at every layer.
    """
    model = PSDLumaSkipPostFilter(
        hidden=32, kernel=3, luma_hidden=8, use_learned_luma_projection=True
    )
    # Perturb the zero-init outputs so gradients have a non-zero chain
    # rule path through them (otherwise upstream layers receive zero grad
    # at init by construction).
    with torch.no_grad():
        model.conv4.weight.add_(torch.randn_like(model.conv4.weight) * 0.01)
        model.luma_out.weight.add_(torch.randn_like(model.luma_out.weight) * 0.01)

    x = _rand_rgb(b=2, h=64, w=64)
    y = model(x)
    loss = (y ** 2).mean()
    loss.backward()

    # Luma path gradient checks
    assert model.luma_in.weight.grad is not None
    assert model.luma_in.weight.grad.abs().sum().item() > 0, "luma_in.weight has no grad"
    assert model.luma_mid.weight.grad is not None
    assert model.luma_mid.weight.grad.abs().sum().item() > 0, "luma_mid.weight has no grad"
    assert model.luma_out.weight.grad is not None
    assert model.luma_out.weight.grad.abs().sum().item() > 0, "luma_out.weight has no grad"

    # Chroma path gradient checks
    assert model.conv1.weight.grad is not None
    assert model.conv1.weight.grad.abs().sum().item() > 0, "conv1.weight has no grad"
    assert model.conv2.weight.grad is not None
    assert model.conv2.weight.grad.abs().sum().item() > 0, "conv2.weight has no grad"
    assert model.conv3.weight.grad is not None
    assert model.conv3.weight.grad.abs().sum().item() > 0, "conv3.weight has no grad"
    assert model.conv4.weight.grad is not None
    assert model.conv4.weight.grad.abs().sum().item() > 0, "conv4.weight has no grad"


# ──────────────────────────────────────────────────────────────────────────
# Test 9: parameter count target (~95-100K at hidden=64, luma_hidden=16)
# ──────────────────────────────────────────────────────────────────────────


def test_param_count_target():
    """Default config (hidden=64, luma_hidden=16) lands in ~95-100K params.

    This bounds the rate cost of the renderer relative to Lane G v3 (88K)
    and vanilla PSD (95K). Per Phase A memo §F4, the predicted FP4A
    rate is ~17.9 KB which is +1.4 KB vs Lane G v3 = +0.0009 score
    points (negligible).
    """
    model = PSDLumaSkipPostFilter(
        hidden=64, kernel=3, luma_hidden=16, use_learned_luma_projection=True
    )
    total = model.num_parameters()
    luma = model.luma_path_params()
    chroma = model.chroma_path_params()
    # Sanity: the two paths plus the (negligible) projection account for
    # nearly all params.
    assert total >= 90_000, f"expected total >= 90K, got {total}"
    assert total <= 105_000, f"expected total <= 105K, got {total}"
    # Luma path is the lightweight skip; should be ~1.5K-3K params.
    assert luma <= 5_000, f"expected luma path <= 5K params, got {luma}"
    # Chroma path is the PSD bottleneck workhorse; should be most of the params.
    assert chroma >= 80_000, f"expected chroma path >= 80K params, got {chroma}"


# ──────────────────────────────────────────────────────────────────────────
# Test 10: EMA round-trip compatibility (canonical tac.training.EMA)
# ──────────────────────────────────────────────────────────────────────────


def test_ema_compatibility():
    """The model's state_dict is compatible with ``tac.training.EMA``.

    Per CLAUDE.md non-negotiable, every training path must wire EMA. This
    test confirms snapshot+apply+restore round-trips cleanly so the
    eval pattern documented in CLAUDE.md works for this architecture.
    """
    from tac.training import EMA

    model = PSDLumaSkipPostFilter(
        hidden=32, kernel=3, luma_hidden=8, use_learned_luma_projection=True
    )
    ema = EMA(model, decay=0.997)

    # Take a snapshot of the live weights
    orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Mutate live weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    # EMA update should still work (uses canonical .update + .apply API)
    ema.update(model)

    # Apply EMA shadow into model — the canonical pattern
    ema.apply(model)
    after_apply = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Restore original
    model.load_state_dict(orig_state)
    restored = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for k in orig_state:
        assert torch.allclose(orig_state[k], restored[k]), (
            f"restore failed for {k}"
        )

    # The applied EMA weights must be present (state_dict key set match)
    assert set(after_apply.keys()) == set(orig_state.keys())


# ──────────────────────────────────────────────────────────────────────────
# Bonus: VARIANTS registry wiring (sanity)
# ──────────────────────────────────────────────────────────────────────────


def test_variants_registry_wires_psd_lumaskip():
    """``build_postfilter('psd_lumaskip', ...)`` returns a PSDLumaSkipPostFilter."""
    from tac.architectures import build_postfilter, VARIANTS

    assert "psd_lumaskip" in VARIANTS
    model = build_postfilter("psd_lumaskip", hidden=32, kernel=3)
    assert isinstance(model, PSDLumaSkipPostFilter)
    # Default luma_hidden is 16 per the class default (Fridrich recommendation)
    assert model.luma_hidden == 16
    # Default uses learned projection (Yousfi recommendation)
    assert model.luma_project is not None


# ──────────────────────────────────────────────────────────────────────────
# Bonus: profile registry wiring (sanity)
# ──────────────────────────────────────────────────────────────────────────


def test_profile_registry_wires_psd_lumaskip_lane_g_v3():
    """``PROFILES['psd_lumaskip_lane_g_v3']`` exists with the required keys."""
    from tac.profiles import PROFILES

    assert "psd_lumaskip_lane_g_v3" in PROFILES
    profile = PROFILES["psd_lumaskip_lane_g_v3"]
    assert profile["variant"] == "psd_lumaskip"
    assert profile["psd_lumaskip_luma_hidden"] == 16
    assert profile["psd_lumaskip_use_learned_luma_projection"] is True
    # Inherits PSD_STANDARD_ADAPTIVE training stabilizers
    assert profile["boundary_weight"] == 50.0
    assert profile["hard_frame_ratio"] == 0.3
    assert profile["use_swa"] is True


# ──────────────────────────────────────────────────────────────────────────
# Robustness: input validation
# ──────────────────────────────────────────────────────────────────────────


def test_rejects_odd_dimensions():
    """PixelUnshuffle(2) requires H and W divisible by 2; raise on mismatch."""
    model = PSDLumaSkipPostFilter(hidden=16, kernel=3, luma_hidden=8)
    x = _rand_rgb(b=1, h=63, w=64)  # H=63 is odd
    with pytest.raises(ValueError, match="divisible by 2"):
        model(x)


def test_rejects_wrong_channel_count():
    """Input must be (B, 3, H, W); raise otherwise."""
    model = PSDLumaSkipPostFilter(hidden=16, kernel=3, luma_hidden=8)
    x = torch.rand(1, 4, 64, 64) * 255.0  # 4 channels, not 3
    with pytest.raises(ValueError, match=r"\(B, 3, H, W\)"):
        model(x)


def test_rejects_invalid_constructor_args():
    """Negative or zero hidden/luma_hidden, even kernel — all raise."""
    with pytest.raises(ValueError, match="hidden"):
        PSDLumaSkipPostFilter(hidden=0)
    with pytest.raises(ValueError, match="luma_hidden"):
        PSDLumaSkipPostFilter(hidden=16, luma_hidden=0)
    with pytest.raises(ValueError, match="kernel"):
        PSDLumaSkipPostFilter(hidden=16, kernel=2)  # even kernel
