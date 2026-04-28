"""Tests for swap_renderer_convs_with_learnable_bits — Lane Ω-V2 splice helper.

Mirrors test_self_compress_renderer.py's swap-helper tests but for the
per-WEIGHT learnable-bit wrapper.

Pins:
  1. Eligible Conv2d layers get swapped to LearnableBitConv2d.
  2. Protected layer-name suffixes (renderer.head, motion.head, FiLM linears,
     fuse_conv) are NEVER swapped.
  3. ConvTranspose2d is never swapped (skip_transposed=True default).
  4. After swap, model.named_parameters() includes the new bit_depth.raw
     parameters.
  5. Forward still works after swap (shape preserved).
  6. Hessian warm-start dict (full {"importance": ..., "metadata": ...}
     OR raw importance dict) initialises bits_used() ∝ √(I/median(I)).
  7. Helper utilities (list_learnable_bit_layers, total_bits, mean_bits,
     compute_learnable_bit_rate_penalty) work on a swapped model.
  8. Rate penalty is 0 when mean_bits ≤ target, > 0 when over target.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.learnable_bit_quant import (
    LearnableBitConv2d,
    compute_learnable_bit_rate_penalty,
    iter_eligible_conv_names,
    list_learnable_bit_layers,
    renderer_average_learnable_bits_per_weight,
    renderer_total_learnable_weight_bits,
    swap_renderer_convs_with_learnable_bits,
)
from tac.self_compress import SC_PROTECTED_NAME_PATTERNS


# ── Toy model that mimics the renderer's protected-layer pattern ─────────


class _ToyRenderer(nn.Module):
    """Minimal model with the same name-pattern surface as a real renderer.

    Has:
      - bulk Conv2d layers (should be swapped)
      - a `renderer.head` Conv2d (PROTECTED)
      - a `motion.head` Conv2d (PROTECTED)
      - a `fuse_conv` Conv2d (PROTECTED)
      - a `film_bottleneck.scale` Linear (PROTECTED — but Linear, not Conv)
      - a ConvTranspose2d (skipped by default)
    """

    def __init__(self):
        super().__init__()
        # Bulk convs
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.body = nn.Conv2d(16, 16, 3, padding=1)
        self.up_conv = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)

        # Sub-namespace "renderer" with a "head" (PROTECTED)
        self.renderer = nn.Module()
        self.renderer.head = nn.Conv2d(16, 3, 1)
        # Bulk conv inside "renderer" namespace
        self.renderer.bulk = nn.Conv2d(16, 16, 3, padding=1)

        # Sub-namespace "motion" with "head" (PROTECTED)
        self.motion = nn.Module()
        self.motion.head = nn.Conv2d(16, 2, 1)

        # Fuse conv (PROTECTED by suffix)
        self.fuse_conv = nn.Conv2d(16, 16, 1)

        # FiLM linear (PROTECTED by suffix; note Linear not Conv2d)
        self.film_bottleneck = nn.Module()
        self.film_bottleneck.scale = nn.Linear(6, 16)


# ── Swap behavior ───────────────────────────────────────────────────────


def test_swap_replaces_bulk_convs():
    model = _ToyRenderer()
    report = swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    swapped_names = report["swapped"]
    # stem, body, renderer.bulk should be swapped
    assert "stem" in swapped_names
    assert "body" in swapped_names
    assert "renderer.bulk" in swapped_names
    # Swapped modules are now LearnableBitConv2d
    assert isinstance(model.stem, LearnableBitConv2d)
    assert isinstance(model.body, LearnableBitConv2d)
    assert isinstance(model.renderer.bulk, LearnableBitConv2d)


def test_swap_skips_protected_renderer_head():
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    assert isinstance(model.renderer.head, nn.Conv2d)
    assert not isinstance(model.renderer.head, LearnableBitConv2d)


def test_swap_skips_protected_motion_head():
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    assert isinstance(model.motion.head, nn.Conv2d)
    assert not isinstance(model.motion.head, LearnableBitConv2d)


def test_swap_skips_fuse_conv_suffix():
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    assert isinstance(model.fuse_conv, nn.Conv2d)
    assert not isinstance(model.fuse_conv, LearnableBitConv2d)


def test_swap_protected_list_matches_sc_list():
    """The Lane Ω-V2 protected list MUST match Lane S's so the same
    weights are protected in both lanes (else cross-lane comparisons are
    invalid)."""
    model = _ToyRenderer()
    report = swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    # Every protected name must end with one of SC_PROTECTED_NAME_PATTERNS
    for name in report["protected"]:
        assert any(
            name == p or name.endswith("." + p) for p in SC_PROTECTED_NAME_PATTERNS
        ), f"protected name {name!r} doesn't match any SC_PROTECTED suffix"


def test_swap_skips_conv_transpose():
    model = _ToyRenderer()
    report = swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    assert isinstance(model.up_conv, nn.ConvTranspose2d)
    # Should appear in the skipped list
    assert any("up_conv" in s for s in report["skipped"])


def test_swap_copies_original_weights():
    model = _ToyRenderer()
    orig_w = model.stem.weight.data.clone()
    orig_b = model.stem.bias.data.clone()
    swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    assert torch.equal(model.stem.conv.weight.data, orig_w)
    assert torch.equal(model.stem.conv.bias.data, orig_b)


def test_swap_extra_protected_pattern_works():
    model = _ToyRenderer()
    report = swap_renderer_convs_with_learnable_bits(
        model, init_bits=8.0, extra_protected_patterns=("body",),
    )
    # body should now be in protected list
    assert "body" in report["protected"]
    assert isinstance(model.body, nn.Conv2d)
    assert not isinstance(model.body, LearnableBitConv2d)


# ── Hessian warm-start ──────────────────────────────────────────────────


def test_hessian_warm_start_raises_bits_for_high_importance():
    """Higher importance → higher initial bits_used()."""
    torch.manual_seed(0)
    model = _ToyRenderer()
    # Build a fake importance dict matching the eligible conv weights
    eligible = list(iter_eligible_conv_names(model))
    importance = {}
    for name in eligible:
        mod = dict(model.named_modules())[name]
        if isinstance(mod, nn.Conv2d):
            shape = mod.weight.shape
            # Half the weights are "high importance" (10x), half "low" (1x)
            t = torch.full(shape, 1.0)
            flat = t.reshape(-1)
            flat[: flat.numel() // 2] = 10.0
            importance[f"{name}.weight"] = t

    swap_report = swap_renderer_convs_with_learnable_bits(
        model, init_bits=4.0, hessian_init={"importance": importance,
                                            "metadata": {}},
    )
    assert len(swap_report["warm_started"]) > 0, (
        "warm_started list should be non-empty when hessian_init given"
    )

    # Pick the first swapped layer; verify its bits show variance (some
    # high, some low) — a uniform init would have zero variance.
    layer = list_learnable_bit_layers(model)[0][1]
    bits = layer.bit_depth.bits_used()
    assert bits.std().item() > 0.1, (
        "warm-started bits should have variance; got "
        f"std={bits.std().item():.4f}"
    )


def test_hessian_warm_start_accepts_bare_dict():
    """Should accept either {'importance': {...}, 'metadata': {...}}
    OR just the importance dict directly."""
    torch.manual_seed(0)
    model = _ToyRenderer()
    eligible = list(iter_eligible_conv_names(model))
    importance = {}
    for name in eligible:
        mod = dict(model.named_modules())[name]
        if isinstance(mod, nn.Conv2d):
            importance[f"{name}.weight"] = torch.rand(mod.weight.shape) + 0.1

    # Pass importance dict directly (no "importance" wrapper)
    swap_renderer_convs_with_learnable_bits(
        model, init_bits=4.0, hessian_init=importance,
    )
    layer = list_learnable_bit_layers(model)[0][1]
    bits = layer.bit_depth.bits_used()
    assert (bits >= 1.0).all() and (bits <= 8.0).all()


# ── After-swap utilities ─────────────────────────────────────────────────


def test_named_parameters_includes_bit_depth_raw():
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    param_names = [n for n, _ in model.named_parameters()]
    # At least one bit_depth.raw should appear
    assert any(".bit_depth.raw" in n for n in param_names), (
        f"named_parameters should include bit_depth.raw; got {param_names[:8]}"
    )


def test_list_learnable_bit_layers_returns_swapped():
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    layers = list_learnable_bit_layers(model)
    assert len(layers) >= 3  # stem, body, renderer.bulk
    for name, layer in layers:
        assert isinstance(layer, LearnableBitConv2d)


def test_renderer_total_bits_is_differentiable():
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    total = renderer_total_learnable_weight_bits(model)
    assert total.requires_grad
    total.backward()
    # At least one bit_depth.raw should have a non-None grad
    grads = [
        p.grad for n, p in model.named_parameters()
        if n.endswith(".bit_depth.raw")
    ]
    assert any(g is not None for g in grads)


def test_renderer_average_bits_close_to_init():
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=4.0)
    mean_bits = renderer_average_learnable_bits_per_weight(model)
    assert abs(mean_bits - 4.0) < 0.01


def test_rate_penalty_negative_when_under_target_controller():
    """Controller path: linear primal penalty is negative under slack.
    Round 4 (codex) restored linear; the dual decay (λ → 0) is what turns
    the rate term off in steady state, NOT a primal hinge. THIS path is
    only valid when lambda_rate is a LagrangianRateController instance
    (Round 5 split semantics).
    """
    from tac.learnable_bit_quant import LagrangianRateController
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=4.0)
    # Set initial_lambda=1.0 (lambda_rate is read-only @property; the
    # constructor's initial_lambda is the canonical setter).
    controller = LagrangianRateController(
        target_bits_per_weight=8.0, initial_lambda=1.0,
    )
    # When using a controller, target_bits_per_weight from the controller
    # overrides the positional arg (kept for legacy float signature).
    pen = compute_learnable_bit_rate_penalty(
        model, target_bits_per_weight=8.0, lambda_rate=controller,
    )
    mean_bits = renderer_average_learnable_bits_per_weight(model)
    expected = 1.0 * (mean_bits - 8.0)
    assert pen.item() == pytest.approx(expected, abs=1e-3)
    assert pen.item() < 0


def test_rate_penalty_zero_under_target_legacy_float():
    """Legacy float path: ReLU-clamped, ZERO under slack (Round 5 codex fix).
    Plain float callers have no dual update to decay λ; if linear primal
    were used here, gradient would be `+λ` constant under slack → SGD would
    push bits below the floor for no benefit.
    """
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=4.0)
    # mean_bits < target → ReLU clamps to zero
    pen = compute_learnable_bit_rate_penalty(
        model, target_bits_per_weight=8.0, lambda_rate=1.0,
    )
    assert pen.item() == 0.0, (
        f"legacy float path must be ReLU-clamped under slack; got {pen.item()} "
        f"— Round 5 split semantics require ReLU for float, linear for controller."
    )


def test_rate_penalty_positive_when_over_target():
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    pen = compute_learnable_bit_rate_penalty(model, target_bits_per_weight=2.0,
                                              lambda_rate=1.0)
    assert pen.item() > 0


def test_rate_penalty_grad_flows_back_to_bits():
    """Backprop through the rate penalty must reach bit_depth.raw."""
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    pen = compute_learnable_bit_rate_penalty(model, target_bits_per_weight=2.0,
                                              lambda_rate=1.0)
    pen.backward()
    grads = [
        p.grad for n, p in model.named_parameters()
        if n.endswith(".bit_depth.raw")
    ]
    # Every non-protected layer's bits should receive grad
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)


# ── Forward still works after swap ───────────────────────────────────────


def test_forward_after_swap_preserves_shape():
    """A simple forward through the bulk conv chain should still work."""
    model = _ToyRenderer()
    swap_renderer_convs_with_learnable_bits(model, init_bits=8.0)
    x = torch.randn(1, 3, 8, 8)
    y1 = model.stem(x)
    y2 = model.body(y1)
    y3 = model.renderer.bulk(y2)
    assert y3.shape == (1, 16, 8, 8)


# ── Empty-model edge case ────────────────────────────────────────────────


def test_helpers_safe_on_model_with_no_swapped_layers():
    """Helpers should return safe defaults on a model with no LearnableBitConv2d."""
    plain_model = nn.Sequential(nn.Linear(3, 3))
    layers = list_learnable_bit_layers(plain_model)
    assert layers == []
    assert renderer_average_learnable_bits_per_weight(plain_model) == 0.0
    total = renderer_total_learnable_weight_bits(plain_model)
    assert total.item() == 0.0
    pen = compute_learnable_bit_rate_penalty(plain_model, 2.0, 1.0)
    assert pen.item() == 0.0
