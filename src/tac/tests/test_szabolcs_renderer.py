# SPDX-License-Identifier: MIT
"""Phase 1 tests for the Lane SZ szabolcs-cs PR#56 renderer replica.

Covers:
  * Forward-pass shape contract for ``SzabolcsRenderer``.
  * Output-range guarantees (sigmoid * 255 -> [0, 255]).
  * Gaussian-softmax LUT shape, normalization, and centroid sharpness.
  * ``encode_luma_to_probability_map`` round-trip from luma -> 5ch.
  * Parameter-count budget: total < 90K with reference settings (matches
    the upper bound from project_szabolcs_full_re_20260426 — competitor
    archive is 279KB total, model alone fits well under 90K params).
  * Frame-index validation (range guard, dtype, dim).

These are CPU-only smoke tests — Phase 1 is architectural only and never
touches CUDA / MPS / training data. The reference forward pass is
deterministic for a fixed seed, so we also assert reproducibility.
"""

from __future__ import annotations

import math

import pytest
import torch

from tac.contrib.szabolcs_renderer import (
    CAMERA_SIZE,
    CLASS_TARGETS,
    LUT_SIGMA,
    SEGMAP_INPUT_SIZE,
    SzabolcsRenderer,
    build_szabolcs_renderer,
    create_gaussian_softmax_lut,
    encode_luma_to_probability_map,
)


# ── Constants for tests ────────────────────────────────────────────────────


# Use a small spatial size for fast CPU tests; the architecture is fully
# spatial-agnostic (1x1 in/out + 3x3 residual blocks with padding=1).
TEST_H = 24
TEST_W = 32
TEST_BATCH = 2


# ── LUT tests ──────────────────────────────────────────────────────────────


def test_lut_shape_and_normalization() -> None:
    lut = create_gaussian_softmax_lut()
    assert lut.shape == (256, 5), f"expected (256, 5), got {lut.shape}"
    # Each row is a softmax distribution -> sums to 1.
    row_sums = lut.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(256), atol=1e-5)


def test_lut_class_centroids_argmax_match_targets() -> None:
    """Each class target's argmax row should pick that class.

    e.g. luma=0 -> class 0 (target=0), luma=255 -> class 1 (target=255),
    luma=64 -> class 2, etc.
    """
    lut = create_gaussian_softmax_lut()
    expected = {target: cls for cls, target in enumerate(CLASS_TARGETS)}
    for target, cls in expected.items():
        argmax = int(lut[target].argmax().item())
        assert argmax == cls, (
            f"luma {target} should map to class {cls} (target {target}); "
            f"got class {argmax}"
        )


def test_lut_sigma_15_softness() -> None:
    """sigma=15 should yield a soft distribution at midpoints between targets.

    Class targets are [0, 64, 128, 192, 255] (sorted). Halfway between 0 and
    64 (luma=32), at sigma=15, the distribution should NOT be one-hot —
    classes 0 and 2 should have non-trivial mass.
    """
    lut = create_gaussian_softmax_lut()
    midpoint_row = lut[32]
    top2 = midpoint_row.topk(2).values
    # Both top-2 classes should have > 0.05 probability — not one-hot.
    assert top2[1].item() > 0.05, (
        f"sigma={LUT_SIGMA} should produce soft distribution at midpoints; "
        f"top-2 was {top2.tolist()}"
    )


# ── encode_luma_to_probability_map ─────────────────────────────────────────


def test_encode_luma_shape_and_range() -> None:
    luma = torch.randint(0, 256, (TEST_BATCH, TEST_H, TEST_W), dtype=torch.float32)
    probs = encode_luma_to_probability_map(luma)
    assert probs.shape == (TEST_BATCH, 5, TEST_H, TEST_W)
    # Per-pixel softmax -> sums to 1 along channel dim.
    channel_sums = probs.sum(dim=1)
    assert torch.allclose(channel_sums, torch.ones_like(channel_sums), atol=1e-5)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_encode_luma_accepts_4d_with_singleton_channel() -> None:
    luma = torch.randint(0, 256, (TEST_BATCH, 1, TEST_H, TEST_W), dtype=torch.float32)
    probs = encode_luma_to_probability_map(luma)
    assert probs.shape == (TEST_BATCH, 5, TEST_H, TEST_W)


def test_encode_luma_clamps_out_of_range() -> None:
    """Negative or > 255 luma values should be clamped, not crash."""
    luma = torch.tensor(
        [[[-50.0, 300.0], [128.0, 64.0]]], dtype=torch.float32
    )  # shape (1, 2, 2)
    probs = encode_luma_to_probability_map(luma)
    assert probs.shape == (1, 5, 2, 2)
    assert torch.isfinite(probs).all()


# ── SzabolcsRenderer forward-pass shape ────────────────────────────────────


def _build_small_renderer(num_blocks: int = 2, hidden: int = 16) -> SzabolcsRenderer:
    """Tiny configuration for fast CPU tests; same shape contract as full."""
    return SzabolcsRenderer(
        hidden=hidden,
        block_hidden=hidden,
        num_blocks=num_blocks,
        max_frame_index=64,
        shared_latent_height=6,
        shared_latent_width=8,
    )


def test_renderer_forward_shape_and_range() -> None:
    torch.manual_seed(0)
    model = _build_small_renderer()
    model.eval()
    probs = torch.rand(TEST_BATCH, 5, TEST_H, TEST_W)
    # Renormalize per-pixel so input matches expected probability semantics.
    probs = probs / probs.sum(dim=1, keepdim=True)
    frame_indices = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        out = model(probs, frame_indices)
    assert out.shape == (TEST_BATCH, 3, TEST_H, TEST_W)
    assert torch.isfinite(out).all()
    # sigmoid * 255 -> strictly within (0, 255).
    assert (out >= 0).all()
    assert (out <= 255).all()


def test_renderer_forward_deterministic_in_eval() -> None:
    torch.manual_seed(0)
    model = _build_small_renderer()
    model.eval()
    probs = torch.rand(TEST_BATCH, 5, TEST_H, TEST_W)
    probs = probs / probs.sum(dim=1, keepdim=True)
    frame_indices = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        out1 = model(probs, frame_indices)
        out2 = model(probs, frame_indices)
    assert torch.equal(out1, out2)


def test_renderer_forward_supports_batch_one() -> None:
    """Batch=1 must work — common when running the inflate loop one frame at a time."""
    torch.manual_seed(0)
    model = _build_small_renderer()
    model.eval()
    probs = torch.rand(1, 5, TEST_H, TEST_W)
    probs = probs / probs.sum(dim=1, keepdim=True)
    frame_indices = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        out = model(probs, frame_indices)
    assert out.shape == (1, 3, TEST_H, TEST_W)


def test_renderer_grad_flows_to_all_params() -> None:
    """Smoke check: backprop reaches latent + affine + conv params."""
    torch.manual_seed(0)
    model = _build_small_renderer()
    probs = torch.rand(TEST_BATCH, 5, TEST_H, TEST_W)
    probs = probs / probs.sum(dim=1, keepdim=True)
    frame_indices = torch.tensor([0, 1], dtype=torch.long)
    out = model(probs, frame_indices)
    out.sum().backward()
    grad_seen = {name: (p.grad is not None and p.grad.abs().sum().item() > 0)
                 for name, p in model.named_parameters()}
    missing = [k for k, v in grad_seen.items() if not v]
    assert not missing, f"params with no gradient: {missing}"


# ── Forward-pass argument validation ───────────────────────────────────────


def test_renderer_rejects_wrong_channel_count() -> None:
    model = _build_small_renderer()
    bad = torch.rand(TEST_BATCH, 4, TEST_H, TEST_W)  # 4ch instead of 5
    frame_indices = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="channel count"):
        model(bad, frame_indices)


def test_renderer_rejects_wrong_input_dim() -> None:
    model = _build_small_renderer()
    bad = torch.rand(TEST_BATCH, 5, TEST_H * TEST_W)  # 3-D
    frame_indices = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        model(bad, frame_indices)


def test_renderer_rejects_wrong_frame_indices_shape() -> None:
    model = _build_small_renderer()
    probs = torch.rand(TEST_BATCH, 5, TEST_H, TEST_W)
    bad = torch.tensor([[0], [1]], dtype=torch.long)  # 2-D
    with pytest.raises(ValueError, match="1-D"):
        model(probs, bad)


def test_renderer_rejects_mismatched_batch() -> None:
    model = _build_small_renderer()
    probs = torch.rand(TEST_BATCH, 5, TEST_H, TEST_W)
    bad = torch.tensor([0, 1, 2], dtype=torch.long)
    with pytest.raises(ValueError, match="batch"):
        model(probs, bad)


# ── Parameter-count budget ─────────────────────────────────────────────────


def test_param_breakdown_keys_and_sum() -> None:
    model = _build_small_renderer()
    breakdown = model.param_breakdown()
    expected_keys = {
        "shared_latent",
        "frame_affine_embedding",
        "layer_in",
        "blocks_total",
        "layer_out",
        "total",
    }
    assert set(breakdown.keys()) == expected_keys
    parts = sum(v for k, v in breakdown.items() if k != "total")
    assert parts == breakdown["total"]
    # Cross-check against nn.Module enumeration.
    nn_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert nn_total == breakdown["total"]


def test_reference_full_size_under_90k_budget() -> None:
    """Reference settings (h=32, blocks=4, latent 30x40) should fit < 90K params.

    Per project_szabolcs_full_re_20260426 the competitor's whole archive is
    279KB; the model packed in block-FP at 1.017 bits/weight uses ~70KB,
    implying the model has roughly 70KB * 8 / 1.017 ~= 550K bit-budget. At
    ~64K params with mixed FP/quant, the under-90K budget is comfortable.
    """
    bundle = build_szabolcs_renderer(quiet=True)
    assert bundle.total_params < 90_000, (
        f"reference SzabolcsRenderer ballooned to {bundle.total_params:,} params"
    )


def test_reference_param_components_match_known_values() -> None:
    """Anchor parameter counts so future refactors can't silently drift.

    Reference architecture (h=32, blocks=4, block_hidden=32, latent 3x30x40,
    1200 frames) has tightly-defined per-component counts. These values are
    derived from the math, not measured — if the constructor is correct
    they will hold byte-for-byte.
    """
    bundle = build_szabolcs_renderer(quiet=True)
    bd = bundle.param_breakdown
    # Shared latent: 1 * 3 * 30 * 40 = 3600.
    assert bd["shared_latent"] == 3600
    # Frame affine: 1200 * 6 = 7200.
    assert bd["frame_affine_embedding"] == 7200
    # layer_in: Conv2d(8, 32, 1) -> 8*32 + 32 = 288.
    assert bd["layer_in"] == 8 * 32 + 32
    # layer_out: Conv2d(32, 3, 1) -> 32*3 + 3 = 99.
    assert bd["layer_out"] == 32 * 3 + 3
    # Each block: 2 * (32 * 32 * 3 * 3 + 32) = 2 * (9216 + 32) = 18496.
    per_block = 2 * (32 * 32 * 3 * 3 + 32)
    assert bd["blocks_total"] == 4 * per_block


# ── Affine-latent geometry ─────────────────────────────────────────────────


def test_affine_latent_channel_shape_matches_request() -> None:
    """The internal affine-latent helper should size its output exactly
    to the requested ``output_height`` x ``output_width`` (not the canvas
    size, which is the 1.25x intermediate).
    """
    model = _build_small_renderer()
    frame_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    latent = model._build_affine_latent_channel(frame_indices, 17, 23)
    assert latent.shape == (3, model.shared_latent_channels, 17, 23)


def test_zero_init_affine_yields_near_identity_warp() -> None:
    """With frame_affine_embedding initialized to zero, the affine should
    be a near-identity transform (zoom=1, aspect=0, shear=0, trans=0) and
    so the affine-latent channel should simply be a center-crop bicubic
    resample of the shared latent.
    """
    model = _build_small_renderer()
    # Verify the init contract.
    assert torch.equal(
        model.frame_affine_embedding.weight,
        torch.zeros_like(model.frame_affine_embedding.weight),
    )
    frame_indices = torch.tensor([0, 1], dtype=torch.long)
    latent = model._build_affine_latent_channel(frame_indices, 16, 24)
    # Both batch elements look at the same identity affine -> same content.
    assert torch.allclose(latent[0], latent[1], atol=1e-6)


# ── Build factory ──────────────────────────────────────────────────────────


def test_build_factory_returns_lut_with_correct_shape() -> None:
    bundle = build_szabolcs_renderer(quiet=True)
    assert bundle.lut.shape == (256, 5)
    assert isinstance(bundle.model, SzabolcsRenderer)
    assert bundle.total_params == bundle.param_breakdown["total"]


def test_build_factory_quiet_mode_does_not_print(capsys: pytest.CaptureFixture[str]) -> None:
    build_szabolcs_renderer(quiet=True)
    out = capsys.readouterr().out
    assert "szabolcs_renderer" not in out


def test_constants_match_reference() -> None:
    """Prevent silent drift of the contest-defined constants."""
    assert CAMERA_SIZE == (1164, 874)
    assert SEGMAP_INPUT_SIZE == (512, 384)
    assert CLASS_TARGETS == (0, 255, 64, 192, 128)
    assert math.isclose(LUT_SIGMA, 15.0)
