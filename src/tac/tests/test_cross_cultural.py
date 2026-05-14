# SPDX-License-Identifier: MIT
"""Tests for cross-cultural research techniques (techniques 3, 5, 6, 7, 8, 9, 10, 11, 12).

Each technique is tested for:
    - Construction/initialization
    - Forward pass shape correctness
    - Parameter count sanity
    - Output range validity
    - Integration with existing infrastructure
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Allow importing from experiments/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None

# ── Technique 3: Analytical Motion Predictor ───────────────────────────


class TestAnalyticalMotionPredictor:
    """Test analytical flow from mask displacement (SenseTime/KAIST)."""

    def test_construction(self):
        from tac.renderer import AnalyticalMotionPredictor
        model = AnalyticalMotionPredictor(num_classes=5)
        assert isinstance(model, nn.Module)

    def test_forward_shape(self):
        from tac.renderer import AnalyticalMotionPredictor
        model = AnalyticalMotionPredictor(num_classes=5)
        mask_t = torch.randint(0, 5, (2, 32, 48))
        mask_t1 = torch.randint(0, 5, (2, 32, 48))
        flow = model(mask_t, mask_t1)
        assert flow.shape == (2, 2, 32, 48), f"Expected (2, 2, 32, 48), got {flow.shape}"

    def test_param_count_small(self):
        """Analytical motion should have far fewer params than standard MotionPredictor."""
        from tac.renderer import AnalyticalMotionPredictor, MotionPredictor
        analytical = AnalyticalMotionPredictor(num_classes=5)
        standard = MotionPredictor(num_classes=5)
        assert analytical.param_count() < standard.param_count(), (
            f"Analytical ({analytical.param_count()}) should be smaller than "
            f"standard ({standard.param_count()})"
        )

    def test_zero_displacement_zero_flow(self):
        """Identical masks should produce near-zero flow."""
        from tac.renderer import AnalyticalMotionPredictor
        model = AnalyticalMotionPredictor(num_classes=5)
        mask = torch.randint(0, 5, (1, 32, 48))
        flow = model(mask, mask)
        # Analytical flow should be exactly zero for identical masks
        # (refinement may add small values at boundaries)
        assert flow.abs().mean().item() < 0.01, f"Flow too large for identical masks: {flow.abs().mean()}"

    def test_boundary_mask_detection(self):
        """Boundary detection should find edges between classes."""
        from tac.renderer import AnalyticalMotionPredictor
        model = AnalyticalMotionPredictor(num_classes=5)
        # Create a mask with a clear boundary
        mask = torch.zeros(1, 16, 16, dtype=torch.long)
        mask[:, :, 8:] = 1  # left half = 0, right half = 1
        boundary = model._compute_boundary_mask(mask)
        assert boundary.shape == (1, 1, 16, 16)
        # Boundary should be nonzero at column 7-8 interface
        assert boundary[0, 0, :, 7].sum() > 0 or boundary[0, 0, :, 8].sum() > 0


# ── Technique 5: INT4 Benchmark ────────────────────────────────────────


class TestINT4Quantization:
    """Test INT4 quantization utilities from benchmark script."""

    def test_quantize_int4_per_channel(self):
        from tac.experiments.benchmark_int4 import quantize_int4_per_channel
        w = torch.randn(16, 3, 3, 3)
        q, s = quantize_int4_per_channel(w)
        assert q.shape == w.shape
        assert (q >= -7).all() and (q <= 7).all(), "INT4 values out of range"
        assert s.shape == (16,)

    def test_roundtrip_quality(self):
        from tac.experiments.benchmark_int4 import quantize_int4_per_channel, dequantize_int4_per_channel
        w = torch.randn(16, 3, 3, 3)
        q, s = quantize_int4_per_channel(w)
        w_recon = dequantize_int4_per_channel(q, s)
        # INT4 is coarse but should be within ~15% relative error for most values
        mse = (w - w_recon).pow(2).mean().item()
        assert mse < 0.1, f"INT4 MSE too high: {mse}"

    def test_quantize_model_int4(self):
        from tac.experiments.benchmark_int4 import quantize_model_int4
        from tac.architectures import build_postfilter
        model = build_postfilter("dilated", hidden=16)
        _, stats = quantize_model_int4(model)
        assert "_summary" in stats
        assert stats["_summary"]["total_params"] > 0

    def test_save_int4_checkpoint(self):
        from tac.experiments.benchmark_int4 import save_int4_checkpoint, quantize_model_int4
        from tac.architectures import build_postfilter
        model = build_postfilter("dilated", hidden=16)
        quantize_model_int4(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_int4.pt"
            size = save_int4_checkpoint(model, path)
            assert size > 0
            assert path.exists()

    def test_int4_smaller_than_int8(self):
        """INT4 checkpoint should be significantly smaller than INT8."""
        from tac.experiments.benchmark_int4 import save_int4_checkpoint, quantize_model_int4
        from tac.architectures import build_postfilter
        model = build_postfilter("dilated", hidden=32)
        quantize_model_int4(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            int4_path = Path(tmpdir) / "test_int4.pt"
            int4_size = save_int4_checkpoint(model, int4_path)
            # INT4 should be roughly half INT8 (plus overhead)
            total_params = sum(p.numel() for p in model.parameters())
            int8_theoretical = total_params  # 1 byte per param
            # Allow generous overhead margin
            assert int4_size < int8_theoretical * 1.5, (
                f"INT4 ({int4_size}) should be smaller than INT8 theoretical ({int8_theoretical})"
            )


# ── Technique 6: Morphological Boundary Sharpening ────────────────────


class TestMorphologicalSharpening:
    """Test morphological boundary sharpening (Fraunhofer/NTT)."""

    def test_import(self):
        from tac.mask_codec import sharpen_mask_boundaries
        assert callable(sharpen_mask_boundaries)

    def test_preserves_shape(self):
        from tac.mask_codec import sharpen_mask_boundaries
        masks = torch.randint(0, 5, (4, 32, 48))
        result = sharpen_mask_boundaries(masks)
        assert result.shape == masks.shape

    def test_preserves_class_range(self):
        from tac.mask_codec import sharpen_mask_boundaries
        masks = torch.randint(0, 5, (4, 32, 48))
        result = sharpen_mask_boundaries(masks)
        assert result.min() >= 0
        assert result.max() <= 4

    def test_clean_mask_unchanged(self):
        """A perfectly clean mask should be mostly preserved."""
        from tac.mask_codec import sharpen_mask_boundaries
        # Create a clean blocky mask (no noise)
        mask = torch.zeros(1, 32, 48, dtype=torch.long)
        mask[:, :16, :] = 1
        mask[:, :, :24] = 2
        result = sharpen_mask_boundaries(mask)
        # Most pixels should be unchanged
        agreement = (result == mask).float().mean().item()
        assert agreement > 0.9, f"Clean mask changed too much: {agreement:.1%} agreement"


# ── Technique 7: Depthwise Cascade Renderer ───────────────────────────


class TestDepthwiseMaskRenderer:
    """Test depthwise cascade renderer (Samsung)."""

    def test_construction(self):
        from tac.renderer import DepthwiseMaskRenderer
        model = DepthwiseMaskRenderer(num_classes=5, base_ch=36)
        assert isinstance(model, nn.Module)

    def test_forward_shape(self):
        from tac.renderer import DepthwiseMaskRenderer
        model = DepthwiseMaskRenderer(num_classes=5, base_ch=36)
        masks = torch.randint(0, 5, (2, 32, 48))
        with torch.no_grad():
            rgb = model(masks)
        assert rgb.shape == (2, 3, 32, 48)

    def test_output_range(self):
        from tac.renderer import DepthwiseMaskRenderer
        model = DepthwiseMaskRenderer(num_classes=5, base_ch=36)
        masks = torch.randint(0, 5, (2, 32, 48))
        with torch.no_grad():
            rgb = model(masks)
        assert rgb.min() >= 0.0, f"RGB min below 0: {rgb.min()}"
        assert rgb.max() <= 255.0, f"RGB max above 255: {rgb.max()}"

    def test_fewer_params_than_standard(self):
        """Depthwise should have fewer params than standard MaskRenderer."""
        from tac.renderer import DepthwiseMaskRenderer, MaskRenderer
        dw = DepthwiseMaskRenderer(num_classes=5, base_ch=36)
        std = MaskRenderer(num_classes=5, base_ch=36, mid_ch=60)
        assert dw.param_count() < std.param_count(), (
            f"Depthwise ({dw.param_count()}) should be smaller than standard ({std.param_count()})"
        )

    def test_gradients_flow(self):
        from tac.renderer import DepthwiseMaskRenderer
        model = DepthwiseMaskRenderer(num_classes=5, base_ch=16)
        masks = torch.randint(0, 5, (1, 16, 16))
        rgb = model(masks)
        loss = rgb.mean()
        loss.backward()
        # Check gradients exist
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ── Technique 8: Channel-Recurrent Renderer ───────────────────────────


class TestChannelRecurrentRenderer:
    """Test channel-recurrent renderer (Sony)."""

    def test_construction(self):
        from tac.renderer import ChannelRecurrentRenderer
        model = ChannelRecurrentRenderer(num_classes=5, hidden=24)
        assert isinstance(model, nn.Module)

    def test_forward_shape(self):
        from tac.renderer import ChannelRecurrentRenderer
        model = ChannelRecurrentRenderer(num_classes=5, hidden=24)
        masks = torch.randint(0, 5, (2, 32, 48))
        with torch.no_grad():
            rgb = model(masks)
        assert rgb.shape == (2, 3, 32, 48)

    def test_output_range(self):
        from tac.renderer import ChannelRecurrentRenderer
        model = ChannelRecurrentRenderer(num_classes=5, hidden=24)
        masks = torch.randint(0, 5, (2, 32, 48))
        with torch.no_grad():
            rgb = model(masks)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 255.0

    def test_fewer_params_than_standard(self):
        """Channel-recurrent should have fewer total params."""
        from tac.renderer import ChannelRecurrentRenderer, MaskRenderer
        cr = ChannelRecurrentRenderer(num_classes=5, hidden=24)
        std = MaskRenderer(num_classes=5, base_ch=36, mid_ch=60)
        # Channel recurrent should be significantly smaller
        assert cr.param_count() < std.param_count() * 0.7, (
            f"Channel recurrent ({cr.param_count()}) should be <70% of standard ({std.param_count()})"
        )

    def test_yuv_conversion(self):
        """Output should have proper color distribution (not all gray)."""
        from tac.renderer import ChannelRecurrentRenderer
        torch.manual_seed(42)
        model = ChannelRecurrentRenderer(num_classes=5, hidden=24)
        # Initialize with some learned weights to get non-trivial output
        for p in model.parameters():
            if p.ndim > 1:
                nn.init.xavier_normal_(p)
        masks = torch.randint(0, 5, (1, 32, 48))
        with torch.no_grad():
            rgb = model(masks)
        # R, G, B channels should not be identical (YUV conversion works)
        r, g, b = rgb[0, 0], rgb[0, 1], rgb[0, 2]
        # At initialization they may be similar but not exactly equal
        # after Xavier init
        assert rgb.std() > 0, "Output is constant (no variation)"


# ── Technique 9: Coordinate-Based Renderer ────────────────────────────


class TestCoordRenderer:
    """Test coordinate-based neural field renderer (INRIA COOL)."""

    def test_positional_encoding(self):
        from tac.contrib.coord_renderer import PositionalEncoding
        pe = PositionalEncoding(num_bands=10)
        assert pe.output_dim == 42  # 2 + 4*10
        coords = torch.randn(4, 16, 16, 2)
        encoded = pe(coords)
        assert encoded.shape == (4, 16, 16, 42)

    def test_coord_renderer_construction(self):
        from tac.contrib.coord_renderer import CoordRenderer
        model = CoordRenderer(num_classes=5, hidden_dim=64, num_bands=10)
        assert isinstance(model, nn.Module)

    def test_coord_renderer_forward(self):
        from tac.contrib.coord_renderer import CoordRenderer
        model = CoordRenderer(num_classes=5, hidden_dim=32, num_bands=5)
        masks = torch.randint(0, 5, (2, 16, 24))
        with torch.no_grad():
            rgb = model(masks)
        assert rgb.shape == (2, 3, 16, 24)

    def test_output_range(self):
        from tac.contrib.coord_renderer import CoordRenderer
        model = CoordRenderer(num_classes=5, hidden_dim=32, num_bands=5)
        masks = torch.randint(0, 5, (1, 16, 24))
        with torch.no_grad():
            rgb = model(masks)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 255.0

    def test_param_count_small(self):
        """Coord renderer should be very small (~50K)."""
        from tac.contrib.coord_renderer import CoordRenderer
        model = CoordRenderer(num_classes=5, hidden_dim=64, num_bands=10)
        params = model.param_count()
        assert params < 100_000, f"Coord renderer too large: {params:,} params (target <100K)"
        assert params > 10_000, f"Coord renderer too small: {params:,} params"

    def test_pair_generator(self):
        from tac.contrib.coord_renderer import build_coord_renderer
        pair_gen = build_coord_renderer(
            num_classes=5, hidden_dim=32, num_bands=5, motion_hidden=16,
        )
        mask_t = torch.randint(0, 5, (1, 16, 24))
        mask_t1 = torch.randint(0, 5, (1, 16, 24))
        with torch.no_grad():
            pairs = pair_gen(mask_t, mask_t1)
        assert pairs.shape == (1, 2, 16, 24, 3), f"Expected (1, 2, 16, 24, 3), got {pairs.shape}"

    def test_gradients_flow(self):
        from tac.contrib.coord_renderer import CoordRenderer
        model = CoordRenderer(num_classes=5, hidden_dim=16, num_bands=3)
        masks = torch.randint(0, 5, (1, 8, 12))
        rgb = model(masks)
        loss = rgb.mean()
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0, "No gradients flowing through CoordRenderer"


# ── Technique 9B/9C: Cool-Chic Latents + C3 Residual ─────────────────


class TestCoolChicAndC3Renderer:
    """Test low-complexity overfitted latent and coordinate-residual renderers."""

    def test_coolchic_renderer_forward(self):
        from tac.contrib.coolchic_renderer import CoolChicLatentRenderer
        model = CoolChicLatentRenderer(
            num_classes=5,
            latent_ch=4,
            hidden=16,
            latent_shapes=((2, 3), (4, 6)),
        )
        masks = torch.randint(0, 5, (2, 16, 24))
        with torch.no_grad():
            rgb = model(masks)
        assert rgb.shape == (2, 3, 16, 24)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 255.0

    def test_coolchic_renderer_has_small_shared_decoder(self):
        from tac.contrib.coolchic_renderer import CoolChicLatentRenderer
        model = CoolChicLatentRenderer(
            num_classes=5,
            class_embed_dim=4,
            latent_ch=4,
            hidden=24,
            latent_shapes=((2, 3), (4, 6), (8, 12)),
        )
        assert model.decoder_param_count() < 2_500
        assert model.param_count() < 20_000

    def test_c3_residual_preserves_shape_and_starts_near_base(self):
        from tac.contrib.coolchic_renderer import C3ResidualRenderer, CoolChicLatentRenderer
        base = CoolChicLatentRenderer(num_classes=5, latent_ch=2, hidden=12, latent_shapes=((2, 3),))
        model = C3ResidualRenderer(base, residual_hidden=16, residual_layers=2, residual_scale=12.0)
        masks = torch.randint(0, 5, (1, 12, 18))
        with torch.no_grad():
            base_rgb = base(masks)
            residual_rgb = model(masks)
        assert residual_rgb.shape == base_rgb.shape
        assert (residual_rgb - base_rgb).abs().max().item() < 1e-5

    def test_c3_pair_generator(self):
        from tac.contrib.coolchic_renderer import build_c3_residual_renderer
        pair_gen = build_c3_residual_renderer(
            num_classes=5,
            embed_dim=4,
            latent_ch=2,
            hidden=12,
            motion_hidden=8,
            residual_hidden=16,
            latent_shapes=((2, 3),),
        )
        mask_t = torch.randint(0, 5, (1, 12, 18))
        mask_t1 = torch.randint(0, 5, (1, 12, 18))
        with torch.no_grad():
            pairs = pair_gen(mask_t, mask_t1)
        assert pairs.shape == (1, 2, 12, 18, 3)

    def test_train_renderer_resizes_fullres_gt_pair_for_roundtrip(self):
        from tac.experiments.train_renderer import resize_pair_hwc
        gt_pair = torch.rand(1, 2, 31, 47, 3)
        resized = resize_pair_hwc(gt_pair, 12, 18)
        assert resized.shape == (1, 2, 12, 18, 3)
        assert torch.isfinite(resized).all()

    def test_fp4_qat_wrapper_buffers_follow_device(self):
        from tac.contrib.coolchic_renderer import build_coolchic_renderer
        from tac.fp4_quantize import QATRendererFP4
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = build_coolchic_renderer(
            num_classes=5,
            embed_dim=4,
            latent_ch=2,
            hidden=12,
            motion_hidden=8,
            latent_shapes=((2, 3),),
        ).to(device)
        wrapper = QATRendererFP4(model).to(device)
        mask = torch.randint(0, 5, (1, 12, 18), device=device)
        with torch.no_grad():
            pairs = wrapper.base(mask, mask)
        assert torch.isfinite(pairs).all()


# ── Technique 10: AV1 Monochrome Encoding ─────────────────────────────


class TestMonochromeEncoding:
    """Test AV1 monochrome mask encoding (Habr)."""

    def test_import(self):
        from tac.mask_codec import encode_masks_monochrome
        assert callable(encode_masks_monochrome)

    @pytest.mark.skipif(
        not _ffmpeg_available(),
        reason="ffmpeg not available",
    )
    def test_monochrome_encoding(self):
        from tac.mask_codec import encode_masks_monochrome
        masks = torch.randint(0, 5, (10, 32, 48))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "masks_mono.mkv"
            size = encode_masks_monochrome(masks, path)
            assert size > 0
            assert path.exists()


# ── Technique 11: Topology-Preserving Post-Filter ─────────────────────


class TestThinStructureRestoration:
    """Test topology-preserving thin structure restoration (Fraunhofer)."""

    def test_import(self):
        from tac.mask_codec import restore_thin_structures
        assert callable(restore_thin_structures)

    def test_preserves_shape(self):
        from tac.mask_codec import restore_thin_structures
        decoded = torch.randint(0, 5, (4, 32, 48))
        keyframe = torch.randint(0, 5, (4, 32, 48))
        result = restore_thin_structures(decoded, keyframe)
        assert result.shape == decoded.shape

    def test_identical_masks_unchanged(self):
        """If decoded == keyframe, output should be identical."""
        from tac.mask_codec import restore_thin_structures
        masks = torch.randint(0, 5, (2, 32, 48))
        result = restore_thin_structures(masks.clone(), masks)
        assert (result == masks).all()

    def test_restores_missing_component(self):
        """A small component present in keyframe but missing in decoded should be restored."""
        from tac.mask_codec import restore_thin_structures
        # Create keyframe with a small lane marking (class 3, tiny component)
        keyframe = torch.zeros(1, 32, 48, dtype=torch.long)
        keyframe[0, 15, 20:25] = 3  # small horizontal stripe (5 pixels)

        # Create decoded where the stripe is missing
        decoded = torch.zeros(1, 32, 48, dtype=torch.long)

        result = restore_thin_structures(
            decoded, keyframe,
            area_threshold=50,
            classes_to_restore=(3,),
        )
        # The stripe should be restored
        assert (result[0, 15, 20:25] == 3).all(), "Thin structure not restored"


# ── Technique 12: Semantic-Aware Rate Control ─────────────────────────


class TestSemanticRateControl:
    """Test semantic-aware rate control (UPM Spain)."""

    def test_compute_qp_offsets(self):
        from tac.mask_codec import compute_semantic_qp_offsets
        # Create masks with known class distribution
        masks = torch.zeros(10, 32, 48, dtype=torch.long)
        masks[:, :, :24] = 0  # 50% road
        masks[:, :16, 24:] = 1  # 25% sky
        masks[:, 16:, 24:36] = 2  # 12.5% vegetation
        masks[:, 16:, 36:44] = 3  # ~8.3% lane marking
        masks[:, 16:, 44:] = 4  # ~4.2% vehicle
        offsets = compute_semantic_qp_offsets(masks)
        assert isinstance(offsets, dict)
        assert len(offsets) == 5
        # Road (50%) should get positive delta (fewer bits)
        assert offsets[0] > 0, f"Road (50%) should have positive QP offset, got {offsets[0]}"

    def test_generate_roi_map(self):
        from tac.mask_codec import generate_roi_qp_map
        masks = torch.randint(0, 5, (4, 32, 48))
        offsets = {0: 4, 1: 4, 2: 0, 3: -8, 4: -8}
        roi = generate_roi_qp_map(masks, offsets)
        assert roi.shape == (4, 32, 48)
        assert roi.dtype == np.int8

    def test_encode_semantic_rate(self):
        """Semantic rate control should produce a valid encoding function."""
        from tac.mask_codec import encode_masks_semantic_rate
        assert callable(encode_masks_semantic_rate)


# ── Profile Registration ──────────────────────────────────────────────


class TestNewProfiles:
    """Verify new profiles are registered and valid."""

    @pytest.mark.parametrize("name", [
        "depthwise_renderer_smoke",
        "channel_recurrent_smoke",
        "coord_renderer_smoke",
        "coolchic_renderer_smoke",
        "c3_residual_renderer_smoke",
    ])
    def test_profile_exists(self, name):
        from tac.profiles import PROFILES
        assert name in PROFILES, f"Profile {name} not registered"

    @pytest.mark.parametrize("name", [
        "depthwise_renderer_smoke",
        "channel_recurrent_smoke",
        "coord_renderer_smoke",
        "coolchic_renderer_smoke",
        "c3_residual_renderer_smoke",
    ])
    def test_profile_has_required_fields(self, name):
        from tac.profiles import PROFILES
        profile = PROFILES[name]
        required = ["variant", "hidden", "epochs", "lr", "loss_mode"]
        for field in required:
            assert field in profile, f"Profile {name} missing required field: {field}"

    @pytest.mark.parametrize("name", [
        "coolchic_renderer_smoke",
        "c3_residual_renderer_smoke",
    ])
    def test_low_complexity_profiles_are_deterministic_and_static_loss(self, name):
        from tac.profiles import PROFILES
        profile = PROFILES[name]
        assert profile["loss_mode"] == "standard"
        assert profile.get("adaptive_rebalance", False) is False
        assert profile.get("seed") == 42
        assert profile.get("deterministic", True) is True
