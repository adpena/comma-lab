# SPDX-License-Identifier: MIT
"""Canonical tests for tac.inverse_steganalysis_real_video_mlx.

Per Catalog #213 + the operator binding "no fake implementations" invariant:
tests use REAL ``upstream/videos/0.mkv`` decoded frames when available, NOT
synthetic random noise. Synthetic test fixtures are used only for unit tests
of the math primitives (where the input distribution does not matter — e.g.
KB-kernel impulse response).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.inverse_steganalysis_real_video_mlx import (
    CANONICAL_HUGO_4_DIRECTION_OFFSETS,
    CANONICAL_HUGO_SPAM_TRUNCATION_T,
    CANONICAL_KB_KERNEL_3X3,
    CANONICAL_N_PAIRS_DEFAULT,
    CANONICAL_RENDERER_RESOLUTION,
    CANONICAL_SOURCE_RESOLUTION,
    CANONICAL_UNIWARD_HH_KERNEL,
    CANONICAL_UNIWARD_HL_KERNEL,
    CANONICAL_UNIWARD_LH_KERNEL,
    CANONICAL_UPSTREAM_VIDEO_PATH,
    MACOS_CPU_ADVISORY_TAG,
    MACOS_MLX_RESEARCH_SIGNAL_TAG,
    PREDICTED_AXIS_TAG,
    CanonicalSmokeResult,
    compute_hill_per_pixel_cost_mlx,
    compute_hugo_per_pixel_spam_delta_mlx,
    compute_mipod_per_pixel_cost_mlx,
    compute_uniward_per_pixel_directional_wavelet_mlx,
    conv2d_mlx,
    decode_upstream_video_frames,
    run_macos_cpu_advisory_smoke,
    wiener_filter_canonical_mlx,
)

# Canonical fixture flag: skip real-video tests when video is not present
# (e.g. CI without the canonical upstream/0.mkv).
_REAL_VIDEO_AVAILABLE = CANONICAL_UPSTREAM_VIDEO_PATH.exists()


# -----------------------------------------------------------------------------
# CANONICAL CONSTANTS TESTS
# -----------------------------------------------------------------------------


class TestCanonicalConstants:
    """Canonical constants are pinned per the canonical references."""

    def test_kb_kernel_is_canonical_li_wang_2014(self):
        """KB kernel matches Li-Wang-Li-Huang 2014 + Ker-Bohme 2008."""
        assert CANONICAL_KB_KERNEL_3X3.shape == (3, 3)
        assert CANONICAL_KB_KERNEL_3X3.dtype == np.float32
        # Center weight is -1.0 per (1/4)*[-4] = -1
        assert CANONICAL_KB_KERNEL_3X3[1, 1] == pytest.approx(-1.0)
        # Sum of all weights is 0 (canonical high-pass property)
        assert float(np.sum(CANONICAL_KB_KERNEL_3X3)) == pytest.approx(0.0, abs=1e-6)
        # Symmetry
        assert CANONICAL_KB_KERNEL_3X3[0, 0] == pytest.approx(-0.25)
        assert CANONICAL_KB_KERNEL_3X3[0, 1] == pytest.approx(0.50)
        assert CANONICAL_KB_KERNEL_3X3[1, 0] == pytest.approx(0.50)

    def test_canonical_resolutions(self):
        """Canonical resolutions match the contest scorer pipeline."""
        assert CANONICAL_SOURCE_RESOLUTION == (1164, 874)
        assert CANONICAL_RENDERER_RESOLUTION == (384, 512)
        assert CANONICAL_N_PAIRS_DEFAULT == 600

    def test_canonical_evidence_tags(self):
        """Catalog #192 evidence tags are canonical strings."""
        assert MACOS_CPU_ADVISORY_TAG == "[macOS-CPU advisory]"
        assert MACOS_MLX_RESEARCH_SIGNAL_TAG == "[macOS-MLX research-signal]"
        assert PREDICTED_AXIS_TAG == "[predicted]"

    def test_uniward_kernels_canonical(self):
        """UNIWARD directional kernels are 3x3 fp32."""
        for k in (
            CANONICAL_UNIWARD_LH_KERNEL,
            CANONICAL_UNIWARD_HL_KERNEL,
            CANONICAL_UNIWARD_HH_KERNEL,
        ):
            assert k.shape == (3, 3)
            assert k.dtype == np.float32

    def test_hugo_offsets_canonical(self):
        """HUGO 4-direction offsets match Pevný-Bas-Fridrich."""
        assert CANONICAL_HUGO_4_DIRECTION_OFFSETS == (
            (0, 1), (1, 0), (1, 1), (1, -1)
        )
        assert CANONICAL_HUGO_SPAM_TRUNCATION_T == 4


# -----------------------------------------------------------------------------
# CANONICAL MLX CONV2D TESTS
# -----------------------------------------------------------------------------


class TestConv2dMlx:
    """Canonical conv2d primitive correctness."""

    def test_conv2d_identity_kernel(self):
        """Identity kernel returns input unchanged."""
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        identity = np.array([[0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0]], dtype=np.float32)
        out = conv2d_mlx(image, identity, use_mlx=False)
        np.testing.assert_allclose(out, image, atol=1e-6)

    def test_conv2d_identity_kernel_mlx(self):
        """Identity kernel returns input unchanged via MLX."""
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        identity = np.array([[0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0]], dtype=np.float32)
        out = conv2d_mlx(image, identity, use_mlx=True)
        np.testing.assert_allclose(out, image, atol=1e-6)

    def test_conv2d_kb_kernel_on_constant_yields_zero(self):
        """KB high-pass on constant image yields ~0 (canonical high-pass property)."""
        image = np.full((16, 16), 0.5, dtype=np.float32)
        out = conv2d_mlx(image, CANONICAL_KB_KERNEL_3X3, use_mlx=True)
        # Interior pixels should be exactly 0; boundary may have artifacts.
        interior = out[2:-2, 2:-2]
        np.testing.assert_allclose(interior, 0.0, atol=1e-5)

    def test_conv2d_kb_kernel_impulse_response(self):
        """KB kernel impulse response matches canonical center weight."""
        # Place an impulse at center; the convolution output at the center pixel
        # equals the kernel center weight scaled by the impulse.
        H, W = 9, 9
        image = np.zeros((H, W), dtype=np.float32)
        image[H // 2, W // 2] = 1.0
        out = conv2d_mlx(image, CANONICAL_KB_KERNEL_3X3, use_mlx=False)
        # Center of output should equal -1.0 (KB center weight)
        assert out[H // 2, W // 2] == pytest.approx(-1.0, abs=1e-5)
        # Cardinal neighbors should equal +0.50
        assert out[H // 2 - 1, W // 2] == pytest.approx(0.50, abs=1e-5)
        assert out[H // 2 + 1, W // 2] == pytest.approx(0.50, abs=1e-5)
        assert out[H // 2, W // 2 - 1] == pytest.approx(0.50, abs=1e-5)
        assert out[H // 2, W // 2 + 1] == pytest.approx(0.50, abs=1e-5)

    def test_conv2d_box_filter_is_local_mean(self):
        """3x3 box filter (1/9 per cell) yields local mean."""
        image = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)
        box = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
        out = conv2d_mlx(image, box, use_mlx=True)
        # Center pixel sees all 9 ones; mean = 1.0
        assert out[1, 1] == pytest.approx(1.0, abs=1e-5)

    def test_conv2d_rejects_2d_only_nonexistent_for_3d(self):
        with pytest.raises(ValueError):
            conv2d_mlx(np.zeros((4, 4, 3), dtype=np.float32),
                       CANONICAL_KB_KERNEL_3X3, use_mlx=False)

    def test_conv2d_rejects_even_kernel(self):
        with pytest.raises(ValueError):
            conv2d_mlx(np.zeros((4, 4), dtype=np.float32),
                       np.zeros((2, 2), dtype=np.float32), use_mlx=False)

    def test_conv2d_mlx_vs_numpy_parity(self):
        """MLX and numpy paths produce same output to fp32 precision."""
        np.random.seed(42)
        image = np.random.rand(32, 32).astype(np.float32)
        out_mlx = conv2d_mlx(image, CANONICAL_KB_KERNEL_3X3, use_mlx=True)
        out_np = conv2d_mlx(image, CANONICAL_KB_KERNEL_3X3, use_mlx=False)
        np.testing.assert_allclose(out_mlx, out_np, atol=1e-5)


# -----------------------------------------------------------------------------
# CANONICAL HILL PER-PIXEL COST MATRIX TESTS
# -----------------------------------------------------------------------------


class TestHillPerPixelCost:
    """Canonical Li-Wang-Li-Huang 2014 HILL per-pixel cost matrix."""

    def test_hill_shape_preserved(self):
        """Output shape matches input shape (per-pixel)."""
        np.random.seed(42)
        luma = np.random.rand(64, 96).astype(np.float32)
        cost = compute_hill_per_pixel_cost_mlx(luma, use_mlx=True)
        assert cost.shape == luma.shape

    def test_hill_cost_positive(self):
        """Cost matrix is strictly positive (1/(intermediate+eps) is positive)."""
        np.random.seed(42)
        luma = np.random.rand(48, 64).astype(np.float32)
        cost = compute_hill_per_pixel_cost_mlx(luma, use_mlx=True)
        assert float(np.min(cost)) > 0

    def test_hill_constant_image_uniform_cost(self):
        """Constant image yields ~uniform cost (interior)."""
        luma = np.full((32, 32), 0.5, dtype=np.float32)
        cost = compute_hill_per_pixel_cost_mlx(luma, use_mlx=True)
        # KB-kernel residual is 0 on constant interior; intermediate is 0;
        # cost_reciprocal is 1/epsilon (LARGE); L2-smoothed is approximately 1/epsilon
        interior = cost[10:-10, 10:-10]
        # All interior cost values should be approximately equal (uniform)
        assert float(np.std(interior)) < 1e3  # very tight on constant input

    def test_hill_textured_vs_flat_discrimination(self):
        """HILL cost discriminates textured from flat regions."""
        # Build a half-flat half-textured image
        luma = np.full((64, 64), 0.5, dtype=np.float32)
        # Add high-frequency texture to right half
        np.random.seed(42)
        luma[:, 32:] = 0.5 + 0.3 * np.random.rand(64, 32).astype(np.float32)
        cost = compute_hill_per_pixel_cost_mlx(luma, use_mlx=True)
        # Per Li-Wang: HIGH cost = LOW embedding admissibility = MORE TEXTURED
        # So the textured (right) half should have LOWER mean cost than flat (left)
        # because residual is non-zero on textured -> intermediate is non-zero ->
        # reciprocal is small -> cost is small
        flat_cost_mean = float(np.mean(cost[:, 5:25]))
        textured_cost_mean = float(np.mean(cost[:, 40:60]))
        # Textured region has LOWER cost (because intermediate is larger -> reciprocal is smaller)
        assert textured_cost_mean < flat_cost_mean

    def test_hill_mlx_vs_numpy_parity(self):
        """MLX and numpy paths produce equivalent HILL cost."""
        np.random.seed(123)
        luma = np.random.rand(32, 48).astype(np.float32)
        cost_mlx = compute_hill_per_pixel_cost_mlx(luma, use_mlx=True)
        cost_np = compute_hill_per_pixel_cost_mlx(luma, use_mlx=False)
        np.testing.assert_allclose(cost_mlx, cost_np, rtol=1e-3, atol=1e-3)

    def test_hill_rejects_3d_input(self):
        with pytest.raises(ValueError):
            compute_hill_per_pixel_cost_mlx(np.zeros((3, 32, 32), dtype=np.float32))

    def test_hill_rejects_even_kernel(self):
        luma = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            compute_hill_per_pixel_cost_mlx(luma, l1_kernel_size=6)
        with pytest.raises(ValueError):
            compute_hill_per_pixel_cost_mlx(luma, l2_kernel_size=8)


# -----------------------------------------------------------------------------
# CANONICAL MiPOD REAL WIENER FILTER TESTS
# -----------------------------------------------------------------------------


class TestWienerFilter:
    """Canonical Sedighi-Cogranne-Fridrich 2016 REAL Wiener filter."""

    def test_wiener_filter_shape_preserved(self):
        np.random.seed(42)
        image = np.random.rand(32, 32).astype(np.float32)
        out = wiener_filter_canonical_mlx(image, use_mlx=True)
        assert out.shape == image.shape

    def test_wiener_filter_on_constant_returns_constant(self):
        """On constant input, Wiener filter returns constant (sigma_local^2 = 0)."""
        image = np.full((32, 32), 0.5, dtype=np.float32)
        out = wiener_filter_canonical_mlx(image, use_mlx=True)
        # Interior should be ~0.5 (boundary may differ due to zero-padding)
        interior = out[5:-5, 5:-5]
        np.testing.assert_allclose(interior, 0.5, atol=1e-4)

    def test_wiener_filter_passes_through_high_snr(self):
        """When sigma_local >> sigma_n, Wiener filter approximates identity."""
        # Build image with strong signal (high local variance)
        np.random.seed(42)
        image = (np.random.rand(32, 32) * 2.0 - 1.0).astype(np.float32) * 10.0
        # Force noise variance estimate to be near 0 (high SNR)
        out = wiener_filter_canonical_mlx(
            image, noise_variance_estimate=1e-8, use_mlx=True
        )
        # With near-zero noise estimate, SNR weight ~1.0, so output ~= input
        np.testing.assert_allclose(out, image, rtol=0.01)

    def test_wiener_filter_smooths_low_snr(self):
        """When sigma_local << sigma_n, Wiener filter approximates local mean."""
        # Build image where noise variance estimate is huge
        np.random.seed(42)
        image = np.random.rand(32, 32).astype(np.float32) * 0.01  # low signal
        out = wiener_filter_canonical_mlx(
            image, noise_variance_estimate=10.0, use_mlx=True
        )
        # When noise_var >> sigma_local, SNR weight ~0, so output ~= mean
        # The interior should be approximately the local mean.
        # On uniform random small input, local mean is ~0.005.
        interior = out[3:-3, 3:-3]
        # Std of output should be MUCH smaller than std of input
        assert float(np.std(interior)) < float(np.std(image)) * 0.5


class TestMipodPerPixelCost:
    """Canonical Sedighi-Cogranne-Fridrich 2016 MiPOD per-pixel Fisher-info cost."""

    def test_mipod_shape_preserved(self):
        np.random.seed(42)
        luma = np.random.rand(48, 64).astype(np.float32)
        cost = compute_mipod_per_pixel_cost_mlx(luma, use_mlx=True)
        assert cost.shape == luma.shape

    def test_mipod_cost_positive(self):
        np.random.seed(42)
        luma = np.random.rand(48, 64).astype(np.float32)
        cost = compute_mipod_per_pixel_cost_mlx(luma, use_mlx=True)
        assert float(np.min(cost)) > 0

    def test_mipod_cost_clipped(self):
        """Cost is clipped to [epsilon, clip_max]."""
        np.random.seed(42)
        luma = np.random.rand(32, 32).astype(np.float32)
        cost = compute_mipod_per_pixel_cost_mlx(
            luma, epsilon=1e-3, clip_max=1e3, use_mlx=True
        )
        assert float(np.max(cost)) <= 1e3
        assert float(np.min(cost)) >= 1e-3

    def test_mipod_flat_vs_textured_discrimination(self):
        """MiPOD inverse-variance cost is HIGH on flat (low-variance) regions."""
        # Build a half-flat half-textured image
        np.random.seed(42)
        luma = np.full((64, 64), 0.5, dtype=np.float32)
        luma[:, 32:] = 0.5 + 0.3 * np.random.rand(64, 32).astype(np.float32)
        cost = compute_mipod_per_pixel_cost_mlx(luma, use_mlx=True)
        flat_cost_mean = float(np.mean(cost[:, 5:25]))
        textured_cost_mean = float(np.mean(cost[:, 40:60]))
        # Flat region has LOW variance -> HIGH Fisher info -> HIGH cost
        assert flat_cost_mean > textured_cost_mean


# -----------------------------------------------------------------------------
# CANONICAL UNIWARD PER-PIXEL TESTS
# -----------------------------------------------------------------------------


class TestUniwardPerPixelDirectionalWavelet:
    """Canonical Holub-Fridrich-Denemark 2014 per-pixel UNIWARD."""

    def test_uniward_shape_preserved(self):
        np.random.seed(42)
        luma = np.random.rand(48, 64).astype(np.float32)
        cost = compute_uniward_per_pixel_directional_wavelet_mlx(luma)
        assert cost.shape == luma.shape

    def test_uniward_cost_positive(self):
        np.random.seed(42)
        luma = np.random.rand(48, 64).astype(np.float32)
        cost = compute_uniward_per_pixel_directional_wavelet_mlx(luma)
        assert float(np.min(cost)) > 0

    def test_uniward_constant_image_uniform_cost(self):
        """On constant image, wavelet coefficients are 0; cost is ~uniform high (1/sigma)."""
        luma = np.full((32, 32), 0.5, dtype=np.float32)
        cost = compute_uniward_per_pixel_directional_wavelet_mlx(luma, sigma=1.0)
        # All directional wavelet coeffs are 0 on constant -> cost = sum(delta_w / sigma)
        # Interior should be uniform
        interior = cost[5:-5, 5:-5]
        assert float(np.std(interior)) < 1e-2

    def test_uniward_textured_region_lower_cost(self):
        """Textured regions have larger |W_d| -> lower cost (paper formulation)."""
        np.random.seed(42)
        luma = np.full((64, 64), 0.5, dtype=np.float32)
        luma[:, 32:] = 0.5 + 0.4 * np.random.rand(64, 32).astype(np.float32)
        cost = compute_uniward_per_pixel_directional_wavelet_mlx(luma)
        flat_cost = float(np.mean(cost[:, 5:25]))
        textured_cost = float(np.mean(cost[:, 40:60]))
        # Per UNIWARD: |W_d| larger on textured -> 1/(|W_d|+sigma) smaller -> cost smaller
        assert textured_cost < flat_cost


# -----------------------------------------------------------------------------
# CANONICAL HUGO PER-PIXEL SPAM-DELTA TESTS
# -----------------------------------------------------------------------------


class TestHugoPerPixelSpamDelta:
    """Canonical Pevný-Filler-Bas 2010 HUGO per-pixel SPAM-delta."""

    def test_hugo_shape_preserved(self):
        np.random.seed(42)
        luma = np.random.rand(48, 64).astype(np.float32)
        cost = compute_hugo_per_pixel_spam_delta_mlx(luma)
        assert cost.shape == luma.shape

    def test_hugo_cost_non_negative(self):
        np.random.seed(42)
        luma = np.random.rand(32, 48).astype(np.float32)
        cost = compute_hugo_per_pixel_spam_delta_mlx(luma)
        assert float(np.min(cost)) >= 0

    def test_hugo_constant_image_uniform_cost(self):
        """Constant image yields uniform low cost (residual = 0 = canonical zero)."""
        luma = np.full((32, 32), 0.5, dtype=np.float32)
        cost = compute_hugo_per_pixel_spam_delta_mlx(luma)
        # On zero residual, the perturbation is not saturated in either
        # direction -> weight = 2.0 -> cost = 2.0 * 4 * pert_magnitude * 255
        # All pixels equal -> uniform cost
        interior = cost[2:-2, 2:-2]
        assert float(np.std(interior)) < 1e-2

    def test_hugo_rejects_bad_truncation(self):
        luma = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            compute_hugo_per_pixel_spam_delta_mlx(luma, truncation_t=0)


# -----------------------------------------------------------------------------
# CANONICAL REAL VIDEO INGESTION TESTS
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REAL_VIDEO_AVAILABLE,
    reason="upstream/videos/0.mkv not available",
)
class TestRealVideoDecode:
    """Canonical real-video frame decode tests (require upstream/videos/0.mkv)."""

    def test_decode_4_frames_rgb_default_resolution(self):
        frames = decode_upstream_video_frames(num_frames=4)
        assert frames.shape == (4, 3, 512, 384)  # (N, C, H, W) per default
        assert frames.dtype == np.float32
        assert 0.0 <= float(frames.min())
        assert float(frames.max()) <= 1.0

    def test_decode_4_frames_luma(self):
        luma = decode_upstream_video_frames(
            num_frames=4, target_resolution=(128, 96), return_format="luma_fp32"
        )
        assert luma.shape == (4, 96, 128)
        assert luma.dtype == np.float32
        assert 0.0 <= float(luma.min())
        assert float(luma.max()) <= 1.0

    def test_decode_rejects_bad_format(self):
        with pytest.raises(ValueError):
            decode_upstream_video_frames(num_frames=1, return_format="bogus")

    def test_decode_rejects_zero_frames(self):
        with pytest.raises(ValueError):
            decode_upstream_video_frames(num_frames=0)

    def test_decode_raises_on_missing_video(self):
        with pytest.raises(FileNotFoundError):
            decode_upstream_video_frames(video_path="nonexistent/video.mkv", num_frames=1)


# -----------------------------------------------------------------------------
# CANONICAL macOS-CPU ADVISORY SMOKE TESTS
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REAL_VIDEO_AVAILABLE,
    reason="upstream/videos/0.mkv not available",
)
class TestRunMacosCpuAdvisorySmoke:
    """Canonical end-to-end smoke runner tests."""

    def test_smoke_hill_per_pixel_real_frames(self):
        result = run_macos_cpu_advisory_smoke(
            target_name="hill_per_pixel_mlx_test",
            cost_function=compute_hill_per_pixel_cost_mlx,
            num_frames=2,
            target_resolution=(96, 72),
            use_mlx=True,
        )
        assert isinstance(result, CanonicalSmokeResult)
        assert result.target_name == "hill_per_pixel_mlx_test"
        assert result.n_frames_decoded == 2
        assert result.frame_resolution == (72, 96)  # (H, W)
        assert result.cost_matrix_shape == (72, 96)
        assert result.elapsed_seconds > 0
        # Canonical routing markers per Catalog #341
        assert result.canonical_routing_markers["predicted_delta_adjustment"] == 0.0
        assert result.canonical_routing_markers["promotable"] is False
        assert result.canonical_routing_markers["score_claim"] is False
        assert result.canonical_routing_markers["axis_tag"] == PREDICTED_AXIS_TAG
        # Canonical Provenance per Catalog #323
        assert result.canonical_provenance["score_claim_valid"] is False
        assert MACOS_CPU_ADVISORY_TAG in result.canonical_provenance["axis_tag"]
        # Dynamic range should be non-trivial on real video frames
        assert result.cost_matrix_dynamic_range_db > 0

    def test_smoke_to_dict_json_safe(self):
        result = run_macos_cpu_advisory_smoke(
            target_name="hill_test_dict",
            cost_function=compute_hill_per_pixel_cost_mlx,
            num_frames=1,
            target_resolution=(64, 48),
        )
        d = result.to_dict()
        # Must be JSON-serializable
        import json
        json_str = json.dumps(d)
        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["target_name"] == "hill_test_dict"
        assert parsed["used_mlx"] is True

    def test_smoke_real_video_yields_non_degenerate_dynamic_range(self):
        """Real video frames produce cost-discrimination (non-trivial dynamic range).

        This is the canonical anti-test against Slot EEE META-finding #1:
        synthetic random noise on inverse-steganalysis cost functions produces
        undifferentiated cost maps. Real video frames have textured AND flat
        regions, so the cost dynamic range must be non-trivial.
        """
        result = run_macos_cpu_advisory_smoke(
            target_name="hill_dynamic_range_test",
            cost_function=compute_hill_per_pixel_cost_mlx,
            num_frames=2,
            target_resolution=(128, 96),
        )
        # On real video, HILL cost should span at least 20 dB dynamic range
        # (canonical natural-image cost-discrimination indicator)
        assert result.cost_matrix_dynamic_range_db > 20.0, (
            f"HILL on real video should show >20 dB cost discrimination; "
            f"got {result.cost_matrix_dynamic_range_db} dB"
        )

    def test_smoke_mipod_per_pixel_real_frames(self):
        result = run_macos_cpu_advisory_smoke(
            target_name="mipod_per_pixel_mlx_test",
            cost_function=compute_mipod_per_pixel_cost_mlx,
            num_frames=1,
            target_resolution=(64, 48),
        )
        assert result.cost_matrix_dynamic_range_db > 0

    def test_smoke_uniward_per_pixel_real_frames(self):
        result = run_macos_cpu_advisory_smoke(
            target_name="uniward_per_pixel_mlx_test",
            cost_function=compute_uniward_per_pixel_directional_wavelet_mlx,
            num_frames=1,
            target_resolution=(64, 48),
        )
        assert result.cost_matrix_dynamic_range_db > 0

    def test_smoke_hugo_per_pixel_real_frames(self):
        result = run_macos_cpu_advisory_smoke(
            target_name="hugo_per_pixel_mlx_test",
            cost_function=compute_hugo_per_pixel_spam_delta_mlx,
            num_frames=1,
            target_resolution=(64, 48),
        )
        # HUGO cost can have low dynamic range when all residuals are non-saturated
        # but the elapsed time should be positive
        assert result.elapsed_seconds > 0
        assert result.cost_matrix_dynamic_range_db >= 0


class TestCanonicalRoutingMarkers:
    """Canonical Tier A routing markers per Catalog #341 + #357 + #317."""

    def test_routing_markers_non_promotable(self):
        from tac.inverse_steganalysis_real_video_mlx import (
            _build_canonical_routing_markers,
        )
        m = _build_canonical_routing_markers()
        assert m["promotable"] is False
        assert m["score_claim"] is False
        assert m["predicted_delta_adjustment"] == 0.0
        assert m["axis_tag"] == PREDICTED_AXIS_TAG
        assert "Catalog #192" in m["rationale"]
        assert "NEVER promotable" in m["rationale"]

    def test_provenance_non_promotable(self):
        from tac.inverse_steganalysis_real_video_mlx import (
            _build_canonical_provenance,
        )
        p = _build_canonical_provenance("test_target", "upstream/videos/0.mkv")
        assert p["score_claim_valid"] is False
        assert MACOS_CPU_ADVISORY_TAG in p["axis_tag"]
        assert p["evidence_grade"] == "predicted"
        assert "macos_arm64_mlx" in p["hardware_substrate"]
