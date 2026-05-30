# SPDX-License-Identifier: MIT
"""Cross-backend parity + canonical-extraction tests for
:mod:`tac.framework_agnostic.canonical_kernels`.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable: every test uses
REAL numpy arrays + computes REAL outputs (not stub fixtures).

Per CLAUDE.md "Apples-to-apples evidence discipline": cross-backend
parity tests assert mathematically equivalent outputs within Slot 16
numerical tolerance per canonical equations:
  * ``mlx_pytorch_conv2d_fp64_accumulation_drift_reduction_v1``
  * ``mlx_pytorch_numerical_equivalence_within_tolerance_per_canonical_helper_v1``
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.framework_agnostic import Backend
from tac.framework_agnostic.canonical_kernels import (
    CANONICAL_CROSS_BACKEND_FP32_ATOL,
    CANONICAL_UNIMIX_ALPHA,
    assert_cross_backend_parity,
    gumbel_softmax_sample,
    rgb_to_yuv6,
)


# -----------------------------------------------------------------------------
# gumbel_softmax_sample canonical contract tests
# -----------------------------------------------------------------------------

class TestGumbelSoftmaxSample:
    def test_output_shape_matches_input(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.NUMPY, seed=42
        )
        assert result.shape == logits.shape

    def test_output_sums_to_one_per_sample(self):
        rng = np.random.default_rng(42)
        logits = rng.standard_normal(size=(4, 8)).astype(np.float32)
        result = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.NUMPY, seed=42
        )
        sums = np.sum(result, axis=-1)
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)

    def test_temperature_must_be_positive(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            gumbel_softmax_sample(
                logits, temperature=0.0, backend=Backend.NUMPY
            )

    def test_temperature_negative_rejected(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            gumbel_softmax_sample(
                logits, temperature=-1.0, backend=Backend.NUMPY
            )

    def test_unimix_alpha_out_of_range_rejected(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="unimix_alpha must be in"):
            gumbel_softmax_sample(
                logits,
                temperature=1.0,
                unimix_alpha=1.5,
                backend=Backend.NUMPY,
            )

    def test_unimix_negative_rejected(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="unimix_alpha must be in"):
            gumbel_softmax_sample(
                logits,
                temperature=1.0,
                unimix_alpha=-0.01,
                backend=Backend.NUMPY,
            )

    def test_unimix_zero_no_mixture_applied(self):
        rng = np.random.default_rng(42)
        logits = rng.standard_normal(size=(2, 5)).astype(np.float32)
        result_no_mix = gumbel_softmax_sample(
            logits,
            temperature=1.0,
            unimix_alpha=0.0,
            backend=Backend.NUMPY,
            seed=42,
        )
        result_with_mix = gumbel_softmax_sample(
            logits,
            temperature=1.0,
            unimix_alpha=0.01,
            backend=Backend.NUMPY,
            seed=42,
        )
        # The two should differ (unimix shifts probabilities)
        assert not np.allclose(result_no_mix, result_with_mix, atol=1e-6)

    def test_canonical_unimix_alpha_constant_pinned(self):
        """Per Wave 3 DreamerV3 math-fidelity audit Hafner 2023 §3."""
        assert CANONICAL_UNIMIX_ALPHA == 0.01

    def test_deterministic_with_seed(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        r1 = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.NUMPY, seed=42
        )
        r2 = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.NUMPY, seed=42
        )
        np.testing.assert_allclose(r1, r2, atol=1e-9)

    def test_different_seeds_different_output(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        r1 = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.NUMPY, seed=42
        )
        r2 = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.NUMPY, seed=43
        )
        assert not np.allclose(r1, r2, atol=1e-3)

    def test_output_all_nonneg(self):
        rng = np.random.default_rng(42)
        logits = rng.standard_normal(size=(8, 16)).astype(np.float32)
        result = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.NUMPY, seed=42
        )
        assert (result >= 0.0).all()

    def test_low_temperature_concentrated_distribution(self):
        """Low temperature → sample concentrates on max-logit category."""
        logits = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
        result = gumbel_softmax_sample(
            logits, temperature=0.01, backend=Backend.NUMPY, seed=42
        )
        # Category 2 (logit=10) should dominate
        assert result[0, 2] > 0.9

    def test_high_temperature_approaches_uniform(self):
        """High temperature → near-uniform distribution."""
        logits = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
        result = gumbel_softmax_sample(
            logits,
            temperature=100.0,
            unimix_alpha=0.0,
            backend=Backend.NUMPY,
            seed=42,
        )
        # Near-uniform: each ~ 1/3
        assert all(abs(p - 1.0 / 3.0) < 0.2 for p in result[0])


# -----------------------------------------------------------------------------
# rgb_to_yuv6 canonical contract tests
# -----------------------------------------------------------------------------


class TestRgbToYuv6:
    def test_output_shape_is_6_channel(self):
        rgb = np.random.RandomState(42).uniform(
            0, 1, size=(2, 3, 4, 4)
        ).astype(np.float32)
        result = rgb_to_yuv6(rgb, backend=Backend.NUMPY)
        assert result.shape == (2, 6, 4, 4)

    def test_rejects_non_3_channel_input(self):
        bad_rgb = np.zeros((1, 4, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="NCHW with 3 channels"):
            rgb_to_yuv6(bad_rgb, backend=Backend.NUMPY)

    def test_rejects_non_nchw_input(self):
        bad_rgb = np.zeros((8, 8, 3), dtype=np.float32)  # HWC
        with pytest.raises(ValueError, match="NCHW with 3 channels"):
            rgb_to_yuv6(bad_rgb, backend=Backend.NUMPY)

    def test_y_channel_grayscale_invariant(self):
        """For grayscale input (R=G=B), Y channel should equal R."""
        gray_value = 0.5
        rgb = np.full((1, 3, 4, 4), gray_value, dtype=np.float32)
        yuv6 = rgb_to_yuv6(rgb, backend=Backend.NUMPY)
        y = yuv6[:, 0]
        # Y = 0.299R + 0.587G + 0.114B ≈ 1.0 * gray_value
        np.testing.assert_allclose(y, gray_value, atol=1e-5)

    def test_uv_centered_at_05_for_gray(self):
        """For grayscale input, U+V channels = 0.5 (centered)."""
        rgb = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
        yuv6 = rgb_to_yuv6(rgb, backend=Backend.NUMPY)
        u = yuv6[:, 1]
        v = yuv6[:, 2]
        np.testing.assert_allclose(u, 0.5, atol=1e-5)
        np.testing.assert_allclose(v, 0.5, atol=1e-5)

    def test_extra_channels_zero_padded(self):
        """Channels 3-5 are zero-padded per canonical contract."""
        rgb = np.random.RandomState(42).uniform(
            0, 1, size=(1, 3, 4, 4)
        ).astype(np.float32)
        yuv6 = rgb_to_yuv6(rgb, backend=Backend.NUMPY)
        for c in (3, 4, 5):
            np.testing.assert_allclose(
                yuv6[:, c], 0.0, atol=1e-9, err_msg=f"channel {c}"
            )


# -----------------------------------------------------------------------------
# assert_cross_backend_parity helper tests
# -----------------------------------------------------------------------------


class TestAssertCrossBackendParity:
    def test_identical_tensors_pass(self):
        x = np.random.RandomState(42).standard_normal(size=(4, 4)).astype(
            np.float32
        )
        assert_cross_backend_parity(x, x, name="identity")

    def test_within_tolerance_pass(self):
        x = np.random.RandomState(42).standard_normal(size=(4, 4)).astype(
            np.float32
        )
        y = x + 1e-7  # within fp32 atol
        assert_cross_backend_parity(x, y, name="within_atol")

    def test_above_tolerance_fail(self):
        x = np.random.RandomState(42).standard_normal(size=(4, 4)).astype(
            np.float32
        )
        y = x + 1e-2
        with pytest.raises(
            AssertionError, match="max abs delta"
        ):
            assert_cross_backend_parity(
                x, y, atol=1e-5, rtol=1e-5, name="above_tol"
            )

    def test_shape_mismatch_fail(self):
        x = np.zeros((4, 4), dtype=np.float32)
        y = np.zeros((4, 8), dtype=np.float32)
        with pytest.raises(AssertionError, match="shape mismatch"):
            assert_cross_backend_parity(x, y, name="bad_shape")

    def test_canonical_atol_constant_pinned(self):
        """Per Slot 16 canonical equation
        ``mlx_pytorch_conv2d_fp64_accumulation_drift_reduction_v1``."""
        assert CANONICAL_CROSS_BACKEND_FP32_ATOL == 1e-5


# -----------------------------------------------------------------------------
# Cross-backend parity tests — only run if multiple backends available
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    "mlx.core" not in __import__("sys").modules
    and not pytest.importorskip("mlx", reason="mlx not installed"),
    reason="MLX not installed",
)
class TestCrossBackendParityMLX:
    """Cross-backend parity (numpy reference ↔ MLX) per Catalog #383."""

    def test_gumbel_softmax_numpy_vs_mlx_parity(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result_numpy = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.NUMPY, seed=42
        )
        result_mlx = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.MLX, seed=42
        )
        assert_cross_backend_parity(
            result_numpy,
            result_mlx,
            atol=CANONICAL_CROSS_BACKEND_FP32_ATOL,
            name="gumbel_softmax_mlx_parity",
        )

    def test_rgb_to_yuv6_numpy_vs_mlx_parity(self):
        rgb = np.random.RandomState(42).uniform(
            0, 1, size=(1, 3, 8, 8)
        ).astype(np.float32)
        result_numpy = rgb_to_yuv6(rgb, backend=Backend.NUMPY)
        result_mlx = rgb_to_yuv6(rgb, backend=Backend.MLX)
        assert_cross_backend_parity(
            result_numpy,
            result_mlx,
            atol=CANONICAL_CROSS_BACKEND_FP32_ATOL,
            name="rgb_to_yuv6_mlx_parity",
        )


# -----------------------------------------------------------------------------
# Tinygrad-availability-conditional parity tests
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    True,  # tinygrad not installed by default; test sister opt-in
    reason="tinygrad not installed (canonical opt-in dep)",
)
class TestCrossBackendParityTinygrad:
    """Cross-backend parity (numpy reference ↔ tinygrad) per Catalog #383."""

    def test_gumbel_softmax_numpy_vs_tinygrad_parity(self):
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result_numpy = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.NUMPY, seed=42
        )
        result_tinygrad = gumbel_softmax_sample(
            logits, temperature=1.0, backend=Backend.TINYGRAD, seed=42
        )
        assert_cross_backend_parity(
            result_numpy,
            result_tinygrad,
            atol=CANONICAL_CROSS_BACKEND_FP32_ATOL,
            name="gumbel_softmax_tinygrad_parity",
        )
