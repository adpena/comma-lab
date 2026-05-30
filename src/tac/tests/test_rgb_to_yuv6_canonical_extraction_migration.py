# SPDX-License-Identifier: MIT
"""Byte-stable parity tests for the rgb_to_yuv6 canonical extraction migration.

Sister of the 2026-05-30 MLX canonicalization audit op-routable #2:
``tac.constrained_gen.rgb_to_yuv6`` / ``tac.saliency.rgb_to_yuv6`` /
``tac.local_acceleration.pr95_hnerv_mlx_training.rgb_to_yuv6_mlx`` all
now route through
:func:`tac.framework_agnostic.canonical_kernels.rgb_to_yuv6`.

``tac.composition.yuv6_chroma_subsampled_perturbation_operator.rgb_to_yuv6_numpy``
is a PRINCIPLED FORK (float64 precision contract; canonical helper is
float32-native; ~2e-5 absolute discrepancy above fp32 epsilon) and
carries a same-line ``CATALOG_383_PRINCIPLED_FORK_OK`` waiver — verified
here for unchanged byte-identical behavior to its pre-migration form.

Tests cover:
    * NCHW byte-identical output (``constrained_gen``)
    * HWC byte-identical output (``saliency``)
    * NHWC byte-identical output (``pr95_hnerv_mlx_training``)
    * Leading-dimension preservation (3D / 4D / 5D)
    * Gradient propagation (PyTorch sisters)
    * Composition operator preserves pre-migration float64 contract
    * Canonical helper output channel order matches sister expectations
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tac.framework_agnostic.canonical_kernels import Backend, rgb_to_yuv6 as canonical_rgb_to_yuv6


# -----------------------------------------------------------------------------
# Sister 1: tac.constrained_gen.rgb_to_yuv6 (PyTorch NCHW, leading dims)
# -----------------------------------------------------------------------------


def _legacy_constrained_gen_rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Reconstructed pre-migration constrained_gen impl for byte-stable parity."""
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, : 2 * H2, : 2 * W2]
    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]
    kYR, kYG, kYB = 0.299, 0.587, 0.114
    Y = (R * kYR + G * kYG + B * kYB).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (
        U[..., 0::2, 0::2]
        + U[..., 1::2, 0::2]
        + U[..., 0::2, 1::2]
        + U[..., 1::2, 1::2]
    ) * 0.25
    V_sub = (
        V[..., 0::2, 0::2]
        + V[..., 1::2, 0::2]
        + V[..., 0::2, 1::2]
        + V[..., 1::2, 1::2]
    ) * 0.25
    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


class TestConstrainedGenRgbToYuv6BytStableMigration:
    """tac.constrained_gen.rgb_to_yuv6 migration parity tests."""

    def test_4d_nchw_byte_identical(self) -> None:
        from tac.constrained_gen import rgb_to_yuv6 as new_cg

        rng = np.random.default_rng(42)
        rgb = torch.from_numpy(rng.uniform(0, 255, size=(2, 3, 16, 16)).astype(np.float32))
        new_out = new_cg(rgb)
        legacy_out = _legacy_constrained_gen_rgb_to_yuv6(rgb)
        assert new_out.shape == legacy_out.shape == (2, 6, 8, 8)
        assert (new_out - legacy_out).abs().max().item() == 0.0

    def test_5d_leading_dim_byte_identical(self) -> None:
        from tac.constrained_gen import rgb_to_yuv6 as new_cg

        rng = np.random.default_rng(42)
        rgb = torch.from_numpy(
            rng.uniform(0, 255, size=(1, 2, 3, 384, 512)).astype(np.float32)
        )
        new_out = new_cg(rgb)
        legacy_out = _legacy_constrained_gen_rgb_to_yuv6(rgb)
        assert new_out.shape == legacy_out.shape == (1, 2, 6, 192, 256)
        assert (new_out - legacy_out).abs().max().item() == 0.0

    def test_6d_deep_leading_dim_byte_identical(self) -> None:
        from tac.constrained_gen import rgb_to_yuv6 as new_cg

        rng = np.random.default_rng(42)
        rgb = torch.from_numpy(
            rng.uniform(0, 255, size=(2, 3, 4, 3, 8, 8)).astype(np.float32)
        )
        new_out = new_cg(rgb)
        legacy_out = _legacy_constrained_gen_rgb_to_yuv6(rgb)
        assert new_out.shape == legacy_out.shape == (2, 3, 4, 6, 4, 4)
        assert (new_out - legacy_out).abs().max().item() == 0.0

    def test_gradient_propagation(self) -> None:
        from tac.constrained_gen import rgb_to_yuv6 as new_cg

        rng = np.random.default_rng(42)
        rgb = torch.from_numpy(rng.uniform(0, 255, size=(1, 3, 8, 8)).astype(np.float32))
        rgb.requires_grad_(True)
        out = new_cg(rgb)
        out.sum().backward()
        assert rgb.grad is not None
        assert torch.isfinite(rgb.grad).all()
        assert rgb.grad.abs().sum().item() > 0.0

    def test_rejects_non_3_channel_input(self) -> None:
        from tac.constrained_gen import rgb_to_yuv6 as new_cg

        bad = torch.zeros((1, 4, 8, 8))
        with pytest.raises(ValueError, match="expects NCHW with 3 channels"):
            new_cg(bad)


# -----------------------------------------------------------------------------
# Sister 2: tac.saliency.rgb_to_yuv6 (PyTorch HWC, leading dims)
# -----------------------------------------------------------------------------


def _legacy_saliency_rgb_to_yuv6(frames: torch.Tensor) -> torch.Tensor:
    """Reconstructed pre-migration saliency impl for byte-stable parity."""
    kYR, kYG, kYB = 0.299, 0.587, 0.114
    R, G, B = frames[..., 0], frames[..., 1], frames[..., 2]
    Y = (R * kYR + G * kYG + B * kYB).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (
        U[..., 0::2, 0::2]
        + U[..., 1::2, 0::2]
        + U[..., 0::2, 1::2]
        + U[..., 1::2, 1::2]
    ) * 0.25
    V_sub = (
        V[..., 0::2, 0::2]
        + V[..., 1::2, 0::2]
        + V[..., 0::2, 1::2]
        + V[..., 1::2, 1::2]
    ) * 0.25
    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


class TestSaliencyRgbToYuv6ByteStableMigration:
    """tac.saliency.rgb_to_yuv6 migration parity tests."""

    def test_3d_hwc_byte_identical(self) -> None:
        from tac.saliency import rgb_to_yuv6 as new_sal

        rng = np.random.default_rng(42)
        rgb = torch.from_numpy(rng.uniform(0, 255, size=(16, 16, 3)).astype(np.float32))
        new_out = new_sal(rgb)
        legacy_out = _legacy_saliency_rgb_to_yuv6(rgb)
        assert new_out.shape == legacy_out.shape == (6, 8, 8)
        assert (new_out - legacy_out).abs().max().item() == 0.0

    def test_4d_hwc_byte_identical(self) -> None:
        from tac.saliency import rgb_to_yuv6 as new_sal

        rng = np.random.default_rng(42)
        rgb = torch.from_numpy(rng.uniform(0, 255, size=(2, 16, 16, 3)).astype(np.float32))
        new_out = new_sal(rgb)
        legacy_out = _legacy_saliency_rgb_to_yuv6(rgb)
        assert new_out.shape == legacy_out.shape == (2, 6, 8, 8)
        assert (new_out - legacy_out).abs().max().item() == 0.0

    def test_5d_hwc_byte_identical_canonical_saliency_pipeline_path(self) -> None:
        """Canonical saliency pipeline path: ``(B, S, H, W, 3)`` -> ``(B, S, 6, H2, W2)``."""
        from tac.saliency import rgb_to_yuv6 as new_sal

        rng = np.random.default_rng(42)
        # 384x512 is the canonical scorer input resolution
        rgb = torch.from_numpy(
            rng.uniform(0, 255, size=(1, 2, 384, 512, 3)).astype(np.float32)
        )
        new_out = new_sal(rgb)
        legacy_out = _legacy_saliency_rgb_to_yuv6(rgb)
        assert new_out.shape == legacy_out.shape == (1, 2, 6, 192, 256)
        assert (new_out - legacy_out).abs().max().item() == 0.0

    def test_gradient_propagation_hwc(self) -> None:
        from tac.saliency import rgb_to_yuv6 as new_sal

        rng = np.random.default_rng(42)
        rgb = torch.from_numpy(rng.uniform(0, 255, size=(1, 8, 8, 3)).astype(np.float32))
        rgb.requires_grad_(True)
        out = new_sal(rgb)
        out.sum().backward()
        assert rgb.grad is not None
        assert torch.isfinite(rgb.grad).all()
        assert rgb.grad.abs().sum().item() > 0.0

    def test_rejects_non_hwc_input(self) -> None:
        from tac.saliency import rgb_to_yuv6 as new_sal

        bad = torch.zeros((8, 8, 4))  # 4 channels at last dim
        with pytest.raises(ValueError, match="expects HWC with 3 channels"):
            new_sal(bad)


# -----------------------------------------------------------------------------
# Sister 3: tac.local_acceleration.pr95_hnerv_mlx_training.rgb_to_yuv6_mlx
# -----------------------------------------------------------------------------


_mlx_core = pytest.importorskip("mlx.core")


def _legacy_pr95_rgb_to_yuv6_mlx(rgb_nhwc):
    """Reconstructed pre-migration pr95_hnerv_mlx_training impl for byte-stable parity."""
    mx = _mlx_core
    height = int(rgb_nhwc.shape[-3])
    width = int(rgb_nhwc.shape[-2])
    h2 = height // 2
    w2 = width // 2
    rgb = rgb_nhwc[..., : 2 * h2, : 2 * w2, :]
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    y = mx.clip(red * 0.299 + green * 0.587 + blue * 0.114, 0.0, 255.0)
    u = mx.clip((blue - y) / 1.772 + 128.0, 0.0, 255.0)
    v = mx.clip((red - y) / 1.402 + 128.0, 0.0, 255.0)
    u_sub = (
        u[..., 0::2, 0::2]
        + u[..., 1::2, 0::2]
        + u[..., 0::2, 1::2]
        + u[..., 1::2, 1::2]
    ) * 0.25
    v_sub = (
        v[..., 0::2, 0::2]
        + v[..., 1::2, 0::2]
        + v[..., 0::2, 1::2]
        + v[..., 1::2, 1::2]
    ) * 0.25
    return mx.stack(
        [
            y[..., 0::2, 0::2],
            y[..., 1::2, 0::2],
            y[..., 0::2, 1::2],
            y[..., 1::2, 1::2],
            u_sub,
            v_sub,
        ],
        axis=-1,
    )


class TestPr95HnervMlxRgbToYuv6ByteStableMigration:
    """tac.local_acceleration.pr95_hnerv_mlx_training.rgb_to_yuv6_mlx migration parity tests."""

    def test_4d_nhwc_byte_identical(self) -> None:
        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx as new_mlx

        rng = np.random.default_rng(42)
        rgb = _mlx_core.array(
            rng.uniform(0, 255, size=(2, 16, 16, 3)).astype(np.float32)
        )
        new_out = np.array(new_mlx(rgb))
        legacy_out = np.array(_legacy_pr95_rgb_to_yuv6_mlx(rgb))
        assert new_out.shape == legacy_out.shape == (2, 8, 8, 6)
        assert float(np.abs(new_out - legacy_out).max()) == 0.0

    def test_odd_crop_byte_identical_per_existing_pr95_test_anchor(self) -> None:
        """Replicates the canonical test_pr95_hnerv_mlx_training odd-crop anchor."""
        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx as new_mlx

        nhwc = np.zeros((1, 17, 19, 3), dtype=np.float32)
        nhwc[:, :, :, 0] = 255.0
        nhwc[:, 0::2, :, 1] = 128.0
        nhwc[:, :, 0::3, 2] = 64.0
        nhwc[:, -1, -1, :] = np.array([17.0, 29.0, 251.0], dtype=np.float32)
        rgb_mx = _mlx_core.array(nhwc)
        new_out = np.array(new_mlx(rgb_mx))
        legacy_out = np.array(_legacy_pr95_rgb_to_yuv6_mlx(rgb_mx))
        assert new_out.shape == legacy_out.shape == (1, 8, 9, 6)
        assert float(np.abs(new_out - legacy_out).max()) == 0.0

    def test_5d_nhwc_leading_dim_byte_identical(self) -> None:
        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx as new_mlx

        rng = np.random.default_rng(42)
        rgb = _mlx_core.array(
            rng.uniform(0, 255, size=(1, 2, 8, 8, 3)).astype(np.float32)
        )
        new_out = np.array(new_mlx(rgb))
        legacy_out = np.array(_legacy_pr95_rgb_to_yuv6_mlx(rgb))
        assert new_out.shape == legacy_out.shape == (1, 2, 4, 4, 6)
        assert float(np.abs(new_out - legacy_out).max()) == 0.0

    def test_rejects_too_small_spatial_dims(self) -> None:
        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx as new_mlx

        bad = _mlx_core.array(np.zeros((1, 1, 1, 3), dtype=np.float32))
        with pytest.raises(Exception, match="requires spatial dims at least 2x2"):
            new_mlx(bad)


# -----------------------------------------------------------------------------
# Sister 4: PRINCIPLED FORK — tac.composition.yuv6_chroma_subsampled_perturbation_operator
# -----------------------------------------------------------------------------


def _legacy_composition_rgb_to_yuv6_numpy(rgb_hwc: np.ndarray) -> np.ndarray:
    """Reconstructed pre-migration composition impl for byte-stable parity.

    The PRINCIPLED FORK rationale (float64 precision contract; canonical
    helper is float32-native) is preserved: this verifies the post-
    migration code is byte-identical to its pre-migration form (NOT
    canonical math).
    """
    BT601_KYR, BT601_KYG, BT601_KYB = 0.299, 0.587, 0.114
    BT601_U_DIVISOR, BT601_V_DIVISOR = 1.772, 1.402
    H, W, _ = rgb_hwc.shape
    H2, W2 = H // 2, W // 2
    rgb = rgb_hwc[: 2 * H2, : 2 * W2, :].astype(np.float64)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    Y = np.clip(R * BT601_KYR + G * BT601_KYG + B * BT601_KYB, 0.0, 255.0)
    U = np.clip((B - Y) / BT601_U_DIVISOR + 128.0, 0.0, 255.0)
    V = np.clip((R - Y) / BT601_V_DIVISOR + 128.0, 0.0, 255.0)
    Y00 = Y[0::2, 0::2]
    Y10 = Y[1::2, 0::2]
    Y01 = Y[0::2, 1::2]
    Y11 = Y[1::2, 1::2]
    U_sub = (
        U[0::2, 0::2] + U[1::2, 0::2] + U[0::2, 1::2] + U[1::2, 1::2]
    ) * 0.25
    V_sub = (
        V[0::2, 0::2] + V[1::2, 0::2] + V[0::2, 1::2] + V[1::2, 1::2]
    ) * 0.25
    return np.stack([Y00, Y10, Y01, Y11, U_sub, V_sub], axis=0)


class TestCompositionPrincipledForkPreservation:
    """tac.composition principled-fork preservation parity tests."""

    def test_even_dims_byte_identical_pre_migration(self) -> None:
        from tac.composition.yuv6_chroma_subsampled_perturbation_operator.operator import (
            rgb_to_yuv6_numpy as new_comp,
        )

        rng = np.random.default_rng(42)
        rgb = rng.uniform(0, 255, size=(16, 16, 3)).astype(np.float32)
        new_out = new_comp(rgb)
        legacy_out = _legacy_composition_rgb_to_yuv6_numpy(rgb)
        assert new_out.shape == legacy_out.shape == (6, 8, 8)
        assert new_out.dtype == np.float64
        assert legacy_out.dtype == np.float64
        assert float(np.abs(new_out - legacy_out).max()) == 0.0

    def test_odd_dims_byte_identical_pre_migration(self) -> None:
        from tac.composition.yuv6_chroma_subsampled_perturbation_operator.operator import (
            rgb_to_yuv6_numpy as new_comp,
        )

        rng = np.random.default_rng(42)
        rgb = rng.uniform(0, 255, size=(17, 19, 3)).astype(np.float32)
        new_out = new_comp(rgb)
        legacy_out = _legacy_composition_rgb_to_yuv6_numpy(rgb)
        assert new_out.shape == legacy_out.shape == (6, 8, 9)
        assert float(np.abs(new_out - legacy_out).max()) == 0.0

    def test_float64_dtype_preserved(self) -> None:
        from tac.composition.yuv6_chroma_subsampled_perturbation_operator.operator import (
            rgb_to_yuv6_numpy as new_comp,
        )

        rgb_f32 = np.zeros((8, 8, 3), dtype=np.float32)
        out = new_comp(rgb_f32)
        assert out.dtype == np.float64, (
            "PRINCIPLED FORK precision contract requires float64 output; "
            "canonical helper is float32-native — confirms NOT delegating."
        )

    def test_canonical_helper_diverges_above_fp32_epsilon_documents_principled_fork(self) -> None:
        """Documents the empirical PRINCIPLED FORK rationale: canonical
        helper's float32-native math diverges from pure float64 by ~2e-5
        (above fp32 epsilon ~3e-7 per element). This test pins the
        empirical anchor that motivated the principled fork."""
        rng = np.random.default_rng(42)
        rgb_f32 = rng.uniform(0, 255, size=(16, 16, 3)).astype(np.float32)
        # Canonical numpy (float32-native)
        rgb_nchw = np.transpose(rgb_f32, (2, 0, 1))[np.newaxis, ...]
        canonical_out = canonical_rgb_to_yuv6(
            rgb_nchw, backend=Backend.NUMPY, value_range=255.0
        )[0].astype(np.float64)
        # Composition impl (pure float64)
        legacy_out = _legacy_composition_rgb_to_yuv6_numpy(rgb_f32)
        divergence = float(np.abs(canonical_out - legacy_out).max())
        # Empirically observed ~2.13e-5 on this fixture; the canonical
        # helper is float32-native and the composition operator's
        # downstream perturbation math depends on float64 precision.
        assert divergence > 1e-6, (
            f"Expected divergence > fp32 epsilon (~3e-7); got {divergence:.2e}. "
            "If this passes the PRINCIPLED FORK rationale may need re-evaluation."
        )
        assert divergence < 1e-3, (
            f"Divergence {divergence:.2e} larger than expected upper bound 1e-3; "
            "audit the canonical helper math against the composition operator."
        )


# -----------------------------------------------------------------------------
# Canonical helper cross-sister byte-stable verification
# -----------------------------------------------------------------------------


class TestCanonicalHelperConsumesByAllMigratedSisters:
    """Verify canonical helper is byte-stable across all 3 delegating sisters
    on their respective shared NCHW contract subsets."""

    def test_constrained_gen_delegates_to_canonical_byte_identical(self) -> None:
        from tac.constrained_gen import rgb_to_yuv6 as cg

        rng = np.random.default_rng(42)
        rgb_nchw = torch.from_numpy(rng.uniform(0, 255, size=(1, 3, 8, 8)).astype(np.float32))
        canonical_out = canonical_rgb_to_yuv6(rgb_nchw, backend=Backend.PYTORCH, value_range=255.0)
        cg_out = cg(rgb_nchw)
        assert (canonical_out - cg_out).abs().max().item() == 0.0

    def test_saliency_delegates_to_canonical_byte_identical_at_hwc_via_layout_adapter(self) -> None:
        from tac.saliency import rgb_to_yuv6 as sal

        rng = np.random.default_rng(42)
        rgb_nchw_np = rng.uniform(0, 255, size=(1, 3, 8, 8)).astype(np.float32)
        canonical_out = canonical_rgb_to_yuv6(
            torch.from_numpy(rgb_nchw_np), backend=Backend.PYTORCH, value_range=255.0
        )
        # saliency expects HWC; transpose and run
        rgb_hwc = torch.from_numpy(rgb_nchw_np.transpose(0, 2, 3, 1))
        sal_out = sal(rgb_hwc)
        assert (canonical_out - sal_out).abs().max().item() == 0.0

    def test_pr95_hnerv_mlx_delegates_to_canonical_byte_identical_via_nhwc_layout_adapter(self) -> None:
        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

        rng = np.random.default_rng(42)
        rgb_nchw_np = rng.uniform(0, 255, size=(1, 3, 8, 8)).astype(np.float32)
        canonical_out = canonical_rgb_to_yuv6(
            rgb_nchw_np, backend=Backend.NUMPY, value_range=255.0
        )
        rgb_nhwc_mx = _mlx_core.array(rgb_nchw_np.transpose(0, 2, 3, 1))
        mlx_out_nhwc = rgb_to_yuv6_mlx(rgb_nhwc_mx)
        mlx_out_nchw = np.array(mlx_out_nhwc).transpose(0, 3, 1, 2)
        assert float(np.abs(canonical_out - mlx_out_nchw).max()) == 0.0
