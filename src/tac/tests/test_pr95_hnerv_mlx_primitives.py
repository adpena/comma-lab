# SPDX-License-Identifier: MIT
"""Canonical primitive parity tests for ``tac.local_acceleration.pr95_hnerv_mlx``.

Per CONSOLIDATE-OP-1 extraction wave 2026-05-26 (charter STEP 4): empirical
parity validation for the canonical primitives that every Path 3 MLX
substrate now delegates to.

Coverage:

- :func:`pixel_shuffle_2x_nhwc` — byte-stable parity vs PyTorch
  ``nn.PixelShuffle(2)`` (canonical channel-FIRST convention; 0.0 absolute
  drift expected per FIX-WAVE-R1 + FIX-WAVE-R1' empirical anchors).
- :func:`bilinear_resize2x_align_corners_false_nhwc` — closed-form 2x
  bilinear parity vs PyTorch
  ``F.interpolate(scale_factor=2, mode='bilinear', align_corners=False)``
  (0.0 absolute drift expected per A=DreamerV3 + F=Z8 anchors).
- :func:`bilinear_resize_nhwc` — generalized bilinear parity vs PyTorch
  ``F.interpolate(size=(target_h, target_w), mode='bilinear',
  align_corners=False)`` (≤ 1e-5 drift expected per D=Z6 align_corners=False
  formula).
- Identity short-circuit for :func:`bilinear_resize_nhwc` when target shape
  equals input shape.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L9
runtime closure: MLX-trained-PyTorch-inflated model MUST be the same
runtime as the MLX trainer observes at convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    bilinear_resize2x_align_corners_false_nhwc,
    bilinear_resize_nhwc,
    pixel_shuffle_2x_nhwc,
)


# ---------------------------------------------------------------------------
# pixel_shuffle_2x_nhwc — canonical channel-FIRST convention parity tests
# ---------------------------------------------------------------------------


def test_pixel_shuffle_2x_nhwc_mlx_pytorch_parity_byte_stable() -> None:
    """Canonical channel-FIRST pixel shuffle MUST match PyTorch byte-for-byte.

    Per FIX-WAVE-R1 (A=DreamerV3 commit `e1b101888`) + FIX-WAVE-R1'
    (F=Z8 commit `4684dbbab`): channel-FIRST convention
    ``(B, H, W, out_C, 2, 2)`` + transpose ``(0, 1, 4, 2, 5, 3)`` is
    empirically PyTorch-byte-stable (0.0 absolute drift). The forbidden
    channel-LAST convention ``(B, H, W, 2, 2, out_C)`` + transpose
    ``(0, 1, 3, 2, 4, 5)`` produced 2.40 / 3.77 absolute drift respectively.

    This test guards against regression of the canonical convention.
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(42)

    # Shape covering all canonical cases: (B, H, W, out_C * 4)
    for batch, h, w, out_c in [(1, 1, 1, 2), (2, 6, 8, 4), (1, 12, 16, 16)]:
        in_c = out_c * 4
        x_np = rng.standard_normal((batch, h, w, in_c)).astype(np.float32)

        # MLX path: NHWC input -> NHWC output
        x_mx = mx.array(x_np)
        y_mx = pixel_shuffle_2x_nhwc(x_mx)
        mx.eval(y_mx)
        y_mx_np = np.asarray(y_mx)

        # PyTorch reference: NCHW input -> NCHW output -> back to NHWC for compare
        x_torch_nchw = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            y_torch_nchw = torch.nn.functional.pixel_shuffle(x_torch_nchw, upscale_factor=2)
        y_torch_nhwc = y_torch_nchw.permute(0, 2, 3, 1).contiguous().numpy()

        assert y_mx_np.shape == y_torch_nhwc.shape == (batch, h * 2, w * 2, out_c), (
            f"shape mismatch: mlx={y_mx_np.shape} torch={y_torch_nhwc.shape}"
        )
        max_abs = float(np.abs(y_mx_np - y_torch_nhwc).max())
        assert max_abs == 0.0, (
            f"canonical pixel_shuffle_2x_nhwc must be PyTorch-byte-stable "
            f"(0.0 absolute drift); got max_abs={max_abs} on shape "
            f"({batch}, {h}, {w}, {in_c})"
        )


def test_pixel_shuffle_2x_nhwc_refuses_non_2x_upscale_factor() -> None:
    """PR95 + Path 3 substrates use only 2x; helper must refuse other factors."""
    x = mx.zeros((1, 1, 1, 16))
    with pytest.raises(ValueError, match="only 2x pixel shuffle"):
        pixel_shuffle_2x_nhwc(x, upscale_factor=3)


def test_pixel_shuffle_2x_nhwc_refuses_channel_count_not_divisible_by_4() -> None:
    """Channel count must be divisible by 2*2=4 for 2x pixel shuffle."""
    x = mx.zeros((1, 1, 1, 6))  # 6 not divisible by 4
    with pytest.raises(ValueError, match="divisible by 4"):
        pixel_shuffle_2x_nhwc(x)


def test_pixel_shuffle_2x_nhwc_refuses_non_4d_input() -> None:
    """Helper requires NHWC 4D input."""
    x = mx.zeros((1, 1, 4))  # 3D
    with pytest.raises(ValueError, match="NHWC"):
        pixel_shuffle_2x_nhwc(x)


# ---------------------------------------------------------------------------
# bilinear_resize2x_align_corners_false_nhwc — 2x closed-form parity
# ---------------------------------------------------------------------------


def test_bilinear_resize2x_mlx_pytorch_parity_below_fp32_noise_floor() -> None:
    """Canonical 2x bilinear MUST match PyTorch within fp32 noise floor.

    Per FIX-WAVE-R1 (A=DreamerV3) + FIX-WAVE-R1' (F=Z8): the canonical
    closed-form 2x implementation produces 0.0 absolute drift vs PyTorch
    align_corners=False. The forbidden ``mx.repeat`` 2x approximation
    produced 0.99 / 1.51 absolute drift respectively.
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(123)

    for batch, h, w, channels in [(1, 4, 4, 1), (2, 6, 8, 3), (1, 12, 16, 8)]:
        x_np = rng.standard_normal((batch, h, w, channels)).astype(np.float32)

        # MLX path: NHWC -> NHWC
        x_mx = mx.array(x_np)
        y_mx = bilinear_resize2x_align_corners_false_nhwc(x_mx)
        mx.eval(y_mx)
        y_mx_np = np.asarray(y_mx)

        # PyTorch reference: NCHW -> NCHW -> back to NHWC
        x_torch_nchw = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            y_torch_nchw = torch.nn.functional.interpolate(
                x_torch_nchw,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
        y_torch_nhwc = y_torch_nchw.permute(0, 2, 3, 1).contiguous().numpy()

        assert y_mx_np.shape == y_torch_nhwc.shape == (batch, h * 2, w * 2, channels)
        max_abs = float(np.abs(y_mx_np - y_torch_nhwc).max())
        # fp32 noise floor for closed-form 2x bilinear; A+F anchors observe 0.0
        assert max_abs <= 1e-5, (
            f"canonical bilinear_resize2x must be PyTorch-byte-stable "
            f"(≤ 1e-5 drift); got max_abs={max_abs} on shape "
            f"({batch}, {h}, {w}, {channels})"
        )


def test_bilinear_resize2x_refuses_non_4d_input() -> None:
    """Helper requires NHWC 4D input."""
    x = mx.zeros((1, 4, 4))  # 3D
    with pytest.raises(ValueError, match="NHWC"):
        bilinear_resize2x_align_corners_false_nhwc(x)


# ---------------------------------------------------------------------------
# bilinear_resize_nhwc — generalized bilinear parity tests
# ---------------------------------------------------------------------------


def test_bilinear_resize_nhwc_mlx_pytorch_parity_below_fp32_noise_floor() -> None:
    """Generalized bilinear MUST match PyTorch within fp32 noise floor.

    Per D=Z6 sister anchor: align_corners=False formula
    ``src = (dst + 0.5) * (src_size / target_size) - 0.5`` matches
    PyTorch ``F.interpolate(size=..., mode='bilinear', align_corners=False)``
    within fp32 accumulation noise (~1e-5 for typical small tensors).
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(456)

    # Cover canonical resize cases: upscale arbitrary, downscale arbitrary,
    # mixed-axis ratio (height stretched, width preserved)
    cases = [
        # (B, H_in, W_in, C, target_h, target_w)
        (1, 6, 8, 3, 12, 16),      # 2x upscale (parity with 2x specialized)
        (1, 6, 8, 3, 9, 12),       # 1.5x upscale (non-integer ratio)
        (2, 12, 16, 4, 6, 8),      # 2x downscale
        (1, 8, 8, 1, 24, 32),      # 3x H, 4x W (non-uniform)
    ]
    for batch, h_in, w_in, channels, target_h, target_w in cases:
        x_np = rng.standard_normal((batch, h_in, w_in, channels)).astype(np.float32)

        x_mx = mx.array(x_np)
        y_mx = bilinear_resize_nhwc(x_mx, target_h=target_h, target_w=target_w)
        mx.eval(y_mx)
        y_mx_np = np.asarray(y_mx)

        # PyTorch reference
        x_torch_nchw = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            y_torch_nchw = torch.nn.functional.interpolate(
                x_torch_nchw,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        y_torch_nhwc = y_torch_nchw.permute(0, 2, 3, 1).contiguous().numpy()

        assert y_mx_np.shape == y_torch_nhwc.shape == (batch, target_h, target_w, channels)
        max_abs = float(np.abs(y_mx_np - y_torch_nhwc).max())
        # fp32 accumulation noise floor for general bilinear
        assert max_abs <= 1e-5, (
            f"canonical bilinear_resize_nhwc must match PyTorch within fp32 "
            f"noise floor; got max_abs={max_abs} on input ({batch}, {h_in}, "
            f"{w_in}, {channels}) -> ({target_h}, {target_w})"
        )


def test_bilinear_resize_nhwc_identity_short_circuit_when_target_matches_input() -> None:
    """When (target_h, target_w) == (H, W) the helper returns input unchanged."""
    x_np = np.random.default_rng(789).standard_normal((1, 6, 8, 3)).astype(np.float32)
    x_mx = mx.array(x_np)

    y_mx = bilinear_resize_nhwc(x_mx, target_h=6, target_w=8)
    mx.eval(y_mx)

    # Identity short-circuit: output IS the input (zero work)
    assert y_mx.shape == x_mx.shape
    np.testing.assert_array_equal(np.asarray(y_mx), x_np)


def test_bilinear_resize_nhwc_refuses_align_corners_true() -> None:
    """Only align_corners=False is supported (canonical PyTorch default)."""
    x = mx.zeros((1, 4, 4, 3))
    with pytest.raises(ValueError, match="align_corners=False"):
        bilinear_resize_nhwc(x, target_h=8, target_w=8, align_corners=True)


def test_bilinear_resize_nhwc_refuses_non_positive_targets() -> None:
    """target_h and target_w must be positive integers."""
    x = mx.zeros((1, 4, 4, 3))
    with pytest.raises(ValueError, match="positive"):
        bilinear_resize_nhwc(x, target_h=0, target_w=8)
    with pytest.raises(ValueError, match="positive"):
        bilinear_resize_nhwc(x, target_h=8, target_w=-1)


def test_bilinear_resize_nhwc_refuses_non_4d_input() -> None:
    """Helper requires NHWC 4D input."""
    x = mx.zeros((1, 4, 4))  # 3D
    with pytest.raises(ValueError, match="NHWC"):
        bilinear_resize_nhwc(x, target_h=8, target_w=8)


# ---------------------------------------------------------------------------
# Sister-substrate delegation regression guards (CONSOLIDATE-OP-1 invariant)
# ---------------------------------------------------------------------------


def test_dreamer_v3_rssm_substrate_delegates_to_canonical_pixel_shuffle() -> None:
    """A=DreamerV3 local _pixel_shuffle_2x_nhwc MUST produce identical output
    to canonical helper (verifies delegation; sister of the same canonical
    convention)."""
    from tac.substrates.dreamer_v3_rssm.module import _pixel_shuffle_2x_nhwc

    rng = np.random.default_rng(1001)
    x_np = rng.standard_normal((2, 6, 8, 16)).astype(np.float32)
    x_mx = mx.array(x_np)

    y_substrate = _pixel_shuffle_2x_nhwc(x_mx)
    y_canonical = pixel_shuffle_2x_nhwc(x_mx)
    mx.eval(y_substrate)
    mx.eval(y_canonical)

    np.testing.assert_array_equal(np.asarray(y_substrate), np.asarray(y_canonical))


def test_z8_hierarchical_predictive_coding_substrate_delegates_to_canonical_pixel_shuffle() -> None:
    """F=Z8 local _pixel_shuffle_2x_nhwc MUST produce identical output to canonical helper."""
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        _pixel_shuffle_2x_nhwc,
    )

    rng = np.random.default_rng(1002)
    x_np = rng.standard_normal((2, 6, 8, 16)).astype(np.float32)
    x_mx = mx.array(x_np)

    y_substrate = _pixel_shuffle_2x_nhwc(x_mx)
    y_canonical = pixel_shuffle_2x_nhwc(x_mx)
    mx.eval(y_substrate)
    mx.eval(y_canonical)

    np.testing.assert_array_equal(np.asarray(y_substrate), np.asarray(y_canonical))


def test_time_traveler_l5_z6_substrate_delegates_to_canonical_pixel_shuffle() -> None:
    """D=Z6 local _pixel_shuffle_2x_nhwc MUST produce identical output to canonical helper."""
    from tac.substrates.time_traveler_l5_z6.mlx_renderer import (
        _pixel_shuffle_2x_nhwc,
    )

    rng = np.random.default_rng(1003)
    x_np = rng.standard_normal((2, 6, 8, 16)).astype(np.float32)
    x_mx = mx.array(x_np)

    y_substrate = _pixel_shuffle_2x_nhwc(x_mx)
    y_canonical = pixel_shuffle_2x_nhwc(x_mx)
    mx.eval(y_substrate)
    mx.eval(y_canonical)

    np.testing.assert_array_equal(np.asarray(y_substrate), np.asarray(y_canonical))


def test_time_traveler_l5_z6_substrate_delegates_to_canonical_bilinear() -> None:
    """D=Z6 local _bilinear_resize_nhwc MUST produce identical output to canonical helper."""
    from tac.substrates.time_traveler_l5_z6.mlx_renderer import (
        _bilinear_resize_nhwc,
    )

    rng = np.random.default_rng(1004)
    x_np = rng.standard_normal((2, 6, 8, 4)).astype(np.float32)
    x_mx = mx.array(x_np)

    y_substrate = _bilinear_resize_nhwc(x_mx, target_h=12, target_w=16)
    y_canonical = bilinear_resize_nhwc(x_mx, target_h=12, target_w=16)
    mx.eval(y_substrate)
    mx.eval(y_canonical)

    np.testing.assert_array_equal(np.asarray(y_substrate), np.asarray(y_canonical))


def test_z8_hierarchical_predictive_coding_substrate_delegates_to_canonical_bilinear_2x() -> None:
    """F=Z8 local _bilinear_resize_2x_nhwc MUST produce identical output to canonical 2x helper."""
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        _bilinear_resize_2x_nhwc,
    )

    rng = np.random.default_rng(1005)
    x_np = rng.standard_normal((2, 6, 8, 4)).astype(np.float32)
    x_mx = mx.array(x_np)

    y_substrate = _bilinear_resize_2x_nhwc(x_mx)
    y_canonical = bilinear_resize2x_align_corners_false_nhwc(x_mx)
    mx.eval(y_substrate)
    mx.eval(y_canonical)

    np.testing.assert_array_equal(np.asarray(y_substrate), np.asarray(y_canonical))


def test_dreamer_v3_rssm_substrate_delegates_to_canonical_bilinear_2x() -> None:
    """A=DreamerV3 local _bilinear_resize_2x_nhwc MUST produce identical output to canonical 2x helper."""
    from tac.substrates.dreamer_v3_rssm.module import _bilinear_resize_2x_nhwc

    rng = np.random.default_rng(1006)
    x_np = rng.standard_normal((2, 6, 8, 4)).astype(np.float32)
    x_mx = mx.array(x_np)

    y_substrate = _bilinear_resize_2x_nhwc(x_mx)
    y_canonical = bilinear_resize2x_align_corners_false_nhwc(x_mx)
    mx.eval(y_substrate)
    mx.eval(y_canonical)

    np.testing.assert_array_equal(np.asarray(y_substrate), np.asarray(y_canonical))
