# SPDX-License-Identifier: MIT
"""Canonical numpy reference primitive tests for
``tac.local_acceleration.pr95_hnerv_numpy_reference``.

Per CONSOLIDATE-OP-1 META-CONSOLIDATE-OP-2 extraction wave 2026-05-26
(charter STEP 3 + STEP 4): empirical parity validation for the 7 canonical
numpy reference primitives extracted from G=NIRVANA per axis-3 portability
discipline.

Coverage:

- Re-export parity: ``nirvana_cascading_nerv.numpy_reference`` re-exports
  MUST be IS the same canonical functions (verifies migration preserved
  back-compat import surface).
- Per-primitive correctness: each of 7 primitives produces expected output
  on small fixtures.
- Cross-validation: numpy reference parity vs MLX canonical (≤ 1e-5 drift
  for fp32 deterministic ops).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.local_acceleration.pr95_hnerv_numpy_reference import (
    bilinear_upsample_2x_nhwc,
    conv2d_nhwc,
    kahan_mean,
    linear,
    mean,
    sigmoid,
    sin,
    to_float32,
)


# ---------------------------------------------------------------------------
# Re-export back-compat regression guards
# ---------------------------------------------------------------------------


def test_nirvana_numpy_reference_reexports_match_canonical() -> None:
    """G=NIRVANA's ``numpy_reference`` module MUST re-export the canonical
    primitives (NOT redefine local copies). Verifies CONSOLIDATE-OP-1 META
    extraction wave preserved back-compat import surface for downstream
    callers (NIRVANA tests + future substrate references)."""
    from tac.substrates.nirvana_cascading_nerv import numpy_reference as nirvana_np_ref

    assert nirvana_np_ref.to_float32 is to_float32
    assert nirvana_np_ref.linear is linear
    assert nirvana_np_ref.conv2d_nhwc is conv2d_nhwc
    assert nirvana_np_ref.bilinear_upsample_2x_nhwc is bilinear_upsample_2x_nhwc
    assert nirvana_np_ref.sigmoid is sigmoid
    assert nirvana_np_ref.sin is sin
    assert nirvana_np_ref.mean is mean
    assert nirvana_np_ref.kahan_mean is kahan_mean


# ---------------------------------------------------------------------------
# Primitive 1: to_float32
# ---------------------------------------------------------------------------


def test_to_float32_casts_int_to_fp32() -> None:
    x_int = np.array([1, 2, 3], dtype=np.int64)
    y = to_float32(x_int)
    assert y.dtype == np.float32
    np.testing.assert_array_equal(y, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_to_float32_accepts_python_list() -> None:
    y = to_float32([0.5, 1.5, 2.5])
    assert y.dtype == np.float32
    np.testing.assert_array_equal(y, np.array([0.5, 1.5, 2.5], dtype=np.float32))


# ---------------------------------------------------------------------------
# Primitive 2: linear
# ---------------------------------------------------------------------------


def test_linear_matches_manual_matmul_with_bias() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    weight = np.array([[0.5, 0.5], [1.0, 0.0]], dtype=np.float32)
    bias = np.array([0.1, 0.2], dtype=np.float32)

    y = linear(x, weight, bias)
    expected = x @ weight.T + bias

    np.testing.assert_allclose(y, expected, atol=1e-6)


def test_linear_no_bias() -> None:
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    weight = np.array([[1.0, 0.0]], dtype=np.float32)

    y = linear(x, weight)
    np.testing.assert_allclose(y, np.array([[1.0]]), atol=1e-6)


# ---------------------------------------------------------------------------
# Primitive 3: conv2d_nhwc
# ---------------------------------------------------------------------------


def test_conv2d_nhwc_identity_kernel_no_padding() -> None:
    """1x1 identity kernel returns input unchanged."""
    x = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32)  # (1, 2, 1, 2)
    weight = np.eye(2, dtype=np.float32).reshape(2, 1, 1, 2)  # (2, 1, 1, 2)
    y = conv2d_nhwc(x, weight)
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_conv2d_nhwc_pytorch_parity_3x3_same_padding() -> None:
    """3x3 conv with padding=1 matches PyTorch F.conv2d output within fp32 noise."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2001)
    x_np = rng.standard_normal((1, 4, 4, 2)).astype(np.float32)
    w_np = rng.standard_normal((3, 3, 3, 2)).astype(np.float32)  # (C_out=3, kH=3, kW=3, C_in=2)

    y_ref = conv2d_nhwc(x_np, w_np, padding=1)

    # PyTorch conv2d: NCHW input + (C_out, C_in, kH, kW) weight
    x_torch_nchw = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous()
    w_torch_nchw = torch.from_numpy(w_np).permute(0, 3, 1, 2).contiguous()  # (C_out, C_in, kH, kW)
    with torch.no_grad():
        y_torch_nchw = torch.nn.functional.conv2d(x_torch_nchw, w_torch_nchw, padding=1)
    y_torch_nhwc = y_torch_nchw.permute(0, 2, 3, 1).contiguous().numpy()

    np.testing.assert_allclose(y_ref, y_torch_nhwc, atol=1e-5)


# ---------------------------------------------------------------------------
# Primitive 4: bilinear_upsample_2x_nhwc
# ---------------------------------------------------------------------------


def test_bilinear_upsample_2x_nhwc_pytorch_parity() -> None:
    """numpy 2x bilinear matches PyTorch align_corners=False within fp32 noise."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2002)
    x_np = rng.standard_normal((1, 4, 4, 2)).astype(np.float32)

    y_ref = bilinear_upsample_2x_nhwc(x_np)

    x_torch_nchw = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        y_torch_nchw = torch.nn.functional.interpolate(
            x_torch_nchw,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
    y_torch_nhwc = y_torch_nchw.permute(0, 2, 3, 1).contiguous().numpy()

    assert y_ref.shape == y_torch_nhwc.shape == (1, 8, 8, 2)
    np.testing.assert_allclose(y_ref, y_torch_nhwc, atol=1e-5)


# ---------------------------------------------------------------------------
# Primitive 5: sigmoid
# ---------------------------------------------------------------------------


def test_sigmoid_at_zero_is_half() -> None:
    y = sigmoid(np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(y, np.array([0.5]), atol=1e-7)


def test_sigmoid_numerically_stable_for_large_negative() -> None:
    """Large-negative inputs must not overflow."""
    x = np.array([-1000.0, -100.0, 100.0, 1000.0], dtype=np.float32)
    y = sigmoid(x)
    assert np.isfinite(y).all()
    # Saturated values
    np.testing.assert_allclose(y[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(y[-1], 1.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Primitive 6: sin
# ---------------------------------------------------------------------------


def test_sin_at_zero_and_pi_over_two() -> None:
    y = sin(np.array([0.0, np.pi / 2, np.pi], dtype=np.float32))
    np.testing.assert_allclose(y, np.array([0.0, 1.0, 0.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Primitive 7: mean + kahan_mean
# ---------------------------------------------------------------------------


def test_mean_matches_numpy_mean() -> None:
    x = np.arange(10, dtype=np.float32)
    y = mean(x)
    np.testing.assert_allclose(y, 4.5, atol=1e-6)


def test_kahan_mean_matches_numpy_mean_on_small_array() -> None:
    """On small arrays Kahan summation matches standard mean."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y_kahan = kahan_mean(x)
    y_std = np.mean(x)
    np.testing.assert_allclose(y_kahan, y_std, atol=1e-7)


def test_kahan_mean_more_accurate_than_naive_summation_at_large_n() -> None:
    """At large N with mixed-magnitude values, Kahan summation is more accurate
    than naive cumulative summation. Regression guard for Catalog #962 / slot
    16 engineering-corrections discipline."""
    # Mixed-magnitude: 1.0 + many epsilon-scale additions
    n = 100_000
    x = np.full(n, 1e-7, dtype=np.float32)
    x[0] = 1.0
    y_kahan = kahan_mean(x)
    expected = (1.0 + (n - 1) * 1e-7) / n
    # Kahan should be tight to expected; naive fp32 summation drifts noticeably
    assert abs(float(y_kahan) - expected) < 1e-9


# ---------------------------------------------------------------------------
# numpy reference vs MLX canonical cross-validation
# ---------------------------------------------------------------------------


def test_numpy_bilinear_matches_mlx_canonical_2x_bilinear() -> None:
    """numpy reference 2x bilinear matches MLX canonical helper within fp32 noise.

    Cross-validation that the canonical helper extraction preserved
    behavior across both numpy + MLX implementations.
    """
    mx = pytest.importorskip("mlx.core")
    from tac.local_acceleration.pr95_hnerv_mlx import (
        bilinear_resize2x_align_corners_false_nhwc,
    )

    rng = np.random.default_rng(2003)
    x_np = rng.standard_normal((1, 6, 8, 3)).astype(np.float32)

    y_numpy = bilinear_upsample_2x_nhwc(x_np)

    y_mlx = bilinear_resize2x_align_corners_false_nhwc(mx.array(x_np))
    mx.eval(y_mlx)
    y_mlx_np = np.asarray(y_mlx)

    assert y_numpy.shape == y_mlx_np.shape
    max_abs = float(np.abs(y_numpy - y_mlx_np).max())
    assert max_abs <= 1e-5, (
        f"numpy reference vs MLX canonical 2x bilinear must agree within "
        f"fp32 noise floor; got max_abs={max_abs}"
    )
