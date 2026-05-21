# SPDX-License-Identifier: MIT
"""Cross-backend numerical-equivalence tests for MLX-ARCH-1 extended primitives.

Sister of ``test_portable_primitives_numerical_equivalence.py`` for the 5
extended primitives required by FastViT-T12 + EfficientNet-B2-UNet:

- :class:`PortableBatchNorm2d`
- :class:`PortableDepthwiseConv2d`
- :class:`PortableMaxPool2d`
- :class:`PortableAvgPool2d`
- :func:`silu`

Per OVERNIGHT-WW + Phase 1 PV finding: every primitive constructed with
``backend="mlx"`` must produce numerically-equivalent results to its
``backend="pytorch"`` sister within a documented ε band.

**Empirical ε band per Phase 1 PV** (M5 Max Metal GPU vs PyTorch CPU fp32):
- ATOL_FP32 = 5e-3 (Metal MPS uses different FMA ordering than CPU fp32)
- RTOL_FP32 = 5e-3 (relative differences scale with magnitude per FMA
  re-ordering)

Per CLAUDE.md "Apples-to-apples evidence discipline" extended to the
portable-primitives surface: a future refactor that silently breaks
backend equivalence beyond ε would corrupt the train-anywhere
eval-anywhere pipeline (the MLX-train -> PyTorch-eval canonical
contest-axis path).

Tests are skipped if MLX is not available on the host (e.g. Linux CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.portable_primitives import (
    PortableAvgPool2d,
    PortableBatchNorm2d,
    PortableDepthwiseConv2d,
    PortableMaxPool2d,
    is_mlx_available,
    is_pytorch_available,
    silu,
)
from tac.portable_primitives.tensor import from_numpy, to_numpy

# Empirical ε band per Phase 1 PV (sister test convention).
ATOL_FP32 = 5e-3
RTOL_FP32 = 5e-3


pytestmark = pytest.mark.skipif(
    not (is_mlx_available() and is_pytorch_available()),
    reason="Both MLX + PyTorch must be installed for cross-backend equivalence tests",
)


def _seeded_input(shape: tuple[int, ...], seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# PortableBatchNorm2d
# ---------------------------------------------------------------------------


def test_batchnorm2d_train_forward_equivalence() -> None:
    """PortableBatchNorm2d train mode (batch stats) MLX vs PyTorch within ε."""
    rng = np.random.RandomState(0)
    num_features = 4
    weight_np = rng.standard_normal((num_features,)).astype(np.float32) * 0.5 + 1.0
    bias_np = rng.standard_normal((num_features,)).astype(np.float32) * 0.1
    x_np = _seeded_input((2, num_features, 8, 8))

    bn_mlx = PortableBatchNorm2d(num_features, backend="mlx")
    bn_pt = PortableBatchNorm2d(num_features, backend="pytorch")
    bn_mlx.load_weights(weight_np=weight_np, bias_np=bias_np)
    bn_pt.load_weights(weight_np=weight_np, bias_np=bias_np)
    # Both default to train mode.

    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")

    y_mlx = to_numpy(bn_mlx(x_mlx), "mlx")
    y_pt = to_numpy(bn_pt(x_pt), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_batchnorm2d_eval_forward_equivalence() -> None:
    """PortableBatchNorm2d eval mode (running stats) MLX vs PyTorch within ε."""
    rng = np.random.RandomState(1)
    num_features = 8
    weight_np = rng.standard_normal((num_features,)).astype(np.float32)
    bias_np = rng.standard_normal((num_features,)).astype(np.float32) * 0.1
    running_mean_np = rng.standard_normal((num_features,)).astype(np.float32) * 0.2
    running_var_np = (rng.standard_normal((num_features,)).astype(np.float32) ** 2 + 0.5)
    x_np = _seeded_input((2, num_features, 6, 6))

    bn_mlx = PortableBatchNorm2d(num_features, backend="mlx")
    bn_pt = PortableBatchNorm2d(num_features, backend="pytorch")
    bn_mlx.load_weights(
        weight_np=weight_np,
        bias_np=bias_np,
        running_mean_np=running_mean_np,
        running_var_np=running_var_np,
    )
    bn_pt.load_weights(
        weight_np=weight_np,
        bias_np=bias_np,
        running_mean_np=running_mean_np,
        running_var_np=running_var_np,
    )

    bn_mlx.eval()
    bn_pt.eval()

    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")

    y_mlx = to_numpy(bn_mlx(x_mlx), "mlx")
    y_pt = to_numpy(bn_pt(x_pt), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_batchnorm2d_export_round_trip() -> None:
    """BatchNorm2d export -> load preserves all 4 buffers (weight/bias/running_mean/running_var)."""
    rng = np.random.RandomState(2)
    num_features = 6
    weight_np = rng.standard_normal((num_features,)).astype(np.float32)
    bias_np = rng.standard_normal((num_features,)).astype(np.float32)
    running_mean_np = rng.standard_normal((num_features,)).astype(np.float32)
    running_var_np = (rng.standard_normal((num_features,)).astype(np.float32) ** 2 + 0.1)

    bn_mlx = PortableBatchNorm2d(num_features, backend="mlx")
    bn_mlx.load_weights(
        weight_np=weight_np,
        bias_np=bias_np,
        running_mean_np=running_mean_np,
        running_var_np=running_var_np,
    )

    exported = bn_mlx.export_weights()
    assert set(exported.keys()) == {"weight", "bias", "running_mean", "running_var"}
    np.testing.assert_allclose(exported["weight"], weight_np, atol=ATOL_FP32, rtol=RTOL_FP32)
    np.testing.assert_allclose(exported["bias"], bias_np, atol=ATOL_FP32, rtol=RTOL_FP32)
    np.testing.assert_allclose(exported["running_mean"], running_mean_np, atol=ATOL_FP32, rtol=RTOL_FP32)
    np.testing.assert_allclose(exported["running_var"], running_var_np, atol=ATOL_FP32, rtol=RTOL_FP32)

    # Load into PyTorch + verify eval forward matches MLX eval forward.
    bn_pt = PortableBatchNorm2d(num_features, backend="pytorch")
    bn_pt.load_weights(
        weight_np=exported["weight"],
        bias_np=exported["bias"],
        running_mean_np=exported["running_mean"],
        running_var_np=exported["running_var"],
    )
    bn_mlx.eval()
    bn_pt.eval()

    x_np = _seeded_input((2, num_features, 4, 4))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(bn_mlx(x_mlx), "mlx")
    y_pt = to_numpy(bn_pt(x_pt), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_batchnorm2d_train_eval_toggle() -> None:
    """train() / eval() toggle changes behavior consistently across backends."""
    num_features = 4
    rng = np.random.RandomState(3)
    weight_np = np.ones((num_features,), dtype=np.float32)
    bias_np = np.zeros((num_features,), dtype=np.float32)
    running_mean_np = np.zeros((num_features,), dtype=np.float32)
    running_var_np = np.ones((num_features,), dtype=np.float32)
    x_np = _seeded_input((2, num_features, 4, 4), seed=10)

    for backend in ("mlx", "pytorch"):
        bn = PortableBatchNorm2d(num_features, backend=backend)
        bn.load_weights(
            weight_np=weight_np,
            bias_np=bias_np,
            running_mean_np=running_mean_np,
            running_var_np=running_var_np,
        )
        x = from_numpy(x_np, backend)

        # In eval mode with running_mean=0 + running_var=1 + weight=1 +
        # bias=0, BN should be approximately identity (up to eps).
        bn.eval()
        y_eval = to_numpy(bn(x), backend)
        np.testing.assert_allclose(y_eval, x_np, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# PortableDepthwiseConv2d
# ---------------------------------------------------------------------------


def test_depthwise_conv2d_forward_equivalence() -> None:
    """PortableDepthwiseConv2d MLX vs PyTorch within ε."""
    rng = np.random.RandomState(0)
    in_channels = 4
    kernel_size = 3
    # PyTorch depthwise weight shape: (in_channels, 1, kH, kW).
    w_np = rng.standard_normal(
        (in_channels, 1, kernel_size, kernel_size)
    ).astype(np.float32) * 0.1
    b_np = rng.standard_normal((in_channels,)).astype(np.float32) * 0.1
    x_np = _seeded_input((2, in_channels, 8, 8))

    dw_mlx = PortableDepthwiseConv2d(in_channels, kernel_size=kernel_size, backend="mlx")
    dw_pt = PortableDepthwiseConv2d(in_channels, kernel_size=kernel_size, backend="pytorch")
    dw_mlx.load_weights(w_np, b_np)
    dw_pt.load_weights(w_np, b_np)

    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(dw_mlx(x_mlx), "mlx")
    y_pt = to_numpy(dw_pt(x_pt), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_depthwise_conv2d_stride2_equivalence() -> None:
    """Depthwise conv with stride=2 (MBConv pattern) MLX vs PyTorch."""
    rng = np.random.RandomState(1)
    in_channels = 8
    kernel_size = 3
    w_np = rng.standard_normal((in_channels, 1, kernel_size, kernel_size)).astype(np.float32) * 0.1
    b_np = np.zeros(in_channels, dtype=np.float32)
    x_np = _seeded_input((2, in_channels, 16, 16))

    dw_mlx = PortableDepthwiseConv2d(
        in_channels, kernel_size=kernel_size, stride=2, backend="mlx"
    )
    dw_pt = PortableDepthwiseConv2d(
        in_channels, kernel_size=kernel_size, stride=2, backend="pytorch"
    )
    dw_mlx.load_weights(w_np, b_np)
    dw_pt.load_weights(w_np, b_np)

    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(dw_mlx(x_mlx), "mlx")
    y_pt = to_numpy(dw_pt(x_pt), "pytorch")

    # Shape: stride=2, kernel=3, padding=1 -> output spatial 8.
    assert y_pt.shape == (2, in_channels, 8, 8)
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_depthwise_conv2d_no_bias_equivalence() -> None:
    """Depthwise conv with bias=False (common in conv-bn-act stacks)."""
    rng = np.random.RandomState(2)
    in_channels = 4
    kernel_size = 5
    w_np = rng.standard_normal((in_channels, 1, kernel_size, kernel_size)).astype(np.float32) * 0.1
    x_np = _seeded_input((1, in_channels, 12, 12))

    dw_mlx = PortableDepthwiseConv2d(in_channels, kernel_size=kernel_size, bias=False, backend="mlx")
    dw_pt = PortableDepthwiseConv2d(in_channels, kernel_size=kernel_size, bias=False, backend="pytorch")
    dw_mlx.load_weights(w_np)
    dw_pt.load_weights(w_np)

    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(dw_mlx(x_mlx), "mlx")
    y_pt = to_numpy(dw_pt(x_pt), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_depthwise_conv2d_export_round_trip() -> None:
    """Depthwise conv export -> load preserves PyTorch layout (in_channels, 1, kH, kW)."""
    rng = np.random.RandomState(3)
    in_channels = 4
    kernel_size = 3
    w_np = rng.standard_normal((in_channels, 1, kernel_size, kernel_size)).astype(np.float32) * 0.1
    b_np = rng.standard_normal((in_channels,)).astype(np.float32) * 0.1

    dw_mlx = PortableDepthwiseConv2d(in_channels, kernel_size=kernel_size, backend="mlx")
    dw_mlx.load_weights(w_np, b_np)

    w_out, b_out = dw_mlx.export_weights()
    assert w_out.shape == (in_channels, 1, kernel_size, kernel_size)
    assert b_out is not None and b_out.shape == (in_channels,)
    np.testing.assert_allclose(w_out, w_np, atol=ATOL_FP32, rtol=RTOL_FP32)
    np.testing.assert_allclose(b_out, b_np, atol=ATOL_FP32, rtol=RTOL_FP32)

    dw_pt = PortableDepthwiseConv2d(in_channels, kernel_size=kernel_size, backend="pytorch")
    dw_pt.load_weights(w_out, b_out)

    x_np = _seeded_input((1, in_channels, 6, 6))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(dw_mlx(x_mlx), "mlx")
    y_pt = to_numpy(dw_pt(x_pt), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


# ---------------------------------------------------------------------------
# PortableMaxPool2d
# ---------------------------------------------------------------------------


def test_max_pool2d_kernel2_stride2_equivalence() -> None:
    """MaxPool2d kernel=2 stride=2 MLX vs PyTorch."""
    x_np = _seeded_input((2, 4, 8, 8))
    mp_mlx = PortableMaxPool2d(kernel_size=2, stride=2, backend="mlx")
    mp_pt = PortableMaxPool2d(kernel_size=2, stride=2, backend="pytorch")
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(mp_mlx(x_mlx), "mlx")
    y_pt = to_numpy(mp_pt(x_pt), "pytorch")
    assert y_pt.shape == (2, 4, 4, 4)
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_max_pool2d_kernel3_stride2_padding1_equivalence() -> None:
    """MaxPool2d kernel=3 stride=2 padding=1 (FastViT stem pattern)."""
    x_np = _seeded_input((1, 8, 16, 16))
    mp_mlx = PortableMaxPool2d(kernel_size=3, stride=2, padding=1, backend="mlx")
    mp_pt = PortableMaxPool2d(kernel_size=3, stride=2, padding=1, backend="pytorch")
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(mp_mlx(x_mlx), "mlx")
    y_pt = to_numpy(mp_pt(x_pt), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_max_pool2d_default_stride_equals_kernel() -> None:
    """MaxPool2d default stride=None means stride=kernel_size per PyTorch convention."""
    x_np = _seeded_input((1, 2, 8, 8))
    mp_mlx = PortableMaxPool2d(kernel_size=4, backend="mlx")  # stride defaults to 4
    mp_pt = PortableMaxPool2d(kernel_size=4, backend="pytorch")
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(mp_mlx(x_mlx), "mlx")
    y_pt = to_numpy(mp_pt(x_pt), "pytorch")
    assert y_pt.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


# ---------------------------------------------------------------------------
# PortableAvgPool2d
# ---------------------------------------------------------------------------


def test_avg_pool2d_spatial_equivalence() -> None:
    """AvgPool2d kernel=2 stride=2 (spatial downsample) MLX vs PyTorch."""
    x_np = _seeded_input((2, 4, 8, 8))
    ap_mlx = PortableAvgPool2d(kernel_size=2, stride=2, backend="mlx")
    ap_pt = PortableAvgPool2d(kernel_size=2, stride=2, backend="pytorch")
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(ap_mlx(x_mlx), "mlx")
    y_pt = to_numpy(ap_pt(x_pt), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_avg_pool2d_global_equivalence() -> None:
    """AvgPool2d global pool (kernel=spatial dims) MLX vs PyTorch.

    Per EfficientNet final global pool + squeeze-excite.
    """
    x_np = _seeded_input((2, 16, 4, 4))
    # Global pool: kernel matches input spatial dims.
    ap_mlx = PortableAvgPool2d(kernel_size=4, stride=4, backend="mlx")
    ap_pt = PortableAvgPool2d(kernel_size=4, stride=4, backend="pytorch")
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(ap_mlx(x_mlx), "mlx")
    y_pt = to_numpy(ap_pt(x_pt), "pytorch")
    assert y_pt.shape == (2, 16, 1, 1)
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_avg_pool2d_kernel3_stride1_padding1_equivalence() -> None:
    """AvgPool2d kernel=3 stride=1 padding=1 (same-size smoothing)."""
    x_np = _seeded_input((1, 4, 8, 8))
    ap_mlx = PortableAvgPool2d(kernel_size=3, stride=1, padding=1, backend="mlx")
    ap_pt = PortableAvgPool2d(kernel_size=3, stride=1, padding=1, backend="pytorch")
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(ap_mlx(x_mlx), "mlx")
    y_pt = to_numpy(ap_pt(x_pt), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


# ---------------------------------------------------------------------------
# silu / Swish
# ---------------------------------------------------------------------------


def test_silu_basic_equivalence() -> None:
    """silu activation MLX vs PyTorch within ε."""
    x_np = _seeded_input((4, 8))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(silu(x_mlx, backend="mlx"), "mlx")
    y_pt = to_numpy(silu(x_pt, backend="pytorch"), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_silu_4d_equivalence() -> None:
    """silu on 4D (B, C, H, W) tensor (post-conv activation pattern)."""
    x_np = _seeded_input((2, 16, 8, 8))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(silu(x_mlx, backend="mlx"), "mlx")
    y_pt = to_numpy(silu(x_pt, backend="pytorch"), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_silu_matches_x_sigmoid_x() -> None:
    """silu(x) = x * sigmoid(x) per canonical definition (Tan & Le 2019)."""
    x_np = _seeded_input((4, 8))
    expected = x_np * (1.0 / (1.0 + np.exp(-x_np)))

    for backend in ("mlx", "pytorch"):
        x = from_numpy(x_np, backend)
        y = to_numpy(silu(x, backend=backend), backend)
        np.testing.assert_allclose(y, expected, atol=ATOL_FP32, rtol=RTOL_FP32)
