"""NO-FAKE tests for the custom Metal grouped/depthwise Conv2d backward.

The custom backward is a THROUGHPUT tool: it MUST produce the same gradient
DIRECTION as the trusted Python-loop reference (``mlx_reference_conv2d_nhwc``).
These tests assert that on the REAL strided-grouped scorer layer shapes the
gradient cosine is ~1.0 AND the absolute drift is fp32 round-off, so a future
edit that zeroes / diverges / mis-indexes the gradient FAILS.

A degenerate stub (zero gradient, wrong-magnitude gradient, or a forward that
ignores the weights) would FAIL ``test_*_cosine`` and/or ``test_*_relmax``.

Authority: torch-CPU exact = d_seg/d_pose authority. These tests run on MLX
(CPU or GPU per default device); NO score claim, NO MPS.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tac.local_acceleration.metal_grouped_conv_backward import (  # noqa: E402
    make_grouped_conv2d_nhwc,
    metal_grouped_conv2d_backend_available,
)
from tac.local_acceleration.mlx_scorer_adapters import (  # noqa: E402
    mlx_reference_conv2d_nhwc,
)

# (name, N, Hin, Win, Cin, Cout, kH, kW, groups, stride) — real scorer shapes.
SCORER_FALLBACK_SHAPES = [
    ("segnet.blocks1.0.conv_dw", 2, 48, 64, 96, 96, 3, 3, 96, 2),
    ("segnet.blocks2.0.conv_dw", 2, 24, 32, 144, 144, 5, 5, 144, 2),
    ("posenet.stem.1.conv_kxk", 2, 48, 64, 64, 64, 3, 3, 64, 2),
    ("posenet.stages1.large_conv", 2, 24, 32, 64, 128, 7, 7, 64, 2),
]


def _build(shape):
    name, N, Hin, Win, Cin, Cout, kH, kW, groups, stride = shape
    Ipg = Cin // groups
    pad = (kH // 2, kW // 2)
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    x = mx.array(rng.standard_normal((N, Hin, Win, Cin)).astype(np.float32))
    w = mx.array(rng.standard_normal((Cout, kH, kW, Ipg)).astype(np.float32))
    conv = make_grouped_conv2d_nhwc(stride=stride, padding=pad, dilation=1, groups=groups)

    def cust_loss(xx, ww):
        return mx.sum(conv(xx, ww) ** 2)

    def ref_loss(xx, ww):
        return mx.sum(
            mlx_reference_conv2d_nhwc(
                xx, ww, None, stride=stride, padding=pad, dilation=1, groups=groups
            )
            ** 2
        )

    gx_c, gw_c = mx.grad(cust_loss, argnums=(0, 1))(x, w)
    gx_r, gw_r = mx.grad(ref_loss, argnums=(0, 1))(x, w)
    mx.eval(gx_c, gw_c, gx_r, gw_r)
    return (np.asarray(gx_c), np.asarray(gw_c), np.asarray(gx_r), np.asarray(gw_r))


def _cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    assert na > 0 and nb > 0
    return float(np.dot(a, b) / (na * nb))


@pytest.mark.parametrize("shape", SCORER_FALLBACK_SHAPES, ids=lambda s: s[0])
def test_grad_input_direction_matches_reference(shape):
    gx_c, _, gx_r, _ = _build(shape)
    cos = _cos(gx_c, gx_r)
    assert cos > 0.999, f"grad_input cosine {cos} too low (native MLX bwd is ~0.025)"


@pytest.mark.parametrize("shape", SCORER_FALLBACK_SHAPES, ids=lambda s: s[0])
def test_grad_weight_direction_matches_reference(shape):
    _, gw_c, _, gw_r = _build(shape)
    cos = _cos(gw_c, gw_r)
    assert cos > 0.999, f"grad_weight cosine {cos} too low"


@pytest.mark.parametrize("shape", SCORER_FALLBACK_SHAPES, ids=lambda s: s[0])
def test_grad_input_magnitude_is_fp32_roundoff(shape):
    gx_c, _, gx_r, _ = _build(shape)
    relmax = np.abs(gx_c - gx_r).max() / max(np.abs(gx_r).max(), 1e-9)
    assert relmax < 1e-3, f"grad_input relmax {relmax} exceeds fp32 drift bound"


def test_gradient_is_non_trivially_nonzero():
    """A zero/degenerate stub gradient would pass cosine vacuously — guard it."""
    gx_c, gw_c, _, _ = _build(SCORER_FALLBACK_SHAPES[0])
    assert np.abs(gx_c).max() > 1e-3
    assert np.abs(gw_c).max() > 1e-3


def test_forward_is_native_conv2d_parity():
    """Custom-function forward must equal native mx.conv2d (the fast path)."""
    name, N, Hin, Win, Cin, Cout, kH, kW, groups, stride = SCORER_FALLBACK_SHAPES[0]
    Ipg = Cin // groups
    pad = (kH // 2, kW // 2)
    rng = np.random.default_rng(7)
    x = mx.array(rng.standard_normal((N, Hin, Win, Cin)).astype(np.float32))
    w = mx.array(rng.standard_normal((Cout, kH, kW, Ipg)).astype(np.float32))
    conv = make_grouped_conv2d_nhwc(stride=stride, padding=pad, dilation=1, groups=groups)
    out_custom = conv(x, w)
    out_native = mx.conv2d(x, w, stride=stride, padding=pad, dilation=1, groups=groups)
    mx.eval(out_custom, out_native)
    assert float(np.abs(np.asarray(out_custom) - np.asarray(out_native)).max()) < 1e-5


def test_bare_grouped_conv2d_vjp_raises_helpful_error():
    """The module-level bare custom_function cannot recover config; must raise."""
    from tac.local_acceleration.metal_grouped_conv_backward import grouped_conv2d_nhwc

    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((1, 8, 8, 4)).astype(np.float32))
    w = mx.array(rng.standard_normal((4, 3, 3, 1)).astype(np.float32))

    def loss(xx):
        return mx.sum(grouped_conv2d_nhwc(xx, w, stride=2, padding=1, groups=4) ** 2)

    with pytest.raises((RuntimeError, TypeError)):
        g = mx.grad(loss)(x)
        mx.eval(g)


def test_backend_available_does_not_raise():
    # Regression guard for the DeviceType comparison bug (mx.gpu has no .type).
    assert isinstance(metal_grouped_conv2d_backend_available(), bool)
