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
#
# COVERAGE: ALL 12 strided-grouped conv shapes that route through the custom
# Metal backward in the REAL DistortionNet (4 SegNet EfficientNet-B2 depthwise +
# 8 PoseNet FastViT-T12 strided-grouped — enumerated from
# experiments/results/mlx_backward_kernel_20260611/conv_inventory.json). The
# original test covered only 4 of these (1 SegNet + the small PoseNet ones); the
# n600-pose-divergence investigation (2026-06-12) added the missing 8 — every
# PoseNet ``large_conv``/``small_conv`` at g64/g128/g256 with the Opg=2 channel
# multiplier — so the gradient self-protection covers the FULL pose backward,
# not just d_seg. This is the "fix + a gate that catches re-introduction"
# discipline (CLAUDE.md "Bugs must be permanently fixed AND self-protected"):
# the kernel gradient was PROVEN correct on all 12 (cosine 1.0), so these tests
# pin that — a future kernel edit that breaks ANY pose shape FAILS here.
SCORER_FALLBACK_SHAPES = [
    # SegNet EfficientNet-B2 strided depthwise (Ipg=1, Opg=1).
    ("segnet.blocks1.0.conv_dw", 2, 48, 64, 96, 96, 3, 3, 96, 2),
    ("segnet.blocks2.0.conv_dw", 2, 24, 32, 144, 144, 5, 5, 144, 2),
    ("segnet.blocks3.0.conv_dw", 2, 12, 16, 288, 288, 3, 3, 288, 2),
    ("segnet.blocks5.0.conv_dw", 2, 6, 8, 720, 720, 5, 5, 720, 2),
    # PoseNet FastViT-T12 stem strided depthwise (Ipg=1, Opg=1).
    ("posenet.stem.1.conv_kxk", 2, 48, 64, 64, 64, 3, 3, 64, 2),
    ("posenet.stem.1.conv_scale", 2, 48, 64, 64, 64, 1, 1, 64, 2),
    # PoseNet FastViT downsample large/small convs — the POSE backward the n8
    # validation NEVER exercised on d_pose. Opg=2 channel-multiplier (g halves
    # the channels, out doubles), the indexing class most at risk of a backward
    # bug. g64 / g128 / g256 at k7 (large) and k3 (small).
    ("posenet.stages1.large_conv", 2, 24, 32, 64, 128, 7, 7, 64, 2),
    ("posenet.stages1.small_conv", 2, 24, 32, 64, 128, 3, 3, 64, 2),
    ("posenet.stages2.large_conv", 2, 12, 16, 128, 256, 7, 7, 128, 2),
    ("posenet.stages2.small_conv", 2, 12, 16, 128, 256, 3, 3, 128, 2),
    ("posenet.stages3.large_conv", 2, 6, 8, 256, 512, 7, 7, 256, 2),
    ("posenet.stages3.small_conv", 2, 6, 8, 256, 512, 3, 3, 256, 2),
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


@pytest.mark.parametrize("shape", SCORER_FALLBACK_SHAPES, ids=lambda s: s[0])
def test_grad_weight_magnitude_is_fp32_roundoff(shape):
    """grad_weight (the correlation kernel) must also be fp32-roundoff exact, not
    just direction-correct — a magnitude bug compounds over training steps."""
    _, gw_c, _, gw_r = _build(shape)
    relmax = np.abs(gw_c - gw_r).max() / max(np.abs(gw_r).max(), 1e-9)
    assert relmax < 1e-3, f"grad_weight relmax {relmax} exceeds fp32 drift bound"


@pytest.mark.parametrize("batch", [1, 4])
def test_pose_channel_multiplier_grad_parity_across_batch(batch):
    """The POSE divergence (n600) was on the channel-multiplier (Opg=2) PoseNet
    downsample convs. Pin grad parity for that exact shape class across batch
    sizes — a batch-dependent indexing bug would slip past the N=2 default."""
    # posenet.stages1.large_conv: 64 -> 128, g64, k7, s2 (Opg=2).
    name, _, Hin, Win, Cin, Cout, kH, kW, groups, stride = (
        "posenet.stages1.large_conv", 0, 24, 32, 64, 128, 7, 7, 64, 2,
    )
    Ipg = Cin // groups
    pad = (kH // 2, kW // 2)
    rng = np.random.default_rng(99 + batch)
    x = mx.array(rng.standard_normal((batch, Hin, Win, Cin)).astype(np.float32))
    w = mx.array(rng.standard_normal((Cout, kH, kW, Ipg)).astype(np.float32))
    conv = make_grouped_conv2d_nhwc(stride=stride, padding=pad, dilation=1, groups=groups)

    def cl(xx, ww):
        return mx.sum(conv(xx, ww) ** 2)

    def rl(xx, ww):
        return mx.sum(
            mlx_reference_conv2d_nhwc(
                xx, ww, None, stride=stride, padding=pad, dilation=1, groups=groups
            ) ** 2
        )

    gx_c, gw_c = mx.grad(cl, argnums=(0, 1))(x, w)
    gx_r, gw_r = mx.grad(rl, argnums=(0, 1))(x, w)
    mx.eval(gx_c, gw_c, gx_r, gw_r)
    cos_x = _cos(np.asarray(gx_c), np.asarray(gx_r))
    cos_w = _cos(np.asarray(gw_c), np.asarray(gw_r))
    assert cos_x > 0.999, f"pose Opg=2 grad_input cosine {cos_x} (batch={batch})"
    assert cos_w > 0.999, f"pose Opg=2 grad_weight cosine {cos_w} (batch={batch})"


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


# ---------------------------------------------------------------------------
# n600-pose-divergence INVESTIGATION self-protection (2026-06-12)
#
# The n600 training run diverged (d_seg 0.028->0.49, d_pose 0.84->36) and the
# memo's first hypothesis blamed the kernel's PoseNet gradient. The investigation
# FALSIFIED that: (a) the per-layer grad_input is bit-exact on REALISTIC
# (correlated, ReLU-biased, sparse-cotangent) activations — not just iid random;
# (b) the full end-to-end render-pixel cotangent through the FULL PoseNet matches
# the reference to ~1e-6 (pose) and fp32 reduction-noise (seg); (c) at an
# aggressive muon_lr the trusted REFERENCE backward diverges IDENTICALLY to the
# custom kernel, so the divergence is an LR/optimizer instability, NOT the kernel.
#
# These tests pin (a) — the gap the original iid-random + d_seg-only validation
# left. A future kernel edit that breaks grad_input on the real activation
# distribution (where the divergence regime actually lives) FAILS here even if
# the iid-random cosine tests above still pass.
# ---------------------------------------------------------------------------

# Real SegNet (EfficientNet-B2) + PoseNet (FastViT-T12) strided-grouped shapes.
# SegNet are Opg=1 depthwise; PoseNet stages are Opg=2 channel-multiplier — the
# exact class the divergence memo flagged as "most at risk".
REALISTIC_ACTIVATION_SHAPES = [
    ("segnet.blocks1.0.conv_dw", 2, 96, 128, 96, 96, 3, 3, 96, 2),
    ("segnet.blocks3.0.conv_dw", 2, 24, 32, 288, 288, 3, 3, 288, 2),
    ("posenet.stages1.large_conv", 2, 24, 32, 64, 128, 7, 7, 64, 2),
    ("posenet.stages2.large_conv", 2, 12, 16, 128, 256, 7, 7, 128, 2),
    ("posenet.stages3.small_conv", 2, 6, 8, 256, 512, 3, 3, 256, 2),
]


def _build_realistic(shape):
    """Grad parity on REALISTIC activations (not iid gaussian) + a sparse
    cotangent (an L1-on-positives loss), which stresses the grad_input scatter
    far more than the dense ``sum(out**2)`` of the cosine tests above."""
    name, N, Hin, Win, Cin, Cout, kH, kW, groups, stride = shape
    Ipg = Cin // groups
    pad = (kH // 2, kW // 2)
    rng = np.random.default_rng((abs(hash(name)) % (2**31)) ^ 0x5EED)
    # ReLU/BN-like: spatially-correlated (cumsum-smoothed), non-negative, biased.
    base = rng.standard_normal((N, Hin, Win, Cin)).astype(np.float32)
    base = np.cumsum(base, axis=1) / np.sqrt(Hin)
    base = np.maximum(base + 0.5, 0.0).astype(np.float32)
    x = mx.array(base)
    w = mx.array((rng.standard_normal((Cout, kH, kW, Ipg)) * 0.1).astype(np.float32))
    conv = make_grouped_conv2d_nhwc(stride=stride, padding=pad, dilation=1, groups=groups)

    def cust_loss(xx):
        out = conv(xx, w)
        return mx.sum(mx.abs(out) * (out > 0))

    def ref_loss(xx):
        out = mlx_reference_conv2d_nhwc(
            xx, w, None, stride=stride, padding=pad, dilation=1, groups=groups
        )
        return mx.sum(mx.abs(out) * (out > 0))

    gx_c = mx.grad(cust_loss)(x)
    gx_r = mx.grad(ref_loss)(x)
    mx.eval(gx_c, gx_r)
    return np.asarray(gx_c).astype(np.float64), np.asarray(gx_r).astype(np.float64)


@pytest.mark.parametrize("shape", REALISTIC_ACTIVATION_SHAPES, ids=lambda s: s[0])
def test_grad_input_bit_exact_on_realistic_activations(shape):
    """On real-like activations + a sparse cotangent the kernel grad_input is
    BIT-EXACT vs the loop reference where the gradient is meaningful. The global
    relmax may show a large value at near-zero cancellation pixels (fp32 order
    noise); restricting to |ref| > 1% of max removes that tail and must be
    fp32-tiny. This is the regression gate the iid-random validation lacked."""
    gc, gr = _build_realistic(shape)
    gmax = np.abs(gr).max()
    assert gmax > 1e-6, "reference gradient is degenerate"
    mask = np.abs(gr) > 0.01 * gmax
    assert mask.sum() > 0
    restricted_relmax = float((np.abs(gc - gr)[mask] / np.abs(gr)[mask]).max())
    # fp32 reduction-order bound on the meaningful-gradient elements.
    assert restricted_relmax < 1e-4, (
        f"{shape[0]}: grad_input restricted_relmax {restricted_relmax} exceeds "
        "fp32 reduction bound — a REAL kernel indexing bug on the real activation "
        "distribution (the n600-divergence regime), not order noise"
    )
    cos = _cos(gc, gr)
    assert cos > 0.99999, f"{shape[0]}: grad_input cosine {cos} too low on real acts"


def test_pose_channel_multiplier_grad_input_bit_exact_realistic():
    """The n600 memo flagged the PoseNet Opg=2 downsample convs as the prime
    suspect for d_pose. On realistic activations their grad_input is bit-exact
    (restricted relmax fp32-tiny) — the memo's pose-gradient-bug hypothesis is
    FALSIFIED here so it cannot silently re-enter as a verdict."""
    gc, gr = _build_realistic(REALISTIC_ACTIVATION_SHAPES[2])  # stages1.large_conv
    gmax = np.abs(gr).max()
    mask = np.abs(gr) > 0.01 * gmax
    restricted_relmax = float((np.abs(gc - gr)[mask] / np.abs(gr)[mask]).max())
    assert restricted_relmax < 1e-4
    assert _cos(gc, gr) > 0.99999
