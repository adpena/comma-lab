#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Localize the SegNet grad_input custom-vs-reference relmax (2.4% @ cos 0.99996).

The e2e diagnostic showed POSE grad is byte-faithful (relmax 2.9e-6) but SEG grad
has a small relmax that grows with training. Is that relmax (a) a real kernel
indexing bug on the SegNet strided-depthwise shapes, or (b) fp32 reduction-order
noise concentrated where the reference grad is near-zero (cancellation)?

Decisive test: per SegNet strided-depthwise shape, on REAL activation
distributions (not iid random — biased + correlated like real feature maps),
compare the kernel grad_input to the loop reference and report BOTH the global
relmax AND the relmax RESTRICTED to elements where |ref_grad| > 1% of max (i.e.
exclude the near-zero cancellation tail). If the restricted relmax is fp32-tiny,
the discrepancy is reduction-order noise, NOT a bug.

$0, local MLX (GPU default), NO MPS, NO score claim.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from tac.local_acceleration.metal_grouped_conv_backward import make_grouped_conv2d_nhwc
from tac.local_acceleration.mlx_scorer_adapters import mlx_reference_conv2d_nhwc

# Real SegNet EfficientNet-B2 strided depthwise shapes (Ipg=1, Opg=1).
SEG_DW = [
    ("segnet.blocks1.0.conv_dw", 4, 192, 256, 96, 96, 3, 3, 96, 2),
    ("segnet.blocks2.0.conv_dw", 4, 96, 128, 144, 144, 5, 5, 144, 2),
    ("segnet.blocks3.0.conv_dw", 4, 48, 64, 288, 288, 3, 3, 288, 2),
    ("segnet.blocks5.0.conv_dw", 4, 24, 32, 720, 720, 5, 5, 720, 2),
]


def run(shape):
    name, N, Hin, Win, Cin, Cout, kH, kW, groups, stride = shape
    Ipg = Cin // groups
    pad = (kH // 2, kW // 2)
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    # Real-ish activations: post-ReLU/BN are non-negative + spatially correlated.
    # Build a smooth, biased, partly-sparse map (NOT iid gaussian).
    base = rng.standard_normal((N, Hin, Win, Cin)).astype(np.float32)
    # spatial smoothing to mimic correlated feature maps
    base = np.cumsum(base, axis=1) / np.sqrt(Hin)
    base = np.maximum(base + 0.5, 0.0)  # ReLU-like non-negative bias
    x = mx.array(base)
    w = mx.array((rng.standard_normal((Cout, kH, kW, Ipg)) * 0.1).astype(np.float32))
    conv = make_grouped_conv2d_nhwc(stride=stride, padding=pad, dilation=1, groups=groups)

    # Use a realistic cotangent (loss grad), not d(sum(out**2)); a 1-hot-ish
    # sparse cotangent stresses the scatter more than a dense one.
    def cust_loss(xx):
        out = conv(xx, w)
        # weighted L1-ish so the cotangent magnitude varies across positions
        return mx.sum(mx.abs(out) * (out > 0))

    def ref_loss(xx):
        out = mlx_reference_conv2d_nhwc(xx, w, None, stride=stride, padding=pad, dilation=1, groups=groups)
        return mx.sum(mx.abs(out) * (out > 0))

    gx_c = mx.grad(cust_loss)(x)
    gx_r = mx.grad(ref_loss)(x)
    mx.eval(gx_c, gx_r)
    gc = np.asarray(gx_c).astype(np.float64)
    gr = np.asarray(gx_r).astype(np.float64)

    gmax = np.abs(gr).max()
    global_relmax = np.abs(gc - gr).max() / max(gmax, 1e-12)
    # Restrict to "meaningful" gradient elements (exclude cancellation tail).
    mask = np.abs(gr) > 0.01 * gmax
    if mask.sum() > 0:
        restricted_relmax = (np.abs(gc - gr)[mask] / np.abs(gr)[mask]).max()
    else:
        restricted_relmax = float("nan")
    cos = float(np.dot(gc.ravel(), gr.ravel()) / (np.linalg.norm(gc.ravel()) * np.linalg.norm(gr.ravel())))
    n_tail = int((~mask).sum())
    return name, cos, global_relmax, restricted_relmax, gmax, n_tail, gr.size


def main():
    print(f"{'layer':30s} {'cos':>10s} {'global_relmax':>14s} {'restricted_relmax':>18s} {'gmax':>10s} {'tail/total':>14s}")
    for shape in SEG_DW:
        name, cos, grm, rrm, gmax, n_tail, total = run(shape)
        print(f"{name:30s} {cos:10.7f} {grm:14.6f} {rrm:18.8f} {gmax:10.3e} {n_tail:>7d}/{total:<7d}")


if __name__ == "__main__":
    main()
