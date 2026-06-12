# SPDX-License-Identifier: MIT
"""Custom Metal backward for strided grouped/depthwise Conv2d on MLX (NHWC).

Why this exists
---------------
MLX's native Metal reverse-mode for grouped/depthwise Conv2d with stride > 1 is
**numerically wrong** (it produces gradients 5-25x too large with near-zero
cosine vs the correct gradient — see
``experiments/results/mlx_backward_kernel_20260611/native_blowup_diagnosis.json``).
The FORWARD ``mx.conv2d`` is bit-exact and fast on Metal; only the strided-grouped
VJP is broken (the stride>1 grad-input scatter mishandles the group x stride
indexing — native grad cosine ~0.025, magnitude 5-25x too large). The repo
therefore forced these layers onto a Python-loop reference backward
(``mlx_reference_conv2d_nhwc``) which is correct but is the dominant backward cost:
swapping the custom kernel in makes the full-scorer (SegNet+PoseNet) backward
**~18x faster** (11,149 ms -> 621 ms at B=4) and the SegNet backward 12.9x (B=4) ->
35.5x (B=8) faster. The backward is >97% of the training step, so this is the
throughput lever that makes MLX-GPU the fast local backward backend. (Measured
2026-06-12, ``.omx/research/mlx_custom_grouped_backward_kernel_makes_mlx_gpu_fast_20260612.md``.)

This module provides a ``@mx.custom_function`` whose FORWARD calls the correct
native ``mx.conv2d`` and whose ``.vjp`` computes ``grad_input`` (a transposed/
scatter conv) and ``grad_weight`` (a correlation) with explicit, correct
group + stride + dilation indexing via two ``mx.fast.metal_kernel`` programs.

IMPORTANT: use ``make_grouped_conv2d_nhwc(...)`` (config bound by closure), NOT
the bare ``grouped_conv2d_nhwc`` — MLX's ``custom_function.vjp`` signature is
``(primals, cotangent, output)`` ONLY and does NOT forward the keyword-only conv
config, so the bare function is not gradient-safe for strided/grouped configs.

Authority note
--------------
This is a GRADIENT THROUGHPUT tool. The gradient it produces is fp32 and
near-exact (validated bit-for-bit vs the trusted Python-loop reference on CPU
and within fp32 reduction-order drift on Metal). It is **never** a d_seg/d_pose
score authority — the FP32-exact torch-CPU scorer remains the only authority.

Contract (NHWC throughout, matching ``mlx_reference_conv2d_nhwc``):
- ``x``      : (N, Hin, Win, Cin)
- ``weight`` : (Cout, kH, kW, Cin // groups)   (O, kH, kW, I/group)  -- MLX layout
- output     : (N, Hout, Wout, Cout)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

__all__ = [
    "grouped_conv2d_nhwc",
    "make_grouped_conv2d_nhwc",
    "metal_grouped_conv2d_backend_available",
]


# --- Metal kernel sources -------------------------------------------------

# grad_input: one thread per input element (n, hi, wi, cin).
# Accumulate contributions from every output position (ho, wo, oc-in-group)
# whose receptive field includes (hi, wi). For an output position to read input
# row hi through kernel tap kh:   ho*stride_h + kh*dil_h - pad_h == hi
# so for each tap (kh, kw) we test stride-alignment and bounds, no atomics.
_GRAD_INPUT_SRC = """
    uint elem = thread_position_in_grid.x;
    int N    = x_shape[0];
    int Hin  = x_shape[1];
    int Win  = x_shape[2];
    int Cin  = x_shape[3];
    if (elem >= (uint)(N * Hin * Win * Cin)) return;

    int cin = elem % Cin;
    int wi  = (elem / Cin) % Win;
    int hi  = (elem / (Cin * Win)) % Hin;
    int n   =  elem / (Cin * Win * Hin);

    int Cout = wt_shape[0];
    int kH   = wt_shape[1];
    int kW   = wt_shape[2];
    int Ipg  = wt_shape[3];          // in-channels per group
    int Opg  = Cout / groups[0];     // out-channels per group
    int g    = cin / Ipg;            // group index of this input channel
    int local_in = cin - g * Ipg;    // local input channel within the group

    int Hout = cot_shape[1];
    int Wout = cot_shape[2];

    float acc = 0.0f;
    for (int kh = 0; kh < kH; ++kh) {
        int num_h = hi + pad[0] - kh * dil[0];
        if (num_h < 0) continue;
        if (num_h % stride[0] != 0) continue;
        int ho = num_h / stride[0];
        if (ho < 0 || ho >= Hout) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int num_w = wi + pad[1] - kw * dil[1];
            if (num_w < 0) continue;
            if (num_w % stride[1] != 0) continue;
            int wo = num_w / stride[1];
            if (wo < 0 || wo >= Wout) continue;
            // sum over the Opg output channels in this group
            for (int oloc = 0; oloc < Opg; ++oloc) {
                int oc = g * Opg + oloc;
                // weight[oc, kh, kw, local_in]
                int widx = ((oc * kH + kh) * kW + kw) * Ipg + local_in;
                // cot[n, ho, wo, oc]
                int cidx = ((n * Hout + ho) * Wout + wo) * Cout + oc;
                acc += wt[widx] * cot[cidx];
            }
        }
    }
    gx[elem] = acc;
"""

# grad_weight: one thread per weight element (oc, kh, kw, local_in).
# gw = sum over (n, ho, wo) of x[n, ho*s+kh*dil-pad, wo*s+kw*dil-pad, in_c]
#                           * cot[n, ho, wo, oc]
_GRAD_WEIGHT_SRC = """
    uint elem = thread_position_in_grid.x;
    int Cout = wt_shape[0];
    int kH   = wt_shape[1];
    int kW   = wt_shape[2];
    int Ipg  = wt_shape[3];
    if (elem >= (uint)(Cout * kH * kW * Ipg)) return;

    int local_in = elem % Ipg;
    int kw =  (elem / Ipg) % kW;
    int kh =  (elem / (Ipg * kW)) % kH;
    int oc =   elem / (Ipg * kW * kH);

    int Opg = Cout / groups[0];
    int g   = oc / Opg;
    int cin = g * Ipg + local_in;    // actual input channel

    int N    = x_shape[0];
    int Hin  = x_shape[1];
    int Win  = x_shape[2];
    int Cin  = x_shape[3];

    int Hout = cot_shape[1];
    int Wout = cot_shape[2];

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int ho = 0; ho < Hout; ++ho) {
            int hi = ho * stride[0] + kh * dil[0] - pad[0];
            if (hi < 0 || hi >= Hin) continue;
            for (int wo = 0; wo < Wout; ++wo) {
                int wi = wo * stride[1] + kw * dil[1] - pad[1];
                if (wi < 0 || wi >= Win) continue;
                int xidx = ((n * Hin + hi) * Win + wi) * Cin + cin;
                int cidx = ((n * Hout + ho) * Wout + wo) * Cout + oc;
                acc += x[xidx] * cot[cidx];
            }
        }
    }
    gw[elem] = acc;
"""

_grad_input_kernel = None
_grad_weight_kernel = None


def _kernels():
    global _grad_input_kernel, _grad_weight_kernel
    if _grad_input_kernel is None:
        _grad_input_kernel = mx.fast.metal_kernel(
            name="grouped_conv2d_grad_input",
            input_names=["x", "wt", "cot", "stride", "pad", "dil", "groups"],
            output_names=["gx"],
            source=_GRAD_INPUT_SRC,
        )
        _grad_weight_kernel = mx.fast.metal_kernel(
            name="grouped_conv2d_grad_weight",
            input_names=["x", "wt", "cot", "stride", "pad", "dil", "groups"],
            output_names=["gw"],
            source=_GRAD_WEIGHT_SRC,
        )
    return _grad_input_kernel, _grad_weight_kernel


def metal_grouped_conv2d_backend_available() -> bool:
    """Return True when an MLX GPU (Metal) device is the default device."""

    try:
        # mx.gpu is a DeviceType enum (not a Device); compare against it directly.
        return mx.default_device().type == mx.gpu
    except Exception:  # pragma: no cover - device introspection guard
        return False


def _i32(values: tuple[int, int]) -> Any:
    return mx.array([int(values[0]), int(values[1])], dtype=mx.int32)


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


@mx.custom_function
def grouped_conv2d_nhwc(
    x: Any,
    weight: Any,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Any:
    """Grouped Conv2d (NHWC) whose forward is native ``mx.conv2d``.

    The backward (registered below via ``.vjp``) uses custom, numerically
    correct Metal kernels for ``grad_input`` and ``grad_weight`` so strided
    grouped/depthwise conv has a fast AND correct Metal reverse-mode.
    """

    return mx.conv2d(
        x,
        weight,
        stride=_pair(stride),
        padding=_pair(padding),
        dilation=_pair(dilation),
        groups=int(groups),
    )


def make_grouped_conv2d_nhwc(
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Any:
    """Return a config-bound ``@mx.custom_function`` grouped Conv2d (NHWC).

    Why a factory: MLX's ``custom_function.vjp`` receives only
    ``(primals, cotangent, output)`` — it does NOT forward the keyword-only
    conv config (stride/padding/dilation/groups) to the vjp. The bare
    ``grouped_conv2d_nhwc`` above therefore raises ``TypeError`` under
    ``mx.grad`` because its vjp asks for those kwargs and never gets them. This
    factory closes the config over the vjp, so the returned callable is a
    correct, gradient-enabled drop-in. Forward = native ``mx.conv2d`` (bit-exact,
    fast); backward = the two correct Metal kernels.

    The returned callable's signature is ``conv(x, weight) -> output`` with the
    same NHWC layout as ``mlx_reference_conv2d_nhwc``.
    """

    s = _pair(stride)
    p = _pair(padding)
    d = _pair(dilation)
    g = int(groups)
    sh = _i32(s)
    pd = _i32(p)
    dl = _i32(d)
    gp = mx.array([g], dtype=mx.int32)

    @mx.custom_function
    def conv(x: Any, weight: Any) -> Any:
        return mx.conv2d(x, weight, stride=s, padding=p, dilation=d, groups=g)

    @conv.vjp
    def _conv_vjp(primals, cotangent, output):
        x, weight = primals
        grad_input_kernel, grad_weight_kernel = _kernels()
        (gx,) = grad_input_kernel(
            inputs=[x, weight, cotangent, sh, pd, dl, gp],
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
            grid=(int(x.size), 1, 1),
            threadgroup=(256, 1, 1),
        )
        (gw,) = grad_weight_kernel(
            inputs=[x, weight, cotangent, sh, pd, dl, gp],
            output_shapes=[weight.shape],
            output_dtypes=[weight.dtype],
            grid=(int(weight.size), 1, 1),
            threadgroup=(256, 1, 1),
        )
        return gx, gw

    return conv


@grouped_conv2d_nhwc.vjp
def _grouped_conv2d_nhwc_vjp(primals, cotangent, output):
    # NOTE: MLX's custom_function.vjp signature is (primals, cotangent, output)
    # ONLY — the keyword-only conv config is NOT forwarded here. This bare vjp
    # cannot recover stride/groups/etc., so the module-level
    # ``grouped_conv2d_nhwc`` is NOT gradient-safe for strided/grouped configs.
    # Use ``make_grouped_conv2d_nhwc(...)`` (config bound by closure) instead.
    raise RuntimeError(
        "grouped_conv2d_nhwc.vjp cannot recover the conv config from MLX's "
        "(primals, cotangent, output) vjp signature. Use "
        "make_grouped_conv2d_nhwc(stride=..., padding=..., dilation=..., "
        "groups=...) which binds the config by closure."
    )
