# SPDX-License-Identifier: MIT
"""Custom Metal forward kernels for frozen-SegNet pointwise/depthwise conv.

This module is a local throughput research surface.  It is deliberately
default-OFF, forward-only, and never score authority.  The frozen Torch-fp32
scorer and a fixed-order NumPy-fp32 reference remain the numerical authorities.

The pointwise kernel flattens NHWC ``(B,H,W,Cin)`` to an implicit GEMM
``(M,Cin) @ (Cin,Cout)``.  One SIMD group computes one 8x8 output tile.  Frozen
weights can be fp16, symmetric per-output-channel int8, or packed symmetric
per-output-channel int4; quantized weights are dequantized into half
threadgroup tiles before SIMD-group matrix multiply.  The destination fragment
is float, so the requested arithmetic is half input multiply with float
accumulation.  A Metal compiler that does not expose that mixed overload must
fail the opt-in closed rather than silently relabeling a float or half-only MMA.

The depthwise kernel uses fp16 activation/weight traffic and a float scalar
accumulator.  It supports the frozen B2 3x3/5x5, stride-1/2 cases.  Neither
custom kernel has a VJP; the default-OFF wire-in is therefore forward-only.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np

CUSTOM_SEGNET_CONV_FLAG = "TAC_MLX_CUSTOM_SEGNET_CONV"
CUSTOM_SEGNET_CONV_WEIGHT_VARIANT_FLAG = "TAC_MLX_CUSTOM_SEGNET_POINTWISE_WEIGHT"
N600_FIDELITY_GATE = "custom-metal-segnet-conv-n600-fidelity-gate"

PointwiseWeightVariant = Literal["fp16", "int8", "int4"]
VALID_POINTWISE_WEIGHT_VARIANTS = frozenset({"fp16", "int8", "int4"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})

__all__ = [
    "CUSTOM_SEGNET_CONV_FLAG",
    "CUSTOM_SEGNET_CONV_WEIGHT_VARIANT_FLAG",
    "N600_FIDELITY_GATE",
    "VALID_POINTWISE_WEIGHT_VARIANTS",
    "MetalSegNetConv2DAdapter",
    "PointwiseQuantizedWeights",
    "build_custom_metal_segnet_adapter",
    "custom_segnet_conv_enabled",
    "custom_segnet_conv_env_requested",
    "custom_segnet_conv_signature",
    "depthwise_conv2d_metal",
    "depthwise_conv2d_numpy_fp32",
    "dequantize_pointwise_int4",
    "dequantize_pointwise_int8",
    "metal_segnet_conv_backend_available",
    "pack_signed_int4",
    "pointwise_1x1_metal",
    "pointwise_1x1_numpy_fp32",
    "quantize_pointwise_int4",
    "quantize_pointwise_int8",
    "unpack_signed_int4",
]


@dataclass(frozen=True)
class PointwiseQuantizedWeights:
    """Deterministic per-output-channel pointwise weight packet.

    ``values`` is KxN int8 for ``int8`` and a flat packed uint8 byte stream for
    ``int4``.  ``scales`` is float32 with one scale per output channel.
    """

    variant: Literal["int8", "int4"]
    values: np.ndarray
    scales: np.ndarray
    cin: int
    cout: int


def _as_float32_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _validate_weight_kn(weight_kn: Any) -> np.ndarray:
    weight = _as_float32_array(weight_kn, name="weight_kn")
    if weight.ndim != 2 or min(weight.shape) <= 0:
        raise ValueError(f"weight_kn must be non-empty KxN, got {weight.shape}")
    return weight


def _pointwise_weight_kn_from_oihw(weight_oihw: Any) -> np.ndarray:
    """Convert Torch pointwise OIHW weights to the kernel's KxN layout."""

    weight = _as_float32_array(weight_oihw, name="weight_oihw")
    if weight.ndim != 4 or weight.shape[2:] != (1, 1):
        raise ValueError(f"pointwise weight must be OIx1x1, got {weight.shape}")
    return np.ascontiguousarray(weight[:, :, 0, 0].T)


def pointwise_1x1_numpy_fp32(
    x_nhwc: Any,
    weight_kn: Any,
    bias: Any | None = None,
) -> np.ndarray:
    """Fixed-K-order NumPy-fp32 authority for NHWC pointwise convolution.

    Callers choose the source rounding before entry.  For the Metal fp16 path,
    pass fp16-rounded activation/weights; they are promoted to fp32 here and
    accumulated in a fixed ascending-K order.
    """

    x = _as_float32_array(x_nhwc, name="x_nhwc")
    weight = _validate_weight_kn(weight_kn)
    if x.ndim != 4:
        raise ValueError(f"x_nhwc must be rank-4, got {x.shape}")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"Cin mismatch: activation {x.shape[-1]} vs weight {weight.shape[0]}"
        )
    out = np.zeros((*x.shape[:-1], weight.shape[1]), dtype=np.float32)
    for k in range(weight.shape[0]):
        term = x[..., k : k + 1] * weight[k : k + 1, :]
        out = np.asarray(out + term, dtype=np.float32)
    if bias is not None:
        bias_array = _as_float32_array(bias, name="bias").reshape(-1)
        if bias_array.shape != (weight.shape[1],):
            raise ValueError(
                f"bias must have shape ({weight.shape[1]},), got {bias_array.shape}"
            )
        out = np.asarray(out + bias_array.reshape(1, 1, 1, -1), dtype=np.float32)
    return out


def _quantize_symmetric_per_output(
    weight_kn: Any,
    *,
    qmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    weight = _validate_weight_kn(weight_kn)
    max_abs = np.max(np.abs(weight), axis=0).astype(np.float32)
    scales = np.where(max_abs > 0, max_abs / np.float32(qmax), np.float32(1.0))
    quantized = np.rint(weight / scales[None, :])
    quantized = np.clip(quantized, -qmax, qmax).astype(np.int8)
    return np.ascontiguousarray(quantized), np.ascontiguousarray(scales)


def quantize_pointwise_int8(weight_kn: Any) -> PointwiseQuantizedWeights:
    """Symmetric signed int8, one deterministic scale per output channel."""

    weight = _validate_weight_kn(weight_kn)
    values, scales = _quantize_symmetric_per_output(weight, qmax=127)
    return PointwiseQuantizedWeights(
        variant="int8",
        values=values,
        scales=scales,
        cin=int(weight.shape[0]),
        cout=int(weight.shape[1]),
    )


def dequantize_pointwise_int8(packet: PointwiseQuantizedWeights) -> np.ndarray:
    if packet.variant != "int8":
        raise ValueError(f"expected int8 packet, got {packet.variant!r}")
    values = np.asarray(packet.values, dtype=np.int8)
    if values.shape != (packet.cin, packet.cout):
        raise ValueError(
            f"int8 values must be {(packet.cin, packet.cout)}, got {values.shape}"
        )
    scales = np.asarray(packet.scales, dtype=np.float32).reshape(-1)
    if scales.shape != (packet.cout,):
        raise ValueError(f"scales must be ({packet.cout},), got {scales.shape}")
    return np.asarray(values.astype(np.float32) * scales[None, :], dtype=np.float32)


def pack_signed_int4(values: Any) -> np.ndarray:
    """Pack signed [-8,7] nibbles, low nibble first, deterministically."""

    q = np.asarray(values, dtype=np.int8).reshape(-1)
    if np.any(q < -8) or np.any(q > 7):
        raise ValueError("signed int4 values must lie in [-8, 7]")
    nibble = np.bitwise_and(q.astype(np.int16), 0xF).astype(np.uint8)
    if nibble.size % 2:
        nibble = np.concatenate([nibble, np.zeros(1, dtype=np.uint8)])
    packed = nibble[0::2] | np.left_shift(nibble[1::2], np.uint8(4))
    return np.ascontiguousarray(packed, dtype=np.uint8)


def unpack_signed_int4(packed: Any, *, count: int) -> np.ndarray:
    """Inverse of :func:`pack_signed_int4` for exactly ``count`` values."""

    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")
    raw = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if raw.size * 2 < count:
        raise ValueError(f"packed stream has {raw.size * 2} nibbles, needs {count}")
    nibbles = np.empty(raw.size * 2, dtype=np.uint8)
    nibbles[0::2] = raw & np.uint8(0xF)
    nibbles[1::2] = raw >> np.uint8(4)
    signed = nibbles.astype(np.int8)
    signed[signed >= 8] -= np.int8(16)
    return np.ascontiguousarray(signed[:count])


def quantize_pointwise_int4(weight_kn: Any) -> PointwiseQuantizedWeights:
    """Symmetric signed int4 [-7,7], packed in flattened KxN order."""

    weight = _validate_weight_kn(weight_kn)
    values, scales = _quantize_symmetric_per_output(weight, qmax=7)
    return PointwiseQuantizedWeights(
        variant="int4",
        values=pack_signed_int4(values),
        scales=scales,
        cin=int(weight.shape[0]),
        cout=int(weight.shape[1]),
    )


def dequantize_pointwise_int4(packet: PointwiseQuantizedWeights) -> np.ndarray:
    if packet.variant != "int4":
        raise ValueError(f"expected int4 packet, got {packet.variant!r}")
    values = unpack_signed_int4(
        packet.values,
        count=int(packet.cin * packet.cout),
    ).reshape(packet.cin, packet.cout)
    scales = np.asarray(packet.scales, dtype=np.float32).reshape(-1)
    if scales.shape != (packet.cout,):
        raise ValueError(f"scales must be ({packet.cout},), got {scales.shape}")
    return np.asarray(values.astype(np.float32) * scales[None, :], dtype=np.float32)


def depthwise_conv2d_numpy_fp32(
    x_nhwc: Any,
    weight_ckk: Any,
    *,
    stride: int = 1,
    padding: int | None = None,
    bias: Any | None = None,
) -> np.ndarray:
    """Fixed tap-order NumPy-fp32 authority for B2 depthwise convolution."""

    x = _as_float32_array(x_nhwc, name="x_nhwc")
    weight = _as_float32_array(weight_ckk, name="weight_ckk")
    if x.ndim != 4:
        raise ValueError(f"x_nhwc must be rank-4, got {x.shape}")
    if weight.ndim == 4 and weight.shape[-1] == 1:
        weight = weight[..., 0]
    if weight.ndim != 3 or weight.shape[1] != weight.shape[2]:
        raise ValueError(f"weight_ckk must be CxKxK or CxKxKx1, got {weight.shape}")
    channels, kernel_h, kernel_w = weight.shape
    if channels != x.shape[-1] or kernel_h not in (3, 5):
        raise ValueError(
            f"unsupported depthwise shape x={x.shape}, weight={weight.shape}"
        )
    stride = int(stride)
    if stride not in (1, 2):
        raise ValueError(f"stride must be 1 or 2, got {stride}")
    pad = kernel_h // 2 if padding is None else int(padding)
    if pad < 0:
        raise ValueError("padding must be non-negative")
    batch, height, width, _ = x.shape
    out_h = (height + 2 * pad - kernel_h) // stride + 1
    out_w = (width + 2 * pad - kernel_w) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("depthwise output has non-positive spatial extent")
    out = np.zeros((batch, out_h, out_w, channels), dtype=np.float32)
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            for oh in range(out_h):
                ih = oh * stride + kh - pad
                if ih < 0 or ih >= height:
                    continue
                for ow in range(out_w):
                    iw = ow * stride + kw - pad
                    if iw < 0 or iw >= width:
                        continue
                    term = x[:, ih, iw, :] * weight[:, kh, kw]
                    out[:, oh, ow, :] = np.asarray(
                        out[:, oh, ow, :] + term,
                        dtype=np.float32,
                    )
    if bias is not None:
        bias_array = _as_float32_array(bias, name="bias").reshape(-1)
        if bias_array.shape != (channels,):
            raise ValueError(f"bias must be ({channels},), got {bias_array.shape}")
        out = np.asarray(out + bias_array.reshape(1, 1, 1, -1), dtype=np.float32)
    return out


_POINTWISE_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""


def _pointwise_source(variant: PointwiseWeightVariant) -> str:
    if variant == "fp16":
        weight_load = "b_tile[slot] = w[weight_idx];"
    elif variant == "int8":
        weight_load = "b_tile[slot] = half(float(w[weight_idx]) * float(scales[n]));"
    elif variant == "int4":
        weight_load = """
            uchar packed = w[weight_idx >> 1];
            uchar nibble = (weight_idx & 1) ? (packed >> 4) : (packed & 15);
            int q = (nibble >= 8) ? int(nibble) - 16 : int(nibble);
            b_tile[slot] = half(float(q) * float(scales[n]));
        """
    else:  # pragma: no cover - guarded by public API
        raise ValueError(f"unsupported pointwise variant: {variant!r}")
    return f"""
    uint lane = thread_index_in_simdgroup;
    uint tile_n = threadgroup_position_in_grid.x;
    uint tile_m = threadgroup_position_in_grid.y;
    int M = dims[0];
    int K = dims[1];
    int N = dims[2];
    int row0 = int(tile_m) * 8;
    int col0 = int(tile_n) * 8;

    threadgroup half a_tile[64];
    threadgroup half b_tile[64];
    threadgroup float c_tile[64];
    simdgroup_float8x8 accum(0.0f);

    for (int k0 = 0; k0 < K; k0 += 8) {{
        for (uint slot = lane; slot < 64; slot += 32) {{
            int r = int(slot >> 3);
            int c = int(slot & 7);
            int m = row0 + r;
            int k = k0 + c;
            a_tile[slot] = (m < M && k < K) ? x[m * K + k] : half(0.0h);

            k = k0 + r;
            int n = col0 + c;
            if (k < K && n < N) {{
                int weight_idx = k * N + n;
                {weight_load}
            }} else {{
                b_tile[slot] = half(0.0h);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 a_frag;
        simdgroup_half8x8 b_frag;
        simdgroup_load(a_frag, a_tile, 8);
        simdgroup_load(b_frag, b_tile, 8);
        simdgroup_multiply_accumulate(accum, a_frag, b_frag, accum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    simdgroup_store(accum, c_tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint slot = lane; slot < 64; slot += 32) {{
        int r = int(slot >> 3);
        int c = int(slot & 7);
        int m = row0 + r;
        int n = col0 + c;
        if (m < M && n < N) out[m * N + n] = c_tile[slot];
    }}
    """


_DEPTHWISE_SOURCE = """
    #pragma clang fp contract(off)
    uint elem = thread_position_in_grid.x;
    int B = dims[0];
    int H = dims[1];
    int W = dims[2];
    int C = dims[3];
    int K = dims[4];
    int S = dims[5];
    int P = dims[6];
    int OH = dims[7];
    int OW = dims[8];
    int total = B * OH * OW * C;
    if (elem >= (uint)total) return;

    int c = int(elem) % C;
    int ow = (int(elem) / C) % OW;
    int oh = (int(elem) / (C * OW)) % OH;
    int b = int(elem) / (C * OW * OH);
    float acc = 0.0f;
    for (int kh = 0; kh < K; ++kh) {
        int ih = oh * S + kh - P;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < K; ++kw) {
            int iw = ow * S + kw - P;
            if (iw < 0 || iw >= W) continue;
            int xidx = ((b * H + ih) * W + iw) * C + c;
            int widx = (c * K + kh) * K + kw;
            acc = acc + float(x[xidx]) * float(w[widx]);
        }
    }
    out[elem] = acc;
"""

_POINTWISE_KERNELS: dict[str, Any] = {}
_DEPTHWISE_KERNEL: Any | None = None


def _pointwise_kernel(variant: PointwiseWeightVariant) -> Any:
    kernel = _POINTWISE_KERNELS.get(variant)
    if kernel is None:
        import mlx.core as mx

        kernel = mx.fast.metal_kernel(
            name=f"segnet_pointwise_1x1_{variant}_mma8x8",
            input_names=["x", "w", "scales", "dims"],
            output_names=["out"],
            header=_POINTWISE_HEADER,
            source=_pointwise_source(variant),
        )
        _POINTWISE_KERNELS[variant] = kernel
    return kernel


def _depthwise_kernel() -> Any:
    global _DEPTHWISE_KERNEL
    if _DEPTHWISE_KERNEL is None:
        import mlx.core as mx

        _DEPTHWISE_KERNEL = mx.fast.metal_kernel(
            name="segnet_depthwise_fp16_accum_fp32",
            input_names=["x", "w", "dims"],
            output_names=["out"],
            source=_DEPTHWISE_SOURCE,
        )
    return _DEPTHWISE_KERNEL


@lru_cache(maxsize=1)
def metal_segnet_conv_backend_available() -> bool:
    """Probe a real evaluated MLX Metal allocation; device labels alone do not count."""

    try:
        import mlx.core as mx

        if mx.default_device().type != mx.gpu:
            return False
        probe = mx.sum(mx.arange(8, dtype=mx.float32))
        mx.eval(probe)
        return float(probe.item()) == 28.0
    except Exception:
        return False


def custom_segnet_conv_env_requested() -> bool:
    return os.environ.get(CUSTOM_SEGNET_CONV_FLAG, "").strip().lower() in _TRUTHY


def custom_segnet_conv_enabled() -> bool:
    """True only when the explicit opt-in and a working Metal device both exist."""

    return custom_segnet_conv_env_requested() and metal_segnet_conv_backend_available()


def _normalize_variant(value: str | None) -> PointwiseWeightVariant:
    variant = (
        value
        if value is not None
        else os.environ.get(CUSTOM_SEGNET_CONV_WEIGHT_VARIANT_FLAG, "fp16")
    )
    variant = str(variant).strip().lower()
    if variant not in VALID_POINTWISE_WEIGHT_VARIANTS:
        raise ValueError(
            f"pointwise variant must be one of {sorted(VALID_POINTWISE_WEIGHT_VARIANTS)}, "
            f"got {variant!r}"
        )
    return variant  # type: ignore[return-value]


def pointwise_1x1_metal(
    x_nhwc: Any,
    weight: Any,
    *,
    variant: str = "fp16",
    scales: Any | None = None,
    cout: int | None = None,
) -> Any:
    """Run the forward-only pointwise Metal kernel and return NHWC fp32."""

    import mlx.core as mx

    normalized = _normalize_variant(variant)
    if not metal_segnet_conv_backend_available():
        raise RuntimeError("custom SegNet pointwise kernel requires an evaluated Metal device")
    if len(x_nhwc.shape) != 4:
        raise ValueError(f"x_nhwc must be rank-4, got {x_nhwc.shape}")
    batch, height, width, cin = map(int, x_nhwc.shape)
    if normalized == "fp16":
        if len(weight.shape) != 2 or int(weight.shape[0]) != cin:
            raise ValueError(f"fp16 weight must be ({cin}, Cout), got {weight.shape}")
        nout = int(weight.shape[1])
        runtime_weight = weight.astype(mx.float16)
        runtime_scales = mx.ones((nout,), dtype=mx.float16)
    else:
        if cout is None or scales is None:
            raise ValueError(f"{normalized} requires cout and scales")
        nout = int(cout)
        runtime_weight = weight
        runtime_scales = scales.astype(mx.float32)
        expected = cin * nout if normalized == "int8" else (cin * nout + 1) // 2
        if int(runtime_weight.size) != expected:
            raise ValueError(
                f"{normalized} weight payload has {runtime_weight.size} values, expected {expected}"
            )
    m = batch * height * width
    dims = mx.array([m, cin, nout], dtype=mx.int32)
    (out,) = _pointwise_kernel(normalized)(
        inputs=[x_nhwc.astype(mx.float16), runtime_weight, runtime_scales, dims],
        output_shapes=[(batch, height, width, nout)],
        output_dtypes=[mx.float32],
        grid=(((nout + 7) // 8) * 32, (m + 7) // 8, 1),
        threadgroup=(32, 1, 1),
    )
    return out


def depthwise_conv2d_metal(
    x_nhwc: Any,
    weight_ckk1: Any,
    *,
    stride: int = 1,
    padding: int | None = None,
) -> Any:
    """Run the forward-only fp16-traffic/fp32-accumulate depthwise kernel."""

    import mlx.core as mx

    if not metal_segnet_conv_backend_available():
        raise RuntimeError("custom SegNet depthwise kernel requires an evaluated Metal device")
    if len(x_nhwc.shape) != 4 or len(weight_ckk1.shape) != 4:
        raise ValueError(
            f"expected NHWC x and CxKxKx1 weight, got {x_nhwc.shape}, {weight_ckk1.shape}"
        )
    batch, height, width, channels = map(int, x_nhwc.shape)
    w_channels, kernel_h, kernel_w, multiplier = map(int, weight_ckk1.shape)
    if (
        w_channels != channels
        or multiplier != 1
        or kernel_h != kernel_w
        or kernel_h not in (3, 5)
    ):
        raise ValueError(
            f"unsupported depthwise weight {weight_ckk1.shape} for x {x_nhwc.shape}"
        )
    stride = int(stride)
    if stride not in (1, 2):
        raise ValueError(f"stride must be 1 or 2, got {stride}")
    pad = kernel_h // 2 if padding is None else int(padding)
    out_h = (height + 2 * pad - kernel_h) // stride + 1
    out_w = (width + 2 * pad - kernel_w) // stride + 1
    dims = mx.array(
        [batch, height, width, channels, kernel_h, stride, pad, out_h, out_w],
        dtype=mx.int32,
    )
    out_size = batch * out_h * out_w * channels
    (out,) = _depthwise_kernel()(
        inputs=[x_nhwc.astype(mx.float16), weight_ckk1.astype(mx.float16), dims],
        output_shapes=[(batch, out_h, out_w, channels)],
        output_dtypes=[mx.float32],
        grid=(out_size, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


class MetalSegNetConv2DAdapter:
    """Frozen-weight forward adapter for one eligible Torch Conv2d layer."""

    supports_vjp = False

    def __init__(self, torch_conv: Any, *, pointwise_variant: str = "fp16") -> None:
        import mlx.core as mx

        self.variant = _normalize_variant(pointwise_variant)
        self.stride = tuple(map(int, torch_conv.stride))
        self.padding = tuple(map(int, torch_conv.padding))
        self.dilation = tuple(map(int, torch_conv.dilation))
        self.groups = int(torch_conv.groups)
        weight_oihw = np.asarray(torch_conv.weight.detach().cpu(), dtype=np.float32)
        self.bias = (
            None
            if torch_conv.bias is None
            else mx.array(np.asarray(torch_conv.bias.detach().cpu(), dtype=np.float32))
        )
        kh, kw = map(int, torch_conv.kernel_size)
        self.is_pointwise = (
            (kh, kw) == (1, 1)
            and self.groups == 1
            and self.stride == (1, 1)
            and self.dilation == (1, 1)
        )
        self.is_depthwise = (
            kh == kw
            and kh in (3, 5)
            and self.groups == int(torch_conv.in_channels)
            and int(torch_conv.out_channels) == int(torch_conv.in_channels)
            and self.dilation == (1, 1)
            and self.stride[0] == self.stride[1]
            and self.padding[0] == self.padding[1]
        )
        if not self.is_pointwise and not self.is_depthwise:
            raise ValueError("MetalSegNetConv2DAdapter only accepts eligible pointwise/depthwise conv")
        self.cout = int(torch_conv.out_channels)
        self.scales = None
        if self.is_pointwise:
            weight_kn = _pointwise_weight_kn_from_oihw(weight_oihw)
            if self.variant == "fp16":
                self.weight = mx.array(weight_kn, dtype=mx.float16)
            elif self.variant == "int8":
                packet = quantize_pointwise_int8(weight_kn)
                self.weight = mx.array(packet.values, dtype=mx.int8)
                self.scales = mx.array(packet.scales, dtype=mx.float32)
            else:
                packet = quantize_pointwise_int4(weight_kn)
                self.weight = mx.array(packet.values, dtype=mx.uint8)
                self.scales = mx.array(packet.scales, dtype=mx.float32)
        else:
            weight_ckk1 = np.ascontiguousarray(weight_oihw.transpose(0, 2, 3, 1))
            self.weight = mx.array(weight_ckk1, dtype=mx.float16)

    def __call__(self, x_nhwc: Any) -> Any:
        if self.is_pointwise:
            out = pointwise_1x1_metal(
                x_nhwc,
                self.weight,
                variant=self.variant,
                scales=self.scales,
                cout=self.cout,
            )
        else:
            out = depthwise_conv2d_metal(
                x_nhwc,
                self.weight,
                stride=self.stride[0],
                padding=self.padding[0],
            )
        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, -1)
        return out


_ADAPTER_BUILD_LOCK = threading.Lock()


def build_custom_metal_segnet_adapter(
    torch_segnet: Any,
    *,
    pointwise_variant: str | None = None,
    require_opt_in: bool = True,
) -> Any:
    """Build an MLX SegNet with eligible convs replaced, restoring globals safely.

    When the flag is OFF and ``require_opt_in`` is true, this returns the native
    adapter without touching the converter.  When the flag is requested but the
    evaluated Metal probe fails, it raises instead of silently falling back.
    """

    from tac.local_acceleration import mlx_scorer_adapters as adapters

    if require_opt_in and not custom_segnet_conv_env_requested():
        return adapters.torch_segnet_to_mlx(torch_segnet)
    if not metal_segnet_conv_backend_available():
        raise RuntimeError(
            f"{CUSTOM_SEGNET_CONV_FLAG} requested but no evaluated Metal device is available"
        )
    variant = _normalize_variant(pointwise_variant)
    original = adapters.torch_conv2d_to_mlx

    def convert(torch_conv: Any) -> Any:
        kh, kw = map(int, torch_conv.kernel_size)
        pointwise = (
            (kh, kw) == (1, 1)
            and int(torch_conv.groups) == 1
            and tuple(map(int, torch_conv.stride)) == (1, 1)
            and tuple(map(int, torch_conv.dilation)) == (1, 1)
        )
        depthwise = (
            kh == kw
            and kh in (3, 5)
            and int(torch_conv.groups) == int(torch_conv.in_channels)
            and int(torch_conv.out_channels) == int(torch_conv.in_channels)
            and tuple(map(int, torch_conv.dilation)) == (1, 1)
        )
        if pointwise or depthwise:
            return MetalSegNetConv2DAdapter(
                torch_conv,
                pointwise_variant=variant,
            )
        return original(torch_conv)

    with _ADAPTER_BUILD_LOCK:
        adapters.torch_conv2d_to_mlx = convert
        try:
            return adapters.torch_segnet_to_mlx(torch_segnet)
        finally:
            adapters.torch_conv2d_to_mlx = original


def custom_segnet_conv_signature() -> dict[str, Any]:
    """Machine-readable suite registration and promotion boundary."""

    return {
        "schema": "custom_metal_segnet_conv_signature_v1",
        "built": True,
        "research_only": True,
        "axis": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "default_enabled": False,
        "env_flag": CUSTOM_SEGNET_CONV_FLAG,
        "pointwise_weight_variant_env": CUSTOM_SEGNET_CONV_WEIGHT_VARIANT_FLAG,
        "pointwise": {
            "operation": "implicit-gemm-nhwc-1x1",
            "tile": [8, 8, 8],
            "arithmetic": "simdgroup-half-input-multiply-float-destination-accumulate",
            "weight_variants": sorted(VALID_POINTWISE_WEIGHT_VARIANTS),
            "int8": "symmetric-per-output-channel-dequant-on-load",
            "int4": "packed-symmetric-per-output-channel-dequant-on-load",
        },
        "depthwise": {
            "kernels": [3, 5],
            "strides": [1, 2],
            "arithmetic": "fp16-load-fp32-sequential-accumulate",
        },
        "determinism": "fixed traversal; no atomics",
        "vjp": "not-implemented-forward-only-fail-closed",
        "numpy_fp32_reference": True,
        "promotion_gate": N600_FIDELITY_GATE,
        "verdict_scope": "frozen-SegNet local forward throughput formulation only",
    }
