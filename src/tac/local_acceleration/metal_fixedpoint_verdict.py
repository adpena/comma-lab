# SPDX-License-Identifier: MIT
"""Custom Metal fixed-point Conv2d backend for frozen-SegNet verdict probes.

Each layer uses either a calibration-fixed or label-free dynamic max-absolute
activation scale, symmetric per-output-channel integer weights, exact signed
int64 multiply-accumulate, and one deterministic float dequantization/bias
finalization.  The direct kernel covers every frozen SegNet Conv2d geometry
(dense/grouped/depthwise, stride, padding, dilation) and is default OFF.  It is
a candidate local verdict backend, never contest authority without full
n600/cross-process/placement receipts.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from tac.local_acceleration.calibrated_fixedpoint_scorer import qmax_for_bits

METAL_FIXEDPOINT_VERDICT_FLAG = "TAC_MLX_FIXEDPOINT_VERDICT"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class FixedPointConvPacket:
    bits: int
    qmax: int
    activation_absmax: float
    activation_scale: float
    activation_scale_mode: str
    weight_q_ohwi: np.ndarray
    weight_scales: np.ndarray
    bias: np.ndarray
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    in_channels: int
    out_channels: int
    kernel_hw: tuple[int, int]
    accumulator_bound: int
    minimum_signed_accumulator_bits: int


@dataclass(frozen=True)
class FixedPointMetalConstants:
    """One-time device-resident buffers shared by every invocation of a layer."""

    weight_q_ohwi: Any
    weight_scales: Any
    bias: Any


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"expected pair, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def minimum_signed_bits_for_bound(bound: int) -> int:
    bound = int(bound)
    if bound < 0:
        raise ValueError("bound must be non-negative")
    return max(1, int(np.ceil(np.log2(2 * bound + 1)))) if bound else 1


def quantize_activation_numpy(
    value: Any, *, activation_scale: float, qmax: int
) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    scale = float(activation_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("activation_scale must be finite and positive")
    scaled = (source / np.float32(scale)).astype(np.float32)
    rounded = np.rint(scaled).astype(np.int64)
    return np.clip(rounded, -int(qmax), int(qmax)).astype(np.int32)


def build_fixedpoint_conv_packet(
    torch_conv: Any,
    *,
    activation_absmax: float,
    bits: int,
    activation_scale_mode: str = "fixed_calibration",
) -> FixedPointConvPacket:
    qmax = qmax_for_bits(bits)
    maximum = float(activation_absmax)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError("activation_absmax must be finite and non-negative")
    if activation_scale_mode not in {"fixed_calibration", "dynamic_exact_absmax"}:
        raise ValueError("unsupported activation_scale_mode")
    activation_scale = maximum / float(qmax) if maximum > 0.0 else 1.0
    weight = np.asarray(torch_conv.weight.detach().cpu(), dtype=np.float32)
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be OIHW, got {weight.shape}")
    reduce_axes = (1, 2, 3)
    weight_maximum = np.max(np.abs(weight), axis=reduce_axes).astype(np.float32)
    weight_scales = np.where(
        weight_maximum > 0.0,
        weight_maximum / np.float32(qmax),
        np.float32(1.0),
    ).astype(np.float32)
    rounded_weight = np.rint(
        weight / weight_scales[:, None, None, None]
    ).astype(np.int64)
    quantized = np.clip(rounded_weight, -int(qmax), int(qmax)).astype(np.int32)
    out_channels, in_per_group, kernel_h, kernel_w = map(int, weight.shape)
    groups = int(torch_conv.groups)
    in_channels = int(torch_conv.in_channels)
    if in_channels != groups * in_per_group or out_channels % groups:
        raise ValueError("invalid grouped Conv2d geometry")
    fan_in = in_per_group * kernel_h * kernel_w
    accumulator_bound = fan_in * qmax * qmax
    minimum_bits = minimum_signed_bits_for_bound(accumulator_bound)
    if accumulator_bound > np.iinfo(np.int64).max:
        raise OverflowError(
            f"W{bits}A{bits} Conv2d requires >int64: bound={accumulator_bound}"
        )
    bias = (
        np.zeros((out_channels,), dtype=np.float32)
        if torch_conv.bias is None
        else np.asarray(torch_conv.bias.detach().cpu(), dtype=np.float32).reshape(out_channels)
    )
    return FixedPointConvPacket(
        bits=int(bits),
        qmax=qmax,
        activation_absmax=maximum,
        activation_scale=float(activation_scale),
        activation_scale_mode=activation_scale_mode,
        weight_q_ohwi=np.ascontiguousarray(quantized.transpose(0, 2, 3, 1)),
        weight_scales=np.ascontiguousarray(weight_scales),
        bias=np.ascontiguousarray(bias),
        stride=_pair(torch_conv.stride),
        padding=_pair(torch_conv.padding),
        dilation=_pair(torch_conv.dilation),
        groups=groups,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_hw=(kernel_h, kernel_w),
        accumulator_bound=int(accumulator_bound),
        minimum_signed_accumulator_bits=minimum_bits,
    )


def _activation_scale_numpy(value: Any, packet: FixedPointConvPacket) -> float:
    if packet.activation_scale_mode == "fixed_calibration":
        return packet.activation_scale
    source = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(source)):
        raise ValueError("dynamic fixed-point activation contains non-finite values")
    maximum = float(np.max(np.abs(source))) if source.size else 0.0
    return maximum / float(packet.qmax) if maximum > 0.0 else 1.0


def fixedpoint_conv2d_numpy(x_nhwc: Any, packet: FixedPointConvPacket) -> np.ndarray:
    activation_scale = _activation_scale_numpy(x_nhwc, packet)
    xq = quantize_activation_numpy(
        x_nhwc, activation_scale=activation_scale, qmax=packet.qmax
    )
    if xq.ndim != 4 or int(xq.shape[-1]) != packet.in_channels:
        raise ValueError(f"expected NHWC Cin={packet.in_channels}, got {xq.shape}")
    batch, height, width, _ = map(int, xq.shape)
    kernel_h, kernel_w = packet.kernel_hw
    stride_h, stride_w = packet.stride
    pad_h, pad_w = packet.padding
    dilation_h, dilation_w = packet.dilation
    out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    output = np.zeros((batch, out_h, out_w, packet.out_channels), dtype=np.float32)
    out_per_group = packet.out_channels // packet.groups
    in_per_group = packet.in_channels // packet.groups
    for b in range(batch):
        for oh in range(out_h):
            for ow in range(out_w):
                for oc in range(packet.out_channels):
                    group = oc // out_per_group
                    accumulator = np.int64(0)
                    for kh in range(kernel_h):
                        ih = oh * stride_h + kh * dilation_h - pad_h
                        if ih < 0 or ih >= height:
                            continue
                        for kw in range(kernel_w):
                            iw = ow * stride_w + kw * dilation_w - pad_w
                            if iw < 0 or iw >= width:
                                continue
                            for channel in range(in_per_group):
                                ic = group * in_per_group + channel
                                accumulator += np.int64(xq[b, ih, iw, ic]) * np.int64(
                                    packet.weight_q_ohwi[oc, kh, kw, channel]
                                )
                    output[b, oh, ow, oc] = np.float32(
                        float(accumulator)
                        * activation_scale
                        * float(packet.weight_scales[oc])
                        + float(packet.bias[oc])
                    )
    return output


_QUANTIZE_KERNEL: Any | None = None
_CONV_KERNEL: Any | None = None
_KERNEL_LOCK = threading.Lock()


def _quantize_kernel() -> Any:
    global _QUANTIZE_KERNEL
    with _KERNEL_LOCK:
        if _QUANTIZE_KERNEL is None:
            import mlx.core as mx

            _QUANTIZE_KERNEL = mx.fast.metal_kernel(
                name="fixedpoint_verdict_quantize_activation_i32",
                input_names=["inp", "dims", "scale", "qmax_value"],
                output_names=["out"],
                source="""
                    uint gid = thread_position_in_grid.x;
                    int count = dims[0];
                    float activation_scale = scale[0];
                    int qmax = qmax_value[0];
                    if (gid >= (uint)count) return;
                    int q = int(rint(inp[gid] / activation_scale));
                    out[gid] = clamp(q, -qmax, qmax);
                """,
            )
    return _QUANTIZE_KERNEL


def _conv_kernel() -> Any:
    global _CONV_KERNEL
    with _KERNEL_LOCK:
        if _CONV_KERNEL is None:
            import mlx.core as mx

            _CONV_KERNEL = mx.fast.metal_kernel(
                name="fixedpoint_verdict_conv2d_i64_accum",
                input_names=[
                    "xq",
                    "wq",
                    "wscale",
                    "bias",
                    "dims",
                    "activation_scale",
                ],
                output_names=["out"],
                source="""
                    uint gid = thread_position_in_grid.x;
                    int B = dims[0];
                    int H = dims[1];
                    int W = dims[2];
                    int Cin = dims[3];
                    int Cout = dims[4];
                    int KH = dims[5];
                    int KW = dims[6];
                    int SH = dims[7];
                    int SW = dims[8];
                    int PH = dims[9];
                    int PW = dims[10];
                    int DH = dims[11];
                    int DW = dims[12];
                    int groups = dims[13];
                    int OH = dims[14];
                    int OW = dims[15];
                    int total = B * OH * OW * Cout;
                    if (gid >= (uint)total) return;
                    int oc = int(gid) % Cout;
                    int ow = (int(gid) / Cout) % OW;
                    int oh = (int(gid) / (Cout * OW)) % OH;
                    int b = int(gid) / (Cout * OW * OH);
                    int out_per_group = Cout / groups;
                    int in_per_group = Cin / groups;
                    int group = oc / out_per_group;
                    long accumulator = 0;
                    for (int kh = 0; kh < KH; ++kh) {
                        int ih = oh * SH + kh * DH - PH;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            int iw = ow * SW + kw * DW - PW;
                            if (iw < 0 || iw >= W) continue;
                            for (int channel = 0; channel < in_per_group; ++channel) {
                                int ic = group * in_per_group + channel;
                                int xidx = ((b * H + ih) * W + iw) * Cin + ic;
                                int widx = ((oc * KH + kh) * KW + kw) * in_per_group + channel;
                                accumulator += long(xq[xidx]) * long(wq[widx]);
                            }
                        }
                    }
                    out[gid] = float(accumulator) * activation_scale[0] * wscale[oc] + bias[oc];
                """,
            )
    return _CONV_KERNEL


@lru_cache(maxsize=1)
def metal_fixedpoint_backend_available() -> bool:
    try:
        import mlx.core as mx

        if mx.default_device().type != mx.gpu:
            return False
        probe = mx.sum(mx.arange(8, dtype=mx.float32))
        mx.eval(probe)
        return float(probe.item()) == 28.0
    except Exception:
        return False


def quantize_activation_metal(
    value: Any, *, activation_scale: Any, qmax: int
) -> Any:
    import mlx.core as mx

    if not metal_fixedpoint_backend_available():
        raise RuntimeError("fixed-point verdict kernel requires evaluated MLX Metal")
    count = int(value.size)
    dims = mx.array([count], dtype=mx.int32)
    scale = (
        activation_scale.astype(mx.float32).reshape((1,))
        if hasattr(activation_scale, "astype")
        else mx.array([float(activation_scale)], dtype=mx.float32)
    )
    qmax_value = mx.array([int(qmax)], dtype=mx.int32)
    (output,) = _quantize_kernel()(
        inputs=[value.astype(mx.float32), dims, scale, qmax_value],
        output_shapes=[value.shape],
        output_dtypes=[mx.int32],
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return output


def prepare_fixedpoint_conv_packet_metal(
    packet: FixedPointConvPacket,
) -> FixedPointMetalConstants:
    """Materialize immutable packet arrays once instead of per forward call."""

    import mlx.core as mx

    if not metal_fixedpoint_backend_available():
        raise RuntimeError("fixed-point verdict constants require evaluated MLX Metal")
    constants = FixedPointMetalConstants(
        weight_q_ohwi=mx.array(packet.weight_q_ohwi, dtype=mx.int32),
        weight_scales=mx.array(packet.weight_scales, dtype=mx.float32),
        bias=mx.array(packet.bias, dtype=mx.float32),
    )
    mx.eval(
        constants.weight_q_ohwi,
        constants.weight_scales,
        constants.bias,
    )
    return constants


def fixedpoint_conv2d_metal(
    x_nhwc: Any,
    packet: FixedPointConvPacket,
    *,
    constants: FixedPointMetalConstants | None = None,
) -> Any:
    import mlx.core as mx

    if not metal_fixedpoint_backend_available():
        raise RuntimeError("fixed-point verdict kernel requires evaluated MLX Metal")
    if len(x_nhwc.shape) != 4 or int(x_nhwc.shape[-1]) != packet.in_channels:
        raise ValueError(f"expected NHWC Cin={packet.in_channels}, got {x_nhwc.shape}")
    batch, height, width, _ = map(int, x_nhwc.shape)
    kernel_h, kernel_w = packet.kernel_hw
    stride_h, stride_w = packet.stride
    pad_h, pad_w = packet.padding
    dilation_h, dilation_w = packet.dilation
    out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if packet.activation_scale_mode == "dynamic_exact_absmax":
        maximum = mx.max(mx.abs(x_nhwc.astype(mx.float32)))
        activation_scale = mx.where(
            maximum > 0.0,
            maximum / float(packet.qmax),
            mx.array(1.0, dtype=mx.float32),
        )
    else:
        activation_scale = mx.array(packet.activation_scale, dtype=mx.float32)
    xq = quantize_activation_metal(
        x_nhwc, activation_scale=activation_scale, qmax=packet.qmax
    )
    device = constants or prepare_fixedpoint_conv_packet_metal(packet)
    if (
        tuple(device.weight_q_ohwi.shape) != tuple(packet.weight_q_ohwi.shape)
        or tuple(device.weight_scales.shape) != tuple(packet.weight_scales.shape)
        or tuple(device.bias.shape) != tuple(packet.bias.shape)
    ):
        raise ValueError("fixed-point Metal constant-buffer geometry differs from packet")
    dims = mx.array(
        [
            batch,
            height,
            width,
            packet.in_channels,
            packet.out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            packet.groups,
            out_h,
            out_w,
        ],
        dtype=mx.int32,
    )
    total = batch * out_h * out_w * packet.out_channels
    (output,) = _conv_kernel()(
        inputs=[
            xq,
            device.weight_q_ohwi,
            device.weight_scales,
            device.bias,
            dims,
            activation_scale.reshape((1,)),
        ],
        output_shapes=[(batch, out_h, out_w, packet.out_channels)],
        output_dtypes=[mx.float32],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return output


class MetalFixedPointConv2DAdapter:
    supports_vjp = False

    def __init__(
        self,
        torch_conv: Any,
        *,
        activation_absmax: float,
        bits: int,
        activation_scale_mode: str,
    ) -> None:
        self.packet = build_fixedpoint_conv_packet(
            torch_conv,
            activation_absmax=activation_absmax,
            bits=bits,
            activation_scale_mode=activation_scale_mode,
        )
        self.constants = prepare_fixedpoint_conv_packet_metal(self.packet)

    def __call__(self, x_nhwc: Any) -> Any:
        return fixedpoint_conv2d_metal(
            x_nhwc,
            self.packet,
            constants=self.constants,
        )


_ADAPTER_LOCK = threading.Lock()


def build_metal_fixedpoint_segnet_adapter(
    torch_segnet: Any,
    *,
    operator_absmax: dict[str, float],
    bits: int,
    activation_scale_mode: str = "fixed_calibration",
    require_opt_in: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Convert all 125 frozen SegNet Conv2d layers to the integer Metal adapter."""

    import torch

    from tac.local_acceleration import mlx_scorer_adapters as adapters

    if (
        require_opt_in
        and os.environ.get(METAL_FIXEDPOINT_VERDICT_FLAG, "").strip().lower()
        not in _TRUTHY
    ):
        raise RuntimeError(f"set {METAL_FIXEDPOINT_VERDICT_FLAG}=1 to request this backend")
    if not metal_fixedpoint_backend_available():
        raise RuntimeError("fixed-point SegNet adapter requested without evaluated Metal")
    modules = {
        id(module): name
        for name, module in torch_segnet.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    }
    expected_paths = set(modules.values())
    if set(operator_absmax) != expected_paths:
        missing = sorted(expected_paths - set(operator_absmax))
        extra = sorted(set(operator_absmax) - expected_paths)
        raise ValueError(f"calibration/SegNet operator mismatch missing={missing} extra={extra}")
    consumed: list[str] = []
    original_converter = adapters.torch_conv2d_to_mlx
    original_explicit = adapters.MLXExplicitSpatialConv2dAdapter

    def convert(torch_conv: Any) -> Any:
        path = modules.get(id(torch_conv))
        if path is None:
            raise RuntimeError("unregistered Conv2d reached fixed-point converter")
        consumed.append(path)
        return MetalFixedPointConv2DAdapter(
            torch_conv,
            activation_absmax=float(operator_absmax[path]),
            bits=bits,
            activation_scale_mode=activation_scale_mode,
        )

    with _ADAPTER_LOCK:
        adapters.torch_conv2d_to_mlx = convert
        adapters.MLXExplicitSpatialConv2dAdapter = convert
        try:
            converted = adapters.torch_segnet_to_mlx(torch_segnet)
        finally:
            adapters.torch_conv2d_to_mlx = original_converter
            adapters.MLXExplicitSpatialConv2dAdapter = original_explicit
    if set(consumed) != expected_paths or len(consumed) != len(expected_paths):
        raise RuntimeError(
            f"fixed-point conversion coverage failed: consumed={len(consumed)} "
            f"unique={len(set(consumed))} expected={len(expected_paths)}"
        )
    packets = [
        build_fixedpoint_conv_packet(
            module,
            activation_absmax=float(operator_absmax[name]),
            bits=bits,
            activation_scale_mode=activation_scale_mode,
        )
        for name, module in torch_segnet.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    manifest = {
        "schema": "metal_fixedpoint_segnet_adapter.v1",
        "bits": int(bits),
        "activation_scale_mode": activation_scale_mode,
        "operator_count": len(consumed),
        "operator_paths": sorted(consumed),
        "all_convs_replaced": True,
        "arithmetic": "integer activation/weight; exact int64 MAC; fp32 dequant+bias",
        "constant_buffers_cached": True,
        "bound_kind": "STATIC_WORST_CASE_FAN_IN_QMAX_PRODUCT",
        "maximum_accumulator_bound": max(packet.accumulator_bound for packet in packets),
        "maximum_minimum_signed_accumulator_bits": max(
            packet.minimum_signed_accumulator_bits for packet in packets
        ),
        "default_enabled": False,
        "score_claim": False,
        "promotion_gate": "real n600 exact argmax + cross-process digest + positive latency",
    }
    return converted, manifest


def fixedpoint_verdict_signature() -> dict[str, Any]:
    return {
        "schema": "metal_fixedpoint_verdict_signature.v1",
        "built": True,
        "default_enabled": False,
        "env_flag": METAL_FIXEDPOINT_VERDICT_FLAG,
        "operation": "frozen-SegNet all-Conv2d fixed-point forward",
        "kernel": "direct NHWC grouped Conv2d; exact int64 accumulator",
        "constant_buffers_cached": True,
        "activation_quantization": (
            "separate int32 kernel; exact integer-domain qmax clamp; fixed calibration "
            "or dynamic max-absolute scale"
        ),
        "activation_scale_modes": ["fixed_calibration", "dynamic_exact_absmax"],
        "supported_bits": list(range(2, 27)),
        "numpy_integer_reference": True,
        "score_claim": False,
        "terminal_authority": "exact contest CPU/CUDA replay until promotion receipt",
        "verdict_scope": "custom Metal frozen-SegNet forward candidate",
    }


__all__ = [
    "METAL_FIXEDPOINT_VERDICT_FLAG",
    "FixedPointConvPacket",
    "FixedPointMetalConstants",
    "MetalFixedPointConv2DAdapter",
    "build_fixedpoint_conv_packet",
    "build_metal_fixedpoint_segnet_adapter",
    "fixedpoint_conv2d_metal",
    "fixedpoint_conv2d_numpy",
    "fixedpoint_verdict_signature",
    "metal_fixedpoint_backend_available",
    "minimum_signed_bits_for_bound",
    "prepare_fixedpoint_conv_packet_metal",
    "quantize_activation_metal",
    "quantize_activation_numpy",
]
