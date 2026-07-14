# SPDX-License-Identifier: MIT
"""Custom-Metal twin of the geometry-safe W26..W30 int64 SegNet map."""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

from tac.local_acceleration.metal_fixedpoint_verdict import (
    METAL_FIXEDPOINT_VERDICT_FLAG,
    FixedPointConvPacket,
    FixedPointMetalConstants,
    fixedpoint_conv2d_metal,
    metal_fixedpoint_backend_available,
    minimum_signed_bits_for_bound,
    prepare_fixedpoint_conv_packet_metal,
)
from tac.local_acceleration.mixed_int64_fixedpoint_scorer import (
    MAXIMUM_BITS,
    MINIMUM_BITS,
    MixedInt64Conv2d,
    maximum_safe_bits,
)
from tac.local_acceleration.weight_l1_int64_fixedpoint_scorer import (
    MAXIMUM_WEIGHT_L1_BITS,
    WeightL1Int64Conv2d,
    maximum_weight_l1_safe_bits,
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_ADAPTER_LOCK = threading.Lock()


def build_mixed_fixedpoint_conv_packet(torch_conv: Any, *, bits: int) -> FixedPointConvPacket:
    cpu_twin = MixedInt64Conv2d(torch_conv, bits=bits).eval()
    weight_oihw = cpu_twin.weight_q.detach().cpu().numpy().astype(np.int32, copy=False)
    kernel_h, kernel_w = map(int, torch_conv.kernel_size)
    qmax = int(cpu_twin.qmax)
    return FixedPointConvPacket(
        bits=int(bits),
        qmax=qmax,
        activation_absmax=1.0,
        activation_scale=1.0 / float(qmax),
        activation_scale_mode="dynamic_exact_absmax",
        weight_q_ohwi=np.ascontiguousarray(weight_oihw.transpose(0, 2, 3, 1)),
        weight_scales=np.ascontiguousarray(
            cpu_twin.weight_scales.detach().cpu().numpy().astype(np.float32, copy=False)
        ),
        bias=np.ascontiguousarray(cpu_twin.bias_fp32.detach().cpu().numpy().astype(np.float32, copy=False)),
        stride=tuple(map(int, torch_conv.stride)),
        padding=tuple(map(int, torch_conv.padding)),
        dilation=tuple(map(int, torch_conv.dilation)),
        groups=int(torch_conv.groups),
        in_channels=int(torch_conv.in_channels),
        out_channels=int(torch_conv.out_channels),
        kernel_hw=(kernel_h, kernel_w),
        accumulator_bound=int(cpu_twin.accumulator_bound),
        minimum_signed_accumulator_bits=minimum_signed_bits_for_bound(cpu_twin.accumulator_bound),
    )


def build_weight_l1_fixedpoint_conv_packet(
    torch_conv: Any,
    *,
    bits: int,
) -> FixedPointConvPacket:
    cpu_twin = WeightL1Int64Conv2d(torch_conv, bits=bits).eval()
    weight_oihw = (
        cpu_twin.weight_q.detach().cpu().numpy().astype(np.int32, copy=False)
    )
    kernel_h, kernel_w = map(int, torch_conv.kernel_size)
    qmax = int(cpu_twin.qmax)
    return FixedPointConvPacket(
        bits=int(bits),
        qmax=qmax,
        activation_absmax=1.0,
        activation_scale=1.0 / float(qmax),
        activation_scale_mode="dynamic_exact_absmax",
        weight_q_ohwi=np.ascontiguousarray(weight_oihw.transpose(0, 2, 3, 1)),
        weight_scales=np.ascontiguousarray(
            cpu_twin.weight_scales.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        ),
        bias=np.ascontiguousarray(
            cpu_twin.bias_fp32.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        ),
        stride=tuple(map(int, torch_conv.stride)),
        padding=tuple(map(int, torch_conv.padding)),
        dilation=tuple(map(int, torch_conv.dilation)),
        groups=int(torch_conv.groups),
        in_channels=int(torch_conv.in_channels),
        out_channels=int(torch_conv.out_channels),
        kernel_hw=(kernel_h, kernel_w),
        accumulator_bound=int(cpu_twin.accumulator_bound),
        minimum_signed_accumulator_bits=minimum_signed_bits_for_bound(
            cpu_twin.accumulator_bound
        ),
    )


class MetalMixedInt64Conv2DAdapter:
    supports_vjp = False

    def __init__(self, torch_conv: Any, *, bits: int) -> None:
        self.packet = build_mixed_fixedpoint_conv_packet(torch_conv, bits=bits)
        self.constants = prepare_fixedpoint_conv_packet_metal(self.packet)

    def __call__(self, value: Any) -> Any:
        return fixedpoint_conv2d_metal(
            value,
            self.packet,
            constants=self.constants,
        )


class MetalWeightL1Int64Conv2DAdapter:
    supports_vjp = False

    def __init__(self, torch_conv: Any, *, bits: int) -> None:
        self.packet = build_weight_l1_fixedpoint_conv_packet(torch_conv, bits=bits)
        self.constants = prepare_fixedpoint_conv_packet_metal(self.packet)

    def __call__(self, value: Any) -> Any:
        return fixedpoint_conv2d_metal(
            value,
            self.packet,
            constants=self.constants,
        )


def derive_mixed_precision_map(torch_segnet: Any) -> dict[str, int]:
    import torch

    return {
        name: maximum_safe_bits(
            module,
            minimum_bits=MINIMUM_BITS,
            maximum_bits=MAXIMUM_BITS,
        )
        for name, module in torch_segnet.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    }


def derive_weight_l1_precision_map(torch_segnet: Any) -> dict[str, int]:
    import torch

    return {
        name: maximum_weight_l1_safe_bits(
            module,
            minimum_bits=MINIMUM_BITS,
            maximum_bits=MAXIMUM_WEIGHT_L1_BITS,
        )
        for name, module in torch_segnet.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    }


def build_metal_mixed_int64_segnet_adapter(
    torch_segnet: Any,
    *,
    operator_absmax: dict[str, float],
    require_opt_in: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Convert all SegNet Conv2d using the geometry-derived W26..W30 map."""

    import torch

    from tac.local_acceleration import mlx_scorer_adapters as adapters

    if require_opt_in and os.environ.get(METAL_FIXEDPOINT_VERDICT_FLAG, "").strip().lower() not in _TRUTHY:
        raise RuntimeError(f"set {METAL_FIXEDPOINT_VERDICT_FLAG}=1 to request this backend")
    if not metal_fixedpoint_backend_available():
        raise RuntimeError("mixed fixed-point SegNet requested without evaluated Metal")
    modules = {id(module): name for name, module in torch_segnet.named_modules() if isinstance(module, torch.nn.Conv2d)}
    expected = set(modules.values())
    if set(operator_absmax) != expected:
        raise ValueError("calibration/SegNet operator set differs for mixed fixed-point")
    precision = derive_mixed_precision_map(torch_segnet)
    consumed: list[str] = []
    packets: list[FixedPointConvPacket] = []
    constants: list[FixedPointMetalConstants] = []
    original_converter = adapters.torch_conv2d_to_mlx
    original_explicit = adapters.MLXExplicitSpatialConv2dAdapter

    def convert(torch_conv: Any) -> Any:
        path = modules.get(id(torch_conv))
        if path is None:
            raise RuntimeError("unregistered Conv2d reached mixed fixed-point converter")
        packet = build_mixed_fixedpoint_conv_packet(torch_conv, bits=precision[path])
        packets.append(packet)
        consumed.append(path)
        adapter = MetalMixedInt64Conv2DAdapter.__new__(MetalMixedInt64Conv2DAdapter)
        adapter.packet = packet
        adapter.constants = prepare_fixedpoint_conv_packet_metal(packet)
        constants.append(adapter.constants)
        return adapter

    with _ADAPTER_LOCK:
        adapters.torch_conv2d_to_mlx = convert
        adapters.MLXExplicitSpatialConv2dAdapter = convert
        try:
            converted = adapters.torch_segnet_to_mlx(torch_segnet)
        finally:
            adapters.torch_conv2d_to_mlx = original_converter
            adapters.MLXExplicitSpatialConv2dAdapter = original_explicit
    if set(consumed) != expected or len(consumed) != len(expected):
        raise RuntimeError("mixed fixed-point conversion coverage failed")
    if len(constants) != len(expected):
        raise RuntimeError("mixed fixed-point constant-buffer cache coverage failed")
    histogram: dict[str, int] = {}
    for bits in precision.values():
        histogram[str(bits)] = histogram.get(str(bits), 0) + 1
    return converted, {
        "schema": "metal_mixed_int64_segnet_adapter.v1",
        "minimum_bits": MINIMUM_BITS,
        "maximum_bits": MAXIMUM_BITS,
        "assignment_rule": "largest_geometry_safe_bits_with_signed_int64_static_bound",
        "precision_by_path": dict(sorted(precision.items())),
        "precision_histogram": dict(sorted(histogram.items())),
        "operator_count": len(consumed),
        "all_convs_replaced": True,
        "activation_scale_mode": "dynamic_exact_absmax",
        "arithmetic": "integer activation/weight; exact int64 MAC; fp32 dequant+bias",
        "constant_buffers_cached": True,
        "maximum_accumulator_bound": max(packet.accumulator_bound for packet in packets),
        "default_enabled": False,
        "score_claim": False,
        "promotion_gate": "real n600 exact argmax + cross-process digest + positive latency",
    }


def mixed_fixedpoint_verdict_signature() -> dict[str, Any]:
    return {
        "schema": "metal_mixed_int64_verdict_signature.v1",
        "built": True,
        "default_enabled": False,
        "operation": "frozen-SegNet all-Conv2d geometry-safe mixed fixed-point forward",
        "minimum_bits": MINIMUM_BITS,
        "maximum_bits": MAXIMUM_BITS,
        "assignment_rule": "largest_geometry_safe_bits_with_signed_int64_static_bound",
        "kernel": "direct NHWC grouped Conv2d; exact int64 accumulator",
        "constant_buffers_cached": True,
        "activation_scale_mode": "dynamic_exact_absmax",
        "numpy_cpu_integer_twin": True,
        "score_claim": False,
    }


def build_metal_weight_l1_int64_segnet_adapter(
    torch_segnet: Any,
    *,
    operator_absmax: dict[str, float],
    require_opt_in: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Convert all SegNet Conv2d using frozen-weight-L1-safe W26..W31."""

    import torch

    from tac.local_acceleration import mlx_scorer_adapters as adapters

    if (
        require_opt_in
        and os.environ.get(METAL_FIXEDPOINT_VERDICT_FLAG, "").strip().lower()
        not in _TRUTHY
    ):
        raise RuntimeError(f"set {METAL_FIXEDPOINT_VERDICT_FLAG}=1 to request this backend")
    if not metal_fixedpoint_backend_available():
        raise RuntimeError("weight-L1 fixed-point SegNet requested without evaluated Metal")
    modules = {
        id(module): name
        for name, module in torch_segnet.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    }
    expected = set(modules.values())
    if set(operator_absmax) != expected:
        raise ValueError("calibration/SegNet operator set differs for weight-L1 fixed-point")
    precision = derive_weight_l1_precision_map(torch_segnet)
    consumed: list[str] = []
    packets: list[FixedPointConvPacket] = []
    constants: list[FixedPointMetalConstants] = []
    original_converter = adapters.torch_conv2d_to_mlx
    original_explicit = adapters.MLXExplicitSpatialConv2dAdapter

    def convert(torch_conv: Any) -> Any:
        path = modules.get(id(torch_conv))
        if path is None:
            raise RuntimeError("unregistered Conv2d reached weight-L1 converter")
        packet = build_weight_l1_fixedpoint_conv_packet(
            torch_conv,
            bits=precision[path],
        )
        packets.append(packet)
        consumed.append(path)
        adapter = MetalWeightL1Int64Conv2DAdapter.__new__(
            MetalWeightL1Int64Conv2DAdapter
        )
        adapter.packet = packet
        adapter.constants = prepare_fixedpoint_conv_packet_metal(packet)
        constants.append(adapter.constants)
        return adapter

    with _ADAPTER_LOCK:
        adapters.torch_conv2d_to_mlx = convert
        adapters.MLXExplicitSpatialConv2dAdapter = convert
        try:
            converted = adapters.torch_segnet_to_mlx(torch_segnet)
        finally:
            adapters.torch_conv2d_to_mlx = original_converter
            adapters.MLXExplicitSpatialConv2dAdapter = original_explicit
    if set(consumed) != expected or len(consumed) != len(expected):
        raise RuntimeError("weight-L1 fixed-point conversion coverage failed")
    if len(constants) != len(expected):
        raise RuntimeError("weight-L1 fixed-point constant-buffer cache coverage failed")
    histogram: dict[str, int] = {}
    for bits in precision.values():
        histogram[str(bits)] = histogram.get(str(bits), 0) + 1
    return converted, {
        "schema": "metal_weight_l1_int64_segnet_adapter.v1",
        "minimum_bits": MINIMUM_BITS,
        "maximum_bits": MAXIMUM_WEIGHT_L1_BITS,
        "assignment_rule": (
            "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
        ),
        "bound_kind": (
            "activation_qmax_times_max_output_quantized_weight_l1"
        ),
        "precision_by_path": dict(sorted(precision.items())),
        "precision_histogram": dict(sorted(histogram.items())),
        "operator_count": len(consumed),
        "all_convs_replaced": True,
        "activation_scale_mode": "dynamic_exact_absmax",
        "arithmetic": "integer activation/weight; exact int64 MAC; fp32 dequant+bias",
        "constant_buffers_cached": True,
        "maximum_accumulator_bound": max(
            packet.accumulator_bound for packet in packets
        ),
        "label_or_frame_dependent": False,
        "default_enabled": False,
        "score_claim": False,
        "promotion_gate": (
            "real n600 exact argmax + cross-process digest + positive latency"
        ),
    }


def weight_l1_fixedpoint_verdict_signature() -> dict[str, Any]:
    return {
        "schema": "metal_weight_l1_int64_verdict_signature.v1",
        "built": True,
        "default_enabled": False,
        "operation": "frozen-SegNet all-Conv2d weight-L1-safe fixed-point forward",
        "minimum_bits": MINIMUM_BITS,
        "maximum_bits": MAXIMUM_WEIGHT_L1_BITS,
        "assignment_rule": (
            "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
        ),
        "bound_kind": (
            "activation_qmax_times_max_output_quantized_weight_l1"
        ),
        "kernel": "direct NHWC grouped Conv2d; exact int64 accumulator",
        "constant_buffers_cached": True,
        "activation_scale_mode": "dynamic_exact_absmax",
        "numpy_cpu_integer_twin": True,
        "label_or_frame_dependent": False,
        "score_claim": False,
    }


__all__ = [
    "MetalMixedInt64Conv2DAdapter",
    "MetalWeightL1Int64Conv2DAdapter",
    "build_metal_mixed_int64_segnet_adapter",
    "build_metal_weight_l1_int64_segnet_adapter",
    "build_mixed_fixedpoint_conv_packet",
    "build_weight_l1_fixedpoint_conv_packet",
    "derive_mixed_precision_map",
    "derive_weight_l1_precision_map",
    "mixed_fixedpoint_verdict_signature",
    "weight_l1_fixedpoint_verdict_signature",
]
