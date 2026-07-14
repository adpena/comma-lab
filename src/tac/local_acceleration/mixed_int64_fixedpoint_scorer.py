# SPDX-License-Identifier: MIT
"""Static overflow-safe per-layer precision for exact-int64 SegNet Conv2d.

Uniform W26 is the largest precision valid for the worst frozen-SegNet fan-in,
but it needlessly constrains smaller layers.  This module assigns each Conv2d
the largest signed precision in a preregistered range whose worst-case
``fan_in * qmax**2`` fits signed int64.  The assignment is derived only from
operator geometry, never labels or measured flips.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

SIGNED_INT64_MAX = (1 << 63) - 1
MINIMUM_BITS = 26
MAXIMUM_BITS = 30


def signed_qmax(bits: int) -> int:
    if bits < 2 or bits > 31:
        raise ValueError("signed int32-backed precision must be in 2..31")
    return (1 << (int(bits) - 1)) - 1


def conv_fan_in(conv: nn.Conv2d) -> int:
    return int(conv.in_channels // conv.groups) * int(conv.kernel_size[0]) * int(conv.kernel_size[1])


def static_accumulator_bound(*, fan_in: int, bits: int) -> int:
    if fan_in <= 0:
        raise ValueError("fan_in must be positive")
    return int(fan_in) * signed_qmax(bits) ** 2


def maximum_safe_bits(
    conv: nn.Conv2d,
    *,
    minimum_bits: int = MINIMUM_BITS,
    maximum_bits: int = MAXIMUM_BITS,
) -> int:
    if minimum_bits > maximum_bits:
        raise ValueError("minimum_bits exceeds maximum_bits")
    fan_in = conv_fan_in(conv)
    safe = [
        bits
        for bits in range(int(minimum_bits), int(maximum_bits) + 1)
        if static_accumulator_bound(fan_in=fan_in, bits=bits) <= SIGNED_INT64_MAX
    ]
    if not safe:
        raise OverflowError(f"no signed-int64-safe precision in [{minimum_bits},{maximum_bits}] for fan_in={fan_in}")
    return max(safe)


def _round_clamp(value: torch.Tensor, *, qmax: int) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise ValueError("mixed fixed-point tensor contains non-finite values")
    return torch.clamp(
        torch.round(value.to(torch.float32)).to(torch.int64),
        min=-int(qmax),
        max=int(qmax),
    )


class MixedInt64Conv2d(nn.Module):
    """One geometry-selected precision Conv2d with exact int64 accumulation."""

    supports_vjp = False

    def __init__(self, conv: nn.Conv2d, *, bits: int) -> None:
        super().__init__()
        self.bits = int(bits)
        self.qmax = signed_qmax(bits)
        self.stride = tuple(conv.stride)
        self.padding = tuple(conv.padding)
        self.dilation = tuple(conv.dilation)
        self.groups = int(conv.groups)
        self.in_channels = int(conv.in_channels)
        self.out_channels = int(conv.out_channels)
        self.fan_in = conv_fan_in(conv)
        self.accumulator_bound = static_accumulator_bound(fan_in=self.fan_in, bits=self.bits)
        if self.accumulator_bound > SIGNED_INT64_MAX:
            raise OverflowError("mixed Conv2d accumulator bound exceeds signed int64")

        weight = conv.weight.detach().to(torch.float32)
        maximum = weight.abs().amax(dim=(1, 2, 3), keepdim=True)
        scales = torch.where(
            maximum > 0.0,
            maximum / float(self.qmax),
            torch.ones_like(maximum),
        )
        weight_q = _round_clamp(weight / scales, qmax=self.qmax)
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("weight_scales", scales.reshape(-1).to(torch.float32))
        self.register_buffer(
            "bias_fp32",
            (
                torch.zeros((conv.out_channels,), dtype=torch.float32)
                if conv.bias is None
                else conv.bias.detach().to(torch.float32)
            ),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and value.requires_grad:
            raise RuntimeError("MixedInt64Conv2d is inference-only and has no VJP")
        source = value.to(torch.float32)
        if not torch.isfinite(source).all():
            raise ValueError("mixed fixed-point activation contains non-finite values")
        maximum = source.abs().max()
        scale = torch.where(
            maximum > 0.0,
            maximum / float(self.qmax),
            torch.ones_like(maximum),
        )
        activation_q = _round_clamp(source / scale, qmax=self.qmax)
        accumulator = functional.conv2d(
            activation_q,
            self.weight_q,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return accumulator.to(torch.float32) * scale * self.weight_scales.reshape(1, -1, 1, 1) + self.bias_fp32.reshape(
            1, -1, 1, 1
        )


@dataclass(frozen=True)
class MixedInt64ModelManifest:
    minimum_bits: int
    maximum_bits: int
    bits_by_path: tuple[tuple[str, int], ...]
    precision_histogram: tuple[tuple[int, int], ...]
    converted_conv2d_count: int
    maximum_accumulator_bound: int
    assignment_rule: str = "largest_geometry_safe_bits_with_signed_int64_static_bound"
    accumulation: str = "exact_signed_int64"
    finalization: str = "single_fp32_scale_and_bias_per_output"
    activation_scale_mode: str = "dynamic_exact_absmax"
    default_enabled: bool = False
    native_speed_claim: bool = False
    score_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_bits": self.minimum_bits,
            "maximum_bits": self.maximum_bits,
            "bits_by_path": dict(self.bits_by_path),
            "precision_histogram": {str(bits): count for bits, count in self.precision_histogram},
            "converted_conv2d_count": self.converted_conv2d_count,
            "maximum_accumulator_bound": self.maximum_accumulator_bound,
            "assignment_rule": self.assignment_rule,
            "accumulation": self.accumulation,
            "finalization": self.finalization,
            "activation_scale_mode": self.activation_scale_mode,
            "default_enabled": self.default_enabled,
            "native_speed_claim": self.native_speed_claim,
            "score_claim": self.score_claim,
        }


def build_mixed_int64_model(
    model: nn.Module,
    *,
    minimum_bits: int = MINIMUM_BITS,
    maximum_bits: int = MAXIMUM_BITS,
) -> tuple[nn.Module, MixedInt64ModelManifest]:
    candidate = copy.deepcopy(model).eval()
    expected = tuple(name for name, module in model.named_modules() if isinstance(module, nn.Conv2d))
    bits_by_path: list[tuple[str, int]] = []
    bounds: list[int] = []

    def replace(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                bits = maximum_safe_bits(
                    child,
                    minimum_bits=minimum_bits,
                    maximum_bits=maximum_bits,
                )
                wrapper = MixedInt64Conv2d(child, bits=bits)
                setattr(parent, name, wrapper)
                bits_by_path.append((path, bits))
                bounds.append(wrapper.accumulator_bound)
            else:
                replace(child, path)

    replace(candidate)
    if tuple(path for path, _ in bits_by_path) != expected:
        raise RuntimeError("mixed exact-int64 Conv2d conversion coverage mismatch")
    histogram: dict[int, int] = {}
    for _, bits in bits_by_path:
        histogram[bits] = histogram.get(bits, 0) + 1
    manifest = MixedInt64ModelManifest(
        minimum_bits=int(minimum_bits),
        maximum_bits=int(maximum_bits),
        bits_by_path=tuple(bits_by_path),
        precision_histogram=tuple(sorted(histogram.items())),
        converted_conv2d_count=len(bits_by_path),
        maximum_accumulator_bound=max(bounds, default=0),
    )
    return candidate, manifest


__all__ = [
    "MAXIMUM_BITS",
    "MINIMUM_BITS",
    "SIGNED_INT64_MAX",
    "MixedInt64Conv2d",
    "MixedInt64ModelManifest",
    "build_mixed_int64_model",
    "conv_fan_in",
    "maximum_safe_bits",
    "signed_qmax",
    "static_accumulator_bound",
]
