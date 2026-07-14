# SPDX-License-Identifier: MIT
"""Frozen-weight-L1-safe per-layer precision for exact-int64 SegNet Conv2d.

The coarse ``fan_in * qmax**2`` bound assumes every frozen weight code has
maximum magnitude.  For a fixed scorer, the exact per-output-channel L1 norm
of the quantized weights gives a strictly tighter, still input-independent
bound:

``abs(accumulator_oc) <= activation_qmax * sum_i(abs(weight_q[oc, i]))``.

No frame, label, logit, or margin participates in the precision assignment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from tac.local_acceleration.mixed_int64_fixedpoint_scorer import (
    MINIMUM_BITS,
    SIGNED_INT64_MAX,
    signed_qmax,
)

MAXIMUM_WEIGHT_L1_BITS = 31


def _quantized_weight_and_scales(
    conv: nn.Conv2d,
    *,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = signed_qmax(bits)
    weight = conv.weight.detach().to(torch.float32)
    maximum = weight.abs().amax(dim=(1, 2, 3), keepdim=True)
    scales = torch.where(
        maximum > 0.0,
        maximum / float(qmax),
        torch.ones_like(maximum),
    )
    weight_q = torch.clamp(
        torch.round(weight / scales).to(torch.int64),
        min=-int(qmax),
        max=int(qmax),
    )
    return weight_q, scales.reshape(-1).to(torch.float32)


def quantized_weight_l1_accumulator_bound(conv: nn.Conv2d, *, bits: int) -> int:
    """Return the exact worst output-channel bound for bounded activations."""

    weight_q, _ = _quantized_weight_and_scales(conv, bits=bits)
    maximum_weight_l1 = int(weight_q.abs().sum(dim=(1, 2, 3)).max().item())
    return signed_qmax(bits) * maximum_weight_l1


def maximum_weight_l1_safe_bits(
    conv: nn.Conv2d,
    *,
    minimum_bits: int = MINIMUM_BITS,
    maximum_bits: int = MAXIMUM_WEIGHT_L1_BITS,
) -> int:
    if minimum_bits > maximum_bits:
        raise ValueError("minimum_bits exceeds maximum_bits")
    safe = [
        bits
        for bits in range(int(minimum_bits), int(maximum_bits) + 1)
        if quantized_weight_l1_accumulator_bound(conv, bits=bits)
        <= SIGNED_INT64_MAX
    ]
    if not safe:
        raise OverflowError(
            "no signed-int64-safe precision in "
            f"[{minimum_bits},{maximum_bits}] under the frozen-weight L1 bound"
        )
    return max(safe)


def _round_clamp(value: torch.Tensor, *, qmax: int) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise ValueError("weight-L1 fixed-point tensor contains non-finite values")
    return torch.clamp(
        torch.round(value.to(torch.float32)).to(torch.int64),
        min=-int(qmax),
        max=int(qmax),
    )


class WeightL1Int64Conv2d(nn.Module):
    """Inference-only Conv2d with a frozen-weight-L1 int64 proof."""

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
        weight_q, weight_scales = _quantized_weight_and_scales(conv, bits=bits)
        maximum_weight_l1 = int(weight_q.abs().sum(dim=(1, 2, 3)).max().item())
        self.accumulator_bound = int(self.qmax) * maximum_weight_l1
        if self.accumulator_bound > SIGNED_INT64_MAX:
            raise OverflowError("frozen-weight-L1 accumulator bound exceeds signed int64")
        self.maximum_weight_l1 = maximum_weight_l1
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("weight_scales", weight_scales)
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
            raise RuntimeError("WeightL1Int64Conv2d is inference-only and has no VJP")
        source = value.to(torch.float32)
        if not torch.isfinite(source).all():
            raise ValueError("weight-L1 fixed-point activation contains non-finite values")
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
        return (
            accumulator.to(torch.float32)
            * scale
            * self.weight_scales.reshape(1, -1, 1, 1)
            + self.bias_fp32.reshape(1, -1, 1, 1)
        )


@dataclass(frozen=True)
class WeightL1Int64ModelManifest:
    minimum_bits: int
    maximum_bits: int
    bits_by_path: tuple[tuple[str, int], ...]
    accumulator_bound_by_path: tuple[tuple[str, int], ...]
    precision_histogram: tuple[tuple[int, int], ...]
    converted_conv2d_count: int
    maximum_accumulator_bound: int
    assignment_rule: str = "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
    bound_kind: str = "activation_qmax_times_max_output_quantized_weight_l1"
    accumulation: str = "exact_signed_int64"
    finalization: str = "single_fp32_scale_and_bias_per_output"
    activation_scale_mode: str = "dynamic_exact_absmax"
    label_or_frame_dependent: bool = False
    default_enabled: bool = False
    native_speed_claim: bool = False
    score_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_bits": self.minimum_bits,
            "maximum_bits": self.maximum_bits,
            "bits_by_path": dict(self.bits_by_path),
            "accumulator_bound_by_path": dict(self.accumulator_bound_by_path),
            "precision_histogram": {
                str(bits): count for bits, count in self.precision_histogram
            },
            "converted_conv2d_count": self.converted_conv2d_count,
            "maximum_accumulator_bound": self.maximum_accumulator_bound,
            "assignment_rule": self.assignment_rule,
            "bound_kind": self.bound_kind,
            "accumulation": self.accumulation,
            "finalization": self.finalization,
            "activation_scale_mode": self.activation_scale_mode,
            "label_or_frame_dependent": self.label_or_frame_dependent,
            "default_enabled": self.default_enabled,
            "native_speed_claim": self.native_speed_claim,
            "score_claim": self.score_claim,
        }


def build_weight_l1_int64_model(
    model: nn.Module,
    *,
    minimum_bits: int = MINIMUM_BITS,
    maximum_bits: int = MAXIMUM_WEIGHT_L1_BITS,
) -> tuple[nn.Module, WeightL1Int64ModelManifest]:
    candidate = copy.deepcopy(model).eval()
    expected = tuple(
        name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)
    )
    bits_by_path: list[tuple[str, int]] = []
    bounds_by_path: list[tuple[str, int]] = []

    def replace(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                bits = maximum_weight_l1_safe_bits(
                    child,
                    minimum_bits=minimum_bits,
                    maximum_bits=maximum_bits,
                )
                wrapper = WeightL1Int64Conv2d(child, bits=bits)
                setattr(parent, name, wrapper)
                bits_by_path.append((path, bits))
                bounds_by_path.append((path, wrapper.accumulator_bound))
            else:
                replace(child, path)

    replace(candidate)
    if tuple(path for path, _ in bits_by_path) != expected:
        raise RuntimeError("weight-L1 exact-int64 Conv2d conversion coverage mismatch")
    histogram: dict[int, int] = {}
    for _, bits in bits_by_path:
        histogram[bits] = histogram.get(bits, 0) + 1
    manifest = WeightL1Int64ModelManifest(
        minimum_bits=int(minimum_bits),
        maximum_bits=int(maximum_bits),
        bits_by_path=tuple(bits_by_path),
        accumulator_bound_by_path=tuple(bounds_by_path),
        precision_histogram=tuple(sorted(histogram.items())),
        converted_conv2d_count=len(bits_by_path),
        maximum_accumulator_bound=max(
            (bound for _, bound in bounds_by_path),
            default=0,
        ),
    )
    return candidate, manifest


__all__ = [
    "MAXIMUM_WEIGHT_L1_BITS",
    "WeightL1Int64Conv2d",
    "WeightL1Int64ModelManifest",
    "build_weight_l1_int64_model",
    "maximum_weight_l1_safe_bits",
    "quantized_weight_l1_accumulator_bound",
]
