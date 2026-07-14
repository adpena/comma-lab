# SPDX-License-Identifier: MIT
"""Exact-int64 CPU twin for the custom fixed-point SegNet backend.

This module is a numerical-authority bridge, not a throughput backend.  It
keeps activation and weight codes as signed integers through Conv2d, performs
the multiply-accumulate in signed int64, and applies one fp32 finalization per
output.  The arithmetic mirrors ``metal_fixedpoint_verdict`` closely enough to
separate fixed-point quantization error from the fp32-accumulation error in a
PyTorch QDQ feasibility sweep.

Every converted layer proves its static worst-case accumulator bound fits in
int64 before execution.  The wrapper is inference-only and deliberately
refuses autograd.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from tac.local_acceleration.metal_fixedpoint_verdict import (
    FixedPointConvPacket,
    build_fixedpoint_conv_packet,
)


def _exact_round_clamp_int64(value: torch.Tensor, *, qmax: int) -> torch.Tensor:
    """Round fp32 ratios and clamp in exact signed-int64 code space."""

    if not torch.isfinite(value).all():
        raise ValueError("fixed-point activation contains non-finite values")
    rounded = torch.round(value.to(torch.float32)).to(torch.int64)
    return torch.clamp(rounded, min=-int(qmax), max=int(qmax))


class ExactInt64Conv2d(nn.Module):
    """Inference-only Conv2d with exact signed-int64 multiply-accumulate."""

    supports_vjp = False

    def __init__(self, packet: FixedPointConvPacket) -> None:
        super().__init__()
        self.bits = int(packet.bits)
        self.qmax = int(packet.qmax)
        self.activation_scale_mode = str(packet.activation_scale_mode)
        self.fixed_activation_scale = float(packet.activation_scale)
        self.stride = tuple(packet.stride)
        self.padding = tuple(packet.padding)
        self.dilation = tuple(packet.dilation)
        self.groups = int(packet.groups)
        self.in_channels = int(packet.in_channels)
        self.out_channels = int(packet.out_channels)
        self.accumulator_bound = int(packet.accumulator_bound)
        self.minimum_signed_accumulator_bits = int(packet.minimum_signed_accumulator_bits)

        weight_oihw = np.ascontiguousarray(packet.weight_q_ohwi.transpose(0, 3, 1, 2))
        self.register_buffer("weight_q", torch.from_numpy(weight_oihw).to(torch.int64))
        self.register_buffer(
            "weight_scales",
            torch.from_numpy(np.ascontiguousarray(packet.weight_scales)).to(torch.float32),
        )
        self.register_buffer(
            "bias_fp32",
            torch.from_numpy(np.ascontiguousarray(packet.bias)).to(torch.float32),
        )

    @classmethod
    def from_torch_conv(
        cls,
        conv: nn.Conv2d,
        *,
        bits: int,
        activation_scale_mode: str = "dynamic_exact_absmax",
        activation_absmax: float = 1.0,
    ) -> ExactInt64Conv2d:
        packet = build_fixedpoint_conv_packet(
            conv,
            activation_absmax=activation_absmax,
            bits=bits,
            activation_scale_mode=activation_scale_mode,
        )
        return cls(packet)

    def _activation_scale(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation_scale_mode == "fixed_calibration":
            return torch.as_tensor(
                self.fixed_activation_scale,
                dtype=torch.float32,
                device=value.device,
            )
        source = value.to(torch.float32)
        if not torch.isfinite(source).all():
            raise ValueError("dynamic fixed-point activation contains non-finite values")
        maximum = source.abs().max()
        return torch.where(
            maximum > 0.0,
            maximum / float(self.qmax),
            torch.ones_like(maximum),
        )

    def quantize_activation(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source = value.to(torch.float32)
        scale = self._activation_scale(source)
        return _exact_round_clamp_int64(source / scale, qmax=self.qmax), scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and value.requires_grad:
            raise RuntimeError("ExactInt64Conv2d is inference-only and has no VJP")
        activation_q, activation_scale = self.quantize_activation(value)
        accumulator = functional.conv2d(
            activation_q,
            self.weight_q,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        # One deterministic fp32 finalization, matching the custom-Metal kernel.
        return accumulator.to(torch.float32) * activation_scale * self.weight_scales.reshape(
            1, -1, 1, 1
        ) + self.bias_fp32.reshape(1, -1, 1, 1)


@dataclass(frozen=True)
class ExactInt64ModelManifest:
    bits: int
    activation_scale_mode: str
    converted_paths: tuple[str, ...]
    converted_conv2d_count: int
    maximum_accumulator_bound: int
    maximum_minimum_signed_accumulator_bits: int
    accumulation: str = "exact_signed_int64"
    finalization: str = "single_fp32_scale_and_bias_per_output"
    default_enabled: bool = False
    native_speed_claim: bool = False
    score_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "activation_scale_mode": self.activation_scale_mode,
            "converted_paths": list(self.converted_paths),
            "converted_conv2d_count": self.converted_conv2d_count,
            "maximum_accumulator_bound": self.maximum_accumulator_bound,
            "maximum_minimum_signed_accumulator_bits": (self.maximum_minimum_signed_accumulator_bits),
            "accumulation": self.accumulation,
            "finalization": self.finalization,
            "default_enabled": self.default_enabled,
            "native_speed_claim": self.native_speed_claim,
            "score_claim": self.score_claim,
        }


def build_exact_int64_model(
    model: nn.Module,
    *,
    bits: int,
    activation_scale_mode: str = "dynamic_exact_absmax",
    operator_absmax: dict[str, float] | None = None,
) -> tuple[nn.Module, ExactInt64ModelManifest]:
    """Deep-copy ``model`` and replace every Conv2d with the exact-int64 twin."""

    candidate = copy.deepcopy(model).eval()
    converted_paths: list[str] = []
    bounds: list[int] = []
    minimum_bits: list[int] = []

    def replace(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                if activation_scale_mode == "fixed_calibration":
                    if operator_absmax is None or path not in operator_absmax:
                        raise KeyError(f"missing fixed activation absmax for {path}")
                    activation_absmax = float(operator_absmax[path])
                else:
                    activation_absmax = 1.0
                wrapper = ExactInt64Conv2d.from_torch_conv(
                    child,
                    bits=bits,
                    activation_scale_mode=activation_scale_mode,
                    activation_absmax=activation_absmax,
                )
                setattr(parent, name, wrapper)
                converted_paths.append(path)
                bounds.append(wrapper.accumulator_bound)
                minimum_bits.append(wrapper.minimum_signed_accumulator_bits)
            else:
                replace(child, path)

    expected_paths = tuple(name for name, module in model.named_modules() if isinstance(module, nn.Conv2d))
    replace(candidate)
    if tuple(converted_paths) != expected_paths:
        raise RuntimeError("exact-int64 Conv2d conversion coverage mismatch")
    if any(isinstance(module, nn.Conv2d) for module in candidate.modules()):
        raise RuntimeError("unconverted Conv2d remains in exact-int64 candidate")
    manifest = ExactInt64ModelManifest(
        bits=int(bits),
        activation_scale_mode=activation_scale_mode,
        converted_paths=tuple(converted_paths),
        converted_conv2d_count=len(converted_paths),
        maximum_accumulator_bound=max(bounds, default=0),
        maximum_minimum_signed_accumulator_bits=max(minimum_bits, default=1),
    )
    return candidate, manifest
