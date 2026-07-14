# SPDX-License-Identifier: MIT
"""Calibrated fixed-point/QDQ primitives for frozen scorer feasibility probes.

The QDQ model uses calibrated, fixed per-operator activation scales and
symmetric per-output-channel weight scales.  Convolution/linear accumulation
remains fp32, so this module is a numerical feasibility oracle—not a native
integer-kernel or speed claim.  The same immutable calibration packets are
consumed by the custom Metal and CoreML/ANE builders.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FixedPointForwardPolicy:
    bits: int
    activation_grouping: str = "per_operator_input_tensor"
    activation_scale_mode: str = "fixed_calibration"
    weight_grouping: str = "per_output_channel"
    mode: str = "symmetric_signed_round_to_nearest_even"
    accumulation: str = "fp32_qdq_feasibility"
    calibration_split: tuple[int, int] = (0, 120)
    skipped_module_prefixes: tuple[str, ...] = ()

    @property
    def qmax(self) -> int:
        return qmax_for_bits(self.bits)

    def validate(self) -> None:
        qmax_for_bits(self.bits)
        if self.calibration_split != (0, 120):
            raise ValueError("Task #494 calibration split is frozen at [0,120)")
        if self.accumulation != "fp32_qdq_feasibility":
            raise ValueError("this module does not claim native integer accumulation")
        if self.activation_scale_mode not in {
            "fixed_calibration",
            "dynamic_exact_absmax",
        }:
            raise ValueError(
                "activation_scale_mode must be fixed_calibration or dynamic_exact_absmax"
            )


@dataclass(frozen=True)
class ActivationCalibration:
    split_start: int
    split_stop: int
    operator_absmax: dict[str, float]
    operator_observations: dict[str, int]
    model_kind: str

    def validate(self, expected_paths: Iterable[str]) -> None:
        if (self.split_start, self.split_stop) != (0, 120):
            raise ValueError("calibration split drift")
        expected = set(expected_paths)
        actual = set(self.operator_absmax)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"calibration operator mismatch missing={missing} extra={extra}")
        for path in expected:
            value = float(self.operator_absmax[path])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid activation absmax for {path}: {value}")
            if int(self.operator_observations.get(path, 0)) <= 0:
                raise ValueError(f"no activation observations for {path}")

    def digest(self) -> str:
        payload = {
            "split_start": self.split_start,
            "split_stop": self.split_stop,
            "operator_absmax": self.operator_absmax,
            "operator_observations": self.operator_observations,
            "model_kind": self.model_kind,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def qmax_for_bits(bits: int) -> int:
    bits = int(bits)
    if bits < 2 or bits > 26:
        raise ValueError("signed fixed-point feasibility bits must be in 2..26")
    return (1 << (bits - 1)) - 1


def weight_bearing_modules(model: Any) -> dict[str, Any]:
    import torch

    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    }


def _round_clamp_codes(value: Any, *, qmax: int) -> Any:
    """Round fp32 ratios, then clamp in exact signed-int64 code space.

    The integer-domain clamp is load-bearing above 24 bits: fp32 cannot
    represent W26's positive qmax=33,554,431 and a float-domain clamp can
    silently admit 33,554,432.
    """

    import torch

    rounded = torch.round(value.to(torch.float32)).to(torch.int64)
    return torch.clamp(rounded, min=-int(qmax), max=int(qmax)).to(torch.float32)


def quantize_weight_per_output(weight: Any, *, bits: int) -> tuple[Any, np.ndarray]:
    """Return per-output-channel QDQ weight and immutable float32 scales."""

    import torch

    qmax = qmax_for_bits(bits)
    source = weight.detach().to(dtype=torch.float32)
    if source.ndim < 2:
        raise ValueError(f"weight must have output/input axes, got {tuple(source.shape)}")
    reduce_dims = tuple(range(1, source.ndim))
    maximum = source.abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.where(
        maximum > 0.0,
        maximum / float(qmax),
        torch.ones_like(maximum),
    )
    quantized = _round_clamp_codes(source / scale, qmax=qmax)
    dequantized = quantized * scale
    scales = scale.detach().reshape(source.shape[0]).cpu().numpy().astype(np.float32)
    return dequantized.to(dtype=weight.dtype), np.ascontiguousarray(scales)


def quantize_activation_fixed(value: Any, *, absmax: float, bits: int) -> Any:
    import torch

    qmax = qmax_for_bits(bits)
    maximum = float(absmax)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError(f"activation absmax must be finite/non-negative, got {maximum}")
    if maximum == 0.0:
        return torch.zeros_like(value)
    scale = maximum / float(qmax)
    return _round_clamp_codes(value.to(torch.float32) / scale, qmax=qmax) * scale


def quantize_activation_dynamic(value: Any, *, bits: int) -> Any:
    """QDQ with a label-free, order-invariant max-absolute runtime scale."""

    import torch

    qmax = qmax_for_bits(bits)
    source = value.to(torch.float32)
    if not bool(torch.isfinite(source).all().item()):
        raise ValueError("dynamic fixed-point activation contains non-finite values")
    maximum = source.abs().amax()
    if float(maximum.item()) == 0.0:
        return torch.zeros_like(source).to(dtype=value.dtype)
    scale = maximum / float(qmax)
    output = _round_clamp_codes(source / scale, qmax=qmax) * scale
    return output.to(dtype=value.dtype)


class ActivationAbsMaxCalibrator:
    """Observe only inputs to every Conv2d/Linear in a frozen fp32 model."""

    def __init__(self, model: Any, *, model_kind: str) -> None:
        self.model = model
        self.model_kind = str(model_kind)
        self.absmax: dict[str, float] = {}
        self.observations: dict[str, int] = {}
        self._handles: list[Any] = []
        for name, module in weight_bearing_modules(model).items():

            def hook(
                layer: Any,
                inputs: tuple[Any, ...],
                *,
                name: str = name,
            ) -> None:
                del layer
                if not inputs:
                    raise RuntimeError(f"{name} received no activation")
                value = inputs[0]
                current = float(value.detach().abs().max().cpu().item())
                self.absmax[name] = max(self.absmax.get(name, 0.0), current)
                self.observations[name] = self.observations.get(name, 0) + 1

            self._handles.append(module.register_forward_pre_hook(hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def freeze(self) -> ActivationCalibration:
        if self._handles:
            self.close()
        calibration = ActivationCalibration(
            split_start=0,
            split_stop=120,
            operator_absmax=dict(sorted(self.absmax.items())),
            operator_observations=dict(sorted(self.observations.items())),
            model_kind=self.model_kind,
        )
        calibration.validate(weight_bearing_modules(self.model))
        return calibration


def build_calibrated_qdq_model(
    fp32_model: Any,
    calibration: ActivationCalibration,
    policy: FixedPointForwardPolicy,
) -> tuple[Any, dict[str, Any]]:
    """Deep-copy a scorer, QDQ weights, and install fixed activation pre-hooks."""

    policy.validate()
    original_paths = weight_bearing_modules(fp32_model)
    calibration.validate(original_paths)
    candidate = copy.deepcopy(fp32_model).eval()
    modules = weight_bearing_modules(candidate)
    handles: list[Any] = []
    rows: list[dict[str, Any]] = []
    skipped = tuple(policy.skipped_module_prefixes)
    for name, module in modules.items():
        should_skip = any(name == prefix or name.startswith(f"{prefix}.") for prefix in skipped)
        if should_skip:
            rows.append({"path": name, "precision": "fp32", "reason": "policy_skip_prefix"})
            continue
        dequantized, weight_scales = quantize_weight_per_output(module.weight, bits=policy.bits)
        module.weight.data.copy_(dequantized)
        activation_absmax = float(calibration.operator_absmax[name])

        def pre_hook(
            layer: Any,
            inputs: tuple[Any, ...],
            *,
            activation_absmax: float = activation_absmax,
            bits: int = policy.bits,
        ) -> tuple[Any, ...]:
            del layer
            if not inputs:
                raise RuntimeError("calibrated QDQ operator received no input")
            if policy.activation_scale_mode == "dynamic_exact_absmax":
                return (
                    quantize_activation_dynamic(inputs[0], bits=bits),
                    *inputs[1:],
                )
            return (
                quantize_activation_fixed(inputs[0], absmax=activation_absmax, bits=bits),
                *inputs[1:],
            )

        handles.append(module.register_forward_pre_hook(pre_hook))
        rows.append(
            {
                "path": name,
                "precision": f"w{policy.bits}a{policy.bits}",
                "activation_absmax": activation_absmax,
                "activation_scale": (
                    activation_absmax / float(policy.qmax)
                    if activation_absmax
                    and policy.activation_scale_mode == "fixed_calibration"
                    else None
                ),
                "activation_scale_mode": policy.activation_scale_mode,
                "dynamic_scale_reduction": (
                    "max(abs(x)); commutative/idempotent; label-free"
                    if policy.activation_scale_mode == "dynamic_exact_absmax"
                    else None
                ),
                "weight_scale_sha256": hashlib.sha256(weight_scales.tobytes()).hexdigest(),
                "weight_output_channels": int(weight_scales.size),
                "bias_precision": "fp32" if module.bias is not None else "none",
                "accumulation": policy.accumulation,
                "quantized_code_clamp": "round_fp32_then_exact_signed_int64_clamp",
            }
        )
    candidate._task494_qdq_hook_handles = handles
    manifest = {
        "schema": "calibrated_fixedpoint_scorer_instrumentation.v1",
        "policy": asdict(policy),
        "qmax": policy.qmax,
        "calibration_digest": calibration.digest(),
        "model_kind": calibration.model_kind,
        "operators": rows,
        "operator_count": len(rows),
        "quantized_operator_count": sum(row["precision"] != "fp32" for row in rows),
        "native_integer_kernel_claim": False,
        "quality_interpretation": "CALIBRATED_WA_QDQ_WITH_FP32_ACCUMULATION",
    }
    return candidate, manifest


def fixedpoint_accumulator_bound(
    module: Any, *, activation_bits: int, weight_bits: int
) -> dict[str, Any]:
    """Static integer-MAC bound for a future native kernel packet."""

    import torch

    if isinstance(module, torch.nn.Conv2d):
        fan_in = (
            int(module.in_channels) // int(module.groups)
        ) * int(module.kernel_size[0]) * int(module.kernel_size[1])
    elif isinstance(module, torch.nn.Linear):
        fan_in = int(module.in_features)
    else:
        raise TypeError(f"unsupported weight-bearing module {type(module).__name__}")
    activation_qmax = qmax_for_bits(activation_bits)
    weight_qmax = qmax_for_bits(weight_bits)
    bound = fan_in * activation_qmax * weight_qmax
    minimum_bits = max(1, int(np.ceil(np.log2(2 * bound + 1)))) if bound else 1
    return {
        "bound_kind": "STATIC_WORST_CASE_FAN_IN_QMAX_PRODUCT",
        "fan_in": fan_in,
        "activation_qmax": activation_qmax,
        "weight_qmax": weight_qmax,
        "max_abs_accumulator_bound": int(bound),
        "minimum_signed_accumulator_bits": minimum_bits,
        "int32_safe": bound <= np.iinfo(np.int32).max,
    }


__all__ = [
    "ActivationAbsMaxCalibrator",
    "ActivationCalibration",
    "FixedPointForwardPolicy",
    "build_calibrated_qdq_model",
    "fixedpoint_accumulator_bound",
    "qmax_for_bits",
    "quantize_activation_dynamic",
    "quantize_activation_fixed",
    "quantize_weight_per_output",
    "weight_bearing_modules",
]
