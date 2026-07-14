# SPDX-License-Identifier: MIT
"""Margin-adaptive exact-int64 precision maps for the frozen SegNet forward.

This module composes the already-landed frozen-weight-L1 integer Conv2d and
custom-Metal kernels.  It adds three things without weakening their arithmetic
contract:

* arbitrary, fully covered per-layer precision maps with per-output-channel
  weight scales;
* a finite-ladder reverse-waterfill that chooses the cheapest profile at each
  output pixel whose top1/rival interval is strictly separated; and
* an executable custom-Metal adapter for one selected per-layer map.

The interval helper is deliberately agnostic about how its error radius was
obtained.  A caller must label ``bound_kind``.  In the n600 probe the radius is
the observed per-pixel absolute fp32-vs-fixed-point logit error, so the result
is a source-corpus certificate, not unseen-input IBP.  The per-pixel waterfill
is a lower bound over a finite profile ladder; global squeeze-excite and the
measured full-frame halo prevent it from being called a native sparse-kernel
speedup.  Only the selected single per-layer map is executable and timed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import nn

from tac.local_acceleration.metal_fixedpoint_verdict import (
    METAL_FIXEDPOINT_VERDICT_FLAG,
    FixedPointConvPacket,
    FixedPointMetalConstants,
    integer_storage_bits_for_precision,
    metal_fixedpoint_backend_available,
    prepare_fixedpoint_conv_packet_metal,
)
from tac.local_acceleration.metal_mixed_int64_fixedpoint_verdict import (
    MetalWeightL1Int64Conv2DAdapter,
    build_weight_l1_fixedpoint_conv_packet,
)
from tac.local_acceleration.mixed_int64_fixedpoint_scorer import (
    SIGNED_INT64_MAX,
)
from tac.local_acceleration.weight_l1_int64_fixedpoint_scorer import (
    MAXIMUM_WEIGHT_L1_BITS,
    WeightL1Int64Conv2d,
    maximum_weight_l1_safe_bits,
    quantized_weight_l1_accumulator_bound,
)

MINIMUM_PROFILE_BITS = 8
DEFAULT_PROFILE_CAPS: tuple[int, ...] = (
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    27,
    28,
    29,
    30,
    31,
)
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _conv_modules(model: nn.Module) -> dict[str, nn.Conv2d]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }


def _map_sha256(precision_by_path: Mapping[str, int]) -> str:
    encoded = json.dumps(
        {str(path): int(bits) for path, bits in sorted(precision_by_path.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_weight_l1_safety_ceilings(model: nn.Module) -> dict[str, int]:
    """Return each Conv2d's largest frozen-weight-L1-safe signed precision."""

    return {
        path: maximum_weight_l1_safe_bits(
            conv,
            minimum_bits=2,
            maximum_bits=MAXIMUM_WEIGHT_L1_BITS,
        )
        for path, conv in _conv_modules(model).items()
    }


def derive_capped_precision_map(
    model: nn.Module,
    *,
    cap_bits: int,
    minimum_bits: int = MINIMUM_PROFILE_BITS,
) -> dict[str, int]:
    """Derive one nested per-layer profile from a global precision cap.

    The cap orders a finite ladder.  A layer receives ``min(cap, safe_ceiling)``;
    therefore high-fan-in layers can remain below the cap while small/depthwise
    layers consume the added precision.  The map is frame/label independent;
    margin evidence chooses among maps rather than changing a map at runtime.
    """

    cap = int(cap_bits)
    floor = int(minimum_bits)
    if floor < 2 or cap < floor or cap > MAXIMUM_WEIGHT_L1_BITS:
        raise ValueError("precision cap must satisfy 2 <= minimum_bits <= cap <= 31")
    ceilings = derive_weight_l1_safety_ceilings(model)
    return {path: max(floor, min(cap, safe)) for path, safe in ceilings.items()}


def validate_precision_map(
    model: nn.Module,
    precision_by_path: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """Validate exact coverage and the signed-int64 bound for every layer."""

    modules = _conv_modules(model)
    observed = {str(path): int(bits) for path, bits in precision_by_path.items()}
    if set(observed) != set(modules):
        missing = sorted(set(modules) - set(observed))
        extra = sorted(set(observed) - set(modules))
        raise ValueError(f"precision-map coverage differs: missing={missing}, extra={extra}")
    ceilings = derive_weight_l1_safety_ceilings(model)
    bounds: dict[str, int] = {}
    for path, conv in modules.items():
        bits = observed[path]
        if bits < 2 or bits > ceilings[path]:
            raise ValueError(
                f"precision {bits} for {path} is outside signed-int64-safe 2..{ceilings[path]}"
            )
        bound = quantized_weight_l1_accumulator_bound(conv, bits=bits)
        if bound > SIGNED_INT64_MAX:
            raise OverflowError(f"precision map overflows signed int64 at {path}")
        bounds[path] = int(bound)
    return ceilings, bounds


def weighted_average_bits(
    precision_by_path: Mapping[str, int],
    work_by_path: Mapping[str, int | float],
) -> float:
    """Return the work-weighted mean precision, refusing partial coverage."""

    if set(precision_by_path) != set(work_by_path):
        raise ValueError("precision/work map coverage differs")
    weights = {path: float(value) for path, value in work_by_path.items()}
    if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("work weights must be finite and non-negative")
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("work weights must have positive total")
    return float(
        sum(int(precision_by_path[path]) * weights[path] for path in weights) / total
    )


@dataclass(frozen=True)
class MarginAdaptiveInt64Manifest:
    precision_by_path: tuple[tuple[str, int], ...]
    safety_ceiling_by_path: tuple[tuple[str, int], ...]
    accumulator_bound_by_path: tuple[tuple[str, int], ...]
    precision_histogram: tuple[tuple[int, int], ...]
    integer_storage_bits_by_path: tuple[tuple[str, int], ...]
    integer_storage_histogram: tuple[tuple[int, int], ...]
    parameter_weighted_average_bits: float
    parameter_weighted_average_storage_bits: float
    converted_conv2d_count: int
    maximum_accumulator_bound: int
    precision_map_sha256: str
    assignment_rule: str = "margin_selected_capped_frozen_weight_l1_safe_per_layer_map"
    bound_kind: str = "activation_qmax_times_max_output_quantized_weight_l1"
    scale_granularity: str = "per_output_channel_weight_scale; per_layer_dynamic_activation_scale"
    accumulation: str = "exact_signed_int64"
    finalization: str = "single_fp32_scale_and_bias_per_output"
    activation_scale_mode: str = "dynamic_exact_absmax"
    region_runtime_claim: bool = False
    label_or_frame_dependent_runtime: bool = False
    native_speed_claim: bool = False
    score_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "margin_adaptive_int64_model_manifest.v1",
            "precision_by_path": dict(self.precision_by_path),
            "safety_ceiling_by_path": dict(self.safety_ceiling_by_path),
            "accumulator_bound_by_path": dict(self.accumulator_bound_by_path),
            "precision_histogram": {
                str(bits): count for bits, count in self.precision_histogram
            },
            "integer_storage_bits_by_path": dict(self.integer_storage_bits_by_path),
            "integer_storage_histogram": {
                str(bits): count for bits, count in self.integer_storage_histogram
            },
            "parameter_weighted_average_bits": self.parameter_weighted_average_bits,
            "parameter_weighted_average_storage_bits": (
                self.parameter_weighted_average_storage_bits
            ),
            "converted_conv2d_count": self.converted_conv2d_count,
            "maximum_accumulator_bound": self.maximum_accumulator_bound,
            "precision_map_sha256": self.precision_map_sha256,
            "assignment_rule": self.assignment_rule,
            "bound_kind": self.bound_kind,
            "scale_granularity": self.scale_granularity,
            "accumulation": self.accumulation,
            "finalization": self.finalization,
            "activation_scale_mode": self.activation_scale_mode,
            "region_runtime_claim": self.region_runtime_claim,
            "label_or_frame_dependent_runtime": self.label_or_frame_dependent_runtime,
            "native_speed_claim": self.native_speed_claim,
            "score_claim": self.score_claim,
        }


def _manifest(
    model: nn.Module,
    precision_by_path: Mapping[str, int],
    ceilings: Mapping[str, int],
    bounds: Mapping[str, int],
) -> MarginAdaptiveInt64Manifest:
    modules = _conv_modules(model)
    histogram: dict[int, int] = {}
    storage_by_path = {
        path: integer_storage_bits_for_precision(bits)
        for path, bits in precision_by_path.items()
    }
    storage_histogram: dict[int, int] = {}
    for bits in precision_by_path.values():
        histogram[int(bits)] = histogram.get(int(bits), 0) + 1
    for bits in storage_by_path.values():
        storage_histogram[int(bits)] = storage_histogram.get(int(bits), 0) + 1
    parameter_work = {
        path: int(conv.weight.numel()) for path, conv in modules.items()
    }
    return MarginAdaptiveInt64Manifest(
        precision_by_path=tuple(sorted((path, int(bits)) for path, bits in precision_by_path.items())),
        safety_ceiling_by_path=tuple(sorted((path, int(bits)) for path, bits in ceilings.items())),
        accumulator_bound_by_path=tuple(sorted((path, int(bound)) for path, bound in bounds.items())),
        precision_histogram=tuple(sorted(histogram.items())),
        integer_storage_bits_by_path=tuple(sorted(storage_by_path.items())),
        integer_storage_histogram=tuple(sorted(storage_histogram.items())),
        parameter_weighted_average_bits=weighted_average_bits(
            precision_by_path,
            parameter_work,
        ),
        parameter_weighted_average_storage_bits=weighted_average_bits(
            storage_by_path,
            parameter_work,
        ),
        converted_conv2d_count=len(modules),
        maximum_accumulator_bound=max(bounds.values(), default=0),
        precision_map_sha256=_map_sha256(precision_by_path),
    )


def build_margin_adaptive_int64_model(
    model: nn.Module,
    *,
    precision_by_path: Mapping[str, int],
) -> tuple[nn.Module, MarginAdaptiveInt64Manifest]:
    """Build the deterministic CPU numerical twin for one selected map."""

    normalized = {str(path): int(bits) for path, bits in precision_by_path.items()}
    ceilings, bounds = validate_precision_map(model, normalized)
    manifest = _manifest(model, normalized, ceilings, bounds)
    candidate = copy.deepcopy(model).eval()

    def replace(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                setattr(parent, name, WeightL1Int64Conv2d(child, bits=normalized[path]))
            else:
                replace(child, path)

    replace(candidate)
    converted = {
        name for name, module in candidate.named_modules() if isinstance(module, WeightL1Int64Conv2d)
    }
    if converted != set(normalized):
        raise RuntimeError("margin-adaptive CPU conversion coverage failed")
    return candidate, manifest


def interval_argmax_certificate_mask(
    center_logits: Any,
    abs_error_bound: Any,
    *,
    expected_argmax: Any | None = None,
    class_axis: int = 1,
) -> np.ndarray:
    """Return pixels satisfying ``L_top1 > max(U_other)``.

    ``center_logits +/- abs_error_bound`` is the supplied interval.  The
    function proves only the interval statement; it does not infer whether the
    radius is analytic, affine, calibration-derived, or corpus-observed.
    """

    center = np.asarray(center_logits, dtype=np.float32)
    radius = np.asarray(abs_error_bound, dtype=np.float32)
    if center.ndim < 2:
        raise ValueError("logits must include a class dimension")
    if radius.shape != center.shape:
        try:
            radius = np.broadcast_to(radius, center.shape)
        except ValueError as exc:
            raise ValueError("error bound is not broadcastable to logits") from exc
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(radius)):
        raise ValueError("interval inputs contain non-finite values")
    if np.any(radius < 0.0):
        raise ValueError("absolute error bounds must be non-negative")
    axis = int(class_axis)
    if axis < 0:
        axis += center.ndim
    if axis < 0 or axis >= center.ndim:
        raise ValueError("class axis is out of range")
    classes = int(center.shape[axis])
    if classes < 2:
        raise ValueError("argmax certificate requires at least two classes")
    winner = (
        np.argmax(center, axis=axis)
        if expected_argmax is None
        else np.asarray(expected_argmax, dtype=np.int64)
    )
    target_shape = center.shape[:axis] + center.shape[axis + 1 :]
    if winner.shape != target_shape:
        raise ValueError("expected argmax shape differs from logits without class axis")
    if np.any(winner < 0) or np.any(winner >= classes):
        raise ValueError("expected argmax contains an invalid class")
    lower = center - radius
    upper = center + radius
    index = np.expand_dims(winner, axis=axis)
    winner_lower = np.take_along_axis(lower, index, axis=axis).squeeze(axis=axis)
    class_indices = np.arange(classes, dtype=np.int64)
    reshape = [1] * center.ndim
    reshape[axis] = classes
    rivals = np.where(
        class_indices.reshape(reshape) == index,
        np.float32(-np.inf),
        upper,
    )
    rival_upper = np.max(rivals, axis=axis)
    return np.asarray(winner_lower > rival_upper, dtype=np.bool_)


@dataclass(frozen=True)
class ProfileCertificate:
    name: str
    average_bits: float
    certified_mask: np.ndarray


@dataclass(frozen=True)
class FiniteLadderWaterfill:
    selected_profile_index: np.ndarray
    certified_mask: np.ndarray
    average_selected_bits: float | None
    profile_histogram: tuple[tuple[str, int], ...]
    profile_order: tuple[str, ...]
    optimality_scope: str = "exact pointwise minimum over supplied finite profile ladder"
    native_region_execution_claim: bool = False

    def to_summary(self) -> dict[str, Any]:
        total = int(self.certified_mask.size)
        certified = int(np.count_nonzero(self.certified_mask))
        return {
            "certified_pixels": certified,
            "pixels": total,
            "certified_fraction": float(certified / total) if total else 0.0,
            "average_selected_bits": self.average_selected_bits,
            "profile_histogram": dict(self.profile_histogram),
            "profile_order": list(self.profile_order),
            "optimality_scope": self.optimality_scope,
            "native_region_execution_claim": self.native_region_execution_claim,
        }


def solve_finite_profile_waterfill(
    profiles: Sequence[ProfileCertificate],
) -> FiniteLadderWaterfill:
    """Choose the lowest-average-bit certifying profile independently per pixel."""

    if not profiles:
        raise ValueError("profile ladder is empty")
    ordered = sorted(profiles, key=lambda row: (float(row.average_bits), row.name))
    shape = np.asarray(ordered[0].certified_mask, dtype=np.bool_).shape
    selected = np.full(shape, -1, dtype=np.int16)
    certified = np.zeros(shape, dtype=np.bool_)
    histogram: list[tuple[str, int]] = []
    selected_bits_sum = 0.0
    for index, profile in enumerate(ordered):
        mask = np.asarray(profile.certified_mask, dtype=np.bool_)
        if mask.shape != shape:
            raise ValueError("certificate masks in the profile ladder differ in shape")
        choose = mask & ~certified
        count = int(np.count_nonzero(choose))
        selected[choose] = index
        certified |= choose
        histogram.append((profile.name, count))
        selected_bits_sum += count * float(profile.average_bits)
    certified_count = int(np.count_nonzero(certified))
    average = (
        float(selected_bits_sum / certified_count) if certified_count else None
    )
    return FiniteLadderWaterfill(
        selected_profile_index=selected,
        certified_mask=certified,
        average_selected_bits=average,
        profile_histogram=tuple(histogram),
        profile_order=tuple(profile.name for profile in ordered),
    )


def build_metal_margin_adaptive_int64_segnet_adapter(
    torch_segnet: nn.Module,
    *,
    precision_by_path: Mapping[str, int],
    operator_absmax: Mapping[str, float],
    require_opt_in: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Convert all frozen-SegNet Conv2d with one validated per-layer map."""

    from tac.local_acceleration import mlx_scorer_adapters as adapters
    from tac.local_acceleration.metal_mixed_int64_fixedpoint_verdict import (
        _ADAPTER_LOCK,
    )

    if (
        require_opt_in
        and os.environ.get(METAL_FIXEDPOINT_VERDICT_FLAG, "").strip().lower()
        not in _TRUTHY
    ):
        raise RuntimeError(f"set {METAL_FIXEDPOINT_VERDICT_FLAG}=1 to request this backend")
    if not metal_fixedpoint_backend_available():
        raise RuntimeError("margin-adaptive fixed-point SegNet requested without evaluated Metal")
    normalized = {str(path): int(bits) for path, bits in precision_by_path.items()}
    ceilings, bounds = validate_precision_map(torch_segnet, normalized)
    manifest = _manifest(torch_segnet, normalized, ceilings, bounds)
    modules = {id(module): name for name, module in _conv_modules(torch_segnet).items()}
    expected = set(modules.values())
    if set(operator_absmax) != expected:
        raise ValueError("calibration/SegNet operator set differs for margin-adaptive map")
    consumed: list[str] = []
    packets: list[FixedPointConvPacket] = []
    constants: list[FixedPointMetalConstants] = []
    original_converter = adapters.torch_conv2d_to_mlx
    original_explicit = adapters.MLXExplicitSpatialConv2dAdapter

    def convert(torch_conv: nn.Conv2d) -> Any:
        path = modules.get(id(torch_conv))
        if path is None:
            raise RuntimeError("unregistered Conv2d reached margin-adaptive converter")
        packet = build_weight_l1_fixedpoint_conv_packet(
            torch_conv,
            bits=normalized[path],
        )
        adapter = MetalWeightL1Int64Conv2DAdapter.__new__(
            MetalWeightL1Int64Conv2DAdapter
        )
        adapter.packet = packet
        adapter.constants = prepare_fixedpoint_conv_packet_metal(
            packet,
            integer_storage_bits=integer_storage_bits_for_precision(normalized[path]),
        )
        packets.append(packet)
        constants.append(adapter.constants)
        consumed.append(path)
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
        raise RuntimeError("margin-adaptive Metal conversion coverage failed")
    if len(constants) != len(expected):
        raise RuntimeError("margin-adaptive Metal constant-buffer cache coverage failed")
    payload = manifest.to_dict()
    payload.update(
        schema="metal_margin_adaptive_int64_segnet_adapter.v1",
        all_convs_replaced=True,
        constant_buffers_cached=True,
        arithmetic="integer activation/weight; exact int64 MAC; fp32 dequant+bias",
        integer_operand_storage="per-layer narrowest exact signed bucket in {int8,int16,int32}",
        native_speed_claim=True,
        promotion_gate=(
            "frozen design-selected map exact on untouched second validation + "
            "real n600 cross-process digest + positive native latency"
        ),
    )
    return converted, payload


__all__ = [
    "DEFAULT_PROFILE_CAPS",
    "MINIMUM_PROFILE_BITS",
    "FiniteLadderWaterfill",
    "MarginAdaptiveInt64Manifest",
    "ProfileCertificate",
    "build_margin_adaptive_int64_model",
    "build_metal_margin_adaptive_int64_segnet_adapter",
    "derive_capped_precision_map",
    "derive_weight_l1_safety_ceilings",
    "interval_argmax_certificate_mask",
    "solve_finite_profile_waterfill",
    "validate_precision_map",
    "weighted_average_bits",
]
