"""Score-aware per-tensor weights for lossy coarsening allocators.

The existing Lagrangian per-tensor allocator already accepts a non-negative
``weights`` vector and applies it as::

    cost(t, K) = byte_proxy(t, K) + lambda * weight[t] * rel_err(t, K)^2

This module builds that vector from beta-Fisher sensitivity maps plus optional
boundary-mass and texture/film-grain capacity priors. It does not add decoder
noise, score archives, or dispatch jobs; it emits allocator input for
downstream byte-closed archive builders.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tac.codec.cost_curves import TensorBlob, precompute_per_tensor_K_curves
from tac.optimization.lagrangian_per_tensor_allocation import (
    JointEncoderHook,
    LagrangianPerTensorAllocator,
    compute_local_variance_proxy,
)
from tac.sensitivity_map import (
    SensitivityMapError,
    load_sensitivity_map,
    real_sensitivity_metadata_blockers,
    validate_sensitivity_vector,
)

if TYPE_CHECKING:
    import torch

SCHEMA_VERSION = "beta_fisher_lossy_coarsening_tensor_weights.v1"
MODULE_NAME = "tac.optimization.beta_fisher_lossy_weights"
DEFAULT_MIN_WEIGHT = 1e-6
DEFAULT_MAX_WEIGHT = 1e6


class BetaFisherWeightError(ValueError):
    """Raised when score-aware tensor-weight inputs are malformed."""


@dataclass(frozen=True)
class TensorWeightTarget:
    """Tensor metadata used to build score-aware allocator weights."""

    name: str
    shape: tuple[int, ...]
    symbols: np.ndarray | None = None

    @property
    def n_values(self) -> int:
        return int(math.prod(self.shape)) if self.shape else 1

    @property
    def out_channels(self) -> int:
        return int(self.shape[0]) if self.shape else 1


@dataclass(frozen=True)
class ScoreWeightConfig:
    """Knobs for the beta-Fisher x lossy-coarsening weight export."""

    fisher_beta: float = 1.0
    boundary_alpha: float = 0.5
    film_grain_alpha: float = 0.25
    min_weight: float = DEFAULT_MIN_WEIGHT
    max_weight: float = DEFAULT_MAX_WEIGHT
    missing_sensitivity_weight: float = 1.0
    use_variance_as_film_grain_capacity: bool = True

    def validate(self) -> None:
        for label, value in {
            "fisher_beta": self.fisher_beta,
            "boundary_alpha": self.boundary_alpha,
            "film_grain_alpha": self.film_grain_alpha,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "missing_sensitivity_weight": self.missing_sensitivity_weight,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise BetaFisherWeightError(f"{label} must be a finite numeric value")
        if self.fisher_beta < 0.0:
            raise BetaFisherWeightError("fisher_beta must be non-negative")
        if self.boundary_alpha < 0.0:
            raise BetaFisherWeightError("boundary_alpha must be non-negative")
        if self.film_grain_alpha < 0.0:
            raise BetaFisherWeightError("film_grain_alpha must be non-negative")
        if self.min_weight <= 0.0:
            raise BetaFisherWeightError("min_weight must be positive")
        if self.max_weight < self.min_weight:
            raise BetaFisherWeightError("max_weight must be >= min_weight")
        if self.missing_sensitivity_weight <= 0.0:
            raise BetaFisherWeightError("missing_sensitivity_weight must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fisher_beta": float(self.fisher_beta),
            "boundary_alpha": float(self.boundary_alpha),
            "film_grain_alpha": float(self.film_grain_alpha),
            "min_weight": float(self.min_weight),
            "max_weight": float(self.max_weight),
            "missing_sensitivity_weight": float(self.missing_sensitivity_weight),
            "use_variance_as_film_grain_capacity": bool(self.use_variance_as_film_grain_capacity),
        }


@dataclass(frozen=True)
class SensitivityScalar:
    """Resolved scalar sensitivity for one tensor."""

    value: float
    source_key: str
    reduction: str
    matched: bool


def _as_nonnegative_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BetaFisherWeightError(f"{label} must be a non-negative number")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise BetaFisherWeightError(f"{label} must be a finite non-negative number")
    return out


def _candidate_sensitivity_keys(name: str) -> list[str]:
    keys = [name]
    if name.endswith(".bias"):
        keys.append(f"{name[:-5]}.weight")
    if name.endswith(".weight"):
        keys.append(name[:-7])
    else:
        keys.append(f"{name}.weight")
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _vector_scalar(value: torch.Tensor, *, key: str, target: TensorWeightTarget) -> tuple[float, str]:
    vec = validate_sensitivity_vector(
        value.reshape(-1),
        expected_channels=int(value.numel()),
        name=key,
    )
    n = int(vec.numel())
    if n == 1:
        return float(vec.item()), "scalar"
    if n == target.out_channels:
        return float(vec.mean().item()), "mean_output_channels"
    if n == target.n_values:
        return float(vec.mean().item()), "mean_all_values"
    return float(vec.mean().item()), f"mean_vector_len_{n}"


def resolve_sensitivity_scalar(
    target: TensorWeightTarget,
    sensitivities: Mapping[str, torch.Tensor],
    *,
    missing_value: float = 1.0,
) -> SensitivityScalar:
    """Resolve a per-tensor scalar from a tensor or channel sensitivity map."""

    for key in _candidate_sensitivity_keys(target.name):
        value = sensitivities.get(key)
        if value is None:
            continue
        scalar, reduction = _vector_scalar(value, key=key, target=target)
        return SensitivityScalar(
            value=scalar,
            source_key=key,
            reduction=reduction,
            matched=True,
        )
    return SensitivityScalar(
        value=float(missing_value),
        source_key="",
        reduction="missing_sensitivity_default",
        matched=False,
    )


def _unit_mean(values: Sequence[float], *, floor: float = DEFAULT_MIN_WEIGHT) -> list[float]:
    if not values:
        return []
    arr = np.array([max(float(v), floor) for v in values], dtype=np.float64)
    mean = float(arr.mean())
    if not math.isfinite(mean) or mean <= 0.0:
        return [1.0] * len(values)
    return [float(v) for v in arr / mean]


def _unit_mean_positive(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    arr = np.array([max(float(v), 0.0) for v in values], dtype=np.float64)
    mean = float(arr.mean())
    if not math.isfinite(mean) or mean <= 0.0:
        return [0.0] * len(values)
    return [float(v) for v in arr / mean]


def _normalise_final_weights(
    raw_weights: Sequence[float],
    *,
    min_weight: float,
    max_weight: float,
) -> list[float]:
    if not raw_weights:
        return []
    clipped = [min(max(float(v), min_weight), max_weight) for v in raw_weights]
    unit = _unit_mean(clipped, floor=min_weight)
    return [min(max(float(v), min_weight), max_weight) for v in unit]


def _scalar_from_mapping_value(value: Any, *, value_key: str, label: str) -> float | None:
    if isinstance(value, Mapping):
        for key in (value_key, "value", "mass", "score_weight", "capacity"):
            if key in value:
                return _as_nonnegative_float(value[key], label=label)
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return _as_nonnegative_float(value, label=label)
    return None


def load_tensor_scalar_json(path: str | Path, *, value_key: str) -> dict[str, float]:
    """Load a flexible tensor->scalar JSON mapping.

    Accepted shapes:
      * ``{"tensor.name": 1.2}``
      * ``{"tensors": {"tensor.name": {"boundary_mass": 1.2}}}``
      * ``{"rows": [{"tensor_name": "tensor.name", "boundary_mass": 1.2}]}``
      * ``{"per_tensor": [{"name": "tensor.name", "value": 1.2}]}``
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: dict[str, float] = {}

    def add(name: Any, value: Any) -> None:
        tensor_name = str(name or "").strip()
        if not tensor_name:
            return
        scalar = _scalar_from_mapping_value(
            value,
            value_key=value_key,
            label=f"{Path(path)}:{tensor_name}",
        )
        if scalar is not None:
            rows[tensor_name] = scalar

    if isinstance(payload, Mapping):
        tensors = payload.get("tensors")
        if isinstance(tensors, Mapping):
            for name, value in tensors.items():
                add(name, value)
        for list_key in ("rows", "per_tensor", "tensors"):
            seq = payload.get(list_key)
            if not isinstance(seq, list):
                continue
            for row in seq:
                if not isinstance(row, Mapping):
                    continue
                name = row.get("tensor_name", row.get("name", row.get("tensor")))
                add(name, row)
        if not rows:
            for name, value in payload.items():
                if name in {"schema", "format", "metadata", "tool"}:
                    continue
                add(name, value)
    else:
        raise BetaFisherWeightError(f"{path}: expected JSON object")

    return rows


def build_tensor_weight_rows(
    targets: Sequence[TensorWeightTarget],
    sensitivities: Mapping[str, torch.Tensor],
    *,
    config: ScoreWeightConfig | None = None,
    boundary_mass: Mapping[str, float] | None = None,
    film_grain_capacity: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build allocator weights in the exact order of ``targets``."""

    cfg = config or ScoreWeightConfig()
    cfg.validate()
    if not targets:
        raise BetaFisherWeightError("at least one tensor target is required")

    boundary_mass = dict(boundary_mass or {})
    film_grain_capacity = dict(film_grain_capacity or {})
    local_variances = (
        compute_local_variance_proxy([target.symbols for target in targets if target.symbols is not None])
        if all(target.symbols is not None for target in targets)
        else [0.0] * len(targets)
    )
    if len(local_variances) != len(targets):
        local_variances = [0.0] * len(targets)

    sensitivity_scalars = [
        resolve_sensitivity_scalar(
            target,
            sensitivities,
            missing_value=cfg.missing_sensitivity_weight,
        )
        for target in targets
    ]
    sensitivity_values = [scalar.value for scalar in sensitivity_scalars]
    sensitivity_unit = _unit_mean(sensitivity_values, floor=cfg.min_weight)
    boundary_values = [float(boundary_mass.get(target.name, 0.0)) for target in targets]
    boundary_unit = _unit_mean_positive(boundary_values)
    explicit_grain_values = [float(film_grain_capacity.get(target.name, 0.0)) for target in targets]
    if any(value > 0.0 for value in explicit_grain_values):
        grain_values = explicit_grain_values
        grain_source = "explicit_film_grain_capacity_json"
    elif cfg.use_variance_as_film_grain_capacity:
        grain_values = list(local_variances)
        grain_source = "local_symbol_variance_proxy"
    else:
        grain_values = [0.0] * len(targets)
        grain_source = "disabled"
    grain_unit = _unit_mean_positive(grain_values)

    raw_weights: list[float] = []
    rows: list[dict[str, Any]] = []
    for idx, target in enumerate(targets):
        sensitivity_factor = sensitivity_unit[idx] ** cfg.fisher_beta
        boundary_factor = 1.0 + cfg.boundary_alpha * boundary_unit[idx]
        grain_factor = 1.0 / (1.0 + cfg.film_grain_alpha * grain_unit[idx])
        raw = sensitivity_factor * boundary_factor * grain_factor
        raw_weights.append(raw)
        rows.append(
            {
                "tensor_index": idx,
                "tensor_name": target.name,
                "shape": list(target.shape),
                "n_values": target.n_values,
                "sensitivity_raw": round(float(sensitivity_values[idx]), 12),
                "sensitivity_unit_mean": round(float(sensitivity_unit[idx]), 12),
                "sensitivity_source_key": sensitivity_scalars[idx].source_key,
                "sensitivity_reduction": sensitivity_scalars[idx].reduction,
                "sensitivity_matched": bool(sensitivity_scalars[idx].matched),
                "boundary_mass_raw": round(float(boundary_values[idx]), 12),
                "boundary_mass_unit_mean": round(float(boundary_unit[idx]), 12),
                "boundary_multiplier": round(float(boundary_factor), 12),
                "film_grain_capacity_raw": round(float(grain_values[idx]), 12),
                "film_grain_capacity_unit_mean": round(float(grain_unit[idx]), 12),
                "film_grain_capacity_source": grain_source,
                "film_grain_multiplier": round(float(grain_factor), 12),
                "local_symbol_variance": round(float(local_variances[idx]), 12),
                "raw_allocator_weight": round(float(raw), 12),
            }
        )

    weights = _normalise_final_weights(
        raw_weights,
        min_weight=cfg.min_weight,
        max_weight=cfg.max_weight,
    )
    for row, weight in zip(rows, weights, strict=True):
        row["allocator_weight"] = round(float(weight), 12)

    missing = [row["tensor_name"] for row in rows if not row["sensitivity_matched"]]
    blockers = []
    if missing:
        blockers.append("missing_sensitivity_for_some_tensors")
    if not any(value > 0.0 for value in sensitivity_values):
        blockers.append("sensitivity_values_have_no_positive_mass")

    return {
        "schema": "beta_fisher_tensor_weight_rows_v1",
        "config": cfg.to_dict(),
        "tensor_count": len(rows),
        "missing_sensitivity_count": len(missing),
        "missing_sensitivity_tensors": missing,
        "blockers": blockers,
        "allocator_input": {
            "tensor_order": [target.name for target in targets],
            "weights": [round(float(weight), 12) for weight in weights],
            "weight_semantics": "cost = bytes + lambda * weight[t] * rel_err[t]^2",
            "normalization": "unit_mean_after_beta_boundary_and_film_grain_factors",
        },
        "per_tensor": rows,
    }


def select_weighted_k_allocations(
    tensors: Sequence[TensorBlob],
    weights: Sequence[float],
    *,
    rms_targets: Sequence[float],
    k_range: Sequence[int],
    joint_encoder: JointEncoderHook | None = None,
) -> list[dict[str, Any]]:
    """Run the existing Lagrangian allocator with exported tensor weights."""

    if len(tensors) != len(weights):
        raise BetaFisherWeightError(
            f"weights length {len(weights)} does not match tensor count {len(tensors)}"
        )
    if not tensors:
        raise BetaFisherWeightError("at least one tensor is required")
    for weight in weights:
        if (
            isinstance(weight, bool)
            or not isinstance(weight, int | float)
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise BetaFisherWeightError("weights must be finite non-negative numbers")
    for target in rms_targets:
        if (
            isinstance(target, bool)
            or not isinstance(target, int | float)
            or not math.isfinite(float(target))
            or float(target) < 0.0
        ):
            raise BetaFisherWeightError("rms_targets must be non-negative numbers")
    k_values: list[int] = []
    for k in k_range:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise BetaFisherWeightError("k_range must contain positive integers")
        k_values.append(int(k))
    if not k_values:
        raise BetaFisherWeightError("k_range must contain positive integers")

    curves = precompute_per_tensor_K_curves(tensors, K_range=k_values)
    allocator = LagrangianPerTensorAllocator(weights=list(weights), joint_encoder=joint_encoder)
    allocations: list[dict[str, Any]] = []
    for target in rms_targets:
        result = allocator.bisect_for_rms_target(curves, float(target), lam_hi=1e15)
        selected_ks = [int(selection["K"]) for selection in result.selections]
        allocations.append(
            {
                "rms_target": float(target),
                "lambda": float(result.lam),
                "rel_err": float(result.rel_err),
                "total_bytes": int(result.total_bytes),
                "selected_Ks": selected_ks,
                "selected_K_by_tensor": [
                    {
                        "tensor_index": idx,
                        "tensor_name": tensor.name,
                        "K": selected_ks[idx],
                        "allocator_weight": round(float(weights[idx]), 12),
                        "rel_err": float(result.selections[idx]["rel_err"]),
                        "byte_proxy": int(result.selections[idx].get("byte_proxy", result.selections[idx].get("bytes", 0))),
                    }
                    for idx, tensor in enumerate(tensors)
                ],
                "joint_encoder_extras": dict(result.joint_extras),
            }
        )
    return allocations


def sensitivity_artifact_status(
    sensitivity_path: str | Path,
    metadata: Mapping[str, Any],
    *,
    allow_diagnostic: bool,
) -> dict[str, Any]:
    """Return fail-closed status for the sensitivity map input."""

    blockers = real_sensitivity_metadata_blockers(metadata)
    status = "diagnostic_allowed" if blockers and allow_diagnostic else "passed"
    if blockers and not allow_diagnostic:
        status = "blocked"
    return {
        "path": str(sensitivity_path),
        "metadata_blockers": blockers,
        "allow_diagnostic_sensitivity": bool(allow_diagnostic),
        "status": status,
    }


def load_sensitivity_map_for_weight_export(
    sensitivity_path: str | Path,
    *,
    allow_diagnostic: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Load a sensitivity map and classify whether it is real or diagnostic."""

    sensitivities, metadata = load_sensitivity_map(sensitivity_path)
    status = sensitivity_artifact_status(
        sensitivity_path,
        metadata,
        allow_diagnostic=allow_diagnostic,
    )
    if status["status"] == "blocked":
        joined = "; ".join(status["metadata_blockers"])
        raise SensitivityMapError(
            f"{sensitivity_path}: diagnostic/stub sensitivity rejected: {joined}"
        )
    return sensitivities, metadata, status


__all__ = [
    "MODULE_NAME",
    "SCHEMA_VERSION",
    "BetaFisherWeightError",
    "ScoreWeightConfig",
    "SensitivityScalar",
    "TensorWeightTarget",
    "build_tensor_weight_rows",
    "load_sensitivity_map_for_weight_export",
    "load_tensor_scalar_json",
    "resolve_sensitivity_scalar",
    "select_weighted_k_allocations",
    "sensitivity_artifact_status",
]
