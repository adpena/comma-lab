"""Jacobian/Fisher importance-weighted quantization allocator.

This module is a CPU-side planning primitive for the "pixel is weight is
deterministic" lane.  It consumes externally produced per-weight or per-tensor
importance values plus per-tensor byte/error candidate curves, then selects one
quantization/coarsening candidate per tensor under either:

* a weighted distortion budget, or
* a total byte budget.

It intentionally does not compute scorer gradients, load the scorer, build an
archive, launch GPU work, or claim score movement.  Outputs are tagged as
planning evidence only and carry blockers for CUDA pixel-Jacobian pullback and
exact auth eval before any promotion/ranking decision.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "jacobian_fisher_importance_allocator.v1"
MODULE_NAME = "tac.optimization.jacobian_fisher_importance_allocator"
EVIDENCE_GRADE = "[CPU/MPS/proxy-planning empirical/prediction jacobian-fisher-importance allocator]"
EVIDENCE_SEMANTICS = (
    "cpu_mps_proxy_importance_weighted_quantization_allocation_no_score_no_dispatch"
)

DEFAULT_DISPATCH_BLOCKERS = [
    "cpu_planning_allocator_not_score_authority",
    "cpu_mps_proxy_importance_inputs_not_score_authority",
    "requires_cuda_pixel_jacobian_or_fisher_pullback",
    "requires_importance_artifact_custody_and_calibration",
    "requires_byte_closed_archive_rebuild",
    "requires_static_pre_submission_compliance",
    "requires_exact_cuda_auth_eval_before_score_claim",
    "requires_exact_archive_cuda_score_custody_before_rank_promotion_or_kill",
]

_PLANNING_ONLY_FALSE_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "dispatch_attempted",
    "score_affecting_payload_changed",
    "charged_bits_changed",
    "family_falsified",
)
_DISALLOWED_DECISION_USES = [
    "score_claim",
    "leaderboard_or_frontier_ranking",
    "candidate_promotion",
    "lane_or_family_kill",
    "exact_eval_dispatch_readiness",
]
_REQUIRED_SCORE_CUSTODY_BLOCKERS = (
    "cpu_mps_proxy_importance_inputs_not_score_authority",
    "requires_exact_archive_cuda_score_custody_before_rank_promotion_or_kill",
)

_BYTE_KEYS = ("bytes", "byte_proxy", "rate_bytes")
_ERROR_KEYS = ("rel_err", "error", "distortion", "rms_error")
_CHOICE_KEYS = ("K", "precision", "bits", "bit_depth", "choice", "candidate_id", "id")
_TENSOR_NAME_KEYS = ("tensor_name", "name", "tensor", "key")
_CURVE_KEYS = ("candidates", "curve", "rows", "choices")
_IMPORTANCE_NAME_KEYS = ("tensor_name", "name", "tensor", "key")
_IMPORTANCE_VALUE_KEYS = (
    "importance",
    "weight",
    "fisher",
    "fisher_importance",
    "jacobian_importance",
    "boundary_mass",
    "value",
    "mass",
)


class ImportanceAllocationError(ValueError):
    """Raised when allocator inputs are malformed or infeasible."""


@dataclass(frozen=True)
class ImportanceConfig:
    """Configuration for converting raw importance into protection weights."""

    fisher_beta: float = 1.0
    boundary_alpha: float = 0.5
    texture_capacity_alpha: float = 0.25
    min_weight: float = 1e-9
    max_weight: float = 1e9
    target_mean: float = 1.0
    per_weight_reducer: str = "mean"

    def validate(self) -> None:
        for label, value in {
            "fisher_beta": self.fisher_beta,
            "boundary_alpha": self.boundary_alpha,
            "texture_capacity_alpha": self.texture_capacity_alpha,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "target_mean": self.target_mean,
        }.items():
            _finite_float(value, label=label)
        if self.fisher_beta < 0.0:
            raise ImportanceAllocationError("fisher_beta must be non-negative")
        if self.boundary_alpha < 0.0:
            raise ImportanceAllocationError("boundary_alpha must be non-negative")
        if self.texture_capacity_alpha < 0.0:
            raise ImportanceAllocationError("texture_capacity_alpha must be non-negative")
        if self.min_weight <= 0.0:
            raise ImportanceAllocationError("min_weight must be positive")
        if self.max_weight < self.min_weight:
            raise ImportanceAllocationError("max_weight must be >= min_weight")
        if self.target_mean <= 0.0:
            raise ImportanceAllocationError("target_mean must be positive")
        if self.per_weight_reducer not in {"mean", "sum", "max", "rms"}:
            raise ImportanceAllocationError(
                "per_weight_reducer must be one of mean, sum, max, rms"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fisher_beta": float(self.fisher_beta),
            "boundary_alpha": float(self.boundary_alpha),
            "texture_capacity_alpha": float(self.texture_capacity_alpha),
            "min_weight": float(self.min_weight),
            "max_weight": float(self.max_weight),
            "target_mean": float(self.target_mean),
            "per_weight_reducer": self.per_weight_reducer,
        }


@dataclass(frozen=True)
class TensorCandidate:
    """Normalized byte/error candidate for one tensor."""

    tensor_name: str
    candidate_id: str
    bytes: int
    error: float
    row: dict[str, Any]
    original_index: int

    def selected_row(self) -> dict[str, Any]:
        out = dict(self.row)
        out["candidate_id"] = self.candidate_id
        out["bytes"] = int(self.bytes)
        out["error"] = float(self.error)
        return out


@dataclass(frozen=True)
class TensorImportanceRow:
    """Per-tensor raw importance and normalized allocator weight."""

    tensor_name: str
    importance_raw: float
    importance_unit_mean: float
    boundary_mass_raw: float
    boundary_mass_unit_mean: float
    texture_capacity_raw: float
    texture_capacity_unit_mean: float
    raw_allocator_weight: float
    allocator_weight: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "importance_raw": round(float(self.importance_raw), 12),
            "importance_unit_mean": round(float(self.importance_unit_mean), 12),
            "boundary_mass_raw": round(float(self.boundary_mass_raw), 12),
            "boundary_mass_unit_mean": round(float(self.boundary_mass_unit_mean), 12),
            "texture_capacity_raw": round(float(self.texture_capacity_raw), 12),
            "texture_capacity_unit_mean": round(float(self.texture_capacity_unit_mean), 12),
            "raw_allocator_weight": round(float(self.raw_allocator_weight), 12),
            "allocator_weight": round(float(self.allocator_weight), 12),
            "source": self.source,
        }


@dataclass(frozen=True)
class ImportanceWeights:
    """Allocator weights in tensor order."""

    tensor_names: list[str]
    weights: list[float]
    rows: list[TensorImportanceRow]
    config: ImportanceConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "importance_allocator_weights_v1",
            "config": self.config.to_dict(),
            "allocator_input": {
                "tensor_order": list(self.tensor_names),
                "weights": [round(float(value), 12) for value in self.weights],
                "weight_semantics": "higher weight protects that tensor from quantization error",
                "cost_semantics": (
                    "distortion objective uses weighted squared candidate error; "
                    "compatible with LagrangianPerTensorAllocator weight convention"
                ),
            },
            "per_tensor": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class AllocationPlan:
    """Selected quantization/coarsening candidates plus non-promotable metadata."""

    objective: str
    lambda_value: float
    tensor_names: list[str]
    weights: list[float]
    selections: list[TensorCandidate]
    weighted_rms_error: float
    unweighted_rms_error: float
    max_error: float
    total_bytes: int
    target_distortion: float | None = None
    byte_budget: int | None = None
    iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        selected_rows: list[dict[str, Any]] = []
        for idx, (name, weight, candidate) in enumerate(
            zip(self.tensor_names, self.weights, self.selections, strict=True)
        ):
            row = {
                "tensor_index": idx,
                "tensor_name": name,
                "allocator_weight": round(float(weight), 12),
                "candidate_id": candidate.candidate_id,
                "bytes": int(candidate.bytes),
                "error": float(candidate.error),
                "weighted_error_sq": float(weight) * float(candidate.error) ** 2,
            }
            for key in _CHOICE_KEYS:
                if key in candidate.row:
                    row[key] = candidate.row[key]
            selected_rows.append(row)
        return {
            "objective": self.objective,
            "lambda": float(self.lambda_value),
            "target_distortion": self.target_distortion,
            "byte_budget": self.byte_budget,
            "iterations": int(self.iterations),
            "total_bytes": int(self.total_bytes),
            "weighted_rms_error": float(self.weighted_rms_error),
            "unweighted_rms_error": float(self.unweighted_rms_error),
            "max_error": float(self.max_error),
            "selected_by_tensor": selected_rows,
            "selected_candidates": [candidate.selected_row() for candidate in self.selections],
            "metadata": dict(self.metadata),
        }


def _utc_iso() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _planning_only_metadata(*, curve_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    blockers = list(DEFAULT_DISPATCH_BLOCKERS)
    metadata: dict[str, Any] = {
        "planning_only": True,
        "proxy_or_diagnostic_evidence": True,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "evidence_device_class": "cpu_mps_proxy_planning_only",
        "score_authority": "none",
        "score_authority_required": "exact_cuda_auth_eval_archive_path",
        "promotion_requires_evidence_grade": "A/A++ exact CUDA archive evidence",
        "downstream_selection_can_change_charged_bits": True,
        "falsification_scope": "none_cpu_mps_proxy_allocator_planning_only",
        "disallowed_decision_uses": list(_DISALLOWED_DECISION_USES),
        "dispatch_blockers": list(blockers),
        "score_claim_blockers": list(blockers),
    }
    for key in _PLANNING_ONLY_FALSE_FIELDS:
        metadata[key] = False
    if curve_summary is not None:
        metadata["curve_summary"] = dict(curve_summary)
    return metadata


def _assert_planning_only_score_custody(section: Mapping[str, Any], *, label: str) -> None:
    for key in _PLANNING_ONLY_FALSE_FIELDS:
        if section.get(key) is not False:
            raise ImportanceAllocationError(f"{label}.{key} must remain false")
    if section.get("planning_only") is not True:
        raise ImportanceAllocationError(f"{label}.planning_only must remain true")
    if section.get("score_authority") != "none":
        raise ImportanceAllocationError(f"{label}.score_authority must remain none")
    disallowed = section.get("disallowed_decision_uses")
    if not isinstance(disallowed, Sequence) or isinstance(disallowed, (str, bytes)):
        raise ImportanceAllocationError(f"{label}.disallowed_decision_uses must be a list")
    for use in _DISALLOWED_DECISION_USES:
        if use not in disallowed:
            raise ImportanceAllocationError(
                f"{label}.disallowed_decision_uses missing {use!r}"
            )
    for blocker_key in ("dispatch_blockers", "score_claim_blockers"):
        blockers = section.get(blocker_key)
        if not isinstance(blockers, Sequence) or isinstance(blockers, (str, bytes)):
            raise ImportanceAllocationError(f"{label}.{blocker_key} must be a list")
        for blocker in _REQUIRED_SCORE_CUSTODY_BLOCKERS:
            if blocker not in blockers:
                raise ImportanceAllocationError(
                    f"{label}.{blocker_key} missing {blocker!r}"
                )


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ImportanceAllocationError(f"{label} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ImportanceAllocationError(f"{label} must be finite")
    return out


def _nonnegative_float(value: Any, *, label: str) -> float:
    out = _finite_float(value, label=label)
    if out < 0.0:
        raise ImportanceAllocationError(f"{label} must be non-negative")
    return out


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ImportanceAllocationError(f"{label} must be a non-negative integer")
    if value < 0:
        raise ImportanceAllocationError(f"{label} must be non-negative")
    return int(value)


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, Any]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return "", None


def _name_from_mapping(mapping: Mapping[str, Any], *, label: str) -> str:
    key, value = _first_present(mapping, _TENSOR_NAME_KEYS)
    if not key or not isinstance(value, str) or not value.strip():
        raise ImportanceAllocationError(f"{label} lacks tensor name")
    return value.strip()


def _candidate_id(row: Mapping[str, Any], *, fallback: int) -> str:
    for key in _CHOICE_KEYS:
        if key in row:
            return f"{key}={row[key]}"
    return f"candidate_index={fallback}"


def _normalize_candidate_row(
    tensor_name: str,
    row: Mapping[str, Any],
    *,
    index: int,
) -> TensorCandidate:
    bytes_key, bytes_value = _first_present(row, _BYTE_KEYS)
    error_key, error_value = _first_present(row, _ERROR_KEYS)
    if not bytes_key:
        raise ImportanceAllocationError(f"{tensor_name}[{index}] lacks bytes/byte_proxy")
    if not error_key:
        raise ImportanceAllocationError(f"{tensor_name}[{index}] lacks rel_err/error")
    byte_count = _nonnegative_int(bytes_value, label=f"{tensor_name}[{index}].{bytes_key}")
    error = _nonnegative_float(error_value, label=f"{tensor_name}[{index}].{error_key}")
    normalized = dict(row)
    normalized["source_bytes_key"] = bytes_key
    normalized["source_error_key"] = error_key
    return TensorCandidate(
        tensor_name=tensor_name,
        candidate_id=_candidate_id(row, fallback=index),
        bytes=byte_count,
        error=error,
        row=normalized,
        original_index=index,
    )


def _prune_dominated(candidates: list[TensorCandidate]) -> list[TensorCandidate]:
    kept: list[TensorCandidate] = []
    for idx, candidate in enumerate(candidates):
        dominated = False
        for jdx, other in enumerate(candidates):
            if idx == jdx:
                continue
            if (
                other.bytes <= candidate.bytes
                and other.error <= candidate.error
                and (other.bytes < candidate.bytes or other.error < candidate.error)
            ):
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    if not kept:
        raise ImportanceAllocationError("all candidates were dominated")
    return kept


def normalize_candidate_curves(
    candidate_curves: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]],
    *,
    prune_dominated: bool = True,
) -> tuple[list[str], list[list[TensorCandidate]], dict[str, Any]]:
    """Validate and normalize flexible per-tensor byte/error curves.

    Accepted shapes:

    * ``{"tensor.name": [{"K": 1, "byte_proxy": 100, "rel_err": 0.0}]}``
    * ``[{"tensor_name": "tensor.name", "candidates": [...]}]``
    """

    tensor_names: list[str] = []
    raw_rows: list[tuple[str, Sequence[Mapping[str, Any]]]] = []
    if isinstance(candidate_curves, Mapping):
        for name, rows in candidate_curves.items():
            raw_rows.append((str(name), rows))
    elif isinstance(candidate_curves, Sequence) and not isinstance(candidate_curves, str | bytes):
        for idx, entry in enumerate(candidate_curves):
            if not isinstance(entry, Mapping):
                raise ImportanceAllocationError(f"candidate_curves[{idx}] must be an object")
            name = _name_from_mapping(entry, label=f"candidate_curves[{idx}]")
            curve_key, rows = _first_present(entry, _CURVE_KEYS)
            if not curve_key:
                raise ImportanceAllocationError(f"candidate_curves[{idx}] lacks candidates/curve/rows")
            raw_rows.append((name, rows))
    else:
        raise ImportanceAllocationError("candidate_curves must be a mapping or sequence")

    if not raw_rows:
        raise ImportanceAllocationError("candidate_curves must not be empty")

    curves: list[list[TensorCandidate]] = []
    candidate_counts_before: list[int] = []
    candidate_counts_after: list[int] = []
    seen: set[str] = set()
    for tensor_idx, (name, rows) in enumerate(raw_rows):
        if not name:
            raise ImportanceAllocationError(f"candidate_curves[{tensor_idx}] has empty tensor name")
        if name in seen:
            raise ImportanceAllocationError(f"duplicate candidate curve for tensor {name!r}")
        seen.add(name)
        if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
            raise ImportanceAllocationError(f"{name} candidate rows must be a sequence")
        if not rows:
            raise ImportanceAllocationError(f"{name} candidate rows must not be empty")
        normalized: list[TensorCandidate] = []
        for row_idx, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ImportanceAllocationError(f"{name}[{row_idx}] must be an object")
            normalized.append(_normalize_candidate_row(name, row, index=row_idx))
        candidate_counts_before.append(len(normalized))
        if prune_dominated:
            normalized = _prune_dominated(normalized)
        candidate_counts_after.append(len(normalized))
        tensor_names.append(name)
        curves.append(normalized)

    return tensor_names, curves, {
        "tensor_count": len(tensor_names),
        "candidate_count_before_prune": int(sum(candidate_counts_before)),
        "candidate_count_after_prune": int(sum(candidate_counts_after)),
        "prune_dominated": bool(prune_dominated),
    }


def _as_scalar_mapping(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    label: str,
    allow_missing: bool = True,
) -> dict[str, float]:
    if value is None:
        if allow_missing:
            return {}
        raise ImportanceAllocationError(f"{label} is required")
    rows: dict[str, float] = {}

    def add(name: Any, raw: Any, source: str) -> None:
        tensor_name = str(name or "").strip()
        if not tensor_name:
            raise ImportanceAllocationError(f"{source} lacks tensor name")
        if tensor_name in rows:
            raise ImportanceAllocationError(f"duplicate {label} entry for {tensor_name!r}")
        if isinstance(raw, Mapping):
            _, raw_value = _first_present(raw, _IMPORTANCE_VALUE_KEYS)
            if raw_value is None:
                raise ImportanceAllocationError(f"{source} lacks scalar importance value")
        else:
            raw_value = raw
        rows[tensor_name] = _nonnegative_float(raw_value, label=source)

    if isinstance(value, Mapping):
        for name, raw in value.items():
            add(name, raw, f"{label}.{name}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for idx, row in enumerate(value):
            if not isinstance(row, Mapping):
                raise ImportanceAllocationError(f"{label}[{idx}] must be an object")
            name = _name_from_mapping(row, label=f"{label}[{idx}]")
            add(name, row, f"{label}[{idx}]")
    else:
        raise ImportanceAllocationError(f"{label} must be a mapping or list")
    return rows


def reduce_per_weight_importance(
    per_weight_importance: Mapping[str, Sequence[float] | np.ndarray],
    *,
    reducer: str = "mean",
) -> dict[str, float]:
    """Reduce per-weight non-negative importance arrays to tensor scalars."""

    if reducer not in {"mean", "sum", "max", "rms"}:
        raise ImportanceAllocationError("reducer must be one of mean, sum, max, rms")
    if not per_weight_importance:
        raise ImportanceAllocationError("per_weight_importance must not be empty")
    out: dict[str, float] = {}
    for name, values in per_weight_importance.items():
        tensor_name = str(name or "").strip()
        if not tensor_name:
            raise ImportanceAllocationError("per_weight_importance contains empty tensor name")
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ImportanceAllocationError(f"per_weight_importance[{tensor_name!r}] is empty")
        if not np.isfinite(arr).all():
            raise ImportanceAllocationError(
                f"per_weight_importance[{tensor_name!r}] contains non-finite values"
            )
        if (arr < 0.0).any():
            raise ImportanceAllocationError(
                f"per_weight_importance[{tensor_name!r}] must be non-negative"
            )
        if reducer == "mean":
            scalar = float(arr.mean())
        elif reducer == "sum":
            scalar = float(arr.sum())
        elif reducer == "max":
            scalar = float(arr.max())
        else:
            scalar = float(math.sqrt(float(np.mean(arr * arr))))
        out[tensor_name] = scalar
    return out


def _unit_mean(values: Sequence[float], *, floor: float) -> list[float]:
    if not values:
        return []
    arr = np.asarray([max(float(value), floor) for value in values], dtype=np.float64)
    mean = float(arr.mean())
    if not math.isfinite(mean) or mean <= 0.0:
        raise ImportanceAllocationError("cannot normalize non-positive importance mean")
    return [float(value) for value in arr / mean]


def _unit_mean_positive(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray([max(float(value), 0.0) for value in values], dtype=np.float64)
    mean = float(arr.mean())
    if not math.isfinite(mean) or mean <= 0.0:
        return [0.0] * len(values)
    return [float(value) for value in arr / mean]


def _normalize_final_weights(
    raw_weights: Sequence[float],
    *,
    min_weight: float,
    max_weight: float,
    target_mean: float,
) -> list[float]:
    if not raw_weights:
        return []
    clipped = [min(max(float(value), min_weight), max_weight) for value in raw_weights]
    unit = _unit_mean(clipped, floor=min_weight)
    scaled = [float(value) * target_mean for value in unit]
    return [min(max(value, min_weight), max_weight) for value in scaled]


def build_importance_weights(
    tensor_names: Sequence[str],
    *,
    per_tensor_importance: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    per_weight_importance: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    boundary_mass: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    texture_capacity: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    config: ImportanceConfig | None = None,
) -> ImportanceWeights:
    """Build unit-mean protection weights in the provided tensor order."""

    cfg = config or ImportanceConfig()
    cfg.validate()
    names = [str(name) for name in tensor_names]
    if not names:
        raise ImportanceAllocationError("at least one tensor name is required")
    if len(set(names)) != len(names):
        raise ImportanceAllocationError("tensor names must be unique")
    if per_tensor_importance is None and per_weight_importance is None:
        raise ImportanceAllocationError(
            "per_tensor_importance or per_weight_importance is required"
        )
    if per_tensor_importance is not None and per_weight_importance is not None:
        raise ImportanceAllocationError(
            "provide per_tensor_importance or per_weight_importance, not both"
        )

    if per_weight_importance is not None:
        raw_importance = reduce_per_weight_importance(
            per_weight_importance,
            reducer=cfg.per_weight_reducer,
        )
        source = f"per_weight:{cfg.per_weight_reducer}"
    else:
        raw_importance = _as_scalar_mapping(
            per_tensor_importance,
            label="per_tensor_importance",
            allow_missing=False,
        )
        source = "per_tensor"

    missing = [name for name in names if name not in raw_importance]
    if missing:
        raise ImportanceAllocationError(f"importance missing tensor(s): {missing}")

    raw_values = [float(raw_importance[name]) for name in names]
    if not any(value > 0.0 for value in raw_values):
        raise ImportanceAllocationError("importance must contain at least one positive value")

    boundary = _as_scalar_mapping(boundary_mass, label="boundary_mass")
    capacity = _as_scalar_mapping(texture_capacity, label="texture_capacity")
    boundary_values = [float(boundary.get(name, 0.0)) for name in names]
    capacity_values = [float(capacity.get(name, 0.0)) for name in names]

    importance_unit = _unit_mean(raw_values, floor=cfg.min_weight)
    boundary_unit = _unit_mean_positive(boundary_values)
    capacity_unit = _unit_mean_positive(capacity_values)

    raw_weights: list[float] = []
    rows: list[TensorImportanceRow] = []
    for idx, name in enumerate(names):
        importance_factor = importance_unit[idx] ** float(cfg.fisher_beta)
        boundary_factor = 1.0 + float(cfg.boundary_alpha) * boundary_unit[idx]
        capacity_factor = 1.0 + float(cfg.texture_capacity_alpha) * capacity_unit[idx]
        raw_weight = importance_factor * boundary_factor / max(capacity_factor, cfg.min_weight)
        raw_weights.append(raw_weight)
        rows.append(
            TensorImportanceRow(
                tensor_name=name,
                importance_raw=raw_values[idx],
                importance_unit_mean=importance_unit[idx],
                boundary_mass_raw=boundary_values[idx],
                boundary_mass_unit_mean=boundary_unit[idx],
                texture_capacity_raw=capacity_values[idx],
                texture_capacity_unit_mean=capacity_unit[idx],
                raw_allocator_weight=raw_weight,
                allocator_weight=0.0,
                source=source,
            )
        )

    weights = _normalize_final_weights(
        raw_weights,
        min_weight=cfg.min_weight,
        max_weight=cfg.max_weight,
        target_mean=cfg.target_mean,
    )
    rows = [
        TensorImportanceRow(
            tensor_name=row.tensor_name,
            importance_raw=row.importance_raw,
            importance_unit_mean=row.importance_unit_mean,
            boundary_mass_raw=row.boundary_mass_raw,
            boundary_mass_unit_mean=row.boundary_mass_unit_mean,
            texture_capacity_raw=row.texture_capacity_raw,
            texture_capacity_unit_mean=row.texture_capacity_unit_mean,
            raw_allocator_weight=row.raw_allocator_weight,
            allocator_weight=weights[idx],
            source=row.source,
        )
        for idx, row in enumerate(rows)
    ]
    return ImportanceWeights(
        tensor_names=names,
        weights=[float(value) for value in weights],
        rows=rows,
        config=cfg,
    )


def _allocation_metrics(
    selections: Sequence[TensorCandidate],
    weights: Sequence[float],
) -> tuple[int, float, float, float]:
    total_bytes = int(sum(candidate.bytes for candidate in selections))
    errors = [float(candidate.error) for candidate in selections]
    if not errors:
        return total_bytes, 0.0, 0.0, 0.0
    denom = float(sum(weights))
    if denom <= 0.0 or not math.isfinite(denom):
        raise ImportanceAllocationError("allocator weights must have positive finite sum")
    weighted_rms = math.sqrt(
        sum(float(weight) * err * err for weight, err in zip(weights, errors, strict=True))
        / denom
    )
    unweighted_rms = math.sqrt(sum(err * err for err in errors) / len(errors))
    return total_bytes, float(weighted_rms), float(unweighted_rms), float(max(errors))


def _select_candidates(
    curves: Sequence[Sequence[TensorCandidate]],
    weights: Sequence[float],
    *,
    lam: float,
    objective: str,
) -> list[TensorCandidate]:
    selections: list[TensorCandidate] = []
    for tensor_curve, weight in zip(curves, weights, strict=True):
        if objective == "target_distortion":
            chosen = min(
                tensor_curve,
                key=lambda candidate: (
                    candidate.bytes + lam * float(weight) * candidate.error**2,
                    float(weight) * candidate.error**2,
                    candidate.bytes,
                    candidate.original_index,
                ),
            )
        elif objective == "byte_budget":
            chosen = min(
                tensor_curve,
                key=lambda candidate: (
                    float(weight) * candidate.error**2 + lam * candidate.bytes,
                    float(weight) * candidate.error**2,
                    candidate.bytes,
                    candidate.original_index,
                ),
            )
        else:  # pragma: no cover - internal guard
            raise ImportanceAllocationError(f"unknown objective {objective!r}")
        selections.append(chosen)
    return selections


def _make_plan(
    *,
    objective: str,
    lam: float,
    tensor_names: Sequence[str],
    weights: Sequence[float],
    selections: Sequence[TensorCandidate],
    target_distortion: float | None,
    byte_budget: int | None,
    iterations: int,
    metadata: Mapping[str, Any],
) -> AllocationPlan:
    total_bytes, weighted_rms, unweighted_rms, max_error = _allocation_metrics(
        selections,
        weights,
    )
    return AllocationPlan(
        objective=objective,
        lambda_value=float(lam),
        tensor_names=list(tensor_names),
        weights=[float(value) for value in weights],
        selections=list(selections),
        weighted_rms_error=weighted_rms,
        unweighted_rms_error=unweighted_rms,
        max_error=max_error,
        total_bytes=total_bytes,
        target_distortion=target_distortion,
        byte_budget=byte_budget,
        iterations=int(iterations),
        metadata=dict(metadata),
    )


def _best_quality_selection(
    curves: Sequence[Sequence[TensorCandidate]],
) -> list[TensorCandidate]:
    return [
        min(curve, key=lambda candidate: (candidate.error, candidate.bytes, candidate.original_index))
        for curve in curves
    ]


def _minimum_byte_selection(
    curves: Sequence[Sequence[TensorCandidate]],
) -> list[TensorCandidate]:
    return [
        min(curve, key=lambda candidate: (candidate.bytes, candidate.error, candidate.original_index))
        for curve in curves
    ]


def _validate_budget_args(
    *,
    target_distortion: float | None,
    byte_budget: int | None,
) -> tuple[float | None, int | None, str]:
    if (target_distortion is None) == (byte_budget is None):
        raise ImportanceAllocationError(
            "provide exactly one of target_distortion or byte_budget"
        )
    if target_distortion is not None:
        target = _nonnegative_float(target_distortion, label="target_distortion")
        return target, None, "target_distortion"
    budget = _nonnegative_int(byte_budget, label="byte_budget")
    return None, budget, "byte_budget"


def allocate_importance_weighted_candidates(
    candidate_curves: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]],
    *,
    per_tensor_importance: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    per_weight_importance: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    boundary_mass: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    texture_capacity: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    target_distortion: float | None = None,
    byte_budget: int | None = None,
    config: ImportanceConfig | None = None,
    max_iter: int = 80,
    tol: float = 1e-12,
    prune_dominated: bool = True,
) -> tuple[AllocationPlan, ImportanceWeights, dict[str, Any]]:
    """Select one candidate per tensor under a weighted distortion or byte cap."""

    if max_iter < 1:
        raise ImportanceAllocationError("max_iter must be positive")
    if tol <= 0.0 or not math.isfinite(float(tol)):
        raise ImportanceAllocationError("tol must be finite and positive")
    target, budget, objective = _validate_budget_args(
        target_distortion=target_distortion,
        byte_budget=byte_budget,
    )
    tensor_names, curves, curve_summary = normalize_candidate_curves(
        candidate_curves,
        prune_dominated=prune_dominated,
    )
    importance_weights = build_importance_weights(
        tensor_names,
        per_tensor_importance=per_tensor_importance,
        per_weight_importance=per_weight_importance,
        boundary_mass=boundary_mass,
        texture_capacity=texture_capacity,
        config=config,
    )
    weights = importance_weights.weights
    metadata = _planning_only_metadata(curve_summary=curve_summary)
    metadata["schema"] = "importance_weighted_candidate_allocation_v1"
    _assert_planning_only_score_custody(metadata, label="allocation.metadata")

    if objective == "target_distortion":
        assert target is not None
        best_quality = _best_quality_selection(curves)
        min_plan = _make_plan(
            objective=objective,
            lam=float("inf"),
            tensor_names=tensor_names,
            weights=weights,
            selections=best_quality,
            target_distortion=target,
            byte_budget=None,
            iterations=0,
            metadata=metadata,
        )
        if min_plan.weighted_rms_error > target + tol:
            raise ImportanceAllocationError(
                "target_distortion is infeasible: "
                f"minimum weighted_rms_error={min_plan.weighted_rms_error:.12g} "
                f"> target={target:.12g}"
            )
        lam_lo = 0.0
        selections_lo = _select_candidates(curves, weights, lam=lam_lo, objective=objective)
        plan_lo = _make_plan(
            objective=objective,
            lam=lam_lo,
            tensor_names=tensor_names,
            weights=weights,
            selections=selections_lo,
            target_distortion=target,
            byte_budget=None,
            iterations=0,
            metadata=metadata,
        )
        if plan_lo.weighted_rms_error <= target + tol:
            return plan_lo, importance_weights, curve_summary

        lam_hi = 1.0
        best: AllocationPlan | None = None
        grow_iters = 0
        while grow_iters < 128:
            selections_hi = _select_candidates(curves, weights, lam=lam_hi, objective=objective)
            plan_hi = _make_plan(
                objective=objective,
                lam=lam_hi,
                tensor_names=tensor_names,
                weights=weights,
                selections=selections_hi,
                target_distortion=target,
                byte_budget=None,
                iterations=grow_iters + 1,
                metadata=metadata,
            )
            if plan_hi.weighted_rms_error <= target + tol:
                best = plan_hi
                break
            lam_lo = lam_hi
            lam_hi *= 2.0
            grow_iters += 1
        if best is None:
            raise ImportanceAllocationError("failed to bracket target_distortion")

        iterations = grow_iters + 1
        for _ in range(max_iter):
            iterations += 1
            mid = 0.5 * (lam_lo + lam_hi)
            selections_mid = _select_candidates(curves, weights, lam=mid, objective=objective)
            plan_mid = _make_plan(
                objective=objective,
                lam=mid,
                tensor_names=tensor_names,
                weights=weights,
                selections=selections_mid,
                target_distortion=target,
                byte_budget=None,
                iterations=iterations,
                metadata=metadata,
            )
            if plan_mid.weighted_rms_error <= target + tol:
                best = plan_mid
                lam_hi = mid
            else:
                lam_lo = mid
            if abs(lam_hi - lam_lo) <= tol * max(1.0, abs(lam_hi)):
                break
        assert best is not None
        return best, importance_weights, curve_summary

    assert budget is not None
    min_bytes = _minimum_byte_selection(curves)
    min_bytes_plan = _make_plan(
        objective=objective,
        lam=float("inf"),
        tensor_names=tensor_names,
        weights=weights,
        selections=min_bytes,
        target_distortion=None,
        byte_budget=budget,
        iterations=0,
        metadata=metadata,
    )
    if min_bytes_plan.total_bytes > budget:
        raise ImportanceAllocationError(
            "byte_budget is infeasible: "
            f"minimum total_bytes={min_bytes_plan.total_bytes} > byte_budget={budget}"
        )

    lam_lo = 0.0
    best_quality = _select_candidates(curves, weights, lam=lam_lo, objective=objective)
    plan_lo = _make_plan(
        objective=objective,
        lam=lam_lo,
        tensor_names=tensor_names,
        weights=weights,
        selections=best_quality,
        target_distortion=None,
        byte_budget=budget,
        iterations=0,
        metadata=metadata,
    )
    if plan_lo.total_bytes <= budget:
        return plan_lo, importance_weights, curve_summary

    lam_hi = 1.0
    best = None
    grow_iters = 0
    while grow_iters < 128:
        selections_hi = _select_candidates(curves, weights, lam=lam_hi, objective=objective)
        plan_hi = _make_plan(
            objective=objective,
            lam=lam_hi,
            tensor_names=tensor_names,
            weights=weights,
            selections=selections_hi,
            target_distortion=None,
            byte_budget=budget,
            iterations=grow_iters + 1,
            metadata=metadata,
        )
        if plan_hi.total_bytes <= budget:
            best = plan_hi
            break
        lam_lo = lam_hi
        lam_hi *= 2.0
        grow_iters += 1
    if best is None:
        raise ImportanceAllocationError("failed to bracket byte_budget")

    iterations = grow_iters + 1
    for _ in range(max_iter):
        iterations += 1
        mid = 0.5 * (lam_lo + lam_hi)
        selections_mid = _select_candidates(curves, weights, lam=mid, objective=objective)
        plan_mid = _make_plan(
            objective=objective,
            lam=mid,
            tensor_names=tensor_names,
            weights=weights,
            selections=selections_mid,
            target_distortion=None,
            byte_budget=budget,
            iterations=iterations,
            metadata=metadata,
        )
        if plan_mid.total_bytes <= budget:
            best = plan_mid
            lam_hi = mid
        else:
            lam_lo = mid
        if abs(lam_hi - lam_lo) <= tol * max(1.0, abs(lam_hi)):
            break
    assert best is not None
    return best, importance_weights, curve_summary


def build_importance_allocation_manifest(
    candidate_curves: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]],
    *,
    per_tensor_importance: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    per_weight_importance: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    boundary_mass: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    texture_capacity: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    target_distortion: float | None = None,
    byte_budget: int | None = None,
    config: ImportanceConfig | None = None,
    producer_tool: str | None = None,
    extra_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-promotable planning manifest around an allocation result."""

    plan, weights, curve_summary = allocate_importance_weighted_candidates(
        candidate_curves,
        per_tensor_importance=per_tensor_importance,
        per_weight_importance=per_weight_importance,
        boundary_mass=boundary_mass,
        texture_capacity=texture_capacity,
        target_distortion=target_distortion,
        byte_budget=byte_budget,
        config=config,
    )
    score_custody = _planning_only_metadata()
    manifest = {
        "schema": SCHEMA_VERSION,
        "module": MODULE_NAME,
        "tool": producer_tool,
        "created_utc": _utc_iso(),
        **score_custody,
        "dispatch_blockers": list(DEFAULT_DISPATCH_BLOCKERS),
        "score_claim_blockers": list(DEFAULT_DISPATCH_BLOCKERS),
        "inputs": dict(extra_inputs or {}),
        "curve_summary": curve_summary,
        "importance": weights.to_dict(),
        "allocation": plan.to_dict(),
        "integration_point": {
            "current_allocator": "tac.optimization.lagrangian_per_tensor_allocation",
            "lossy_coarsening_curve_fields": ["K", "byte_proxy", "rel_err"],
            "selected_output_field": "allocation.selected_by_tensor",
            "next_archive_builder_requirement": (
                "Consume selected K/precision values in a byte-closed archive rebuild; "
                "record old/new payload SHA-256 and no-op proof before exact CUDA auth eval."
            ),
        },
        "pixel_jacobian_cuda_blockers": [
            "compute deterministic decoder VJP/JVP on CUDA for official pixel/component target",
            "reduce per-weight or per-channel pullback to tensor-order importance",
            "calibrate top selected perturbations with byte-level finite differences",
            "rebuild charged archive bytes and run exact CUDA auth eval before ranking",
        ],
    }
    _assert_planning_only_score_custody(manifest, label="manifest")
    allocation = manifest["allocation"]
    if not isinstance(allocation, Mapping):
        raise ImportanceAllocationError("manifest.allocation must be an object")
    metadata = allocation.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ImportanceAllocationError("manifest.allocation.metadata must be an object")
    _assert_planning_only_score_custody(metadata, label="manifest.allocation.metadata")
    return manifest


__all__ = [
    "DEFAULT_DISPATCH_BLOCKERS",
    "EVIDENCE_GRADE",
    "EVIDENCE_SEMANTICS",
    "MODULE_NAME",
    "SCHEMA_VERSION",
    "AllocationPlan",
    "ImportanceAllocationError",
    "ImportanceConfig",
    "ImportanceWeights",
    "TensorCandidate",
    "TensorImportanceRow",
    "allocate_importance_weighted_candidates",
    "build_importance_allocation_manifest",
    "build_importance_weights",
    "normalize_candidate_curves",
    "reduce_per_weight_importance",
]
