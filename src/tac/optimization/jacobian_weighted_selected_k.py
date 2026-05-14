# SPDX-License-Identifier: MIT
"""Jacobian-weighted selected-K producer scaffold.

This module consumes an externally produced per-tensor or per-channel
importance manifest, checks that the manifest is CUDA-authored and not marked
as diagnostic/proxy data, then runs the canonical
:class:`JacobianWeightedAllocator` over K curves.  It emits
``weighted_k_allocations[].selected_Ks`` in the same planning schema consumed
by the no-dead-K archive builder.

The producer does not load a scorer, build an archive, launch GPU work, or make
score/rank/kill claims.  It is a CPU-safe bridge from future CUDA pullback
artifacts into the existing byte-closed builder interface.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.codec.cost_curves import TensorBlob, precompute_per_tensor_K_curves
from tac.optimization.lagrangian_per_tensor_allocation import (
    JacobianWeightedAllocator,
    JointEncoderHook,
)
from tac.repo_io import sha256_file
from tac.sensitivity_map import (
    SensitivityMapError,
    real_sensitivity_metadata_blockers,
    require_authoritative_device,
)

SCHEMA_VERSION = "jacobian_weighted_selected_k_allocations.v1"
MODULE_NAME = "tac.optimization.jacobian_weighted_selected_k"
EVIDENCE_GRADE = "[CPU-planning jacobian-weighted selected-K producer]"
EVIDENCE_SEMANTICS = "cpu_jacobian_importance_selected_k_producer_no_score_no_dispatch"

DEFAULT_DISPATCH_BLOCKERS = [
    "selected_Ks_cpu_planning_not_score_authority",
    "requires_byte_closed_no_dead_K_archive_rebuild",
    "requires_static_archive_preflight",
    "requires_exact_cuda_auth_eval_before_score_claim",
]

_SCALAR_IMPORTANCE_KEYS = (
    "importance",
    "jacobian_importance",
    "score_importance",
    "pullback_importance",
    "value",
    "mass",
    "weight",
)
_CHANNEL_IMPORTANCE_KEYS = (
    "channel_importance",
    "per_channel_importance",
    "channels",
    "values",
    "per_channel",
)
_NAME_KEYS = ("tensor_name", "name", "tensor", "key")
_TEXT_DIAGNOSTIC_MARKERS = (
    "advisory-only",
    "advisory_only",
    "debug",
    "diagnostic",
    "dummy",
    "fake",
    "local_proxy",
    "non_authoritative",
    "non-promotable",
    "non_promotable",
    "placeholder",
    "planning-only",
    "planning_only",
    "proxy",
    "smoke",
    "stub",
    "synthetic",
)
_BOOL_DIAGNOSTIC_KEYS = {
    "advisory_only",
    "debug",
    "diagnostic",
    "dummy",
    "fake",
    "is_debug",
    "is_planning",
    "is_stub",
    "local_proxy",
    "non_authoritative",
    "non_promotable",
    "planning",
    "planning_only",
    "proxy",
    "proxy_only",
    "smoke",
    "stub",
    "synthetic",
}
_DEVICE_KEYS = (
    "device",
    "producer_device",
    "source_device",
    "pullback_device",
    "jacobian_device",
)


class JacobianSelectedKError(ValueError):
    """Raised when an importance manifest cannot drive selected-K output."""


@dataclass(frozen=True)
class ResolvedImportance:
    """Importance vector resolved in tensor order."""

    importance: list[float]
    texture_capacity: list[float] | None
    per_tensor: list[dict[str, Any]]
    metadata_gate: dict[str, Any]


def _utc_iso() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_importance_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON importance manifest and require an object payload."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JacobianSelectedKError(f"{path}: importance manifest must be a JSON object")
    return payload


def _flatten_mapping(payload: Mapping[str, Any], prefix: str = ""):
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _flatten_mapping(value, full_key)
        else:
            yield full_key, value


def _metadata_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw_metadata = payload.get("metadata")
    if isinstance(raw_metadata, Mapping):
        metadata.update({str(k): v for k, v in raw_metadata.items()})
    for key in (
        "device",
        "producer_device",
        "source_device",
        "pullback_device",
        "jacobian_device",
        "component",
        "component_scope",
        "kind",
        "mode",
        "source",
        "source_kind",
        "status",
        "tag",
    ):
        if key in payload and key not in metadata:
            metadata[key] = payload[key]
    return metadata


def _metadata_value(metadata: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in metadata:
            return metadata[key]
    for container in ("producer", "source", "pullback", "jacobian", "certification"):
        nested = metadata.get(container)
        if not isinstance(nested, Mapping):
            continue
        for key in keys:
            if key in nested:
                return nested[key]
    return None


def _diagnostic_metadata_blockers(metadata: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for full_key, value in _flatten_mapping(metadata):
        key = full_key.rsplit(".", 1)[-1].lower().replace("-", "_")
        if key in _BOOL_DIAGNOSTIC_KEYS and value is True:
            blockers.append(f"{full_key}=true")
        if isinstance(value, str):
            value_norm = value.lower().replace("_", "-")
            for marker in _TEXT_DIAGNOSTIC_MARKERS:
                if marker.replace("_", "-") in value_norm:
                    blockers.append(f"{full_key} contains {marker!r}")
                    break
    return blockers


def validate_importance_manifest_metadata(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless manifest metadata is CUDA-authored and non-proxy."""

    metadata = _metadata_payload(payload)
    if not metadata:
        raise JacobianSelectedKError(
            "importance manifest metadata is required for CUDA/proxy gating"
        )

    blockers = []
    blockers.extend(real_sensitivity_metadata_blockers(metadata))
    blockers.extend(_diagnostic_metadata_blockers(metadata))

    device = _metadata_value(metadata, _DEVICE_KEYS)
    try:
        require_authoritative_device(device)
    except SensitivityMapError as exc:
        blockers.append(f"cuda_device_gate rejected: {exc}")

    if blockers:
        joined = "; ".join(sorted(set(blockers)))
        raise JacobianSelectedKError(f"importance manifest rejected: {joined}")

    return {
        "status": "passed",
        "device": str(device),
        "metadata_blockers": [],
        "metadata": dict(metadata),
    }


def _as_nonnegative_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise JacobianSelectedKError(f"{label} must be a finite non-negative number")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise JacobianSelectedKError(f"{label} must be a finite non-negative number")
    return out


def _numeric_sequence(value: Any, *, label: str) -> list[float] | None:
    if not isinstance(value, list | tuple):
        return None
    out = [_as_nonnegative_float(item, label=f"{label}[]") for item in value]
    if not out:
        raise JacobianSelectedKError(f"{label} must not be empty")
    return out


def _scalar_from_value(
    value: Any,
    *,
    label: str,
    scalar_keys: Sequence[str] = _SCALAR_IMPORTANCE_KEYS,
    channel_keys: Sequence[str] = _CHANNEL_IMPORTANCE_KEYS,
) -> tuple[float, str, int]:
    if isinstance(value, Mapping):
        for key in scalar_keys:
            if key in value:
                return (
                    _as_nonnegative_float(value[key], label=f"{label}.{key}"),
                    f"scalar:{key}",
                    1,
                )
        for key in channel_keys:
            if key in value:
                vec = _numeric_sequence(value[key], label=f"{label}.{key}")
                if vec is None:
                    raise JacobianSelectedKError(f"{label}.{key} must be a numeric list")
                return float(np.mean(vec)), f"mean_channels:{key}", len(vec)
        raise JacobianSelectedKError(
            f"{label} lacks one of {list(scalar_keys)} or {list(channel_keys)}"
        )
    vec = _numeric_sequence(value, label=label)
    if vec is not None:
        return float(np.mean(vec)), "mean_channels:direct_list", len(vec)
    return _as_nonnegative_float(value, label=label), "scalar:direct", 1


def _row_name(row: Mapping[str, Any], *, label: str) -> str:
    for key in _NAME_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise JacobianSelectedKError(f"{label} lacks tensor name")


def _candidate_keys(name: str) -> list[str]:
    candidates = [name]
    if name.endswith(".bias"):
        candidates.append(f"{name[:-5]}.weight")
    if name.endswith(".weight"):
        candidates.append(name[:-7])
    else:
        candidates.append(f"{name}.weight")
    out: list[str] = []
    seen: set[str] = set()
    for key in candidates:
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _collect_scalar_rows(
    payload: Mapping[str, Any],
    *,
    section_keys: Sequence[str],
    scalar_keys: Sequence[str] = _SCALAR_IMPORTANCE_KEYS,
    channel_keys: Sequence[str] = _CHANNEL_IMPORTANCE_KEYS,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    def add(name: str, value: Any, source: str) -> None:
        if name in rows:
            raise JacobianSelectedKError(f"duplicate importance entry for tensor {name!r}")
        scalar, reduction, n_values = _scalar_from_value(
            value,
            label=source,
            scalar_keys=scalar_keys,
            channel_keys=channel_keys,
        )
        rows[name] = {
            "source_key": name,
            "source_section": source,
            "value": scalar,
            "reduction": reduction,
            "source_value_count": n_values,
        }

    for section_key in section_keys:
        section = payload.get(section_key)
        if isinstance(section, Mapping):
            for name, value in section.items():
                add(str(name), value, f"{section_key}.{name}")
        elif isinstance(section, list):
            for idx, item in enumerate(section):
                if not isinstance(item, Mapping):
                    raise JacobianSelectedKError(f"{section_key}[{idx}] must be an object")
                name = _row_name(item, label=f"{section_key}[{idx}]")
                add(name, item, f"{section_key}[{idx}]")
        elif section is not None:
            raise JacobianSelectedKError(f"{section_key} must be an object or list")

    return rows


def _resolve_optional_capacity(
    payload: Mapping[str, Any],
    tensor_names: Sequence[str],
) -> tuple[list[float] | None, dict[str, dict[str, Any]]]:
    rows = _collect_scalar_rows(
        payload,
        section_keys=("texture_capacity", "film_grain_capacity", "capacity"),
        scalar_keys=("texture_capacity", "film_grain_capacity", "capacity", "value"),
        channel_keys=("channel_capacity", "per_channel_capacity", "channels", "values"),
    )
    if not rows:
        return None, {}

    out: list[float] = []
    resolved: dict[str, dict[str, Any]] = {}
    for name in tensor_names:
        source = next((key for key in _candidate_keys(name) if key in rows), None)
        if source is None:
            raise JacobianSelectedKError(
                f"capacity section is present but lacks tensor {name!r}"
            )
        row = rows[source]
        out.append(float(row["value"]))
        resolved[name] = row
    return out, resolved


def resolve_importance_manifest(
    payload: Mapping[str, Any],
    tensor_names: Sequence[str],
    *,
    reject_uniform: bool = True,
) -> ResolvedImportance:
    """Resolve a CUDA-gated importance manifest into tensor-order scalars."""

    if not tensor_names:
        raise JacobianSelectedKError("at least one tensor name is required")
    metadata_gate = validate_importance_manifest_metadata(payload)
    rows = _collect_scalar_rows(
        payload,
        section_keys=("per_tensor", "per_channel", "tensors", "importance", "rows"),
    )
    if not rows:
        raise JacobianSelectedKError(
            "importance manifest must contain per_tensor, per_channel, tensors, importance, or rows"
        )

    importance: list[float] = []
    per_tensor: list[dict[str, Any]] = []
    for idx, name in enumerate(tensor_names):
        source = next((key for key in _candidate_keys(name) if key in rows), None)
        if source is None:
            raise JacobianSelectedKError(f"importance manifest lacks tensor {name!r}")
        row = rows[source]
        value = float(row["value"])
        importance.append(value)
        per_tensor.append(
            {
                "tensor_index": idx,
                "tensor_name": name,
                "importance_raw": round(value, 12),
                "importance_source_key": row["source_key"],
                "importance_source_section": row["source_section"],
                "importance_reduction": row["reduction"],
                "importance_source_value_count": int(row["source_value_count"]),
            }
        )

    arr = np.asarray(importance, dtype=np.float64)
    if not np.isfinite(arr).all() or (arr < 0.0).any():
        raise JacobianSelectedKError("importance values must be finite and non-negative")
    if float(arr.sum()) <= 0.0:
        raise JacobianSelectedKError("importance manifest has no positive scorer signal")
    if reject_uniform and arr.size > 1 and np.allclose(arr, arr[0], rtol=0.0, atol=1e-12):
        raise JacobianSelectedKError(
            "importance manifest is uniform; dummy/proxy importance is non-promotable"
        )

    capacity, capacity_rows = _resolve_optional_capacity(payload, tensor_names)
    if capacity is not None:
        for row in per_tensor:
            cap = capacity_rows[row["tensor_name"]]
            row["texture_capacity_raw"] = round(float(cap["value"]), 12)
            row["texture_capacity_source_key"] = cap["source_key"]
            row["texture_capacity_reduction"] = cap["reduction"]

    return ResolvedImportance(
        importance=[float(value) for value in importance],
        texture_capacity=capacity,
        per_tensor=per_tensor,
        metadata_gate=metadata_gate,
    )


def _validate_numeric_targets(values: Sequence[float], *, label: str) -> list[float]:
    out: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise JacobianSelectedKError(f"{label} must contain finite non-negative numbers")
        out.append(float(value))
    if not out:
        raise JacobianSelectedKError(f"{label} must not be empty")
    return out


def _validate_k_range(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise JacobianSelectedKError("k_range must contain positive integers")
        out.append(int(value))
    if not out:
        raise JacobianSelectedKError("k_range must contain positive integers")
    return out


def select_jacobian_weighted_k_allocations(
    tensors: Sequence[TensorBlob],
    importance: Sequence[float],
    *,
    rms_targets: Sequence[float],
    k_range: Sequence[int],
    texture_capacity: Sequence[float] | None = None,
    joint_encoder: JointEncoderHook | None = None,
    curves: Sequence[Sequence[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Run ``JacobianWeightedAllocator`` and emit no-dead-K selected_K rows."""

    if not tensors:
        raise JacobianSelectedKError("at least one tensor is required")
    if len(importance) != len(tensors):
        raise JacobianSelectedKError(
            f"importance length {len(importance)} does not match tensor count {len(tensors)}"
        )
    if texture_capacity is not None and len(texture_capacity) != len(tensors):
        raise JacobianSelectedKError(
            "texture_capacity length "
            f"{len(texture_capacity)} does not match tensor count {len(tensors)}"
        )
    rms_values = _validate_numeric_targets(rms_targets, label="rms_targets")
    k_values = _validate_k_range(k_range)
    curve_rows = list(curves) if curves is not None else precompute_per_tensor_K_curves(tensors, K_range=k_values)
    if len(curve_rows) != len(tensors):
        raise JacobianSelectedKError(
            f"curves length {len(curve_rows)} does not match tensor count {len(tensors)}"
        )

    allocator = JacobianWeightedAllocator(
        importance=list(importance),
        texture_capacity=list(texture_capacity) if texture_capacity is not None else None,
        joint_encoder=joint_encoder,
    )
    weights = allocator.importance_weights
    allocations: list[dict[str, Any]] = []
    for target in rms_values:
        result = allocator.bisect_for_rms_target(curve_rows, target, lam_hi=1e15)
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
                        "importance_raw": round(float(importance[idx]), 12),
                        "allocator_weight": round(float(weights[idx]), 12),
                        "rel_err": float(result.selections[idx]["rel_err"]),
                        "byte_proxy": int(
                            result.selections[idx].get(
                                "byte_proxy",
                                result.selections[idx].get("bytes", 0),
                            )
                        ),
                    }
                    for idx, tensor in enumerate(tensors)
                ],
                "joint_encoder_extras": dict(result.joint_extras),
            }
        )
    return allocations, weights


def build_jacobian_selected_k_manifest(
    *,
    tensors: Sequence[TensorBlob],
    importance_payload: Mapping[str, Any],
    rms_targets: Sequence[float],
    k_range: Sequence[int],
    joint_encoder: JointEncoderHook | None = None,
    curves: Sequence[Sequence[dict[str, Any]]] | None = None,
    importance_manifest_path: str | Path | None = None,
    producer_tool: str | None = None,
    extra_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a planning manifest with weighted_k_allocations[].selected_Ks."""

    tensor_names = [tensor.name for tensor in tensors]
    resolved = resolve_importance_manifest(importance_payload, tensor_names)
    weighted_allocations, allocator_weights = select_jacobian_weighted_k_allocations(
        tensors,
        resolved.importance,
        rms_targets=rms_targets,
        k_range=k_range,
        texture_capacity=resolved.texture_capacity,
        joint_encoder=joint_encoder,
        curves=curves,
    )

    inputs: dict[str, Any] = {
        "rms_targets": [float(value) for value in rms_targets],
        "K_range": [int(min(k_range)), int(max(k_range))],
    }
    if importance_manifest_path is not None:
        path = Path(importance_manifest_path)
        inputs["importance_manifest"] = str(path)
        if path.is_file():
            inputs["importance_manifest_sha256"] = sha256_file(path)
    if extra_inputs:
        inputs.update(dict(extra_inputs))

    per_tensor_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(resolved.per_tensor):
        merged = dict(row)
        merged["allocator_weight"] = round(float(allocator_weights[idx]), 12)
        per_tensor_rows.append(merged)

    return {
        "schema": SCHEMA_VERSION,
        "tool": producer_tool,
        "module": MODULE_NAME,
        "created_utc": _utc_iso(),
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "downstream_selected_Ks_can_change_charged_bits": True,
        "family_falsified": False,
        "falsification_scope": "none_selected_K_planning_only",
        "dispatch_blockers": list(DEFAULT_DISPATCH_BLOCKERS),
        "inputs": inputs,
        "importance_artifact": {
            "source_schema": importance_payload.get("schema", importance_payload.get("format")),
            **resolved.metadata_gate,
        },
        "tensor_importance": {
            "tensor_count": len(per_tensor_rows),
            "importance_semantics": "higher value increases JacobianWeightedAllocator protection",
            "weight_semantics": "cost = bytes + lambda * weight[t] * rel_err[t]^2",
            "normalization": "JacobianWeightedAllocator unit-mean normalized importance",
            "allocator_input": {
                "tensor_order": tensor_names,
                "importance": [round(float(v), 12) for v in resolved.importance],
                "texture_capacity": (
                    [round(float(v), 12) for v in resolved.texture_capacity]
                    if resolved.texture_capacity is not None
                    else None
                ),
                "weights": [round(float(v), 12) for v in allocator_weights],
            },
            "per_tensor": per_tensor_rows,
        },
        "weighted_k_allocations": weighted_allocations,
        "integration_point": {
            "target_tool": "tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py",
            "selected_Ks_field": "weighted_k_allocations[].selected_Ks",
            "next_exact_gate": (
                "Rebuild a byte-closed no-dead-K archive from this selected_Ks row, "
                "run static pre-submission compliance, then run exact CUDA auth eval "
                "with a dispatch claim before any score/promotion/rank/kill status."
            ),
        },
    }


__all__ = [
    "DEFAULT_DISPATCH_BLOCKERS",
    "EVIDENCE_GRADE",
    "EVIDENCE_SEMANTICS",
    "MODULE_NAME",
    "SCHEMA_VERSION",
    "JacobianSelectedKError",
    "ResolvedImportance",
    "build_jacobian_selected_k_manifest",
    "load_importance_manifest",
    "resolve_importance_manifest",
    "select_jacobian_weighted_k_allocations",
    "validate_importance_manifest_metadata",
]
