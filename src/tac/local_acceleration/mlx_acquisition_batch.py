# SPDX-License-Identifier: MIT
"""Group local MLX scorer/training evidence into acquisition operation sets."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from tac.local_acceleration import (
    EVIDENCE_GRADE_MLX,
    EVIDENCE_TAG_MLX,
    MLX_ACQUISITION_BATCH_OPERATION_SET_SCHEMA,
    MLX_ACQUISITION_BATCH_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)

SCHEMA_VERSION = MLX_ACQUISITION_BATCH_SCHEMA
OPERATION_SET_SCHEMA = MLX_ACQUISITION_BATCH_OPERATION_SET_SCHEMA
MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA = (
    "mlx_effective_spend_triage_candidate_selection.v1"
)
MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_ROW_SCHEMA = (
    "mlx_effective_spend_triage_candidate_row.v1"
)
TOOL_NAME = "tac.local_acceleration.mlx_acquisition_batch"


class MLXAcquisitionBatchError(ValueError):
    """Raised when local MLX acquisition batch input is malformed."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _false_authority(row: Mapping[str, Any], *blockers: str) -> dict[str, Any]:
    return apply_proxy_evidence_boundary(
        {
            **dict(row),
            **PROXY_FALSE_AUTHORITY_FIELDS,
            "candidate_generation_only": True,
            "planning_only": True,
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "evidence_tag": EVIDENCE_TAG_MLX,
            "score_axis": EVIDENCE_TAG_MLX,
        },
        dispatch_blockers=blockers,
    )


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MLXAcquisitionBatchError(f"{label} must be an object")
    return dict(value)


def _list_rows(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        raise MLXAcquisitionBatchError(f"{label} must be a non-empty list")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        out.append(_as_mapping(item, label=f"{label}[{index}]"))
    return out


def _text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise MLXAcquisitionBatchError(f"{label} is required")
    return text


def _float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise MLXAcquisitionBatchError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MLXAcquisitionBatchError(f"{label} must be numeric") from exc
    if minimum is not None and parsed < minimum:
        raise MLXAcquisitionBatchError(f"{label} must be >= {minimum}")
    return parsed


def _int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise MLXAcquisitionBatchError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MLXAcquisitionBatchError(f"{label} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise MLXAcquisitionBatchError(f"{label} must be >= {minimum}")
    return parsed


def _int_list(value: Any, *, label: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise MLXAcquisitionBatchError(f"{label} must be a list")
    return [_int(item, f"{label}[{index}]", minimum=0) for index, item in enumerate(value)]


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_") or "row"


def _selection_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    require_no_truthy_authority_fields(selection, context="mlx_acquisition_selection")
    if selection.get("schema") != MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA:
        raise MLXAcquisitionBatchError(
            "selection schema must be "
            f"{MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA}"
        )
    if selection.get("candidate_generation_only") is not True:
        raise MLXAcquisitionBatchError("selection candidate_generation_only must be true")
    if selection.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise MLXAcquisitionBatchError(
            f"selection evidence_grade must be {EVIDENCE_GRADE_MLX}"
        )
    if selection.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise MLXAcquisitionBatchError(
            f"selection evidence_tag must be {EVIDENCE_TAG_MLX}"
        )
    rows = _list_rows(selection.get("selected_rows"), label="selection.selected_rows")
    for index, row in enumerate(rows):
        require_no_truthy_authority_fields(
            row,
            context=f"mlx_acquisition_selection.selected_rows[{index}]",
        )
        if row.get("schema") != MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_ROW_SCHEMA:
            raise MLXAcquisitionBatchError(f"selection row {index} schema mismatch")
        if row.get("candidate_generation_only") is not True:
            raise MLXAcquisitionBatchError(
                f"selection row {index} candidate_generation_only must be true"
            )
    return rows


def _row_gain(row: Mapping[str, Any], *, index: int) -> float:
    gain = _float(
        row.get("normalized_full_video_scorer_gain_vs_baseline"),
        f"selected_rows[{index}].normalized_full_video_scorer_gain_vs_baseline",
        minimum=0.0,
    )
    if gain <= 0.0:
        raise MLXAcquisitionBatchError(
            f"selected_rows[{index}] normalized_full_video_scorer_gain_vs_baseline must be positive"
        )
    return gain


def _operation_set_from_selection_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    set_index: int,
    source_path: str | None,
) -> dict[str, Any]:
    if not rows:
        raise MLXAcquisitionBatchError("operation set rows must not be empty")
    selected_operations: list[dict[str, Any]] = []
    selected_unit_ids: list[str] = []
    pair_indices: list[int] = []
    expected_score_gain = 0.0
    candidate_saved_bytes = 0
    families: list[str] = []
    row_refs: list[dict[str, Any]] = []
    for offset, row in enumerate(rows):
        row_index = set_index + offset
        row_id = _text(row.get("row_id") or row.get("candidate_id"), "selection row_id")
        candidate_id = _text(row.get("candidate_id"), "selection candidate_id")
        family = str(row.get("family") or "mlx_scorer_response")
        families.append(family)
        unit_id = f"mlx_row_{_slug(row_id)}"
        selected_unit_ids.append(unit_id)
        row_pairs = _int_list(
            row.get("pair_indices") or row.get("source_pair_window"),
            label=f"selected_rows[{row_index}].pair_indices",
        )
        pair_indices.extend(row_pairs)
        expected_score_gain += _row_gain(row, index=row_index)
        added_bytes = int(row.get("added_archive_bytes") or 0)
        if added_bytes < 0:
            candidate_saved_bytes += abs(added_bytes)
        selected_operations.append(
            {
                "operation_id": f"materialize_mlx_response_{_slug(row_id)}",
                "operation_family": "materialize_scorer_response_candidate",
                "unit_id": unit_id,
                "unit_kind": "scorer_response_row",
                "target_kind": "mlx_scorer_response_candidate_v1",
                "candidate_id": candidate_id,
                "candidate_saved_bytes": max(0, -added_bytes),
                "predicted_quality_score_delta": -_row_gain(row, index=row_index),
                "pair_indices": row_pairs,
                "params": {
                    "source_row_id": row_id,
                    "source_path": row.get("source_path"),
                    "archive_sha256": row.get("archive_sha256"),
                    "raw_sha256": row.get("raw_sha256"),
                },
                "blockers": [
                    "mlx_acquisition_operation_requires_candidate_materializer",
                    "requires_exact_auth_eval_before_score_claim",
                ],
                **PROXY_FALSE_AUTHORITY_FIELDS,
            }
        )
        row_refs.append(
            {
                "row_id": row_id,
                "candidate_id": candidate_id,
                "family": family,
                "source_path": row.get("source_path"),
            }
        )
    first = rows[0]
    source_candidate_id = _text(first.get("candidate_id"), "selection candidate_id")
    operation_set_id = f"mlx_opset_{set_index:04d}_{_slug(source_candidate_id)}"
    return _false_authority(
        {
            "schema": OPERATION_SET_SCHEMA,
            "operation_set_id": operation_set_id,
            "candidate_id": source_candidate_id,
            "operation_set_rank": set_index + 1,
            "resource_kind": "local_mlx",
            "component": "scorer",
            "source_path": source_path,
            "source_schema": MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA,
            "operation_families": ["materialize_scorer_response_candidate"],
            "source_families": sorted(set(families)),
            "selected_unit_ids": selected_unit_ids,
            "selected_operations": selected_operations,
            "chosen_operation_sequence": [dict(item) for item in selected_operations],
            "chosen_operation_sequence_source": "mlx_acquisition_batch_order",
            "active_interactions": [],
            "pair_indices": sorted(set(pair_indices)),
            "candidate_saved_bytes": candidate_saved_bytes,
            "expected_score_gain": expected_score_gain,
            "expected_delta_score": -expected_score_gain,
            "quality_cost_score": -expected_score_gain,
            "row_refs": row_refs,
        },
        "mlx_acquisition_batch_is_planning_only",
        "requires_byte_closed_materialization_before_dispatch",
        "requires_exact_auth_eval_before_score_claim",
    )


def build_mlx_acquisition_batch_from_selection(
    selection: Mapping[str, Any],
    *,
    source_path: str | None = None,
    set_size: int = 1,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return grouped MLX acquisition operation sets from strict selection rows."""

    if isinstance(set_size, bool) or set_size < 1:
        raise MLXAcquisitionBatchError("set_size must be >= 1")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise MLXAcquisitionBatchError("limit must be >= 1 when provided")
    rows = _selection_rows(selection)
    selected = rows[:limit] if limit is not None else rows
    operation_sets = [
        _operation_set_from_selection_rows(
            selected[index : index + set_size],
            set_index=index // set_size,
            source_path=source_path,
        )
        for index in range(0, len(selected), set_size)
    ]
    return _false_authority(
        {
            "schema": SCHEMA_VERSION,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "source_schema": selection.get("schema"),
            "source_path": source_path,
            "set_size": set_size,
            "source_row_count": len(rows),
            "operation_set_count": len(operation_sets),
            "operation_sets": operation_sets,
            "summary": {
                "operation_set_count": len(operation_sets),
                "selected_operation_count": sum(
                    len(item["selected_operations"]) for item in operation_sets
                ),
                "expected_score_gain_sum": sum(
                    float(item["expected_score_gain"]) for item in operation_sets
                ),
                "candidate_saved_bytes_sum": sum(
                    int(item["candidate_saved_bytes"]) for item in operation_sets
                ),
                "resource_kinds": sorted(
                    {str(item.get("resource_kind") or "") for item in operation_sets}
                ),
            },
        },
        "mlx_acquisition_batch_is_planning_only",
        "requires_byte_closed_materialization_before_dispatch",
        "requires_exact_auth_eval_before_score_claim",
    )


def validate_mlx_acquisition_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a MLX acquisition batch."""

    require_no_truthy_authority_fields(batch, context="mlx_acquisition_batch")
    if batch.get("schema") != SCHEMA_VERSION:
        raise MLXAcquisitionBatchError(f"batch schema must be {SCHEMA_VERSION}")
    if batch.get("candidate_generation_only") is not True:
        raise MLXAcquisitionBatchError("batch candidate_generation_only must be true")
    if batch.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise MLXAcquisitionBatchError(f"batch evidence_grade must be {EVIDENCE_GRADE_MLX}")
    if batch.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise MLXAcquisitionBatchError(f"batch evidence_tag must be {EVIDENCE_TAG_MLX}")
    operation_sets = _list_rows(batch.get("operation_sets"), label="batch.operation_sets")
    normalized_sets: list[dict[str, Any]] = []
    for index, operation_set in enumerate(operation_sets):
        require_no_truthy_authority_fields(
            operation_set,
            context=f"mlx_acquisition_batch.operation_sets[{index}]",
        )
        if operation_set.get("schema") != OPERATION_SET_SCHEMA:
            raise MLXAcquisitionBatchError(f"operation set {index} schema mismatch")
        if operation_set.get("candidate_generation_only") is not True:
            raise MLXAcquisitionBatchError(
                f"operation set {index} candidate_generation_only must be true"
            )
        if not _list_rows(
            operation_set.get("selected_operations"),
            label=f"operation_sets[{index}].selected_operations",
        ):
            raise MLXAcquisitionBatchError(
                f"operation_sets[{index}].selected_operations must not be empty"
            )
        normalized_sets.append(dict(operation_set))
    return {**dict(batch), "operation_sets": normalized_sets}


__all__ = [
    "OPERATION_SET_SCHEMA",
    "SCHEMA_VERSION",
    "TOOL_NAME",
    "MLXAcquisitionBatchError",
    "build_mlx_acquisition_batch_from_selection",
    "validate_mlx_acquisition_batch",
]
