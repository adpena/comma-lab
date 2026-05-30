# SPDX-License-Identifier: MIT
"""Adapt strict MLX spend-triage selections into learned-sweep candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.exact_eval_custody import CONTEST_EXACT_SAMPLE_COUNT
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.mlx_dynamic_learned_sweep import (
    FALSE_AUTHORITY,
    QUALITY_EVIDENCE_SCHEMA,
)
from tac.optimization.mlx_effective_spend_triage_selection import (
    ROW_SCHEMA as SELECTION_ROW_SCHEMA,
)
from tac.optimization.mlx_effective_spend_triage_selection import (
    SCHEMA as SELECTION_SCHEMA,
)
from tac.optimization.normalized_objective import (
    NormalizedObjectiveError,
    require_normalized_full_video_objective,
)
from tac.optimization.proxy_candidate_contract import (
    require_no_truthy_authority_fields,
)
from tac.repo_io import write_json_artifact

SCHEMA = "mlx_effective_spend_triage_learned_sweep_candidates.v1"
ROW_SCHEMA = "mlx_effective_spend_triage_learned_sweep_candidate.v1"
REFUSAL_SCHEMA = "mlx_effective_spend_triage_learned_sweep_refusal.v1"
TOOL = "tac.optimization.mlx_effective_spend_triage_learned_sweep_adapter"
CLI_TOOL = "tools/adapt_mlx_effective_spend_triage_to_learned_sweep.py"
DEFAULT_VARIANCE_FLOOR = 2.5e-10


class MLXEffectiveSpendTriageLearnedSweepAdapterError(ValueError):
    """Raised when MLX spend-triage evidence is not safe for learned sweep."""


def dumps_json(payload: Mapping[str, Any]) -> str:
    """Return deterministic JSON text for adapter payloads."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    allow_overwrite: bool = False,
    expected_existing_sha256: str | None = None,
) -> None:
    """Write deterministic JSON through the guarded artifact writer."""

    write_json_artifact(
        path,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_existing_sha256,
    )


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object with adapter-specific validation errors."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{path}: expected JSON object"
        )
    return payload


def file_sha256(path: Path) -> str:
    """Return SHA-256 for a source artifact consumed by this adapter."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_artifact_metadata(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Build deterministic source metadata for adapter manifests."""

    out: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        out[label] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return out


def build_refusal_payload(
    *,
    error: str,
    source_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a durable fail-closed refusal artifact for operator audit."""

    return {
        "schema": REFUSAL_SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "status": "refused",
        "error": str(error),
        "source_artifacts": dict(source_artifacts or {}),
        "allowed_use": "fail_closed_adapter_refusal_no_score_or_dispatch_authority",
    }


def build_mlx_effective_spend_triage_learned_sweep_candidates(
    selection: Mapping[str, Any],
    *,
    incumbent_score: float,
    top_k: int | None = None,
    source_artifacts: Mapping[str, Any] | None = None,
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
) -> dict[str, Any]:
    """Adapt normalized MLX spend-triage rows into learned-sweep candidates.

    The output is deliberately non-authoritative. It exists only to let the
    dynamic learned-sweep planner consume strict, normalized MLX quality
    evidence without treating raw timing/planning queues as score signal.
    """

    if selection.get("schema") != SELECTION_SCHEMA:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"selection schema must be {SELECTION_SCHEMA}"
        )
    _require_explicit_false_authority(selection, label="selection")
    if selection.get("candidate_generation_only") is not True:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "selection candidate_generation_only must be true"
        )
    if selection.get("requires_exact_auth_eval_before_score_claim") is not True:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "selection requires_exact_auth_eval_before_score_claim must be true"
        )
    if selection.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"selection evidence_tag must be {EVIDENCE_TAG_MLX}"
        )
    if selection.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"selection evidence_grade must be {EVIDENCE_GRADE_MLX}"
        )
    incumbent = _finite_float(incumbent_score, label="incumbent_score")
    floor = _finite_float(variance_floor, label="variance_floor")
    if incumbent <= 0.0:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "incumbent_score must be positive"
        )
    if floor < 0.0:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "variance_floor must be non-negative"
        )
    if top_k is not None and int(top_k) <= 0:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "top_k must be positive when provided"
        )

    gate_statuses = _selection_gate_statuses(selection)
    selected_rows = selection.get("selected_rows")
    if not isinstance(selected_rows, list):
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "selection selected_rows[] missing"
        )

    rows = selected_rows[: int(top_k)] if top_k is not None else selected_rows
    candidates = [
        _candidate_from_selection_row(
            row,
            index=index,
            incumbent_score=incumbent,
            variance_floor=floor,
            gate_statuses=gate_statuses,
        )
        for index, row in enumerate(rows)
        if isinstance(row, Mapping)
    ]
    if len(candidates) != len(rows):
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "selection selected_rows[] must contain only objects"
        )
    if not candidates:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "no selection rows available for learned-sweep adaptation"
        )

    return {
        "schema": SCHEMA,
        "producer": TOOL,
        "source_schema": SELECTION_SCHEMA,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "quality_evidence_schema": QUALITY_EVIDENCE_SCHEMA,
        "allowed_use": (
            "learned_sweep_candidate_intake_after_strict_effective_mlx_spend_triage"
        ),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "source_artifacts": dict(source_artifacts or {}),
        "summary": {
            "source_selected_count": len(selected_rows),
            "adapted_candidate_count": len(candidates),
            "top_k": top_k,
            "incumbent_score": incumbent,
            "variance_floor": floor,
            "gate_statuses": gate_statuses,
            "best_predicted_score_mean": min(
                candidate["predicted_score_mean"] for candidate in candidates
            ),
        },
        "candidates": candidates,
    }


def _candidate_from_selection_row(
    row: Mapping[str, Any],
    *,
    index: int,
    incumbent_score: float,
    variance_floor: float,
    gate_statuses: Mapping[str, str],
) -> dict[str, Any]:
    if row.get("schema") != SELECTION_ROW_SCHEMA:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"selected row {index} schema must be {SELECTION_ROW_SCHEMA}"
        )
    _require_explicit_false_authority(row, label=f"selected row {index}")
    candidate_id = str(
        row.get("candidate_id") or row.get("row_id") or f"mlx_triage_row_{index:04d}"
    )
    if row.get("selection_basis") != "normalized_full_video_mlx_singleton_response_gain":
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{candidate_id}.selection_basis must be "
            "normalized_full_video_mlx_singleton_response_gain"
        )
    if row.get("requires_exact_auth_eval_before_score_claim") is not True:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{candidate_id}.requires_exact_auth_eval_before_score_claim must be true"
        )
    if row.get("source_evidence_grade") != EVIDENCE_GRADE_MLX:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{candidate_id}.source_evidence_grade must be {EVIDENCE_GRADE_MLX}"
        )
    if row.get("source_evidence_tag") != EVIDENCE_TAG_MLX:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{candidate_id}.source_evidence_tag must be {EVIDENCE_TAG_MLX}"
        )
    if "canonical_provenance" in row and not isinstance(row["canonical_provenance"], Mapping):
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{candidate_id}.canonical_provenance must be an object when present"
        )
    try:
        metrics = require_normalized_full_video_objective(
            row,
            label=f"{candidate_id}.normalized_objective",
        )
    except NormalizedObjectiveError as exc:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(str(exc)) from exc
    if row.get("full_video_denominator") != CONTEST_EXACT_SAMPLE_COUNT:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{candidate_id}.full_video_denominator must be "
            f"{CONTEST_EXACT_SAMPLE_COUNT}"
        )

    projected_delta = metrics["projected_delta"]
    normalized_gain = metrics["normalized_gain"]
    observed_gain = _finite_float(
        row.get("observed_scorer_gain_vs_baseline"),
        label=f"{candidate_id}.observed_scorer_gain_vs_baseline",
    )
    added_archive_bytes = _finite_float(
        row.get("added_archive_bytes"),
        label=f"{candidate_id}.added_archive_bytes",
    )
    if added_archive_bytes < 0.0:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{candidate_id}.added_archive_bytes must be non-negative"
        )
    calibrated_gap = _finite_float(
        row.get("calibrated_min_mlx_gap_for_spend_triage"),
        label=f"{candidate_id}.calibrated_min_mlx_gap_for_spend_triage",
    )
    if calibrated_gap < 0.0:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{candidate_id}.calibrated_min_mlx_gap_for_spend_triage must be non-negative"
        )
    predicted_delta = row.get("predicted_delta_vs_baseline_score")
    disagreement = 0.0
    if predicted_delta is not None:
        disagreement = abs(
            projected_delta
            - _finite_float(
                predicted_delta,
                label=f"{candidate_id}.predicted_delta_vs_baseline_score",
            )
        )
    predicted_score = incumbent_score + projected_delta
    pair_indices = _optional_list(row.get("pair_indices"))
    selected_pair_count = _selected_pair_count(row, pair_indices=pair_indices)
    quality_evidence = {
        "schema": QUALITY_EVIDENCE_SCHEMA,
        **FALSE_AUTHORITY,
        "source_schema": SELECTION_SCHEMA,
        "source_row_schema": SELECTION_ROW_SCHEMA,
        "source_evidence_grade": row.get("source_evidence_grade"),
        "source_evidence_tag": row.get("source_evidence_tag"),
        "canonical_provenance": row.get("canonical_provenance"),
        "adapter_tool": CLI_TOOL,
        "calibrated_score_mean": predicted_score,
        "calibrated_gain_mean": normalized_gain,
        "calibrated_delta_mean": projected_delta,
        "calibrated_min_mlx_gap_for_spend_triage": calibrated_gap,
        "gate_statuses": dict(gate_statuses),
    }
    return {
        "schema": ROW_SCHEMA,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "candidate_id": candidate_id,
        "source_row_id": row.get("row_id"),
        "family": str(row.get("family") or "mlx_decoder_q"),
        "rank": row.get("rank"),
        "predicted_score_mean": predicted_score,
        "predicted_score_variance": max(
            variance_floor,
            calibrated_gap * calibrated_gap,
            disagreement * disagreement,
        ),
        "predicted_score_source": "mlx_effective_spend_triage_quality_adapter",
        "predicted_score_scope": "candidate_specific_normalized_full_video",
        "payload_bytes": round(added_archive_bytes),
        "selected_pair_indices": pair_indices,
        "selected_pair_count": selected_pair_count,
        "pair_indices": pair_indices,
        "non_authoritative_mlx_gain_sum": normalized_gain,
        "non_authoritative_normalized_full_video_gain_sum": normalized_gain,
        "non_authoritative_mlx_window_gain_sum": (
            normalized_gain * float(CONTEST_EXACT_SAMPLE_COUNT)
        ),
        "full_video_denominator": CONTEST_EXACT_SAMPLE_COUNT,
        "observed_window_scorer_gain_vs_baseline": observed_gain,
        "projected_full_video_delta_vs_baseline_score": projected_delta,
        "break_even_added_bytes_from_normalized_full_video_gain": metrics[
            "break_even_added_bytes"
        ],
        "normalized_full_video_byte_budget_margin_vs_break_even": metrics[
            "normalized_margin"
        ],
        "component_axis_context": {
            "schema": "mlx_effective_spend_triage_component_axis_context.v1",
            "axis": EVIDENCE_TAG_MLX,
            "source_evidence_grade": EVIDENCE_GRADE_MLX,
            "observed_scorer_gain_vs_baseline": observed_gain,
            "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
            "projected_full_video_delta_vs_baseline_score": projected_delta,
            **FALSE_AUTHORITY,
        },
        "frame_pair_context": {
            "schema": "mlx_effective_spend_triage_frame_pair_context.v1",
            "source_pair_window": _optional_list(row.get("source_pair_window")),
            "pair_indices": pair_indices,
            "source_n_samples": row.get("source_n_samples"),
            "source_batch_pairs": row.get("source_batch_pairs"),
            **FALSE_AUTHORITY,
        },
        "waterbucket_context": {
            "schema": "mlx_effective_spend_triage_waterbucket_context.v1",
            "selection_basis": row.get("selection_basis"),
            "normalized_gain": normalized_gain,
            "projected_delta": projected_delta,
            "break_even_added_bytes": metrics["break_even_added_bytes"],
            "byte_budget_margin": metrics["normalized_margin"],
            "added_archive_bytes": added_archive_bytes,
            **FALSE_AUTHORITY,
        },
        "quality_evidence": quality_evidence,
    }


def _selection_gate_statuses(selection: Mapping[str, Any]) -> dict[str, str]:
    gates = selection.get("gates")
    if not isinstance(gates, Mapping):
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "selection gates object missing"
        )
    effective_gate = gates.get("effective_mlx_spend_triage_gate")
    if not isinstance(effective_gate, Mapping):
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "selection effective_mlx_spend_triage_gate missing"
        )
    if effective_gate.get("mlx_exact_eval_spend_triage_allowed") is not True:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "effective_mlx_spend_triage_gate must allow spend triage"
        )
    if gates.get("response_validation_status") != "passed":
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            "response_validation_status must be passed"
        )
    gate_statuses = {
        "calibration": str(gates.get("score_calibration_status") or ""),
        "parity": str(gates.get("torch_parity_status") or ""),
        "production_contract": str(gates.get("production_contract_status") or ""),
        "effective_spend_triage": str(effective_gate.get("status") or ""),
    }
    for gate, status in gate_statuses.items():
        if status != "strict_pass":
            raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
                f"selection gate {gate} must be strict_pass, got {status!r}"
            )
    return gate_statuses


def _selected_pair_count(
    row: Mapping[str, Any],
    *,
    pair_indices: list[Any] | None,
) -> int:
    raw = row.get("source_n_samples", row.get("source_batch_pairs"))
    if raw is not None:
        parsed = _finite_int(raw, label="source_n_samples")
        if parsed <= 0:
            raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
                "source_n_samples must be positive"
            )
        return parsed
    if pair_indices:
        return max(1, len(pair_indices) // 2)
    return 1


def _optional_list(value: Any) -> list[Any] | None:
    return list(value) if isinstance(value, list | tuple) else None


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{label} must be numeric"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{label} must be numeric"
        ) from exc
    if not math.isfinite(result):
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{label} must be finite"
        )
    return result


def _finite_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{label} must be an integer"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{label} must be an integer"
        ) from exc
    if parsed != value and not (isinstance(value, str) and str(parsed) == value):
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
            f"{label} must be integral"
        )
    return parsed


def _require_explicit_false_authority(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise MLXEffectiveSpendTriageLearnedSweepAdapterError(
                f"{label} {key} must be explicit false"
            )
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise MLXEffectiveSpendTriageLearnedSweepAdapterError(str(exc)) from exc


__all__ = [
    "CLI_TOOL",
    "DEFAULT_VARIANCE_FLOOR",
    "REFUSAL_SCHEMA",
    "ROW_SCHEMA",
    "SCHEMA",
    "TOOL",
    "MLXEffectiveSpendTriageLearnedSweepAdapterError",
    "build_mlx_effective_spend_triage_learned_sweep_candidates",
    "build_refusal_payload",
    "dumps_json",
    "file_sha256",
    "load_json_object",
    "source_artifact_metadata",
    "write_json",
]
