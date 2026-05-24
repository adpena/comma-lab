# SPDX-License-Identifier: MIT
"""Harvest local MLX learned-sweep rows into observation JSONL."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.mlx_dynamic_learned_sweep import FALSE_AUTHORITY
from tac.optimization.mlx_dynamic_sweep_observations import (
    ROW_SCHEMA as OBSERVATION_ROW_SCHEMA,
)
from tac.optimization.mlx_dynamic_sweep_observations import (
    build_observation_row,
    json_text,
    summarize_observations,
)
from tac.optimization.mlx_effective_spend_triage_selection import (
    ROW_SCHEMA as SELECTION_ROW_SCHEMA,
)
from tac.optimization.mlx_effective_spend_triage_selection import (
    SCHEMA as SELECTION_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.repo_io import sha256_bytes, sha256_file

SCHEMA = "mlx_dynamic_learned_sweep_observation_harvest.v1"
TOOL = "tac.optimization.mlx_dynamic_learned_sweep_observation_harvest"
DEFAULT_SWEEP_CONFIG_ID = "mlx_local_response"
DEFAULT_OPTIMIZATION_PASS_ID = "smoke"


class MLXDynamicLearnedSweepObservationHarvestError(ValueError):
    """Raised when learned-sweep rows cannot become valid observations."""


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MLXDynamicLearnedSweepObservationHarvestError(
            f"{path}: expected JSON object"
        )
    return payload


def build_observation_rows_from_learned_sweep_plan(
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    planner_artifact_path: str | Path | None = None,
    planner_artifact_sha256: str | None = None,
    max_rows: int | None = None,
    sweep_config_id: str | None = DEFAULT_SWEEP_CONFIG_ID,
    optimization_pass_id: str | None = DEFAULT_OPTIMIZATION_PASS_ID,
    ready_local_only: bool = True,
    source_artifact_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build observation rows from already-measured local MLX response evidence."""

    if plan.get("schema") != "mlx_dynamic_learned_sweep_plan.v1":
        raise MLXDynamicLearnedSweepObservationHarvestError("plan schema mismatch")
    if selection.get("schema") != SELECTION_SCHEMA:
        raise MLXDynamicLearnedSweepObservationHarvestError("selection schema mismatch")
    _require_false_authority(plan, label="plan")
    _require_false_authority(selection, label="selection")
    if max_rows is not None and max_rows <= 0:
        raise MLXDynamicLearnedSweepObservationHarvestError("max_rows must be positive")
    selection_by_candidate = _selection_rows_by_candidate(selection)
    rows = plan.get("ranked_sweep_rows")
    if not isinstance(rows, list):
        raise MLXDynamicLearnedSweepObservationHarvestError(
            "plan ranked_sweep_rows[] missing"
        )
    out: list[dict[str, Any]] = []
    for sweep_row in rows:
        if not isinstance(sweep_row, Mapping):
            continue
        if ready_local_only and sweep_row.get("ready_for_local_sweep") is not True:
            continue
        if sweep_config_id is not None and sweep_row.get("sweep_config_id") != sweep_config_id:
            continue
        if (
            optimization_pass_id is not None
            and sweep_row.get("optimization_pass_id") != optimization_pass_id
        ):
            continue
        candidate_id = str(sweep_row.get("candidate_id") or "")
        source_row = selection_by_candidate.get(candidate_id)
        if source_row is None:
            raise MLXDynamicLearnedSweepObservationHarvestError(
                f"{candidate_id}: missing source selection row"
            )
        out.append(
            _observation_from_rows(
                sweep_row,
                source_row,
                planner_artifact_path=planner_artifact_path,
                planner_artifact_sha256=planner_artifact_sha256,
                source_artifact_root=source_artifact_root,
            )
        )
        if max_rows is not None and len(out) >= max_rows:
            break
    if not out:
        raise MLXDynamicLearnedSweepObservationHarvestError(
            "no learned-sweep rows matched observation harvest filters"
        )
    return out


def build_observation_harvest_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan_path: str | Path | None = None,
    selection_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a manifest summarizing harvested observation rows."""

    normalized_rows = [dict(row) for row in rows]
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "allowed_use": "local_mlx_learned_sweep_observation_feedback_only",
        "row_schema": OBSERVATION_ROW_SCHEMA,
        "row_count": len(normalized_rows),
        "source_artifacts": _source_artifacts(plan_path=plan_path, selection_path=selection_path),
        "summary": summarize_observations(normalized_rows),
    }


def write_observation_jsonl(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    replace: bool = False,
) -> None:
    """Write observation rows as deterministic JSONL."""

    if output_path.exists() and not replace:
        raise MLXDynamicLearnedSweepObservationHarvestError(
            f"refusing to overwrite existing observation JSONL: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    output_path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any], *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise MLXDynamicLearnedSweepObservationHarvestError(
            f"refusing to overwrite existing JSON artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(payload), encoding="utf-8")


def _observation_from_rows(
    sweep_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    *,
    planner_artifact_path: str | Path | None,
    planner_artifact_sha256: str | None,
    source_artifact_root: str | Path | None,
) -> dict[str, Any]:
    candidate_id = _required_text(sweep_row, "candidate_id")
    source_path = _source_artifact_path(
        source_row,
        key="source_path",
        source_artifact_root=source_artifact_root,
    )
    component_deltas = _component_deltas(
        source_row,
        source_path=source_path,
        source_artifact_root=source_artifact_root,
    )
    source_path_text = None if source_path is None else str(source_path)
    source_sha = sha256_file(source_path) if source_path is not None and source_path.is_file() else None
    extra = {
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "selected_pair_indices": source_row.get("pair_indices")
        or sweep_row.get("selected_pair_indices"),
        "selected_pair_count": sweep_row.get("selected_pair_count"),
        "source_schema": str(source_row.get("schema") or SELECTION_ROW_SCHEMA),
        "source_row": {
            "schema": str(source_row.get("schema") or ""),
            "candidate_id": source_row.get("candidate_id"),
            "row_id": source_row.get("row_id"),
            "selection_basis": source_row.get("selection_basis"),
            "source_path": source_row.get("source_path"),
            **FALSE_AUTHORITY,
        },
        "planner_artifact_path": None
        if planner_artifact_path is None
        else str(planner_artifact_path),
        "planner_artifact_sha256": planner_artifact_sha256,
        "component_delta_baseline_policy": (
            "source_window_component_terms_scaled_to_full_video_denominator"
        ),
        "notes": (
            "Harvested from strict local MLX scorer-response evidence already "
            "consumed by the learned-sweep quality adapter; observation is "
            "replanning-only and carries no score or dispatch authority."
        ),
    }
    extra = {key: value for key, value in extra.items() if value is not None}
    return build_observation_row(
        candidate_id=candidate_id,
        sweep_config_id=_required_text(sweep_row, "sweep_config_id"),
        optimization_pass_id=_required_text(sweep_row, "optimization_pass_id"),
        family=_required_text(sweep_row, "family"),
        observed_axis=EVIDENCE_TAG_MLX,
        evidence_tag=EVIDENCE_TAG_MLX,
        observed_score_or_delta=_required_float(
            source_row,
            "projected_full_video_delta_vs_baseline_score",
        ),
        archive_sha256=_required_sha(source_row, "archive_sha256"),
        runtime_sha256=_runtime_identity_sha256(source_row),
        raw_output_or_cache_sha256=_cache_identity_sha256(source_row),
        component_deltas=component_deltas,
        source_artifact_path=source_path_text,
        source_artifact_sha256=source_sha,
        extra=extra,
    )


def _selection_rows_by_candidate(selection: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = selection.get("selected_rows")
    if not isinstance(rows, list):
        raise MLXDynamicLearnedSweepObservationHarvestError(
            "selection selected_rows[] missing"
        )
    out: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise MLXDynamicLearnedSweepObservationHarvestError(
                f"selection row {index} must be object"
            )
        if row.get("schema") != SELECTION_ROW_SCHEMA:
            raise MLXDynamicLearnedSweepObservationHarvestError(
                f"selection row {index} schema mismatch"
            )
        _require_false_authority(row, label=f"selection row {index}")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            raise MLXDynamicLearnedSweepObservationHarvestError(
                f"selection row {index} candidate_id missing"
            )
        out[candidate_id] = row
    return out


def _component_deltas(
    source_row: Mapping[str, Any],
    *,
    source_path: Path | None,
    source_artifact_root: str | Path | None,
) -> dict[str, float]:
    scale = _normalization_scale(source_row)
    if _has_component_terms(source_row):
        seg_delta = (
            _required_float(source_row, "seg_term")
            - _required_float(source_row, "window_baseline_seg_term")
        ) * scale
        pose_delta = (
            _required_float(source_row, "pose_term")
            - _required_float(source_row, "window_baseline_pose_term")
        ) * scale
        rate_delta = _optional_float(source_row.get("rate_delta_vs_baseline"), default=0.0)
    else:
        candidate_terms = _artifact_component_terms(
            source_path,
            label="source_path",
        )
        baseline_path = _source_artifact_path(
            source_row,
            key="window_baseline_source_path",
            source_artifact_root=source_artifact_root,
        )
        baseline_terms = _artifact_component_terms(
            baseline_path,
            label="window_baseline_source_path",
        )
        seg_delta = (candidate_terms["seg_term"] - baseline_terms["seg_term"]) * scale
        pose_delta = (candidate_terms["pose_term"] - baseline_terms["pose_term"]) * scale
        rate_delta = _optional_float(
            source_row.get("rate_delta_vs_baseline"),
            default=candidate_terms["rate_term"] - baseline_terms["rate_term"],
        )
    return {
        "segnet_delta": seg_delta,
        "posenet_delta": pose_delta,
        "rate_delta": rate_delta,
        "scorer_delta": seg_delta + pose_delta,
    }


def _has_component_terms(source_row: Mapping[str, Any]) -> bool:
    return all(
        source_row.get(key) is not None
        for key in (
            "seg_term",
            "window_baseline_seg_term",
            "pose_term",
            "window_baseline_pose_term",
        )
    )


def _artifact_component_terms(path: Path | None, *, label: str) -> dict[str, float]:
    if path is None or not path.is_file():
        raise MLXDynamicLearnedSweepObservationHarvestError(
            f"{label} component terms missing and artifact file is unavailable"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MLXDynamicLearnedSweepObservationHarvestError(
            f"{label} component terms missing and artifact is not valid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise MLXDynamicLearnedSweepObservationHarvestError(
            f"{label} component artifact must be a JSON object"
        )
    seg_dist = _required_float(payload, "avg_segnet_dist")
    pose_dist = _required_float(payload, "avg_posenet_dist")
    if seg_dist < 0.0 or pose_dist < 0.0:
        raise MLXDynamicLearnedSweepObservationHarvestError(
            f"{label} component distortions must be non-negative"
        )
    return {
        "seg_term": 100.0 * seg_dist,
        "pose_term": math.sqrt(10.0 * pose_dist),
        "rate_term": _artifact_rate_term(payload),
    }


def _artifact_rate_term(payload: Mapping[str, Any]) -> float:
    if payload.get("score_rate_contribution") is not None:
        return _required_float(payload, "score_rate_contribution")
    return 25.0 * _required_float(payload, "rate_unscaled")


def _normalization_scale(source_row: Mapping[str, Any]) -> float:
    source_n = _required_float(source_row, "source_n_samples")
    denominator = _required_float(source_row, "full_video_denominator")
    if source_n <= 0.0 or denominator <= 0.0 or source_n > denominator:
        raise MLXDynamicLearnedSweepObservationHarvestError(
            "source_n_samples/full_video_denominator invalid"
        )
    return source_n / denominator


def _runtime_identity_sha256(source_row: Mapping[str, Any]) -> str:
    return _hash_payload(
        {
            "schema": "mlx_dynamic_learned_sweep_local_runtime_identity.v1",
            "source_evidence_grade": source_row.get("source_evidence_grade"),
            "source_evidence_tag": source_row.get("source_evidence_tag"),
            "source_schema": source_row.get("source_schema"),
            "source_posenet_sha256": source_row.get("source_posenet_sha256"),
            "source_segnet_sha256": source_row.get("source_segnet_sha256"),
        }
    )


def _cache_identity_sha256(source_row: Mapping[str, Any]) -> str:
    cache = source_row.get("source_candidate_cache_array_sha256")
    if isinstance(cache, Mapping) and cache:
        return _hash_payload(
            {
                "schema": "mlx_dynamic_learned_sweep_cache_identity.v1",
                "source_candidate_cache_array_sha256": dict(sorted(cache.items())),
            }
        )
    return _required_sha(source_row, "source_inflated_outputs_aggregate_sha256")


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


def _source_artifact_path(
    source_row: Mapping[str, Any],
    *,
    key: str,
    source_artifact_root: str | Path | None,
) -> Path | None:
    raw = source_row.get(key)
    if raw is None:
        return None
    path = Path(str(raw))
    if path.is_file():
        return path
    if source_artifact_root is not None:
        rooted = Path(source_artifact_root) / path
        if rooted.is_file():
            return rooted
    return path


def _source_artifacts(
    *,
    plan_path: str | Path | None,
    selection_path: str | Path | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw in (("plan", plan_path), ("selection", selection_path)):
        if raw is None:
            continue
        path = Path(raw)
        out[key] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return out


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise MLXDynamicLearnedSweepObservationHarvestError(f"{key} is required")
    return text


def _required_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MLXDynamicLearnedSweepObservationHarvestError(
            f"{key} must be numeric"
        ) from exc


def _optional_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MLXDynamicLearnedSweepObservationHarvestError(
            "optional float value must be numeric"
        ) from exc


def _required_sha(row: Mapping[str, Any], key: str) -> str:
    value = _required_text(row, key).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise MLXDynamicLearnedSweepObservationHarvestError(
            f"{key} must be a 64-character SHA-256 hex digest"
        )
    return value


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise MLXDynamicLearnedSweepObservationHarvestError(
                f"{label} {key} must be explicit false"
            )
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise MLXDynamicLearnedSweepObservationHarvestError(str(exc)) from exc


__all__ = [
    "DEFAULT_OPTIMIZATION_PASS_ID",
    "DEFAULT_SWEEP_CONFIG_ID",
    "SCHEMA",
    "TOOL",
    "MLXDynamicLearnedSweepObservationHarvestError",
    "build_observation_harvest_manifest",
    "build_observation_rows_from_learned_sweep_plan",
    "load_json_object",
    "write_json",
    "write_observation_jsonl",
]
