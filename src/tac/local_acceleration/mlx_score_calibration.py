# SPDX-License-Identifier: MIT
"""Calibration summaries for local MLX scorer-response artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

SCHEMA_VERSION = "mlx_score_calibration.v1"

AUTHORITY_FALSE_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_mlx_score_calibration_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mlx_score_calibration_manifest(
    rows: list[dict[str, Any]],
    *,
    repo_root: str | Path = ".",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a false-authority calibration table from MLX and exact-axis rows."""

    root = Path(repo_root)
    normalized = [
        _normalize_row(row, repo_root=root, index=index)
        for index, row in enumerate(rows)
    ]
    _attach_ranks(normalized, "mlx_score", "mlx_rank")
    if all(row.get("cpu_score") is not None for row in normalized):
        _attach_ranks(normalized, "cpu_score", "cpu_rank")
    if all(row.get("cuda_score") is not None for row in normalized):
        _attach_ranks(normalized, "cuda_score", "cuda_rank")

    pairwise = _pairwise_order(normalized)
    summary = _build_summary(normalized, pairwise)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "row_count": len(normalized),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "calibration_role": "local_mlx_decision_quality_calibration",
        "rows": normalized,
        "pairwise_order": pairwise,
        "summary": summary,
        "authority_status": (
            "This manifest measures MLX local signal quality only. Exact contest "
            "CPU/CUDA auth eval remains required for score claims, promotion, "
            "rank/kill decisions, and dispatch readiness."
        ),
    }


def _normalize_row(row: dict[str, Any], *, repo_root: Path, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"calibration row {index} is not an object")
    label = str(row.get("label") or f"row_{index}")
    mlx_response_path = _required_path(row, "mlx_response_path", repo_root)
    mlx_response = load_json_object(mlx_response_path)
    _require_mlx_response_false_authority(mlx_response, mlx_response_path)

    cpu_score = _resolve_axis_score(row, "cpu", repo_root)
    cuda_score = _resolve_axis_score(row, "cuda", repo_root)
    local_cpu_score = _optional_float(row.get("local_cpu_score"))
    mlx_score = _finite_float(mlx_response.get("canonical_score"), "mlx_response.canonical_score")

    out: dict[str, Any] = {
        "index": index,
        "label": label,
        "pr_number": row.get("pr_number"),
        "archive_sha256": mlx_response.get("archive_sha256"),
        "inflated_outputs_aggregate_sha256": mlx_response.get(
            "inflated_outputs_aggregate_sha256"
        ),
        "archive_size_bytes": int(mlx_response.get("archive_size_bytes")),
        "n_samples": int(mlx_response.get("n_samples")),
        "batch_pairs": int(mlx_response.get("batch_pairs")),
        "mlx_response_path": str(mlx_response_path),
        "mlx_score": mlx_score,
        "mlx_avg_posenet_dist": _finite_float(
            mlx_response.get("avg_posenet_dist"), "mlx_response.avg_posenet_dist"
        ),
        "mlx_avg_segnet_dist": _finite_float(
            mlx_response.get("avg_segnet_dist"), "mlx_response.avg_segnet_dist"
        ),
        "mlx_batch_shape_research_signal_allowed": bool(
            mlx_response.get("batch_shape_research_signal_allowed")
        ),
    }
    if cpu_score is not None:
        out["cpu_score"] = cpu_score
        out["mlx_minus_cpu"] = mlx_score - cpu_score
        out["cpu_source"] = row.get("cpu_source") or row.get("cpu_auth_eval_path")
    if cuda_score is not None:
        out["cuda_score"] = cuda_score
        out["cuda_minus_mlx"] = cuda_score - mlx_score
        out["cuda_source"] = row.get("cuda_source") or row.get("cuda_auth_eval_path")
    if local_cpu_score is not None:
        out["local_cpu_score"] = local_cpu_score
        out["mlx_minus_local_cpu"] = mlx_score - local_cpu_score
    if cpu_score is not None and cuda_score is not None:
        out["cuda_minus_cpu"] = cuda_score - cpu_score
    return out


def _required_path(row: dict[str, Any], key: str, repo_root: Path) -> Path:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"calibration row missing {key}")
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        raise ValueError(f"{key} does not exist: {path}")
    return path


def _resolve_axis_score(row: dict[str, Any], axis: str, repo_root: Path) -> float | None:
    direct_key = f"{axis}_score"
    path_key = f"{axis}_auth_eval_path"
    if row.get(direct_key) is not None:
        return _finite_float(row[direct_key], direct_key)
    if row.get(path_key) is None:
        return None
    path = _required_path(row, path_key, repo_root)
    payload = load_json_object(path)
    return _score_from_auth_eval_payload(payload, path)


def _score_from_auth_eval_payload(payload: dict[str, Any], path: Path) -> float:
    for key in ("canonical_score", "score_recomputed_from_components"):
        if payload.get(key) is not None:
            return _finite_float(payload[key], f"{path}:{key}")
    raise ValueError(f"auth eval payload has no canonical score: {path}")


def _require_mlx_response_false_authority(payload: dict[str, Any], path: Path) -> None:
    if payload.get("schema_version") != "mlx_scorer_response.v1":
        raise ValueError(f"not an mlx_scorer_response.v1 payload: {path}")
    if payload.get("evidence_grade") not in {None, EVIDENCE_GRADE_MLX}:
        raise ValueError(f"MLX response evidence grade is not local MLX: {path}")
    for field in AUTHORITY_FALSE_FIELDS:
        if payload.get(field) is not False:
            raise ValueError(f"MLX response {path} has non-false {field}")


def _attach_ranks(rows: list[dict[str, Any]], score_key: str, rank_key: str) -> None:
    ordered = sorted(range(len(rows)), key=lambda idx: float(rows[idx][score_key]))
    for rank, idx in enumerate(ordered, start=1):
        rows[idx][rank_key] = rank


def _pairwise_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            item: dict[str, Any] = {
                "left_index": left["index"],
                "right_index": right["index"],
                "left_label": left["label"],
                "right_label": right["label"],
                "mlx_order": _order(left.get("mlx_score"), right.get("mlx_score")),
            }
            for axis in ("cpu", "cuda"):
                key = f"{axis}_score"
                if left.get(key) is not None and right.get(key) is not None:
                    item[f"{axis}_order"] = _order(left.get(key), right.get(key))
                    item[f"mlx_matches_{axis}"] = item["mlx_order"] == item[f"{axis}_order"]
            out.append(item)
    return out


def _build_summary(rows: list[dict[str, Any]], pairwise: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    _attach_delta_stats(summary, rows, "mlx_minus_cpu")
    _attach_delta_stats(summary, rows, "mlx_minus_local_cpu")
    _attach_delta_stats(summary, rows, "cuda_minus_mlx")
    _attach_delta_stats(summary, rows, "cuda_minus_cpu")
    for axis in ("cpu", "cuda"):
        match_key = f"mlx_matches_{axis}"
        comparable = [item for item in pairwise if match_key in item]
        summary[f"mlx_{axis}_pairwise_comparison_count"] = len(comparable)
        summary[f"mlx_{axis}_rank_inversions"] = sum(
            1 for item in comparable if item[match_key] is not True
        )
    return summary


def _attach_delta_stats(summary: dict[str, Any], rows: list[dict[str, Any]], key: str) -> None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return
    summary[f"{key}_mean"] = sum(values) / len(values)
    summary[f"{key}_min"] = min(values)
    summary[f"{key}_max"] = max(values)
    summary[f"{key}_max_abs"] = max(abs(value) for value in values)


def _order(left: Any, right: Any) -> int:
    left_f = _finite_float(left, "left")
    right_f = _finite_float(right, "right")
    return (left_f > right_f) - (left_f < right_f)


def _finite_float(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite float, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return out


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float(value, "optional_float")


__all__ = [
    "SCHEMA_VERSION",
    "build_mlx_score_calibration_manifest",
    "load_json_object",
    "write_mlx_score_calibration_manifest",
]
