# SPDX-License-Identifier: MIT
"""Calibration summaries for local MLX scorer-response artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tac.auth_eval_schema import (
    CONTEST_AUTH_AXIS_BY_EVIDENCE_GRADE,
    FULL_CONTEST_SAMPLE_COUNT,
    eval_metric_summary,
    required_contest_auth_axis_payload_blockers,
)
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

SCHEMA_VERSION = "mlx_score_calibration.v1"
DEFAULT_DECISION_SAFETY_FACTOR = 5.0
STRICT_AUTH_AXIS_SPEND_TRIAGE_ALLOWED_USE = (
    "local_spend_triage_only_after_strict_auth_axis_calibration"
)

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
    decision_safety_factor: float = DEFAULT_DECISION_SAFETY_FACTOR,
) -> dict[str, Any]:
    """Build a false-authority calibration table from MLX and exact-axis rows."""

    root = Path(repo_root)
    safety_factor = _positive_float(decision_safety_factor, "decision_safety_factor")
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
    decision_policy = _build_decision_policy(summary, safety_factor)
    _attach_decision_certification(pairwise, decision_policy)
    _attach_decision_summary(summary, pairwise, decision_policy)
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
        "decision_policy": decision_policy,
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

    mlx_archive_size_bytes = int(mlx_response.get("archive_size_bytes"))
    cpu_score = _resolve_axis_score(
        row,
        "cpu",
        repo_root,
        expected_archive_bytes=mlx_archive_size_bytes,
    )
    cuda_score = _resolve_axis_score(
        row,
        "cuda",
        repo_root,
        expected_archive_bytes=mlx_archive_size_bytes,
    )
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
        "archive_size_bytes": mlx_archive_size_bytes,
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


def _resolve_axis_score(
    row: dict[str, Any],
    axis: str,
    repo_root: Path,
    *,
    expected_archive_bytes: int,
) -> float | None:
    direct_key = f"{axis}_score"
    path_key = f"{axis}_auth_eval_path"
    if row.get(direct_key) is not None and row.get(path_key) is None:
        raise ValueError(
            f"{direct_key} direct scalar is not accepted for MLX calibration; "
            f"use {path_key} with a strict contest auth-eval payload"
        )
    if row.get(path_key) is None:
        return None
    direct_score = (
        None
        if row.get(direct_key) is None
        else _finite_float(row[direct_key], direct_key)
    )
    path = _required_path(row, path_key, repo_root)
    payload = load_json_object(path)
    payload_score = _score_from_auth_eval_payload(
        payload,
        path,
        axis,
        expected_archive_bytes=expected_archive_bytes,
    )
    if direct_score is not None and not math.isclose(
        direct_score,
        payload_score,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{direct_key} direct scalar does not match strict auth-eval payload: "
            f"direct={direct_score}:payload={payload_score}:path={path}"
        )
    return payload_score


def _score_from_auth_eval_payload(
    payload: dict[str, Any],
    path: Path,
    axis: str,
    *,
    expected_archive_bytes: int,
) -> float:
    expected_score_axis = {
        "cpu": "contest_cpu",
        "cuda": "contest_cuda",
    }.get(axis)
    if expected_score_axis is None:
        raise ValueError(f"unknown auth axis {axis!r}")
    metrics = eval_metric_summary(payload)
    blockers = required_contest_auth_axis_payload_blockers(
        payload,
        metrics,
        expected_archive_bytes=expected_archive_bytes,
        expected_n_samples=FULL_CONTEST_SAMPLE_COUNT,
    )
    if payload.get("score_axis") != expected_score_axis:
        blockers.append(
            f"{axis}_auth_eval_score_axis_mismatch:"
            f"expected={expected_score_axis}:actual={payload.get('score_axis')}"
        )
    evidence_grade = payload.get("evidence_grade")
    expected_from_grade = CONTEST_AUTH_AXIS_BY_EVIDENCE_GRADE.get(str(evidence_grade))
    if expected_from_grade != expected_score_axis:
        blockers.append(
            f"{axis}_auth_eval_evidence_grade_axis_mismatch:"
            f"grade={evidence_grade}:expected_score_axis={expected_score_axis}"
        )
    if blockers:
        raise ValueError(
            f"auth eval payload is not a strict {axis} contest auth-axis source: "
            f"{path}: {sorted(set(blockers))}"
        )
    score = metrics.get("score")
    if score is None:
        raise ValueError(f"auth eval payload has no canonical score: {path}")
    return _finite_float(score, f"{path}:score")


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
            mlx_gap = float(left["mlx_score"]) - float(right["mlx_score"])
            item: dict[str, Any] = {
                "left_index": left["index"],
                "right_index": right["index"],
                "left_label": left["label"],
                "right_label": right["label"],
                "mlx_order": _order(left.get("mlx_score"), right.get("mlx_score")),
                "mlx_score_gap": mlx_gap,
                "mlx_score_gap_abs": abs(mlx_gap),
            }
            for axis in ("cpu", "cuda"):
                key = f"{axis}_score"
                if left.get(key) is not None and right.get(key) is not None:
                    axis_gap = float(left[key]) - float(right[key])
                    item[f"{axis}_order"] = _order(left.get(key), right.get(key))
                    item[f"{axis}_score_gap"] = axis_gap
                    item[f"{axis}_score_gap_abs"] = abs(axis_gap)
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


def _build_decision_policy(
    summary: dict[str, Any],
    safety_factor: float,
) -> dict[str, Any]:
    cpu_error = summary.get("mlx_minus_cpu_max_abs")
    cuda_error = summary.get("cuda_minus_mlx_max_abs")
    available_errors = [
        float(value)
        for value in (cpu_error, cuda_error)
        if value is not None and math.isfinite(float(value))
    ]
    calibration_uncertainty_score = max(available_errors) if available_errors else None
    min_gap = (
        None
        if calibration_uncertainty_score is None
        else calibration_uncertainty_score * safety_factor
    )
    basis = "mlx_minus_cpu_max_abs"
    if cuda_error is not None and cpu_error is None:
        basis = "cuda_minus_mlx_max_abs"
    elif cuda_error is not None and cpu_error is not None:
        basis = "max(mlx_minus_cpu_max_abs, cuda_minus_mlx_max_abs)"
    return {
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "decision_safety_factor": safety_factor,
        "calibration_uncertainty_basis": basis if available_errors else None,
        "calibration_uncertainty_score": calibration_uncertainty_score,
        "recommended_min_mlx_gap_for_spend_triage": min_gap,
        "allowed_use": (
            STRICT_AUTH_AXIS_SPEND_TRIAGE_ALLOWED_USE
            if available_errors
            else "diagnostic_only_auth_axis_calibration_missing"
        ),
        "forbidden_use": "score_claim_or_rank_or_kill_or_promotion",
    }


def _attach_decision_certification(
    pairwise: list[dict[str, Any]],
    decision_policy: dict[str, Any],
) -> None:
    min_gap = decision_policy.get("recommended_min_mlx_gap_for_spend_triage")
    for item in pairwise:
        gap = _finite_float(item["mlx_score_gap_abs"], "pairwise.mlx_score_gap_abs")
        certified = min_gap is not None and gap >= float(min_gap)
        item["mlx_spend_triage_decision_certified"] = certified
        item["mlx_spend_triage_uncertain"] = not certified
        item["mlx_spend_triage_min_gap"] = min_gap


def _attach_decision_summary(
    summary: dict[str, Any],
    pairwise: list[dict[str, Any]],
    decision_policy: dict[str, Any],
) -> None:
    certified = [
        item for item in pairwise if item.get("mlx_spend_triage_decision_certified") is True
    ]
    uncertain = [item for item in pairwise if item.get("mlx_spend_triage_uncertain") is True]
    summary["mlx_spend_triage_pairwise_certified_count"] = len(certified)
    summary["mlx_spend_triage_pairwise_uncertain_count"] = len(uncertain)
    summary["mlx_spend_triage_pairwise_total_count"] = len(pairwise)
    summary["recommended_min_mlx_gap_for_spend_triage"] = decision_policy.get(
        "recommended_min_mlx_gap_for_spend_triage"
    )
    summary["calibration_uncertainty_score"] = decision_policy.get(
        "calibration_uncertainty_score"
    )


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


def _positive_float(value: Any, label: str) -> float:
    out = _finite_float(value, label)
    if out <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return out


__all__ = [
    "DEFAULT_DECISION_SAFETY_FACTOR",
    "SCHEMA_VERSION",
    "STRICT_AUTH_AXIS_SPEND_TRIAGE_ALLOWED_USE",
    "build_mlx_score_calibration_manifest",
    "load_json_object",
    "write_mlx_score_calibration_manifest",
]
