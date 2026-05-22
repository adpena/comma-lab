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
from tac.exact_eval_custody import (
    extract_runtime_tree_sha256,
    validate_exact_eval_evidence,
)
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

SCHEMA_VERSION = "mlx_score_calibration.v1"
DEFAULT_DECISION_SAFETY_FACTOR = 5.0
MIN_AXIS_ROWS_FOR_SPEND_TRIAGE = 3
MIN_AXIS_PAIRWISE_COMPARISONS_FOR_SPEND_TRIAGE = 3
STRICT_AUTH_AXIS_SPEND_TRIAGE_ALLOWED_USE = (
    "local_spend_triage_only_after_strict_auth_axis_calibration"
)

AUTHORITY_FALSE_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "promotable",
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
        "promotable": False,
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
    mlx_archive_sha256 = _required_sha256(
        mlx_response.get("archive_sha256"),
        f"{mlx_response_path}:archive_sha256",
    )
    mlx_inflated_outputs_aggregate_sha256 = _required_sha256(
        mlx_response.get("inflated_outputs_aggregate_sha256"),
        f"{mlx_response_path}:inflated_outputs_aggregate_sha256",
    )
    cpu_score = _resolve_axis_score(
        row,
        "cpu",
        repo_root,
        expected_archive_bytes=mlx_archive_size_bytes,
        expected_archive_sha256=mlx_archive_sha256,
        expected_inflated_outputs_aggregate_sha256=mlx_inflated_outputs_aggregate_sha256,
    )
    cuda_score = _resolve_axis_score(
        row,
        "cuda",
        repo_root,
        expected_archive_bytes=mlx_archive_size_bytes,
        expected_archive_sha256=mlx_archive_sha256,
        expected_inflated_outputs_aggregate_sha256=mlx_inflated_outputs_aggregate_sha256,
    )
    local_cpu_score = _optional_float(row.get("local_cpu_score"))
    mlx_score = _finite_float(mlx_response.get("canonical_score"), "mlx_response.canonical_score")
    components = mlx_response.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"MLX response {mlx_response_path} components missing")
    candidate_identity = _response_candidate_cache_identity(mlx_response)
    reference_identity = _response_cache_identity(mlx_response, "reference")

    out: dict[str, Any] = {
        "index": index,
        "label": label,
        "pr_number": row.get("pr_number"),
        "archive_sha256": mlx_archive_sha256,
        "inflated_outputs_aggregate_sha256": mlx_inflated_outputs_aggregate_sha256,
        "archive_size_bytes": mlx_archive_size_bytes,
        "n_samples": int(mlx_response.get("n_samples")),
        "batch_pairs": int(mlx_response.get("batch_pairs")),
        "pair_window": _required_pair_window(mlx_response.get("pair_window"), mlx_response_path),
        "response_family": mlx_response.get("response_family"),
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
        "mlx_components": {
            "posenet_sha256": _required_sha256(
                components.get("posenet_sha256"),
                f"{mlx_response_path}:components.posenet_sha256",
            ),
            "segnet_sha256": _required_sha256(
                components.get("segnet_sha256"),
                f"{mlx_response_path}:components.segnet_sha256",
            ),
            "posenet_shape": components.get("posenet_shape"),
            "segnet_shape": components.get("segnet_shape"),
        },
        "candidate_cache_identity": _public_cache_identity(candidate_identity),
        "reference_cache_identity": _public_cache_identity(reference_identity),
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
    expected_archive_sha256: str,
    expected_inflated_outputs_aggregate_sha256: str,
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
        expected_archive_sha256=expected_archive_sha256,
        expected_inflated_outputs_aggregate_sha256=expected_inflated_outputs_aggregate_sha256,
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
    expected_archive_sha256: str,
    expected_inflated_outputs_aggregate_sha256: str,
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
    exact_custody = _auth_axis_exact_eval_custody_validation(
        payload,
        metrics,
        path,
        axis=axis,
        expected_score_axis=expected_score_axis,
        expected_archive_sha256=expected_archive_sha256,
    )
    blockers.extend(
        f"{axis}_auth_eval_custody_{blocker}"
        for blocker in exact_custody.blockers
    )
    actual_archive_sha256 = _auth_archive_sha256(payload)
    if actual_archive_sha256 != expected_archive_sha256:
        blockers.append(
            f"{axis}_auth_eval_archive_sha256_mismatch:"
            f"expected={expected_archive_sha256}:actual={actual_archive_sha256}"
        )
    actual_inflated_sha256 = _auth_inflated_outputs_aggregate_sha256(payload)
    if actual_inflated_sha256 != expected_inflated_outputs_aggregate_sha256:
        blockers.append(
            f"{axis}_auth_eval_inflated_outputs_aggregate_sha256_mismatch:"
            f"expected={expected_inflated_outputs_aggregate_sha256}:"
            f"actual={actual_inflated_sha256}"
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


def _auth_axis_exact_eval_custody_validation(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    path: Path,
    *,
    axis: str,
    expected_score_axis: str,
    expected_archive_sha256: str,
):
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    inflated_manifest = _auth_inflated_output_manifest(payload)
    command = _auth_eval_command(payload, provenance)
    evidence = {
        "axis": _optional_str(payload.get("score_axis")) or expected_score_axis,
        "archive_sha256": _auth_archive_sha256(payload),
        "runtime_tree_sha256": extract_runtime_tree_sha256(payload),
        "score": metrics.get("score"),
        "seg_dist": metrics.get("seg_avg"),
        "pose_dist": metrics.get("pose_avg"),
        "archive_bytes": metrics.get("archive_size_bytes"),
        "n_samples": metrics.get("n_samples"),
        "hardware": _auth_hardware(payload, provenance, expected_score_axis),
        "inflate_device": _auth_inflate_device(payload, provenance, command),
        "eval_device": _auth_eval_device(payload, provenance, command),
        "auth_eval_command": command,
        "log_path": _first_text(
            payload.get("log_path"),
            payload.get("auth_eval_log_path"),
            provenance.get("log_path"),
            provenance.get("auth_eval_log_path"),
        ),
        "artifact_path": _first_text(
            payload.get("artifact_path"),
            payload.get("exact_eval_artifact_path"),
            payload.get("auth_eval_artifact_path"),
            provenance.get("artifact_path"),
            provenance.get("exact_eval_artifact_path"),
            provenance.get("auth_eval_artifact_path"),
        ),
        "artifact_sha256": _first_text(
            payload.get("artifact_sha256"),
            payload.get("exact_eval_artifact_sha256"),
            payload.get("auth_eval_artifact_sha256"),
            provenance.get("artifact_sha256"),
            provenance.get("exact_eval_artifact_sha256"),
            provenance.get("auth_eval_artifact_sha256"),
        ),
        "inflated_outputs_manifest_path": _first_text(
            payload.get("inflated_outputs_manifest_path"),
            payload.get("inflated_output_manifest_path"),
            inflated_manifest.get("path"),
        ),
        "inflated_outputs_manifest_sha256": _first_text(
            payload.get("inflated_outputs_manifest_sha256"),
            payload.get("inflated_output_manifest_sha256"),
            inflated_manifest.get("sha256"),
        ),
        "raw_output_aggregate_sha256": _auth_inflated_outputs_aggregate_sha256(payload),
    }
    return validate_exact_eval_evidence(
        evidence,
        expected_axis=expected_score_axis,
        expected_archive_sha256=expected_archive_sha256,
        require_artifact_path=True,
        require_artifact_sha256=True,
        require_devices=True,
        require_inflated_outputs_manifest=True,
        require_raw_output_aggregate_sha256=True,
        artifact_base_dir=path.parent,
        annotation_prefix=f"{axis}_auth_eval_custody",
    )


def _auth_inflated_output_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload.get("inflated_output_manifest")
    if isinstance(manifest, dict):
        return manifest
    manifest = payload.get("inflated_outputs_manifest")
    if isinstance(manifest, dict):
        return manifest
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        manifest = provenance.get("inflated_output_manifest")
        if isinstance(manifest, dict):
            return manifest
        manifest = provenance.get("inflated_outputs_manifest")
        if isinstance(manifest, dict):
            return manifest
    return {}


def _auth_eval_command(payload: dict[str, Any], provenance: dict[str, Any]) -> str:
    return _command_text(
        _first_present(
            payload.get("auth_eval_command"),
            payload.get("command"),
            payload.get("sys_argv"),
            provenance.get("auth_eval_command"),
            provenance.get("command"),
            provenance.get("sys_argv"),
        )
    )


def _auth_hardware(
    payload: dict[str, Any],
    provenance: dict[str, Any],
    expected_score_axis: str,
) -> str:
    direct = _first_text(
        payload.get("hardware"),
        payload.get("hardware_substrate"),
        provenance.get("hardware"),
        provenance.get("hardware_substrate"),
        payload.get("gpu_model"),
        provenance.get("gpu_model"),
    )
    if direct:
        return direct
    system = _first_text(payload.get("platform_system"), provenance.get("platform_system"))
    machine = _first_text(payload.get("platform_machine"), provenance.get("platform_machine"))
    if expected_score_axis == "contest_cpu" and system and machine:
        return f"{system} {machine} cpu"
    if expected_score_axis == "contest_cuda" and (
        payload.get("gpu_t4_match") is True or provenance.get("gpu_t4_match") is True
    ):
        return "T4 CUDA"
    return ""


def _auth_inflate_device(
    payload: dict[str, Any],
    provenance: dict[str, Any],
    command: str,
) -> str:
    return _first_text(
        payload.get("inflate_device"),
        payload.get("inflate_device_policy"),
        provenance.get("inflate_device"),
        provenance.get("inflate_device_policy"),
        _command_flag_value(command, "--inflate-device"),
    )


def _auth_eval_device(
    payload: dict[str, Any],
    provenance: dict[str, Any],
    command: str,
) -> str:
    return _first_text(
        payload.get("eval_device"),
        payload.get("actual_device"),
        payload.get("device"),
        provenance.get("eval_device"),
        provenance.get("actual_device"),
        provenance.get("device"),
        _command_flag_value(command, "--device"),
    )


def _require_mlx_response_false_authority(payload: dict[str, Any], path: Path) -> None:
    if payload.get("schema_version") != "mlx_scorer_response.v1":
        raise ValueError(f"not an mlx_scorer_response.v1 payload: {path}")
    if payload.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise ValueError(f"MLX response evidence grade is not local MLX: {path}")
    if payload.get("evidence_tag") != EVIDENCE_TAG_MLX:
        raise ValueError(f"MLX response evidence tag is not local MLX: {path}")
    if payload.get("score_axis") != EVIDENCE_TAG_MLX:
        raise ValueError(f"MLX response score axis is not local MLX: {path}")
    if payload.get("candidate_generation_only") is not True:
        raise ValueError(f"MLX response {path} is not candidate_generation_only")
    if payload.get("requires_exact_eval_before_promotion") is not True:
        raise ValueError(
            f"MLX response {path} does not require exact eval before promotion"
        )
    for field in AUTHORITY_FALSE_FIELDS:
        if payload.get(field) is not False:
            raise ValueError(f"MLX response {path} has non-false {field}")
    candidate_identity = _response_candidate_cache_identity(payload)
    audit = candidate_identity.get("auth_eval_identity_audit")
    if candidate_identity.get("eligible_for_local_mlx_transfer_calibration") is not True:
        raise ValueError(
            f"MLX response {path} candidate cache is not eligible for local MLX "
            "transfer calibration"
        )
    if not isinstance(audit, dict):
        raise ValueError(f"MLX response {path} candidate cache audit is missing")
    if audit.get("verdict") != "PASS_CACHE_AUTH_EVAL_IDENTITY":
        raise ValueError(
            f"MLX response {path} candidate cache audit did not pass: "
            f"{audit.get('verdict')}"
        )
    if audit.get("passed") is not True or audit.get("identity_residual") != 0:
        raise ValueError(
            f"MLX response {path} candidate cache audit is not zero-residual"
        )


def _response_candidate_cache_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return _response_cache_identity(payload, "candidate")


def _response_cache_identity(payload: dict[str, Any], side: str) -> dict[str, Any]:
    cache_identity = payload.get("cache_identity")
    if not isinstance(cache_identity, dict):
        raise ValueError("MLX response cache_identity is missing")
    item = cache_identity.get(side)
    if not isinstance(item, dict):
        raise ValueError(f"MLX response {side} cache_identity is missing")
    return item


def _public_cache_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": identity.get("path"),
        "archive_sha256": identity.get("archive_sha256"),
        "inflated_outputs_aggregate_sha256": identity.get("inflated_outputs_aggregate_sha256"),
        "raw_sha256": identity.get("raw_sha256"),
        "pair_count": identity.get("pair_count"),
        "hash_domain": identity.get("hash_domain"),
        "array_sha256": identity.get("array_sha256"),
        "segnet_last_rgb_shape": identity.get("segnet_last_rgb_shape"),
        "posenet_yuv6_pair_shape": identity.get("posenet_yuv6_pair_shape"),
        "pair_indices_shape": identity.get("pair_indices_shape"),
        "eligible_for_local_mlx_transfer_calibration": identity.get(
            "eligible_for_local_mlx_transfer_calibration"
        ),
    }


def _required_pair_window(value: Any, path: Path) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"MLX response {path} pair_window missing")
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MLX response {path} pair_window invalid") from exc


def _auth_archive_sha256(payload: dict[str, Any]) -> str | None:
    value = _optional_str(payload.get("archive_sha256"))
    if value:
        return value
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        return _optional_str(provenance.get("archive_sha256"))
    return None


def _auth_inflated_outputs_aggregate_sha256(payload: dict[str, Any]) -> str | None:
    value = _optional_str(payload.get("inflated_outputs_aggregate_sha256"))
    if value:
        return value
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return None
    manifest = provenance.get("inflated_output_manifest")
    if isinstance(manifest, dict):
        payload_obj = manifest.get("payload")
        if isinstance(payload_obj, dict):
            nested = _optional_str(payload_obj.get("aggregate_sha256"))
            if nested:
                return nested
        return _optional_str(manifest.get("aggregate_sha256"))
    return None


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
        "promotable": False,
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
        rows_with_axis = [row for row in rows if row.get(f"{axis}_score") is not None]
        summary[f"{axis}_auth_axis_row_count"] = len(rows_with_axis)
        summary[f"mlx_{axis}_pairwise_comparison_count"] = len(comparable)
        summary[f"mlx_{axis}_rank_inversions"] = sum(
            1 for item in comparable if item[match_key] is not True
        )
    summary["planning_advisory_dual_axis_summary"] = _build_dual_axis_summary(rows)
    return summary


def _build_decision_policy(
    summary: dict[str, Any],
    safety_factor: float,
) -> dict[str, Any]:
    axis_policies = {
        axis: _build_axis_decision_policy(summary, axis, safety_factor)
        for axis in ("cpu", "cuda")
    }
    cuda_policy = axis_policies["cuda"]
    calibration_uncertainty_score = cuda_policy.get("calibration_uncertainty_score")
    min_gap = (
        cuda_policy.get("min_gap_for_spend_triage")
        if cuda_policy.get("spend_triage_allowed") is True
        else None
    )
    blockers: list[str] = []
    warnings: list[str] = []
    for axis, policy in axis_policies.items():
        warnings.extend(str(item) for item in policy.get("warnings", []))
        if axis == "cuda":
            blockers.extend(str(item) for item in policy.get("blockers", []))
    cpu_policy = axis_policies["cpu"]
    if (
        cpu_policy.get("has_strict_auth_axis_calibration") is True
        and cuda_policy.get("has_strict_auth_axis_calibration") is not True
    ):
        blockers.append("cpu_only_calibration_cannot_authorize_cuda_routing")
    return {
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "decision_safety_factor": safety_factor,
        "axis_decision_policies": axis_policies,
        "calibration_uncertainty_basis": (
            cuda_policy.get("calibration_uncertainty_basis")
            if cuda_policy.get("spend_triage_allowed") is True
            else None
        ),
        "calibration_uncertainty_score": calibration_uncertainty_score,
        "recommended_min_mlx_gap_for_spend_triage": min_gap,
        "allowed_use": (
            STRICT_AUTH_AXIS_SPEND_TRIAGE_ALLOWED_USE
            if cuda_policy.get("spend_triage_allowed") is True
            else "diagnostic_only_cuda_auth_axis_calibration_missing_or_insufficient"
        ),
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "forbidden_use": "score_claim_or_rank_or_kill_or_promotion",
    }


def _attach_decision_certification(
    pairwise: list[dict[str, Any]],
    decision_policy: dict[str, Any],
) -> None:
    min_gap = decision_policy.get("recommended_min_mlx_gap_for_spend_triage")
    axis_policies = decision_policy.get("axis_decision_policies")
    if not isinstance(axis_policies, dict):
        axis_policies = {}
    for item in pairwise:
        for axis in ("cpu", "cuda"):
            policy = axis_policies.get(axis)
            if not isinstance(policy, dict):
                policy = {}
            axis_gap_key = f"{axis}_score_gap_abs"
            if axis_gap_key not in item:
                continue
            axis_min_gap = policy.get("min_gap_for_spend_triage")
            axis_gap = _finite_float(item[axis_gap_key], f"pairwise.{axis_gap_key}")
            axis_certified = (
                policy.get("spend_triage_allowed") is True
                and axis_min_gap is not None
                and axis_gap >= float(axis_min_gap)
            )
            item[f"mlx_{axis}_spend_triage_decision_certified"] = axis_certified
            item[f"mlx_{axis}_spend_triage_uncertain"] = not axis_certified
            item[f"mlx_{axis}_spend_triage_min_gap"] = axis_min_gap
        gap = _finite_float(item["mlx_score_gap_abs"], "pairwise.mlx_score_gap_abs")
        certified = (
            min_gap is not None
            and "cuda_score_gap_abs" in item
            and gap >= float(min_gap)
            and axis_policies.get("cuda", {}).get("spend_triage_allowed") is True
        )
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
    axis_policies = decision_policy.get("axis_decision_policies")
    if isinstance(axis_policies, dict):
        summary["axis_specific_min_gap_for_spend_triage"] = {
            axis: (
                policy.get("min_gap_for_spend_triage")
                if isinstance(policy, dict)
                else None
            )
            for axis, policy in axis_policies.items()
        }
        summary["axis_calibration"] = axis_policies
        for axis in ("cpu", "cuda"):
            axis_certified = [
                item
                for item in pairwise
                if item.get(f"mlx_{axis}_spend_triage_decision_certified") is True
            ]
            axis_uncertain = [
                item for item in pairwise if item.get(f"mlx_{axis}_spend_triage_uncertain") is True
            ]
            summary[f"mlx_{axis}_spend_triage_pairwise_certified_count"] = len(
                axis_certified
            )
            summary[f"mlx_{axis}_spend_triage_pairwise_uncertain_count"] = len(
                axis_uncertain
            )
    summary["calibration_uncertainty_score"] = decision_policy.get(
        "calibration_uncertainty_score"
    )


def _build_axis_decision_policy(
    summary: dict[str, Any],
    axis: str,
    safety_factor: float,
) -> dict[str, Any]:
    error_key = {
        "cpu": "mlx_minus_cpu_max_abs",
        "cuda": "cuda_minus_mlx_max_abs",
    }[axis]
    row_count = int(summary.get(f"{axis}_auth_axis_row_count") or 0)
    pairwise_count = int(summary.get(f"mlx_{axis}_pairwise_comparison_count") or 0)
    raw_error = summary.get(error_key)
    calibration_uncertainty_score = (
        None
        if raw_error is None
        else _finite_float(raw_error, f"summary.{error_key}")
    )
    min_gap = (
        None
        if calibration_uncertainty_score is None
        else calibration_uncertainty_score * safety_factor
    )
    has_calibration = calibration_uncertainty_score is not None
    sample_scarce = has_calibration and (
        row_count < MIN_AXIS_ROWS_FOR_SPEND_TRIAGE
        or pairwise_count < MIN_AXIS_PAIRWISE_COMPARISONS_FOR_SPEND_TRIAGE
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not has_calibration:
        blockers.append(f"{axis}_auth_axis_calibration_missing")
    elif sample_scarce:
        blockers.append(f"{axis}_auth_axis_calibration_sample_scarce")
        warnings.append(
            f"{axis}_auth_axis_calibration_sample_scarce:"
            f"rows={row_count}:pairwise={pairwise_count}:"
            f"min_rows={MIN_AXIS_ROWS_FOR_SPEND_TRIAGE}:"
            f"min_pairwise={MIN_AXIS_PAIRWISE_COMPARISONS_FOR_SPEND_TRIAGE}"
        )
    allowed = has_calibration and not sample_scarce
    return {
        "auth_axis": f"contest_{axis}",
        "has_strict_auth_axis_calibration": has_calibration,
        "spend_triage_allowed": allowed,
        "row_count": row_count,
        "pairwise_comparison_count": pairwise_count,
        "rank_inversions_vs_mlx": int(summary.get(f"mlx_{axis}_rank_inversions") or 0),
        "sample_scarce": bool(sample_scarce),
        "min_rows_for_spend_triage": MIN_AXIS_ROWS_FOR_SPEND_TRIAGE,
        "min_pairwise_comparisons_for_spend_triage": (
            MIN_AXIS_PAIRWISE_COMPARISONS_FOR_SPEND_TRIAGE
        ),
        "calibration_uncertainty_basis": error_key if has_calibration else None,
        "calibration_uncertainty_score": calibration_uncertainty_score,
        "min_gap_for_spend_triage": min_gap,
        "recommended_min_mlx_gap_for_spend_triage": min_gap,
        "allowed_use": (
            f"local_spend_triage_only_after_strict_{axis}_auth_axis_calibration"
            if allowed
            else f"diagnostic_only_{axis}_auth_axis_calibration_missing_or_insufficient"
        ),
        "blockers": blockers,
        "warnings": warnings,
    }


def _build_dual_axis_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_with_both = [
        row
        for row in rows
        if row.get("cpu_score") is not None and row.get("cuda_score") is not None
    ]
    groups_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows_with_both:
        family = str(row.get("response_family") or "unknown_family")
        archive_sha256 = str(row.get("archive_sha256") or "unknown_archive")
        groups_by_key.setdefault((family, archive_sha256), []).append(row)
    groups = [
        _summarize_cpu_cuda_group(
            group_rows,
            response_family=family,
            archive_sha256=archive_sha256,
        )
        for (family, archive_sha256), group_rows in sorted(groups_by_key.items())
    ]
    overall = _summarize_cpu_cuda_group(
        rows_with_both,
        response_family="all",
        archive_sha256="all",
    )
    return {
        "schema": "mlx_cpu_cuda_dual_axis_planning_summary.v1",
        "advisory_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "row_count_with_both_axes": len(rows_with_both),
        "sample_scarcity_thresholds": {
            "min_rows": MIN_AXIS_ROWS_FOR_SPEND_TRIAGE,
            "min_pairwise_comparisons": MIN_AXIS_PAIRWISE_COMPARISONS_FOR_SPEND_TRIAGE,
        },
        "overall": overall,
        "groups": groups,
    }


def _summarize_cpu_cuda_group(
    rows: list[dict[str, Any]],
    *,
    response_family: str,
    archive_sha256: str,
) -> dict[str, Any]:
    row_count = len(rows)
    pairwise_count = 0
    cpu_improvement_count = 0
    cuda_regression_count = 0
    cuda_non_improvement_count = 0
    cuda_confirms_count = 0
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            pairwise_count += 1
            cpu_order = _order(left.get("cpu_score"), right.get("cpu_score"))
            cuda_order = _order(left.get("cuda_score"), right.get("cuda_score"))
            if cpu_order == 0:
                continue
            cpu_improvement_count += 1
            if cuda_order == cpu_order:
                cuda_confirms_count += 1
            else:
                cuda_non_improvement_count += 1
                if cuda_order == -cpu_order:
                    cuda_regression_count += 1
    sample_scarce = (
        row_count < MIN_AXIS_ROWS_FOR_SPEND_TRIAGE
        or pairwise_count < MIN_AXIS_PAIRWISE_COMPARISONS_FOR_SPEND_TRIAGE
    )
    out: dict[str, Any] = {
        "response_family": response_family,
        "archive_sha256": archive_sha256,
        "row_count": row_count,
        "pairwise_comparison_count": pairwise_count,
        "cpu_improvement_comparison_count": cpu_improvement_count,
        "cuda_confirms_cpu_improvement_count": cuda_confirms_count,
        "cuda_non_improvement_given_cpu_improvement_count": cuda_non_improvement_count,
        "cuda_regression_given_cpu_improvement_count": cuda_regression_count,
        "p_cuda_regression_given_cpu_improvement_empirical": (
            None
            if cpu_improvement_count == 0
            else cuda_regression_count / cpu_improvement_count
        ),
        "p_cuda_regression_given_cpu_improvement_conservative_count_based": (
            1.0
            if cpu_improvement_count == 0
            else (cuda_regression_count + 1) / (cpu_improvement_count + 2)
        ),
        "sample_scarce": sample_scarce,
        "warnings": [],
    }
    if sample_scarce:
        out["warnings"].append(
            "dual_axis_cpu_cuda_sample_scarce_fail_closed:"
            f"rows={row_count}:pairwise={pairwise_count}:"
            f"min_rows={MIN_AXIS_ROWS_FOR_SPEND_TRIAGE}:"
            f"min_pairwise={MIN_AXIS_PAIRWISE_COMPARISONS_FOR_SPEND_TRIAGE}"
        )
    _attach_observation_stats(
        out,
        "cuda_minus_cpu",
        [float(row["cuda_minus_cpu"]) for row in rows if row.get("cuda_minus_cpu") is not None],
    )
    return out


def _attach_observation_stats(
    summary: dict[str, Any],
    key: str,
    values: list[float],
) -> None:
    if not values:
        return
    summary[f"{key}_mean"] = sum(values) / len(values)
    summary[f"{key}_min"] = min(values)
    summary[f"{key}_max"] = max(values)
    summary[f"{key}_max_abs"] = max(abs(value) for value in values)


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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _first_text(*values: Any) -> str:
    value = _first_present(*values)
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value).strip()
    return str(value).strip()


def _command_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value).strip()
    return str(value or "").strip()


def _command_flag_value(command: str, flag: str) -> str:
    parts = command.split()
    for idx, part in enumerate(parts[:-1]):
        if part == flag:
            return parts[idx + 1]
    prefix = f"{flag}="
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


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


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _required_sha256(value: Any, label: str) -> str:
    text = _optional_str(value)
    if text is None or len(text) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        raise ValueError(f"{label} must be a SHA-256 hex string")
    return text.lower()


__all__ = [
    "DEFAULT_DECISION_SAFETY_FACTOR",
    "MIN_AXIS_PAIRWISE_COMPARISONS_FOR_SPEND_TRIAGE",
    "MIN_AXIS_ROWS_FOR_SPEND_TRIAGE",
    "SCHEMA_VERSION",
    "STRICT_AUTH_AXIS_SPEND_TRIAGE_ALLOWED_USE",
    "build_mlx_score_calibration_manifest",
    "load_json_object",
    "write_mlx_score_calibration_manifest",
]
