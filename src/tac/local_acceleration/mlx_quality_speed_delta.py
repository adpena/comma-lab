# SPDX-License-Identifier: MIT
"""Quality/speed delta manifests for local MLX scorer-response debugging.

The output is intentionally non-authoritative.  It is a compact bridge between
local CPU/advisory anchors, MLX scorer responses, and calibration rows so the
solver can reason about whether a local MLX run is a useful acquisition signal
or a drift probe that needs repair.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tac.auth_eval_schema import eval_metric_summary
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

SCHEMA_VERSION = "mlx_quality_speed_delta.v1"

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


def write_quality_speed_delta_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_quality_speed_delta_manifest(
    *,
    anchor_payload: dict[str, Any],
    mlx_payloads: list[dict[str, Any]],
    anchor_path: str | Path | None = None,
    mlx_paths: list[str | Path] | None = None,
    calibration_payload: dict[str, Any] | None = None,
    calibration_path: str | Path | None = None,
    run_id: str | None = None,
    frontier_score: float | None = None,
    calibration_safety_factor: float = 5.0,
) -> dict[str, Any]:
    """Compare one CPU/advisory anchor against one or more MLX responses."""

    if not isinstance(anchor_payload, dict):
        raise ValueError("anchor_payload must be an object")
    if not mlx_payloads:
        raise ValueError("mlx_payloads must not be empty")
    if mlx_paths is not None and len(mlx_paths) != len(mlx_payloads):
        raise ValueError("mlx_paths length must match mlx_payloads length")
    if calibration_safety_factor <= 0 or not math.isfinite(float(calibration_safety_factor)):
        raise ValueError("calibration_safety_factor must be positive and finite")

    anchor = _anchor_metrics(anchor_payload, anchor_path)
    calibration = _calibration_summary(
        calibration_payload,
        calibration_path=calibration_path,
        safety_factor=float(calibration_safety_factor),
    )
    rows = [
        _mlx_row(
            payload=payload,
            path=None if mlx_paths is None else mlx_paths[index],
            anchor=anchor,
            calibration=calibration,
        )
        for index, payload in enumerate(mlx_payloads)
    ]
    best_speed = max(rows, key=lambda row: _none_to_neg_inf(row.get("speedup_vs_anchor_elapsed")))
    smallest_abs_delta = min(rows, key=lambda row: abs(float(row["score_delta"])))
    blockers = sorted({blocker for row in rows for blocker in row["blockers"]})
    frontier = _frontier_block(frontier_score, anchor, rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "anchor": anchor,
        "calibration": calibration,
        "frontier": frontier,
        "rows": rows,
        "summary": {
            "row_count": len(rows),
            "blocking_reason_count": len(blockers),
            "blockers": blockers,
            "best_speed_row": _row_pointer(best_speed),
            "smallest_abs_score_delta_row": _row_pointer(smallest_abs_delta),
            "all_rows_blocked_for_spend_triage": all(
                row["spend_triage_allowed"] is False for row in rows
            ),
            "any_singleton_cpu_row": any(
                row["device_type"] == "cpu" and row["batch_pairs"] == 1 for row in rows
            ),
            "any_gpu_or_batch_shape_research_row": any(
                row["device_type"] == "gpu" or row["batch_pairs"] != 1 for row in rows
            ),
        },
        "authority_status": (
            "Quality/speed deltas are local MLX debugging and acquisition evidence "
            "only. Exact contest CPU/CUDA auth eval remains required for score "
            "claims, promotion, rank/kill decisions, and dispatch readiness."
        ),
    }


def _anchor_metrics(payload: dict[str, Any], path: str | Path | None) -> dict[str, Any]:
    metrics = eval_metric_summary(payload)
    score = _finite_metric(metrics.get("score"), "anchor score")
    seg = _finite_metric(metrics.get("seg_avg"), "anchor seg_avg")
    pose = _finite_metric(metrics.get("pose_avg"), "anchor pose_avg")
    archive_size = metrics.get("archive_size_bytes")
    n_samples = metrics.get("n_samples")
    elapsed = _first_finite(
        payload.get("contest_auth_eval_elapsed_seconds"),
        payload.get("evaluate_elapsed_seconds"),
        payload.get("elapsed_seconds"),
    )
    return {
        "path": None if path is None else str(path),
        "score_axis": payload.get("score_axis"),
        "evidence_grade": payload.get("evidence_grade"),
        "evidence_semantics": payload.get("evidence_semantics"),
        "canonical_score": score,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": None if archive_size is None else int(archive_size),
        "n_samples": None if n_samples is None else int(n_samples),
        "elapsed_seconds": elapsed,
        "archive_sha256": _payload_archive_sha256(payload),
        "inflated_outputs_aggregate_sha256": _payload_inflated_sha256(payload),
        "raw_sha256": _payload_raw_sha256(payload),
        "authority_fields": {
            field: payload.get(field) for field in AUTHORITY_FALSE_FIELDS if field in payload
        },
    }


def _mlx_row(
    *,
    payload: dict[str, Any],
    path: str | Path | None,
    anchor: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("each MLX payload must be an object")
    metrics = eval_metric_summary(payload)
    score = _finite_metric(metrics.get("score"), "mlx score")
    seg = _finite_metric(metrics.get("seg_avg"), "mlx seg_avg")
    pose = _finite_metric(metrics.get("pose_avg"), "mlx pose_avg")
    elapsed = _first_finite(payload.get("elapsed_seconds"))
    batch_pairs = int(payload.get("batch_pairs", 0))
    n_samples = int(payload.get("n_samples", 0))
    device_type = _device_type(payload)
    score_delta = score - float(anchor["canonical_score"])
    seg_delta = seg - float(anchor["avg_segnet_dist"])
    pose_delta = pose - float(anchor["avg_posenet_dist"])
    score_seg_delta = 100.0 * seg_delta
    score_pose_delta = _pose_contribution(pose) - _pose_contribution(float(anchor["avg_posenet_dist"]))
    score_rate_delta = _rate_contribution(payload) - _rate_contribution(anchor)
    candidate_identity = _candidate_cache_identity(payload)
    audited = _candidate_cache_audited(candidate_identity)
    identity_blockers = _identity_blockers(
        payload=payload,
        anchor=anchor,
        candidate_identity=candidate_identity,
        n_samples=n_samples,
    )
    blockers = _mlx_blockers(
        payload=payload,
        device_type=device_type,
        batch_pairs=batch_pairs,
        candidate_cache_audited=audited,
        score_delta=score_delta,
        calibration=calibration,
    )
    blockers.extend(identity_blockers)
    blockers = sorted(set(blockers))
    row = {
        "path": None if path is None else str(path),
        "label": Path(path).stem if path is not None else str(payload.get("run_id") or "mlx_response"),
        "device_type": device_type,
        "hardware_substrate": payload.get("hardware_substrate"),
        "batch_pairs": batch_pairs,
        "n_samples": n_samples,
        "pair_window": payload.get("pair_window"),
        "elapsed_seconds": elapsed,
        "pairs_per_second": None if not elapsed or elapsed <= 0 else n_samples / elapsed,
        "speedup_vs_anchor_elapsed": _speedup(anchor.get("elapsed_seconds"), elapsed),
        "canonical_score": score,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "score_delta": score_delta,
        "score_abs_delta": abs(score_delta),
        "segnet_dist_delta": seg_delta,
        "posenet_dist_delta": pose_delta,
        "score_component_delta": {
            "segnet": score_seg_delta,
            "pose": score_pose_delta,
            "rate": score_rate_delta,
            "sum": score_seg_delta + score_pose_delta + score_rate_delta,
        },
        "archive_sha256": payload.get("archive_sha256"),
        "inflated_outputs_aggregate_sha256": payload.get("inflated_outputs_aggregate_sha256"),
        "raw_sha256": payload.get("raw_sha256"),
        "candidate_cache_identity_audited": audited,
        "candidate_cache_identity": _public_cache_identity(candidate_identity),
        "identity_blockers": identity_blockers,
        "component_hashes": {
            "posenet_sha256": (payload.get("components") or {}).get("posenet_sha256"),
            "segnet_sha256": (payload.get("components") or {}).get("segnet_sha256"),
        },
        "calibration_gap_ratio": _gap_ratio(abs(score_delta), calibration.get("decision_band")),
        "blockers": blockers,
        "spend_triage_allowed": not blockers,
        "recommended_use": (
            "local_mlx_acquisition_signal_after_calibration"
            if not blockers
            else "diagnostic_drift_probe_only"
        ),
    }
    return row


def _identity_blockers(
    *,
    payload: dict[str, Any],
    anchor: dict[str, Any],
    candidate_identity: dict[str, Any],
    n_samples: int,
) -> list[str]:
    blockers: list[str] = []
    anchor_n = anchor.get("n_samples")
    if anchor_n != 600:
        blockers.append("anchor_n_samples_not_full_contest")
    if n_samples != 600:
        blockers.append("mlx_n_samples_not_full_contest")
    if anchor_n is not None and n_samples != int(anchor_n):
        blockers.append("mlx_n_samples_mismatch_anchor")
    pair_window = payload.get("pair_window")
    if pair_window != [0, 600]:
        blockers.append("mlx_pair_window_not_full_contest")
    _append_identity_match_blocker(
        blockers,
        label="archive_sha256",
        anchor_value=anchor.get("archive_sha256"),
        payload_value=payload.get("archive_sha256"),
        cache_value=candidate_identity.get("archive_sha256"),
    )
    _append_identity_match_blocker(
        blockers,
        label="inflated_outputs_aggregate_sha256",
        anchor_value=anchor.get("inflated_outputs_aggregate_sha256"),
        payload_value=payload.get("inflated_outputs_aggregate_sha256"),
        cache_value=candidate_identity.get("inflated_outputs_aggregate_sha256"),
    )
    _append_identity_match_blocker(
        blockers,
        label="raw_sha256",
        anchor_value=anchor.get("raw_sha256"),
        payload_value=payload.get("raw_sha256"),
        cache_value=candidate_identity.get("raw_sha256"),
    )
    pair_count = candidate_identity.get("pair_count")
    if pair_count != 600:
        blockers.append("candidate_cache_pair_count_not_full_contest")
    return blockers


def _append_identity_match_blocker(
    blockers: list[str],
    *,
    label: str,
    anchor_value: Any,
    payload_value: Any,
    cache_value: Any,
) -> None:
    values = {
        "anchor": anchor_value,
        "mlx_payload": payload_value,
        "candidate_cache": cache_value,
    }
    missing = [name for name, value in values.items() if not isinstance(value, str) or not value]
    if missing:
        blockers.append(f"{label}_identity_missing:" + ",".join(missing))
        return
    if len({str(value) for value in values.values()}) != 1:
        blockers.append(f"{label}_identity_mismatch")


def _mlx_blockers(
    *,
    payload: dict[str, Any],
    device_type: str,
    batch_pairs: int,
    candidate_cache_audited: bool,
    score_delta: float,
    calibration: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if payload.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        blockers.append("mlx_evidence_grade_missing_or_not_mlx")
    if payload.get("evidence_tag") != EVIDENCE_TAG_MLX:
        blockers.append("mlx_evidence_tag_missing_or_not_mlx")
    if payload.get("score_axis") != EVIDENCE_TAG_MLX:
        blockers.append("mlx_score_axis_missing_or_not_mlx")
    for field in AUTHORITY_FALSE_FIELDS:
        if payload.get(field) is True:
            blockers.append(f"mlx_payload_attempts_{field}")
    if not candidate_cache_audited:
        blockers.append("candidate_cache_missing_pass_cache_auth_eval_identity")
    if device_type == "gpu":
        blockers.append("mlx_gpu_response_requires_separate_cpu_transfer_calibration")
    if batch_pairs != 1:
        blockers.append("mlx_non_singleton_batch_shape_requires_passing_invariance_gate")
    band = calibration.get("decision_band")
    if isinstance(band, (int, float)) and math.isfinite(float(band)):
        if abs(score_delta) > float(band):
            blockers.append("score_delta_exceeds_calibration_decision_band")
    else:
        blockers.append("calibration_decision_band_missing")
    return sorted(set(blockers))


def _calibration_summary(
    payload: dict[str, Any] | None,
    *,
    calibration_path: str | Path | None,
    safety_factor: float,
) -> dict[str, Any]:
    if payload is None:
        return {
            "path": None if calibration_path is None else str(calibration_path),
            "available": False,
            "decision_band": None,
            "blockers": ["calibration_payload_missing"],
        }
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    policy = payload.get("decision_policy") if isinstance(payload.get("decision_policy"), dict) else {}
    max_abs_local = _first_finite(
        summary.get("max_abs_mlx_minus_local_cpu"),
        summary.get("mlx_minus_local_cpu_max_abs"),
    )
    max_abs_cpu = _first_finite(
        summary.get("max_abs_mlx_minus_cpu"),
        summary.get("mlx_minus_cpu_max_abs"),
    )
    decision_band = _first_finite(
        summary.get("recommended_min_mlx_gap_for_spend_triage"),
        policy.get("recommended_min_mlx_gap_for_spend_triage"),
    )
    basis = "reported"
    if decision_band is None:
        basis_value = max_abs_cpu if max_abs_cpu is not None else max_abs_local
        decision_band = None if basis_value is None else float(basis_value) * safety_factor
        basis = "safety_factor_times_max_abs_mlx_minus_cpu_or_local"
    return {
        "path": None if calibration_path is None else str(calibration_path),
        "available": True,
        "schema_version": payload.get("schema_version"),
        "row_count": payload.get("row_count") or len(payload.get("rows") or []),
        "safety_factor": safety_factor,
        "decision_band": decision_band,
        "decision_band_basis": basis,
        "max_abs_mlx_minus_cpu": max_abs_cpu,
        "max_abs_mlx_minus_local_cpu": max_abs_local,
        "mean_mlx_minus_cpu": _first_finite(summary.get("mean_mlx_minus_cpu")),
        "mean_mlx_minus_local_cpu": _first_finite(summary.get("mean_mlx_minus_local_cpu")),
        "rank_inversions": {
            "mlx_cpu": summary.get("mlx_cpu_rank_inversions"),
            "mlx_cuda": summary.get("mlx_cuda_rank_inversions"),
            "cuda_cpu": summary.get("cuda_cpu_rank_inversions"),
        },
        "blockers": [] if decision_band is not None else ["calibration_decision_band_missing"],
    }


def _frontier_block(
    frontier_score: float | None,
    anchor: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if frontier_score is None:
        return {"frontier_score": None, "available": False}
    frontier = float(frontier_score)
    return {
        "frontier_score": frontier,
        "available": True,
        "anchor_minus_frontier": float(anchor["canonical_score"]) - frontier,
        "mlx_rows_minus_frontier": [
            {
                "label": row["label"],
                "score_minus_frontier": float(row["canonical_score"]) - frontier,
                "axis": EVIDENCE_TAG_MLX,
                "authority": "non_authoritative",
            }
            for row in rows
        ],
    }


def _row_pointer(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row.get("label"),
        "path": row.get("path"),
        "device_type": row.get("device_type"),
        "batch_pairs": row.get("batch_pairs"),
        "canonical_score": row.get("canonical_score"),
        "score_delta": row.get("score_delta"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "speedup_vs_anchor_elapsed": row.get("speedup_vs_anchor_elapsed"),
    }


def _finite_metric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} missing or not finite")
    return float(value)


def _first_finite(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _device_type(payload: dict[str, Any]) -> str | None:
    value = str(payload.get("hardware_substrate") or "").lower()
    if "gpu" in value:
        return "gpu"
    if "cpu" in value:
        return "cpu"
    contract = payload.get("device_contract")
    if isinstance(contract, dict) and contract.get("gpu_research_signal_required") is True:
        return "gpu"
    return None


def _candidate_cache_identity(payload: dict[str, Any]) -> dict[str, Any]:
    cache_identity = payload.get("cache_identity")
    if not isinstance(cache_identity, dict):
        return {}
    candidate = cache_identity.get("candidate")
    return candidate if isinstance(candidate, dict) else {}


def _candidate_cache_audited(candidate_identity: dict[str, Any]) -> bool:
    audit = candidate_identity.get("auth_eval_identity_audit")
    return bool(
        candidate_identity.get("eligible_for_local_mlx_transfer_calibration") is True
        and isinstance(audit, dict)
        and audit.get("verdict") == "PASS_CACHE_AUTH_EVAL_IDENTITY"
        and audit.get("passed") is True
        and audit.get("identity_residual") == 0
    )


def _public_cache_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        key: identity.get(key)
        for key in (
            "path",
            "archive_sha256",
            "inflated_outputs_aggregate_sha256",
            "raw_sha256",
            "hash_domain",
            "pair_count",
            "eligible_for_local_mlx_transfer_calibration",
        )
        if key in identity
    }


def _pose_contribution(pose: float) -> float:
    return math.sqrt(10.0 * float(pose))


def _rate_contribution(payload: dict[str, Any]) -> float:
    value = payload.get("score_rate_contribution")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    metrics = eval_metric_summary(payload)
    rate = metrics.get("rate")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool):
        return float(rate)
    archive_size = metrics.get("archive_size_bytes")
    if isinstance(archive_size, int):
        from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES

        return 25.0 * archive_size / ORIGINAL_VIDEO_BYTES
    return 0.0


def _speedup(anchor_elapsed: Any, mlx_elapsed: Any) -> float | None:
    if not isinstance(anchor_elapsed, (int, float)) or not isinstance(mlx_elapsed, (int, float)):
        return None
    if float(anchor_elapsed) <= 0 or float(mlx_elapsed) <= 0:
        return None
    return float(anchor_elapsed) / float(mlx_elapsed)


def _gap_ratio(value: float, band: Any) -> float | None:
    if not isinstance(band, (int, float)) or float(band) <= 0:
        return None
    return float(value) / float(band)


def _none_to_neg_inf(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float("-inf")


def _payload_archive_sha256(payload: dict[str, Any]) -> str | None:
    value = payload.get("archive_sha256")
    if isinstance(value, str) and value:
        return value
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        value = provenance.get("archive_sha256")
        if isinstance(value, str) and value:
            return value
    return None


def _payload_inflated_sha256(payload: dict[str, Any]) -> str | None:
    value = payload.get("inflated_outputs_aggregate_sha256")
    if isinstance(value, str) and value:
        return value
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        value = provenance.get("inflated_outputs_aggregate_sha256")
        if isinstance(value, str) and value:
            return value
    return None


def _payload_raw_sha256(payload: dict[str, Any]) -> str | None:
    value = payload.get("raw_sha256")
    if isinstance(value, str) and value:
        return value
    raw = payload.get("raw")
    if isinstance(raw, dict):
        value = raw.get("sha256")
        if isinstance(value, str) and value:
            return value
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        value = provenance.get("raw_sha256") or provenance.get("raw_file_sha256")
        if isinstance(value, str) and value:
            return value
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


__all__ = [
    "SCHEMA_VERSION",
    "build_quality_speed_delta_manifest",
    "load_json_object",
    "write_quality_speed_delta_manifest",
]
