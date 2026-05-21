# SPDX-License-Identifier: MIT
"""Audit MLX scorer-input cache custody against an auth-eval target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tac.auth_eval_schema import eval_metric_summary
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

__all__ = [
    "audit_mlx_scorer_input_cache_against_auth_eval",
    "write_cache_audit",
]

SCHEMA_VERSION = "mlx_scorer_input_cache_auth_eval_audit.v1"
PASS_VERDICT = "PASS_CACHE_AUTH_EVAL_IDENTITY"
FAIL_VERDICT = "FAIL_CACHE_AUTH_EVAL_IDENTITY"


def audit_mlx_scorer_input_cache_against_auth_eval(
    cache_manifest: dict[str, Any],
    auth_eval_payload: dict[str, Any],
    *,
    expected_pair_count: int | None = None,
) -> dict[str, Any]:
    """Return pass/fail custody audit for a cache and auth-eval JSON."""

    blockers: list[str] = []
    metrics = eval_metric_summary(auth_eval_payload)
    cache_archive_sha = _string(cache_manifest.get("archive_sha256"))
    auth_archive_sha = _archive_sha256(auth_eval_payload)
    cache_inflated_sha = _string(cache_manifest.get("inflated_outputs_aggregate_sha256"))
    auth_inflated_sha = _inflated_outputs_aggregate_sha256(auth_eval_payload)
    cache_pair_count = _int(cache_manifest.get("pair_count"))
    auth_n_samples = _int(metrics.get("n_samples"))
    expected = expected_pair_count if expected_pair_count is not None else auth_n_samples

    if not cache_archive_sha or cache_archive_sha != auth_archive_sha:
        blockers.append("archive_sha256_mismatch_or_missing")
    if not cache_inflated_sha or cache_inflated_sha != auth_inflated_sha:
        blockers.append("inflated_outputs_aggregate_sha256_mismatch_or_missing")
    if cache_pair_count is None:
        blockers.append("cache_pair_count_missing")
    elif expected is not None and cache_pair_count != expected:
        blockers.append(f"cache_pair_count_mismatch:cache={cache_pair_count}:expected={expected}")
    if auth_n_samples is None:
        blockers.append("auth_eval_n_samples_missing")
    if cache_manifest.get("score_claim") is True or cache_manifest.get("promotion_eligible") is True:
        blockers.append("cache_manifest_attempts_score_authority")

    passed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "passed": passed,
        "blockers": blockers,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cache": {
            "archive_sha256": cache_archive_sha,
            "inflated_outputs_aggregate_sha256": cache_inflated_sha,
            "raw_sha256": _string(cache_manifest.get("raw_sha256")),
            "pair_count": cache_pair_count,
            "segnet_last_rgb_shape": cache_manifest.get("segnet_last_rgb_shape"),
            "posenet_yuv6_pair_shape": cache_manifest.get("posenet_yuv6_pair_shape"),
            "artifacts": cache_manifest.get("artifacts"),
            "array_sha256": cache_manifest.get("array_sha256"),
        },
        "auth_eval": {
            "archive_sha256": auth_archive_sha,
            "inflated_outputs_aggregate_sha256": auth_inflated_sha,
            "raw_file_sha256": _first_raw_file_sha256(auth_eval_payload),
            "n_samples": auth_n_samples,
            "score": metrics.get("score"),
            "pose_avg": metrics.get("pose_avg"),
            "seg_avg": metrics.get("seg_avg"),
            "rate": metrics.get("rate"),
            "evidence_grade": auth_eval_payload.get("evidence_grade"),
            "score_axis": auth_eval_payload.get("score_axis"),
            "lane_tag": auth_eval_payload.get("lane_tag"),
        },
        "allowed_use": (
            [
                "local_mlx_training_transfer_calibration",
                "surrogate_error_measurement_against_matching_auth_axis",
            ]
            if passed
            else [
                "local_tensor_ingestion_debug_only",
                "do_not_use_for_auth_axis_transfer_calibration",
            ]
        ),
    }


def write_cache_audit(audit: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _archive_sha256(payload: dict[str, Any]) -> str | None:
    value = _string(payload.get("archive_sha256"))
    if value:
        return value
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        return _string(provenance.get("archive_sha256"))
    return None


def _inflated_outputs_aggregate_sha256(payload: dict[str, Any]) -> str | None:
    value = _string(payload.get("inflated_outputs_aggregate_sha256"))
    if value:
        return value
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return None
    manifest = provenance.get("inflated_output_manifest")
    if isinstance(manifest, dict):
        payload_obj = manifest.get("payload")
        if isinstance(payload_obj, dict):
            value = _string(payload_obj.get("aggregate_sha256"))
            if value:
                return value
        return _string(manifest.get("aggregate_sha256"))
    return None


def _first_raw_file_sha256(payload: dict[str, Any]) -> str | None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return None
    manifest = provenance.get("inflated_output_manifest")
    if not isinstance(manifest, dict):
        return None
    payload_obj = manifest.get("payload")
    if not isinstance(payload_obj, dict):
        return None
    files = payload_obj.get("files")
    if not isinstance(files, list) or not files:
        return None
    first = files[0]
    if not isinstance(first, dict):
        return None
    return _string(first.get("sha256"))


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None
