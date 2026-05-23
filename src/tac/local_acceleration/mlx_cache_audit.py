# SPDX-License-Identifier: MIT
"""Audit MLX scorer-input cache custody against an auth-eval target."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tac.auth_eval_schema import (
    CONTEST_AUTH_AXIS_BY_EVIDENCE_GRADE,
    FULL_CONTEST_SAMPLE_COUNT,
    eval_metric_summary,
    required_contest_auth_axis_payload_blockers,
)
from tac.canonical_equations.scorer_input_cache_hash_identity import (
    scorer_input_cache_hash_identity,
)
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

__all__ = [
    "audit_mlx_scorer_input_cache_against_auth_eval",
    "audit_mlx_scorer_input_cache_against_local_cpu_advisory",
    "cache_audit_stamp_blockers",
    "write_cache_audit",
]

SCHEMA_VERSION = "mlx_scorer_input_cache_auth_eval_audit.v1"
LOCAL_CPU_ADVISORY_SCHEMA_VERSION = "mlx_scorer_input_cache_local_cpu_advisory_audit.v1"
PASS_VERDICT = "PASS_CACHE_AUTH_EVAL_IDENTITY"
FAIL_VERDICT = "FAIL_CACHE_AUTH_EVAL_IDENTITY"
PASS_LOCAL_CPU_ADVISORY_VERDICT = "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY"
FAIL_LOCAL_CPU_ADVISORY_VERDICT = "FAIL_CACHE_LOCAL_CPU_ADVISORY_IDENTITY"
AUTH_EVAL_AXIS_BY_GRADE = CONTEST_AUTH_AXIS_BY_EVIDENCE_GRADE
AUTHORITY_FALSE_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "promotable",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


def audit_mlx_scorer_input_cache_against_auth_eval(
    cache_manifest: dict[str, Any],
    auth_eval_payload: dict[str, Any],
    *,
    expected_pair_count: int | None = None,
    reference_cache_manifest: dict[str, Any] | None = None,
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
    auth_raw_sha = _first_raw_file_sha256(auth_eval_payload)
    auth_hash_manifest = _auth_scorer_input_manifest_payload(auth_eval_payload)
    hash_reference = auth_hash_manifest or reference_cache_manifest or {}
    hash_reference_source = (
        "auth_eval_provenance"
        if auth_hash_manifest
        else "reference_cache_manifest"
        if reference_cache_manifest
        else "missing"
    )
    auth_scorer_input_hashes = _hash_mapping(hash_reference.get("array_sha256"))
    auth_shapes = _shape_mapping(hash_reference)
    auth_hash_domain = _manifest_hash_domain(hash_reference)
    if auth_raw_sha is None:
        auth_raw_sha = _string(hash_reference.get("raw_sha256"))

    identity = scorer_input_cache_hash_identity(
        cache_archive_sha256=cache_archive_sha,
        auth_archive_sha256=auth_archive_sha,
        cache_inflated_outputs_aggregate_sha256=cache_inflated_sha,
        auth_inflated_outputs_aggregate_sha256=auth_inflated_sha,
        cache_raw_sha256=_string(cache_manifest.get("raw_sha256")),
        auth_raw_sha256=auth_raw_sha,
        cache_pair_count=cache_pair_count,
        auth_n_samples=auth_n_samples,
        cache_hash_domain=_manifest_hash_domain(cache_manifest),
        auth_hash_domain=auth_hash_domain,
        cache_array_sha256=_hash_mapping(cache_manifest.get("array_sha256")),
        auth_scorer_input_array_sha256=auth_scorer_input_hashes,
        cache_shapes=_shape_mapping(cache_manifest),
        auth_shapes=auth_shapes,
    )

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
    if auth_hash_manifest:
        _append_auth_scorer_input_manifest_blockers(
            blockers,
            auth_hash_manifest,
            auth_archive_sha=auth_archive_sha,
            auth_inflated_sha=auth_inflated_sha,
            auth_raw_sha=auth_raw_sha,
            auth_n_samples=auth_n_samples,
        )
    if reference_cache_manifest is not None:
        ref_archive_sha = _string(reference_cache_manifest.get("archive_sha256"))
        ref_inflated_sha = _string(
            reference_cache_manifest.get("inflated_outputs_aggregate_sha256")
        )
        ref_pair_count = _int(reference_cache_manifest.get("pair_count"))
        ref_raw_sha = _string(reference_cache_manifest.get("raw_sha256"))
        if ref_archive_sha and auth_archive_sha and ref_archive_sha != auth_archive_sha:
            blockers.append("reference_archive_sha256_mismatch_with_auth_eval")
        if ref_inflated_sha and auth_inflated_sha and ref_inflated_sha != auth_inflated_sha:
            blockers.append("reference_inflated_outputs_aggregate_sha256_mismatch_with_auth_eval")
        if ref_raw_sha and auth_raw_sha and ref_raw_sha != auth_raw_sha:
            blockers.append("reference_raw_sha256_mismatch_with_auth_eval")
        if ref_pair_count is not None and auth_n_samples is not None and ref_pair_count != auth_n_samples:
            blockers.append(
                f"reference_pair_count_mismatch:reference={ref_pair_count}:auth={auth_n_samples}"
            )
        if not auth_hash_manifest:
            _append_reference_cache_manifest_blockers(
                blockers,
                reference_cache_manifest,
                auth_archive_sha=auth_archive_sha,
                auth_inflated_sha=auth_inflated_sha,
                auth_raw_sha=auth_raw_sha,
                auth_n_samples=auth_n_samples,
            )
    for field in AUTHORITY_FALSE_FIELDS:
        if cache_manifest.get(field) is not False:
            blockers.append(f"cache_manifest_{field}_not_false")
    _append_auth_eval_authority_blockers(
        blockers,
        auth_eval_payload,
        metrics,
    )
    for blocker in identity["blockers"]:
        if blocker not in blockers:
            blockers.append(blocker)

    passed = not blockers
    transfer_eligible = bool(
        passed and identity.get("eligible_for_local_mlx_transfer_calibration")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "passed": passed,
        "blockers": blockers,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "eligible_for_local_mlx_transfer_calibration": transfer_eligible,
        "identity_residual": identity.get("identity_residual"),
        "cache": {
            "archive_sha256": cache_archive_sha,
            "inflated_outputs_aggregate_sha256": cache_inflated_sha,
            "raw_sha256": _string(cache_manifest.get("raw_sha256")),
            "pair_count": cache_pair_count,
            "segnet_last_rgb_shape": cache_manifest.get("segnet_last_rgb_shape"),
            "posenet_yuv6_pair_shape": cache_manifest.get("posenet_yuv6_pair_shape"),
            "pair_indices_shape": cache_manifest.get("pair_indices_shape"),
            "hash_domain": _manifest_hash_domain(cache_manifest),
            "artifacts": cache_manifest.get("artifacts"),
            "array_sha256": cache_manifest.get("array_sha256"),
        },
        "canonical_equation": identity,
        "auth_eval": {
            "archive_sha256": auth_archive_sha,
            "inflated_outputs_aggregate_sha256": auth_inflated_sha,
            "raw_file_sha256": auth_raw_sha,
            "n_samples": auth_n_samples,
            "score": metrics.get("score"),
            "pose_avg": metrics.get("pose_avg"),
            "seg_avg": metrics.get("seg_avg"),
            "rate": metrics.get("rate"),
            "evidence_grade": auth_eval_payload.get("evidence_grade"),
            "score_axis": auth_eval_payload.get("score_axis"),
            "lane_tag": auth_eval_payload.get("lane_tag"),
            "scorer_input_hash_reference_source": hash_reference_source,
            "scorer_input_hash_domain": auth_hash_domain,
            "scorer_input_array_sha256": auth_scorer_input_hashes,
            "scorer_input_shapes": auth_shapes,
        },
        "auth_eval_contract": {
            "evidence_grade": auth_eval_payload.get("evidence_grade"),
            "score_axis": auth_eval_payload.get("score_axis"),
            "accepted_grade_axis_pairs": dict(AUTH_EVAL_AXIS_BY_GRADE),
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


def cache_audit_stamp_blockers(
    cache_manifest: dict[str, Any],
    *,
    cache_root: str | Path | None,
    stamp_key: str,
    expected_verdict: str,
    require_identity_residual_zero: bool = True,
    require_cache_shapes: bool = True,
) -> list[str]:
    """Verify an embedded cache-audit stamp by loading its referenced audit JSON."""

    stamp = cache_manifest.get(stamp_key)
    if stamp is None:
        return [f"{stamp_key}_missing"]
    if not isinstance(stamp, dict):
        return [f"{stamp_key}_not_object"]
    blockers: list[str] = []
    if stamp.get("verdict") != expected_verdict:
        blockers.append(f"{stamp_key}_verdict_not_{expected_verdict}")
    if stamp.get("passed") is not True:
        blockers.append(f"{stamp_key}_passed_not_true")
    if require_identity_residual_zero and stamp.get("identity_residual") != 0:
        blockers.append(f"{stamp_key}_identity_residual_not_zero")
    _append_manifest_authority_blockers(blockers, stamp, prefix=stamp_key)

    audit_path = _stamp_path(cache_root=cache_root, stamp=stamp)
    expected_sha = _string(stamp.get("sha256"))
    if audit_path is None:
        blockers.append(f"{stamp_key}_path_missing")
    elif not audit_path.is_file():
        blockers.append(f"{stamp_key}_path_not_found")
    if expected_sha is None:
        blockers.append(f"{stamp_key}_sha256_missing")
    audit: dict[str, Any] | None = None
    if audit_path is not None and audit_path.is_file() and expected_sha is not None:
        actual_sha = _sha256_file(audit_path)
        if actual_sha != expected_sha:
            blockers.append(f"{stamp_key}_sha256_mismatch")
        else:
            try:
                payload = json.loads(audit_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                blockers.append(f"{stamp_key}_audit_json_invalid:{type(exc).__name__}")
            else:
                if not isinstance(payload, dict):
                    blockers.append(f"{stamp_key}_audit_json_not_object")
                else:
                    audit = payload
    if audit is None:
        return blockers
    if audit.get("verdict") != expected_verdict:
        blockers.append(f"{stamp_key}_audit_verdict_not_{expected_verdict}")
    if audit.get("passed") is not True:
        blockers.append(f"{stamp_key}_audit_passed_not_true")
    if require_identity_residual_zero and audit.get("identity_residual") != 0:
        blockers.append(f"{stamp_key}_audit_identity_residual_not_zero")
    _append_manifest_authority_blockers(blockers, audit, prefix=f"{stamp_key}_audit")
    blockers.extend(
        _audit_cache_identity_blockers(
            cache_manifest,
            audit.get("cache"),
            stamp_key=stamp_key,
            require_cache_shapes=require_cache_shapes,
        )
    )
    return blockers


def _stamp_path(*, cache_root: str | Path | None, stamp: dict[str, Any]) -> Path | None:
    raw = _string(stamp.get("path"))
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if cache_root is None:
        return path.resolve()
    return (Path(cache_root) / path).resolve()


def _audit_cache_identity_blockers(
    cache_manifest: dict[str, Any],
    audit_cache: Any,
    *,
    stamp_key: str,
    require_cache_shapes: bool,
) -> list[str]:
    if not isinstance(audit_cache, dict):
        return [f"{stamp_key}_audit_cache_missing"]
    blockers: list[str] = []
    comparisons: dict[str, Any] = {
        "archive_sha256": _string(cache_manifest.get("archive_sha256")),
        "inflated_outputs_aggregate_sha256": _string(
            cache_manifest.get("inflated_outputs_aggregate_sha256")
        ),
        "raw_sha256": _string(cache_manifest.get("raw_sha256")),
        "pair_count": _int(cache_manifest.get("pair_count")),
        "hash_domain": _manifest_hash_domain(cache_manifest),
        "array_sha256": _hash_mapping(cache_manifest.get("array_sha256")),
    }
    if require_cache_shapes:
        comparisons.update(
            {
                "segnet_last_rgb_shape": cache_manifest.get("segnet_last_rgb_shape"),
                "posenet_yuv6_pair_shape": cache_manifest.get("posenet_yuv6_pair_shape"),
                "pair_indices_shape": cache_manifest.get("pair_indices_shape"),
            }
        )
    for key, expected in comparisons.items():
        if audit_cache.get(key) != expected:
            blockers.append(f"{stamp_key}_audit_cache_{key}_mismatch")
    return blockers


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_mlx_scorer_input_cache_against_local_cpu_advisory(
    cache_manifest: dict[str, Any],
    advisory_payload: dict[str, Any],
    *,
    expected_pair_count: int | None = FULL_CONTEST_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Return pass/fail cache identity audit for a local CPU advisory payload.

    This is deliberately weaker than :func:`audit_mlx_scorer_input_cache_against_auth_eval`:
    it proves local raw/cache custody for MLX debugging, not contest-axis transfer
    calibration.  Downstream exact-eval spend triage must continue to require the
    strict auth-eval identity audit.
    """

    blockers: list[str] = []
    metrics = eval_metric_summary(advisory_payload)
    provenance = advisory_payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        blockers.append("local_advisory_provenance_missing")
    cache_archive_sha = _string(cache_manifest.get("archive_sha256"))
    advisory_archive_sha = _string(provenance.get("archive_sha256"))
    cache_inflated_sha = _string(cache_manifest.get("inflated_outputs_aggregate_sha256"))
    advisory_inflated_sha = _local_advisory_inflated_outputs_sha256(advisory_payload)
    cache_raw_sha = _string(cache_manifest.get("raw_sha256"))
    advisory_raw_sha = _local_advisory_raw_sha256(advisory_payload)
    cache_pair_count = _int(cache_manifest.get("pair_count"))
    advisory_n_samples = _int(metrics.get("n_samples"))
    expected = expected_pair_count if expected_pair_count is not None else advisory_n_samples

    if advisory_payload.get("score_axis") != "cpu_advisory":
        blockers.append("local_advisory_score_axis_not_cpu_advisory")
    if advisory_payload.get("score_claim") is not False:
        blockers.append("local_advisory_score_claim_not_false")
    for field in AUTHORITY_FALSE_FIELDS:
        if cache_manifest.get(field) is not False:
            blockers.append(f"cache_manifest_{field}_not_false")
    if not cache_archive_sha or cache_archive_sha != advisory_archive_sha:
        blockers.append("archive_sha256_mismatch_or_missing")
    if not cache_inflated_sha or cache_inflated_sha != advisory_inflated_sha:
        blockers.append("inflated_outputs_aggregate_sha256_mismatch_or_missing")
    if not cache_raw_sha or cache_raw_sha != advisory_raw_sha:
        blockers.append("raw_sha256_mismatch_or_missing")
    if cache_pair_count is None:
        blockers.append("cache_pair_count_missing")
    elif expected is not None and cache_pair_count != expected:
        blockers.append(f"cache_pair_count_mismatch:cache={cache_pair_count}:expected={expected}")
    if advisory_n_samples is None:
        blockers.append("local_advisory_n_samples_missing")
    elif expected is not None and advisory_n_samples != expected:
        blockers.append(
            f"local_advisory_n_samples_mismatch:advisory={advisory_n_samples}:expected={expected}"
        )
    _append_required_cache_manifest_blockers(blockers, cache_manifest)

    passed = not blockers
    return {
        "schema_version": LOCAL_CPU_ADVISORY_SCHEMA_VERSION,
        "verdict": PASS_LOCAL_CPU_ADVISORY_VERDICT if passed else FAIL_LOCAL_CPU_ADVISORY_VERDICT,
        "passed": passed,
        "blockers": blockers,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "eligible_for_local_mlx_transfer_calibration": False,
        "eligible_for_local_mlx_local_advisory_debug": passed,
        "cache": {
            "archive_sha256": cache_archive_sha,
            "inflated_outputs_aggregate_sha256": cache_inflated_sha,
            "raw_sha256": cache_raw_sha,
            "pair_count": cache_pair_count,
            "array_sha256": cache_manifest.get("array_sha256"),
            "hash_domain": _manifest_hash_domain(cache_manifest),
        },
        "local_cpu_advisory": {
            "archive_sha256": advisory_archive_sha,
            "inflated_outputs_aggregate_sha256": advisory_inflated_sha,
            "raw_file_sha256": advisory_raw_sha,
            "n_samples": advisory_n_samples,
            "score": metrics.get("score"),
            "pose_avg": metrics.get("pose_avg"),
            "seg_avg": metrics.get("seg_avg"),
            "score_axis": advisory_payload.get("score_axis"),
            "evidence_grade": advisory_payload.get("evidence_grade"),
            "platform_system": provenance.get("platform_system"),
            "platform_machine": provenance.get("platform_machine"),
        },
        "allowed_use": (
            [
                "local_mlx_debug_against_matching_local_cpu_advisory_raw",
                "local_speed_quality_delta_measurement",
            ]
            if passed
            else [
                "local_tensor_ingestion_debug_only",
                "do_not_use_for_local_mlx_cpu_advisory_comparison",
            ]
        ),
        "forbidden_use": [
            "auth_eval",
            "score_claim",
            "promotion",
            "rank_or_kill",
            "exact_eval_spend_triage_calibration",
            "replacement_for_contest_cpu_or_cuda_auth_eval",
        ],
        "authority_status": (
            "This audit proves local advisory raw/cache identity only. It is not a "
            "contest auth-axis cache identity and must not unlock score claims or "
            "exact-eval spend triage."
        ),
    }


def _append_auth_eval_authority_blockers(
    blockers: list[str],
    payload: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    for blocker in required_contest_auth_axis_payload_blockers(
        payload,
        metrics,
        expected_n_samples=FULL_CONTEST_SAMPLE_COUNT,
    ):
        if blocker not in blockers:
            blockers.append(blocker)


def _append_reference_cache_manifest_blockers(
    blockers: list[str],
    reference: dict[str, Any],
    *,
    auth_archive_sha: str | None,
    auth_inflated_sha: str | None,
    auth_raw_sha: str | None,
    auth_n_samples: int | None,
) -> None:
    required_strings = (
        "archive_sha256",
        "inflated_outputs_aggregate_sha256",
        "raw_sha256",
        "hash_domain",
    )
    for key in required_strings:
        if not _string(reference.get(key)):
            blockers.append(f"reference_{key}_missing")
    if _int(reference.get("pair_count")) is None:
        blockers.append("reference_pair_count_missing")
    for key in ("segnet_last_rgb_shape", "posenet_yuv6_pair_shape", "pair_indices_shape"):
        if not isinstance(reference.get(key), list) or not reference.get(key):
            blockers.append(f"reference_{key}_missing")
    hashes = _hash_mapping(reference.get("array_sha256"))
    for key in ("segnet_last_rgb", "posenet_yuv6_pair", "pair_indices"):
        if not _string(hashes.get(key)):
            blockers.append(f"reference_array_sha256_{key}_missing")
    if (
        _string(reference.get("archive_sha256"))
        and auth_archive_sha
        and _string(reference.get("archive_sha256")) != auth_archive_sha
    ):
        blockers.append("reference_archive_sha256_mismatch_with_auth_eval")
    if (
        _string(reference.get("inflated_outputs_aggregate_sha256"))
        and auth_inflated_sha
        and _string(reference.get("inflated_outputs_aggregate_sha256")) != auth_inflated_sha
    ):
        blockers.append("reference_inflated_outputs_aggregate_sha256_mismatch_with_auth_eval")
    if (
        _string(reference.get("raw_sha256"))
        and auth_raw_sha
        and _string(reference.get("raw_sha256")) != auth_raw_sha
    ):
        blockers.append("reference_raw_sha256_mismatch_with_auth_eval")
    pair_count = _int(reference.get("pair_count"))
    if pair_count is not None and auth_n_samples is not None and pair_count != auth_n_samples:
        blockers.append(
            f"reference_pair_count_mismatch:reference={pair_count}:auth={auth_n_samples}"
        )


def _append_auth_scorer_input_manifest_blockers(
    blockers: list[str],
    manifest: dict[str, Any],
    *,
    auth_archive_sha: str | None,
    auth_inflated_sha: str | None,
    auth_raw_sha: str | None,
    auth_n_samples: int | None,
) -> None:
    schema = _string(manifest.get("schema_version"))
    require_full_cache_identity = schema == "mlx_scorer_input_cache.v1"
    manifest_archive_sha = _string(manifest.get("archive_sha256"))
    manifest_inflated_sha = _string(manifest.get("inflated_outputs_aggregate_sha256"))
    manifest_raw_sha = _string(manifest.get("raw_sha256"))
    manifest_pair_count = _int(manifest.get("pair_count"))

    _append_manifest_authority_blockers(
        blockers,
        manifest,
        prefix="auth_scorer_input_manifest",
    )

    if require_full_cache_identity and not manifest_archive_sha:
        blockers.append("auth_scorer_input_manifest_archive_sha256_missing")
    if manifest_archive_sha and auth_archive_sha and manifest_archive_sha != auth_archive_sha:
        blockers.append("auth_scorer_input_manifest_archive_sha256_mismatch_with_auth_eval")

    if require_full_cache_identity and not manifest_inflated_sha:
        blockers.append("auth_scorer_input_manifest_inflated_outputs_aggregate_sha256_missing")
    if (
        manifest_inflated_sha
        and auth_inflated_sha
        and manifest_inflated_sha != auth_inflated_sha
    ):
        blockers.append(
            "auth_scorer_input_manifest_inflated_outputs_aggregate_sha256_mismatch_with_auth_eval"
        )

    if require_full_cache_identity and not manifest_raw_sha:
        blockers.append("auth_scorer_input_manifest_raw_sha256_missing")
    if manifest_raw_sha and auth_raw_sha and manifest_raw_sha != auth_raw_sha:
        blockers.append("auth_scorer_input_manifest_raw_sha256_mismatch_with_auth_eval")

    if require_full_cache_identity and manifest_pair_count is None:
        blockers.append("auth_scorer_input_manifest_pair_count_missing")
    if (
        manifest_pair_count is not None
        and auth_n_samples is not None
        and manifest_pair_count != auth_n_samples
    ):
        blockers.append(
            "auth_scorer_input_manifest_pair_count_mismatch:"
            f"manifest={manifest_pair_count}:auth={auth_n_samples}"
        )


def _append_manifest_authority_blockers(
    blockers: list[str],
    manifest: dict[str, Any],
    *,
    prefix: str,
) -> None:
    for field in AUTHORITY_FALSE_FIELDS:
        if field in manifest and manifest.get(field) is not False:
            blockers.append(f"{prefix}_{field}_not_false")


def _append_required_cache_manifest_blockers(
    blockers: list[str],
    manifest: dict[str, Any],
) -> None:
    if _manifest_hash_domain(manifest) is None:
        blockers.append("cache_manifest_hash_domain_missing")
    hashes = _hash_mapping(manifest.get("array_sha256"))
    for key in ("segnet_last_rgb", "posenet_yuv6_pair", "pair_indices"):
        if key not in hashes:
            blockers.append(f"cache_manifest_array_sha256_{key}_missing")
    shapes = _shape_mapping(manifest)
    for key in ("segnet_last_rgb", "posenet_yuv6_pair", "pair_indices"):
        if key not in shapes:
            blockers.append(f"cache_manifest_{key}_shape_missing")


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


def _local_advisory_inflated_outputs_sha256(payload: dict[str, Any]) -> str | None:
    return _inflated_outputs_aggregate_sha256(payload)


def _local_advisory_raw_sha256(payload: dict[str, Any]) -> str | None:
    return _first_raw_file_sha256(payload)


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


def _auth_scorer_input_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return {}
    for key in ("scorer_input_cache_hash_manifest", "scorer_input_cache_tensor_manifest"):
        manifest = provenance.get(key)
        if not isinstance(manifest, dict):
            continue
        payload_obj = manifest.get("payload")
        if isinstance(payload_obj, dict):
            return payload_obj
    return {}


def _manifest_hash_domain(payload: dict[str, Any]) -> str | None:
    value = _string(payload.get("hash_domain"))
    if value:
        return value
    if payload.get("schema_version") == "mlx_scorer_input_cache.v1":
        return "_array_sha256(dtype_string + json_shape + contiguous_bytes)"
    return None


def _shape_mapping(payload: dict[str, Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for name in ("segnet_last_rgb", "posenet_yuv6_pair", "pair_indices"):
        shape = payload.get(f"{name}_shape")
        if not isinstance(shape, list):
            continue
        dims: list[int] = []
        valid = True
        for dim in shape:
            if isinstance(dim, bool) or not isinstance(dim, int):
                valid = False
                break
            dims.append(int(dim))
        if valid:
            out[name] = dims
    return out


def _hash_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        if isinstance(key, str) and isinstance(raw, str) and raw.strip():
            out[key] = raw.strip().lower()
    return out


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
