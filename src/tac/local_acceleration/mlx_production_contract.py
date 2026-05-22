# SPDX-License-Identifier: MIT
"""Production contract checks for local MLX scorer-response artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_response import (
    GPU_BATCH_SHAPE_BLOCKER,
    GPU_RESEARCH_SIGNAL_BLOCKER,
)

SCHEMA_VERSION = "mlx_scorer_production_contract.v1"
PASS_VERDICT = "PASS_MLX_SCORER_PRODUCTION_CONTRACT"
FAIL_VERDICT = "FAIL_MLX_SCORER_PRODUCTION_CONTRACT"

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
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def write_production_contract_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mlx_scorer_production_contract_manifest(
    response_payload: dict[str, Any],
    *,
    profile_stability: dict[str, Any] | None = None,
    batch_invariance: dict[str, Any] | None = None,
    run_id: str | None = None,
    require_cache_identity: bool = True,
    require_profile_stability: bool = True,
    require_batch_invariance: bool = True,
) -> dict[str, Any]:
    """Validate an MLX scorer-response artifact as a production local signal.

    Passing this contract means the artifact is suitable for production use as
    local MLX acceleration, profiling, or candidate-generation signal. It does
    not make the artifact a contest-authoritative score.
    """

    blockers: list[str] = []
    warnings: list[str] = []

    _check_response_payload(
        response_payload,
        blockers=blockers,
        warnings=warnings,
        require_cache_identity=require_cache_identity,
    )
    if profile_stability is None:
        missing = "profile_stability_manifest_not_supplied"
        if require_profile_stability:
            blockers.append(missing)
        else:
            warnings.append(missing)
    else:
        _check_optional_gate_manifest(
            profile_stability,
            blockers=blockers,
            warnings=warnings,
            schema_prefix="mlx_scorer_response_profile_stability",
            label="profile_stability",
        )
    if batch_invariance is None:
        missing = "batch_invariance_manifest_not_supplied"
        if require_batch_invariance:
            blockers.append(missing)
        else:
            warnings.append(missing)
    else:
        _check_batch_invariance_gate_manifest(
            batch_invariance,
            blockers=blockers,
            warnings=warnings,
            response_device=_response_device(response_payload),
            response_batch_pairs=_safe_int(response_payload.get("batch_pairs")),
        )

    passed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "passed": passed,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "blockers": blockers,
        "warnings": warnings,
        "production_deployment_role": "local_mlx_scorer_acceleration_non_authoritative",
        "score_authority": False,
        "contest_authority": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "response_summary": {
            "schema_version": response_payload.get("schema_version"),
            "hardware_substrate": response_payload.get("hardware_substrate"),
            "batch_pairs": response_payload.get("batch_pairs"),
            "n_samples": response_payload.get("n_samples"),
            "pair_window": response_payload.get("pair_window"),
            "archive_sha256": response_payload.get("archive_sha256"),
            "inflated_outputs_aggregate_sha256": response_payload.get(
                "inflated_outputs_aggregate_sha256"
            ),
        },
        "required_gates": {
            "cache_identity": bool(require_cache_identity),
            "profile_stability": bool(require_profile_stability),
            "batch_invariance": bool(require_batch_invariance),
        },
        "authority_status": (
            "Contract pass means production-safe local MLX scorer acceleration "
            "signal only; paired contest CPU/CUDA auth eval remains required "
            "for score claims, promotion, rank/kill, and dispatch readiness."
        ),
    }


def _check_response_payload(
    payload: dict[str, Any],
    *,
    blockers: list[str],
    warnings: list[str],
    require_cache_identity: bool,
) -> None:
    if not isinstance(payload, dict):
        blockers.append("response_payload_not_object")
        return
    if payload.get("schema_version") != "mlx_scorer_response.v1":
        blockers.append("response_schema_version_not_mlx_scorer_response_v1")
    if payload.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        blockers.append(f"response_evidence_grade_not_{EVIDENCE_GRADE_MLX}")
    if payload.get("evidence_tag") != EVIDENCE_TAG_MLX:
        blockers.append("response_evidence_tag_not_mlx_research_signal")
    for field in AUTHORITY_FALSE_FIELDS:
        if payload.get(field) is not False:
            blockers.append(f"response_{field}_not_false")
    if payload.get("candidate_generation_only") is not True:
        blockers.append("response_candidate_generation_only_not_true")
    if payload.get("requires_exact_eval_before_promotion") is not True:
        blockers.append("response_requires_exact_eval_before_promotion_not_true")
    if payload.get("score_axis") != EVIDENCE_TAG_MLX:
        blockers.append("response_score_axis_not_mlx_research_signal")

    device = _response_device(payload)
    batch_pairs = _int_field(payload, "batch_pairs", blockers)
    if device == "gpu":
        if payload.get("gpu_research_signal_allowed") is not True:
            blockers.append(GPU_RESEARCH_SIGNAL_BLOCKER)
        if batch_pairs != 1:
            blockers.append(GPU_BATCH_SHAPE_BLOCKER)
    elif device not in {"cpu", "gpu"}:
        blockers.append(f"response_unknown_mlx_device:{device!r}")

    n_samples = _int_field(payload, "n_samples", blockers)
    pair_window = payload.get("pair_window")
    if isinstance(pair_window, list) and len(pair_window) == 2:
        try:
            if n_samples is not None and int(pair_window[1]) - int(pair_window[0]) != n_samples:
                blockers.append("response_pair_window_n_samples_mismatch")
        except (TypeError, ValueError):
            blockers.append("response_pair_window_non_integer")
    else:
        blockers.append("response_pair_window_invalid")

    for key in (
        "canonical_score",
        "score_recomputed_from_components",
        "avg_posenet_dist",
        "avg_segnet_dist",
    ):
        _finite_float_field(payload, key, blockers)
    if payload.get("canonical_score_source") != "score_recomputed_from_components":
        blockers.append("response_canonical_score_source_not_recomputed_from_components")
    try:
        if not math.isclose(
            float(payload.get("canonical_score")),
            float(payload.get("score_recomputed_from_components")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            blockers.append("response_canonical_score_recompute_mismatch")
    except (TypeError, ValueError):
        pass

    components = payload.get("components")
    if not isinstance(components, dict):
        blockers.append("response_components_missing")
    else:
        for key in ("posenet_sha256", "segnet_sha256"):
            if not _is_sha256(str(components.get(key, ""))):
                blockers.append(f"response_components_{key}_invalid")
        _check_component_shape(components, "posenet_shape", n_samples, blockers)
        _check_component_shape(components, "segnet_shape", n_samples, blockers)

    contract = payload.get("device_contract")
    if not isinstance(contract, dict):
        blockers.append("response_device_contract_missing")
    else:
        forbidden = contract.get("forbidden_uses")
        if not isinstance(forbidden, list) or "score_claim" not in forbidden:
            blockers.append("response_device_contract_missing_score_claim_forbidden_use")
        if device == "gpu":
            if contract.get("gpu_research_signal_blocker") != GPU_RESEARCH_SIGNAL_BLOCKER:
                blockers.append("response_device_contract_missing_gpu_research_blocker")
            if contract.get("gpu_batch_shape_blocker") != GPU_BATCH_SHAPE_BLOCKER:
                blockers.append("response_device_contract_missing_gpu_batch_shape_blocker")

    if require_cache_identity:
        _check_cache_identity(payload, blockers=blockers)
    elif "cache_identity" not in payload:
        warnings.append("response_cache_identity_not_checked")


def _check_cache_identity(payload: dict[str, Any], *, blockers: list[str]) -> None:
    identity = payload.get("cache_identity")
    if not isinstance(identity, dict):
        blockers.append("response_cache_identity_missing")
        return
    if identity.get("pair_indices_equal") is not True:
        blockers.append("response_cache_pair_indices_not_equal")
    for side in ("reference", "candidate"):
        item = identity.get(side)
        if not isinstance(item, dict):
            blockers.append(f"response_cache_identity_{side}_missing")
            continue
        archive_hashes_valid = all(
            _is_sha256(str(item.get(key, "")))
            for key in ("archive_sha256", "inflated_outputs_aggregate_sha256", "raw_sha256")
        )
        array_hashes = item.get("array_sha256")
        array_hashes_valid = isinstance(array_hashes, dict) and all(
            _is_sha256(str(array_hashes.get(key, "")))
            for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb")
        )
        if side == "candidate" and not archive_hashes_valid:
            blockers.append(f"response_cache_identity_{side}_archive_raw_hashes_invalid")
        if side == "reference" and not archive_hashes_valid and not array_hashes_valid:
            blockers.append(f"response_cache_identity_{side}_hash_identity_invalid")
        for key in ("archive_sha256", "inflated_outputs_aggregate_sha256", "raw_sha256"):
            if item.get(key) is None:
                continue
            if not _is_sha256(str(item.get(key, ""))):
                blockers.append(f"response_cache_identity_{side}_{key}_invalid")
        if isinstance(array_hashes, dict):
            for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
                if not _is_sha256(str(array_hashes.get(key, ""))):
                    blockers.append(f"response_cache_identity_{side}_array_sha256_{key}_invalid")


def _check_optional_gate_manifest(
    manifest: dict[str, Any],
    *,
    blockers: list[str],
    warnings: list[str],
    schema_prefix: str,
    label: str,
) -> None:
    if not isinstance(manifest, dict):
        blockers.append(f"{label}_manifest_not_object")
        return
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.startswith(schema_prefix):
        blockers.append(f"{label}_schema_version_invalid")
    for field in AUTHORITY_FALSE_FIELDS:
        if manifest.get(field) is not False:
            blockers.append(f"{label}_{field}_not_false")
    if manifest.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        blockers.append(f"{label}_evidence_grade_not_{EVIDENCE_GRADE_MLX}")
    if manifest.get("evidence_tag") not in {None, EVIDENCE_TAG_MLX}:
        blockers.append(f"{label}_evidence_tag_not_mlx_research_signal")
    if manifest.get("candidate_generation_only") not in {None, True}:
        blockers.append(f"{label}_candidate_generation_only_not_true")
    if manifest.get("passed") is not True:
        blockers.append(f"{label}_not_passing")


def _check_batch_invariance_gate_manifest(
    manifest: dict[str, Any],
    *,
    blockers: list[str],
    warnings: list[str],
    response_device: str,
    response_batch_pairs: int | None,
) -> None:
    _check_optional_gate_manifest(
        manifest,
        blockers=blockers,
        warnings=warnings,
        schema_prefix="mlx_scorer_batch_invariance",
        label="batch_invariance",
    )
    if not isinstance(manifest, dict):
        return
    if response_batch_pairs is None or response_batch_pairs <= 1:
        return

    gate_device = str(manifest.get("device_type") or "").lower()
    if not gate_device:
        blockers.append("batch_invariance_device_type_missing")
    elif gate_device != response_device:
        blockers.append(
            "batch_invariance_device_type_mismatch:"
            f"response={response_device}:gate={gate_device}"
        )

    gate_batch_pairs = _safe_int(manifest.get("batch_pairs"))
    if gate_batch_pairs is None:
        blockers.append("batch_invariance_batch_pairs_invalid")
    elif gate_batch_pairs != response_batch_pairs:
        blockers.append(
            "batch_invariance_batch_pairs_mismatch:"
            f"response={response_batch_pairs}:gate={gate_batch_pairs}"
        )


def _response_device(payload: dict[str, Any]) -> str:
    hardware = str(payload.get("hardware_substrate") or "")
    parts = hardware.strip().split()
    if len(parts) == 2 and parts[0].lower() == "mlx":
        return parts[1].lower()
    return str(payload.get("device_type") or "").lower()


def _int_field(payload: dict[str, Any], key: str, blockers: list[str]) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool):
        blockers.append(f"response_{key}_invalid_bool")
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        blockers.append(f"response_{key}_invalid")
        return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _check_component_shape(
    components: dict[str, Any],
    key: str,
    n_samples: int | None,
    blockers: list[str],
) -> None:
    shape = components.get(key)
    if not isinstance(shape, list) or not shape:
        blockers.append(f"response_components_{key}_invalid")
        return
    try:
        first_dim = int(shape[0])
    except (TypeError, ValueError):
        blockers.append(f"response_components_{key}_invalid")
        return
    if n_samples is not None and first_dim != n_samples:
        blockers.append(f"response_components_{key}_n_samples_mismatch")


def _finite_float_field(payload: dict[str, Any], key: str, blockers: list[str]) -> None:
    try:
        value = float(payload.get(key))
    except (TypeError, ValueError):
        blockers.append(f"response_{key}_invalid")
        return
    if not math.isfinite(value):
        blockers.append(f"response_{key}_non_finite")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


__all__ = [
    "FAIL_VERDICT",
    "PASS_VERDICT",
    "SCHEMA_VERSION",
    "build_mlx_scorer_production_contract_manifest",
    "load_json_object",
    "write_production_contract_manifest",
]
