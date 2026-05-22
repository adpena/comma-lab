# SPDX-License-Identifier: MIT
"""Production contract checks for local MLX scorer-response artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_cache_audit import PASS_VERDICT as CACHE_AUDIT_PASS_VERDICT
from tac.local_acceleration.mlx_score_calibration import (
    SCHEMA_VERSION as SCORE_CALIBRATION_SCHEMA_VERSION,
)
from tac.local_acceleration.mlx_score_calibration import STRICT_AUTH_AXIS_SPEND_TRIAGE_ALLOWED_USE
from tac.local_acceleration.mlx_scorer_response import (
    GPU_BATCH_SHAPE_BLOCKER,
    GPU_RESEARCH_SIGNAL_BLOCKER,
)
from tac.local_acceleration.mlx_scorer_torch_parity import (
    PASS_SWEEP_VERDICT as TORCH_PARITY_PASS_SWEEP_VERDICT,
)
from tac.local_acceleration.mlx_scorer_torch_parity import (
    PASS_VERDICT as TORCH_PARITY_PASS_VERDICT,
)
from tac.local_acceleration.mlx_scorer_torch_parity import (
    SCHEMA_VERSION as TORCH_PARITY_SCHEMA_VERSION,
)
from tac.local_acceleration.mlx_scorer_torch_parity import (
    SWEEP_SCHEMA_VERSION as TORCH_PARITY_SWEEP_SCHEMA_VERSION,
)

SCHEMA_VERSION = "mlx_scorer_production_contract.v2"
GATE_SET_VERSION = "mlx_scorer_production_gate_set.v2.cache_auth_torch_profile"
PASS_VERDICT = "PASS_MLX_SCORER_PRODUCTION_CONTRACT"
FAIL_VERDICT = "FAIL_MLX_SCORER_PRODUCTION_CONTRACT"
ADVISORY_VERDICT = "ADVISORY_MLX_SCORER_DEV_CONTRACT"
MAX_ALLOWED_TORCH_PARITY_ARGMAX_DIFF_PIXELS = 1

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
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def write_production_contract_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mlx_scorer_production_contract_manifest(
    response_payload: dict[str, Any],
    *,
    cache_auth_audit: dict[str, Any] | None = None,
    torch_parity: dict[str, Any] | None = None,
    reference_torch_parity: dict[str, Any] | None = None,
    profile_stability: dict[str, Any] | None = None,
    batch_invariance: dict[str, Any] | None = None,
    score_calibration: dict[str, Any] | None = None,
    run_id: str | None = None,
    require_cache_identity: bool = True,
    require_cache_auth_audit: bool = True,
    require_torch_parity: bool = True,
    require_profile_stability: bool = True,
    require_batch_invariance: bool = True,
    require_score_calibration: bool = False,
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
    response_batch_pairs = _safe_int(response_payload.get("batch_pairs"))
    if response_batch_pairs is not None and response_batch_pairs != 1:
        blockers.append("response_non_singleton_batch_pairs_not_production_supported")
    batch_invariance_required = bool(
        require_batch_invariance
        and (response_batch_pairs is None or response_batch_pairs > 1)
    )
    if cache_auth_audit is None:
        missing = "cache_auth_audit_manifest_not_supplied"
        if require_cache_auth_audit:
            blockers.append(missing)
        else:
            warnings.append(missing)
    else:
        _check_cache_auth_audit_manifest(
            cache_auth_audit,
            response_payload=response_payload,
            blockers=blockers,
        )
    if torch_parity is None:
        missing = "torch_parity_manifest_not_supplied"
        if require_torch_parity:
            blockers.append(missing)
        else:
            warnings.append(missing)
    else:
        _check_torch_parity_manifest(
            torch_parity,
            response_payload=response_payload,
            cache_side="candidate",
            blockers=blockers,
        )
    if reference_torch_parity is None:
        missing = "reference_torch_parity_manifest_not_supplied"
        if require_torch_parity:
            blockers.append(missing)
        else:
            warnings.append(missing)
    else:
        _check_torch_parity_manifest(
            reference_torch_parity,
            response_payload=response_payload,
            cache_side="reference",
            blockers=blockers,
        )
    if profile_stability is None:
        missing = "profile_stability_manifest_not_supplied"
        if require_profile_stability:
            blockers.append(missing)
        else:
            warnings.append(missing)
    else:
        _check_profile_stability_manifest(
            profile_stability,
            response_payload=response_payload,
            blockers=blockers,
            warnings=warnings,
        )
    if batch_invariance is None:
        missing = "batch_invariance_manifest_not_supplied"
        if batch_invariance_required:
            blockers.append(missing)
        elif require_batch_invariance:
            warnings.append("batch_invariance_not_required_for_singleton_response")
        else:
            warnings.append(missing)
    else:
        _check_batch_invariance_gate_manifest(
            batch_invariance,
            response_payload=response_payload,
            blockers=blockers,
            warnings=warnings,
            response_device=_response_device(response_payload),
            response_batch_pairs=response_batch_pairs,
            required=batch_invariance_required,
        )
    if score_calibration is None:
        missing = "score_calibration_manifest_not_supplied"
        if require_score_calibration:
            blockers.append(missing)
    else:
        _check_score_calibration_gate_manifest(
            score_calibration,
            response_payload=response_payload,
            blockers=blockers,
        )

    strict_gate_policy = bool(
        require_cache_identity
        and require_cache_auth_audit
        and require_torch_parity
        and require_profile_stability
        and require_batch_invariance
    )
    advisory_only = not strict_gate_policy
    production_passed = not blockers and not advisory_only
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_set_version": GATE_SET_VERSION,
        "run_id": run_id,
        "passed": production_passed,
        "advisory_passed": bool(not blockers),
        "verdict": (
            PASS_VERDICT
            if production_passed
            else ADVISORY_VERDICT
            if not blockers
            else FAIL_VERDICT
        ),
        "blockers": blockers,
        "warnings": (
            [*warnings, "production_required_gate_policy_bypassed"]
            if advisory_only
            else warnings
        ),
        "production_deployment_role": "local_mlx_scorer_acceleration_non_authoritative",
        "score_authority": False,
        "contest_authority": False,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "score_axis": EVIDENCE_TAG_MLX,
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
            "cache_auth_audit": bool(require_cache_auth_audit),
            "torch_parity": bool(require_torch_parity),
            "reference_torch_parity": bool(require_torch_parity),
            "profile_stability": bool(require_profile_stability),
            "batch_invariance": batch_invariance_required,
            "batch_invariance_policy_requested": bool(require_batch_invariance),
            "score_calibration": bool(require_score_calibration),
            "strict_gate_policy": strict_gate_policy,
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
        audit = item.get("auth_eval_identity_audit")
        if side == "candidate":
            if not isinstance(audit, dict):
                blockers.append("response_cache_identity_candidate_audit_missing")
            else:
                for field in AUTHORITY_FALSE_FIELDS:
                    if audit.get(field) is not False:
                        blockers.append(
                            f"response_cache_identity_candidate_audit_{field}_not_false"
                        )


def _check_optional_gate_manifest(
    manifest: dict[str, Any],
    *,
    blockers: list[str],
    warnings: list[str],
    schema_prefix: str,
    label: str,
    require_passed: bool = True,
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
    if require_passed and manifest.get("passed") is not True:
        blockers.append(f"{label}_not_passing")


def _check_cache_auth_audit_manifest(
    manifest: dict[str, Any],
    *,
    response_payload: dict[str, Any],
    blockers: list[str],
) -> None:
    if not isinstance(manifest, dict):
        blockers.append("cache_auth_audit_manifest_not_object")
        return
    if manifest.get("schema_version") != "mlx_scorer_input_cache_auth_eval_audit.v1":
        blockers.append("cache_auth_audit_schema_version_invalid")
    if manifest.get("passed") is not True:
        blockers.append("cache_auth_audit_not_passing")
    if manifest.get("verdict") != CACHE_AUDIT_PASS_VERDICT:
        blockers.append("cache_auth_audit_verdict_not_pass")
    audit_blockers = manifest.get("blockers")
    if isinstance(audit_blockers, list):
        for blocker in audit_blockers:
            if blocker:
                blockers.append(f"cache_auth_audit_blocker:{blocker}")
    for field in AUTHORITY_FALSE_FIELDS:
        if manifest.get(field) is not False:
            blockers.append(f"cache_auth_audit_{field}_not_false")
    allowed_use = manifest.get("allowed_use")
    if (
        not isinstance(allowed_use, list)
        or "local_mlx_training_transfer_calibration" not in allowed_use
    ):
        blockers.append("cache_auth_audit_missing_transfer_calibration_allowed_use")
    canonical_equation = manifest.get("canonical_equation")
    if not isinstance(canonical_equation, dict):
        blockers.append("cache_auth_audit_canonical_equation_missing")
    else:
        if canonical_equation.get("eligible_for_local_mlx_transfer_calibration") is not True:
            blockers.append("cache_auth_audit_canonical_equation_not_transfer_eligible")
        if canonical_equation.get("identity_residual") != 0:
            blockers.append("cache_auth_audit_canonical_equation_identity_residual_nonzero")
        if canonical_equation.get("score_claim") is not False:
            blockers.append("cache_auth_audit_canonical_equation_score_claim_not_false")
        for key in ("hash_domain", "compared_scorer_input_hashes", "compared_scorer_input_shapes"):
            if not canonical_equation.get(key):
                blockers.append(f"cache_auth_audit_canonical_equation_{key}_missing")
    cache = manifest.get("cache")
    if not isinstance(cache, dict):
        blockers.append("cache_auth_audit_cache_identity_missing")
        return
    _require_hash_match(
        response_payload.get("archive_sha256"),
        cache.get("archive_sha256"),
        "cache_auth_audit_archive_sha256_mismatch",
        blockers,
    )
    _require_hash_match(
        response_payload.get("inflated_outputs_aggregate_sha256"),
        cache.get("inflated_outputs_aggregate_sha256"),
        "cache_auth_audit_inflated_outputs_aggregate_sha256_mismatch",
        blockers,
    )
    _check_int_match(
        _response_cache_side(response_payload, "candidate").get("pair_count"),
        cache.get("pair_count"),
        "cache_auth_audit_pair_count_mismatch",
        blockers,
    )
    if not _hash_mapping(cache.get("array_sha256")):
        blockers.append("cache_auth_audit_array_sha256_missing")
    if not _shape_mapping(cache):
        blockers.append("cache_auth_audit_shapes_missing")
    if not cache.get("hash_domain"):
        blockers.append("cache_auth_audit_hash_domain_missing")
    auth_eval = manifest.get("auth_eval")
    if not isinstance(auth_eval, dict):
        blockers.append("cache_auth_audit_auth_eval_summary_missing")
    else:
        if _safe_int(auth_eval.get("n_samples")) is None:
            blockers.append("cache_auth_audit_auth_eval_n_samples_missing")
        if not auth_eval.get("scorer_input_hash_domain"):
            blockers.append("cache_auth_audit_auth_eval_hash_domain_missing")
        if not _hash_mapping(auth_eval.get("scorer_input_array_sha256")):
            blockers.append("cache_auth_audit_auth_eval_array_sha256_missing")
        if not _shape_mapping_from_auth_summary(auth_eval.get("scorer_input_shapes")):
            blockers.append("cache_auth_audit_auth_eval_shapes_missing")
    candidate = _response_cache_side(response_payload, "candidate")
    if candidate:
        _require_hash_match(
            candidate.get("archive_sha256"),
            cache.get("archive_sha256"),
            "cache_auth_audit_candidate_archive_sha256_mismatch",
            blockers,
        )
        _require_hash_match(
            candidate.get("inflated_outputs_aggregate_sha256"),
            cache.get("inflated_outputs_aggregate_sha256"),
            "cache_auth_audit_candidate_inflated_outputs_aggregate_sha256_mismatch",
            blockers,
        )
        _require_hash_match(
            candidate.get("raw_sha256"),
            cache.get("raw_sha256"),
            "cache_auth_audit_candidate_raw_sha256_mismatch",
            blockers,
        )


def _check_torch_parity_manifest(
    manifest: dict[str, Any],
    *,
    response_payload: dict[str, Any],
    cache_side: str = "candidate",
    blockers: list[str],
) -> None:
    if cache_side not in {"candidate", "reference"}:
        blockers.append(f"torch_parity_cache_side_invalid:{cache_side!r}")
        return
    if not isinstance(manifest, dict):
        blockers.append("torch_parity_manifest_not_object")
        return
    schema_version = manifest.get("schema_version")
    if schema_version not in {TORCH_PARITY_SCHEMA_VERSION, TORCH_PARITY_SWEEP_SCHEMA_VERSION}:
        blockers.append("torch_parity_schema_version_invalid")
    if manifest.get("passed") is not True:
        blockers.append("torch_parity_not_passing")
    if schema_version == TORCH_PARITY_SCHEMA_VERSION:
        if manifest.get("verdict") != TORCH_PARITY_PASS_VERDICT:
            blockers.append("torch_parity_verdict_not_pass")
    elif schema_version == TORCH_PARITY_SWEEP_SCHEMA_VERSION:
        if manifest.get("verdict") != TORCH_PARITY_PASS_SWEEP_VERDICT:
            blockers.append("torch_parity_verdict_not_pass")
        summary = manifest.get("summary")
        if isinstance(summary, dict) and summary.get("failed_windows") not in {None, 0}:
            blockers.append("torch_parity_failed_windows_nonzero")
    for field in AUTHORITY_FALSE_FIELDS:
        if manifest.get(field) is not False:
            blockers.append(f"torch_parity_{field}_not_false")
    if manifest.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        blockers.append(f"torch_parity_evidence_grade_not_{EVIDENCE_GRADE_MLX}")
    if manifest.get("evidence_tag") != EVIDENCE_TAG_MLX:
        blockers.append("torch_parity_evidence_tag_not_mlx_research_signal")
    if manifest.get("score_axis") != EVIDENCE_TAG_MLX:
        blockers.append("torch_parity_score_axis_not_mlx_research_signal")
    _check_torch_parity_thresholds(manifest, blockers=blockers)
    cache_identity = manifest.get("cache_identity")
    if not isinstance(cache_identity, dict):
        blockers.append("torch_parity_cache_identity_missing")
        return
    parity_archive = cache_identity.get("archive_sha256")
    expected = _response_cache_side(response_payload, cache_side)
    archive_required = cache_side == "candidate" or expected.get("archive_sha256") is not None
    if archive_required and not _is_sha256(str(parity_archive)):
        blockers.append("torch_parity_cache_identity_archive_sha256_invalid")
    expected_archive = expected.get("archive_sha256")
    if (
        expected_archive is not None
        and parity_archive is not None
        and (
            not _is_sha256(str(parity_archive))
            or str(parity_archive) != str(expected_archive)
        )
    ):
        blockers.append("torch_parity_cache_identity_archive_sha256_mismatch")
    _check_cache_path_match(
        expected.get("path"),
        cache_identity.get("path") or manifest.get("cache_dir"),
        "torch_parity_cache_dir_mismatch",
        blockers,
    )
    _check_int_match(
        expected.get("pair_count"),
        cache_identity.get("pair_count") or manifest.get("total_pair_count"),
        "torch_parity_pair_count_mismatch",
        blockers,
    )
    if cache_side == "candidate":
        _require_hash_match(
            expected.get("inflated_outputs_aggregate_sha256"),
            cache_identity.get("inflated_outputs_aggregate_sha256"),
            "torch_parity_inflated_outputs_aggregate_sha256_mismatch",
            blockers,
        )
        _require_hash_match(
            expected.get("raw_sha256"),
            cache_identity.get("raw_sha256"),
            "torch_parity_raw_sha256_mismatch",
            blockers,
        )
    else:
        _check_hash_match(
            expected.get("inflated_outputs_aggregate_sha256"),
            cache_identity.get("inflated_outputs_aggregate_sha256"),
            "torch_parity_inflated_outputs_aggregate_sha256_mismatch",
            blockers,
        )
        _check_hash_match(
            expected.get("raw_sha256"),
            cache_identity.get("raw_sha256"),
            "torch_parity_raw_sha256_mismatch",
            blockers,
        )
    response_window = _pair_window(response_payload)
    covered_window = _pair_window({"pair_window": manifest.get("covered_pair_window")})
    response_batch_pairs = _safe_int(response_payload.get("batch_pairs"))
    if response_window is not None:
        if covered_window is None:
            blockers.append("torch_parity_covered_pair_window_missing")
        elif not (
            covered_window[0] <= response_window[0]
            and response_window[1] <= covered_window[1]
        ):
            blockers.append(
                "torch_parity_window_does_not_cover_response:"
                f"response={response_window}:covered={covered_window}"
            )
    _check_torch_parity_batch_shape(
        manifest,
        schema_version=schema_version,
        response_window=response_window,
        response_batch_pairs=response_batch_pairs,
        blockers=blockers,
    )
    if not cache_identity.get("hash_domain"):
        blockers.append("torch_parity_cache_identity_hash_domain_missing")
    if not _hash_mapping(cache_identity.get("array_sha256")):
        blockers.append("torch_parity_cache_identity_array_sha256_missing")
    else:
        _require_array_hash_mapping_match(
            expected.get("array_sha256"),
            cache_identity.get("array_sha256"),
            "torch_parity_cache_identity_array_sha256_mismatch",
            blockers,
        )
    if not _shape_mapping(cache_identity):
        blockers.append("torch_parity_cache_identity_shapes_missing")
    else:
        _require_cache_shape_mapping_match(
            expected,
            cache_identity,
            "torch_parity_cache_identity_shapes_mismatch",
            blockers,
        )


def _check_torch_parity_batch_shape(
    manifest: dict[str, Any],
    *,
    schema_version: Any,
    response_window: list[int] | None,
    response_batch_pairs: int | None,
    blockers: list[str],
) -> None:
    if response_batch_pairs is None:
        blockers.append("torch_parity_response_batch_pairs_invalid")
        return
    if schema_version == TORCH_PARITY_SWEEP_SCHEMA_VERSION:
        window_pairs = _safe_int(manifest.get("window_pairs"))
        stride_pairs = _safe_int(manifest.get("stride_pairs"))
        if window_pairs is None:
            blockers.append("torch_parity_window_pairs_missing")
        elif window_pairs != response_batch_pairs:
            blockers.append(
                "torch_parity_window_pairs_mismatch:"
                f"response_batch_pairs={response_batch_pairs}:parity_window_pairs={window_pairs}"
            )
        if stride_pairs is None:
            blockers.append("torch_parity_stride_pairs_missing")
        elif stride_pairs != response_batch_pairs:
            blockers.append(
                "torch_parity_stride_pairs_mismatch:"
                f"response_batch_pairs={response_batch_pairs}:parity_stride_pairs={stride_pairs}"
            )
        if response_window is not None:
            _check_torch_parity_sweep_tiles_response(
                manifest,
                response_window=response_window,
                response_batch_pairs=response_batch_pairs,
                blockers=blockers,
            )
        return

    if schema_version == TORCH_PARITY_SCHEMA_VERSION:
        parity_window = _pair_window(manifest)
        if parity_window is None:
            blockers.append("torch_parity_pair_window_missing")
            return
        parity_pairs = parity_window[1] - parity_window[0]
        if parity_pairs != response_batch_pairs:
            blockers.append(
                "torch_parity_window_pairs_mismatch:"
                f"response_batch_pairs={response_batch_pairs}:parity_window_pairs={parity_pairs}"
            )


def _check_torch_parity_thresholds(
    manifest: dict[str, Any],
    *,
    blockers: list[str],
) -> None:
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict):
        blockers.append("torch_parity_thresholds_missing")
        return
    argmax = _safe_int(thresholds.get("max_segnet_argmax_diff_pixels"))
    if argmax is None:
        blockers.append("torch_parity_threshold_max_segnet_argmax_diff_pixels_missing")
    elif argmax > MAX_ALLOWED_TORCH_PARITY_ARGMAX_DIFF_PIXELS:
        blockers.append(
            "torch_parity_threshold_max_segnet_argmax_diff_pixels_too_loose:"
            f"{argmax}>{MAX_ALLOWED_TORCH_PARITY_ARGMAX_DIFF_PIXELS}"
        )
    for key, limit in (
        ("max_posenet_output_abs_delta", 2.0e-3),
        ("max_segnet_logit_abs_delta", 1.0e-2),
        ("max_posenet_component_abs_delta", 2.0e-5),
    ):
        value = thresholds.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            blockers.append(f"torch_parity_threshold_{key}_invalid")
            continue
        if not math.isfinite(parsed) or parsed > limit:
            blockers.append(f"torch_parity_threshold_{key}_too_loose:{parsed}>{limit}")


def _check_torch_parity_sweep_tiles_response(
    manifest: dict[str, Any],
    *,
    response_window: list[int],
    response_batch_pairs: int,
    blockers: list[str],
) -> None:
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        blockers.append("torch_parity_sweep_rows_missing")
        return
    expected_start = response_window[0]
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            blockers.append(f"torch_parity_sweep_row_not_object:index={index}")
            continue
        window = _pair_window(row)
        if window is None:
            blockers.append(f"torch_parity_sweep_row_pair_window_invalid:index={index}")
            continue
        if window[0] != expected_start:
            blockers.append(
                "torch_parity_sweep_rows_not_contiguous:"
                f"index={index}:expected_start={expected_start}:window={window}"
            )
        width = window[1] - window[0]
        if width < 1 or width > response_batch_pairs:
            blockers.append(
                "torch_parity_sweep_row_width_invalid:"
                f"index={index}:width={width}:response_batch_pairs={response_batch_pairs}"
            )
        if window[1] > response_window[1]:
            blockers.append(
                "torch_parity_sweep_row_exceeds_response:"
                f"index={index}:response={response_window}:window={window}"
            )
        expected_start = window[1]
    if expected_start != response_window[1]:
        blockers.append(
            "torch_parity_sweep_rows_do_not_cover_response_end:"
            f"expected_end={response_window[1]}:actual_end={expected_start}"
        )


def _check_score_calibration_gate_manifest(
    manifest: dict[str, Any],
    *,
    response_payload: dict[str, Any],
    blockers: list[str],
) -> None:
    if not isinstance(manifest, dict):
        blockers.append("score_calibration_manifest_not_object")
        return
    if manifest.get("schema_version") != SCORE_CALIBRATION_SCHEMA_VERSION:
        blockers.append("score_calibration_schema_version_invalid")
    for field in AUTHORITY_FALSE_FIELDS:
        if manifest.get(field) is not False:
            blockers.append(f"score_calibration_{field}_not_false")
    decision_policy = manifest.get("decision_policy")
    if not isinstance(decision_policy, dict):
        blockers.append("score_calibration_decision_policy_missing")
        return
    if (
        decision_policy.get("allowed_use")
        != STRICT_AUTH_AXIS_SPEND_TRIAGE_ALLOWED_USE
    ):
        blockers.append("score_calibration_allowed_use_not_strict_auth_axis")
    if decision_policy.get("recommended_min_mlx_gap_for_spend_triage") is None:
        blockers.append("score_calibration_min_gap_missing")
    summary = manifest.get("summary")
    if isinstance(summary, dict):
        uncertain = _safe_int(summary.get("mlx_spend_triage_pairwise_uncertain_count"))
        if uncertain is not None and uncertain > 0:
            blockers.append("score_calibration_uncertain_pairwise_triage")
        certified = _safe_int(summary.get("mlx_spend_triage_pairwise_certified_count"))
        total = _safe_int(summary.get("mlx_spend_triage_pairwise_total_count"))
        if certified is None or certified <= 0:
            blockers.append("score_calibration_certified_pairwise_count_missing")
        if total is None or total <= 0:
            blockers.append("score_calibration_total_pairwise_count_missing")
        if decision_policy.get("calibration_uncertainty_score") is None:
            blockers.append("score_calibration_uncertainty_missing")
    else:
        blockers.append("score_calibration_summary_missing")
    _check_score_calibration_response_binding(
        manifest,
        response_payload=response_payload,
        blockers=blockers,
    )


def _check_score_calibration_response_binding(
    manifest: dict[str, Any],
    *,
    response_payload: dict[str, Any],
    blockers: list[str],
) -> None:
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        blockers.append("score_calibration_rows_missing")
        return
    matches = [
        row for row in rows if _score_calibration_row_matches_response(row, response_payload)
    ]
    if not matches:
        blockers.append("score_calibration_no_row_matches_response_identity")


def _score_calibration_row_matches_response(
    row: Any,
    response_payload: dict[str, Any],
) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("archive_sha256") != response_payload.get("archive_sha256"):
        return False
    if row.get("inflated_outputs_aggregate_sha256") != response_payload.get(
        "inflated_outputs_aggregate_sha256"
    ):
        return False
    for key in ("n_samples", "batch_pairs"):
        if _safe_int(row.get(key)) != _safe_int(response_payload.get(key)):
            return False
    if _pair_window({"pair_window": row.get("pair_window")}) != _pair_window(response_payload):
        return False
    if row.get("response_family") != response_payload.get("response_family"):
        return False
    try:
        if not math.isclose(
            float(row.get("mlx_score")),
            float(response_payload.get("canonical_score")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return False
        if not math.isclose(
            float(row.get("mlx_avg_posenet_dist")),
            float(response_payload.get("avg_posenet_dist")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return False
        if not math.isclose(
            float(row.get("mlx_avg_segnet_dist")),
            float(response_payload.get("avg_segnet_dist")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return False
    except (TypeError, ValueError):
        return False
    components = response_payload.get("components")
    row_components = row.get("mlx_components")
    if not isinstance(components, dict) or not isinstance(row_components, dict):
        return False
    if row_components.get("posenet_sha256") != components.get("posenet_sha256"):
        return False
    if row_components.get("segnet_sha256") != components.get("segnet_sha256"):
        return False
    for side in ("candidate", "reference"):
        response_side = _response_cache_side(response_payload, side)
        row_side = row.get(f"{side}_cache_identity")
        if not isinstance(row_side, dict):
            return False
        if row_side.get("hash_domain") != response_side.get("hash_domain"):
            return False
        if _hash_mapping(row_side.get("array_sha256")) != _hash_mapping(
            response_side.get("array_sha256")
        ):
            return False
        if _shape_mapping(row_side) != _shape_mapping(response_side):
            return False
    return True


def _check_profile_stability_manifest(
    manifest: dict[str, Any],
    *,
    response_payload: dict[str, Any],
    blockers: list[str],
    warnings: list[str],
) -> None:
    _check_optional_gate_manifest(
        manifest,
        blockers=blockers,
        warnings=warnings,
        schema_prefix="mlx_scorer_response_profile_stability",
        label="profile_stability",
    )
    if not isinstance(manifest, dict):
        return
    summary = manifest.get("profile_summary")
    if not isinstance(summary, dict):
        blockers.append("profile_stability_summary_missing")
        return
    row_count = _safe_int(summary.get("row_count"))
    if row_count is None or row_count < 2:
        blockers.append("profile_stability_row_count_lt_2")
    _check_int_match(
        response_payload.get("archive_size_bytes"),
        summary.get("archive_size_bytes"),
        "profile_stability_archive_size_bytes_mismatch",
        blockers,
    )
    response_window = _pair_window(response_payload)
    profile_start = _safe_int(summary.get("start_pair"))
    profile_max_pairs = _safe_int(summary.get("max_pairs"))
    if response_window is not None:
        if profile_start is None or profile_max_pairs is None:
            blockers.append("profile_stability_window_missing")
        elif response_window != [profile_start, profile_start + profile_max_pairs]:
            blockers.append(
                "profile_stability_pair_window_mismatch:"
                f"response={response_window}:profile={[profile_start, profile_start + profile_max_pairs]}"
            )
    _check_cache_path_match(
        _response_cache_side(response_payload, "candidate").get("path"),
        summary.get("candidate_cache_dir"),
        "profile_stability_candidate_cache_dir_mismatch",
        blockers,
    )
    _check_cache_path_match(
        _response_cache_side(response_payload, "reference").get("path"),
        summary.get("reference_cache_dir"),
        "profile_stability_reference_cache_dir_mismatch",
        blockers,
    )
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        blockers.append("profile_stability_selection_missing")
        return
    recommended = selection.get("recommended_row")
    if not isinstance(recommended, dict):
        blockers.append("profile_stability_recommended_row_missing")
        return
    response_device = _response_device(response_payload)
    response_batch_pairs = _safe_int(response_payload.get("batch_pairs"))
    recommended_device = str(recommended.get("device") or "").lower()
    recommended_batch_pairs = _safe_int(recommended.get("batch_pairs"))
    if recommended_device != response_device:
        blockers.append(
            "profile_stability_recommended_device_mismatch:"
            f"response={response_device}:profile={recommended_device}"
        )
    if recommended_batch_pairs != response_batch_pairs:
        blockers.append(
            "profile_stability_recommended_batch_pairs_mismatch:"
            f"response={response_batch_pairs}:profile={recommended_batch_pairs}"
        )
    _check_float_match(
        response_payload.get("canonical_score"),
        recommended.get("canonical_score"),
        "profile_stability_recommended_score_mismatch",
        blockers,
        abs_tol=1.0e-12,
    )
    _check_float_match(
        response_payload.get("avg_posenet_dist"),
        recommended.get("avg_posenet_dist"),
        "profile_stability_recommended_posenet_avg_mismatch",
        blockers,
        abs_tol=1.0e-12,
    )
    _check_float_match(
        response_payload.get("avg_segnet_dist"),
        recommended.get("avg_segnet_dist"),
        "profile_stability_recommended_segnet_avg_mismatch",
        blockers,
        abs_tol=1.0e-12,
    )
    components = response_payload.get("components")
    if not isinstance(components, dict):
        blockers.append("profile_stability_response_components_missing")
    else:
        _require_hash_match(
            components.get("posenet_sha256"),
            recommended.get("posenet_sha256"),
            "profile_stability_recommended_posenet_sha256_mismatch",
            blockers,
        )
        _require_hash_match(
            components.get("segnet_sha256"),
            recommended.get("segnet_sha256"),
            "profile_stability_recommended_segnet_sha256_mismatch",
            blockers,
        )


def _check_batch_invariance_gate_manifest(
    manifest: dict[str, Any],
    *,
    response_payload: dict[str, Any],
    blockers: list[str],
    warnings: list[str],
    response_device: str,
    response_batch_pairs: int | None,
    required: bool,
) -> None:
    _check_optional_gate_manifest(
        manifest,
        blockers=blockers,
        warnings=warnings,
        schema_prefix="mlx_scorer_batch_invariance",
        label="batch_invariance",
        require_passed=required,
    )
    if not isinstance(manifest, dict):
        return
    if not required:
        if manifest.get("passed") is False:
            warnings.append("batch_invariance_supplied_but_not_required_and_failing")
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
    _check_cache_path_match(
        _response_cache_side(response_payload, "candidate").get("path"),
        manifest.get("cache_dir"),
        "batch_invariance_candidate_cache_dir_mismatch",
        blockers,
    )
    _check_int_match(
        _response_cache_side(response_payload, "candidate").get("pair_count"),
        manifest.get("total_pair_count"),
        "batch_invariance_total_pair_count_mismatch",
        blockers,
    )
    response_window = _pair_window(response_payload)
    gate_start = _safe_int(manifest.get("start_pair"))
    if response_window is not None:
        if gate_start is None or gate_batch_pairs is None:
            blockers.append("batch_invariance_pair_window_missing")
        elif not (
            response_window[0] <= gate_start
            and gate_start + gate_batch_pairs <= response_window[1]
        ):
            blockers.append(
                "batch_invariance_window_not_within_response_window:"
                f"response={response_window}:gate={[gate_start, gate_start + gate_batch_pairs]}"
            )


def _response_cache_side(payload: dict[str, Any], side: str) -> dict[str, Any]:
    identity = payload.get("cache_identity")
    if not isinstance(identity, dict):
        return {}
    item = identity.get(side)
    return item if isinstance(item, dict) else {}


def _check_hash_match(
    lhs: Any,
    rhs: Any,
    blocker: str,
    blockers: list[str],
) -> None:
    if lhs is None or rhs is None:
        return
    if not _is_sha256(str(lhs)) or not _is_sha256(str(rhs)) or str(lhs) != str(rhs):
        blockers.append(blocker)


def _require_hash_match(
    lhs: Any,
    rhs: Any,
    blocker: str,
    blockers: list[str],
) -> None:
    if not _is_sha256(str(lhs)) or not _is_sha256(str(rhs)) or str(lhs) != str(rhs):
        blockers.append(blocker)


def _check_int_match(lhs: Any, rhs: Any, blocker: str, blockers: list[str]) -> None:
    if lhs is None or rhs is None:
        blockers.append(blocker)
        return
    left = _safe_int(lhs)
    right = _safe_int(rhs)
    if left is None or right is None or left != right:
        blockers.append(blocker)


def _check_float_match(
    lhs: Any,
    rhs: Any,
    blocker: str,
    blockers: list[str],
    *,
    abs_tol: float,
) -> None:
    try:
        left = float(lhs)
        right = float(rhs)
    except (TypeError, ValueError):
        blockers.append(blocker)
        return
    if not math.isfinite(left) or not math.isfinite(right):
        blockers.append(blocker)
        return
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=abs_tol):
        blockers.append(blocker)


def _check_cache_path_match(lhs: Any, rhs: Any, blocker: str, blockers: list[str]) -> None:
    if not lhs or not rhs:
        blockers.append(blocker)
        return
    if _normalize_path_text(str(lhs)) != _normalize_path_text(str(rhs)):
        blockers.append(blocker)


def _normalize_path_text(value: str) -> str:
    return str(Path(value)).rstrip("/")


def _pair_window(payload: dict[str, Any]) -> list[int] | None:
    window = payload.get("pair_window")
    if not isinstance(window, list) or len(window) != 2:
        return None
    start = _safe_int(window[0])
    stop = _safe_int(window[1])
    if start is None or stop is None:
        return None
    return [start, stop]


def _hash_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
        item = value.get(key)
        if _is_sha256(str(item)):
            out[key] = str(item)
    return out if len(out) == 3 else {}


def _require_array_hash_mapping_match(
    lhs: Any,
    rhs: Any,
    blocker: str,
    blockers: list[str],
) -> None:
    left = _hash_mapping(lhs)
    right = _hash_mapping(rhs)
    if not left or not right:
        blockers.append(blocker)
        return
    for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
        if left.get(key) != right.get(key):
            blockers.append(f"{blocker}:{key}")


def _shape_mapping(value: dict[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for key in ("pair_indices_shape", "posenet_yuv6_pair_shape", "segnet_last_rgb_shape"):
        item = value.get(key)
        if isinstance(item, list) and item:
            out[key] = item
    return out if len(out) == 3 else {}


def _require_cache_shape_mapping_match(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    blocker: str,
    blockers: list[str],
) -> None:
    left = _shape_mapping(lhs)
    right = _shape_mapping(rhs)
    if not left or not right:
        blockers.append(blocker)
        return
    for key in ("pair_indices_shape", "posenet_yuv6_pair_shape", "segnet_last_rgb_shape"):
        if list(left.get(key, [])) != list(right.get(key, [])):
            blockers.append(f"{blocker}:{key}")


def _shape_mapping_from_auth_summary(value: Any) -> dict[str, list[Any]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[Any]] = {}
    for key in ("pair_indices", "posenet_yuv6_pair", "segnet_last_rgb"):
        item = value.get(key)
        if isinstance(item, list) and item:
            out[key] = item
    return out if len(out) == 3 else {}


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
    "ADVISORY_VERDICT",
    "FAIL_VERDICT",
    "GATE_SET_VERSION",
    "PASS_VERDICT",
    "SCHEMA_VERSION",
    "build_mlx_scorer_production_contract_manifest",
    "load_json_object",
    "write_production_contract_manifest",
]
