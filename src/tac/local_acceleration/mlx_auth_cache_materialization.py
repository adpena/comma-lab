# SPDX-License-Identifier: MIT
"""Plan auth-surface cache materialization for local MLX scorer acceleration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_cache_audit import PASS_VERDICT as CACHE_AUDIT_PASS_VERDICT

SCHEMA_VERSION = "mlx_auth_cache_materialization_plan.v1"
READY_VERDICT = "READY_USE_EXISTING_AUTH_TRANSFER_CACHE"
REQUIRED_VERDICT = "AUTH_CACHE_MATERIALIZATION_REQUIRED"

AUTHORITY_FALSE_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "promotable",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)

_RAW_OR_TENSOR_ACTION = "materialize_auth_axis_tensor_cache_from_modal_linux_raw_or_export_linux_tensor_cache"
_HASH_ACTION = "rerun_auth_eval_with_scorer_input_cache_hashes"
_PAIR_COUNT_ACTION = "rebuild_full_sample_cache_with_matching_pair_count"
_WRONG_ARCHIVE_ACTION = "stop_wrong_archive"
_READY_ACTION = "use_existing_cache_for_local_mlx_transfer_calibration"
_GENERIC_ACTION = "resolve_cache_auth_blockers_before_training"


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def write_materialization_plan(plan: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mlx_auth_cache_materialization_plan(
    cache_auth_audit: dict[str, Any],
    *,
    production_contract: dict[str, Any] | None = None,
    cache_audit_path: str | None = None,
    production_contract_path: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Turn a cache/auth audit into an actionable auth-cache materialization plan.

    The plan is deliberately non-authoritative: it can route local engineering
    work, but it cannot make an MLX cache a score claim or promotion artifact.
    """

    audit_passed = _audit_passed(cache_auth_audit)
    blockers = _string_list(cache_auth_audit.get("blockers"))
    cache = _object(cache_auth_audit.get("cache"))
    auth_eval = _object(cache_auth_audit.get("auth_eval"))
    canonical_equation = _object(cache_auth_audit.get("canonical_equation"))
    production = _object(production_contract)

    classification = _classify_surfaces(cache=cache, auth_eval=auth_eval)
    actions = _next_actions(
        audit_passed=audit_passed,
        blockers=blockers,
        classification=classification,
        auth_eval=auth_eval,
    )
    production_blockers = _string_list(production.get("blockers")) if production else []

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "verdict": READY_VERDICT if audit_passed else REQUIRED_VERDICT,
        "passed": audit_passed,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "allowed_use": (
            [
                "local_mlx_training_transfer_calibration",
                "auth_axis_cache_materialized_for_local_tensor_training",
            ]
            if audit_passed
            else [
                "local_cache_materialization_planning_only",
                "do_not_use_for_auth_axis_transfer_calibration",
                "do_not_use_for_local_mlx_training_targets",
            ]
        ),
        "next_materialization_action": actions[0],
        "next_actions": actions,
        "blockers": blockers,
        "surface_classification": classification,
        "canonical_equation": {
            "equation_id": canonical_equation.get("equation_id"),
            "identity_residual": canonical_equation.get("identity_residual"),
            "eligible_for_local_mlx_transfer_calibration": canonical_equation.get(
                "eligible_for_local_mlx_transfer_calibration"
            ),
            "blockers": _string_list(canonical_equation.get("blockers")),
        },
        "production_contract": {
            "path": production_contract_path,
            "passed": production.get("passed") if production else None,
            "verdict": production.get("verdict") if production else None,
            "blockers": production_blockers,
        },
        "required_artifacts": _required_artifacts(
            audit_passed=audit_passed,
            classification=classification,
            auth_eval=auth_eval,
        ),
        "recommended_commands": _recommended_commands(
            action=actions[0],
            cache_audit_path=cache_audit_path,
        ),
        "source_artifacts": {
            "cache_auth_audit": cache_audit_path,
            "production_contract": production_contract_path,
        },
        "authority_status": (
            "This plan is an MLX local-acceleration routing artifact only. "
            "Exact contest CPU/CUDA auth eval remains required for every score, "
            "frontier, promotion, rank/kill, or submission claim."
        ),
    }


def _audit_passed(audit: dict[str, Any]) -> bool:
    return bool(audit.get("passed") is True and audit.get("verdict") == CACHE_AUDIT_PASS_VERDICT)


def _classify_surfaces(*, cache: dict[str, Any], auth_eval: dict[str, Any]) -> dict[str, Any]:
    archive_match = _match(cache.get("archive_sha256"), auth_eval.get("archive_sha256"))
    aggregate_match = _match(
        cache.get("inflated_outputs_aggregate_sha256"),
        auth_eval.get("inflated_outputs_aggregate_sha256"),
    )
    raw_match = _match(cache.get("raw_sha256"), auth_eval.get("raw_file_sha256"))
    cache_hashes = _object(cache.get("array_sha256"))
    auth_hashes = _object(auth_eval.get("scorer_input_array_sha256"))
    array_matches = {
        name: _match(cache_hashes.get(name), auth_hashes.get(name))
        for name in ("pair_indices", "segnet_last_rgb", "posenet_yuv6_pair")
    }
    scorer_hashes_present = bool(auth_eval.get("scorer_input_hash_domain")) and bool(auth_hashes)
    decoded_surface_match = bool(aggregate_match["match"] and raw_match["match"])
    scorer_input_surface_match = scorer_hashes_present and all(
        item["match"] for item in array_matches.values()
    )

    if archive_match["match"] and not decoded_surface_match:
        decoded_class = "same_archive_different_decoded_raw_surface"
    elif decoded_surface_match:
        decoded_class = "decoded_raw_surface_identity"
    elif not archive_match["match"]:
        decoded_class = "archive_identity_mismatch"
    else:
        decoded_class = "decoded_raw_surface_unresolved"

    if scorer_input_surface_match:
        scorer_class = "scorer_input_tensor_identity"
    elif not scorer_hashes_present:
        scorer_class = "auth_scorer_input_hashes_missing"
    elif archive_match["match"]:
        scorer_class = "same_archive_different_scorer_input_tensors"
    else:
        scorer_class = "scorer_input_tensor_identity_unresolved"

    return {
        "archive_identity": archive_match,
        "decoded_raw_surface": {
            "classification": decoded_class,
            "inflated_outputs_aggregate_sha256": aggregate_match,
            "raw_sha256": raw_match,
            "match": decoded_surface_match,
        },
        "scorer_input_surface": {
            "classification": scorer_class,
            "hash_domain": {
                "cache": cache.get("hash_domain"),
                "auth_eval": auth_eval.get("scorer_input_hash_domain"),
                "match": _string(cache.get("hash_domain")) is not None
                and _string(cache.get("hash_domain"))
                == _string(auth_eval.get("scorer_input_hash_domain")),
            },
            "array_sha256": array_matches,
            "match": scorer_input_surface_match,
        },
        "pair_count": {
            "cache": cache.get("pair_count"),
            "auth_eval": auth_eval.get("n_samples"),
            "match": cache.get("pair_count") == auth_eval.get("n_samples"),
        },
        "auth_axis": {
            "evidence_grade": auth_eval.get("evidence_grade"),
            "score_axis": auth_eval.get("score_axis"),
            "score": auth_eval.get("score"),
            "pose_avg": auth_eval.get("pose_avg"),
            "seg_avg": auth_eval.get("seg_avg"),
        },
    }


def _next_actions(
    *,
    audit_passed: bool,
    blockers: list[str],
    classification: dict[str, Any],
    auth_eval: dict[str, Any],
) -> list[str]:
    if audit_passed:
        return [_READY_ACTION]
    if classification["archive_identity"]["match"] is False:
        return [_WRONG_ARCHIVE_ACTION, _GENERIC_ACTION]
    if any("pair_count" in blocker or "n_samples" in blocker for blocker in blockers):
        return [_PAIR_COUNT_ACTION, _GENERIC_ACTION]
    if classification["scorer_input_surface"]["classification"] == "auth_scorer_input_hashes_missing":
        return [_HASH_ACTION, _GENERIC_ACTION]
    raw_or_tensor_blocker = any(
        blocker.startswith("raw_sha256_mismatch")
        or blocker.startswith("inflated_outputs_aggregate_sha256_mismatch")
        or blocker.startswith("scorer_input_array_sha256_mismatch")
        for blocker in blockers
    )
    if raw_or_tensor_blocker:
        return [_RAW_OR_TENSOR_ACTION, _HASH_ACTION, _GENERIC_ACTION]
    if not _string(auth_eval.get("scorer_input_hash_domain")):
        return [_HASH_ACTION, _GENERIC_ACTION]
    return [_GENERIC_ACTION]


def _required_artifacts(
    *,
    audit_passed: bool,
    classification: dict[str, Any],
    auth_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    if audit_passed:
        return []
    artifacts: list[dict[str, Any]] = []
    if classification["decoded_raw_surface"]["match"] is not True:
        artifacts.append(
            {
                "name": "modal_linux_inflated_raw_or_equivalent_tensor_export",
                "required": True,
                "expected_raw_sha256": auth_eval.get("raw_file_sha256"),
                "expected_inflated_outputs_aggregate_sha256": auth_eval.get(
                    "inflated_outputs_aggregate_sha256"
                ),
                "purpose": (
                    "Build the local MLX scorer-input cache from the same Linux auth-eval "
                    "raw bytes, or export the Linux scorer-input tensors directly."
                ),
            }
        )
    if classification["scorer_input_surface"]["match"] is not True:
        artifacts.append(
            {
                "name": "auth_axis_scorer_input_tensor_hashes",
                "required": True,
                "expected_hash_domain": auth_eval.get("scorer_input_hash_domain"),
                "expected_array_sha256": auth_eval.get("scorer_input_array_sha256"),
                "purpose": "Verify the materialized local tensors are byte-identical to auth-side scorer inputs.",
            }
        )
    return artifacts


def _recommended_commands(*, action: str, cache_audit_path: str | None) -> list[str]:
    commands: list[str] = []
    if cache_audit_path:
        commands.append(
            "jq '{passed, verdict, blockers, surface_action: .next_materialization_action}' "
            "<plan.json>"
        )
    if action == _RAW_OR_TENSOR_ACTION:
        commands.append(
            "recover or export the Modal/Linux auth raw surface, then rebuild "
            "tools/build_mlx_scorer_input_cache.py from that raw/tensor source"
        )
        commands.append(
            "rerun tools/audit_mlx_scorer_input_cache.py and require "
            "PASS_CACHE_AUTH_EVAL_IDENTITY before MLX training-target use"
        )
    elif action == _HASH_ACTION:
        commands.append(
            "rerun auth eval with --scorer-input-cache-hashes-out and attach the hash manifest"
        )
    elif action == _READY_ACTION:
        commands.append(
            "run tools/check_mlx_scorer_production_contract.py with the passing cache-auth audit"
        )
    return commands


def _match(lhs: Any, rhs: Any) -> dict[str, Any]:
    left = _string(lhs)
    right = _string(rhs)
    return {
        "cache": left,
        "auth_eval": right,
        "match": bool(left and right and left == right),
    }


def _string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = [
    "READY_VERDICT",
    "REQUIRED_VERDICT",
    "SCHEMA_VERSION",
    "build_mlx_auth_cache_materialization_plan",
    "load_json_object",
    "write_materialization_plan",
]
