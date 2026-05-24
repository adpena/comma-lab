# SPDX-License-Identifier: MIT
"""Adapter for PR95 HNeRV/Muon local-training probe manifests.

The probe measures training portability and candidate quality. It does not
grant score, promotion, dispatch, rank, or kill authority. This adapter turns
its manifest into the same planning-only queue row shape used by optimizer and
MLX sweep artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.optimization.local_training_runtime_profile import (
    LocalTrainingRuntimeProfileError,
    runtime_profile_summary_from_training_manifest,
)
from tac.optimization.optimizer_training_signal_bridge import (
    build_optimizer_training_signal_wire_in,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    auth_bridge_score_rankable,
    require_no_truthy_authority_fields,
)

SCHEMA = "pr95_local_training_probe_manifest_v1"
PLAN_SCHEMA = "pr95_local_training_probe_plan_v1"
CANDIDATE_PAYLOAD_SCHEMA = "pr95_muon_local_training_candidate_payload.v1"
LANE_ID = "lane_pr95_local_mps_source_faithful_training_probe_20260519"
REQUIRED_FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "promotable",
    "dispatch_attempted",
    "gpu_launched",
    "dispatch_packet_ready",
)
PR95_LOCAL_TRAINING_BLOCKERS: tuple[str, ...] = (
    "pr95_local_training_probe_is_proxy_signal",
    "requires_byte_closed_archive_export",
    "requires_runtime_consumption_proof",
    "requires_exact_cpu_cuda_auth_eval_before_score_claim",
    "requires_lane_claim_before_dispatch",
)


class PR95MuonLocalTrainingIntegrationError(ValueError):
    """Raised when a PR95 local-training manifest leaks authority."""


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _stage_modules(payload: Mapping[str, Any]) -> list[str]:
    modules: list[str] = []
    for stage in _as_list(payload.get("stages")):
        if isinstance(stage, Mapping) and stage.get("module"):
            modules.append(str(stage["module"]))
    for result in _as_list(payload.get("results")):
        if isinstance(result, Mapping) and result.get("stage_module"):
            module = str(result["stage_module"])
            if module not in modules:
                modules.append(module)
    return modules


def _best_training_score(payload: Mapping[str, Any]) -> float | None:
    scores = [
        score
        for result in _as_list(payload.get("results"))
        if isinstance(result, Mapping)
        for score in (_finite_float(result.get("best_score")),)
        if score is not None
    ]
    return min(scores) if scores else None


def _latest_training_score(payload: Mapping[str, Any]) -> float | None:
    for result in reversed(_as_list(payload.get("results"))):
        if isinstance(result, Mapping):
            score = _finite_float(result.get("best_score"))
            if score is not None:
                return score
    return None


def _auth_eval_score(payload: Mapping[str, Any]) -> float | None:
    bridge = payload.get("auth_eval_bridge")
    if isinstance(bridge, Mapping):
        return _finite_float(bridge.get("auth_eval_canonical_score"))
    return None


def _archive_zip(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    archive_zip = payload.get("archive_zip")
    return archive_zip if isinstance(archive_zip, Mapping) else {}


def _auth_eval_bridge(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    bridge = payload.get("auth_eval_bridge")
    return bridge if isinstance(bridge, Mapping) else {}


def validate_pr95_local_training_manifest(payload: Mapping[str, Any]) -> None:
    """Reject PR95 manifests that carry any authority-bearing flag."""

    if payload.get("schema") not in {SCHEMA, PLAN_SCHEMA}:
        raise PR95MuonLocalTrainingIntegrationError("PR95 manifest schema mismatch")
    for key in REQUIRED_FALSE_AUTHORITY_FIELDS:
        if key in payload and payload.get(key) is not False:
            raise PR95MuonLocalTrainingIntegrationError(f"{key} must be explicit false when present")
    bridge = _auth_eval_bridge(payload)
    for key in REQUIRED_FALSE_AUTHORITY_FIELDS:
        if key in bridge and bridge.get(key) is not False:
            raise PR95MuonLocalTrainingIntegrationError(f"auth_eval_bridge.{key} must be false")
    try:
        require_no_truthy_authority_fields(payload, context="pr95_local_training_manifest")
    except ValueError as exc:
        raise PR95MuonLocalTrainingIntegrationError(str(exc)) from exc


def adapt_pr95_local_training_manifest_to_candidate(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Return one planning-only optimizer queue row for a PR95 probe manifest."""

    validate_pr95_local_training_manifest(payload)
    stages = _stage_modules(payload)
    best_score = _best_training_score(payload)
    latest_score = _latest_training_score(payload)
    auth_score = _auth_eval_score(payload)
    try:
        runtime_profile_summary = runtime_profile_summary_from_training_manifest(payload)
    except LocalTrainingRuntimeProfileError as exc:
        raise PR95MuonLocalTrainingIntegrationError(str(exc)) from exc
    archive_zip = _archive_zip(payload)
    bridge = _auth_eval_bridge(payload)
    seed = _finite_int(payload.get("seed"))
    stage_count = _finite_int(payload.get("stage_count")) or len(stages)
    device = str(payload.get("device_selected") or payload.get("device_requested") or "unknown")
    has_archive = bool(archive_zip.get("sha256") and archive_zip.get("bytes"))
    has_auth_bridge = bool(bridge.get("ok") is True and bridge.get("auth_eval_json_sha256"))
    rankable_auth_score = (
        auth_score if auth_score is not None and auth_bridge_score_rankable(bridge) else None
    )
    runtime_rank_score = _finite_float(
        runtime_profile_summary.get("best_timing_value_seconds")
    )
    if rankable_auth_score is not None:
        rank_score = rankable_auth_score
        rank_score_field = "contest_auth_eval_bridge_score_not_authority"
    elif best_score is not None:
        rank_score = best_score
        rank_score_field = "training_best_score_proxy_not_authority"
    elif runtime_rank_score is not None:
        rank_score = runtime_rank_score
        timing_field = str(
            runtime_profile_summary.get("best_timing_field") or "runtime_seconds"
        )
        rank_score_field = f"{timing_field}_cost_signal_not_score"
    else:
        rank_score = None
        rank_score_field = "no_rank_score_noncomparable_auth_bridge"
    candidate_id = str(
        payload.get("candidate_id")
        or f"pr95_muon_hnerv_local_{device}_stages{stage_count}_seed{seed or 'unknown'}"
    )
    blockers = [
        *PR95_LOCAL_TRAINING_BLOCKERS,
        *([] if has_archive else ["local_training_probe_archive_export_missing"]),
        *([] if has_auth_bridge else ["local_training_probe_auth_eval_bridge_missing"]),
        *([] if best_score is not None else ["local_training_probe_best_score_missing"]),
        *[
            str(item)
            for item in runtime_profile_summary.get("blockers", [])
            if str(item)
        ],
    ]
    optimizer_recipe = _mapping(payload.get("optimizer_recipe"))
    optimizer_params = {
        "stage_count": stage_count,
        "seed": seed,
        "device": device,
        "full_curriculum": bool(payload.get("full_curriculum", False)),
        "uses_stage8_muon": "stage8_muon_finetune" in stages,
        "archive_exported": has_archive,
        "auth_eval_bridge_present": has_auth_bridge,
    }
    for key in (
        "optimizer_descriptor_id",
        "optimizer_config_sha256",
        "optimizer_backend_status",
        "parameter_group_lr_policy_id",
        "parameter_group_lr_policy_sha256",
        "parameter_group_fingerprint_sha256",
    ):
        if optimizer_recipe.get(key) is not None:
            optimizer_params[key] = optimizer_recipe[key]
    if runtime_profile_summary.get("profile_count"):
        optimizer_params.update(
            {
                "best_local_backend": runtime_profile_summary.get("best_local_backend"),
                "best_runtime_timing_field": runtime_profile_summary.get(
                    "best_timing_field"
                ),
                "best_runtime_timing_value_seconds": runtime_profile_summary.get(
                    "best_timing_value_seconds"
                ),
                "kernel_fusion_strategy_ids": runtime_profile_summary.get(
                    "kernel_fusion_strategy_ids"
                ),
            }
        )
    consumer_payload = {
        "schema": CANDIDATE_PAYLOAD_SCHEMA,
        "pr95_muon_local_training": {
            "stage_count": stage_count,
            "stage_modules": stages,
            "muon_partition": {
                "hidden_2d_plus_weights": "Muon",
                "bias_norm_scalar_stem_rgb_head": "AdamW",
                "source": "PR95 hnerv_muon stage8 convention",
            },
            "timing_smoke": {
                "wall_seconds": _finite_float(payload.get("wall_seconds")),
                "runtime_profile_summary": dict(runtime_profile_summary),
                "results": [
                    {
                        "stage_index": result.get("stage_index"),
                        "stage_module": result.get("stage_module"),
                        "epochs_run": result.get("epochs_run"),
                        "wall_seconds": result.get("wall_seconds"),
                        "best_score": result.get("best_score"),
                    }
                    for result in _as_list(payload.get("results"))
                    if isinstance(result, Mapping)
                ],
            },
            "archive_export": {
                "present": has_archive,
                "path": archive_zip.get("path"),
                "bytes": archive_zip.get("bytes"),
                "sha256": archive_zip.get("sha256"),
                "member_sha256": archive_zip.get("member_sha256"),
            },
            "auth_bridge": {
                "present": has_auth_bridge,
                "axis": bridge.get("score_axis"),
                "json_sha256": bridge.get("auth_eval_json_sha256"),
                "canonical_score": bridge.get("auth_eval_canonical_score"),
                "score_comparable": bridge.get("score_comparable"),
            },
            "missing_blockers": blockers,
            "source_tree_sha256": payload.get("source_tree_sha256"),
            "torch_version": payload.get("torch_version"),
            "platform": payload.get("platform"),
            "packet_compiler_bridge": (
                runtime_profile_summary.get("profiles", [{}])[0].get(
                    "packet_compiler_bridge"
                )
                if runtime_profile_summary.get("profile_count")
                else None
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
    }
    row = {
        "candidate_id": candidate_id,
        "source_paths": [_repo_rel(source_path, repo_root)],
        "lane_id": str(payload.get("lane_id") or LANE_ID),
        "lane_class": "pr95_hnerv_muon_local_training_proxy",
        "candidate_family": "pr95_hnerv_muon_training_probe",
        "representation_family": "hnerv",
        "substrate_family": "nerv_family",
        "training_signal_kind": "local_representation_training_optimizer_schedule_probe",
        "profile": "pr95_hnerv_muon_training_smoke",
        "param_schema": "pr95_hnerv_muon_local_training_manifest_params_v1",
        "candidate_params": optimizer_params,
        "op_params": optimizer_params,
        "rank_score": rank_score,
        "rank_score_field": rank_score_field,
        "training_best_score": best_score,
        "training_latest_score": latest_score,
        "auth_eval_bridge_score": rankable_auth_score,
        "advisory_auth_eval_bridge_score": auth_score if rankable_auth_score is None else None,
        "auth_eval_bridge_score_axis": bridge.get("score_axis"),
        "auth_eval_bridge_score_comparable": bridge.get("score_comparable"),
        "stage_count": stage_count,
        "stage_modules": stages,
        "device_selected": device,
        "source_tree_sha256": payload.get("source_tree_sha256"),
        "archive_path": archive_zip.get("path"),
        "archive_sha256": archive_zip.get("sha256"),
        "archive_bytes": archive_zip.get("bytes"),
        "candidate_archive_path": archive_zip.get("path"),
        "candidate_archive_sha256": archive_zip.get("sha256"),
        "candidate_archive_bytes": archive_zip.get("bytes"),
        "score_affecting_payload_changed": bool(has_archive),
        "charged_bits_changed": bool(has_archive),
        "score_affecting_runtime_changed": bool(has_archive),
        "consumer_payload": consumer_payload,
        "solver_stack_wire_in": build_optimizer_training_signal_wire_in(
            candidate_id=candidate_id,
            profile_id="pr95_hnerv_muon_training_smoke",
            lane_id=str(payload.get("lane_id") or LANE_ID),
            lane_class="pr95_hnerv_muon_local_training_proxy",
            candidate_family="pr95_hnerv_muon_training_probe",
            representation_family="hnerv",
            substrate_family="nerv_family",
            training_signal_kind="local_representation_training_optimizer_schedule_probe",
            param_schema="pr95_hnerv_muon_local_training_manifest_params_v1",
            candidate_params=optimizer_params,
            source_anchor="PR95 HNeRV Muon local training probe manifest",
            score_lowering_hypothesis=(
                "Local PR95/HNeRV training smokes rank optimizer/scheduler variants "
                "for byte-closed export and exact auth replay."
            ),
            dispatch_blockers=blockers,
        ),
        "evidence_semantics": "pr95_hnerv_muon_local_training_proxy_not_exact_auth_eval",
        "evidence_grade": "local_training_portability_probe_advisory",
    }
    return apply_proxy_evidence_boundary(row, dispatch_blockers=blockers)


__all__ = [
    "CANDIDATE_PAYLOAD_SCHEMA",
    "LANE_ID",
    "PLAN_SCHEMA",
    "PR95_LOCAL_TRAINING_BLOCKERS",
    "REQUIRED_FALSE_AUTHORITY_FIELDS",
    "SCHEMA",
    "PR95MuonLocalTrainingIntegrationError",
    "adapt_pr95_local_training_manifest_to_candidate",
    "validate_pr95_local_training_manifest",
]
