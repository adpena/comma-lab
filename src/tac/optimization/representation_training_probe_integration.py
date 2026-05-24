# SPDX-License-Identifier: MIT
"""Adapter for representation-training probe manifests.

The manifest schema is intentionally substrate-family agnostic: PR95/HNeRV,
new HNeRV variants, broader NeRV-family models, non-NeRV learned codecs, and
non-neural representation experiments can all enter the same planning queue.
Rows remain proxy evidence until archive/runtime custody and exact auth eval
prove score authority.
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
    ordered_unique,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    auth_bridge_score_rankable,
    require_no_truthy_authority_fields,
)

SCHEMA = "representation_training_probe_manifest_v1"
PLAN_SCHEMA = "representation_training_probe_plan_v1"
CANDIDATE_PAYLOAD_SCHEMA = "representation_training_candidate_payload.v1"
DEFAULT_LANE_ID = "offline_representation_training_probe"
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
REPRESENTATION_TRAINING_BLOCKERS: tuple[str, ...] = (
    "representation_training_probe_is_proxy_signal",
    "requires_byte_closed_archive_export",
    "requires_runtime_consumption_proof",
    "requires_exact_cpu_cuda_auth_eval_before_score_claim",
    "requires_lane_claim_before_dispatch",
)


class RepresentationTrainingProbeIntegrationError(ValueError):
    """Raised when a representation-training manifest leaks authority."""


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _str_list(value: Any) -> list[str]:
    return ordered_unique(str(item) for item in _as_list(value))


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
    return min(scores) if scores else _finite_float(payload.get("best_training_score"))


def _latest_training_score(payload: Mapping[str, Any]) -> float | None:
    for result in reversed(_as_list(payload.get("results"))):
        if isinstance(result, Mapping):
            score = _finite_float(result.get("best_score"))
            if score is not None:
                return score
    return _finite_float(payload.get("latest_training_score"))


def _auth_eval_score(payload: Mapping[str, Any]) -> float | None:
    bridge = _mapping(payload.get("auth_eval_bridge"))
    return _finite_float(bridge.get("auth_eval_canonical_score"))


def _reject_truthy_authority(payload: Mapping[str, Any], *, prefix: str = "") -> None:
    for key in REQUIRED_FALSE_AUTHORITY_FIELDS:
        if key in payload and payload.get(key) is not False:
            dotted = f"{prefix}.{key}" if prefix else key
            raise RepresentationTrainingProbeIntegrationError(
                f"{dotted} must be explicit false when present"
            )


def validate_representation_training_manifest(payload: Mapping[str, Any]) -> None:
    """Reject generic training manifests that carry any authority-bearing flag."""

    if payload.get("schema") not in {SCHEMA, PLAN_SCHEMA}:
        raise RepresentationTrainingProbeIntegrationError(
            "representation training manifest schema mismatch"
        )
    _reject_truthy_authority(payload)
    bridge = _mapping(payload.get("auth_eval_bridge"))
    _reject_truthy_authority(bridge, prefix="auth_eval_bridge")
    _reject_truthy_authority(
        _mapping(payload.get("source_video_training_target")),
        prefix="source_video_training_target",
    )
    try:
        require_no_truthy_authority_fields(
            payload,
            context="representation_training_manifest",
        )
    except ValueError as exc:
        raise RepresentationTrainingProbeIntegrationError(str(exc)) from exc
    for attr in ("candidate_id", "representation_family", "substrate_family"):
        if not str(payload.get(attr) or "").strip():
            raise RepresentationTrainingProbeIntegrationError(f"{attr} is required")


def _candidate_params(
    payload: Mapping[str, Any],
    *,
    runtime_profile_summary: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer_recipe = _mapping(payload.get("optimizer_recipe"))
    explicit = payload.get("candidate_params")
    if isinstance(explicit, Mapping):
        params = dict(explicit)
    else:
        scheduler_recipe = _mapping(payload.get("scheduler_recipe"))
        training_recipe = _mapping(payload.get("training_recipe"))
        stage_count = _finite_int(payload.get("stage_count")) or len(_stage_modules(payload))
        params = {
            "stage_count": stage_count,
            "seed": _finite_int(payload.get("seed")),
            "device": payload.get("device_selected") or payload.get("device_requested"),
            "optimizer_recipe_id": optimizer_recipe.get("id"),
            "scheduler_recipe_id": scheduler_recipe.get("id"),
            "training_recipe_id": training_recipe.get("id"),
        }
    for key in (
        "optimizer_descriptor_id",
        "optimizer_config_sha256",
        "optimizer_backend_status",
        "parameter_group_lr_policy_id",
        "parameter_group_lr_policy_sha256",
        "parameter_group_fingerprint_sha256",
    ):
        if params.get(key) is None and optimizer_recipe.get(key) is not None:
            params[key] = optimizer_recipe[key]
    if runtime_profile_summary.get("profile_count"):
        params.update(
            {
                "best_local_backend": runtime_profile_summary.get("best_local_backend"),
                "best_runtime_timing_field": runtime_profile_summary.get(
                    "best_timing_field"
                ),
                "best_runtime_timing_value_seconds": runtime_profile_summary.get(
                    "best_timing_value_seconds"
                ),
                "best_scheduler_resource_kind": runtime_profile_summary.get(
                    "best_scheduler_resource_kind"
                ),
                "kernel_fusion_strategy_ids": runtime_profile_summary.get(
                    "kernel_fusion_strategy_ids"
                ),
            }
        )
    return params


def _extra_blockers(payload: Mapping[str, Any]) -> list[str]:
    return ordered_unique(
        [
            *REPRESENTATION_TRAINING_BLOCKERS,
            *_str_list(payload.get("dispatch_blockers")),
        ]
    )


def _source_faithful_preprocess_signal(payload: Mapping[str, Any]) -> dict[str, Any]:
    smoke = _mapping(payload.get("source_faithful_preprocess_smoke"))
    if not smoke:
        return {
            "present": False,
            "source_faithful_preprocess_ready": False,
            "gradient_reachable": False,
            "exact_readiness_ready": False,
            "exact_readiness_blockers": [],
        }
    gradient_probe = _mapping(smoke.get("gradient_probe"))
    exact_readiness = _mapping(smoke.get("exact_readiness_refusal"))
    return {
        "present": True,
        "schema": smoke.get("schema"),
        "source_faithful_preprocess_ready": (
            smoke.get("source_faithful_preprocess_ready") is True
        ),
        "input_shape": _as_list(smoke.get("input_shape")),
        "camera_hw": _as_list(smoke.get("camera_hw")),
        "roundtrip_output_shape": _as_list(smoke.get("roundtrip_output_shape")),
        "yuv6_output_shape": _as_list(smoke.get("yuv6_output_shape")),
        "elapsed_seconds": _finite_float(smoke.get("elapsed_seconds")),
        "gradient_reachable": gradient_probe.get("gradient_reachable") is True,
        "gradient_probe_schema": gradient_probe.get("schema"),
        "max_abs_gradient": _finite_float(gradient_probe.get("max_abs_gradient")),
        "nonzero_gradient_count": _finite_int(
            gradient_probe.get("nonzero_gradient_count")
        ),
        "exact_readiness_ready": exact_readiness.get("ready") is True,
        "exact_readiness_blockers": _str_list(exact_readiness.get("blockers")),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _source_video_preprocess_signal(payload: Mapping[str, Any]) -> dict[str, Any]:
    smoke = _mapping(payload.get("source_video_preprocess_smoke"))
    if not smoke:
        return {
            "present": False,
            "source_video_loader_ready": False,
            "source_video_preprocess_ready": False,
            "gradient_reachable": False,
            "exact_readiness_ready": False,
            "exact_readiness_blockers": [],
        }
    gradient_probe = _mapping(smoke.get("gradient_probe"))
    exact_readiness = _mapping(smoke.get("exact_readiness_refusal"))
    return {
        "present": True,
        "schema": smoke.get("schema"),
        "source_video_loader_ready": smoke.get("source_video_loader_ready") is True,
        "source_video_preprocess_ready": (
            smoke.get("source_video_preprocess_ready") is True
        ),
        "video_path": smoke.get("video_path"),
        "video_sha256": smoke.get("video_sha256"),
        "upstream_dir": smoke.get("upstream_dir"),
        "pair_indices": _as_list(smoke.get("pair_indices")),
        "frame_indices": _as_list(smoke.get("frame_indices")),
        "source_frame_pair_shape": _as_list(smoke.get("source_frame_pair_shape")),
        "scorer_rgb_shape": _as_list(smoke.get("scorer_rgb_shape")),
        "yuv6_output_shape": _as_list(smoke.get("yuv6_output_shape")),
        "frame_reader_kind": smoke.get("frame_reader_kind"),
        "elapsed_seconds": _finite_float(smoke.get("elapsed_seconds")),
        "gradient_reachable": gradient_probe.get("gradient_reachable") is True,
        "gradient_probe_schema": gradient_probe.get("schema"),
        "max_abs_gradient": _finite_float(gradient_probe.get("max_abs_gradient")),
        "nonzero_gradient_count": _finite_int(
            gradient_probe.get("nonzero_gradient_count")
        ),
        "exact_readiness_ready": exact_readiness.get("ready") is True,
        "exact_readiness_blockers": _str_list(exact_readiness.get("blockers")),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _source_video_training_target_signal(payload: Mapping[str, Any]) -> dict[str, Any]:
    target = _mapping(payload.get("source_video_training_target"))
    if not target:
        return {
            "present": False,
            "source_video_target_loss_training": False,
            "exact_readiness_ready": False,
            "exact_readiness_blockers": [],
        }
    exact_readiness = _mapping(target.get("exact_readiness_refusal"))
    return {
        "present": True,
        "schema": target.get("schema"),
        "source_video_target_loss_training": True,
        "training_loss_surface": target.get("training_loss_surface"),
        "target_source": dict(_mapping(target.get("target_source"))),
        "target_source_kind": _mapping(target.get("target_source")).get("kind"),
        "target_shape_n2chw": _as_list(target.get("target_shape_n2chw")),
        "exact_readiness_ready": exact_readiness.get("ready") is True,
        "exact_readiness_blockers": _str_list(exact_readiness.get("blockers")),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _build_wire_in(
    payload: Mapping[str, Any],
    *,
    candidate_id: str,
    profile: str,
    lane_id: str,
    lane_class: str,
    candidate_family: str,
    representation_family: str,
    substrate_family: str,
    param_schema: str,
    params: Mapping[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "candidate_id": candidate_id,
        "profile_id": profile,
        "lane_id": lane_id,
        "lane_class": lane_class,
        "candidate_family": candidate_family,
        "representation_family": representation_family,
        "substrate_family": substrate_family,
        "training_signal_kind": str(
            payload.get("training_signal_kind") or "representation_training_probe"
        ),
        "param_schema": param_schema,
        "candidate_params": dict(params),
        "source_anchor": str(
            payload.get("source_anchor") or "representation training probe manifest"
        ),
        "score_lowering_hypothesis": str(
            payload.get("score_lowering_hypothesis")
            or "Use local training telemetry to rank representation variants before byte-closed exact eval."
        ),
        "dispatch_blockers": blockers,
    }
    optional_lists = {
        "canonical_equation_refs": _str_list(payload.get("canonical_equation_refs")),
        "master_gradient_features": _str_list(payload.get("master_gradient_features")),
        "xray_primitives": _str_list(payload.get("xray_primitives")),
        "deterministic_solution_refs": _str_list(payload.get("deterministic_solution_refs")),
        "variant_axes": _str_list(payload.get("variant_axes")),
        "paired_modes": _str_list(payload.get("paired_modes")),
    }
    kwargs.update({key: value for key, value in optional_lists.items() if value})
    return build_optimizer_training_signal_wire_in(**kwargs)


def adapt_representation_training_manifest_to_candidate(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Return one planning-only optimizer queue row for any training probe."""

    validate_representation_training_manifest(payload)
    stages = _stage_modules(payload)
    stage_count = _finite_int(payload.get("stage_count")) or len(stages)
    archive_zip = _mapping(payload.get("archive_zip"))
    bridge = _mapping(payload.get("auth_eval_bridge"))
    try:
        runtime_profile_summary = runtime_profile_summary_from_training_manifest(payload)
    except LocalTrainingRuntimeProfileError as exc:
        raise RepresentationTrainingProbeIntegrationError(str(exc)) from exc
    best_score = _best_training_score(payload)
    latest_score = _latest_training_score(payload)
    auth_score = _auth_eval_score(payload)
    has_archive = bool(archive_zip.get("sha256") and archive_zip.get("bytes"))
    has_auth_bridge = bool(bridge.get("ok") is True and bridge.get("auth_eval_json_sha256"))
    preprocess_signal = _source_faithful_preprocess_signal(payload)
    source_video_signal = _source_video_preprocess_signal(payload)
    source_video_target_signal = _source_video_training_target_signal(payload)
    blockers = _extra_blockers(payload)
    blockers.extend(_str_list(preprocess_signal.get("exact_readiness_blockers")))
    blockers.extend(_str_list(source_video_signal.get("exact_readiness_blockers")))
    blockers.extend(
        _str_list(source_video_target_signal.get("exact_readiness_blockers"))
    )
    blockers.extend(_str_list(runtime_profile_summary.get("blockers")))
    if not has_archive:
        blockers.append("representation_training_archive_export_missing")
    if not has_auth_bridge:
        blockers.append("representation_training_auth_eval_bridge_missing")
    if best_score is None:
        blockers.append("representation_training_best_score_missing")
    blockers = ordered_unique(blockers)

    candidate_id = str(payload["candidate_id"])
    representation_family = str(payload["representation_family"])
    substrate_family = str(payload["substrate_family"])
    lane_id = str(payload.get("lane_id") or DEFAULT_LANE_ID)
    lane_class = str(payload.get("lane_class") or f"{substrate_family}_training_proxy")
    candidate_family = str(payload.get("candidate_family") or f"{representation_family}_training_probe")
    profile = str(payload.get("profile") or "representation_training_probe")
    param_schema = str(payload.get("param_schema") or "representation_training_manifest_params_v1")
    params = _candidate_params(
        payload,
        runtime_profile_summary=runtime_profile_summary,
    )
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
    optimizer_recipe = _mapping(payload.get("optimizer_recipe"))
    scheduler_recipe = _mapping(payload.get("scheduler_recipe"))
    training_recipe = _mapping(payload.get("training_recipe"))

    consumer_payload = {
        "schema": CANDIDATE_PAYLOAD_SCHEMA,
        "representation_training_probe": {
            "representation_family": representation_family,
            "substrate_family": substrate_family,
            "stage_count": stage_count,
            "stage_modules": stages,
            "training_recipe": dict(training_recipe),
            "optimizer_recipe": dict(optimizer_recipe),
            "scheduler_recipe": dict(scheduler_recipe),
            "timing_smoke": {
                "wall_seconds": _finite_float(payload.get("wall_seconds")),
                "memory_peak_bytes": _finite_int(payload.get("memory_peak_bytes")),
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
            "source_faithful_preprocess": dict(preprocess_signal),
            "source_video_preprocess": dict(source_video_signal),
            "source_video_training_target": dict(source_video_target_signal),
            "missing_blockers": blockers,
            "source_tree_sha256": payload.get("source_tree_sha256"),
            "runtime_tree_sha256": payload.get("runtime_tree_sha256"),
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
        "lane_id": lane_id,
        "lane_class": lane_class,
        "candidate_family": candidate_family,
        "representation_family": representation_family,
        "substrate_family": substrate_family,
        "profile": profile,
        "param_schema": param_schema,
        "candidate_params": params,
        "op_params": params,
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
        "device_selected": payload.get("device_selected") or payload.get("device_requested"),
        "source_tree_sha256": payload.get("source_tree_sha256"),
        "runtime_tree_sha256": payload.get("runtime_tree_sha256"),
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
        "solver_stack_wire_in": _build_wire_in(
            payload,
            candidate_id=candidate_id,
            profile=profile,
            lane_id=lane_id,
            lane_class=lane_class,
            candidate_family=candidate_family,
            representation_family=representation_family,
            substrate_family=substrate_family,
            param_schema=param_schema,
            params=params,
            blockers=blockers,
        ),
        "evidence_semantics": "representation_training_probe_proxy_not_exact_auth_eval",
        "evidence_grade": str(payload.get("evidence_grade") or "local_training_probe_advisory"),
    }
    return apply_proxy_evidence_boundary(row, dispatch_blockers=blockers)


__all__ = [
    "CANDIDATE_PAYLOAD_SCHEMA",
    "DEFAULT_LANE_ID",
    "PLAN_SCHEMA",
    "REPRESENTATION_TRAINING_BLOCKERS",
    "REQUIRED_FALSE_AUTHORITY_FIELDS",
    "SCHEMA",
    "RepresentationTrainingProbeIntegrationError",
    "adapt_representation_training_manifest_to_candidate",
    "validate_representation_training_manifest",
]
