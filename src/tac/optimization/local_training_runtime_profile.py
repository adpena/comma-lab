# SPDX-License-Identifier: MIT
"""Planning-only runtime profile contract for local training substrates.

Local CPU/MLX/MPS/CUDA timing observations are valuable because they change
which expensive representation-training runs are worth doing locally before
cloud anchoring. They are not score evidence. This module turns those timing
and backend-kernel observations into typed queue rows while preserving the same
false-authority boundary used by optimizer and MLX sweep artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)

SCHEMA = "trainer_runtime_profile_observation.v1"
CANDIDATE_PAYLOAD_SCHEMA = "trainer_runtime_profile_candidate_payload.v1"

RUNTIME_PROFILE_BLOCKERS: tuple[str, ...] = (
    "trainer_runtime_profile_is_cost_signal_not_score",
    "requires_paired_same_seed_quality_observation_before_candidate_rank",
    "requires_byte_closed_archive_export_before_dispatch_readiness",
    "requires_packet_compiler_or_archive_export_gate_before_exact_eval",
    "requires_exact_auth_eval_result_before_score_claim",
)

SUPPORTED_LOCAL_BACKENDS: frozenset[str] = frozenset(
    {"cpu", "mlx", "mps", "cuda", "torch", "unknown"}
)

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "dispatch_packet_ready": False,
}


class LocalTrainingRuntimeProfileError(ValueError):
    """Raised when runtime telemetry is malformed or over-authoritative."""


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


def _normalized_backend(payload: Mapping[str, Any]) -> str:
    backend = str(
        payload.get("training_backend")
        or payload.get("backend")
        or payload.get("device_selected")
        or payload.get("device")
        or "unknown"
    ).lower()
    if backend.startswith("mlx"):
        return "mlx"
    if backend.startswith("mps"):
        return "mps"
    if backend.startswith("cuda"):
        return "cuda"
    if backend.startswith("cpu"):
        return "cpu"
    return backend if backend in SUPPORTED_LOCAL_BACKENDS else "unknown"


def _primary_timing_metric(payload: Mapping[str, Any]) -> tuple[str | None, float | None]:
    for key in ("seconds_per_epoch", "seconds_per_candidate", "seconds_per_step"):
        value = _finite_float(payload.get(key))
        if value is not None and value > 0.0:
            return key, value
    timing = _mapping(payload.get("timing"))
    for key in ("seconds_per_epoch", "seconds_per_candidate", "seconds_per_step"):
        value = _finite_float(timing.get(key))
        if value is not None and value > 0.0:
            return key, value
    return None, None


def _kernel_fusion_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    fusion = dict(_mapping(payload.get("kernel_fusion")))
    strategy_id = str(
        payload.get("kernel_fusion_strategy_id")
        or fusion.get("strategy_id")
        or fusion.get("kernel_fusion_strategy_id")
        or "none"
    )
    backend_contract = _mapping(
        payload.get("backend_kernel_contract") or fusion.get("backend_kernel_contract")
    )
    operator_mix = _mapping(payload.get("operator_mix") or fusion.get("operator_mix"))
    numerical_drift = _mapping(
        payload.get("numerical_drift_profile") or fusion.get("numerical_drift_profile")
    )
    return {
        "kernel_fusion_strategy_id": strategy_id,
        "backend_kernel_contract": dict(backend_contract),
        "eligible_patterns": _str_list(
            payload.get("eligible_patterns") or fusion.get("eligible_patterns")
        ),
        "operator_mix": dict(operator_mix),
        "numerical_drift_profile": dict(numerical_drift),
        "ineligible_reason": payload.get("ineligible_reason")
        or fusion.get("ineligible_reason"),
        "measured": bool(
            payload.get("kernel_fusion_measured", fusion.get("measured", False))
        ),
    }


def _packet_compiler_bridge(payload: Mapping[str, Any]) -> dict[str, Any]:
    bridge = _mapping(
        payload.get("packet_compiler_bridge")
        or payload.get("archive_export_gate")
        or payload.get("compiler_bridge")
    )
    return {
        "packet_compiler_target_declared": bool(
            bridge.get("packet_compiler_target_declared", False)
        ),
        "archive_export_schema": bridge.get("archive_export_schema"),
        "archive_export_tool": bridge.get("archive_export_tool"),
        "runtime_consumption_proof_required": bool(
            bridge.get("runtime_consumption_proof_required", True)
        ),
        "runtime_consumption_proof_present": bool(
            bridge.get("runtime_consumption_proof_present", False)
        ),
        "blockers": ordered_unique(str(item) for item in _as_list(bridge.get("blockers"))),
    }


def _local_cloud_substitution(payload: Mapping[str, Any]) -> dict[str, Any]:
    substitution = _mapping(
        payload.get("local_cloud_substitution")
        or payload.get("cloud_replacement")
        or payload.get("cloud_gpu_reference")
    )
    return {
        "intended_to_replace_cloud_gpu_training": bool(
            payload.get(
                "intended_to_replace_cloud_gpu_training",
                substitution.get("intended_to_replace_cloud_gpu_training", False),
            )
        ),
        "cloud_gpu_reference": substitution.get("cloud_gpu_reference")
        or substitution.get("reference"),
        "local_cost_usd": _finite_float(
            payload.get("local_cost_usd", substitution.get("local_cost_usd"))
        ),
        "cloud_cost_usd": _finite_float(
            payload.get("cloud_cost_usd", substitution.get("cloud_cost_usd"))
        ),
        "estimated_cloud_cost_saved_usd": _finite_float(
            payload.get(
                "estimated_cloud_cost_saved_usd",
                substitution.get("estimated_cloud_cost_saved_usd"),
            )
        ),
    }


def validate_runtime_profile_observation(payload: Mapping[str, Any]) -> None:
    """Validate one runtime observation without granting score authority."""

    if payload.get("schema") != SCHEMA:
        raise LocalTrainingRuntimeProfileError(f"expected schema {SCHEMA}")
    try:
        require_no_truthy_authority_fields(
            payload,
            context="trainer_runtime_profile_observation",
        )
    except ValueError as exc:
        raise LocalTrainingRuntimeProfileError(str(exc)) from exc
    for key, expected in FALSE_AUTHORITY.items():
        if key in payload and payload.get(key) is not expected:
            raise LocalTrainingRuntimeProfileError(f"{key} must be explicit false")
    if not str(payload.get("candidate_id") or payload.get("profile_id") or "").strip():
        raise LocalTrainingRuntimeProfileError("candidate_id or profile_id is required")
    _metric, timing_value = _primary_timing_metric(payload)
    if timing_value is None:
        raise LocalTrainingRuntimeProfileError(
            "seconds_per_epoch, seconds_per_candidate, or seconds_per_step is required"
        )


def normalize_runtime_profile_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical JSON-safe runtime profile summary."""

    validate_runtime_profile_observation(payload)
    backend = _normalized_backend(payload)
    timing_field, timing_value = _primary_timing_metric(payload)
    assert timing_field is not None and timing_value is not None
    peak_memory = _finite_int(
        payload.get("peak_memory_bytes")
        or _mapping(payload.get("timing")).get("peak_memory_bytes")
    )
    packet_bridge = _packet_compiler_bridge(payload)
    profile_blockers = ordered_unique(
        [
            *RUNTIME_PROFILE_BLOCKERS,
            *(
                ["local_mlx_training_profile_not_score_authority"]
                if backend == "mlx"
                else []
            ),
            *(
                ["packet_compiler_target_missing"]
                if not packet_bridge["packet_compiler_target_declared"]
                else []
            ),
            *(
                ["runtime_consumption_proof_missing"]
                if packet_bridge["runtime_consumption_proof_required"]
                and not packet_bridge["runtime_consumption_proof_present"]
                else []
            ),
            *packet_bridge["blockers"],
            *[
                str(item)
                for item in _as_list(payload.get("dispatch_blockers"))
                if str(item)
            ],
        ]
    )
    return {
        "schema": SCHEMA,
        "profile_id": str(payload.get("profile_id") or "local_training_runtime_profile"),
        "candidate_id": str(payload.get("candidate_id") or payload.get("profile_id")),
        "lane_id": payload.get("lane_id"),
        "representation_family": payload.get("representation_family"),
        "substrate_family": payload.get("substrate_family"),
        "training_backend": backend,
        "device": payload.get("device") or payload.get("device_selected") or backend,
        "hardware_substrate": payload.get("hardware_substrate"),
        "seed": _finite_int(payload.get("seed")),
        "stage_id": payload.get("stage_id"),
        "stage_index": _finite_int(payload.get("stage_index")),
        "timing_field": timing_field,
        "timing_value_seconds": timing_value,
        "seconds_per_epoch": _finite_float(payload.get("seconds_per_epoch")),
        "seconds_per_candidate": _finite_float(payload.get("seconds_per_candidate")),
        "seconds_per_step": _finite_float(payload.get("seconds_per_step")),
        "examples_per_second": _finite_float(payload.get("examples_per_second")),
        "peak_memory_bytes": peak_memory,
        "state_bytes": _finite_int(payload.get("state_bytes")),
        "kernel_fusion": _kernel_fusion_context(payload),
        "packet_compiler_bridge": packet_bridge,
        "local_cloud_substitution": _local_cloud_substitution(payload),
        "blockers": profile_blockers,
        **FALSE_AUTHORITY,
    }


def runtime_profiles_from_training_manifest(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract normalized runtime profiles embedded in a training manifest."""

    try:
        require_no_truthy_authority_fields(
            payload,
            context="training_manifest_runtime_profile_root",
        )
    except ValueError as exc:
        raise LocalTrainingRuntimeProfileError(str(exc)) from exc

    raw_profiles: list[Mapping[str, Any]] = []
    single = payload.get("runtime_profile")
    if isinstance(single, Mapping):
        raw_profiles.append(single)
    for item in _as_list(payload.get("runtime_profiles")):
        if isinstance(item, Mapping):
            raw_profiles.append(item)
    profiles: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_profiles):
        enriched = {
            "candidate_id": (
                payload.get("candidate_id")
                or raw.get("candidate_id")
                or payload.get("profile")
                or payload.get("lane_id")
                or "training_manifest_runtime_profile"
            ),
            "lane_id": payload.get("lane_id") or raw.get("lane_id"),
            "representation_family": payload.get("representation_family")
            or raw.get("representation_family"),
            "substrate_family": payload.get("substrate_family")
            or raw.get("substrate_family"),
            **dict(raw),
        }
        if "schema" not in enriched:
            enriched["schema"] = SCHEMA
        try:
            profiles.append(normalize_runtime_profile_observation(enriched))
        except LocalTrainingRuntimeProfileError as exc:
            raise LocalTrainingRuntimeProfileError(
                f"runtime_profiles[{index}]: {exc}"
            ) from exc
    return profiles


def runtime_profile_summary_from_training_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an aggregate runtime-cost context for representation manifests."""

    profiles = runtime_profiles_from_training_manifest(payload)
    if not profiles:
        return {
            "schema": "trainer_runtime_profile_summary.v1",
            "profile_count": 0,
            "best_local_backend": None,
            "best_timing_field": None,
            "best_timing_value_seconds": None,
            "kernel_fusion_strategy_ids": [],
            "operator_mix_keys": [],
            "blockers": ["local_training_runtime_profile_missing"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        }
    best = min(
        profiles,
        key=lambda row: (
            float(row["timing_value_seconds"]),
            str(row["training_backend"]),
            str(row["candidate_id"]),
        ),
    )
    operator_mix_keys: list[str] = []
    for profile in profiles:
        kernel = _mapping(profile.get("kernel_fusion"))
        operator_mix_keys.extend(str(key) for key in _mapping(kernel.get("operator_mix")))
    return {
        "schema": "trainer_runtime_profile_summary.v1",
        "profile_count": len(profiles),
        "best_local_backend": best["training_backend"],
        "best_timing_field": best["timing_field"],
        "best_timing_value_seconds": best["timing_value_seconds"],
        "kernel_fusion_strategy_ids": ordered_unique(
            str(_mapping(profile.get("kernel_fusion")).get("kernel_fusion_strategy_id"))
            for profile in profiles
        ),
        "operator_mix_keys": ordered_unique(operator_mix_keys),
        "profiles": profiles,
        "blockers": ordered_unique(
            blocker
            for profile in profiles
            for blocker in _as_list(profile.get("blockers"))
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
    }


def adapt_runtime_profile_observation_to_candidate(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Return one candidate-queue row from a standalone runtime profile."""

    profile = normalize_runtime_profile_observation(payload)
    candidate_id = str(profile["candidate_id"])
    backend = str(profile["training_backend"])
    timing_field = str(profile["timing_field"])
    row = {
        "candidate_id": f"{candidate_id}::runtime_profile::{backend}",
        "source_candidate_id": candidate_id,
        "source_paths": [_repo_rel(source_path, repo_root)],
        "lane_id": profile.get("lane_id") or "local_training_runtime_profile",
        "lane_class": "local_training_runtime_profile",
        "candidate_family": "local_training_runtime_profile_cost_model",
        "representation_family": profile.get("representation_family"),
        "substrate_family": profile.get("substrate_family"),
        "training_signal_kind": "local_training_runtime_cost_profile",
        "profile": profile["profile_id"],
        "param_schema": "trainer_runtime_profile_params_v1",
        "candidate_params": {
            "training_backend": backend,
            "device": profile.get("device"),
            "seed": profile.get("seed"),
            "stage_id": profile.get("stage_id"),
            "kernel_fusion_strategy_id": profile["kernel_fusion"][
                "kernel_fusion_strategy_id"
            ],
        },
        "rank_score": profile["timing_value_seconds"],
        "rank_score_field": f"{timing_field}_cost_signal_not_score",
        "evidence_semantics": "local_training_runtime_profile_cost_signal_not_score",
        "evidence_grade": (
            "[macOS-MLX research-signal]"
            if backend == "mlx"
            else "[local-training-runtime-profile]"
        ),
        "consumer_payload": {
            "schema": CANDIDATE_PAYLOAD_SCHEMA,
            "runtime_profile": profile,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
        },
    }
    return apply_proxy_evidence_boundary(row, dispatch_blockers=profile["blockers"])


__all__ = [
    "CANDIDATE_PAYLOAD_SCHEMA",
    "FALSE_AUTHORITY",
    "RUNTIME_PROFILE_BLOCKERS",
    "SCHEMA",
    "LocalTrainingRuntimeProfileError",
    "adapt_runtime_profile_observation_to_candidate",
    "normalize_runtime_profile_observation",
    "runtime_profile_summary_from_training_manifest",
    "runtime_profiles_from_training_manifest",
    "validate_runtime_profile_observation",
]
