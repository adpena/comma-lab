# SPDX-License-Identifier: MIT
"""Convert harvested local-training queues into reusable planning intelligence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.optimizer_scheduler_registry import (
    TELEMETRY_SCHEMA,
    OptimizerSchedulerRegistry,
    OptimizerSchedulerRegistryError,
    build_optimizer_scheduler_telemetry_record,
    default_optimizer_scheduler_registry,
)
from tac.optimization.optimizer_signal_atoms import build_optimizer_signal_atom_ledger
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)

LOCAL_TRAINING_HARVEST_INTELLIGENCE_SCHEMA = (
    "local_training_optimizer_harvest_intelligence.v1"
)
OPTIMIZER_SCHEDULER_TELEMETRY_LEDGER_SCHEMA = (
    "optimizer_scheduler_telemetry_ledger.v1"
)
SUPPORTED_HARVEST_QUEUE_SCHEMA = "optimizer_candidate_queue_v1"
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


class LocalTrainingHarvestIntelligenceError(ValueError):
    """Raised when harvested local-training signal is unsafe or malformed."""


def _repo_rel(path: str | Path | None, *, repo_root: Path) -> str | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve(strict=False).relative_to(
            repo_root.resolve(strict=False)
        ).as_posix()
    except ValueError:
        return candidate.as_posix()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _int_or_default(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _require_queue(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SUPPORTED_HARVEST_QUEUE_SCHEMA:
        raise LocalTrainingHarvestIntelligenceError(
            f"expected {SUPPORTED_HARVEST_QUEUE_SCHEMA}"
        )
    try:
        require_no_truthy_authority_fields(
            payload,
            context="local_training_harvest_intelligence_queue",
        )
    except ValueError as exc:
        raise LocalTrainingHarvestIntelligenceError(str(exc)) from exc
    rows = payload.get("top_k")
    if not isinstance(rows, list):
        raise LocalTrainingHarvestIntelligenceError("optimizer queue top_k must be a list")


def _runtime_profiles_from_row(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    probe = _mapping(_mapping(row.get("consumer_payload")).get("representation_training_probe"))
    timing = _mapping(probe.get("timing_smoke"))
    summary = _mapping(timing.get("runtime_profile_summary"))
    profiles = _list_of_mappings(summary.get("profiles"))
    if profiles:
        return profiles
    single = _mapping(probe.get("runtime_profile"))
    return [single] if single else []


def _timing_kwargs(profile: Mapping[str, Any]) -> dict[str, float | None]:
    field = str(profile.get("timing_field") or "")
    value = _float_or_none(profile.get("timing_value_seconds"))
    seconds_per_epoch = _float_or_none(profile.get("seconds_per_epoch"))
    seconds_per_candidate = _float_or_none(profile.get("seconds_per_candidate"))
    seconds_per_step = _float_or_none(profile.get("seconds_per_step"))
    if field == "seconds_per_epoch" and seconds_per_epoch is None:
        seconds_per_epoch = value
    if field == "seconds_per_candidate" and seconds_per_candidate is None:
        seconds_per_candidate = value
    if field == "seconds_per_step" and seconds_per_step is None:
        seconds_per_step = value
    return {
        "seconds_per_epoch": seconds_per_epoch,
        "seconds_per_candidate": seconds_per_candidate,
        "seconds_per_step": seconds_per_step,
    }


def _profile_axis_tag(
    profile: Mapping[str, Any],
    *,
    allowed_axis_tags: Sequence[str],
) -> str:
    for key in ("evidence_grade", "evidence_tag", "axis_tag"):
        value = str(profile.get(key) or "")
        if value in allowed_axis_tags:
            return value
    if "[macOS-MLX research-signal]" in allowed_axis_tags:
        return "[macOS-MLX research-signal]"
    return allowed_axis_tags[0]


def _archive_ready(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("candidate_archive_sha256")
        or row.get("archive_sha256")
        or row.get("candidate_archive_path")
        or row.get("archive_path")
    )


def _export_ready(profile: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    bridge = _mapping(profile.get("packet_compiler_bridge"))
    candidate_params = _mapping(row.get("candidate_params"))
    return bool(
        bridge.get("runtime_consumption_proof_present")
        or candidate_params.get("pr95_runtime_consumption_proof_present")
    )


def build_optimizer_scheduler_telemetry_from_harvest_queue(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    repo_root: str | Path = ".",
    registry: OptimizerSchedulerRegistry | None = None,
) -> dict[str, Any]:
    """Build typed scheduler telemetry from harvested local-training rows."""

    _require_queue(payload)
    repo = Path(repo_root)
    active_registry = registry or default_optimizer_scheduler_registry()
    records: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row_index, row_any in enumerate(payload.get("top_k") or []):
        if not isinstance(row_any, Mapping):
            refusals.append({"row_index": row_index, "reason": "row_not_mapping"})
            continue
        row = row_any
        try:
            require_no_truthy_authority_fields(
                row,
                context=f"local_training_harvest_intelligence.top_k[{row_index}]",
            )
        except ValueError as exc:
            raise LocalTrainingHarvestIntelligenceError(str(exc)) from exc
        params = _mapping(row.get("candidate_params"))
        descriptor_id = str(params.get("optimizer_descriptor_id") or "")
        if not descriptor_id:
            refusals.append(
                {"row_index": row_index, "candidate_id": row.get("candidate_id"), "reason": "optimizer_descriptor_id_missing"}
            )
            continue
        try:
            descriptor = active_registry.get(descriptor_id)
        except OptimizerSchedulerRegistryError as exc:
            refusals.append(
                {"row_index": row_index, "candidate_id": row.get("candidate_id"), "reason": str(exc)}
            )
            continue
        expected_sha = params.get("optimizer_config_sha256")
        if expected_sha and expected_sha != descriptor.config_sha256:
            refusals.append(
                {
                    "row_index": row_index,
                    "candidate_id": row.get("candidate_id"),
                    "reason": "optimizer_config_sha256_mismatch",
                    "expected": descriptor.config_sha256,
                    "actual": expected_sha,
                }
            )
            continue
        for profile_index, profile in enumerate(_runtime_profiles_from_row(row)):
            timing_kwargs = _timing_kwargs(profile)
            if not any(value is not None for value in timing_kwargs.values()):
                refusals.append(
                    {
                        "row_index": row_index,
                        "profile_index": profile_index,
                        "candidate_id": row.get("candidate_id"),
                        "reason": "timing_metric_missing",
                    }
                )
                continue
            seed = _int_or_default(profile.get("seed"), _int_or_default(params.get("seed"), 0))
            kernel = _mapping(profile.get("kernel_fusion"))
            bridge = _mapping(profile.get("packet_compiler_bridge"))
            blockers = ordered_unique(
                [
                    *[str(item) for item in row.get("dispatch_blockers") or [] if str(item)],
                    *[str(item) for item in profile.get("blockers") or [] if str(item)],
                    *[str(item) for item in bridge.get("blockers") or [] if str(item)],
                ]
            )
            metadata = {
                "source_queue_path": _repo_rel(source_path, repo_root=repo),
                "source_candidate_id": row.get("candidate_id"),
                "source_paths": row.get("source_paths") or [],
                "row_rank_score": row.get("rank_score"),
                "row_rank_score_field": row.get("rank_score_field"),
                "stage_index": params.get("stage_index") or profile.get("stage_index"),
                "stage_module": params.get("stage_module") or profile.get("stage_id"),
                "base_channels": params.get("base_channels"),
                "latent_dim": params.get("latent_dim"),
                "runtime_profile_candidate_id": profile.get("candidate_id"),
                "runtime_profile_id": profile.get("profile_id"),
                "runtime_profile_blockers": list(profile.get("blockers") or []),
                "local_cloud_substitution": dict(
                    _mapping(profile.get("local_cloud_substitution"))
                ),
                **FALSE_AUTHORITY,
            }
            record = build_optimizer_scheduler_telemetry_record(
                descriptor=descriptor,
                axis_tag=_profile_axis_tag(
                    profile,
                    allowed_axis_tags=descriptor.allowed_axis_tags,
                ),
                substrate=descriptor.substrate,
                seed=seed,
                seed_budget=1,
                slice_budget=max(1, _int_or_default(params.get("steps"), 1)),
                state_bytes=max(0, _int_or_default(profile.get("state_bytes"), 0)),
                backend=profile.get("training_backend") or profile.get("device"),
                kernel_fusion_strategy_id=kernel.get("kernel_fusion_strategy_id"),
                backend_kernel_contract=dict(
                    _mapping(kernel.get("backend_kernel_contract"))
                ),
                operator_mix=dict(_mapping(kernel.get("operator_mix"))),
                numerical_drift_profile=dict(
                    _mapping(kernel.get("numerical_drift_profile"))
                ),
                ineligible_reason=kernel.get("ineligible_reason"),
                archive_ready=_archive_ready(row),
                export_ready=_export_ready(profile, row),
                archive_export_blockers=blockers,
                metadata=metadata,
                **timing_kwargs,
            )
            identity = (
                record["descriptor_id"],
                record["config_sha256"],
                record["axis_tag"],
                record["seed"],
                record.get("seconds_per_epoch"),
                record.get("seconds_per_candidate"),
                record.get("seconds_per_step"),
                record.get("state_bytes"),
                record["metadata"].get("source_candidate_id"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            records.append(record)
    return {
        "schema": OPTIMIZER_SCHEDULER_TELEMETRY_LEDGER_SCHEMA,
        "record_schema": TELEMETRY_SCHEMA,
        "source_schema": str(payload.get("schema") or ""),
        "source_path": _repo_rel(source_path, repo_root=repo),
        "record_count": len(records),
        "refusal_count": len(refusals),
        "refusals": refusals,
        "records": records,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_local_training_harvest_intelligence(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    repo_root: str | Path = ".",
    max_atoms: int | None = None,
) -> dict[str, Any]:
    """Build atoms and scheduler telemetry from a harvested candidate queue."""

    _require_queue(payload)
    atom_ledger = build_optimizer_signal_atom_ledger(
        payload,
        source_path=_repo_rel(source_path, repo_root=Path(repo_root)),
        max_atoms=max_atoms,
    )
    telemetry_ledger = build_optimizer_scheduler_telemetry_from_harvest_queue(
        payload,
        source_path=source_path,
        repo_root=repo_root,
    )
    return {
        "schema": LOCAL_TRAINING_HARVEST_INTELLIGENCE_SCHEMA,
        "source_schema": str(payload.get("schema") or ""),
        "source_path": _repo_rel(source_path, repo_root=Path(repo_root)),
        "atom_ledger_schema": atom_ledger["schema"],
        "atom_count": atom_ledger["atom_count"],
        "telemetry_ledger_schema": telemetry_ledger["schema"],
        "telemetry_record_count": telemetry_ledger["record_count"],
        "telemetry_refusal_count": telemetry_ledger["refusal_count"],
        "optimizer_signal_atom_ledger": atom_ledger,
        "optimizer_scheduler_telemetry": telemetry_ledger,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_semantics": "harvested_local_training_signal_planning_only",
    }


__all__ = [
    "FALSE_AUTHORITY",
    "LOCAL_TRAINING_HARVEST_INTELLIGENCE_SCHEMA",
    "OPTIMIZER_SCHEDULER_TELEMETRY_LEDGER_SCHEMA",
    "SUPPORTED_HARVEST_QUEUE_SCHEMA",
    "LocalTrainingHarvestIntelligenceError",
    "build_local_training_harvest_intelligence",
    "build_optimizer_scheduler_telemetry_from_harvest_queue",
]
