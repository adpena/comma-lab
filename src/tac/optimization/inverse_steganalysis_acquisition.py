# SPDX-License-Identifier: MIT
"""Planning-only inverse-steganalysis acquisition surface.

The rows here are scorer-in-the-loop search signal, not score authority. They
encode multiscale atoms plus local/proxy calibration observations so schedulers
can rank next probes by expected score gain per second, GB, and resource kind.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.local_acceleration.mlx_acquisition_batch import (
    validate_mlx_acquisition_batch,
)
from tac.optimization.byte_shaving_campaign import (
    COUPLED_OPERATION_SET_SCHEMA as BYTE_SHAVING_OPERATION_SET_SCHEMA,
)
from tac.optimization.byte_shaving_campaign import (
    PLAN_SCHEMA as BYTE_SHAVING_CAMPAIGN_PLAN_SCHEMA,
)
from tac.optimization.byte_shaving_campaign import (
    build_byte_shaving_campaign_plan,
)
from tac.optimization.byte_shaving_campaign import (
    validate_signal_surface as validate_byte_shaving_signal_surface,
)
from tac.optimization.candidate_evidence_contract import is_sha256_hex
from tac.optimization.proxy_candidate_contract import (
    CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    CONTEST_AUTH_SCORE_AXES,
    CONTEST_AUTH_SCORE_AXIS_PREFIXES,
    PROXY_FALSE_AUTHORITY_FIELDS,
    apply_proxy_evidence_boundary,
    ordered_unique,
    truthy_authority_field_violations,
)
from tac.score_composition import (
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
)

SCHEMA = "inverse_steganalysis_acquisition_plan.v1"
ATOM_SCHEMA = "inverse_steganalysis_atom.v1"
OBSERVATION_SCHEMA = "inverse_steganalysis_observation.v1"
EXACT_AUTH_CALIBRATION_SCHEMA = "inverse_steganalysis_exact_auth_calibration.v1"
QUEUE_PERFORMANCE_SUMMARY_SCHEMA = "experiment_queue_performance_summary.v1"
QUEUE_OBSERVATION_SCHEMA = "experiment_queue_observation.v1"
PRIORITY_SCHEMA = "inverse_steganalysis_acquisition_priority.v1"
ACTION_FUNCTIONAL_SCHEMA = "inverse_steganalysis_discrete_action_functional.v1"
ACTION_CELL_SCHEMA = "inverse_steganalysis_action_cell.v1"
QUEUE_HEALTH_FEEDBACK_SCHEMA = "inverse_steganalysis_queue_health_feedback.v1"
MATERIALIZER_ARCHIVE_DELTA_FEEDBACK_SCHEMA = "inverse_steganalysis_materializer_archive_delta_feedback.v1"
INVERSE_SCORER_SURFACE_SCHEMA = "scorer_inverse_decision_surface.v1"
BYTE_SHAVING_OPERATION_SET_PROVENANCE_SCHEMA = "inverse_steganalysis_byte_shaving_operation_set_provenance.v1"
BYTE_SHAVING_UNIT_PROVENANCE_SCHEMA = "inverse_steganalysis_byte_shaving_ranked_unit_provenance.v1"
MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA = "mlx_effective_spend_triage_candidate_selection.v1"
MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_ROW_SCHEMA = "mlx_effective_spend_triage_candidate_row.v1"
MLX_EFFECTIVE_SPEND_TRIAGE_PROVENANCE_SCHEMA = "inverse_steganalysis_mlx_effective_spend_triage_row_provenance.v1"
MLX_ACQUISITION_BATCH_PROVENANCE_SCHEMA = "inverse_steganalysis_mlx_acquisition_batch_operation_set_provenance.v1"
TOOL = "tac.optimization.inverse_steganalysis_acquisition"
CONTEST_RATE_DENOM_BYTES = CANONICAL_RATE_DENOM_BYTES
CONTEST_RATE_SCORE_PER_BYTE = CANONICAL_RATE_MULTIPLIER / float(CANONICAL_RATE_DENOM_BYTES)
MLX_EVIDENCE_GRADE = "macOS-MLX-research-signal"
MLX_EVIDENCE_TAG = "[macOS-MLX research-signal]"
QUEUE_HEALTH_BLOCKER_OBSERVATION_KINDS = frozenset(
    {
        "queue_observation_health_blocker",
        "queue_observation_global_health_blocker",
    }
)
MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND = "materializer_chain_archive_delta"
FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA = "family_agnostic_materializer_empirical_observation.v1"
FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA = "family_agnostic_materializer_empirical_sweep.v1"
FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_KIND = "family_agnostic_materializer_empirical_observation"
MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KINDS = frozenset(
    {
        MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND,
        FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_KIND,
    }
)
QUEUE_MATERIALIZER_OBSERVATION_ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
QUEUE_MATERIALIZER_OBSERVATION_ARTIFACT_MAX_ROWS = 4096

ALLOWED_SCALES = frozenset(
    {
        "candidate",
        "frame",
        "pair",
        "frame_pair",
        "region",
        "frequency",
        "byte",
        "component",
        "coherence",
        "region_frequency",
        "byte_range",
        "multiscale",
    }
)
SCOPE_AXES = frozenset(
    {
        "bytes",
        "pixels",
        "regions",
        "boundaries",
        "frames",
        "pairs",
        "batches",
        "full_video",
    }
)
SCALE_TO_SCOPE_AXIS = {
    "byte": "bytes",
    "byte_range": "bytes",
    "region": "regions",
    "region_frequency": "regions",
    "frame": "frames",
    "frame_pair": "pairs",
    "pair": "pairs",
    "frequency": "pixels",
    "component": "full_video",
    "coherence": "batches",
    "candidate": "full_video",
    "multiscale": "full_video",
}
RUNTIME_IDENTITY_KEYS = frozenset(
    {
        "runtime_content_tree_sha256",
        "runtime_tree_sha256",
        "runtime_manifest_sha256",
        "runtime_sha256",
        "inflate_runtime_sha256",
        "inflate_script_sha256",
        "scorer_runtime_sha256",
        "scorer_version",
        "runtime_contract_sha256",
    }
)
CACHE_IDENTITY_KEYS = frozenset(
    {
        "cache_sha256",
        "cache_key",
        "input_cache_sha256",
        "raw_sha256",
        "inflated_outputs_aggregate_sha256",
        "array_sha256",
        "candidate_cache_array_sha256",
        "reference_cache_array_sha256",
        "pair_indices_sha256",
    }
)
AUTHORITY_FIELDS = tuple(
    dict.fromkeys(
        (
            *CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS,
            *PROXY_FALSE_AUTHORITY_FIELDS.keys(),
        )
    )
)
RESOURCE_MULTIPLIERS: dict[str, float] = {
    "local_mlx": 1.0,
    "macos_mlx": 1.0,
    "macos_mlx_research_signal": 1.0,
    "local_cpu": 1.25,
    "macos_cpu_advisory": 1.25,
    "local_io_heavy": 1.5,
    "local_gpu": 2.0,
    "remote_cpu": 3.0,
    "modal_t4": 4.0,
    "remote_gpu": 5.0,
    "exact_auth_calibration": 8.0,
    "contest_exact_eval": 8.0,
}
DEFAULT_RESOURCE_MULTIPLIER = 2.0
MIN_ELAPSED_SECONDS = 1.0
MIN_ARTIFACT_GB = 1.0e-6


class InverseSteganalysisAcquisitionError(ValueError):
    """Raised when inverse-steganalysis planning rows are malformed."""


@dataclass(frozen=True)
class AcquisitionPriorityTerms:
    """Resource-normalized score-lowering priority terms."""

    predicted_score_gain: float
    expected_score_gain: float
    uncertainty_bonus: float
    calibration_penalty: float
    elapsed_seconds: float
    artifact_bytes: int
    artifact_gb: float
    resource_kind: str
    resource_multiplier: float
    score_gain_per_second: float
    score_gain_per_gb: float
    acquisition_priority: float

    def to_dict(self) -> dict[str, Any]:
        return {"schema": PRIORITY_SCHEMA, **self.__dict__}


def normalize_inverse_steganalysis_atom(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a multiscale inverse-scorer atom and force proxy authority."""

    _reject_truthy_authority(row, label="inverse-steganalysis atom")
    scale = _scale(row.get("scale"))
    effects = _predicted_effects(row)
    action_terms = _action_surface_terms(row, effects)
    out = {
        "schema": ATOM_SCHEMA,
        "atom_id": _text(row.get("atom_id"), "atom_id"),
        "candidate_id": _text(row.get("candidate_id"), "candidate_id"),
        "scale": scale,
        "scope_axis": _scope_axis(row.get("scope_axis"), scale),
        "parent_unit_id": _optional_text(row.get("parent_unit_id")),
        "frame_range": _range(row.get("frame_range"), "frame_range"),
        "pair_indices": _int_list(row.get("pair_indices"), "pair_indices"),
        "region_bbox": _bbox(row.get("region_bbox")),
        "frequency_band": _optional_text(row.get("frequency_band")),
        "byte_range": _range(row.get("byte_range"), "byte_range"),
        "component": _text(row.get("component"), "component"),
        "coherence_group": _optional_text(row.get("coherence_group")),
        "sparsity_prior": _float(row.get("sparsity_prior", 0.0), "sparsity_prior", minimum=0.0),
        **effects,
        **action_terms,
        "uncertainty": _float(
            row.get("uncertainty", row.get("prediction_uncertainty", 0.0)),
            "uncertainty",
            minimum=0.0,
        ),
        "calibration_error": _float(row.get("calibration_error", 0.0), "calibration_error", minimum=0.0),
        "elapsed_seconds": _float_or_none(row.get("elapsed_seconds"), "elapsed_seconds", minimum=0.0, exclusive=True),
        "artifact_bytes": _int(
            row.get("artifact_bytes", row.get("source_artifact_bytes", 0)), "artifact_bytes", minimum=0
        ),
        "resource_kind": _resource(row.get("resource_kind", "local_cpu")),
        "source_provenance": _optional_mapping(row.get("source_provenance"), "source_provenance"),
        "operation_set_compiler": _optional_mapping(
            row.get("operation_set_compiler"),
            "operation_set_compiler",
        ),
        "operation_set_target_kind": _optional_text(
            _first(row.get("operation_set_target_kind"), row.get("target_kind"))
        ),
        "operation_set_operation_family": _optional_text(
            _first(row.get("operation_set_operation_family"), row.get("operation_family"))
        ),
        "operation_set_params": _optional_mapping(
            _first(row.get("operation_set_params"), row.get("params")),
            "operation_set_params",
        ),
        "candidate_generation_only": True,
        "planning_only": True,
        "allowed_use": "planning_rank_for_candidate_generation_or_exact_eval_followup",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        "tool": TOOL,
    }
    return _false_authority(
        out,
        "inverse_steganalysis_atom_is_not_score_authority",
        "requires_byte_closed_archive_and_exact_auth_eval_before_promotion",
    )


def normalize_inverse_steganalysis_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a local/proxy scorer observation used by acquisition ranking."""

    _reject_truthy_authority(row, label="inverse-steganalysis observation")
    candidate_id = _text(row.get("candidate_id"), "candidate_id")
    axis = _text(_first(row.get("axis"), row.get("score_axis")), "axis")
    axis_normalized = _token(axis).replace("_", "-")
    resource_kind = _resource(row.get("resource_kind", "local_cpu"))
    saved_bytes = _optional_int(row.get("saved_bytes"), "saved_bytes")
    observed_rate_gain = _float_or_none(
        row.get("observed_rate_gain"),
        "observed_rate_gain",
        minimum=0.0,
    )
    explicit_rate_positive = row.get("rate_positive")
    rate_positive = explicit_rate_positive is True or (
        explicit_rate_positive is None
        and saved_bytes is not None
        and saved_bytes > 0
        and observed_rate_gain is not None
        and observed_rate_gain > 0.0
    )
    if _looks_like_contest_auth_axis(axis_normalized) or resource_kind == "contest_exact_eval":
        raise InverseSteganalysisAcquisitionError(
            "inverse-steganalysis observations must not masquerade as contest "
            "auth evidence; use auth-eval payload validation for contest axes"
        )
    out = {
        "schema": OBSERVATION_SCHEMA,
        "source_observation_schema": _optional_text(row.get("schema")),
        "observation_id": _optional_text(row.get("observation_id")) or f"obs_{_slug(candidate_id)}_{_slug(axis)}",
        "observation_kind": _optional_text(row.get("observation_kind")),
        "candidate_id": candidate_id,
        "axis": axis,
        "axis_normalized": axis_normalized,
        "target_kind": _optional_text(row.get("target_kind")),
        "materializer_id": _optional_text(_first(row.get("materializer_id"), row.get("materializer"))),
        "portability_contract": (
            dict(row["portability_contract"]) if isinstance(row.get("portability_contract"), Mapping) else None
        ),
        "receiver_contract_kind": _optional_text(row.get("receiver_contract_kind")),
        "source_path": _optional_text(row.get("source_path")),
        "queue_id": _optional_text(row.get("queue_id")),
        "experiment_id": _optional_text(row.get("experiment_id")),
        "step_id": _optional_text(row.get("step_id")),
        "performance_bucket_key": _optional_text(row.get("performance_bucket_key")),
        "performance_summary_schema": _optional_text(row.get("performance_summary_schema")),
        "queue_observation_schema": _optional_text(row.get("queue_observation_schema")),
        "queue_observation_health": (
            None if row.get("queue_observation_health") is None else bool(row.get("queue_observation_health"))
        ),
        "queue_observation_status": _optional_text(row.get("queue_observation_status")),
        "queue_observation_blockers": _list_strings(row.get("queue_observation_blockers")),
        "archive_delta_status": _optional_text(row.get("archive_delta_status")),
        "archive_delta_bytes": _optional_int(
            row.get("archive_delta_bytes"),
            "archive_delta_bytes",
        ),
        "source_archive_bytes": _optional_int(
            row.get("source_archive_bytes"),
            "source_archive_bytes",
            minimum=0,
        ),
        "candidate_archive_bytes": _optional_int(
            row.get("candidate_archive_bytes"),
            "candidate_archive_bytes",
            minimum=0,
        ),
        "source_archive_sha256": _sha256_or_none(
            row.get("source_archive_sha256"),
            "source_archive_sha256",
        ),
        "candidate_archive_sha256": _sha256_or_none(
            row.get("candidate_archive_sha256"),
            "candidate_archive_sha256",
        ),
        "savings_realized": row.get("savings_realized") is True,
        "inflate_parity_satisfied": row.get("inflate_parity_satisfied") is True,
        "quality_spend_allowed": row.get("quality_spend_allowed") is True,
        "materializer_rate_outcome": _optional_text(row.get("materializer_rate_outcome")),
        "signal_semantics": _optional_text(row.get("signal_semantics")),
        "readiness_blockers": _list_strings(row.get("readiness_blockers")),
        "dispatch_blockers": _list_strings(row.get("dispatch_blockers")),
        "expected_artifact_paths": _list_strings(row.get("expected_artifact_paths")),
        "candidate_ids": _list_strings(row.get("candidate_ids")),
        "work_ids": _list_strings(row.get("work_ids")),
        "backlog_keys": _list_strings(row.get("backlog_keys")),
        "source_unit_ids": _list_strings(row.get("source_unit_ids")),
        "source_selection_ids": _list_strings(row.get("source_selection_ids")),
        "run_count": _optional_int(row.get("run_count"), "run_count", minimum=0),
        "success_count": _optional_int(row.get("success_count"), "success_count", minimum=0),
        "failure_count": _optional_int(row.get("failure_count"), "failure_count", minimum=0),
        "saved_bytes": saved_bytes,
        "observed_rate_gain": observed_rate_gain,
        "rate_positive": rate_positive,
        "receiver_contract_satisfied": row.get("receiver_contract_satisfied") is True,
        "artifact_record_count": _optional_int(
            row.get("artifact_record_count"),
            "artifact_record_count",
            minimum=0,
        ),
        "artifact_record_raw_bytes_mean": _float_or_none(
            row.get("artifact_record_raw_bytes_mean"),
            "artifact_record_raw_bytes_mean",
            minimum=0.0,
        ),
        "runtime_identity": _identity(row.get("runtime_identity"), RUNTIME_IDENTITY_KEYS, "runtime_identity"),
        "cache_identity": _identity(row.get("cache_identity"), CACHE_IDENTITY_KEYS, "cache_identity"),
        "observed_score_gain": _float_or_none(
            _first(
                row.get("observed_score_gain"),
                row.get("observed_scorer_gain_vs_baseline"),
                row.get("normalized_full_video_scorer_gain_vs_baseline"),
            ),
            "observed_score_gain",
            minimum=0.0,
        ),
        "calibration_error": _float(
            _first(
                row.get("calibration_error"),
                row.get("absolute_calibration_error"),
                row.get("calibration_uncertainty_score"),
                0.0,
            ),
            "calibration_error",
            minimum=0.0,
        ),
        "elapsed_seconds": _float_or_none(row.get("elapsed_seconds"), "elapsed_seconds", minimum=0.0, exclusive=True),
        "artifact_bytes": _int(
            _first(row.get("artifact_bytes"), row.get("source_artifact_bytes"), 0), "artifact_bytes", minimum=0
        ),
        "resource_kind": resource_kind,
        "candidate_generation_only": True,
        "planning_only": True,
        "allowed_use": "local_or_proxy_acquisition_ranking_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        "tool": TOOL,
    }
    if row.get("exact_auth_calibration") is not None:
        if not isinstance(row.get("exact_auth_calibration"), Mapping):
            raise InverseSteganalysisAcquisitionError("exact_auth_calibration must be an object")
        out["exact_auth_calibration"] = dict(row["exact_auth_calibration"])
    return _false_authority(
        out,
        "inverse_steganalysis_observation_is_not_score_authority",
        "local_or_proxy_observation_requires_exact_auth_eval_before_promotion",
    )


def observations_from_queue_performance_summary(
    summary: Mapping[str, Any],
    *,
    runtime_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    axis: str = "[local-queue-performance advisory]",
    source_path: str | None = None,
    candidate_id_by_experiment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert queue timing/artifact telemetry into planning-only observations.

    Queue performance rows do not carry scorer deltas. They only calibrate the
    acquisition denominator: seconds, artifact footprint, and resource class.
    Runtime/cache identity remains required so the observation cannot silently
    cross into score or promotion authority.
    """

    _reject_truthy_authority(summary, label="queue performance summary")
    if summary.get("schema") != QUEUE_PERFORMANCE_SUMMARY_SCHEMA:
        raise InverseSteganalysisAcquisitionError(f"summary schema must be {QUEUE_PERFORMANCE_SUMMARY_SCHEMA}")
    queue_id = _text(summary.get("queue_id"), "summary.queue_id")
    by_step = summary.get("by_step")
    if not isinstance(by_step, Mapping):
        raise InverseSteganalysisAcquisitionError("summary.by_step must be an object")
    embedded_lookup = _parse_candidate_lookup(
        summary.get("candidate_id_by_experiment") if summary.get("candidate_id_by_experiment") is not None else None,
        label="summary.candidate_id_by_experiment",
    )
    explicit_lookup = _parse_candidate_lookup(
        candidate_id_by_experiment,
        label="candidate_id_by_experiment",
    )
    candidate_lookup = _merge_candidate_lookup(
        embedded_lookup,
        explicit_lookup,
        conflict_label="summary.candidate_id_by_experiment",
    )
    if by_step and not candidate_lookup:
        raise InverseSteganalysisAcquisitionError(
            "queue performance summary missing candidate_id_by_experiment; "
            "pass candidate_id_by_experiment only for legacy summaries"
        )

    observations: list[dict[str, Any]] = []
    for bucket_key, bucket in sorted(by_step.items(), key=lambda item: str(item[0])):
        if not isinstance(bucket, Mapping):
            raise InverseSteganalysisAcquisitionError("summary.by_step values must be objects")
        experiment_id, step_id = _split_performance_step_key(bucket_key)
        run_count = _int(
            bucket.get("run_count", 0),
            f"summary.by_step[{bucket_key!r}].run_count",
            minimum=0,
        )
        if run_count == 0:
            continue
        resource_kind = _resource(
            bucket.get("dominant_resource_kind")
            or _dominant_resource_from_counts(bucket.get("resource_kind_counts"))
            or "local_cpu"
        )
        elapsed_seconds = _float_or_none(
            bucket.get("elapsed_seconds_mean"),
            f"summary.by_step[{bucket_key!r}].elapsed_seconds_mean",
            minimum=0.0,
            exclusive=True,
        )
        artifact_bytes = _queue_performance_artifact_bytes(bucket)
        candidate_ids = candidate_lookup.get(experiment_id)
        if candidate_ids is None:
            raise InverseSteganalysisAcquisitionError(
                f"summary.candidate_id_by_experiment missing experiment {experiment_id!r}"
            )
        base_observation_id = f"queue_perf_{_slug(queue_id)}_{_slug(experiment_id)}_{_slug(step_id)}"
        for candidate_id in candidate_ids:
            observation_id = base_observation_id
            if len(candidate_ids) > 1:
                observation_id = f"{base_observation_id}_{_slug(candidate_id)}"
            observations.append(
                normalize_inverse_steganalysis_observation(
                    {
                        "observation_id": observation_id,
                        "observation_kind": "queue_performance_step",
                        "candidate_id": candidate_id,
                        "axis": axis,
                        "source_path": source_path,
                        "queue_id": queue_id,
                        "experiment_id": experiment_id,
                        "step_id": step_id,
                        "performance_bucket_key": str(bucket_key),
                        "performance_summary_schema": QUEUE_PERFORMANCE_SUMMARY_SCHEMA,
                        "candidate_ids": bucket.get("candidate_ids"),
                        "work_ids": bucket.get("work_ids"),
                        "backlog_keys": bucket.get("backlog_keys"),
                        "source_unit_ids": bucket.get("source_unit_ids"),
                        "source_selection_ids": bucket.get("source_selection_ids"),
                        "runtime_identity": runtime_identity,
                        "cache_identity": cache_identity,
                        "elapsed_seconds": elapsed_seconds,
                        "artifact_bytes": artifact_bytes,
                        "resource_kind": resource_kind,
                        "run_count": run_count,
                        "success_count": bucket.get("success_count"),
                        "failure_count": bucket.get("failure_count"),
                        "artifact_record_count": bucket.get("artifact_record_count"),
                        "artifact_record_raw_bytes_mean": bucket.get("artifact_record_raw_bytes_mean"),
                    }
                )
            )
    return observations


def observations_from_queue_observation(
    observation: Mapping[str, Any],
    *,
    runtime_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    axis: str = "[local-queue-observation advisory]",
    source_path: str | None = None,
    candidate_id_by_experiment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert queue health/blocker observations into action feedback rows.

    ``experiment_queue_observation.v1`` carries richer failed/blocked/orphan
    state than pure performance telemetry. These rows are never score evidence;
    they are local planning blockers used to prevent the next water-bucket pass
    from blindly refilling a bucket whose queue proof path is unhealthy.
    """

    _reject_truthy_authority(observation, label="queue observation")
    if observation.get("schema") != QUEUE_OBSERVATION_SCHEMA:
        raise InverseSteganalysisAcquisitionError(f"queue observation schema must be {QUEUE_OBSERVATION_SCHEMA}")
    queue_id = _text(observation.get("queue_id"), "queue_observation.queue_id")
    explicit_lookup = _parse_candidate_lookup(
        candidate_id_by_experiment,
        label="candidate_id_by_experiment",
    )
    performance = observation.get("performance")
    performance_lookup: dict[str, tuple[str, ...]] = {}
    performance_identity: dict[str, dict[str, list[str]]] = {}
    observations: list[dict[str, Any]] = []
    if isinstance(performance, Mapping):
        performance_lookup = _parse_candidate_lookup(
            performance.get("candidate_id_by_experiment")
            if performance.get("candidate_id_by_experiment") is not None
            else None,
            label="queue_observation.performance.candidate_id_by_experiment",
        )
        performance_identity = _performance_identity_by_experiment(performance)
        observations.extend(
            observations_from_queue_performance_summary(
                performance,
                runtime_identity=runtime_identity,
                cache_identity=cache_identity,
                axis=axis,
                source_path=source_path,
                candidate_id_by_experiment=explicit_lookup or None,
            )
        )
    elif performance is not None:
        raise InverseSteganalysisAcquisitionError("queue_observation.performance must be an object when present")
    candidate_lookup = _merge_candidate_lookup(
        performance_lookup,
        explicit_lookup,
        conflict_label="queue_observation.performance.candidate_id_by_experiment",
    )

    health_sections = (
        ("failed_steps", "queue_observation_failed_step"),
        ("blocked_steps", "queue_observation_blocked_step"),
        (
            "succeeded_artifact_failure_steps",
            "queue_observation_succeeded_artifact_failure",
        ),
        ("orphaned_steps", "queue_observation_orphaned_step"),
    )
    health_rows_before = len(observations)
    for section, kind in health_sections:
        for step in _sequence_of_mappings(observation.get(section)):
            if section == "orphaned_steps" and not _orphaned_step_blocks_queue(step):
                continue
            observations.extend(
                _queue_health_observations_for_step(
                    step,
                    queue_id=queue_id,
                    section=section,
                    kind=kind,
                    axis=axis,
                    source_path=source_path,
                    runtime_identity=runtime_identity,
                    cache_identity=cache_identity,
                    candidate_lookup=candidate_lookup,
                    performance_identity=performance_identity,
                    observation_blockers=_list_strings(observation.get("blockers")),
                    healthy=observation.get("healthy") is True,
                )
            )
    if (
        observation.get("healthy") is False
        and _queue_observation_has_blocking_global_health(observation)
        and len(observations) == health_rows_before
    ):
        observations.extend(
            _queue_global_health_observations(
                observation,
                queue_id=queue_id,
                axis=axis,
                source_path=source_path,
                runtime_identity=runtime_identity,
                cache_identity=cache_identity,
                candidate_lookup=candidate_lookup,
            )
        )
    for step in _sequence_of_mappings(observation.get("succeeded_artifact_steps")):
        observations.extend(
            _queue_materializer_delta_observations_for_observed_step(
                step,
                queue_id=queue_id,
                section="succeeded_artifact_steps",
                axis=axis,
                source_path=source_path,
                runtime_identity=runtime_identity,
                cache_identity=cache_identity,
                candidate_lookup=candidate_lookup,
                performance_identity=performance_identity,
            )
        )
    return observations


def observations_from_materializer_chain_manifest(
    manifest: Mapping[str, Any],
    *,
    runtime_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    axis: str = "[local-materializer-archive-delta advisory]",
    source_path: str | None = None,
    candidate_manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert byte-closed materializer chain economics into planner feedback.

    A materializer chain can be receiver/parity-correct while still adding bytes.
    That is valuable negative evidence: the next water-bucket pass must not
    refill the same matched bucket just because the local proof chain succeeded.
    """

    _reject_truthy_authority(manifest, label="materializer chain manifest")
    if candidate_manifest is not None:
        _reject_truthy_authority(
            candidate_manifest,
            label="materializer chain candidate manifest",
        )
    delta = manifest.get("serialized_archive_delta")
    if not isinstance(delta, Mapping):
        raise InverseSteganalysisAcquisitionError(
            "materializer chain manifest serialized_archive_delta must be an object"
        )
    _reject_truthy_authority(delta, label="materializer chain serialized_archive_delta")
    status = _text(delta.get("status"), "serialized_archive_delta.status")
    realized_saved_bytes = _int(
        delta.get("realized_saved_bytes"),
        "serialized_archive_delta.realized_saved_bytes",
    )
    archive_delta_bytes = _optional_int(
        delta.get("archive_delta_bytes"),
        "serialized_archive_delta.archive_delta_bytes",
    )
    source_archive = manifest.get("source_archive")
    candidate_archive = manifest.get("candidate_archive")
    if not isinstance(source_archive, Mapping):
        source_archive = {}
    if not isinstance(candidate_archive, Mapping):
        candidate_archive = {}
    source_archive_bytes = _optional_int(
        _first(
            delta.get("source_archive_bytes"),
            manifest.get("source_archive_bytes"),
            source_archive.get("bytes"),
            source_archive.get("archive_bytes"),
        ),
        "source_archive_bytes",
        minimum=0,
    )
    candidate_archive_bytes = _optional_int(
        _first(
            delta.get("candidate_archive_bytes"),
            manifest.get("candidate_archive_bytes"),
            candidate_archive.get("bytes"),
            candidate_archive.get("archive_bytes"),
        ),
        "candidate_archive_bytes",
        minimum=0,
    )
    source_archive_sha256 = _sha256_or_none(
        _first(
            manifest.get("source_archive_sha256"),
            source_archive.get("sha256"),
            source_archive.get("archive_sha256"),
        ),
        "source_archive_sha256",
    )
    candidate_archive_sha256 = _sha256_or_none(
        _first(
            manifest.get("candidate_archive_sha256"),
            candidate_archive.get("sha256"),
            candidate_archive.get("archive_sha256"),
        ),
        "candidate_archive_sha256",
    )
    rate_positive = realized_saved_bytes > 0 and status == "realized_saving"
    observed_rate_gain = CONTEST_RATE_SCORE_PER_BYTE * float(realized_saved_bytes) if rate_positive else 0.0
    selected_cells = (
        _sequence_of_mappings(candidate_manifest.get("selected_cells"))
        if isinstance(candidate_manifest, Mapping)
        else []
    )
    if not selected_cells:
        selected_cells = _sequence_of_mappings(manifest.get("selected_cells"))
    if not selected_cells:
        selected_cells = [
            {
                "candidate_id": _first(
                    manifest.get("candidate_id"),
                    (f"materializer_chain_{candidate_archive_sha256[:12]}" if candidate_archive_sha256 else None),
                    "materializer_chain_unknown_candidate",
                ),
                "atom_id": _first(
                    manifest.get("atom_id"),
                    (f"materializer_chain_{candidate_archive_sha256[:12]}" if candidate_archive_sha256 else None),
                ),
            }
        ]
    expected_artifact_paths = _materializer_chain_artifact_paths(manifest)
    readiness_blockers = ordered_unique(
        str(item)
        for item in (
            *(_list_strings(manifest.get("readiness_blockers"))),
            *(
                _list_strings(candidate_manifest.get("readiness_blockers"))
                if isinstance(candidate_manifest, Mapping)
                else []
            ),
            *(_list_strings(delta.get("blockers"))),
        )
        if str(item)
    )
    dispatch_blockers = ordered_unique(
        str(item)
        for item in (
            *(_list_strings(manifest.get("dispatch_blockers"))),
            *(
                _list_strings(candidate_manifest.get("dispatch_blockers"))
                if isinstance(candidate_manifest, Mapping)
                else []
            ),
        )
        if str(item)
    )
    observations: list[dict[str, Any]] = []
    for index, cell in enumerate(selected_cells):
        atom_id = _optional_text(cell.get("atom_id"))
        candidate_id = _optional_text(cell.get("candidate_id")) or _optional_text(manifest.get("candidate_id"))
        if candidate_id is None:
            candidate_id = (
                f"materializer_chain_{candidate_archive_sha256[:12]}"
                if candidate_archive_sha256
                else f"materializer_chain_cell_{index:04d}"
            )
        source_unit_ids = ordered_unique(
            str(item)
            for item in (
                atom_id,
                f"inverse_action_{atom_id}" if atom_id else None,
                *_list_strings(cell.get("source_unit_ids")),
            )
            if str(item or "")
        )
        observations.append(
            normalize_inverse_steganalysis_observation(
                {
                    "observation_id": (
                        f"materializer_chain_delta_{_slug(candidate_id)}_{_slug(atom_id or str(index))}"
                    ),
                    "observation_kind": MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND,
                    "candidate_id": candidate_id,
                    "axis": axis,
                    "source_path": source_path,
                    "runtime_identity": runtime_identity,
                    "cache_identity": cache_identity,
                    "target_kind": (
                        _optional_text(
                            _first(
                                candidate_manifest.get("target_kind")
                                if isinstance(candidate_manifest, Mapping)
                                else None,
                                manifest.get("target_kind"),
                            )
                        )
                    ),
                    "materializer_id": (
                        _optional_text(
                            _first(
                                candidate_manifest.get("materializer_id")
                                if isinstance(candidate_manifest, Mapping)
                                else None,
                                manifest.get("materializer_id"),
                            )
                        )
                    ),
                    "receiver_contract_kind": (
                        _optional_text(
                            _first(
                                candidate_manifest.get("receiver_contract_kind")
                                if isinstance(candidate_manifest, Mapping)
                                else None,
                                manifest.get("receiver_contract_kind"),
                            )
                        )
                    ),
                    "saved_bytes": realized_saved_bytes,
                    "observed_rate_gain": observed_rate_gain,
                    "observed_score_gain": observed_rate_gain,
                    "rate_positive": rate_positive,
                    "savings_realized": delta.get("savings_realized") is True,
                    "quality_spend_allowed": manifest.get("quality_spend_allowed") is True,
                    "materializer_rate_outcome": status,
                    "signal_semantics": (
                        "realized_archive_saving"
                        if rate_positive
                        else "successful_quality_spend_not_byte_saving_progress"
                    ),
                    "receiver_contract_satisfied": (manifest.get("receiver_contract_satisfied") is True),
                    "inflate_parity_satisfied": (manifest.get("inflate_parity_satisfied") is True),
                    "archive_delta_status": status,
                    "archive_delta_bytes": archive_delta_bytes,
                    "source_archive_bytes": source_archive_bytes,
                    "candidate_archive_bytes": candidate_archive_bytes,
                    "source_archive_sha256": source_archive_sha256,
                    "candidate_archive_sha256": candidate_archive_sha256,
                    "artifact_bytes": candidate_archive_bytes or 0,
                    "resource_kind": "local_cpu",
                    "candidate_ids": [candidate_id],
                    "source_unit_ids": source_unit_ids,
                    "source_selection_ids": _list_strings(cell.get("source_selection_ids")),
                    "expected_artifact_paths": expected_artifact_paths,
                    "readiness_blockers": readiness_blockers,
                    "dispatch_blockers": dispatch_blockers,
                }
            )
        )
    return observations


def paired_exact_auth_calibration_observations_from_review_packets(
    packets: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    packet_paths: Sequence[str] | None = None,
    observation_id: str | None = None,
    source_path: str | None = None,
) -> list[dict[str, Any]]:
    """Convert paired CPU/CUDA result-review packets into calibration signal.

    The returned observation is intentionally not a contest-axis observation.
    It is a false-authority trust-region update for the inverse-steganalysis
    planner. The exact auth scores remain nested metadata so downstream
    consumers cannot confuse this with a score claim or rank/kill surface.
    """

    candidate = _text(candidate_id, "candidate_id")
    if not isinstance(packets, Sequence) or isinstance(packets, bytes | str):
        raise InverseSteganalysisAcquisitionError("packets must be a sequence")
    packet_rows = [dict(packet) for packet in packets]
    if len(packet_rows) != 2:
        raise InverseSteganalysisAcquisitionError("paired exact-auth calibration requires exactly two review packets")
    paths = [str(path) for path in (packet_paths or [])]
    if packet_paths is not None and len(paths) != len(packet_rows):
        raise InverseSteganalysisAcquisitionError("packet_paths length must match packets length")

    by_axis = {_review_packet_axis(packet): packet for packet in packet_rows}
    if set(by_axis) != {"contest_cpu", "contest_cuda"}:
        raise InverseSteganalysisAcquisitionError(
            "paired exact-auth calibration requires one contest_cpu and one contest_cuda packet"
        )
    path_by_axis = {
        _review_packet_axis(packet): paths[index] if index < len(paths) else None
        for index, packet in enumerate(packet_rows)
    }
    _validate_paired_exact_auth_packets(by_axis)

    axis_rows = {
        axis: _exact_auth_axis_calibration_row(
            packet,
            axis=axis,
            packet_path=path_by_axis.get(axis),
        )
        for axis, packet in sorted(by_axis.items())
    }
    regression_penalty = sum(max(0.0, float(row["delta_vs_axis_baseline"])) for row in axis_rows.values())
    improvement_gain = sum(max(0.0, -float(row["delta_vs_axis_baseline"])) for row in axis_rows.values())
    observed_score_gain = 0.0 if regression_penalty > 0.0 else improvement_gain
    pair_status = (
        "paired_exact_auth_regressed_vs_axis_baselines"
        if regression_penalty > 0.0
        else "paired_exact_auth_improved_vs_axis_baselines"
        if improvement_gain > 0.0
        else "paired_exact_auth_tied_axis_baselines"
    )
    custody = _paired_exact_auth_custody(by_axis)
    runtime_identity = _paired_exact_auth_runtime_identity(by_axis)
    cache_identity = _paired_exact_auth_cache_identity(by_axis)
    source = source_path or ",".join(path for path in paths if path) or None
    raw = {
        "observation_id": observation_id
        or f"exact_auth_calibration_{_slug(candidate)}_{_slug(custody['archive_sha256'])}",
        "observation_kind": "paired_exact_auth_calibration",
        "candidate_id": candidate,
        "axis": "[paired exact-auth calibration]",
        "source_path": source,
        "runtime_identity": runtime_identity,
        "cache_identity": cache_identity,
        "observed_score_gain": observed_score_gain,
        "calibration_error": regression_penalty,
        "artifact_bytes": custody["archive_bytes"],
        "resource_kind": "exact_auth_calibration",
    }
    observation = normalize_inverse_steganalysis_observation(raw)
    observation["exact_auth_calibration"] = _false_authority(
        {
            "schema": EXACT_AUTH_CALIBRATION_SCHEMA,
            "candidate_id": candidate,
            "pair_status": pair_status,
            "archive_sha256": custody["archive_sha256"],
            "archive_bytes": custody["archive_bytes"],
            "n_samples": custody["n_samples"],
            "axis_rows": axis_rows,
            "regression_penalty_sum": regression_penalty,
            "improvement_gain_sum": improvement_gain,
            "observed_score_gain_policy": ("zero_when_any_exact_auth_axis_regresses_vs_baseline"),
            "runtime_identity": runtime_identity,
            "cache_identity": cache_identity,
            "packet_paths": [path_by_axis[axis] for axis in ("contest_cpu", "contest_cuda") if path_by_axis.get(axis)],
            "allowed_use": ("planner_calibration_only_measured_config_not_family_retirement"),
        },
        "exact_auth_calibration_is_planning_only",
        "does_not_convert_cpu_to_cuda_or_cuda_to_cpu",
        "measured_config_only_no_family_falsification",
    )
    return [observation]


def compute_acquisition_priority(
    atom: Mapping[str, Any],
    observation: Mapping[str, Any] | None = None,
    *,
    uncertainty_weight: float = 0.25,
) -> dict[str, Any]:
    """Compute score-gain priority normalized by time, artifact bytes, resource."""

    atom_row = dict(atom) if atom.get("schema") == ATOM_SCHEMA else normalize_inverse_steganalysis_atom(atom)
    obs_row = None
    if observation is not None:
        obs_row = (
            dict(observation)
            if observation.get("schema") == OBSERVATION_SCHEMA
            else normalize_inverse_steganalysis_observation(observation)
        )
    if not math.isfinite(uncertainty_weight) or uncertainty_weight < 0.0:
        raise InverseSteganalysisAcquisitionError("uncertainty_weight must be finite and non-negative")

    predicted_gain = _float(atom_row.get("predicted_score_gain"), "predicted_score_gain", minimum=0.0)
    first_order = _float(atom_row.get("first_order_marginal_effect"), "first_order_marginal_effect")
    second_order = _float(atom_row.get("second_order_interaction_effect"), "second_order_interaction_effect")
    fragility_penalty = _float(atom_row.get("fragility_penalty"), "fragility_penalty", minimum=0.0)
    observed_gain = _float_or_none(
        None if obs_row is None else obs_row.get("observed_score_gain"),
        "observed_score_gain",
        minimum=0.0,
    )
    action_gain = max(0.0, first_order + second_order)
    base_gain = max(predicted_gain, action_gain) if observed_gain is None else observed_gain
    uncertainty_bonus = _float(atom_row.get("uncertainty"), "uncertainty", minimum=0.0) * uncertainty_weight
    calibration_penalty = _float(
        _first(
            None if obs_row is None else obs_row.get("calibration_error"),
            atom_row.get("calibration_error"),
            0.0,
        ),
        "calibration_error",
        minimum=0.0,
    )
    elapsed_seconds = max(
        _float(
            _first(
                None if obs_row is None else obs_row.get("elapsed_seconds"),
                atom_row.get("elapsed_seconds"),
                MIN_ELAPSED_SECONDS,
            ),
            "elapsed_seconds",
            minimum=0.0,
            exclusive=True,
        ),
        MIN_ELAPSED_SECONDS,
    )
    artifact_bytes = _int(
        _first(
            None if obs_row is None else obs_row.get("artifact_bytes"),
            atom_row.get("artifact_bytes"),
            0,
        ),
        "artifact_bytes",
        minimum=0,
    )
    resource_kind = _resource(
        _first(
            None if obs_row is None else obs_row.get("resource_kind"),
            atom_row.get("resource_kind"),
            "local_cpu",
        )
    )
    resource_multiplier = RESOURCE_MULTIPLIERS.get(resource_kind, DEFAULT_RESOURCE_MULTIPLIER)
    queue_health_blocked = obs_row is not None and _is_queue_health_blocker_observation(obs_row)
    archive_delta_blocked = obs_row is not None and _materializer_archive_delta_blocks_water_bucket(obs_row)
    expected_gain = (
        0.0
        if queue_health_blocked or archive_delta_blocked
        else max(0.0, base_gain - calibration_penalty - fragility_penalty) + uncertainty_bonus
    )
    artifact_gb = max(artifact_bytes / 1_000_000_000.0, MIN_ARTIFACT_GB)
    terms = AcquisitionPriorityTerms(
        predicted_score_gain=predicted_gain,
        expected_score_gain=expected_gain,
        uncertainty_bonus=uncertainty_bonus,
        calibration_penalty=calibration_penalty,
        elapsed_seconds=elapsed_seconds,
        artifact_bytes=artifact_bytes,
        artifact_gb=artifact_gb,
        resource_kind=resource_kind,
        resource_multiplier=resource_multiplier,
        score_gain_per_second=expected_gain / elapsed_seconds,
        score_gain_per_gb=expected_gain / artifact_gb,
        acquisition_priority=(expected_gain / elapsed_seconds) / (resource_multiplier * (1.0 + artifact_gb)),
    )
    return terms.to_dict()


def build_inverse_steganalysis_acquisition_plan(
    atoms: Iterable[Mapping[str, Any]],
    *,
    observations: Iterable[Mapping[str, Any]] = (),
    top_k: int | None = None,
) -> dict[str, Any]:
    """Rank atoms using optional local/proxy calibration observations."""

    obs_rows = [normalize_inverse_steganalysis_observation(row) for row in observations]
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for obs in obs_rows:
        by_candidate.setdefault(str(obs["candidate_id"]), []).append(obs)

    ranked: list[dict[str, Any]] = []
    for index, raw_atom in enumerate(atoms):
        atom = normalize_inverse_steganalysis_atom(raw_atom)
        candidate_obs = _matching_observations_for_atom(
            atom,
            obs_rows,
            by_candidate,
        )
        best_obs = _best_observation(atom, candidate_obs)
        queue_health_feedback = _queue_health_feedback_for_observations(
            candidate_obs,
            atom=atom,
        )
        materializer_archive_delta_feedback = _materializer_archive_delta_feedback_for_observations(
            candidate_obs,
            atom=atom,
        )
        priority = _apply_queue_health_feedback_to_priority(
            compute_acquisition_priority(atom, best_obs),
            queue_health_feedback,
        )
        priority = _apply_materializer_archive_delta_feedback_to_priority(
            priority,
            materializer_archive_delta_feedback,
        )
        ranked.append(
            _false_authority(
                {
                    **atom,
                    "source_atom_index": index,
                    "best_observation_id": None if best_obs is None else best_obs["observation_id"],
                    "observation_count": len(candidate_obs),
                    "queue_health_feedback": queue_health_feedback,
                    "materializer_archive_delta_feedback": (materializer_archive_delta_feedback),
                    "queue_health_blocked": bool(queue_health_feedback["blocks_water_bucket"]),
                    "materializer_archive_delta_blocked": bool(
                        materializer_archive_delta_feedback["blocks_water_bucket"]
                    ),
                    "priority": priority,
                },
                "ranked_inverse_steganalysis_row_is_planning_only",
                "requires_exact_eval_before_score_or_promotion_claim",
            )
        )

    ranked.sort(
        key=lambda row: (
            -float(row["priority"]["acquisition_priority"]),
            -float(row["priority"]["expected_score_gain"]),
            float(row["priority"]["elapsed_seconds"]),
            str(row["atom_id"]),
        )
    )
    if top_k is not None:
        if top_k < 1:
            raise InverseSteganalysisAcquisitionError("top_k must be positive")
        ranked = ranked[:top_k]
    for rank, row in enumerate(ranked, start=1):
        row["acquisition_rank"] = rank

    return _false_authority(
        {
            "schema": SCHEMA,
            "tool": TOOL,
            "candidate_generation_only": True,
            "planning_only": True,
            "authority": "false_authority_proxy_acquisition_only",
            "ranked_atoms": ranked,
            "summary": {
                "atom_count": len(ranked),
                "observation_count": len(obs_rows),
                "queue_health_blocked_count": sum(1 for row in ranked if row.get("queue_health_blocked") is True),
                "materializer_archive_delta_blocked_count": sum(
                    1 for row in ranked if row.get("materializer_archive_delta_blocked") is True
                ),
                "top_atom_id": None if not ranked else ranked[0]["atom_id"],
                "top_candidate_id": None if not ranked else ranked[0]["candidate_id"],
            },
        },
        "inverse_steganalysis_acquisition_plan_is_not_score_authority",
        "planner_rank_requires_exact_eval_before_promotion_or_rank_kill",
    )


def build_discrete_scorer_action_functional(
    atoms: Iterable[Mapping[str, Any]],
    *,
    observations: Iterable[Mapping[str, Any]] = (),
    total_byte_budget: int | None = None,
    lambda_rate: float = CONTEST_RATE_SCORE_PER_BYTE,
    max_cells: int | None = None,
) -> dict[str, Any]:
    """Approximate hydrated auth eval as a coupled discrete action surface.

    The returned rows are a Riemann-sum style planning model over byte, pixel,
    region, frame, pair, batch, and full-video cells.  They carry local first-
    order score marginals, second-order synergy/antagonism terms, discontinuity
    barriers, and a rate shadow price so deterministic materializers can choose
    the next water bucket without treating proxy evidence as score authority.
    """

    if not math.isfinite(lambda_rate) or lambda_rate < 0.0:
        raise InverseSteganalysisAcquisitionError("lambda_rate must be finite and non-negative")
    if total_byte_budget is not None and total_byte_budget < 1:
        raise InverseSteganalysisAcquisitionError("total_byte_budget must be positive")
    if max_cells is not None and max_cells < 1:
        raise InverseSteganalysisAcquisitionError("max_cells must be positive")

    obs_rows = [normalize_inverse_steganalysis_observation(row) for row in observations]
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for obs in obs_rows:
        by_candidate.setdefault(str(obs["candidate_id"]), []).append(obs)

    cells: list[dict[str, Any]] = []
    total_first_order = 0.0
    total_second_order = 0.0
    total_synergy = 0.0
    total_antagonism = 0.0
    total_fragility_penalty = 0.0
    total_expected_gain = 0.0
    blocked_cells = 0
    queue_health_blocked_cells = 0
    materializer_archive_delta_blocked_cells = 0
    for index, raw_atom in enumerate(atoms):
        if max_cells is not None and index >= max_cells:
            raise InverseSteganalysisAcquisitionError(
                f"inverse action functional cell count exceeds max_cells={max_cells}"
            )
        atom = normalize_inverse_steganalysis_atom(raw_atom)
        matched_observations = _matching_observations_for_atom(atom, obs_rows, by_candidate)
        best_obs = _best_observation(atom, matched_observations)
        queue_health_feedback = _queue_health_feedback_for_observations(
            matched_observations,
            atom=atom,
        )
        materializer_archive_delta_feedback = _materializer_archive_delta_feedback_for_observations(
            matched_observations,
            atom=atom,
        )
        priority = _apply_queue_health_feedback_to_priority(
            compute_acquisition_priority(atom, best_obs),
            queue_health_feedback,
        )
        priority = _apply_materializer_archive_delta_feedback_to_priority(
            priority,
            materializer_archive_delta_feedback,
        )
        measure = _cell_measure(atom)
        first_order = _float(atom["first_order_marginal_effect"], "first_order_marginal_effect")
        second_order = _float(atom["second_order_interaction_effect"], "second_order_interaction_effect")
        synergy = _float(atom["synergy_effect"], "synergy_effect", minimum=0.0)
        antagonism = _float(atom["antagonism_effect"], "antagonism_effect", minimum=0.0)
        fragility = _float(atom["fragility_penalty"], "fragility_penalty", minimum=0.0)
        expected_gain = _float(priority["expected_score_gain"], "expected_score_gain", minimum=0.0)
        byte_cost = int(measure["water_fill_cost_bytes"])
        marginal_utility = expected_gain / float(byte_cost)
        residual = marginal_utility - lambda_rate
        guard = dict(atom["discontinuity_guard"])
        queue_health_blocked = bool(queue_health_feedback["blocks_water_bucket"])
        materializer_archive_delta_blocked = bool(materializer_archive_delta_feedback["blocks_water_bucket"])
        blocked = bool(guard.get("blocked")) or queue_health_blocked or materializer_archive_delta_blocked
        if blocked:
            blocked_cells += 1
        if queue_health_blocked:
            queue_health_blocked_cells += 1
        if materializer_archive_delta_blocked:
            materializer_archive_delta_blocked_cells += 1
        total_first_order += first_order
        total_second_order += second_order
        total_synergy += synergy
        total_antagonism += antagonism
        total_fragility_penalty += fragility
        total_expected_gain += expected_gain
        cells.append(
            _false_authority(
                {
                    "schema": ACTION_CELL_SCHEMA,
                    "cell_index": index,
                    "atom_id": atom["atom_id"],
                    "candidate_id": atom["candidate_id"],
                    "candidate_generation_only": True,
                    "planning_only": True,
                    "scale": atom["scale"],
                    "scope_axis": atom["scope_axis"],
                    "component": atom["component"],
                    "measure": measure,
                    "first_order_marginal_effect": first_order,
                    "second_order_interaction_effect": second_order,
                    "synergy_effect": synergy,
                    "antagonism_effect": antagonism,
                    "fragility_penalty": fragility,
                    "expected_score_gain": expected_gain,
                    "lambda_rate": lambda_rate,
                    "marginal_utility_per_byte": marginal_utility,
                    "euler_lagrange_residual": residual,
                    "water_bucket_selectable": residual > 0.0 and not blocked,
                    "discontinuity_guard": guard,
                    "queue_health_feedback": queue_health_feedback,
                    "queue_health_blocked": queue_health_blocked,
                    "materializer_archive_delta_feedback": (materializer_archive_delta_feedback),
                    "materializer_archive_delta_blocked": (materializer_archive_delta_blocked),
                    "queue_health_group_ids": queue_health_feedback["group_ids"],
                    "queue_health_repeat_count": queue_health_feedback["repeated_observation_count"],
                    "queue_health_penalty_applied": queue_health_feedback["queue_health_penalty_applied"],
                    "best_observation_id": None if best_obs is None else best_obs["observation_id"],
                    "best_observation_kind": (None if best_obs is None else best_obs.get("observation_kind")),
                    "exact_auth_calibration": (
                        best_obs.get("exact_auth_calibration")
                        if isinstance(best_obs, Mapping) and isinstance(best_obs.get("exact_auth_calibration"), Mapping)
                        else None
                    ),
                    "source_provenance": atom.get("source_provenance"),
                    "operation_set_compiler": atom.get("operation_set_compiler"),
                    "operation_set_target_kind": atom.get("operation_set_target_kind"),
                    "operation_set_operation_family": atom.get("operation_set_operation_family"),
                    "operation_set_params": atom.get("operation_set_params"),
                    "priority": priority,
                },
                "inverse_steganalysis_action_cell_is_planning_only",
                "requires_materialized_archive_and_exact_auth_eval_before_score_claim",
            )
        )

    water_bucket = _water_bucket_fill(cells, total_byte_budget=total_byte_budget)
    cells.sort(
        key=lambda row: (
            -float(row["euler_lagrange_residual"]),
            -float(row["expected_score_gain"]),
            int(row["measure"]["water_fill_cost_bytes"]),
            str(row["atom_id"]),
        )
    )
    return _false_authority(
        {
            "schema": ACTION_FUNCTIONAL_SCHEMA,
            "tool": TOOL,
            "candidate_generation_only": True,
            "planning_only": True,
            "authority": "false_authority_discrete_action_surface_only",
            "math_model": {
                "representation": "discrete_riemann_sum_with_second_order_interactions",
                "coordinates": [
                    "bytes",
                    "pixels",
                    "regions",
                    "boundaries",
                    "frames",
                    "pairs",
                    "batches",
                    "full_video",
                    "scorer_component",
                ],
                "objective_terms": [
                    "segnet_error_field",
                    "posenet_geometry_field",
                    "rate_shadow_price",
                    "second_order_synergy_antagonism_kernel",
                    "discontinuity_barrier",
                    "calibration_residual",
                ],
                "stationarity_rule": "select positive euler_lagrange_residual cells under byte budget and guard barriers",
                "lambda_rate": lambda_rate,
            },
            "observation_feedback": _observation_feedback_summary(obs_rows),
            "queue_health_feedback": _queue_health_feedback_for_observations(obs_rows),
            "materializer_archive_delta_feedback": (_materializer_archive_delta_feedback_for_observations(obs_rows)),
            "integral_totals": {
                "cell_count": len(cells),
                "blocked_cell_count": blocked_cells,
                "queue_health_blocked_cell_count": queue_health_blocked_cells,
                "materializer_archive_delta_blocked_cell_count": (materializer_archive_delta_blocked_cells),
                "first_order_marginal_effect_sum": total_first_order,
                "second_order_interaction_effect_sum": total_second_order,
                "synergy_effect_sum": total_synergy,
                "antagonism_effect_sum": total_antagonism,
                "fragility_penalty_sum": total_fragility_penalty,
                "expected_score_gain_sum": total_expected_gain,
                "net_action_gain_after_fragility": max(
                    0.0,
                    total_first_order + total_second_order - total_fragility_penalty,
                ),
            },
            "water_bucket": water_bucket,
            "cells": cells,
        },
        "inverse_steganalysis_discrete_action_functional_is_not_score_authority",
        "requires_byte_closed_candidate_generation_before_dispatch",
        "requires_exact_auth_eval_before_promotion_or_rank_kill",
    )


def action_atoms_from_byte_shaving_signal_surface(
    surface: Mapping[str, Any],
    *,
    source_path: str | None = None,
    candidate_id: str | None = None,
    elapsed_seconds: float | None = None,
    artifact_bytes: int | None = None,
    resource_kind: str = "local_mlx",
    max_k: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a family-agnostic byte-shaving surface into action atoms.

    This is the bridge from HNeRV sections, BoostNeRV/NeRV tensor units,
    packet/member recodes, and non-NeRV candidate matrices into the same
    inverse-steganalysis action functional used by scorer-response atoms.  It
    preserves coupled operation sets and interaction terms as provenance; the
    emitted rows remain planning-only.
    """

    _reject_truthy_authority(surface, label="byte-shaving signal surface")
    validate_byte_shaving_signal_surface(surface)
    plan = build_byte_shaving_campaign_plan(surface, max_k=max_k)
    return action_atoms_from_byte_shaving_campaign_plan(
        plan,
        source_path=source_path,
        candidate_id=candidate_id,
        elapsed_seconds=elapsed_seconds,
        artifact_bytes=artifact_bytes,
        resource_kind=resource_kind,
    )


def action_atoms_from_byte_shaving_campaign_plan(
    plan: Mapping[str, Any],
    *,
    source_path: str | None = None,
    candidate_id: str | None = None,
    elapsed_seconds: float | None = None,
    artifact_bytes: int | None = None,
    resource_kind: str = "local_mlx",
) -> list[dict[str, Any]]:
    """Convert byte-shaving operation sets into inverse-action atoms."""

    _reject_truthy_authority(plan, label="byte-shaving campaign plan")
    if plan.get("schema") != BYTE_SHAVING_CAMPAIGN_PLAN_SCHEMA:
        raise InverseSteganalysisAcquisitionError(f"plan schema must be {BYTE_SHAVING_CAMPAIGN_PLAN_SCHEMA}")
    if elapsed_seconds is not None:
        elapsed_seconds = _float(
            elapsed_seconds,
            "elapsed_seconds",
            minimum=0.0,
            exclusive=True,
        )
    if artifact_bytes is not None:
        artifact_bytes = _int(artifact_bytes, "artifact_bytes", minimum=0)

    atoms: list[dict[str, Any]] = []
    operation_sets = [
        item
        for item in _sequence_of_mappings(plan.get("operation_set_ladder"))
        if item.get("schema") == BYTE_SHAVING_OPERATION_SET_SCHEMA
    ]
    for index, operation_set in enumerate(operation_sets):
        atoms.append(
            normalize_inverse_steganalysis_atom(
                _action_atom_from_byte_shaving_operation_set(
                    operation_set,
                    plan=plan,
                    index=index,
                    source_path=source_path,
                    default_candidate_id=candidate_id,
                    elapsed_seconds=elapsed_seconds,
                    artifact_bytes=artifact_bytes,
                    resource_kind=resource_kind,
                )
            )
        )
    if atoms:
        return atoms

    ranked_units = _sequence_of_mappings(plan.get("ranked_units"))
    if not ranked_units:
        raise InverseSteganalysisAcquisitionError("byte-shaving plan must contain operation_set_ladder or ranked_units")
    for index, unit in enumerate(ranked_units):
        atoms.append(
            normalize_inverse_steganalysis_atom(
                _action_atom_from_byte_shaving_ranked_unit(
                    unit,
                    plan=plan,
                    index=index,
                    source_path=source_path,
                    default_candidate_id=candidate_id,
                    elapsed_seconds=elapsed_seconds,
                    artifact_bytes=artifact_bytes,
                    resource_kind=resource_kind,
                )
            )
        )
    return atoms


def action_atoms_from_inverse_scorer_surface(
    surface: Mapping[str, Any],
    *,
    candidate_id: str | None = None,
    elapsed_seconds: float | None = None,
    artifact_bytes: int | None = None,
    resource_kind: str = "local_mlx",
) -> list[dict[str, Any]]:
    """Convert inverse scorer decision cells into normalized action atoms."""

    _reject_truthy_authority(surface, label="inverse scorer decision surface")
    if surface.get("schema") != INVERSE_SCORER_SURFACE_SCHEMA:
        raise InverseSteganalysisAcquisitionError(f"surface schema must be {INVERSE_SCORER_SURFACE_SCHEMA}")
    cells = surface.get("cells")
    if not isinstance(cells, list) or not cells:
        raise InverseSteganalysisAcquisitionError("surface.cells must be a non-empty list")
    if elapsed_seconds is not None:
        elapsed_seconds = _float(
            elapsed_seconds,
            "elapsed_seconds",
            minimum=0.0,
            exclusive=True,
        )
    if artifact_bytes is not None:
        artifact_bytes = _int(artifact_bytes, "artifact_bytes", minimum=0)

    atoms: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise InverseSteganalysisAcquisitionError("surface.cells rows must be objects")
        atoms.append(
            normalize_inverse_steganalysis_atom(
                _action_atom_from_inverse_cell(
                    cell,
                    index=index,
                    default_candidate_id=candidate_id,
                    elapsed_seconds=elapsed_seconds,
                    artifact_bytes=artifact_bytes,
                    resource_kind=resource_kind,
                )
            )
        )
    return atoms


def inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection(
    selection: Mapping[str, Any],
    *,
    source_path: str | None = None,
    elapsed_seconds: float | None = None,
    artifact_bytes: int | None = None,
    resource_kind: str = "local_mlx",
) -> list[dict[str, Any]]:
    """Convert strict MLX spend-triage selections into action atoms.

    The MLX selection is a candidate-generation signal only.  This bridge keeps
    it in that evidence space by requiring the strict effective, parity,
    calibration, and production gates and by rejecting truthy authority flags
    anywhere in the manifest before emitting false-authority action atoms.
    """

    rows = _validated_mlx_effective_spend_triage_rows(selection)
    if elapsed_seconds is not None:
        elapsed_seconds = _float(
            elapsed_seconds,
            "elapsed_seconds",
            minimum=0.0,
            exclusive=True,
        )
    if artifact_bytes is not None:
        artifact_bytes = _int(artifact_bytes, "artifact_bytes", minimum=0)

    atoms: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        atoms.append(
            normalize_inverse_steganalysis_atom(
                _action_atom_from_mlx_effective_spend_triage_row(
                    row,
                    selection=selection,
                    index=index,
                    source_path=source_path,
                    elapsed_seconds=elapsed_seconds,
                    artifact_bytes=artifact_bytes,
                    resource_kind=resource_kind,
                )
            )
        )
    return atoms


def action_atoms_from_mlx_acquisition_batch(
    batch: Mapping[str, Any],
    *,
    source_path: str | None = None,
    candidate_id: str | None = None,
    elapsed_seconds: float | None = None,
    artifact_bytes: int | None = None,
    resource_kind: str | None = None,
) -> list[dict[str, Any]]:
    """Convert grouped local MLX acquisition operation sets into action atoms."""

    _reject_truthy_authority(batch, label="MLX acquisition batch")
    try:
        normalized_batch = validate_mlx_acquisition_batch(batch)
    except ValueError as exc:
        raise InverseSteganalysisAcquisitionError(str(exc)) from exc
    if elapsed_seconds is not None:
        elapsed_seconds = _float(
            elapsed_seconds,
            "elapsed_seconds",
            minimum=0.0,
            exclusive=True,
        )
    if artifact_bytes is not None:
        artifact_bytes = _int(artifact_bytes, "artifact_bytes", minimum=0)
    atoms: list[dict[str, Any]] = []
    for index, operation_set in enumerate(_sequence_of_mappings(normalized_batch.get("operation_sets"))):
        atoms.append(
            normalize_inverse_steganalysis_atom(
                _action_atom_from_mlx_acquisition_operation_set(
                    operation_set,
                    batch=normalized_batch,
                    index=index,
                    source_path=source_path,
                    default_candidate_id=candidate_id,
                    elapsed_seconds=elapsed_seconds,
                    artifact_bytes=artifact_bytes,
                    resource_kind=resource_kind,
                )
            )
        )
    return atoms


def _validated_mlx_effective_spend_triage_rows(
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _reject_truthy_authority(selection, label="MLX effective spend-triage selection")
    _require_explicit_false_authority(
        selection,
        label="MLX effective spend-triage selection",
    )
    if selection.get("schema") != MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA:
        raise InverseSteganalysisAcquisitionError(
            f"selection schema must be {MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA}"
        )
    if selection.get("candidate_generation_only") is not True:
        raise InverseSteganalysisAcquisitionError("MLX selection candidate_generation_only must be true")
    if selection.get("archive_materialization_required") is not True:
        raise InverseSteganalysisAcquisitionError("MLX selection archive_materialization_required must be true")
    if selection.get("requires_exact_auth_eval_before_score_claim") is not True:
        raise InverseSteganalysisAcquisitionError("MLX selection must require exact auth eval before score claim")
    if selection.get("evidence_grade") != MLX_EVIDENCE_GRADE:
        raise InverseSteganalysisAcquisitionError(f"MLX selection evidence_grade must be {MLX_EVIDENCE_GRADE}")
    if selection.get("evidence_tag") != MLX_EVIDENCE_TAG:
        raise InverseSteganalysisAcquisitionError(f"MLX selection evidence_tag must be {MLX_EVIDENCE_TAG}")
    if selection.get("score_axis") != MLX_EVIDENCE_TAG:
        raise InverseSteganalysisAcquisitionError(f"MLX selection score_axis must be {MLX_EVIDENCE_TAG}")

    gates = selection.get("gates")
    if not isinstance(gates, Mapping):
        raise InverseSteganalysisAcquisitionError("MLX selection gates must be an object")
    effective_gate = gates.get("effective_mlx_spend_triage_gate")
    if not isinstance(effective_gate, Mapping):
        raise InverseSteganalysisAcquisitionError("MLX selection effective_mlx_spend_triage_gate missing")
    if effective_gate.get("status") != "strict_pass":
        raise InverseSteganalysisAcquisitionError("MLX effective spend-triage gate must be strict_pass")
    if effective_gate.get("mlx_exact_eval_spend_triage_allowed") is not True:
        raise InverseSteganalysisAcquisitionError("MLX effective spend-triage gate must allow spend triage")
    for key, label in (
        ("torch_parity_status", "MLX parity gate"),
        ("score_calibration_status", "MLX score calibration gate"),
        ("production_contract_status", "MLX production contract gate"),
    ):
        if gates.get(key) != "strict_pass":
            raise InverseSteganalysisAcquisitionError(f"{label} must be strict_pass")

    policy = selection.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise InverseSteganalysisAcquisitionError("MLX selection selection_policy must be an object")
    if policy.get("planning_value_accessor") != "scorer_response_planning_value_for_target":
        raise InverseSteganalysisAcquisitionError("MLX selection must use scorer_response_planning_value_for_target")
    if policy.get("planning_value_scope") != "normalized_full_video":
        raise InverseSteganalysisAcquisitionError("MLX selection planning_value_scope must be normalized_full_video")
    if policy.get("require_singleton_windows") is not True:
        raise InverseSteganalysisAcquisitionError("MLX selection must require singleton windows")

    rows = selection.get("selected_rows")
    if not isinstance(rows, list) or not rows:
        raise InverseSteganalysisAcquisitionError("MLX selection selected_rows must be a non-empty list")
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise InverseSteganalysisAcquisitionError(f"MLX selection row {index} must be an object")
        label = f"MLX selection row {index}"
        _reject_truthy_authority(row, label=label)
        _require_explicit_false_authority(row, label=label)
        if row.get("schema") != MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_ROW_SCHEMA:
            raise InverseSteganalysisAcquisitionError(f"{label} schema mismatch")
        if row.get("candidate_generation_only") is not True:
            raise InverseSteganalysisAcquisitionError(f"{label} candidate_generation_only must be true")
        if row.get("archive_materialization_required") is not True:
            raise InverseSteganalysisAcquisitionError(f"{label} archive_materialization_required must be true")
        if row.get("requires_exact_auth_eval_before_score_claim") is not True:
            raise InverseSteganalysisAcquisitionError(f"{label} must require exact auth eval before score claim")
        if row.get("selection_basis") != "normalized_full_video_mlx_singleton_response_gain":
            raise InverseSteganalysisAcquisitionError(
                f"{label} selection_basis must be normalized_full_video_mlx_singleton_response_gain"
            )
        if row.get("selection_planning_value_accessor") != ("scorer_response_planning_value_for_target"):
            raise InverseSteganalysisAcquisitionError(f"{label} must use scorer_response_planning_value_for_target")
        if row.get("selection_planning_value_scope") != "normalized_full_video":
            raise InverseSteganalysisAcquisitionError(
                f"{label} selection_planning_value_scope must be normalized_full_video"
            )
        normalized_gain = _mlx_selection_score_gain(row, label=label)
        projected_delta = _mlx_selection_projected_delta(row, label=label)
        if projected_delta >= 0.0:
            raise InverseSteganalysisAcquisitionError(f"{label} projected full-video delta must be negative")
        margin = _mlx_selection_byte_margin(row, label=label)
        if margin < 0.0:
            raise InverseSteganalysisAcquisitionError(f"{label} normalized full-video margin must be non-negative")
        if normalized_gain <= 0.0:
            raise InverseSteganalysisAcquisitionError(f"{label} normalized full-video gain must be positive")
        _validate_mlx_selection_row_geometry_and_identity(
            row,
            label=label,
            normalized_gain=normalized_gain,
            projected_delta=projected_delta,
            margin=margin,
        )
        validated.append(dict(row))
    return validated


def _action_atom_from_mlx_effective_spend_triage_row(
    row: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    index: int,
    source_path: str | None,
    elapsed_seconds: float | None,
    artifact_bytes: int | None,
    resource_kind: str,
) -> dict[str, Any]:
    label = f"MLX selection row {index}"
    row_id = _optional_text(row.get("row_id")) or f"row_{index:04d}"
    candidate_id = _text(row.get("candidate_id") or row_id, f"{label}.candidate_id")
    family = _optional_text(row.get("family")) or "mlx_effective_spend_triage"
    projected_delta = _mlx_selection_projected_delta(row, label=label)
    normalized_gain = _mlx_selection_score_gain(row, label=label)
    calibrated_gap = _float(
        row.get("calibrated_min_mlx_gap_for_spend_triage", 0.0),
        f"{label}.calibrated_min_mlx_gap_for_spend_triage",
        minimum=0.0,
    )
    predicted_delta = _float_or_none(
        row.get("predicted_delta_vs_baseline_score"),
        f"{label}.predicted_delta_vs_baseline_score",
    )
    prediction_gap = 0.0 if predicted_delta is None else abs(projected_delta - predicted_delta)
    added_archive_bytes = _float_or_none(
        row.get("added_archive_bytes"),
        f"{label}.added_archive_bytes",
    )
    added_bytes = max(0, math.ceil(added_archive_bytes or 0.0))
    saved_bytes = (
        max(0, math.ceil(abs(added_archive_bytes or 0.0)))
        if (added_archive_bytes is not None and added_archive_bytes < 0.0)
        else 0
    )
    row_artifact_bytes = artifact_bytes
    if row_artifact_bytes is None:
        row_artifact_bytes = max(1, added_bytes, saved_bytes)
    pair_indices = _mlx_selection_pair_indices(row, label=label)
    component = _optional_text(row.get("component")) or "scorer"
    predicted_score_gain = max(0.0, -projected_delta)
    provenance = _false_authority(
        {
            "schema": MLX_EFFECTIVE_SPEND_TRIAGE_PROVENANCE_SCHEMA,
            "selection_schema": selection.get("schema"),
            "selection_source_path": source_path,
            "selection_rank": row.get("rank", index + 1),
            "source_row_id": row_id,
            "source_candidate_id": candidate_id,
            "source_family": family,
            "source_path": row.get("source_path"),
            "window_baseline_source_path": row.get("window_baseline_source_path"),
            "source_pair_window": row.get("source_pair_window"),
            "pair_indices": pair_indices,
            "archive_sha256": row.get("archive_sha256"),
            "raw_sha256": row.get("raw_sha256"),
            "source_inflated_outputs_aggregate_sha256": row.get("source_inflated_outputs_aggregate_sha256"),
            "source_candidate_cache_array_sha256": row.get("source_candidate_cache_array_sha256"),
            "source_reference_cache_array_sha256": row.get("source_reference_cache_array_sha256"),
            "window_baseline_candidate_cache_array_sha256": row.get("window_baseline_candidate_cache_array_sha256"),
            "window_baseline_reference_cache_array_sha256": row.get("window_baseline_reference_cache_array_sha256"),
            "evidence_grade": selection.get("evidence_grade"),
            "evidence_tag": selection.get("evidence_tag"),
            "score_axis": selection.get("score_axis"),
            "selection_basis": row.get("selection_basis"),
            "selection_planning_value_accessor": row.get("selection_planning_value_accessor"),
            "selection_planning_value_scope": row.get("selection_planning_value_scope"),
            "normalized_full_video_scorer_gain_vs_baseline": normalized_gain,
            "projected_full_video_delta_vs_baseline_score": projected_delta,
            "normalized_full_video_byte_budget_margin_vs_break_even": _mlx_selection_byte_margin(row, label=label),
            "observed_scorer_gain_vs_baseline": row.get("observed_scorer_gain_vs_baseline"),
            "observed_delta_vs_baseline_score": row.get("observed_delta_vs_baseline_score"),
            "byte_budget_margin_vs_break_even": row.get("byte_budget_margin_vs_break_even"),
            "calibrated_min_mlx_gap_for_spend_triage": calibrated_gap,
            "prediction_field": row.get("prediction_field"),
            "predicted_delta_vs_baseline_score": predicted_delta,
            "operation_set_compiler": _optional_mapping(
                row.get("operation_set_compiler"),
                f"{label}.operation_set_compiler",
            ),
        },
        "mlx_effective_spend_triage_row_provenance_is_not_score_authority",
    )
    return {
        "atom_id": f"mlx_effective_spend_triage_{_slug(row_id)}_{index:04d}",
        "candidate_id": candidate_id,
        "scale": "pair" if pair_indices else "candidate",
        "scope_axis": "pairs" if pair_indices else "full_video",
        "parent_unit_id": f"mlx_effective_spend_triage:{row_id}",
        "pair_indices": pair_indices,
        "component": component,
        "frequency_band": "mlx_effective_spend_triage",
        "coherence_group": family,
        "sparsity_prior": 0.5,
        "predicted_segnet_gain": 0.0,
        "predicted_posenet_gain": 0.0,
        "predicted_rate_gain": CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes),
        "predicted_rate_cost": CONTEST_RATE_SCORE_PER_BYTE * float(added_bytes),
        "predicted_score_gain": predicted_score_gain,
        "first_order_marginal_effect": predicted_score_gain,
        "second_order_interaction_effect": 0.0,
        "discontinuity_risk": 0.1 if row.get("prediction_agrees_with_observed_gain") is True else 0.25,
        "discontinuity_threshold": 0.75,
        "uncertainty": max(calibrated_gap, prediction_gap),
        "calibration_error": calibrated_gap,
        "elapsed_seconds": elapsed_seconds,
        "artifact_bytes": row_artifact_bytes,
        "resource_kind": resource_kind,
        "operation_set_compiler": _optional_mapping(
            row.get("operation_set_compiler"),
            f"{label}.operation_set_compiler",
        ),
        "source_provenance": provenance,
    }


def _mlx_selection_pair_indices(row: Mapping[str, Any], *, label: str) -> list[int] | None:
    pair_indices = _int_list(row.get("pair_indices"), f"{label}.pair_indices")
    if pair_indices is not None:
        return pair_indices
    window = _range(row.get("source_pair_window"), f"{label}.source_pair_window")
    if window is None:
        return None
    return list(range(window[0], window[1] + 1))


def _mlx_selection_score_gain(row: Mapping[str, Any], *, label: str) -> float:
    return _float(
        row.get("normalized_full_video_scorer_gain_vs_baseline"),
        f"{label}.normalized_full_video_scorer_gain_vs_baseline",
        minimum=0.0,
        exclusive=True,
    )


def _mlx_selection_projected_delta(row: Mapping[str, Any], *, label: str) -> float:
    return _float(
        row.get("projected_full_video_delta_vs_baseline_score"),
        f"{label}.projected_full_video_delta_vs_baseline_score",
    )


def _mlx_selection_byte_margin(row: Mapping[str, Any], *, label: str) -> float:
    return _float(
        row.get("normalized_full_video_byte_budget_margin_vs_break_even"),
        f"{label}.normalized_full_video_byte_budget_margin_vs_break_even",
    )


def _validate_mlx_selection_row_geometry_and_identity(
    row: Mapping[str, Any],
    *,
    label: str,
    normalized_gain: float,
    projected_delta: float,
    margin: float,
) -> None:
    denominator = _int(row.get("full_video_denominator"), f"{label}.full_video_denominator", minimum=1)
    if denominator != 600:
        raise InverseSteganalysisAcquisitionError(f"{label} full_video_denominator must be 600")
    pair_indices = _mlx_selection_pair_indices(row, label=label)
    window = _range(row.get("source_pair_window"), f"{label}.source_pair_window")
    if pair_indices is None or window is None:
        raise InverseSteganalysisAcquisitionError(f"{label} pair_indices and source_pair_window are required")
    expected = list(range(window[0], window[1] + 1))
    if pair_indices != expected:
        raise InverseSteganalysisAcquisitionError(f"{label} pair_indices must match source_pair_window")
    observed_gain = _float_or_none(
        row.get("observed_scorer_gain_vs_baseline"),
        f"{label}.observed_scorer_gain_vs_baseline",
        minimum=0.0,
        exclusive=True,
    )
    if (
        observed_gain is not None
        and "normalized_full_video_scorer_gain_vs_baseline" in row
        and not math.isclose(normalized_gain, observed_gain / denominator, rel_tol=1e-9, abs_tol=1e-12)
    ):
        raise InverseSteganalysisAcquisitionError(f"{label} normalized gain inconsistent with full_video_denominator")
    observed_delta = _float_or_none(
        row.get("observed_delta_vs_baseline_score"),
        f"{label}.observed_delta_vs_baseline_score",
    )
    if (
        observed_delta is not None
        and "projected_full_video_delta_vs_baseline_score" in row
        and not math.isclose(projected_delta, observed_delta / denominator, rel_tol=1e-9, abs_tol=1e-12)
    ):
        raise InverseSteganalysisAcquisitionError(f"{label} projected delta inconsistent with full_video_denominator")
    added_bytes = _float_or_none(
        row.get("added_archive_bytes"),
        f"{label}.added_archive_bytes",
    )
    break_even = row.get(
        "break_even_added_bytes_from_normalized_full_video_gain",
        row.get("break_even_added_bytes_from_scorer_gain"),
    )
    if break_even is not None:
        break_even_bytes = _float(
            break_even,
            f"{label}.break_even_added_bytes_from_score_gain",
            minimum=0.0,
        )
        if not math.isclose(
            break_even_bytes,
            normalized_gain / CONTEST_RATE_SCORE_PER_BYTE,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise InverseSteganalysisAcquisitionError(f"{label} break-even bytes inconsistent with normalized gain")
        if added_bytes is not None and not math.isclose(
            margin,
            break_even_bytes - added_bytes,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise InverseSteganalysisAcquisitionError(f"{label} byte budget margin inconsistent with added bytes")
    for key in (
        "archive_sha256",
        "raw_sha256",
        "source_inflated_outputs_aggregate_sha256",
        "source_posenet_sha256",
        "source_segnet_sha256",
    ):
        value = row.get(key)
        if value is not None:
            _sha256_value(value, f"{label}.{key}")
    for key in (
        "source_candidate_cache_array_sha256",
        "source_reference_cache_array_sha256",
        "window_baseline_candidate_cache_array_sha256",
        "window_baseline_reference_cache_array_sha256",
    ):
        value = row.get(key)
        if value is not None:
            if not isinstance(value, Mapping):
                raise InverseSteganalysisAcquisitionError(f"{label}.{key} must be an object")
            _sha256_value(value, f"{label}.{key}")


def _action_atom_from_inverse_cell(
    cell: Mapping[str, Any],
    *,
    index: int,
    default_candidate_id: str | None,
    elapsed_seconds: float | None,
    artifact_bytes: int | None,
    resource_kind: str,
) -> dict[str, Any]:
    cell_id = _text(cell.get("cell_id"), "cell.cell_id")
    source_candidates = cell.get("source_candidate_ids")
    source_candidate_id = (
        str(source_candidates[0]) if isinstance(source_candidates, list) and source_candidates else None
    )
    candidate_id = default_candidate_id or source_candidate_id or f"inverse_surface_candidate_{_slug(cell_id)}"
    decision_class = _text(
        cell.get("decision_surface_class"),
        "cell.decision_surface_class",
    )
    dominant_axis = _token(cell.get("dominant_receiver_axis") or "mixed")
    pair_indices = _pair_indices_from_bucket(cell.get("pair_bucket"))
    saved_bytes = _int(cell.get("candidate_saved_bytes", 0), "candidate_saved_bytes", minimum=0)
    median_delta = _float(
        cell.get("median_projected_delta_vs_baseline_score"),
        "median_projected_delta_vs_baseline_score",
    )
    best_delta = _float(
        cell.get("best_projected_delta_vs_baseline_score", median_delta),
        "best_projected_delta_vs_baseline_score",
    )
    worst_delta = _float(
        cell.get("worst_projected_delta_vs_baseline_score", median_delta),
        "worst_projected_delta_vs_baseline_score",
    )
    scorer_delta = _float(
        cell.get("median_scorer_delta_vs_baseline", 0.0),
        "median_scorer_delta_vs_baseline",
    )
    predicted_gain = max(0.0, -median_delta)
    best_gain = max(0.0, -best_delta)
    rate_gain = CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes)
    receiver_gain = max(0.0, predicted_gain - rate_gain, -scorer_delta)
    second_order = max(0.0, best_gain - predicted_gain)
    spread = abs(worst_delta - best_delta)
    component = _component_from_dominant_axis(dominant_axis)
    seg_gain = receiver_gain if component == "segnet" else 0.0
    pose_gain = receiver_gain if component == "posenet" else 0.0
    risk = _decision_surface_risk(decision_class)
    return {
        "atom_id": f"inverse_surface_{_slug(cell_id)}_{index:04d}",
        "candidate_id": candidate_id,
        "scale": "pair" if pair_indices else "component",
        "scope_axis": "pairs" if pair_indices else "full_video",
        "parent_unit_id": f"inverse_surface_{cell_id}",
        "pair_indices": pair_indices,
        "component": component,
        "frequency_band": decision_class,
        "coherence_group": str(cell.get("coordinate_key") or cell_id),
        "sparsity_prior": 1.0 if decision_class == "rate_only_null_space" else 0.5,
        "predicted_segnet_gain": seg_gain,
        "predicted_posenet_gain": pose_gain,
        "predicted_rate_gain": rate_gain,
        "predicted_rate_cost": 0.0,
        "predicted_score_gain": predicted_gain,
        "first_order_marginal_effect": predicted_gain,
        "second_order_interaction_effect": second_order,
        "discontinuity_risk": risk,
        "discontinuity_threshold": 0.75,
        "uncertainty": spread,
        "calibration_error": spread * 0.25,
        "elapsed_seconds": elapsed_seconds,
        "artifact_bytes": (artifact_bytes if artifact_bytes is not None else max(1, saved_bytes)),
        "resource_kind": resource_kind,
        "operation_set_compiler": _optional_mapping(
            cell.get("operation_set_compiler"),
            "inverse_cell.operation_set_compiler",
        ),
        "operation_set_target_kind": _optional_text(
            _first(
                cell.get("operation_set_target_kind"),
                cell.get("target_kind"),
                cell.get("recommended_operation_target_kind"),
            )
        ),
        "operation_set_operation_family": _optional_text(
            _first(
                cell.get("operation_set_operation_family"),
                cell.get("operation_family"),
                cell.get("recommended_operation_family"),
            )
        ),
        "operation_set_params": _optional_mapping(
            _first(
                cell.get("operation_set_params"),
                cell.get("recommended_operation_params"),
                cell.get("operation_params"),
                cell.get("params"),
            ),
            "inverse_cell.operation_set_params",
        ),
    }


def _action_atom_from_byte_shaving_operation_set(
    operation_set: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    index: int,
    source_path: str | None,
    default_candidate_id: str | None,
    elapsed_seconds: float | None,
    artifact_bytes: int | None,
    resource_kind: str,
) -> dict[str, Any]:
    operation_set_id = _text(
        operation_set.get("operation_set_id"),
        "operation_set.operation_set_id",
    )
    candidate_id = (
        default_candidate_id or _optional_text(plan.get("candidate_id")) or f"byte_shaving_{_slug(operation_set_id)}"
    )
    unit_kinds = ordered_unique(
        str(operation.get("unit_kind") or "")
        for operation in _sequence_of_mappings(operation_set.get("selected_operations"))
        if str(operation.get("unit_kind") or "")
    )
    scale = _byte_shaving_scale(unit_kinds)
    scope_axis = _byte_shaving_scope_axis(unit_kinds)
    saved_bytes = _int(
        _first(operation_set.get("candidate_saved_bytes"), 0),
        "operation_set.candidate_saved_bytes",
        minimum=0,
    )
    expected_gain = _expected_score_gain(operation_set, "operation_set")
    second_order = _byte_shaving_second_order_gain(operation_set)
    first_order = expected_gain - second_order
    active_interactions = _sequence_of_mappings(operation_set.get("active_interactions"))
    blockers = ordered_unique(
        str(item)
        for item in (
            *(_list_strings(operation_set.get("dispatch_blockers"))),
            *(_list_strings(operation_set.get("blockers"))),
        )
    )
    risk = _byte_shaving_discontinuity_risk(
        blockers=blockers,
        active_interactions=active_interactions,
        unit_kinds=unit_kinds,
    )
    uncertainty = max(
        abs(second_order),
        float(operation_set.get("quality_cost_score") or 0.0) * 0.25,
    )
    row_artifact_bytes = artifact_bytes
    if row_artifact_bytes is None:
        row_artifact_bytes = max(1, saved_bytes)
    return {
        "atom_id": f"byte_shaving_opset_{_slug(operation_set_id)}_{index:04d}",
        "candidate_id": str(candidate_id),
        "scale": scale,
        "scope_axis": scope_axis,
        "parent_unit_id": f"byte_shaving_operation_set:{operation_set_id}",
        "pair_indices": _byte_shaving_pair_indices(operation_set),
        "component": _byte_shaving_component(operation_set, plan),
        "frequency_band": _byte_shaving_frequency_band(operation_set),
        "byte_range": _byte_shaving_byte_range(operation_set),
        "coherence_group": _byte_shaving_coherence_group(operation_set, plan),
        "sparsity_prior": _byte_shaving_sparsity_prior(unit_kinds),
        "predicted_segnet_gain": 0.0,
        "predicted_posenet_gain": 0.0,
        "predicted_rate_gain": CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes),
        "predicted_rate_cost": 0.0,
        "predicted_score_gain": max(0.0, expected_gain),
        "first_order_marginal_effect": first_order,
        "second_order_interaction_effect": second_order,
        "discontinuity_risk": risk,
        "discontinuity_threshold": 0.75,
        "uncertainty": uncertainty,
        "calibration_error": uncertainty * 0.25,
        "elapsed_seconds": elapsed_seconds,
        "artifact_bytes": row_artifact_bytes,
        "resource_kind": resource_kind,
        "operation_set_compiler": _optional_mapping(
            operation_set.get("operation_set_compiler"),
            "operation_set.operation_set_compiler",
        ),
        "operation_set_target_kind": _optional_text(
            _first(operation_set.get("operation_set_target_kind"), operation_set.get("target_kind"))
        ),
        "operation_set_operation_family": _optional_text(
            _first(
                operation_set.get("operation_set_operation_family"),
                operation_set.get("operation_family"),
            )
        ),
        "operation_set_params": _optional_mapping(
            _first(operation_set.get("operation_set_params"), operation_set.get("params")),
            "operation_set.operation_set_params",
        ),
        "source_provenance": _byte_shaving_operation_set_provenance(
            operation_set,
            plan=plan,
            source_path=source_path,
            expected_gain=expected_gain,
            first_order=first_order,
            second_order=second_order,
        ),
    }


def _action_atom_from_mlx_acquisition_operation_set(
    operation_set: Mapping[str, Any],
    *,
    batch: Mapping[str, Any],
    index: int,
    source_path: str | None,
    default_candidate_id: str | None,
    elapsed_seconds: float | None,
    artifact_bytes: int | None,
    resource_kind: str | None,
) -> dict[str, Any]:
    operation_set_id = _text(
        operation_set.get("operation_set_id"),
        "mlx_operation_set.operation_set_id",
    )
    candidate_id = (
        default_candidate_id
        or _optional_text(operation_set.get("candidate_id"))
        or f"mlx_acquisition_{_slug(operation_set_id)}"
    )
    selected_operations = _sequence_of_mappings(operation_set.get("selected_operations"))
    unit_kinds = ordered_unique(
        str(operation.get("unit_kind") or "scorer_response_row") for operation in selected_operations
    )
    pair_indices = _byte_shaving_pair_indices(operation_set)
    saved_bytes = _int(
        _first(operation_set.get("candidate_saved_bytes"), 0),
        "mlx_operation_set.candidate_saved_bytes",
        minimum=0,
    )
    expected_gain = _expected_score_gain(operation_set, "mlx_operation_set")
    second_order = _byte_shaving_second_order_gain(operation_set)
    first_order = expected_gain - second_order
    active_interactions = _sequence_of_mappings(operation_set.get("active_interactions"))
    blockers = ordered_unique(
        str(item)
        for item in (
            *(_list_strings(operation_set.get("dispatch_blockers"))),
            *(_list_strings(operation_set.get("blockers"))),
        )
    )
    risk = _byte_shaving_discontinuity_risk(
        blockers=blockers,
        active_interactions=active_interactions,
        unit_kinds=unit_kinds,
    )
    uncertainty = max(
        abs(second_order),
        float(operation_set.get("uncertainty", 0.0) or 0.0),
        float(operation_set.get("quality_cost_score", 0.0) or 0.0) * 0.25,
    )
    row_artifact_bytes = artifact_bytes
    if row_artifact_bytes is None:
        row_artifact_bytes = max(1, saved_bytes, len(selected_operations))
    return {
        "atom_id": f"mlx_acquisition_opset_{_slug(operation_set_id)}_{index:04d}",
        "candidate_id": str(candidate_id),
        "scale": "pair" if pair_indices else _byte_shaving_scale(unit_kinds),
        "scope_axis": "pairs" if pair_indices else _byte_shaving_scope_axis(unit_kinds),
        "parent_unit_id": f"mlx_acquisition_operation_set:{operation_set_id}",
        "pair_indices": pair_indices,
        "component": str(operation_set.get("component") or "scorer"),
        "frequency_band": _byte_shaving_frequency_band(operation_set),
        "byte_range": _byte_shaving_byte_range(operation_set),
        "coherence_group": ":".join(
            item
            for item in (
                str(batch.get("source_schema") or "mlx_acquisition"),
                str(operation_set_id),
            )
            if item
        ),
        "sparsity_prior": _byte_shaving_sparsity_prior(unit_kinds),
        "predicted_segnet_gain": 0.0,
        "predicted_posenet_gain": 0.0,
        "predicted_rate_gain": CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes),
        "predicted_rate_cost": 0.0,
        "predicted_score_gain": max(0.0, expected_gain),
        "first_order_marginal_effect": first_order,
        "second_order_interaction_effect": second_order,
        "discontinuity_risk": risk,
        "discontinuity_threshold": 0.75,
        "uncertainty": uncertainty,
        "calibration_error": uncertainty * 0.25,
        "elapsed_seconds": elapsed_seconds,
        "artifact_bytes": row_artifact_bytes,
        "resource_kind": resource_kind or str(operation_set.get("resource_kind") or "local_mlx"),
        "operation_set_compiler": _optional_mapping(
            operation_set.get("operation_set_compiler"),
            "mlx_operation_set.operation_set_compiler",
        ),
        "source_provenance": _mlx_acquisition_operation_set_provenance(
            operation_set,
            batch=batch,
            source_path=source_path,
            expected_gain=expected_gain,
            first_order=first_order,
            second_order=second_order,
        ),
    }


def _action_atom_from_byte_shaving_ranked_unit(
    unit: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    index: int,
    source_path: str | None,
    default_candidate_id: str | None,
    elapsed_seconds: float | None,
    artifact_bytes: int | None,
    resource_kind: str,
) -> dict[str, Any]:
    unit_id = _text(unit.get("unit_id"), "ranked_unit.unit_id")
    candidate_id = (
        default_candidate_id
        or _optional_text(unit.get("source_candidate_id"))
        or _optional_text(plan.get("candidate_id"))
        or f"byte_shaving_{_slug(unit_id)}"
    )
    unit_kind = str(unit.get("unit_kind") or "byte_range")
    saved_bytes = _int(
        _first(unit.get("candidate_saved_bytes"), 0),
        "ranked_unit.candidate_saved_bytes",
        minimum=0,
    )
    expected_gain = _expected_score_gain(unit, "ranked_unit")
    blockers = ordered_unique(
        str(item)
        for item in (
            *(_list_strings(unit.get("dispatch_blockers"))),
            *(_list_strings(unit.get("blockers"))),
        )
    )
    risk = _byte_shaving_discontinuity_risk(
        blockers=blockers,
        active_interactions=(),
        unit_kinds=[unit_kind],
    )
    uncertainty = max(0.0, float(unit.get("quality_cost_score") or 0.0) * 0.25)
    row_artifact_bytes = artifact_bytes
    if row_artifact_bytes is None:
        row_artifact_bytes = max(1, saved_bytes)
    source_span = unit.get("source_span")
    return {
        "atom_id": f"byte_shaving_unit_{_slug(unit_id)}_{index:04d}",
        "candidate_id": str(candidate_id),
        "scale": _byte_shaving_scale([unit_kind]),
        "scope_axis": _byte_shaving_scope_axis([unit_kind]),
        "parent_unit_id": f"byte_shaving_ranked_unit:{unit_id}",
        "pair_indices": _byte_shaving_pair_indices(unit),
        "component": _byte_shaving_component(unit, plan),
        "frequency_band": str(unit.get("recommended_operation_family") or unit_kind),
        "byte_range": _source_span_byte_range(source_span),
        "coherence_group": str(
            unit.get("recommended_operation_family") or unit.get("unit_kind") or "byte_shaving_unit"
        ),
        "sparsity_prior": _byte_shaving_sparsity_prior([unit_kind]),
        "predicted_segnet_gain": 0.0,
        "predicted_posenet_gain": 0.0,
        "predicted_rate_gain": CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes),
        "predicted_rate_cost": 0.0,
        "predicted_score_gain": max(0.0, expected_gain),
        "first_order_marginal_effect": expected_gain,
        "second_order_interaction_effect": 0.0,
        "discontinuity_risk": risk,
        "discontinuity_threshold": 0.75,
        "uncertainty": uncertainty,
        "calibration_error": uncertainty * 0.25,
        "elapsed_seconds": elapsed_seconds,
        "artifact_bytes": row_artifact_bytes,
        "resource_kind": resource_kind,
        "operation_set_compiler": _optional_mapping(
            unit.get("operation_set_compiler"),
            "ranked_unit.operation_set_compiler",
        ),
        "operation_set_target_kind": _optional_text(
            _first(
                unit.get("operation_set_target_kind"),
                unit.get("recommended_operation_target_kind"),
                unit.get("target_kind"),
            )
        ),
        "operation_set_operation_family": _optional_text(
            _first(
                unit.get("operation_set_operation_family"),
                unit.get("recommended_operation_family"),
                unit.get("operation_family"),
            )
        ),
        "operation_set_params": _optional_mapping(
            _first(
                unit.get("operation_set_params"),
                unit.get("recommended_operation_params"),
                unit.get("operation_params"),
                unit.get("params"),
            ),
            "ranked_unit.operation_set_params",
        ),
        "source_provenance": _byte_shaving_unit_provenance(
            unit,
            plan=plan,
            source_path=source_path,
            expected_gain=expected_gain,
        ),
    }


def _byte_shaving_operation_set_provenance(
    operation_set: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    source_path: str | None,
    expected_gain: float,
    first_order: float,
    second_order: float,
) -> dict[str, Any]:
    return _false_authority(
        {
            "schema": BYTE_SHAVING_OPERATION_SET_PROVENANCE_SCHEMA,
            "source_path": source_path,
            "source_plan_schema": plan.get("schema"),
            "source_campaign_id": plan.get("campaign_id"),
            "source_plan_candidate_id": plan.get("candidate_id"),
            "source_lane_id": plan.get("lane_id"),
            "operation_set_id": operation_set.get("operation_set_id"),
            "combo_id": operation_set.get("combo_id"),
            "operation_set_rank": operation_set.get("operation_set_rank"),
            "selected_unit_ids": _list_strings(operation_set.get("selected_unit_ids")),
            "operation_families": _list_strings(operation_set.get("operation_families")),
            "chosen_operation_sequence": _sequence_of_mappings(operation_set.get("chosen_operation_sequence")),
            "chosen_operation_sequence_source": operation_set.get("chosen_operation_sequence_source"),
            "operation_set_compiler": _optional_mapping(
                operation_set.get("operation_set_compiler"),
                "operation_set.operation_set_compiler",
            ),
            "selected_operations": _sequence_of_mappings(operation_set.get("selected_operations")),
            "active_interactions": _sequence_of_mappings(operation_set.get("active_interactions")),
            "candidate_saved_bytes": operation_set.get("candidate_saved_bytes"),
            "base_saved_bytes": operation_set.get("base_saved_bytes"),
            "interaction_extra_saved_bytes": operation_set.get("interaction_extra_saved_bytes"),
            "interaction_shared_overhead_bytes": operation_set.get("interaction_shared_overhead_bytes"),
            "quality_cost_score": operation_set.get("quality_cost_score"),
            "interaction_delta_score": operation_set.get("interaction_delta_score"),
            "expected_delta_score": operation_set.get("expected_delta_score"),
            "expected_score_gain": expected_gain,
            "first_order_marginal_effect": first_order,
            "second_order_interaction_effect": second_order,
            "partial_materialization_allowed": operation_set.get("partial_materialization_allowed"),
            "dispatch_blockers": _list_strings(operation_set.get("dispatch_blockers")),
            "source_signal_refs": _sequence_of_mappings(plan.get("source_signal_refs")),
            "scorer_response_refs": _sequence_of_mappings(plan.get("scorer_response_refs")),
            "inverse_scorer_surface_refs": _sequence_of_mappings(plan.get("inverse_scorer_surface_refs")),
            "mlx_calibration_refs": _sequence_of_mappings(plan.get("mlx_calibration_refs")),
            "allowed_use": "inverse_steganalysis_planning_rank_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        },
        "byte_shaving_operation_set_provenance_is_planning_only",
    )


def _mlx_acquisition_operation_set_provenance(
    operation_set: Mapping[str, Any],
    *,
    batch: Mapping[str, Any],
    source_path: str | None,
    expected_gain: float,
    first_order: float,
    second_order: float,
) -> dict[str, Any]:
    return _false_authority(
        {
            "schema": MLX_ACQUISITION_BATCH_PROVENANCE_SCHEMA,
            "source_path": source_path or batch.get("source_path"),
            "source_batch_schema": batch.get("schema"),
            "source_batch_tool": batch.get("tool"),
            "source_schema": batch.get("source_schema"),
            "operation_set_schema": operation_set.get("schema"),
            "operation_set_id": operation_set.get("operation_set_id"),
            "operation_set_rank": operation_set.get("operation_set_rank"),
            "candidate_id": operation_set.get("candidate_id"),
            "resource_kind": operation_set.get("resource_kind"),
            "selected_unit_ids": _list_strings(operation_set.get("selected_unit_ids")),
            "operation_families": _list_strings(operation_set.get("operation_families")),
            "source_families": _list_strings(operation_set.get("source_families")),
            "source_family_classes": _list_strings(operation_set.get("source_family_classes")),
            "representation_contracts": _sequence_of_mappings(operation_set.get("representation_contracts")),
            "receiver_contract_kinds": _list_strings(operation_set.get("receiver_contract_kinds")),
            "materializer_contract_kinds": _list_strings(operation_set.get("materializer_contract_kinds")),
            "operation_portability": operation_set.get("operation_portability"),
            "operation_set_compiler": _optional_mapping(
                operation_set.get("operation_set_compiler"),
                "mlx_operation_set.operation_set_compiler",
            ),
            "selected_operations": _sequence_of_mappings(operation_set.get("selected_operations")),
            "chosen_operation_sequence": _sequence_of_mappings(operation_set.get("chosen_operation_sequence")),
            "chosen_operation_sequence_source": operation_set.get("chosen_operation_sequence_source"),
            "active_interactions": _sequence_of_mappings(operation_set.get("active_interactions")),
            "row_refs": _sequence_of_mappings(operation_set.get("row_refs")),
            "pair_indices": _int_list(operation_set.get("pair_indices"), "pair_indices"),
            "candidate_saved_bytes": operation_set.get("candidate_saved_bytes"),
            "expected_delta_score": operation_set.get("expected_delta_score"),
            "expected_score_gain": expected_gain,
            "first_order_marginal_effect": first_order,
            "second_order_interaction_effect": second_order,
            "allowed_use": "inverse_steganalysis_mlx_operation_set_rank_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        },
        "mlx_acquisition_operation_set_provenance_is_planning_only",
    )


def _byte_shaving_unit_provenance(
    unit: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    source_path: str | None,
    expected_gain: float,
) -> dict[str, Any]:
    return _false_authority(
        {
            "schema": BYTE_SHAVING_UNIT_PROVENANCE_SCHEMA,
            "source_path": source_path,
            "source_plan_schema": plan.get("schema"),
            "source_campaign_id": plan.get("campaign_id"),
            "source_plan_candidate_id": plan.get("candidate_id"),
            "source_lane_id": plan.get("lane_id"),
            "unit_id": unit.get("unit_id"),
            "unit_kind": unit.get("unit_kind"),
            "atom_ids": _list_strings(unit.get("atom_ids")),
            "source_candidate_id": unit.get("source_candidate_id"),
            "source_span": unit.get("source_span"),
            "candidate_saved_bytes": unit.get("candidate_saved_bytes"),
            "quality_cost_score": unit.get("quality_cost_score"),
            "expected_delta_score": unit.get("expected_delta_score"),
            "expected_score_gain": expected_gain,
            "recommended_operation_id": unit.get("recommended_operation_id"),
            "recommended_operation_family": unit.get("recommended_operation_family"),
            "recommended_operation_materializer": unit.get("recommended_operation_materializer"),
            "recommended_operation_target_kind": unit.get("recommended_operation_target_kind"),
            "recommended_operation_params": _optional_mapping(
                unit.get("recommended_operation_params"),
                "ranked_unit.recommended_operation_params",
            ),
            "operation_set_compiler": _optional_mapping(
                unit.get("operation_set_compiler"),
                "ranked_unit.operation_set_compiler",
            ),
            "operation_candidates": _sequence_of_mappings(unit.get("operation_candidates")),
            "source_paths": _list_strings(unit.get("source_paths")),
            "master_gradient_signal": unit.get("master_gradient_signal"),
            "inverse_scorer_signal": unit.get("inverse_scorer_signal"),
            "bit_allocator_signal": unit.get("bit_allocator_signal"),
            "dqs1_outcome_signal": unit.get("dqs1_outcome_signal"),
            "engineered_correction_signal": unit.get("engineered_correction_signal"),
            "canonical_equation_provenance": unit.get("canonical_equation_provenance"),
            "candidate_trust_region_blockers": _list_strings(unit.get("candidate_trust_region_blockers")),
            "dispatch_blockers": _list_strings(unit.get("dispatch_blockers")),
            "allowed_use": "inverse_steganalysis_planning_rank_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        },
        "byte_shaving_ranked_unit_provenance_is_planning_only",
    )


def _expected_score_gain(row: Mapping[str, Any], label: str) -> float:
    explicit = _float_or_none(row.get("expected_score_gain"), f"{label}.expected_score_gain")
    if explicit is not None:
        return explicit
    expected_delta = _float_or_none(
        row.get("expected_delta_score"),
        f"{label}.expected_delta_score",
    )
    if expected_delta is None:
        saved_bytes = _int(
            _first(row.get("candidate_saved_bytes"), 0),
            f"{label}.candidate_saved_bytes",
            minimum=0,
        )
        quality_cost = _float(
            _first(row.get("quality_cost_score"), 0.0),
            f"{label}.quality_cost_score",
        )
        return (CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes)) - quality_cost
    return -expected_delta


def _byte_shaving_second_order_gain(operation_set: Mapping[str, Any]) -> float:
    direct_gain = -_float(
        _first(operation_set.get("interaction_delta_score"), 0.0),
        "operation_set.interaction_delta_score",
    )
    extra_saved = _int(
        _first(operation_set.get("interaction_extra_saved_bytes"), 0),
        "operation_set.interaction_extra_saved_bytes",
        minimum=0,
    )
    overhead = _int(
        _first(operation_set.get("interaction_shared_overhead_bytes"), 0),
        "operation_set.interaction_shared_overhead_bytes",
        minimum=0,
    )
    quality_gain = -sum(
        _float(
            _first(interaction.get("quality_cost_delta_score"), 0.0),
            "interaction.quality_cost_delta_score",
        )
        for interaction in _sequence_of_mappings(operation_set.get("active_interactions"))
    )
    return direct_gain + quality_gain + (CONTEST_RATE_SCORE_PER_BYTE * float(extra_saved - overhead))


def _byte_shaving_scale(unit_kinds: Sequence[str]) -> str:
    normalized = [str(kind) for kind in unit_kinds if str(kind)]
    if len(set(normalized)) > 1:
        return "multiscale"
    kind = normalized[0] if normalized else "byte_range"
    return {
        "pair": "pair",
        "frame": "frame",
        "byte_range": "byte_range",
        "archive_section": "byte_range",
        "tensor": "component",
        "packet_member": "byte_range",
        "scorer_response_row": "candidate",
        "scorer_inverse_surface_cell": "multiscale",
        "correction_target": "byte",
    }.get(kind, "multiscale")


def _byte_shaving_scope_axis(unit_kinds: Sequence[str]) -> str:
    normalized = [str(kind) for kind in unit_kinds if str(kind)]
    mapped = {
        "pair": "pairs",
        "frame": "frames",
        "byte_range": "bytes",
        "archive_section": "bytes",
        "tensor": "full_video",
        "packet_member": "bytes",
        "scorer_response_row": "full_video",
        "scorer_inverse_surface_cell": "full_video",
        "correction_target": "bytes",
    }
    scopes = {mapped.get(kind, "full_video") for kind in normalized}
    if len(scopes) == 1:
        return next(iter(scopes))
    if "pairs" in scopes and len(scopes) == 1:
        return "pairs"
    return "full_video"


def _byte_shaving_component(row: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    for value in (
        row.get("component"),
        row.get("score_axis"),
        row.get("frontier_axis"),
        plan.get("frontier_axis"),
    ):
        text = _optional_text(value)
        if not text:
            continue
        token = _token(text)
        if "seg" in token:
            return "segnet"
        if "pose" in token:
            return "posenet"
        if "rate" in token or "byte" in token:
            return "rate"
    families = set(_list_strings(row.get("operation_families")))
    if any(
        family in families
        for family in (
            "drop_pair",
            "drop_frame",
            "entropy_recode",
            "null_remove_or_seed",
            "section_entropy_recode",
            "zip_header_elide",
            "member_recompress",
            "member_merge",
        )
    ):
        return "rate"
    return "scorer"


def _byte_shaving_frequency_band(row: Mapping[str, Any]) -> str:
    families = _list_strings(row.get("operation_families"))
    if families:
        return "+".join(families[:4])
    return str(row.get("unit_kind") or "byte_shaving")


def _byte_shaving_coherence_group(row: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    families = _list_strings(row.get("operation_families"))
    unit_ids = _list_strings(row.get("selected_unit_ids"))
    return ":".join(
        item
        for item in (
            str(plan.get("campaign_id") or "byte_shaving"),
            "+".join(families[:4]) if families else "",
            "+".join(unit_ids[:4]) if unit_ids else str(row.get("unit_id") or ""),
        )
        if item
    )


def _byte_shaving_pair_indices(row: Mapping[str, Any]) -> list[int] | None:
    values: list[int] = []
    for key in ("pair_indices", "selected_pair_indices"):
        parsed = _int_list(row.get(key), key)
        if parsed:
            values.extend(parsed)
    for operation in _sequence_of_mappings(row.get("selected_operations")):
        parsed = _int_list(operation.get("pair_indices"), "operation.pair_indices")
        if parsed:
            values.extend(parsed)
        params = operation.get("params")
        if isinstance(params, Mapping):
            parsed = _int_list(params.get("pair_indices"), "operation.params.pair_indices")
            if parsed:
                values.extend(parsed)
    return sorted(set(values)) or None


def _byte_shaving_byte_range(row: Mapping[str, Any]) -> list[int] | None:
    for key in ("byte_range", "archive_byte_range"):
        parsed = _range(row.get(key), key)
        if parsed is not None:
            return parsed
    spans = [
        _source_span_byte_range(operation.get("source_span"))
        for operation in _sequence_of_mappings(row.get("selected_operations"))
    ]
    spans = [span for span in spans if span is not None]
    if len(spans) == 1:
        return spans[0]
    return None


def _source_span_byte_range(value: Any) -> list[int] | None:
    if not isinstance(value, Mapping):
        return None
    start = _optional_int(value.get("start"), "source_span.start", minimum=0)
    end = _optional_int(
        value.get("end_exclusive"),
        "source_span.end_exclusive",
        minimum=0,
    )
    if start is None or end is None or end < start:
        return None
        return [start, end]


def _materializer_chain_artifact_paths(manifest: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    candidate_archive = manifest.get("candidate_archive")
    if isinstance(candidate_archive, Mapping):
        _extend_unique(paths, [_optional_text(candidate_archive.get("path"))])
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        for artifact in artifacts.values():
            if isinstance(artifact, Mapping):
                _extend_unique(paths, [_optional_text(artifact.get("path"))])
    for step in _sequence_of_mappings(manifest.get("chain_steps")):
        artifact = step.get("artifact")
        if isinstance(artifact, Mapping):
            _extend_unique(paths, [_optional_text(artifact.get("path"))])
        archive = step.get("archive")
        if isinstance(archive, Mapping):
            _extend_unique(paths, [_optional_text(archive.get("path"))])
    return paths


def _byte_shaving_sparsity_prior(unit_kinds: Sequence[str]) -> float:
    kinds = {str(kind) for kind in unit_kinds if str(kind)}
    if kinds & {"byte_range", "archive_section", "packet_member"}:
        return 1.0
    if kinds & {"tensor", "scorer_inverse_surface_cell", "scorer_response_row"}:
        return 0.5
    return 0.75


def _byte_shaving_discontinuity_risk(
    *,
    blockers: Sequence[str],
    active_interactions: Sequence[Mapping[str, Any]],
    unit_kinds: Sequence[str],
) -> float:
    blocker_text = " ".join(str(blocker).lower() for blocker in blockers)
    if "fragile" in blocker_text or "discontinuity" in blocker_text or "parity_failed" in blocker_text:
        return 0.8
    if "missing" in blocker_text or "materializer_gap" in blocker_text:
        return 0.45
    if active_interactions:
        return 0.2
    if set(unit_kinds) & {"scorer_inverse_surface_cell", "scorer_response_row"}:
        return 0.3
    return 0.1


def _sequence_of_mappings(value: Any) -> list[dict[str, Any]]:
    return (
        [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, Sequence) and not isinstance(value, str | bytes)
        else []
    )


def _list_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if str(item)]


def _matching_observations_for_atom(
    atom: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: Mapping[str, Any]) -> None:
        key = str(row.get("observation_id") or id(row))
        if key in seen:
            return
        seen.add(key)
        matched.append(dict(row))

    atom_targets = _atom_feedback_target_ids(atom)
    for row in by_candidate.get(str(atom["candidate_id"]), ()):
        if _observation_requires_target_intersection(row):
            continue
        add(row)

    if atom_targets:
        for row in observations:
            row_targets = _observation_feedback_target_ids(row)
            compare_atom_targets = atom_targets
            if _observation_requires_target_intersection(row):
                candidate_ids = _observation_candidate_identity_ids(row)
                row_targets = row_targets - candidate_ids
                compare_atom_targets = atom_targets - candidate_ids
            if row_targets & compare_atom_targets:
                add(row)
    return matched


def _atom_feedback_target_ids(atom: Mapping[str, Any]) -> set[str]:
    ids = {
        str(value)
        for value in (
            atom.get("atom_id"),
            atom.get("candidate_id"),
            atom.get("parent_unit_id"),
        )
        if str(value or "")
    }
    provenance = atom.get("source_provenance")
    if isinstance(provenance, Mapping):
        ids.update(
            str(value)
            for value in (
                provenance.get("operation_set_id"),
                provenance.get("combo_id"),
                provenance.get("unit_id"),
                provenance.get("source_row_id"),
                provenance.get("source_candidate_id"),
                provenance.get("candidate_id"),
            )
            if str(value or "")
        )
        ids.update(_list_strings(provenance.get("atom_ids")))
        ids.update(_list_strings(provenance.get("selected_unit_ids")))
        ids.update(_list_strings(provenance.get("source_unit_ids")))
        ids.update(_list_strings(provenance.get("source_selection_ids")))
        recommended_params = provenance.get("recommended_operation_params")
        if isinstance(recommended_params, Mapping):
            op_atom_id = _optional_text(recommended_params.get("atom_id"))
            if op_atom_id:
                ids.add(op_atom_id)
                ids.add(f"inverse_action_{op_atom_id}")
        for operation in _sequence_of_mappings(provenance.get("selected_operations")):
            unit_id = _optional_text(operation.get("unit_id"))
            if unit_id:
                ids.add(unit_id)
        for operation in _sequence_of_mappings(provenance.get("operation_candidates")):
            params = operation.get("operation_params")
            if isinstance(params, Mapping):
                op_atom_id = _optional_text(params.get("atom_id"))
                if op_atom_id:
                    ids.add(op_atom_id)
                    ids.add(f"inverse_action_{op_atom_id}")
    _add_compiler_feedback_target_ids(ids, atom)
    return ids


def _add_compiler_feedback_target_ids(ids: set[str], atom: Mapping[str, Any]) -> None:
    atom_id = _optional_text(atom.get("atom_id"))
    compiler = atom.get("operation_set_compiler")
    if not isinstance(compiler, Mapping):
        return
    operation_set_id = _optional_text(compiler.get("operation_set_id"))
    if operation_set_id:
        ids.add(operation_set_id)
    target_kind = _optional_text(compiler.get("target_kind"))
    if target_kind:
        ids.add(target_kind)
    materializer = _optional_text(_first(compiler.get("materializer_id"), compiler.get("materializer")))
    if materializer:
        ids.add(materializer)
    receiver_contract_kind = _optional_text(compiler.get("receiver_contract_kind"))
    if receiver_contract_kind:
        ids.add(receiver_contract_kind)
    for op_index, operation in enumerate(_sequence_of_mappings(compiler.get("selected_operations"))):
        unit_id = _optional_text(operation.get("unit_id")) or f"op{op_index:04d}"
        ids.add(unit_id)
        target_kind = _optional_text(operation.get("target_kind"))
        if target_kind:
            ids.add(target_kind)
        materializer = _optional_text(_first(operation.get("materializer_id"), operation.get("materializer")))
        if materializer:
            ids.add(materializer)
        receiver_contract_kind = _optional_text(operation.get("receiver_contract_kind"))
        if receiver_contract_kind:
            ids.add(receiver_contract_kind)
        if atom_id:
            ids.add(f"inverse_action_{atom_id}_{unit_id}_{op_index:04d}")


def _observation_feedback_target_ids(observation: Mapping[str, Any]) -> set[str]:
    ids = {
        str(value)
        for value in (
            observation.get("candidate_id"),
            observation.get("experiment_id"),
            observation.get("step_id"),
            observation.get("performance_bucket_key"),
            observation.get("target_kind"),
            observation.get("materializer_id"),
            observation.get("materializer"),
            observation.get("receiver_contract_kind"),
        )
        if str(value or "")
    }
    for key in (
        "candidate_ids",
        "work_ids",
        "backlog_keys",
        "source_unit_ids",
        "source_selection_ids",
    ):
        ids.update(_list_strings(observation.get(key)))
    return ids


def _observation_requires_target_intersection(observation: Mapping[str, Any]) -> bool:
    observation_kind = str(observation.get("observation_kind") or "")
    if observation_kind not in {
        "queue_observation_health_blocker",
        *MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KINDS,
    }:
        return False
    for key in (
        "work_ids",
        "backlog_keys",
        "source_unit_ids",
        "source_selection_ids",
    ):
        if _list_strings(observation.get(key)):
            return True
    return False


def _observation_candidate_identity_ids(observation: Mapping[str, Any]) -> set[str]:
    ids = {str(observation.get("candidate_id") or "")}
    ids.update(_list_strings(observation.get("candidate_ids")))
    return {item for item in ids if item}


def _best_observation(atom: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not observations:
        return None

    def key(row: Mapping[str, Any]) -> tuple[int, int, float, float, float, str]:
        priority = compute_acquisition_priority(atom, row)
        exact_auth_calibration = row.get("observation_kind") == "paired_exact_auth_calibration"
        has_score_observation = row.get("observed_score_gain") is not None
        return (
            int(exact_auth_calibration),
            int(has_score_observation),
            float(priority["acquisition_priority"]),
            float(priority["expected_score_gain"]),
            -float(priority["elapsed_seconds"]),
            str(row.get("observation_id")),
        )

    return dict(max(observations, key=key))


def _observation_feedback_summary(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    kind_counts: dict[str, int] = {}
    exact_auth_refs: list[dict[str, Any]] = []
    for row in observations:
        kind = str(row.get("observation_kind") or "generic_observation")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        calibration = row.get("exact_auth_calibration")
        if isinstance(calibration, Mapping):
            exact_auth_refs.append(
                _false_authority(
                    {
                        "schema": calibration.get("schema"),
                        "candidate_id": calibration.get("candidate_id"),
                        "pair_status": calibration.get("pair_status"),
                        "archive_sha256": calibration.get("archive_sha256"),
                        "archive_bytes": calibration.get("archive_bytes"),
                        "axes": sorted(
                            str(axis)
                            for axis in (
                                calibration.get("axis_rows", {})
                                if isinstance(calibration.get("axis_rows"), Mapping)
                                else {}
                            )
                        ),
                        "regression_penalty_sum": calibration.get("regression_penalty_sum"),
                        "improvement_gain_sum": calibration.get("improvement_gain_sum"),
                        "packet_paths": calibration.get("packet_paths") or [],
                    },
                    "exact_auth_calibration_ref_is_planning_metadata_only",
                )
            )
    return _false_authority(
        {
            "schema": "inverse_steganalysis_observation_feedback.v1",
            "observation_count": len(observations),
            "observation_kind_counts": dict(sorted(kind_counts.items())),
            "exact_auth_calibration_count": len(exact_auth_refs),
            "exact_auth_calibration_refs": exact_auth_refs,
            "queue_health_group_priors": _queue_health_feedback_for_observations(observations),
            "materializer_archive_delta_feedback": (
                _materializer_archive_delta_feedback_for_observations(observations)
            ),
            "allowed_use": "action_functional_calibration_interpretability_only",
        },
        "observation_feedback_is_not_score_authority",
    )


def _is_queue_health_blocker_observation(row: Mapping[str, Any]) -> bool:
    return str(row.get("observation_kind") or "") in QUEUE_HEALTH_BLOCKER_OBSERVATION_KINDS


def _is_materializer_archive_delta_observation(row: Mapping[str, Any]) -> bool:
    return str(row.get("observation_kind") or "") in (MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KINDS)


def _materializer_archive_delta_blocks_water_bucket(row: Mapping[str, Any]) -> bool:
    return _is_materializer_archive_delta_observation(row) and (
        row.get("receiver_contract_satisfied") is not True
        or (row.get("rate_positive") is not True and row.get("quality_spend_allowed") is not True)
    )


def _materializer_archive_delta_feedback_for_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    atom: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = _merge_materializer_archive_delta_observations(
        [dict(row) for row in observations if _is_materializer_archive_delta_observation(row)]
    )
    blocking_rows = [row for row in rows if _materializer_archive_delta_blocks_water_bucket(row)]
    realized_values = [int(row["saved_bytes"]) for row in rows if row.get("saved_bytes") is not None]
    blockers: list[str] = []
    if any(row.get("rate_positive") is not True for row in blocking_rows):
        blockers.append("rate_negative_materializer_success")
    if any(row.get("receiver_contract_satisfied") is not True for row in blocking_rows):
        blockers.append("receiver_negative_materializer_success")
    observed_ids = [_optional_text(row.get("observation_id")) for row in rows]
    blocking_ids = [_optional_text(row.get("observation_id")) for row in blocking_rows]
    source_unit_ids: list[str] = []
    source_selection_ids: list[str] = []
    candidate_ids: list[str] = []
    archive_delta_statuses: list[str] = []
    signal_semantics: list[str] = []
    for row in rows:
        _extend_unique(candidate_ids, [row.get("candidate_id")])
        _extend_unique(source_unit_ids, _list_strings(row.get("source_unit_ids")))
        _extend_unique(
            source_selection_ids,
            _list_strings(row.get("source_selection_ids")),
        )
        _extend_unique(archive_delta_statuses, [row.get("archive_delta_status")])
        _extend_unique(signal_semantics, [row.get("signal_semantics")])
    return _false_authority(
        {
            "schema": MATERIALIZER_ARCHIVE_DELTA_FEEDBACK_SCHEMA,
            "observation_count": len(rows),
            "blocking_observation_count": len(blocking_rows),
            "observation_ids": [item for item in observed_ids if item],
            "blocking_observation_ids": [item for item in blocking_ids if item],
            "rate_positive_count": sum(1 for row in rows if row.get("rate_positive") is True),
            "rate_nonpositive_count": sum(1 for row in rows if row.get("rate_positive") is not True),
            "realized_saved_bytes_sum": sum(realized_values),
            "min_realized_saved_bytes": min(realized_values) if realized_values else None,
            "max_realized_saved_bytes": max(realized_values) if realized_values else None,
            "blocks_water_bucket": bool(blocking_rows),
            "quality_spend_allowed": any(row.get("quality_spend_allowed") is True for row in rows),
            "blockers": blockers,
            "candidate_ids": candidate_ids,
            "source_unit_ids": source_unit_ids,
            "source_selection_ids": source_selection_ids,
            "archive_delta_statuses": archive_delta_statuses,
            "signal_semantics": signal_semantics,
            "atom_id": None if atom is None else atom.get("atom_id"),
            "allowed_use": "archive_delta_feedback_for_local_planning_only",
        },
        "materializer_archive_delta_feedback_is_not_score_authority",
        "rate_negative_materializer_success_requires_replan_before_water_bucket_selection",
    )


def _queue_health_feedback_for_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    atom: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    health_rows = [dict(row) for row in observations if _is_queue_health_blocker_observation(row)]
    hard_blocking_ids = {
        str(row.get("observation_id"))
        for row in health_rows
        if atom is None or _queue_health_observation_hard_blocks_atom(row, atom)
    }
    groups_by_key: dict[str, dict[str, Any]] = {}
    for row in health_rows:
        group_key = _queue_health_feedback_group_key(row)
        group = groups_by_key.setdefault(
            group_key,
            {
                "group_key": group_key,
                "observation_ids": [],
                "hard_blocking_observation_ids": [],
                "observation_kinds": [],
                "candidate_ids": [],
                "experiment_ids": [],
                "step_ids": [],
                "statuses": [],
                "blockers": [],
                "work_ids": [],
                "backlog_keys": [],
                "source_unit_ids": [],
                "source_selection_ids": [],
                "expected_artifact_paths": [],
            },
        )
        _extend_unique(group["observation_ids"], [_optional_text(row.get("observation_id"))])
        observation_id = _optional_text(row.get("observation_id"))
        if observation_id in hard_blocking_ids:
            _extend_unique(group["hard_blocking_observation_ids"], [observation_id])
        _extend_unique(group["observation_kinds"], [_optional_text(row.get("observation_kind"))])
        _extend_unique(group["candidate_ids"], [row.get("candidate_id")])
        _extend_unique(group["experiment_ids"], [row.get("experiment_id")])
        _extend_unique(group["step_ids"], [row.get("step_id")])
        _extend_unique(group["statuses"], [row.get("queue_observation_status")])
        _extend_unique(group["blockers"], _list_strings(row.get("queue_observation_blockers")))
        _extend_unique(group["work_ids"], _list_strings(row.get("work_ids")))
        _extend_unique(group["backlog_keys"], _list_strings(row.get("backlog_keys")))
        _extend_unique(group["source_unit_ids"], _list_strings(row.get("source_unit_ids")))
        _extend_unique(
            group["source_selection_ids"],
            _list_strings(row.get("source_selection_ids")),
        )
        _extend_unique(
            group["expected_artifact_paths"],
            _list_strings(row.get("expected_artifact_paths")),
        )

    groups = []
    for group in sorted(groups_by_key.values(), key=lambda item: str(item["group_key"])):
        observation_count = len(group["observation_ids"])
        hard_blocking_observation_count = len(group["hard_blocking_observation_ids"])
        groups.append(
            {
                **group,
                "observation_count": observation_count,
                "hard_blocking_observation_count": hard_blocking_observation_count,
                "repeated": observation_count > 1,
                "blocks_water_bucket": (observation_count > 0 if atom is None else hard_blocking_observation_count > 0),
            }
        )
    repeated_groups = [group for group in groups if group["repeated"]]
    repeated_observation_count = sum(int(group["observation_count"]) for group in repeated_groups)
    blocks_water_bucket = bool(health_rows) if atom is None else any(group["blocks_water_bucket"] for group in groups)
    if blocks_water_bucket:
        planning_penalty_multiplier = 0.0
    elif repeated_observation_count:
        planning_penalty_multiplier = 1.0 / float(1 + repeated_observation_count)
    else:
        planning_penalty_multiplier = 1.0
    blockers: list[str] = []
    statuses: list[str] = []
    source_unit_ids: list[str] = []
    source_selection_ids: list[str] = []
    for group in groups:
        _extend_unique(blockers, group["blockers"])
        _extend_unique(statuses, group["statuses"])
        _extend_unique(source_unit_ids, group["source_unit_ids"])
        _extend_unique(source_selection_ids, group["source_selection_ids"])
    return _false_authority(
        {
            "schema": QUEUE_HEALTH_FEEDBACK_SCHEMA,
            "observation_count": len(health_rows),
            "group_count": len(groups),
            "group_ids": [str(group["group_key"]) for group in groups],
            "repeated_group_count": len(repeated_groups),
            "repeated_observation_count": repeated_observation_count,
            "global_blocker_count": sum(
                1 for row in health_rows if row.get("observation_kind") == "queue_observation_global_health_blocker"
            ),
            "step_blocker_count": sum(
                1 for row in health_rows if row.get("observation_kind") == "queue_observation_health_blocker"
            ),
            "blocks_water_bucket": blocks_water_bucket,
            "recovery_required": bool(health_rows),
            "planning_penalty_multiplier": planning_penalty_multiplier,
            "queue_health_penalty_applied": planning_penalty_multiplier < 1.0,
            "blockers": blockers,
            "statuses": statuses,
            "source_unit_ids": source_unit_ids,
            "source_selection_ids": source_selection_ids,
            "groups": groups,
            "allowed_use": "queue_recovery_and_action_planning_only",
        },
        "queue_health_feedback_is_not_score_authority",
        "queue_health_feedback_requires_recovery_before_water_bucket_selection",
    )


def _queue_health_feedback_group_key(row: Mapping[str, Any]) -> str:
    materializer = _optional_text(_first(row.get("materializer_id"), row.get("materializer")))
    receiver_contract_kind = _optional_text(row.get("receiver_contract_kind"))
    target_kind = _optional_text(row.get("target_kind"))
    if materializer and receiver_contract_kind:
        return f"materializer_receiver:{materializer}:{receiver_contract_kind}"
    if materializer and target_kind:
        return f"materializer_target:{materializer}:{target_kind}"
    if receiver_contract_kind and target_kind:
        return f"receiver_target:{receiver_contract_kind}:{target_kind}"
    for key in (
        "source_selection_ids",
        "source_unit_ids",
        "work_ids",
        "backlog_keys",
        "expected_artifact_paths",
    ):
        values = _list_strings(row.get(key))
        if values:
            return f"{key}:{'|'.join(values)}"
    experiment_id = _optional_text(row.get("experiment_id"))
    step_id = _optional_text(row.get("step_id"))
    if experiment_id and step_id:
        return f"step:{experiment_id}.{step_id}"
    candidate_id = _optional_text(row.get("candidate_id"))
    if candidate_id:
        return f"candidate:{candidate_id}"
    observation_id = _optional_text(row.get("observation_id"))
    return f"observation:{observation_id or 'unknown'}"


def _queue_health_observation_hard_blocks_atom(
    row: Mapping[str, Any],
    atom: Mapping[str, Any],
) -> bool:
    if row.get("observation_kind") == "queue_observation_global_health_blocker":
        return True
    if _optional_text(row.get("candidate_id")) == _optional_text(atom.get("candidate_id")):
        return True
    atom_targets = _atom_feedback_target_ids(atom)
    for key in (
        "work_ids",
        "backlog_keys",
        "source_unit_ids",
        "source_selection_ids",
        "expected_artifact_paths",
    ):
        if set(_list_strings(row.get(key))) & atom_targets:
            return True
    return False


def _apply_queue_health_feedback_to_priority(
    priority: Mapping[str, Any],
    queue_health_feedback: Mapping[str, Any],
) -> dict[str, Any]:
    multiplier = _float(
        queue_health_feedback.get("planning_penalty_multiplier", 1.0),
        "queue_health_feedback.planning_penalty_multiplier",
        minimum=0.0,
    )
    if multiplier >= 1.0:
        return dict(priority)
    out = dict(priority)
    for key in (
        "expected_score_gain",
        "score_gain_per_second",
        "score_gain_per_gb",
        "acquisition_priority",
    ):
        out[key] = float(out[key]) * multiplier
    out["queue_health_penalty_multiplier"] = multiplier
    out["queue_health_penalty_applied"] = True
    return out


def _apply_materializer_archive_delta_feedback_to_priority(
    priority: Mapping[str, Any],
    materializer_feedback: Mapping[str, Any],
) -> dict[str, Any]:
    if materializer_feedback.get("blocks_water_bucket") is not True:
        return dict(priority)
    out = dict(priority)
    for key in (
        "expected_score_gain",
        "score_gain_per_second",
        "score_gain_per_gb",
        "acquisition_priority",
    ):
        out[key] = 0.0
    out["materializer_archive_delta_blocked"] = True
    out["materializer_archive_delta_blocker_count"] = int(materializer_feedback.get("blocking_observation_count") or 0)
    return out


def _extend_unique(out: list[str], values: Sequence[Any]) -> None:
    seen = set(out)
    for value in values:
        text = _optional_text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)


def _false_authority(row: Mapping[str, Any], *blockers: str) -> dict[str, Any]:
    return apply_proxy_evidence_boundary(
        dict(row),
        dispatch_blockers=ordered_unique(("inverse_steganalysis_acquisition_false_authority_only", *blockers)),
    )


def _require_explicit_false_authority(row: Mapping[str, Any], *, label: str) -> None:
    for key in (
        "score_claim",
        "score_claim_valid",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
        "promotable",
    ):
        if row.get(key) is not False:
            raise InverseSteganalysisAcquisitionError(f"{label} {key} must be explicit false")


def _reject_truthy_authority(row: Mapping[str, Any], *, label: str) -> None:
    violations = truthy_authority_field_violations(row, fields=AUTHORITY_FIELDS)
    if violations:
        raise InverseSteganalysisAcquisitionError(
            f"{label}: forbidden truthy authority fields: {', '.join(violations)}"
        )


def _predicted_effects(row: Mapping[str, Any]) -> dict[str, float]:
    seg = _float(
        row.get("predicted_segnet_gain", row.get("predicted_segnet_score_gain", 0.0)),
        "predicted_segnet_gain",
        minimum=0.0,
    )
    pose = _float(
        row.get("predicted_posenet_gain", row.get("predicted_posenet_score_gain", 0.0)),
        "predicted_posenet_gain",
        minimum=0.0,
    )
    rate_gain = _float(
        row.get("predicted_rate_gain", row.get("predicted_rate_score_gain", 0.0)), "predicted_rate_gain", minimum=0.0
    )
    rate_cost = _float(
        row.get("predicted_rate_cost", row.get("predicted_rate_score_cost", 0.0)), "predicted_rate_cost", minimum=0.0
    )
    explicit = _float_or_none(row.get("predicted_score_gain"), "predicted_score_gain", minimum=0.0)
    delta = _float_or_none(row.get("predicted_delta_vs_baseline_score"), "predicted_delta_vs_baseline_score")
    return {
        "predicted_segnet_gain": seg,
        "predicted_posenet_gain": pose,
        "predicted_rate_gain": rate_gain,
        "predicted_rate_cost": rate_cost,
        "predicted_score_gain": explicit
        if explicit is not None
        else (max(0.0, -delta) if delta is not None else max(0.0, seg + pose + rate_gain - rate_cost)),
    }


def _action_surface_terms(
    row: Mapping[str, Any],
    effects: Mapping[str, float],
) -> dict[str, Any]:
    first_order = _float(
        row.get("first_order_marginal_effect", effects["predicted_score_gain"]),
        "first_order_marginal_effect",
    )
    second_order = _float(
        row.get(
            "second_order_interaction_effect",
            row.get("synergy_effect", row.get("antagonism_effect", 0.0)),
        ),
        "second_order_interaction_effect",
    )
    discontinuity_risk = _float(
        row.get("discontinuity_risk", row.get("fragility_risk", 0.0)),
        "discontinuity_risk",
        minimum=0.0,
    )
    fragility_penalty = _float(
        row.get("fragility_penalty", discontinuity_risk * abs(first_order + second_order)),
        "fragility_penalty",
        minimum=0.0,
    )
    guard_threshold = _float_or_none(
        row.get("discontinuity_threshold"),
        "discontinuity_threshold",
        minimum=0.0,
    )
    guard_blocked = guard_threshold is not None and discontinuity_risk > guard_threshold
    return {
        "first_order_marginal_effect": first_order,
        "second_order_interaction_effect": second_order,
        "interaction_kind": _interaction_kind(second_order),
        "synergy_effect": max(0.0, second_order),
        "antagonism_effect": max(0.0, -second_order),
        "discontinuity_risk": discontinuity_risk,
        "fragility_penalty": fragility_penalty,
        "discontinuity_guard": {
            "schema": "inverse_steganalysis_discontinuity_guard.v1",
            "risk": discontinuity_risk,
            "threshold": guard_threshold,
            "blocked": guard_blocked,
            "blocker": "discontinuity_risk_exceeds_threshold" if guard_blocked else None,
        },
    }


def _cell_measure(atom: Mapping[str, Any]) -> dict[str, Any]:
    byte_span = _span(atom.get("byte_range"))
    frame_span = _span(atom.get("frame_range"))
    pair_count = len(atom.get("pair_indices") or []) if isinstance(atom.get("pair_indices"), list) else 0
    region_area = _region_area(atom.get("region_bbox"))
    component_count = 1 if atom.get("component") else 0
    water_fill_cost_bytes = max(1, byte_span or _int(atom.get("artifact_bytes", 0), "artifact_bytes", minimum=0))
    return {
        "schema": "inverse_steganalysis_action_cell_measure.v1",
        "byte_span": byte_span,
        "frame_span": frame_span,
        "pair_count": pair_count,
        "region_area": region_area,
        "component_count": component_count,
        "water_fill_cost_bytes": water_fill_cost_bytes,
        "water_fill_cost_bytes_semantics": ("planner_budget_cost_not_serialized_savings"),
    }


def _water_bucket_fill(
    cells: Sequence[Mapping[str, Any]],
    *,
    total_byte_budget: int | None,
) -> dict[str, Any]:
    ordered = sorted(
        (dict(cell) for cell in cells if bool(cell.get("water_bucket_selectable"))),
        key=lambda row: (
            -float(row["euler_lagrange_residual"]),
            -float(row["expected_score_gain"]),
            int(row["measure"]["water_fill_cost_bytes"]),
            str(row["atom_id"]),
        ),
    )
    greedy_selected: list[dict[str, Any]] = []
    greedy_used_bytes = 0
    greedy_expected_gain = 0.0
    for row in ordered:
        cost = int(row["measure"]["water_fill_cost_bytes"])
        if total_byte_budget is not None and greedy_used_bytes + cost > total_byte_budget:
            continue
        greedy_selected.append(dict(row))
        greedy_used_bytes += cost
        greedy_expected_gain += float(row["expected_score_gain"])
    selected_rows, frontier_state_count = _water_bucket_portfolio_search(
        ordered,
        total_byte_budget=total_byte_budget,
    )
    selected = [_water_bucket_selection_row(row) for row in selected_rows]
    used_bytes = sum(int(row["water_fill_cost_bytes"]) for row in selected)
    expected_gain = sum(float(row["expected_score_gain"]) for row in selected)
    selected_lagrangian_gain = sum(float(row["portfolio_objective_gain"]) for row in selected)
    strategy = (
        "positive_residual_unbounded_select_all" if total_byte_budget is None else "bounded_lagrangian_portfolio_search"
    )
    return {
        "schema": "inverse_steganalysis_water_bucket_plan.v1",
        "selection_strategy": strategy,
        "total_byte_budget": total_byte_budget,
        "candidate_pool_count": len(ordered),
        "frontier_state_count": frontier_state_count,
        "selected_count": len(selected),
        "selected_water_fill_cost_bytes": used_bytes,
        "selected_expected_score_gain": expected_gain,
        "selected_lagrangian_gain": selected_lagrangian_gain,
        "greedy_baseline_selected_count": len(greedy_selected),
        "greedy_baseline_water_fill_cost_bytes": greedy_used_bytes,
        "greedy_baseline_expected_score_gain": greedy_expected_gain,
        "greedy_baseline_atom_ids": [str(row["atom_id"]) for row in greedy_selected],
        "selected_cells": selected,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _water_bucket_selection_row(row: Mapping[str, Any]) -> dict[str, Any]:
    measure = row.get("measure")
    if not isinstance(measure, Mapping):
        raise InverseSteganalysisAcquisitionError("water bucket cell measure missing")
    cost = int(measure["water_fill_cost_bytes"])
    lambda_rate = float(row["lambda_rate"])
    expected_gain = float(row["expected_score_gain"])
    out: dict[str, Any] = {
        "atom_id": row["atom_id"],
        "candidate_id": row["candidate_id"],
        "candidate_generation_only": True,
        "planning_only": True,
        "scope_axis": row["scope_axis"],
        "component": row["component"],
        "water_fill_cost_bytes": cost,
        "water_fill_cost_bytes_semantics": ("planner_budget_cost_not_serialized_savings"),
        "expected_score_gain": expected_gain,
        "euler_lagrange_residual": row["euler_lagrange_residual"],
        "portfolio_objective_gain": expected_gain - lambda_rate * float(cost),
    }
    compiler = row.get("operation_set_compiler")
    if isinstance(compiler, Mapping) and compiler:
        out["operation_set_compiler"] = dict(compiler)
        for key, value in _water_bucket_compiler_summary(compiler).items():
            out.setdefault(key, value)
    for source_key, target_key in (
        ("operation_set_target_kind", "operation_set_target_kind"),
        ("operation_set_operation_family", "operation_set_operation_family"),
        ("operation_set_params", "operation_set_params"),
    ):
        value = row.get(source_key)
        if value is not None:
            out[target_key] = dict(value) if isinstance(value, Mapping) else value
    if out.get("operation_set_target_kind") and "target_kind" not in out:
        out["target_kind"] = out["operation_set_target_kind"]
    if out.get("operation_set_operation_family") and "operation_family" not in out:
        out["operation_family"] = out["operation_set_operation_family"]
    return out


def _water_bucket_compiler_summary(compiler: Mapping[str, Any]) -> dict[str, Any]:
    selected_operations = _sequence_of_mappings(compiler.get("selected_operations"))

    def first_text(*values: Any) -> str | None:
        for value in values:
            text = _optional_text(value)
            if text is not None:
                return text
        return None

    def first_operation_text(*keys: str) -> str | None:
        for operation in selected_operations:
            text = first_text(*(operation.get(key) for key in keys))
            if text is not None:
                return text
        return None

    summary: dict[str, Any] = {
        "operation_set_id": first_text(compiler.get("operation_set_id")),
        "operation_set_target_kind": first_text(
            compiler.get("target_kind"),
            compiler.get("operation_set_target_kind"),
            first_operation_text("target_kind", "operation_set_target_kind"),
        ),
        "operation_set_operation_family": first_text(
            compiler.get("operation_family"),
            compiler.get("operation_set_operation_family"),
            first_operation_text("operation_family", "recommended_operation_family"),
        ),
        "materializer_id": first_text(
            compiler.get("materializer_id"),
            compiler.get("materializer"),
            first_operation_text("materializer_id", "materializer"),
        ),
        "receiver_contract_kind": first_text(
            compiler.get("receiver_contract_kind"),
            first_operation_text("receiver_contract_kind"),
        ),
        "selected_operation_count": len(selected_operations),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _water_bucket_portfolio_search(
    ordered: Sequence[Mapping[str, Any]],
    *,
    total_byte_budget: int | None,
) -> tuple[list[dict[str, Any]], int]:
    if total_byte_budget is None:
        return [dict(row) for row in ordered], 1 if ordered else 0
    if not ordered:
        return [], 0

    max_frontier_states = 4096
    states: list[tuple[tuple[int, ...], int, float, float]] = [((), 0, 0.0, 0.0)]
    for index, row in enumerate(ordered):
        cost = int(row["measure"]["water_fill_cost_bytes"])
        expected_gain = float(row["expected_score_gain"])
        lambda_rate = float(row["lambda_rate"])
        objective_gain = expected_gain - lambda_rate * float(cost)
        if cost > total_byte_budget or objective_gain <= 0.0:
            continue
        candidates = list(states)
        for selected_indices, used_bytes, gain, objective in states:
            next_cost = used_bytes + cost
            if next_cost > total_byte_budget:
                continue
            candidates.append(
                (
                    (*selected_indices, index),
                    next_cost,
                    gain + expected_gain,
                    objective + objective_gain,
                )
            )
        states = _water_bucket_pruned_frontier(
            candidates,
            max_frontier_states=max_frontier_states,
        )
    best = max(states, key=_water_bucket_state_sort_key)
    selected = [dict(ordered[index]) for index in best[0]]
    return selected, len(states)


def _water_bucket_pruned_frontier(
    states: Sequence[tuple[tuple[int, ...], int, float, float]],
    *,
    max_frontier_states: int,
) -> list[tuple[tuple[int, ...], int, float, float]]:
    best_by_cost: dict[int, tuple[tuple[int, ...], int, float, float]] = {}
    for state in states:
        prior = best_by_cost.get(state[1])
        if prior is None or _water_bucket_state_sort_key(state) > _water_bucket_state_sort_key(prior):
            best_by_cost[state[1]] = state
    frontier = sorted(
        best_by_cost.values(),
        key=_water_bucket_state_sort_key,
        reverse=True,
    )
    return frontier[:max_frontier_states]


def _water_bucket_state_sort_key(
    state: tuple[tuple[int, ...], int, float, float],
) -> tuple[float, float, int, int, tuple[int, ...]]:
    selected_indices, used_bytes, expected_gain, objective_gain = state
    return (
        objective_gain,
        expected_gain,
        -used_bytes,
        -len(selected_indices),
        tuple(-index for index in selected_indices),
    )


def _span(value: Any) -> int:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        return 0
    start = _int(value[0], "range[0]", minimum=0)
    end = _int(value[1], "range[1]", minimum=0)
    return max(0, end - start)


def _region_area(value: Any) -> float:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        return 0.0
    x_min = _float(value[0], "region_bbox[0]")
    y_min = _float(value[1], "region_bbox[1]")
    x_max = _float(value[2], "region_bbox[2]")
    y_max = _float(value[3], "region_bbox[3]")
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def action_surface_terms(atom: Mapping[str, Any]) -> dict[str, Any]:
    """Return the domain-math/action terms from a normalized or raw atom."""

    row = dict(atom) if atom.get("schema") == ATOM_SCHEMA else normalize_inverse_steganalysis_atom(atom)
    return {
        "scope_axis": row["scope_axis"],
        "first_order_marginal_effect": row["first_order_marginal_effect"],
        "second_order_interaction_effect": row["second_order_interaction_effect"],
        "interaction_kind": row["interaction_kind"],
        "synergy_effect": row["synergy_effect"],
        "antagonism_effect": row["antagonism_effect"],
        "discontinuity_risk": row["discontinuity_risk"],
        "fragility_penalty": row["fragility_penalty"],
        "discontinuity_guard": dict(row["discontinuity_guard"]),
    }


def _interaction_kind(value: float) -> str:
    if value > 0.0:
        return "synergy"
    if value < 0.0:
        return "antagonism"
    return "neutral"


def _split_performance_step_key(value: Any) -> tuple[str, str]:
    text = _text(value, "performance bucket key")
    if "." not in text:
        raise InverseSteganalysisAcquisitionError("performance bucket key must be experiment_id.step_id")
    experiment_id, step_id = text.split(".", 1)
    return _text(experiment_id, "performance experiment_id"), _text(
        step_id,
        "performance step_id",
    )


def _parse_candidate_lookup(
    raw: Mapping[str, Any] | None,
    *,
    label: str,
) -> dict[str, tuple[str, ...]]:
    lookup: dict[str, tuple[str, ...]] = {}
    if raw is None:
        return lookup
    if not isinstance(raw, Mapping):
        raise InverseSteganalysisAcquisitionError(f"{label} must be an object")
    for raw_key, raw_value in dict(raw).items():
        key = _text(raw_key, f"{label} key")
        if isinstance(raw_value, str):
            values = (_text(raw_value, f"{label}[{key!r}]"),)
        elif isinstance(raw_value, Sequence) and not isinstance(
            raw_value,
            (str, bytes, bytearray),
        ):
            values = tuple(_text(value, f"{label}[{key!r}][{index}]") for index, value in enumerate(raw_value))
            if not values:
                raise InverseSteganalysisAcquisitionError(f"{label}[{key!r}] must not be empty")
        else:
            raise InverseSteganalysisAcquisitionError(f"{label}[{key!r}] must be a string or list")
        lookup[key] = values
    return lookup


def _merge_candidate_lookup(
    base: Mapping[str, tuple[str, ...]],
    explicit: Mapping[str, tuple[str, ...]],
    *,
    conflict_label: str,
) -> dict[str, tuple[str, ...]]:
    out = dict(base)
    for key, values in explicit.items():
        existing = out.get(key)
        if existing is not None and existing != values:
            raise InverseSteganalysisAcquisitionError(
                f"candidate_id_by_experiment conflicts with {conflict_label}[{key!r}]"
            )
        out[key] = values
    return out


def _performance_identity_by_experiment(
    performance: Mapping[str, Any],
) -> dict[str, dict[str, list[str]]]:
    work_ids = _parse_single_value_lookup(
        performance.get("work_id_by_experiment"),
        label="performance.work_id_by_experiment",
    )
    backlog_keys = _parse_single_value_lookup(
        performance.get("backlog_key_by_experiment"),
        label="performance.backlog_key_by_experiment",
    )
    source_unit_ids = _parse_candidate_lookup(
        performance.get("source_unit_ids_by_experiment")
        if performance.get("source_unit_ids_by_experiment") is not None
        else None,
        label="performance.source_unit_ids_by_experiment",
    )
    source_selection_ids = _parse_candidate_lookup(
        performance.get("source_selection_ids_by_experiment")
        if performance.get("source_selection_ids_by_experiment") is not None
        else None,
        label="performance.source_selection_ids_by_experiment",
    )
    experiment_ids = set(work_ids) | set(backlog_keys) | set(source_unit_ids) | set(source_selection_ids)
    out: dict[str, dict[str, list[str]]] = {}
    for experiment_id in experiment_ids:
        out[experiment_id] = {
            "work_ids": list(work_ids.get(experiment_id, ())),
            "backlog_keys": list(backlog_keys.get(experiment_id, ())),
            "source_unit_ids": list(source_unit_ids.get(experiment_id, ())),
            "source_selection_ids": list(source_selection_ids.get(experiment_id, ())),
        }
    return out


def _parse_single_value_lookup(
    raw: Any,
    *,
    label: str,
) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise InverseSteganalysisAcquisitionError(f"{label} must be an object")
    out: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_value in raw.items():
        key = _text(raw_key, f"{label} key")
        out[key] = (_text(raw_value, f"{label}[{key!r}]"),)
    return out


def _queue_materializer_delta_observations_for_observed_step(
    step: Mapping[str, Any],
    *,
    queue_id: str,
    section: str,
    axis: str,
    source_path: str | None,
    runtime_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    candidate_lookup: Mapping[str, tuple[str, ...]],
    performance_identity: Mapping[str, Mapping[str, list[str]]],
) -> list[dict[str, Any]]:
    experiment_id = _text(step.get("experiment_id"), f"{section}.experiment_id")
    step_id = _text(step.get("step_id"), f"{section}.step_id")
    candidate_ids = _queue_step_candidate_ids(
        step,
        experiment_id=experiment_id,
        candidate_lookup=candidate_lookup,
        section=section,
    )
    identity = performance_identity.get(experiment_id, {})
    return _queue_materializer_delta_observations_for_step(
        step,
        queue_id=queue_id,
        experiment_id=experiment_id,
        step_id=step_id,
        candidate_ids=candidate_ids,
        axis=axis,
        source_path=source_path,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        resource_kind=_resource(step.get("resource_kind") or "local_cpu"),
        work_ids=ordered_unique([*_list_strings(step.get("work_ids")), *(identity.get("work_ids") or [])]),
        backlog_keys=ordered_unique(
            [
                *_list_strings(step.get("backlog_keys")),
                *(identity.get("backlog_keys") or []),
            ]
        ),
        source_unit_ids=ordered_unique(
            [
                *_list_strings(step.get("source_unit_ids")),
                *(identity.get("source_unit_ids") or []),
            ]
        ),
        source_selection_ids=ordered_unique(
            [
                *_list_strings(step.get("source_selection_ids")),
                *(identity.get("source_selection_ids") or []),
            ]
        ),
    )


def _queue_health_observations_for_step(
    step: Mapping[str, Any],
    *,
    queue_id: str,
    section: str,
    kind: str,
    axis: str,
    source_path: str | None,
    runtime_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    candidate_lookup: Mapping[str, tuple[str, ...]],
    performance_identity: Mapping[str, Mapping[str, list[str]]],
    observation_blockers: Sequence[str],
    healthy: bool,
) -> list[dict[str, Any]]:
    experiment_id = _text(step.get("experiment_id"), f"{section}.experiment_id")
    step_id = _text(step.get("step_id"), f"{section}.step_id")
    candidate_ids = _queue_step_candidate_ids(
        step,
        experiment_id=experiment_id,
        candidate_lookup=candidate_lookup,
        section=section,
    )
    status = _optional_text(step.get("status")) or kind
    blockers = _queue_step_health_blockers(
        step,
        section=section,
        kind=kind,
        status=status,
        observation_blockers=observation_blockers,
    )
    artifact_paths = _queue_step_expected_artifact_paths(step)
    artifact_bytes = _queue_step_artifact_bytes(step)
    readiness_blockers = _queue_step_materializer_readiness_blockers(step)
    receiver_contract_satisfied = _queue_step_receiver_contract_satisfied(step)
    resource_kind = _resource(step.get("resource_kind") or "local_cpu")
    identity = performance_identity.get(experiment_id, {})
    work_ids = ordered_unique([*_list_strings(step.get("work_ids")), *(identity.get("work_ids") or [])])
    backlog_keys = ordered_unique([*_list_strings(step.get("backlog_keys")), *(identity.get("backlog_keys") or [])])
    source_unit_ids = ordered_unique(
        [
            *_list_strings(step.get("source_unit_ids")),
            *(identity.get("source_unit_ids") or []),
        ]
    )
    source_selection_ids = ordered_unique(
        [
            *_list_strings(step.get("source_selection_ids")),
            *(identity.get("source_selection_ids") or []),
        ]
    )
    elapsed_seconds = _float_or_none(
        step.get("timeout_seconds"),
        f"{section}.timeout_seconds",
        minimum=0.0,
        exclusive=True,
    )
    base_observation_id = f"queue_obs_{_slug(queue_id)}_{_slug(experiment_id)}_{_slug(step_id)}_{_slug(kind)}"
    out: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        observation_id = base_observation_id
        if len(candidate_ids) > 1:
            observation_id = f"{base_observation_id}_{_slug(candidate_id)}"
        out.append(
            normalize_inverse_steganalysis_observation(
                {
                    "observation_id": observation_id,
                    "observation_kind": "queue_observation_health_blocker",
                    "candidate_id": candidate_id,
                    "axis": axis,
                    "source_path": source_path,
                    "queue_id": queue_id,
                    "target_kind": _optional_text(step.get("target_kind")),
                    "materializer_id": _optional_text(_first(step.get("materializer_id"), step.get("materializer"))),
                    "receiver_contract_kind": _optional_text(step.get("receiver_contract_kind")),
                    "experiment_id": experiment_id,
                    "step_id": step_id,
                    "queue_observation_schema": QUEUE_OBSERVATION_SCHEMA,
                    "queue_observation_health": healthy,
                    "queue_observation_status": status,
                    "queue_observation_blockers": blockers,
                    "receiver_contract_satisfied": receiver_contract_satisfied,
                    "readiness_blockers": readiness_blockers,
                    "candidate_ids": list(candidate_ids),
                    "work_ids": work_ids,
                    "backlog_keys": backlog_keys,
                    "source_unit_ids": source_unit_ids,
                    "source_selection_ids": source_selection_ids,
                    "expected_artifact_paths": artifact_paths,
                    "runtime_identity": runtime_identity,
                    "cache_identity": cache_identity,
                    "observed_score_gain": 0.0,
                    "calibration_error": 1.0,
                    "elapsed_seconds": elapsed_seconds,
                    "artifact_bytes": artifact_bytes,
                    "resource_kind": resource_kind,
                    "allowed_use": "queue_health_feedback_only",
                    "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
                }
            )
        )
    out.extend(
        _queue_materializer_delta_observations_for_step(
            step,
            queue_id=queue_id,
            experiment_id=experiment_id,
            step_id=step_id,
            candidate_ids=candidate_ids,
            axis=axis,
            source_path=source_path,
            runtime_identity=runtime_identity,
            cache_identity=cache_identity,
            resource_kind=resource_kind,
            work_ids=work_ids,
            backlog_keys=backlog_keys,
            source_unit_ids=source_unit_ids,
            source_selection_ids=source_selection_ids,
        )
    )
    return out


def _resolve_queue_artifact_path(
    path_text: str,
    *,
    source_path: str | None,
) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    if source_path:
        source = Path(source_path).expanduser()
        source_parent = source.parent if source.name else source
        candidate = source_parent / path
        if candidate.exists():
            return candidate.resolve(strict=False)
    return path.resolve(strict=False)


def _queue_materializer_empirical_rows_from_artifact(
    artifact_path: str,
    *,
    source_path: str | None,
) -> list[dict[str, Any]]:
    path = _resolve_queue_artifact_path(artifact_path, source_path=source_path)
    if not path.is_file():
        return []
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InverseSteganalysisAcquisitionError(
            f"queue materializer observation artifact unreadable: {artifact_path}: {exc}"
        ) from exc
    if size > QUEUE_MATERIALIZER_OBSERVATION_ARTIFACT_MAX_BYTES:
        raise InverseSteganalysisAcquisitionError(
            f"queue materializer observation artifact exceeds byte cap: {artifact_path} ({size} bytes)"
        )
    if path.suffix == ".jsonl":
        return _queue_materializer_empirical_rows_from_jsonl(path, artifact_path)
    if path.suffix == ".json":
        return _queue_materializer_empirical_rows_from_json(path, artifact_path)
    return []


def _queue_materializer_empirical_rows_from_jsonl(
    path: Path,
    display_path: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_index, raw_line in enumerate(handle, start=1):
                if line_index > QUEUE_MATERIALIZER_OBSERVATION_ARTIFACT_MAX_ROWS:
                    raise InverseSteganalysisAcquisitionError(
                        f"queue materializer observation artifact exceeds row cap: {display_path}"
                    )
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InverseSteganalysisAcquisitionError(
                        f"queue materializer observation artifact has invalid JSONL: {display_path}:{line_index}: {exc}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise InverseSteganalysisAcquisitionError(
                        f"queue materializer observation artifact row must be an object: {display_path}:{line_index}"
                    )
                if payload.get("schema") == FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA:
                    rows.append(dict(payload))
    except OSError as exc:
        raise InverseSteganalysisAcquisitionError(
            f"queue materializer observation artifact unreadable: {display_path}: {exc}"
        ) from exc
    return rows


def _queue_materializer_empirical_rows_from_json(
    path: Path,
    display_path: str,
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InverseSteganalysisAcquisitionError(
            f"queue materializer observation artifact has invalid JSON: {display_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        return []
    schema = payload.get("schema")
    if schema == FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA:
        return [dict(payload)]
    if schema != FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA:
        return []
    observations = payload.get("observations")
    if not isinstance(observations, Sequence) or isinstance(
        observations,
        (bytes, bytearray, str),
    ):
        raise InverseSteganalysisAcquisitionError(
            f"queue materializer sweep artifact observations must be a list: {display_path}"
        )
    if len(observations) > QUEUE_MATERIALIZER_OBSERVATION_ARTIFACT_MAX_ROWS:
        raise InverseSteganalysisAcquisitionError(f"queue materializer sweep artifact exceeds row cap: {display_path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(observations, start=1):
        if not isinstance(row, Mapping):
            raise InverseSteganalysisAcquisitionError(
                f"queue materializer sweep observation row must be an object: {display_path}:observations[{index}]"
            )
        if row.get("schema") != FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA:
            raise InverseSteganalysisAcquisitionError(
                f"queue materializer sweep observation row has unexpected schema: {display_path}:observations[{index}]"
            )
        rows.append(dict(row))
    return rows


def _queue_materializer_empirical_observations_from_artifact(
    artifact: Mapping[str, Any],
    *,
    artifact_path: str,
    artifact_index: int,
    queue_id: str,
    experiment_id: str,
    step_id: str,
    candidate_ids: Sequence[str],
    axis: str,
    source_path: str | None,
    runtime_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    resource_kind: str,
    work_ids: Sequence[str],
    backlog_keys: Sequence[str],
    source_unit_ids: Sequence[str],
    source_selection_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows = _queue_materializer_empirical_rows_from_artifact(
        artifact_path,
        source_path=source_path,
    )
    out: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(rows):
        label = f"queue materializer observation artifact {artifact_path}[{row_index}]"
        _reject_truthy_authority(raw_row, label=label)
        _require_explicit_false_authority(raw_row, label=label)
        row = dict(raw_row)
        candidate_id = _optional_text(row.get("candidate_id"))
        if candidate_id is None:
            candidate_id = next((item for item in candidate_ids if item), None)
        if candidate_id is None:
            candidate_id = (
                f"queue_materializer_empirical_{_slug(queue_id)}_"
                f"{_slug(experiment_id)}_{_slug(step_id)}_"
                f"{artifact_index:04d}_{row_index:04d}"
            )
        row.setdefault(
            "observation_id",
            (
                f"queue_materializer_empirical_{_slug(queue_id)}_"
                f"{_slug(experiment_id)}_{_slug(step_id)}_"
                f"{artifact_index:04d}_{row_index:04d}"
            ),
        )
        row.setdefault(
            "observation_kind",
            FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_KIND,
        )
        row["candidate_id"] = candidate_id
        row.setdefault("axis", axis)
        row.setdefault("source_path", artifact_path or source_path)
        row.setdefault("queue_id", queue_id)
        row.setdefault("experiment_id", experiment_id)
        row.setdefault("step_id", step_id)
        row.setdefault("runtime_identity", dict(runtime_identity))
        row.setdefault("cache_identity", dict(cache_identity))
        row.setdefault("resource_kind", resource_kind)
        row["candidate_ids"] = ordered_unique(
            [
                *_list_strings(row.get("candidate_ids")),
                candidate_id,
                *_list_strings(artifact.get("candidate_ids")),
                *candidate_ids,
            ]
        )
        row["work_ids"] = ordered_unique([*_list_strings(row.get("work_ids")), *work_ids])
        row["backlog_keys"] = ordered_unique([*_list_strings(row.get("backlog_keys")), *backlog_keys])
        row["source_unit_ids"] = ordered_unique([*_list_strings(row.get("source_unit_ids")), *source_unit_ids])
        row["source_selection_ids"] = ordered_unique(
            [
                *_list_strings(row.get("source_selection_ids")),
                *source_selection_ids,
            ]
        )
        row["expected_artifact_paths"] = ordered_unique(
            [*_list_strings(row.get("expected_artifact_paths")), artifact_path]
        )
        if row.get("materializer_rate_outcome") is None:
            row["materializer_rate_outcome"] = row.get("archive_delta_status")
        if row.get("archive_delta_status") is None:
            row["archive_delta_status"] = row.get("materializer_rate_outcome")
        if row.get("signal_semantics") is None:
            row["signal_semantics"] = (
                "realized_archive_saving"
                if row.get("rate_positive") is True
                else "receiver_or_rate_blocked_materializer_feedback"
            )
        out.append(normalize_inverse_steganalysis_observation(row))
    return out


def _queue_materializer_delta_observations_for_step(
    step: Mapping[str, Any],
    *,
    queue_id: str,
    experiment_id: str,
    step_id: str,
    candidate_ids: Sequence[str],
    axis: str,
    source_path: str | None,
    runtime_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    resource_kind: str,
    work_ids: Sequence[str],
    backlog_keys: Sequence[str],
    source_unit_ids: Sequence[str],
    source_selection_ids: Sequence[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for artifact_index, artifact in enumerate(_sequence_of_mappings(step.get("expected_artifacts"))):
        artifact_path = _optional_text(artifact.get("path"))
        if artifact_path:
            out.extend(
                _queue_materializer_empirical_observations_from_artifact(
                    artifact,
                    artifact_path=artifact_path,
                    artifact_index=artifact_index,
                    queue_id=queue_id,
                    experiment_id=experiment_id,
                    step_id=step_id,
                    candidate_ids=candidate_ids,
                    axis=axis,
                    source_path=source_path,
                    runtime_identity=runtime_identity,
                    cache_identity=cache_identity,
                    resource_kind=resource_kind,
                    work_ids=work_ids,
                    backlog_keys=backlog_keys,
                    source_unit_ids=source_unit_ids,
                    source_selection_ids=source_selection_ids,
                )
            )
        delta_status = _optional_text(artifact.get("serialized_archive_delta_status"))
        saved_bytes = _optional_int(
            _first(
                artifact.get("serialized_archive_delta_realized_saved_bytes"),
                artifact.get("section_recode_saved_bytes"),
                artifact.get("selected_compression_saved_bytes"),
                artifact.get("selected_merge_saved_bytes"),
                artifact.get("selected_payload_saved_bytes"),
                artifact.get("selected_elision_saved_bytes"),
                artifact.get("factorization_saved_bytes"),
            ),
            "serialized_archive_delta_realized_saved_bytes",
        )
        has_receiver_signal = "receiver_contract_satisfied" in artifact
        receiver = artifact.get("receiver_verification")
        if isinstance(receiver, Mapping) and "receiver_contract_satisfied" in receiver:
            has_receiver_signal = True
        if delta_status is None and saved_bytes is None and not has_receiver_signal:
            continue
        target_kind = _optional_text(_first(artifact.get("target_kind"), step.get("target_kind")))
        materializer_id = _optional_text(
            _first(
                artifact.get("materializer_id"),
                step.get("materializer_id"),
                step.get("materializer"),
            )
        )
        receiver_contract_kind = _optional_text(
            _first(
                artifact.get("receiver_contract_kind"),
                step.get("receiver_contract_kind"),
            )
        )
        if target_kind is None and materializer_id is None and receiver_contract_kind is None:
            continue
        if saved_bytes is None:
            saved_bytes = 0
        if delta_status is None:
            if saved_bytes > 0:
                delta_status = "realized_saving"
            elif saved_bytes < 0:
                delta_status = "realized_cost"
        savings_realized = artifact.get("serialized_archive_delta_savings_realized") is True or saved_bytes > 0
        rate_positive = saved_bytes > 0 and delta_status == "realized_saving" and savings_realized
        receiver_contract_satisfied = _artifact_receiver_contract_satisfied(artifact)
        readiness_blockers = ordered_unique(
            [
                *_list_strings(artifact.get("readiness_blockers")),
                *(
                    [f"receiver_verification:{item}" for item in _list_strings(receiver.get("blockers"))]
                    if isinstance(receiver, Mapping)
                    else []
                ),
                *(
                    ["receiver_contract_not_satisfied"]
                    if has_receiver_signal and not receiver_contract_satisfied
                    else []
                ),
            ]
        )
        artifact_bytes = _optional_int(artifact.get("bytes"), "artifact.bytes", minimum=0) or 0
        candidate_archive = artifact.get("candidate_archive")
        candidate_archive_bytes = _optional_int(
            _first(
                artifact.get("serialized_archive_delta_candidate_archive_bytes"),
                artifact.get("section_recode_candidate_archive_bytes"),
                artifact.get("selected_compression_candidate_archive_bytes"),
                artifact.get("selected_merge_candidate_archive_bytes"),
                artifact.get("selected_payload_candidate_archive_bytes"),
                artifact.get("selected_elision_candidate_archive_bytes"),
                artifact.get("factorization_candidate_archive_bytes"),
                candidate_archive.get("bytes") if isinstance(candidate_archive, Mapping) else None,
            ),
            "candidate_archive_bytes",
            minimum=0,
        )
        candidate_archive_sha256 = _sha256_or_none(
            candidate_archive.get("sha256") if isinstance(candidate_archive, Mapping) else None,
            "candidate_archive_sha256",
        )
        base_observation_id = (
            f"queue_materializer_delta_{_slug(queue_id)}_{_slug(experiment_id)}_{_slug(step_id)}_{artifact_index:04d}"
        )
        artifact_candidate_ids = tuple(
            dict.fromkeys(
                [
                    *_list_strings(artifact.get("candidate_ids")),
                    *(
                        [_optional_text(artifact.get("candidate_id"))]
                        if _optional_text(artifact.get("candidate_id")) is not None
                        else []
                    ),
                    *candidate_ids,
                ]
            )
        )
        for candidate_id in artifact_candidate_ids:
            if not candidate_id:
                continue
            observation_id = base_observation_id
            if len(artifact_candidate_ids) > 1:
                observation_id = f"{base_observation_id}_{_slug(candidate_id)}"
            out.append(
                normalize_inverse_steganalysis_observation(
                    {
                        "observation_id": observation_id,
                        "observation_kind": MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND,
                        "candidate_id": candidate_id,
                        "axis": axis,
                        "source_path": source_path,
                        "queue_id": queue_id,
                        "experiment_id": experiment_id,
                        "step_id": step_id,
                        "target_kind": target_kind,
                        "materializer_id": materializer_id,
                        "receiver_contract_kind": receiver_contract_kind,
                        "saved_bytes": saved_bytes,
                        "observed_rate_gain": (
                            CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes) if rate_positive else 0.0
                        ),
                        "observed_score_gain": (
                            CONTEST_RATE_SCORE_PER_BYTE * float(saved_bytes) if rate_positive else 0.0
                        ),
                        "rate_positive": rate_positive,
                        "savings_realized": savings_realized,
                        "receiver_contract_satisfied": receiver_contract_satisfied,
                        "materializer_rate_outcome": delta_status,
                        "signal_semantics": (
                            "realized_archive_saving"
                            if rate_positive
                            else "receiver_or_rate_blocked_materializer_feedback"
                        ),
                        "archive_delta_status": delta_status,
                        "source_archive_bytes": _optional_int(
                            _first(
                                artifact.get("serialized_archive_delta_source_archive_bytes"),
                                artifact.get("section_recode_source_archive_bytes"),
                                artifact.get("selected_compression_source_archive_bytes"),
                                artifact.get("selected_merge_source_archive_bytes"),
                                artifact.get("selected_payload_source_archive_bytes"),
                                artifact.get("selected_elision_source_archive_bytes"),
                                artifact.get("factorization_source_archive_bytes"),
                            ),
                            "source_archive_bytes",
                            minimum=0,
                        ),
                        "candidate_archive_bytes": candidate_archive_bytes,
                        "candidate_archive_sha256": candidate_archive_sha256,
                        "artifact_bytes": artifact_bytes,
                        "resource_kind": resource_kind,
                        "candidate_ids": list(artifact_candidate_ids),
                        "work_ids": list(work_ids),
                        "backlog_keys": list(backlog_keys),
                        "source_unit_ids": list(source_unit_ids),
                        "source_selection_ids": list(source_selection_ids),
                        "expected_artifact_paths": [artifact_path] if artifact_path else [],
                        "readiness_blockers": readiness_blockers,
                        "runtime_identity": runtime_identity,
                        "cache_identity": cache_identity,
                        "allowed_use": "materializer_queue_feedback_only",
                        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
                    }
                )
            )
    return _merge_materializer_archive_delta_observations(out)


_QUEUE_MATERIALIZER_DELTA_MERGE_SEQUENCE_KEYS = (
    "candidate_ids",
    "work_ids",
    "backlog_keys",
    "source_unit_ids",
    "source_selection_ids",
    "expected_artifact_paths",
    "readiness_blockers",
)


def _materializer_archive_delta_merge_key(row: Mapping[str, Any]) -> str:
    artifact_path = _optional_text(_first(*_list_strings(row.get("expected_artifact_paths"))))
    candidate_sha = _optional_text(row.get("candidate_archive_sha256"))
    candidate_id = _optional_text(row.get("candidate_id"))
    target_kind = _optional_text(row.get("target_kind"))
    materializer_id = _optional_text(row.get("materializer_id"))
    receiver_contract_kind = _optional_text(row.get("receiver_contract_kind"))
    identity = candidate_sha or _optional_text(row.get("observation_id")) or artifact_path
    if identity is None:
        identity = _sha256_json_value(dict(row))
    return "|".join(
        str(item or "")
        for item in (
            candidate_id,
            target_kind,
            materializer_id,
            receiver_contract_kind,
            identity,
        )
    )


def _sha256_json_value(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _merge_materializer_archive_delta_observations(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_dict = dict(row)
        key = _materializer_archive_delta_merge_key(row_dict)
        existing = merged.get(key)
        if existing is None:
            merged[key] = row_dict
            continue
        for field in _QUEUE_MATERIALIZER_DELTA_MERGE_SEQUENCE_KEYS:
            values = ordered_unique(
                [
                    *_list_strings(existing.get(field)),
                    *_list_strings(row_dict.get(field)),
                ]
            )
            if values:
                existing[field] = values
        if row_dict.get("receiver_contract_satisfied") is False:
            existing["receiver_contract_satisfied"] = False
        if row_dict.get("rate_positive") is True:
            existing["rate_positive"] = True
        if row_dict.get("savings_realized") is True:
            existing["savings_realized"] = True
        saved_bytes = _merge_saved_bytes(existing.get("saved_bytes"), row_dict.get("saved_bytes"))
        if saved_bytes is not None:
            existing["saved_bytes"] = saved_bytes
        for field in ("observed_rate_gain", "observed_score_gain"):
            current_float = _float_or_none(existing.get(field), field, minimum=0.0)
            incoming_float = _float_or_none(row_dict.get(field), field, minimum=0.0)
            if incoming_float is not None and (current_float is None or incoming_float > current_float):
                existing[field] = incoming_float
        for field in ("source_archive_bytes", "candidate_archive_bytes", "artifact_bytes"):
            current = _optional_int(existing.get(field), field, minimum=0)
            incoming = _optional_int(row_dict.get(field), field, minimum=0)
            if incoming is not None and (current is None or incoming > current):
                existing[field] = incoming
        for field, value in row_dict.items():
            if field in _QUEUE_MATERIALIZER_DELTA_MERGE_SEQUENCE_KEYS:
                continue
            if existing.get(field) in (None, "", [], {}) and value not in (
                None,
                "",
                [],
                {},
            ):
                existing[field] = value
    return list(merged.values())


def _merge_saved_bytes(*values: Any) -> int | None:
    parsed = [parsed_value for value in values if (parsed_value := _optional_int(value, "saved_bytes")) is not None]
    if not parsed:
        return None
    positive = [value for value in parsed if value > 0]
    if positive:
        return max(positive)
    return min(parsed)


def _artifact_receiver_contract_satisfied(artifact: Mapping[str, Any]) -> bool:
    if "receiver_contract_satisfied" in artifact:
        return artifact.get("receiver_contract_satisfied") is True
    receiver = artifact.get("receiver_verification")
    if isinstance(receiver, Mapping) and "receiver_contract_satisfied" in receiver:
        return receiver.get("receiver_contract_satisfied") is True
    return False


def _queue_global_health_observations(
    observation: Mapping[str, Any],
    *,
    queue_id: str,
    axis: str,
    source_path: str | None,
    runtime_identity: Mapping[str, Any],
    cache_identity: Mapping[str, Any],
    candidate_lookup: Mapping[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    candidate_ids = tuple(
        dict.fromkeys(candidate_id for values in candidate_lookup.values() for candidate_id in values)
    )
    if not candidate_ids:
        raise InverseSteganalysisAcquisitionError(
            "unhealthy queue observation has global blockers but no candidate identity; pass candidate_id_by_experiment"
        )
    blockers = _list_strings(observation.get("blockers"))
    out: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        out.append(
            normalize_inverse_steganalysis_observation(
                {
                    "observation_id": (f"queue_obs_{_slug(queue_id)}_global_health_{_slug(candidate_id)}"),
                    "observation_kind": "queue_observation_global_health_blocker",
                    "candidate_id": candidate_id,
                    "axis": axis,
                    "source_path": source_path,
                    "queue_id": queue_id,
                    "queue_observation_schema": QUEUE_OBSERVATION_SCHEMA,
                    "queue_observation_health": False,
                    "queue_observation_status": "global_unhealthy",
                    "queue_observation_blockers": blockers,
                    "candidate_ids": list(candidate_ids),
                    "runtime_identity": runtime_identity,
                    "cache_identity": cache_identity,
                    "observed_score_gain": 0.0,
                    "calibration_error": 1.0,
                    "artifact_bytes": 0,
                    "resource_kind": "local_cpu",
                    "allowed_use": "queue_health_feedback_only",
                    "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
                }
            )
        )
    return out


def _orphaned_step_blocks_queue(step: Mapping[str, Any]) -> bool:
    return str(step.get("status") or "") in {"queued", "running", "blocked"}


def _queue_observation_has_blocking_global_health(
    observation: Mapping[str, Any],
) -> bool:
    blockers = _list_strings(observation.get("blockers"))
    if any(blocker for blocker in blockers if not blocker.startswith("experiment_queue_observation_orphaned_steps")):
        return True
    orphaned_steps = _sequence_of_mappings(observation.get("orphaned_steps"))
    if not orphaned_steps:
        return any(blocker.startswith("experiment_queue_observation_orphaned_steps") for blocker in blockers)
    return any(_orphaned_step_blocks_queue(step) for step in orphaned_steps)


def _queue_step_candidate_ids(
    step: Mapping[str, Any],
    *,
    experiment_id: str,
    candidate_lookup: Mapping[str, tuple[str, ...]],
    section: str,
) -> tuple[str, ...]:
    mapped = candidate_lookup.get(experiment_id)
    if mapped:
        return mapped
    step_candidate_ids = tuple(dict.fromkeys(_list_strings(step.get("candidate_ids"))))
    if step_candidate_ids:
        return step_candidate_ids
    artifact_candidate_ids = tuple(
        dict.fromkeys(
            _text(artifact.get("candidate_id"), f"{section}.artifact.candidate_id")
            for artifact in _sequence_of_mappings(step.get("expected_artifacts"))
            if artifact.get("candidate_id") is not None
        )
    )
    if artifact_candidate_ids:
        return artifact_candidate_ids
    raise InverseSteganalysisAcquisitionError(
        "queue observation health step missing candidate identity for "
        f"experiment {experiment_id!r}; pass candidate_id_by_experiment"
    )


def _queue_step_health_blockers(
    step: Mapping[str, Any],
    *,
    section: str,
    kind: str,
    status: str,
    observation_blockers: Sequence[str],
) -> list[str]:
    blockers = [kind, f"queue_observation_step_status:{status}"]
    blockers.extend(str(item) for item in observation_blockers if str(item))
    for artifact in _sequence_of_mappings(step.get("expected_artifacts")):
        if artifact.get("postcondition_passed") is False:
            path = _optional_text(artifact.get("path")) or "unknown"
            blockers.append(f"queue_observation_artifact_postcondition_failed:{path}")
        if artifact.get("exists") is False:
            path = _optional_text(artifact.get("path")) or "unknown"
            blockers.append(f"queue_observation_artifact_missing:{path}")
        if artifact.get("receiver_contract_satisfied") is False:
            path = _optional_text(artifact.get("path")) or "unknown"
            blockers.append(f"queue_observation_receiver_contract_unsatisfied:{path}")
        for readiness_blocker in _list_strings(artifact.get("readiness_blockers")):
            blockers.append(f"queue_observation_materializer_readiness_blocker:{readiness_blocker}")
        receiver = artifact.get("receiver_verification")
        if isinstance(receiver, Mapping):
            for receiver_blocker in _list_strings(receiver.get("blockers")):
                blockers.append(f"queue_observation_receiver_verification_blocker:{receiver_blocker}")
    return ordered_unique(blockers)


def _queue_step_materializer_readiness_blockers(step: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for artifact in _sequence_of_mappings(step.get("expected_artifacts")):
        blockers.extend(_list_strings(artifact.get("readiness_blockers")))
        receiver = artifact.get("receiver_verification")
        if isinstance(receiver, Mapping):
            blockers.extend(f"receiver_verification:{item}" for item in _list_strings(receiver.get("blockers")))
    return ordered_unique(blockers)


def _queue_step_receiver_contract_satisfied(step: Mapping[str, Any]) -> bool:
    saw_receiver_artifact = False
    for artifact in _sequence_of_mappings(step.get("expected_artifacts")):
        if "receiver_contract_satisfied" in artifact:
            saw_receiver_artifact = True
            if artifact.get("receiver_contract_satisfied") is not True:
                return False
        receiver = artifact.get("receiver_verification")
        if isinstance(receiver, Mapping) and "receiver_contract_satisfied" in receiver:
            saw_receiver_artifact = True
            if receiver.get("receiver_contract_satisfied") is not True:
                return False
    return bool(saw_receiver_artifact)


def _queue_step_expected_artifact_paths(step: Mapping[str, Any]) -> list[str]:
    return [
        path
        for path in (
            _optional_text(artifact.get("path")) for artifact in _sequence_of_mappings(step.get("expected_artifacts"))
        )
        if path is not None
    ]


def _queue_step_artifact_bytes(step: Mapping[str, Any]) -> int:
    total = 0
    for artifact in _sequence_of_mappings(step.get("expected_artifacts")):
        value = artifact.get("bytes")
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            total += parsed
    return total


def _dominant_resource_from_counts(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    counts: list[tuple[str, int]] = []
    for key, item in value.items():
        count = _int(item, "resource_kind_counts value", minimum=0)
        if count > 0:
            counts.append((_resource(key), count))
    if not counts:
        return None
    return sorted(counts, key=lambda item: (-item[1], item[0]))[0][0]


def _queue_performance_artifact_bytes(bucket: Mapping[str, Any]) -> int:
    for key in (
        "artifact_record_bytes_mean",
        "artifact_record_raw_bytes_mean",
        "artifact_record_bytes_sum",
    ):
        value = _float_or_none(bucket.get(key), key, minimum=0.0)
        if value is not None:
            return math.ceil(value)
    return 0


def _pair_indices_from_bucket(value: Any) -> list[int] | None:
    if not isinstance(value, str) or not value.startswith("pair_"):
        return None
    indices: list[int] = []
    for token in value.removeprefix("pair_").split("_"):
        if not token:
            continue
        try:
            indices.append(int(token))
        except ValueError:
            return None
    return indices or None


def _component_from_dominant_axis(axis: str) -> str:
    if axis == "seg":
        return "segnet"
    if axis == "pose":
        return "posenet"
    if axis == "rate":
        return "rate"
    return "scorer"


def _decision_surface_risk(decision_class: str) -> float:
    if decision_class == "rate_only_null_space":
        return 0.05
    if decision_class == "receiver_sufficient_statistic":
        return 0.2
    if decision_class == "fragile_boundary":
        return 1.0
    return 0.4


def _review_packet_axis(packet: Mapping[str, Any]) -> str:
    if packet.get("schema") != "tac_result_review_packet_v1":
        raise InverseSteganalysisAcquisitionError("review packet schema must be tac_result_review_packet_v1")
    axis = _text(packet.get("score_axis"), "review_packet.score_axis")
    if axis not in {"contest_cpu", "contest_cuda"}:
        raise InverseSteganalysisAcquisitionError(f"unsupported review packet score_axis {axis!r}")
    if axis == "contest_cpu" and packet.get("exact_cpu_evidence") is not True:
        raise InverseSteganalysisAcquisitionError("contest_cpu review packet must have exact_cpu_evidence=true")
    if axis == "contest_cuda" and packet.get("exact_cuda_evidence") is not True:
        raise InverseSteganalysisAcquisitionError("contest_cuda review packet must have exact_cuda_evidence=true")
    return axis


def _validate_paired_exact_auth_packets(by_axis: Mapping[str, Mapping[str, Any]]) -> None:
    archive_sha = _review_packet_archive_sha(by_axis["contest_cpu"])
    archive_bytes = _review_packet_archive_bytes(by_axis["contest_cpu"])
    n_samples = _review_packet_n_samples(by_axis["contest_cpu"])
    runtime_content = _runtime_custody_text(
        by_axis["contest_cpu"],
        "runtime_content_tree_sha256",
        required=True,
    )
    for axis, packet in sorted(by_axis.items()):
        _validate_exact_auth_packet_for_calibration(packet, axis=axis)
        if _review_packet_archive_sha(packet) != archive_sha:
            raise InverseSteganalysisAcquisitionError("paired exact-auth packets must share archive_sha256")
        if _review_packet_archive_bytes(packet) != archive_bytes:
            raise InverseSteganalysisAcquisitionError("paired exact-auth packets must share archive_bytes")
        if _review_packet_n_samples(packet) != n_samples:
            raise InverseSteganalysisAcquisitionError("paired exact-auth packets must share n_samples")
        packet_runtime_content = _runtime_custody_text(
            packet,
            "runtime_content_tree_sha256",
            required=True,
        )
        if packet_runtime_content != runtime_content:
            raise InverseSteganalysisAcquisitionError(
                "paired exact-auth packets must share runtime_content_tree_sha256"
            )


def _validate_exact_auth_packet_for_calibration(
    packet: Mapping[str, Any],
    *,
    axis: str,
) -> None:
    if packet.get("score_claim") is not False:
        raise InverseSteganalysisAcquisitionError(f"{axis} review packet score_claim must be false")
    for key in ("promotion_eligible", "rank_or_kill_eligible", "ready_for_exact_eval_dispatch"):
        if packet.get(key) is not False:
            raise InverseSteganalysisAcquisitionError(f"{axis} review packet {key} must be false")
    if packet.get("family_falsified") is True or packet.get("method_family_retired") is True:
        raise InverseSteganalysisAcquisitionError(f"{axis} review packet must not retire a family")
    status = _text(
        packet.get("measured_config_status"),
        f"{axis}.measured_config_status",
    )
    if status.startswith("indeterminate"):
        raise InverseSteganalysisAcquisitionError(f"{axis} review packet status is indeterminate")
    baseline = packet.get("baseline_score")
    if baseline is None:
        raise InverseSteganalysisAcquisitionError(f"{axis} review packet baseline_score is required")
    _float(baseline, f"{axis}.baseline_score", minimum=0.0)
    _float(packet.get("canonical_score"), f"{axis}.canonical_score", minimum=0.0)
    recompute = _packet_mapping(packet, "score_recomputation")
    if recompute.get("available") is not True:
        raise InverseSteganalysisAcquisitionError(f"{axis} score recomputation must be available")
    if recompute.get("matches_reported") is not True:
        raise InverseSteganalysisAcquisitionError(f"{axis} score recomputation must match reported score")
    audit = _packet_mapping(packet, "engineering_forensic_audit")
    if audit.get("engineering_or_config_bug_found") is True:
        raise InverseSteganalysisAcquisitionError(f"{axis} engineering forensic audit has blockers")
    if audit.get("score_formula_reviewed") is not True:
        raise InverseSteganalysisAcquisitionError(f"{axis} engineering forensic audit must review score formula")
    if audit.get("archive_runtime_closure_reviewed") is not True:
        raise InverseSteganalysisAcquisitionError(f"{axis} engineering forensic audit must review runtime closure")
    claim = _packet_mapping(packet, "dispatch_claim_state")
    if claim.get("terminal_status_recorded") is not True:
        raise InverseSteganalysisAcquisitionError(f"{axis} review packet must have a terminal dispatch claim")
    _runtime_custody_text(packet, "runtime_content_tree_sha256", required=True)
    _runtime_custody_text(packet, "inflated_output_aggregate_sha256", required=True)


def _packet_mapping(packet: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = packet.get(key)
    if not isinstance(value, Mapping):
        raise InverseSteganalysisAcquisitionError(f"review packet {key} must be an object")
    return value


def _review_packet_archive_sha(packet: Mapping[str, Any]) -> str:
    custody = _packet_mapping(packet, "custody")
    value = _text(custody.get("archive_sha256"), "review_packet.custody.archive_sha256")
    _sha256_value(value, "review_packet.custody.archive_sha256")
    return value


def _review_packet_archive_bytes(packet: Mapping[str, Any]) -> int:
    custody = _packet_mapping(packet, "custody")
    return _int(
        custody.get("archive_bytes"),
        "review_packet.custody.archive_bytes",
        minimum=1,
    )


def _review_packet_n_samples(packet: Mapping[str, Any]) -> int:
    custody = _packet_mapping(packet, "custody")
    return _int(
        custody.get("n_samples"),
        "review_packet.custody.n_samples",
        minimum=1,
    )


def _runtime_custody_text(
    packet: Mapping[str, Any],
    key: str,
    *,
    required: bool,
) -> str | None:
    custody = _packet_mapping(packet, "runtime_custody")
    value = _optional_text(custody.get(key))
    if required and value is None:
        raise InverseSteganalysisAcquisitionError(f"review_packet.runtime_custody.{key} is required")
    if value is not None and key.endswith("sha256"):
        _sha256_value(value, f"review_packet.runtime_custody.{key}")
    return value


def _paired_exact_auth_custody(
    by_axis: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cpu = by_axis["contest_cpu"]
    return {
        "archive_sha256": _review_packet_archive_sha(cpu),
        "archive_bytes": _review_packet_archive_bytes(cpu),
        "n_samples": _review_packet_n_samples(cpu),
    }


def _paired_exact_auth_runtime_identity(
    by_axis: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    common_content = _runtime_custody_text(
        by_axis["contest_cpu"],
        "runtime_content_tree_sha256",
        required=True,
    )
    by_axis_identity: dict[str, dict[str, Any]] = {}
    for axis, packet in sorted(by_axis.items()):
        fields: dict[str, Any] = {}
        for key in (
            "runtime_tree_sha256",
            "runtime_content_tree_sha256",
            "inflate_script_sha256",
            "inflated_output_manifest_sha256",
        ):
            value = _runtime_custody_text(packet, key, required=False)
            if value:
                fields[key] = value
        by_axis_identity[axis] = fields
    return {
        "runtime_content_tree_sha256": common_content,
        "by_axis": by_axis_identity,
    }


def _paired_exact_auth_cache_identity(
    by_axis: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    aggregates = {
        axis: _runtime_custody_text(
            packet,
            "inflated_output_aggregate_sha256",
            required=True,
        )
        for axis, packet in sorted(by_axis.items())
    }
    return {"inflated_outputs_aggregate_sha256": aggregates}


def _exact_auth_axis_calibration_row(
    packet: Mapping[str, Any],
    *,
    axis: str,
    packet_path: str | None,
) -> dict[str, Any]:
    score = _float(packet.get("canonical_score"), f"{axis}.canonical_score", minimum=0.0)
    baseline = _float(packet.get("baseline_score"), f"{axis}.baseline_score", minimum=0.0)
    delta = score - baseline
    recompute = _packet_mapping(packet, "score_recomputation")
    row = {
        "axis": axis,
        "score": score,
        "axis_baseline_score": baseline,
        "delta_vs_axis_baseline": delta,
        "score_delta_status": (
            "regresses_vs_axis_baseline"
            if delta > 0.0
            else "improves_vs_axis_baseline"
            if delta < 0.0
            else "ties_axis_baseline"
        ),
        "measured_config_status": packet.get("measured_config_status"),
        "failure_class": packet.get("failure_class"),
        "source_json_path": packet.get("source_json_path"),
        "review_packet_path": packet_path,
        "review_packet_score_valid_flag": bool(packet.get("score_claim_valid")),
        "component_terms": {
            "avg_segnet_dist": recompute.get("avg_segnet_dist"),
            "avg_posenet_dist": recompute.get("avg_posenet_dist"),
            "rate_term": recompute.get("rate_term"),
            "recomputed_score": recompute.get("recomputed_score"),
        },
        "runtime_content_tree_sha256": _runtime_custody_text(
            packet,
            "runtime_content_tree_sha256",
            required=True,
        ),
        "inflated_output_aggregate_sha256": _runtime_custody_text(
            packet,
            "inflated_output_aggregate_sha256",
            required=True,
        ),
        "dispatch_claim_latest_status": _packet_mapping(
            packet,
            "dispatch_claim_state",
        ).get("latest_status"),
    }
    return _false_authority(
        row,
        "exact_auth_axis_row_is_nested_calibration_metadata_only",
    )


def _looks_like_contest_auth_axis(axis: str) -> bool:
    return axis in CONTEST_AUTH_SCORE_AXES or any(
        axis.startswith(prefix) for prefix in CONTEST_AUTH_SCORE_AXIS_PREFIXES
    )


def _identity(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise InverseSteganalysisAcquisitionError(f"{label} must be a non-empty object")
    out = dict(value)
    if not any(_has_value(out.get(key)) for key in keys):
        raise InverseSteganalysisAcquisitionError(f"{label} must include one of: {', '.join(sorted(keys))}")
    for key, item in out.items():
        if key.endswith("sha256") and item is not None:
            _sha256_value(item, f"{label}.{key}")
    return out


def _sha256_value(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _sha256_value(item, f"{label}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _sha256_value(item, f"{label}[{index}]")
        return
    if not is_sha256_hex(value):
        raise InverseSteganalysisAcquisitionError(f"{label} must be sha256 hex")


def _sha256_or_none(value: Any, label: str) -> str | None:
    if value is None:
        return None
    text = _optional_text(value)
    if text is None:
        return None
    _sha256_value(text, label)
    return text


def _has_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_value(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_has_value(item) for item in value)
    return value not in (None, "")


def _scale(value: Any) -> str:
    scale = _token(_text(value, "scale"))
    if scale not in ALLOWED_SCALES:
        raise InverseSteganalysisAcquisitionError(f"scale must be one of {sorted(ALLOWED_SCALES)}, got {scale!r}")
    return scale


def _scope_axis(value: Any, scale: str) -> str:
    scope = _token(value) if value is not None else SCALE_TO_SCOPE_AXIS[scale]
    if scope not in SCOPE_AXES:
        raise InverseSteganalysisAcquisitionError(f"scope_axis must be one of {sorted(SCOPE_AXES)}, got {scope!r}")
    return scope


def _resource(value: Any) -> str:
    return _token(_text(value, "resource_kind")) or "local_cpu"


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().strip("[]").lower()).strip("_")


def _slug(value: Any) -> str:
    return _token(value)[:64] or "unknown"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_mapping(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise InverseSteganalysisAcquisitionError(f"{label} must be an object")
    return dict(value)


def _text(value: Any, label: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise InverseSteganalysisAcquisitionError(f"{label} must be a non-empty string")
    return text


def _range(value: Any, label: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 2:
        raise InverseSteganalysisAcquisitionError(f"{label} must be [start, end]")
    start = _int(value[0], f"{label}[0]", minimum=0)
    end = _int(value[1], f"{label}[1]", minimum=0)
    if end < start:
        raise InverseSteganalysisAcquisitionError(f"{label} end must be >= start")
    return [start, end]


def _int_list(value: Any, label: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise InverseSteganalysisAcquisitionError(f"{label} must be a list")
    out = [_int(item, f"{label}[{index}]", minimum=0) for index, item in enumerate(value)]
    if len(set(out)) != len(out):
        raise InverseSteganalysisAcquisitionError(f"{label} contains duplicates")
    return out


def _bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        raise InverseSteganalysisAcquisitionError("region_bbox must be [x0, y0, x1, y1]")
    bbox = [_float(coord, f"region_bbox[{index}]") for index, coord in enumerate(value)]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise InverseSteganalysisAcquisitionError("region_bbox max coordinates must exceed min")
    return bbox


def _int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise InverseSteganalysisAcquisitionError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise InverseSteganalysisAcquisitionError(f"{label} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise InverseSteganalysisAcquisitionError(f"{label} must be >= {minimum}")
    return result


def _optional_int(value: Any, label: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    return _int(value, label, minimum=minimum)


def _float_or_none(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    exclusive: bool = False,
) -> float | None:
    if value is None:
        return None
    return _float(value, label, minimum=minimum, exclusive=exclusive)


def _float(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    exclusive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise InverseSteganalysisAcquisitionError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InverseSteganalysisAcquisitionError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise InverseSteganalysisAcquisitionError(f"{label} must be finite")
    if minimum is not None and (result <= minimum if exclusive else result < minimum):
        op = ">" if exclusive else ">="
        raise InverseSteganalysisAcquisitionError(f"{label} must be {op} {minimum}")
    return result


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


__all__ = [
    "ACTION_CELL_SCHEMA",
    "ACTION_FUNCTIONAL_SCHEMA",
    "ALLOWED_SCALES",
    "ATOM_SCHEMA",
    "CONTEST_RATE_DENOM_BYTES",
    "CONTEST_RATE_SCORE_PER_BYTE",
    "INVERSE_SCORER_SURFACE_SCHEMA",
    "MATERIALIZER_ARCHIVE_DELTA_FEEDBACK_SCHEMA",
    "MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND",
    "MLX_ACQUISITION_BATCH_PROVENANCE_SCHEMA",
    "MLX_EFFECTIVE_SPEND_TRIAGE_PROVENANCE_SCHEMA",
    "MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_ROW_SCHEMA",
    "MLX_EFFECTIVE_SPEND_TRIAGE_SELECTION_SCHEMA",
    "OBSERVATION_SCHEMA",
    "PRIORITY_SCHEMA",
    "QUEUE_HEALTH_FEEDBACK_SCHEMA",
    "QUEUE_OBSERVATION_SCHEMA",
    "QUEUE_PERFORMANCE_SUMMARY_SCHEMA",
    "RESOURCE_MULTIPLIERS",
    "SCHEMA",
    "SCOPE_AXES",
    "TOOL",
    "AcquisitionPriorityTerms",
    "InverseSteganalysisAcquisitionError",
    "action_atoms_from_byte_shaving_campaign_plan",
    "action_atoms_from_byte_shaving_signal_surface",
    "action_atoms_from_inverse_scorer_surface",
    "action_atoms_from_mlx_acquisition_batch",
    "action_surface_terms",
    "build_discrete_scorer_action_functional",
    "build_inverse_steganalysis_acquisition_plan",
    "compute_acquisition_priority",
    "inverse_steganalysis_atoms_from_mlx_effective_spend_triage_selection",
    "normalize_inverse_steganalysis_atom",
    "normalize_inverse_steganalysis_observation",
    "observations_from_materializer_chain_manifest",
    "observations_from_queue_observation",
    "observations_from_queue_performance_summary",
    "paired_exact_auth_calibration_observations_from_review_packets",
]
