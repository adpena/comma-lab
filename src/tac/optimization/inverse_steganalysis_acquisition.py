# SPDX-License-Identifier: MIT
"""Planning-only inverse-steganalysis acquisition surface.

The rows here are scorer-in-the-loop search signal, not score authority. They
encode multiscale atoms plus local/proxy calibration observations so schedulers
can rank next probes by expected score gain per second, GB, and resource kind.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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

SCHEMA = "inverse_steganalysis_acquisition_plan.v1"
ATOM_SCHEMA = "inverse_steganalysis_atom.v1"
OBSERVATION_SCHEMA = "inverse_steganalysis_observation.v1"
EXACT_AUTH_CALIBRATION_SCHEMA = "inverse_steganalysis_exact_auth_calibration.v1"
QUEUE_PERFORMANCE_SUMMARY_SCHEMA = "experiment_queue_performance_summary.v1"
PRIORITY_SCHEMA = "inverse_steganalysis_acquisition_priority.v1"
ACTION_FUNCTIONAL_SCHEMA = "inverse_steganalysis_discrete_action_functional.v1"
ACTION_CELL_SCHEMA = "inverse_steganalysis_action_cell.v1"
INVERSE_SCORER_SURFACE_SCHEMA = "scorer_inverse_decision_surface.v1"
TOOL = "tac.optimization.inverse_steganalysis_acquisition"
CONTEST_RATE_SCORE_PER_BYTE = 25.0 / 50_000_000.0

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
        "artifact_bytes": _int(row.get("artifact_bytes", row.get("source_artifact_bytes", 0)), "artifact_bytes", minimum=0),
        "resource_kind": _resource(row.get("resource_kind", "local_cpu")),
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
    if _looks_like_contest_auth_axis(axis_normalized) or resource_kind == "contest_exact_eval":
        raise InverseSteganalysisAcquisitionError(
            "inverse-steganalysis observations must not masquerade as contest "
            "auth evidence; use auth-eval payload validation for contest axes"
        )
    out = {
        "schema": OBSERVATION_SCHEMA,
        "observation_id": _optional_text(row.get("observation_id")) or f"obs_{_slug(candidate_id)}_{_slug(axis)}",
        "observation_kind": _optional_text(row.get("observation_kind")),
        "candidate_id": candidate_id,
        "axis": axis,
        "axis_normalized": axis_normalized,
        "source_path": _optional_text(row.get("source_path")),
        "queue_id": _optional_text(row.get("queue_id")),
        "experiment_id": _optional_text(row.get("experiment_id")),
        "step_id": _optional_text(row.get("step_id")),
        "performance_bucket_key": _optional_text(row.get("performance_bucket_key")),
        "performance_summary_schema": _optional_text(row.get("performance_summary_schema")),
        "run_count": _optional_int(row.get("run_count"), "run_count", minimum=0),
        "success_count": _optional_int(row.get("success_count"), "success_count", minimum=0),
        "failure_count": _optional_int(row.get("failure_count"), "failure_count", minimum=0),
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
        "artifact_bytes": _int(_first(row.get("artifact_bytes"), row.get("source_artifact_bytes"), 0), "artifact_bytes", minimum=0),
        "resource_kind": resource_kind,
        "candidate_generation_only": True,
        "planning_only": True,
        "allowed_use": "local_or_proxy_acquisition_ranking_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        "tool": TOOL,
    }
    if row.get("exact_auth_calibration") is not None:
        if not isinstance(row.get("exact_auth_calibration"), Mapping):
            raise InverseSteganalysisAcquisitionError(
                "exact_auth_calibration must be an object"
            )
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
    candidate_id_by_experiment: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert queue timing/artifact telemetry into planning-only observations.

    Queue performance rows do not carry scorer deltas. They only calibrate the
    acquisition denominator: seconds, artifact footprint, and resource class.
    Runtime/cache identity remains required so the observation cannot silently
    cross into score or promotion authority.
    """

    _reject_truthy_authority(summary, label="queue performance summary")
    if summary.get("schema") != QUEUE_PERFORMANCE_SUMMARY_SCHEMA:
        raise InverseSteganalysisAcquisitionError(
            f"summary schema must be {QUEUE_PERFORMANCE_SUMMARY_SCHEMA}"
        )
    queue_id = _text(summary.get("queue_id"), "summary.queue_id")
    by_step = summary.get("by_step")
    if not isinstance(by_step, Mapping):
        raise InverseSteganalysisAcquisitionError("summary.by_step must be an object")
    def parse_candidate_lookup(
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
            elif isinstance(raw_value, Sequence) and not isinstance(raw_value, bytes):
                values = tuple(
                    _text(value, f"{label}[{key!r}][{index}]")
                    for index, value in enumerate(raw_value)
                )
                if not values:
                    raise InverseSteganalysisAcquisitionError(
                        f"{label}[{key!r}] must not be empty"
                    )
            else:
                raise InverseSteganalysisAcquisitionError(
                    f"{label}[{key!r}] must be a string or list"
                )
            lookup[key] = values
        return lookup

    embedded_lookup = parse_candidate_lookup(
        summary.get("candidate_id_by_experiment")
        if summary.get("candidate_id_by_experiment") is not None
        else None,
        label="summary.candidate_id_by_experiment",
    )
    explicit_lookup = parse_candidate_lookup(
        candidate_id_by_experiment,
        label="candidate_id_by_experiment",
    )
    candidate_lookup = dict(embedded_lookup)
    for key, values in explicit_lookup.items():
        embedded_values = candidate_lookup.get(key)
        if embedded_values is not None and embedded_values != values:
            raise InverseSteganalysisAcquisitionError(
                "candidate_id_by_experiment conflicts with "
                f"summary.candidate_id_by_experiment[{key!r}]"
            )
        candidate_lookup[key] = values
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
                "summary.candidate_id_by_experiment missing experiment "
                f"{experiment_id!r}"
            )
        base_observation_id = (
            f"queue_perf_{_slug(queue_id)}_"
            f"{_slug(experiment_id)}_{_slug(step_id)}"
        )
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
                        "runtime_identity": runtime_identity,
                        "cache_identity": cache_identity,
                        "elapsed_seconds": elapsed_seconds,
                        "artifact_bytes": artifact_bytes,
                        "resource_kind": resource_kind,
                        "run_count": run_count,
                        "success_count": bucket.get("success_count"),
                        "failure_count": bucket.get("failure_count"),
                        "artifact_record_count": bucket.get("artifact_record_count"),
                        "artifact_record_raw_bytes_mean": bucket.get(
                            "artifact_record_raw_bytes_mean"
                        ),
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
        raise InverseSteganalysisAcquisitionError(
            "paired exact-auth calibration requires exactly two review packets"
        )
    paths = [str(path) for path in (packet_paths or [])]
    if packet_paths is not None and len(paths) != len(packet_rows):
        raise InverseSteganalysisAcquisitionError(
            "packet_paths length must match packets length"
        )

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
    regression_penalty = sum(
        max(0.0, float(row["delta_vs_axis_baseline"]))
        for row in axis_rows.values()
    )
    improvement_gain = sum(
        max(0.0, -float(row["delta_vs_axis_baseline"]))
        for row in axis_rows.values()
    )
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
            "observed_score_gain_policy": (
                "zero_when_any_exact_auth_axis_regresses_vs_baseline"
            ),
            "runtime_identity": runtime_identity,
            "cache_identity": cache_identity,
            "packet_paths": [
                path_by_axis[axis]
                for axis in ("contest_cpu", "contest_cuda")
                if path_by_axis.get(axis)
            ],
            "allowed_use": (
                "planner_calibration_only_measured_config_not_family_retirement"
            ),
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

    atom_row = (
        dict(atom)
        if atom.get("schema") == ATOM_SCHEMA
        else normalize_inverse_steganalysis_atom(atom)
    )
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
    expected_gain = max(0.0, base_gain - calibration_penalty - fragility_penalty) + uncertainty_bonus
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
        candidate_obs = by_candidate.get(str(atom["candidate_id"]), [])
        best_obs = _best_observation(atom, candidate_obs)
        ranked.append(
            _false_authority(
                {
                    **atom,
                    "source_atom_index": index,
                    "best_observation_id": None if best_obs is None else best_obs["observation_id"],
                    "observation_count": len(candidate_obs),
                    "priority": compute_acquisition_priority(atom, best_obs),
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
    for index, raw_atom in enumerate(atoms):
        atom = normalize_inverse_steganalysis_atom(raw_atom)
        best_obs = _best_observation(atom, by_candidate.get(str(atom["candidate_id"]), []))
        priority = compute_acquisition_priority(atom, best_obs)
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
        blocked = bool(guard.get("blocked"))
        if blocked:
            blocked_cells += 1
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
                    "best_observation_id": None if best_obs is None else best_obs["observation_id"],
                    "best_observation_kind": (
                        None if best_obs is None else best_obs.get("observation_kind")
                    ),
                    "exact_auth_calibration": (
                        best_obs.get("exact_auth_calibration")
                        if isinstance(best_obs, Mapping)
                        and isinstance(best_obs.get("exact_auth_calibration"), Mapping)
                        else None
                    ),
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
            "integral_totals": {
                "cell_count": len(cells),
                "blocked_cell_count": blocked_cells,
                "first_order_marginal_effect_sum": total_first_order,
                "second_order_interaction_effect_sum": total_second_order,
                "synergy_effect_sum": total_synergy,
                "antagonism_effect_sum": total_antagonism,
                "fragility_penalty_sum": total_fragility_penalty,
                "expected_score_gain_sum": total_expected_gain,
                "net_action_gain_after_fragility": max(
                    0.0,
                    total_first_order
                    + total_second_order
                    - total_fragility_penalty,
                ),
            },
            "water_bucket": water_bucket,
            "cells": cells,
        },
        "inverse_steganalysis_discrete_action_functional_is_not_score_authority",
        "requires_byte_closed_candidate_generation_before_dispatch",
        "requires_exact_auth_eval_before_promotion_or_rank_kill",
    )


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
        raise InverseSteganalysisAcquisitionError(
            f"surface schema must be {INVERSE_SCORER_SURFACE_SCHEMA}"
        )
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
        str(source_candidates[0])
        if isinstance(source_candidates, list) and source_candidates
        else None
    )
    candidate_id = (
        default_candidate_id
        or source_candidate_id
        or f"inverse_surface_candidate_{_slug(cell_id)}"
    )
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
        "artifact_bytes": (
            artifact_bytes if artifact_bytes is not None else max(1, saved_bytes)
        ),
        "resource_kind": resource_kind,
    }


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
                        "regression_penalty_sum": calibration.get(
                            "regression_penalty_sum"
                        ),
                        "improvement_gain_sum": calibration.get(
                            "improvement_gain_sum"
                        ),
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
            "allowed_use": "action_functional_calibration_interpretability_only",
        },
        "observation_feedback_is_not_score_authority",
    )


def _false_authority(row: Mapping[str, Any], *blockers: str) -> dict[str, Any]:
    return apply_proxy_evidence_boundary(
        dict(row),
        dispatch_blockers=ordered_unique(
            ("inverse_steganalysis_acquisition_false_authority_only", *blockers)
        ),
    )


def _reject_truthy_authority(row: Mapping[str, Any], *, label: str) -> None:
    violations = truthy_authority_field_violations(row, fields=AUTHORITY_FIELDS)
    if violations:
        raise InverseSteganalysisAcquisitionError(
            f"{label}: forbidden truthy authority fields: {', '.join(violations)}"
        )


def _predicted_effects(row: Mapping[str, Any]) -> dict[str, float]:
    seg = _float(row.get("predicted_segnet_gain", row.get("predicted_segnet_score_gain", 0.0)), "predicted_segnet_gain", minimum=0.0)
    pose = _float(row.get("predicted_posenet_gain", row.get("predicted_posenet_score_gain", 0.0)), "predicted_posenet_gain", minimum=0.0)
    rate_gain = _float(row.get("predicted_rate_gain", row.get("predicted_rate_score_gain", 0.0)), "predicted_rate_gain", minimum=0.0)
    rate_cost = _float(row.get("predicted_rate_cost", row.get("predicted_rate_score_cost", 0.0)), "predicted_rate_cost", minimum=0.0)
    explicit = _float_or_none(row.get("predicted_score_gain"), "predicted_score_gain", minimum=0.0)
    delta = _float_or_none(row.get("predicted_delta_vs_baseline_score"), "predicted_delta_vs_baseline_score")
    return {
        "predicted_segnet_gain": seg,
        "predicted_posenet_gain": pose,
        "predicted_rate_gain": rate_gain,
        "predicted_rate_cost": rate_cost,
        "predicted_score_gain": explicit if explicit is not None else (max(0.0, -delta) if delta is not None else max(0.0, seg + pose + rate_gain - rate_cost)),
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
    selected: list[dict[str, Any]] = []
    used_bytes = 0
    expected_gain = 0.0
    for row in ordered:
        cost = int(row["measure"]["water_fill_cost_bytes"])
        if total_byte_budget is not None and used_bytes + cost > total_byte_budget:
            continue
        selected.append(
            {
                "atom_id": row["atom_id"],
                "candidate_id": row["candidate_id"],
                "scope_axis": row["scope_axis"],
                "component": row["component"],
                "water_fill_cost_bytes": cost,
                "expected_score_gain": row["expected_score_gain"],
                "euler_lagrange_residual": row["euler_lagrange_residual"],
            }
        )
        used_bytes += cost
        expected_gain += float(row["expected_score_gain"])
    return {
        "schema": "inverse_steganalysis_water_bucket_plan.v1",
        "total_byte_budget": total_byte_budget,
        "selected_count": len(selected),
        "selected_water_fill_cost_bytes": used_bytes,
        "selected_expected_score_gain": expected_gain,
        "selected_cells": selected,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


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

    row = (
        dict(atom)
        if atom.get("schema") == ATOM_SCHEMA
        else normalize_inverse_steganalysis_atom(atom)
    )
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
        raise InverseSteganalysisAcquisitionError(
            "performance bucket key must be experiment_id.step_id"
        )
    experiment_id, step_id = text.split(".", 1)
    return _text(experiment_id, "performance experiment_id"), _text(
        step_id,
        "performance step_id",
    )


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
        raise InverseSteganalysisAcquisitionError(
            "review packet schema must be tac_result_review_packet_v1"
        )
    axis = _text(packet.get("score_axis"), "review_packet.score_axis")
    if axis not in {"contest_cpu", "contest_cuda"}:
        raise InverseSteganalysisAcquisitionError(
            f"unsupported review packet score_axis {axis!r}"
        )
    if axis == "contest_cpu" and packet.get("exact_cpu_evidence") is not True:
        raise InverseSteganalysisAcquisitionError(
            "contest_cpu review packet must have exact_cpu_evidence=true"
        )
    if axis == "contest_cuda" and packet.get("exact_cuda_evidence") is not True:
        raise InverseSteganalysisAcquisitionError(
            "contest_cuda review packet must have exact_cuda_evidence=true"
        )
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
            raise InverseSteganalysisAcquisitionError(
                "paired exact-auth packets must share archive_sha256"
            )
        if _review_packet_archive_bytes(packet) != archive_bytes:
            raise InverseSteganalysisAcquisitionError(
                "paired exact-auth packets must share archive_bytes"
            )
        if _review_packet_n_samples(packet) != n_samples:
            raise InverseSteganalysisAcquisitionError(
                "paired exact-auth packets must share n_samples"
            )
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
        raise InverseSteganalysisAcquisitionError(
            f"{axis} review packet score_claim must be false"
        )
    for key in ("promotion_eligible", "rank_or_kill_eligible", "ready_for_exact_eval_dispatch"):
        if packet.get(key) is not False:
            raise InverseSteganalysisAcquisitionError(
                f"{axis} review packet {key} must be false"
            )
    if packet.get("family_falsified") is True or packet.get("method_family_retired") is True:
        raise InverseSteganalysisAcquisitionError(
            f"{axis} review packet must not retire a family"
        )
    status = _text(
        packet.get("measured_config_status"),
        f"{axis}.measured_config_status",
    )
    if status.startswith("indeterminate"):
        raise InverseSteganalysisAcquisitionError(
            f"{axis} review packet status is indeterminate"
        )
    baseline = packet.get("baseline_score")
    if baseline is None:
        raise InverseSteganalysisAcquisitionError(
            f"{axis} review packet baseline_score is required"
        )
    _float(baseline, f"{axis}.baseline_score", minimum=0.0)
    _float(packet.get("canonical_score"), f"{axis}.canonical_score", minimum=0.0)
    recompute = _packet_mapping(packet, "score_recomputation")
    if recompute.get("available") is not True:
        raise InverseSteganalysisAcquisitionError(
            f"{axis} score recomputation must be available"
        )
    if recompute.get("matches_reported") is not True:
        raise InverseSteganalysisAcquisitionError(
            f"{axis} score recomputation must match reported score"
        )
    audit = _packet_mapping(packet, "engineering_forensic_audit")
    if audit.get("engineering_or_config_bug_found") is True:
        raise InverseSteganalysisAcquisitionError(
            f"{axis} engineering forensic audit has blockers"
        )
    if audit.get("score_formula_reviewed") is not True:
        raise InverseSteganalysisAcquisitionError(
            f"{axis} engineering forensic audit must review score formula"
        )
    if audit.get("archive_runtime_closure_reviewed") is not True:
        raise InverseSteganalysisAcquisitionError(
            f"{axis} engineering forensic audit must review runtime closure"
        )
    claim = _packet_mapping(packet, "dispatch_claim_state")
    if claim.get("terminal_status_recorded") is not True:
        raise InverseSteganalysisAcquisitionError(
            f"{axis} review packet must have a terminal dispatch claim"
        )
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
        raise InverseSteganalysisAcquisitionError(
            f"review_packet.runtime_custody.{key} is required"
        )
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
    "CONTEST_RATE_SCORE_PER_BYTE",
    "INVERSE_SCORER_SURFACE_SCHEMA",
    "OBSERVATION_SCHEMA",
    "PRIORITY_SCHEMA",
    "QUEUE_PERFORMANCE_SUMMARY_SCHEMA",
    "RESOURCE_MULTIPLIERS",
    "SCHEMA",
    "SCOPE_AXES",
    "TOOL",
    "AcquisitionPriorityTerms",
    "InverseSteganalysisAcquisitionError",
    "action_atoms_from_inverse_scorer_surface",
    "action_surface_terms",
    "build_discrete_scorer_action_functional",
    "build_inverse_steganalysis_acquisition_plan",
    "compute_acquisition_priority",
    "normalize_inverse_steganalysis_atom",
    "normalize_inverse_steganalysis_observation",
    "observations_from_queue_performance_summary",
    "paired_exact_auth_calibration_observations_from_review_packets",
]
