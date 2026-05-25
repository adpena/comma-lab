# SPDX-License-Identifier: MIT
"""Harvest distortion-axis learned-sweep plans into feedback observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.mlx_dynamic_learned_sweep import FALSE_AUTHORITY
from tac.optimization.mlx_dynamic_sweep_observations import (
    ROW_SCHEMA as OBSERVATION_ROW_SCHEMA,
)
from tac.optimization.mlx_dynamic_sweep_observations import (
    append_observation_row,
    build_observation_row,
)
from tac.optimization.proxy_candidate_contract import (
    require_no_truthy_authority_fields,
)

SCHEMA = "distortion_axis_probe_learned_sweep_feedback.v1"
TOOL = "tac.optimization.distortion_axis_probe_learned_sweep_feedback"
CANDIDATE_PAYLOAD_SCHEMA = "distortion_axis_probe_learned_sweep_candidates.v1"
PLAN_SCHEMA = "mlx_dynamic_learned_sweep_plan.v1"
DEFAULT_SWEEP_CONFIG_ID = "macos_cpu_advisory"
DEFAULT_OPTIMIZATION_PASS_ID = "smoke"
MACOS_CPU_ADVISORY_AXIS = "macos_cpu_advisory"
MACOS_CPU_ADVISORY_TAG = "[macOS-CPU advisory]"


class DistortionAxisProbeLearnedSweepFeedbackError(ValueError):
    """Raised when probe feedback cannot be harvested fail-closed."""


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DistortionAxisProbeLearnedSweepFeedbackError(
            f"{path}: expected JSON object"
        )
    return payload


def dumps_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_distortion_axis_probe_feedback_observation(
    *,
    plan: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    sweep_config_id: str = DEFAULT_SWEEP_CONFIG_ID,
    optimization_pass_id: str = DEFAULT_OPTIMIZATION_PASS_ID,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Build one feedback observation from a distortion learned-sweep row.

    The observation is intentionally advisory. It records that the local probe
    signal has already informed this candidate/config/pass tuple, letting the
    learned-sweep replanner suppress duplicate local work without treating the
    probe as an exact score or archive authority.
    """

    _require_schema(plan, PLAN_SCHEMA, label="plan")
    _require_schema(candidate_payload, CANDIDATE_PAYLOAD_SCHEMA, label="candidate_payload")
    _require_false_authority(plan, label="plan")
    _require_false_authority(candidate_payload, label="candidate_payload")
    candidate = _select_candidate(candidate_payload, candidate_id=candidate_id)
    selected_candidate_id = str(candidate["candidate_id"])
    row = _select_plan_row(
        plan,
        candidate_id=selected_candidate_id,
        sweep_config_id=sweep_config_id,
        optimization_pass_id=optimization_pass_id,
    )
    source_artifact = _source_artifact_for_candidate(candidate_payload, candidate)
    source_sha = str(source_artifact["sha256"])
    source_path = str(source_artifact["path"])
    predicted_delta = _required_float(
        candidate,
        "predicted_delta_vs_incumbent_score",
    )
    repair_budget = _required_float(
        candidate,
        "non_authoritative_normalized_full_video_gain_sum",
    )
    identity_payload = {
        "schema": "distortion_axis_probe_feedback_identity.v1",
        "candidate_id": selected_candidate_id,
        "queue_candidate_id": row.get("queue_candidate_id"),
        "sweep_config_id": sweep_config_id,
        "optimization_pass_id": optimization_pass_id,
        "source_artifact_sha256": source_sha,
        **FALSE_AUTHORITY,
    }
    archive_identity_sha = _hash_payload(
        {
            **identity_payload,
            "identity_kind": "probe_verdict_not_submission_archive",
        }
    )
    runtime_identity_sha = _hash_payload(
        {
            **identity_payload,
            "identity_kind": "feedback_harvester_runtime",
            "producer": TOOL,
        }
    )
    observation = build_observation_row(
        candidate_id=selected_candidate_id,
        sweep_config_id=sweep_config_id,
        optimization_pass_id=optimization_pass_id,
        family=str(row["family"]),
        observed_axis=MACOS_CPU_ADVISORY_AXIS,
        evidence_tag=MACOS_CPU_ADVISORY_TAG,
        observed_score_or_delta=predicted_delta,
        archive_sha256=archive_identity_sha,
        runtime_sha256=runtime_identity_sha,
        raw_output_or_cache_sha256=source_sha,
        component_deltas={
            "segnet_delta": predicted_delta,
            "posenet_delta": 0.0,
            "rate_delta": 0.0,
            "scorer_delta": predicted_delta,
            "non_authoritative_repair_budget_score": repair_budget,
        },
        source_artifact_path=source_path,
        source_artifact_sha256=source_sha,
        extra={
            "evidence_grade": "macOS-CPU-advisory",
            "source_schema": SCHEMA,
            "planner_artifact_path": _artifact_path_from_source(plan),
            "selected_pair_indices": row.get("selected_pair_indices"),
            "selected_pair_count": row.get("selected_pair_count"),
            "sweep_rank": row.get("rank"),
            "source_row": {
                "candidate_id": selected_candidate_id,
                "queue_candidate_id": row.get("queue_candidate_id"),
                "sweep_config_id": sweep_config_id,
                "optimization_pass_id": optimization_pass_id,
                "source_probe_id": candidate.get("source_probe_id"),
                **FALSE_AUTHORITY,
            },
            "component_delta_baseline_policy": (
                "probe9_conservative_predicted_delta_vs_incumbent_no_auth_baseline"
            ),
            "score_delta_vs_baseline": predicted_delta,
            "archive_byte_delta_vs_baseline": 0.0,
            "archive_identity_semantics": (
                "hash_identity_for_probe_feedback_not_submission_archive"
            ),
            "feedback_semantics": (
                "local_distortion_probe_feedback_only_not_scorer_execution"
            ),
            "non_authoritative_repair_budget_score": repair_budget,
            "non_authoritative_repair_budget_bytes_equivalent": _repair_budget_bytes(
                candidate
            ),
            "notes": (
                "Harvested from the distortion-axis probe learned-sweep payload; "
                "this row suppresses duplicate local planning work only and "
                "carries no score, rank, promotion, dispatch, or exact-eval authority."
            ),
        },
    )
    return observation


def append_distortion_axis_probe_feedback_observation(
    *,
    plan: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    output_path: Path,
    sweep_config_id: str = DEFAULT_SWEEP_CONFIG_ID,
    optimization_pass_id: str = DEFAULT_OPTIMIZATION_PASS_ID,
    candidate_id: str | None = None,
    allow_duplicate_observation: bool = False,
) -> dict[str, Any]:
    observation = build_distortion_axis_probe_feedback_observation(
        plan=plan,
        candidate_payload=candidate_payload,
        sweep_config_id=sweep_config_id,
        optimization_pass_id=optimization_pass_id,
        candidate_id=candidate_id,
    )
    return append_observation_row(
        observation,
        output_path=output_path,
        allow_duplicate_observation=allow_duplicate_observation,
    )


def build_feedback_summary(
    *,
    observation: Mapping[str, Any],
    observation_jsonl: Path,
    replan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_schema(observation, OBSERVATION_ROW_SCHEMA, label="observation")
    _require_false_authority(observation, label="observation")
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "observation_jsonl": str(observation_jsonl),
        "observation_jsonl_sha256": (
            file_sha256(observation_jsonl) if observation_jsonl.is_file() else None
        ),
        "observation_row_schema": OBSERVATION_ROW_SCHEMA,
        "candidate_id": observation.get("candidate_id"),
        "sweep_config_id": observation.get("sweep_config_id"),
        "optimization_pass_id": observation.get("optimization_pass_id"),
        "observed_axis": observation.get("observed_axis"),
        "observed_score_or_delta": observation.get("observed_score_or_delta"),
        "source_artifact_sha256": observation.get("source_artifact_sha256"),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if replan is not None:
        _require_schema(replan, PLAN_SCHEMA, label="replan")
        _require_false_authority(replan, label="replan")
        summary = replan.get("summary")
        summary_map = summary if isinstance(summary, Mapping) else {}
        out["replan"] = {
            "schema": PLAN_SCHEMA,
            "ranked_row_count": summary_map.get("ranked_row_count"),
            "suppressed_observed_row_count": summary_map.get(
                "suppressed_observed_row_count"
            ),
            "local_ready_row_count": summary_map.get("local_ready_row_count"),
            **FALSE_AUTHORITY,
        }
    return out


def _select_candidate(
    candidate_payload: Mapping[str, Any],
    *,
    candidate_id: str | None,
) -> Mapping[str, Any]:
    rows = candidate_payload.get("candidates")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise DistortionAxisProbeLearnedSweepFeedbackError(
            "candidate_payload candidates[] missing"
        )
    matches: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DistortionAxisProbeLearnedSweepFeedbackError(
                f"candidate_payload candidates[{index}] must be an object"
            )
        _require_false_authority(row, label=f"candidate {index}")
        if candidate_id is None or str(row.get("candidate_id") or "") == candidate_id:
            matches.append(row)
    if not matches:
        raise DistortionAxisProbeLearnedSweepFeedbackError(
            f"candidate_id not found: {candidate_id}"
        )
    if len(matches) > 1 and candidate_id is None:
        raise DistortionAxisProbeLearnedSweepFeedbackError(
            "candidate_id is required when payload contains multiple candidates"
        )
    return matches[0]


def _select_plan_row(
    plan: Mapping[str, Any],
    *,
    candidate_id: str,
    sweep_config_id: str,
    optimization_pass_id: str,
) -> Mapping[str, Any]:
    rows = plan.get("ranked_sweep_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise DistortionAxisProbeLearnedSweepFeedbackError(
            "plan ranked_sweep_rows[] missing"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        _require_false_authority(row, label=f"plan row {index}")
        if (
            str(row.get("candidate_id") or "") == candidate_id
            and str(row.get("sweep_config_id") or "") == sweep_config_id
            and str(row.get("optimization_pass_id") or "") == optimization_pass_id
        ):
            if row.get("ready_for_local_sweep") is not True:
                raise DistortionAxisProbeLearnedSweepFeedbackError(
                    "selected plan row is not local-ready"
                )
            return row
    raise DistortionAxisProbeLearnedSweepFeedbackError(
        "no matching local-ready plan row for "
        f"{candidate_id}::{sweep_config_id}::{optimization_pass_id}"
    )


def _source_artifact_for_candidate(
    candidate_payload: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    source_artifacts = candidate_payload.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise DistortionAxisProbeLearnedSweepFeedbackError(
            "candidate_payload source_artifacts missing"
        )
    index = int(_required_float(candidate, "source_probe_index"))
    key = f"verdict_{index:03d}"
    artifact = source_artifacts.get(key)
    if not isinstance(artifact, Mapping):
        raise DistortionAxisProbeLearnedSweepFeedbackError(
            f"candidate source artifact missing: {key}"
        )
    path = artifact.get("path")
    sha = artifact.get("sha256")
    if not path or not _is_sha256(sha):
        raise DistortionAxisProbeLearnedSweepFeedbackError(
            f"candidate source artifact {key} must include path and sha256"
        )
    return artifact


def _artifact_path_from_source(plan: Mapping[str, Any]) -> str | None:
    artifacts = plan.get("source_artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    payload = artifacts.get("distortion_axis_candidate_payload")
    if isinstance(payload, Mapping) and payload.get("path") is not None:
        return str(payload.get("path"))
    return None


def _repair_budget_bytes(candidate: Mapping[str, Any]) -> float | None:
    context = candidate.get("component_axis_context")
    if not isinstance(context, Mapping):
        return None
    value = context.get("non_authoritative_rate_budget_bytes_equivalent")
    if value is None:
        return None
    return _required_float(context, "non_authoritative_rate_budget_bytes_equivalent")


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _require_schema(payload: Mapping[str, Any], schema: str, *, label: str) -> None:
    if payload.get("schema") != schema:
        raise DistortionAxisProbeLearnedSweepFeedbackError(
            f"{label} schema must be {schema}"
        )


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise DistortionAxisProbeLearnedSweepFeedbackError(
                f"{label} {key} must be explicit false"
            )
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise DistortionAxisProbeLearnedSweepFeedbackError(str(exc)) from exc


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        raise DistortionAxisProbeLearnedSweepFeedbackError(f"{key} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise DistortionAxisProbeLearnedSweepFeedbackError(
            f"{key} must be numeric"
        ) from exc
    return out


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip()
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


__all__ = [
    "DEFAULT_OPTIMIZATION_PASS_ID",
    "DEFAULT_SWEEP_CONFIG_ID",
    "SCHEMA",
    "TOOL",
    "DistortionAxisProbeLearnedSweepFeedbackError",
    "append_distortion_axis_probe_feedback_observation",
    "build_distortion_axis_probe_feedback_observation",
    "build_feedback_summary",
    "dumps_json",
    "file_sha256",
    "load_json_object",
]
