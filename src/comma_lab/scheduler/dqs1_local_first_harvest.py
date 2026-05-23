from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.local_cpu_contest_drift import (
    EUREKA_SIGNAL_SCHEMA,
    LocalCPUContestDriftError,
    local_cpu_advisory_payload_blockers,
    require_eureka_false_authority,
)
from tac.optimization.proxy_candidate_contract import apply_proxy_evidence_boundary

from .dqs1_local_first_queue import (
    DEFAULT_RESULTS_ROOT,
    build_queue_from_action_summary,
    find_latest_cross_family_action_summary,
)
from .experiment_queue import ExperimentQueueError, load_queue_definition

HARVEST_SCHEMA = "dqs1_local_first_harvest.v1"
EXACT_AUTH_ANCHOR_REQUEST_SCHEMA = "exact_auth_anchor_request.v1"
DEFAULT_QUEUE_PATH = "configs/experiment_queues/dqs1_pairset_local_first.yaml"


@dataclass(frozen=True)
class Dqs1HarvestResult:
    harvest_record: dict[str, Any]
    exact_auth_request: dict[str, Any] | None
    rerouted_queue: dict[str, Any] | None = None


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentQueueError(f"{path}: could not load JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentQueueError(f"{path}: JSON root must be an object")
    return payload


def _resolve(path: str | Path, *, repo_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _arg_after(command: list[Any], flag: str, *, step_id: str) -> str:
    try:
        idx = command.index(flag)
        value = command[idx + 1]
    except (ValueError, IndexError) as exc:
        raise ExperimentQueueError(f"{step_id}: command missing {flag}") from exc
    if not isinstance(value, str) or not value.strip():
        raise ExperimentQueueError(f"{step_id}: command {flag} value must be a non-empty string")
    return value


def _single_experiment(queue: dict[str, Any]) -> dict[str, Any]:
    experiments = queue.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 1:
        raise ExperimentQueueError("DQS1 local-first harvest expects exactly one experiment")
    experiment = experiments[0]
    if not isinstance(experiment, dict):
        raise ExperimentQueueError("DQS1 experiment must be an object")
    return experiment


def _step_by_id(experiment: dict[str, Any], step_id: str) -> dict[str, Any]:
    steps = experiment.get("steps")
    if not isinstance(steps, list):
        raise ExperimentQueueError("DQS1 experiment steps must be a list")
    for step in steps:
        if isinstance(step, dict) and step.get("id") == step_id:
            return step
    raise ExperimentQueueError(f"DQS1 experiment missing step {step_id!r}")


def _command(step: dict[str, Any], *, step_id: str) -> list[Any]:
    command = step.get("command")
    if not isinstance(command, list) or not command:
        raise ExperimentQueueError(f"{step_id}: command must be a non-empty list")
    return command


def _validate_advisory(path: Path) -> dict[str, Any]:
    advisory = _load_json_object(path)
    blockers = local_cpu_advisory_payload_blockers(advisory)
    if blockers:
        raise ExperimentQueueError(
            f"{path}: local CPU advisory outside harvest contract: {blockers}"
        )
    return advisory


def _validate_eureka(
    path: Path,
    *,
    candidate_id: str,
    advisory_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    eureka = _load_json_object(path)
    if eureka.get("schema") != EUREKA_SIGNAL_SCHEMA:
        raise ExperimentQueueError(f"{path}: expected schema {EUREKA_SIGNAL_SCHEMA}")
    if eureka.get("candidate_id") != candidate_id:
        raise ExperimentQueueError(
            f"{path}: candidate_id {eureka.get('candidate_id')!r} != {candidate_id!r}"
        )
    try:
        require_eureka_false_authority(eureka, context=f"{path} eureka signal")
    except LocalCPUContestDriftError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    source_artifact = eureka.get("source_artifact")
    if not isinstance(source_artifact, str) or not source_artifact.strip():
        raise ExperimentQueueError(f"{path}: eureka source_artifact missing")
    source_path = _resolve(source_artifact, repo_root=repo_root)
    if source_path.resolve(strict=False) != advisory_path.resolve(strict=False):
        raise ExperimentQueueError(
            f"{path}: eureka source_artifact does not match advisory path"
        )
    if eureka.get("recommended_action") not in {"observe_only", "dispatch_exact_auth_anchor"}:
        raise ExperimentQueueError(f"{path}: unsupported recommended_action")
    if eureka.get("eureka_trigger") not in {True, False}:
        raise ExperimentQueueError(f"{path}: eureka_trigger must be boolean")
    if (
        eureka.get("eureka_trigger") is True
        and eureka.get("recommended_action") != "dispatch_exact_auth_anchor"
    ):
        raise ExperimentQueueError(f"{path}: positive eureka must request exact auth anchor")
    if eureka.get("eureka_trigger") is False and eureka.get("recommended_action") != "observe_only":
        raise ExperimentQueueError(f"{path}: negative eureka must be observe_only")
    return eureka

def _archive_path_from_advisory(advisory: dict[str, Any]) -> str:
    provenance = advisory.get("provenance")
    if isinstance(provenance, dict):
        archive_path = provenance.get("archive_path")
        if isinstance(archive_path, str) and archive_path.strip():
            return archive_path
    archive_path = advisory.get("archive_path")
    if isinstance(archive_path, str) and archive_path.strip():
        return archive_path
    raise ExperimentQueueError("local advisory missing archive_path provenance")


def _archive_sha_from_advisory(advisory: dict[str, Any]) -> str:
    provenance = advisory.get("provenance")
    if isinstance(provenance, dict):
        archive_sha = provenance.get("archive_sha256")
        if isinstance(archive_sha, str) and archive_sha.strip():
            return archive_sha
    archive_sha = advisory.get("archive_sha256")
    if isinstance(archive_sha, str) and archive_sha.strip():
        return archive_sha
    raise ExperimentQueueError("local advisory missing archive sha256 provenance")


def build_dqs1_harvest_result(
    *,
    queue_path: str | Path = DEFAULT_QUEUE_PATH,
    repo_root: str | Path,
    timestamp: str | None = None,
    reroute_observe_only: bool = False,
    output_queue_path: str | Path | None = None,
    action_summary: str | Path = "latest",
    results_root: str = DEFAULT_RESULTS_ROOT,
) -> Dqs1HarvestResult:
    """Validate the active DQS1 result and optionally build the next queue."""

    repo = Path(repo_root)
    queue_file = _resolve(queue_path, repo_root=repo)
    queue = load_queue_definition(queue_file)
    experiment = _single_experiment(queue)
    candidate_id = str(experiment.get("id") or "")
    if not candidate_id:
        raise ExperimentQueueError("DQS1 experiment id is required")

    advisory_step = _step_by_id(experiment, "local_cpu_advisory")
    eureka_step = _step_by_id(experiment, "local_cpu_contest_drift_eureka")
    advisory_rel = _arg_after(
        _command(advisory_step, step_id="local_cpu_advisory"),
        "--json-out",
        step_id="local_cpu_advisory",
    )
    eureka_rel = _arg_after(
        _command(eureka_step, step_id="local_cpu_contest_drift_eureka"),
        "--eureka-out",
        step_id="local_cpu_contest_drift_eureka",
    )
    advisory_path = _resolve(advisory_rel, repo_root=repo)
    eureka_path = _resolve(eureka_rel, repo_root=repo)
    if not advisory_path.exists():
        raise ExperimentQueueError(f"{advisory_path}: local advisory not found")
    if not eureka_path.exists():
        raise ExperimentQueueError(f"{eureka_path}: eureka signal not found")

    advisory = _validate_advisory(advisory_path)
    eureka = _validate_eureka(
        eureka_path,
        candidate_id=candidate_id,
        advisory_path=advisory_path,
        repo_root=repo,
    )
    stamp = timestamp or _utc_stamp()
    recommended_action = str(eureka["recommended_action"])
    local_score = float(eureka["local_score"])
    harvest = apply_proxy_evidence_boundary(
        {
            "schema": HARVEST_SCHEMA,
            "candidate_id": candidate_id,
            "queue_path": str(queue_file),
            "local_cpu_advisory_path": str(advisory_path),
            "eureka_signal_path": str(eureka_path),
            "candidate_archive_path": _archive_path_from_advisory(advisory),
            "candidate_archive_sha256": _archive_sha_from_advisory(advisory),
            "local_score": local_score,
            "auth_frontier_score": eureka.get("auth_frontier_score"),
            "projected_contest_score": eureka.get("projected_contest_score"),
            "conservative_projected_contest_score": eureka.get(
                "conservative_projected_contest_score"
            ),
            "eureka_margin": eureka.get("eureka_margin"),
            "eureka_trigger": eureka.get("eureka_trigger"),
            "recommended_action": recommended_action,
            "calibration_confidence": eureka.get("calibration_confidence"),
            "calibration_anchor_count": eureka.get("calibration_anchor_count"),
            "harvested_at_utc": stamp,
            "authority": "false_authority_dqs1_local_first_harvest",
        },
        dispatch_blockers=(
            "dqs1_harvest_is_not_score_authority",
            "exact_cpu_cuda_auth_eval_required_before_frontier_claim",
        ),
    )
    exact_request = None
    if recommended_action == "dispatch_exact_auth_anchor":
        exact_request = apply_proxy_evidence_boundary(
            {
                "schema": EXACT_AUTH_ANCHOR_REQUEST_SCHEMA,
                "candidate_id": candidate_id,
                "request_reason": "positive_local_cpu_contest_drift_eureka",
                "candidate_archive_path": harvest["candidate_archive_path"],
                "candidate_archive_sha256": harvest["candidate_archive_sha256"],
                "source_harvest_schema": HARVEST_SCHEMA,
                "source_eureka_signal_path": str(eureka_path),
                "requested_axes": ["contest-CPU", "contest-CUDA"],
                "created_at_utc": stamp,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            dispatch_blockers=(
                "requires_lane_dispatch_claim_before_exact_auth",
                "requires_modal_or_contest_runtime_custody_packet",
            ),
        )

    rerouted_queue = None
    if reroute_observe_only and recommended_action == "observe_only":
        summary_path = (
            find_latest_cross_family_action_summary(repo)
            if str(action_summary) == "latest"
            else _resolve(action_summary, repo_root=repo)
        )
        reroute = build_queue_from_action_summary(
            summary_path,
            repo_root=repo,
            results_root=results_root,
            eureka_run_id=stamp,
        )
        rerouted_queue = reroute.queue
        if output_queue_path is not None:
            output_path = _resolve(output_queue_path, repo_root=repo)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(rerouted_queue, indent=2, allow_nan=False) + "\n")

    return Dqs1HarvestResult(
        harvest_record=harvest,
        exact_auth_request=exact_request,
        rerouted_queue=rerouted_queue,
    )


__all__ = [
    "DEFAULT_QUEUE_PATH",
    "EXACT_AUTH_ANCHOR_REQUEST_SCHEMA",
    "HARVEST_SCHEMA",
    "Dqs1HarvestResult",
    "build_dqs1_harvest_result",
]
