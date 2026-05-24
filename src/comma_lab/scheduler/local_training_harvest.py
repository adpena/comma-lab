# SPDX-License-Identifier: MIT
"""Harvest completed local-training queue outputs into optimizer candidates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.optimization.representation_training_probe_integration import (
    SCHEMA as REPRESENTATION_TRAINING_MANIFEST_SCHEMA,
)
from tac.optimization.representation_training_probe_integration import (
    validate_representation_training_manifest,
)
from tac.optimizer.candidate_queue import build_candidate_queue

from .experiment_queue import (
    ExperimentQueueError,
    connect_state_readonly,
    normalize_queue_definition,
)
from .local_training_queue import LOCAL_TRAINING_QUEUE_SCHEMA

LOCAL_TRAINING_HARVEST_SCHEMA = "local_training_optimizer_candidate_harvest.v1"


class LocalTrainingHarvestError(ExperimentQueueError):
    """Raised when local-training harvest would consume unsafe artifacts."""


def _repo_path(path: str, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def _repo_rel(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(
            repo_root.resolve(strict=False)
        ).as_posix()
    except ValueError:
        return path.as_posix()


def _state_rows(state_path: str | Path, *, queue_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    with connect_state_readonly(state_path) as conn:
        rows = conn.execute(
            """
            SELECT experiment_id, step_id, status, attempts, updated_at_utc,
                   last_event_json
            FROM step_state
            WHERE queue_id = ?
            """,
            (queue_id,),
        ).fetchall()
    return {
        (str(row["experiment_id"]), str(row["step_id"])): {
            "status": str(row["status"]),
            "attempts": int(row["attempts"] or 0),
            "updated_at_utc": row["updated_at_utc"],
            "last_event": (
                json.loads(row["last_event_json"])
                if row["last_event_json"]
                else None
            ),
        }
        for row in rows
    }


def _manifest_path_from_step(step: Mapping[str, Any]) -> str | None:
    schema_paths: list[str] = []
    false_authority_paths: set[str] = set()
    for condition in step.get("postconditions") or []:
        if not isinstance(condition, Mapping):
            continue
        path = str(condition.get("path") or "").strip()
        if not path:
            continue
        if condition.get("type") == "json_false_authority":
            false_authority_paths.add(path)
        if (
            condition.get("type") == "json_equals"
            and condition.get("key") == "schema"
            and condition.get("equals") == REPRESENTATION_TRAINING_MANIFEST_SCHEMA
        ):
            schema_paths.append(path)
    if not schema_paths:
        return None
    if len(schema_paths) > 1:
        raise LocalTrainingHarvestError(
            f"{step.get('id')}: multiple representation manifest postconditions"
        )
    path = schema_paths[0]
    if Path(path).name != "representation_training_manifest.json":
        raise LocalTrainingHarvestError(
            "local training harvest refuses non-manifest sidecar path "
            f"{path!r}; expected representation_training_manifest.json"
        )
    if path not in false_authority_paths:
        raise LocalTrainingHarvestError(
            f"{path}: representation manifest missing json_false_authority postcondition"
        )
    return path


def _load_completed_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LocalTrainingHarvestError(f"representation manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LocalTrainingHarvestError(f"{path}: representation manifest must be JSON object")
    if payload.get("schema") != REPRESENTATION_TRAINING_MANIFEST_SCHEMA:
        raise LocalTrainingHarvestError(
            f"{path}: expected schema {REPRESENTATION_TRAINING_MANIFEST_SCHEMA}"
        )
    try:
        validate_representation_training_manifest(payload)
    except ValueError as exc:
        raise LocalTrainingHarvestError(f"{path}: {exc}") from exc
    return payload


def _manifest_value(payload: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _assert_manifest_matches_queue_identity(
    payload: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    path: Path,
) -> None:
    expected_keys = {
        "candidate_id": "candidate_id",
        "stage_index": "candidate_params.stage_index",
        "seed": "seed",
        "optimizer_descriptor_id": "candidate_params.optimizer_descriptor_id",
        "optimizer_config_sha256": "candidate_params.optimizer_config_sha256",
        "parameter_group_lr_policy_id": "candidate_params.parameter_group_lr_policy_id",
        "parameter_group_lr_policy_sha256": "candidate_params.parameter_group_lr_policy_sha256",
    }
    for metadata_key, manifest_key in expected_keys.items():
        if metadata_key not in metadata:
            continue
        expected = metadata[metadata_key]
        if expected is None:
            continue
        actual = _manifest_value(payload, manifest_key)
        if actual != expected:
            raise LocalTrainingHarvestError(
                f"{path}: manifest identity mismatch for {manifest_key} "
                f"expected={expected!r} actual={actual!r}"
            )


def harvest_local_training_optimizer_candidates(
    queue: Mapping[str, Any],
    *,
    state_path: str | Path,
    repo_root: str | Path,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Build an optimizer candidate queue from succeeded local-training steps.

    Harvest is intentionally queue-state driven: it consumes only completed
    ``representation_training_manifest.json`` sidecars from succeeded
    ``local_training_execution_queue_plan.v1`` experiments. Plan sidecars are
    refused so a pre-execution plan cannot masquerade as a completed training
    observation.
    """

    repo = Path(repo_root)
    normalized_queue = normalize_queue_definition(queue)
    rows_by_step = _state_rows(state_path, queue_id=str(normalized_queue["queue_id"]))
    manifest_paths: list[Path] = []
    skipped_steps: list[dict[str, Any]] = []
    for experiment in normalized_queue["experiments"]:
        metadata = dict(experiment.get("metadata") or {})
        if metadata.get("schema") != LOCAL_TRAINING_QUEUE_SCHEMA:
            continue
        for step in experiment["steps"]:
            key = (str(experiment["id"]), str(step["id"]))
            state = rows_by_step.get(key)
            if state is None:
                skipped_steps.append(
                    {
                        "experiment_id": key[0],
                        "step_id": key[1],
                        "reason": "state_missing",
                    }
                )
                continue
            if state["status"] != "succeeded":
                skipped_steps.append(
                    {
                        "experiment_id": key[0],
                        "step_id": key[1],
                        "status": state["status"],
                        "reason": "step_not_succeeded",
                    }
                )
                continue
            manifest_path = _manifest_path_from_step(step)
            if manifest_path is None:
                raise LocalTrainingHarvestError(
                    f"{key[0]}.{key[1]} succeeded without representation manifest "
                    "postcondition"
                )
            resolved = _repo_path(manifest_path, repo_root=repo)
            payload = _load_completed_manifest(resolved)
            _assert_manifest_matches_queue_identity(
                payload,
                metadata=metadata,
                path=resolved,
            )
            manifest_paths.append(resolved)
    if not manifest_paths:
        raise LocalTrainingHarvestError(
            "no succeeded local-training representation manifests available for harvest"
        )
    candidate_queue = build_candidate_queue(
        manifest_paths,
        repo_root=repo,
        top_k=top_k,
    )
    candidate_queue["harvest"] = {
        "schema": LOCAL_TRAINING_HARVEST_SCHEMA,
        "source_queue_id": normalized_queue["queue_id"],
        "state_path": _repo_rel(Path(state_path), repo_root=repo),
        "harvested_representation_manifest_count": len(manifest_paths),
        "harvested_representation_manifest_paths": [
            _repo_rel(path, repo_root=repo) for path in manifest_paths
        ],
        "skipped_steps": skipped_steps,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return candidate_queue


__all__ = [
    "LOCAL_TRAINING_HARVEST_SCHEMA",
    "LocalTrainingHarvestError",
    "harvest_local_training_optimizer_candidates",
]
