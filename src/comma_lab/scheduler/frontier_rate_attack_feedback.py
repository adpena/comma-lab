# SPDX-License-Identifier: MIT
"""Compile frontier materializer feedback into queue-owned follow-up surfaces."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import (
    DQS1_OBSERVATION_SOURCE_SCHEMA,
    DQS1_OBSERVATION_SWEEP_CONFIG_ID,
    FALSE_AUTHORITY,
    build_dqs1_materializer_feedback_bridge,
)
from tac.optimization.materializer_feedback import (
    FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA,
    FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
    materializer_observation_feedback_rows,
)
from tac.optimization.mlx_dynamic_sweep_observations import (
    MLXDynamicSweepObservationError,
    load_observation_rows,
    observation_duplicate_key,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .dqs1_local_first_queue import DEFAULT_QUEUE_ID, DEFAULT_RESULTS_ROOT, build_queue_from_action_summary
from .experiment_queue import ExperimentQueueError, normalize_queue_definition

FEEDBACK_REFRESH_SCHEMA = "frontier_rate_attack_feedback_refresh.v1"
FRONTIER_RATE_ATTACK_FEEDBACK_REFRESH_SCHEMA = FEEDBACK_REFRESH_SCHEMA
MATERIALIZER_FEEDBACK_DISCOVERY_SCHEMA = (
    "frontier_rate_attack_materializer_feedback_discovery.v1"
)
DISCOVERED_MATERIALIZER_FEEDBACK_SCHEMA = (
    "frontier_rate_attack_discovered_materializer_feedback.v1"
)


class FrontierRateAttackFeedbackError(ExperimentQueueError):
    """Raised when frontier feedback discovery or compilation is unsafe."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrontierRateAttackFeedbackError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrontierRateAttackFeedbackError(f"{path}: expected JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FrontierRateAttackFeedbackError(f"{path}: cannot read JSONL") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FrontierRateAttackFeedbackError(
                f"{path}:{index}: invalid JSONL row: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise FrontierRateAttackFeedbackError(
                f"{path}:{index}: expected JSON object row"
            )
        rows.append(row)
    return rows


def _is_materializer_feedback_payload(payload: Mapping[str, Any]) -> bool:
    schema = str(payload.get("schema") or "")
    observation_kind = str(payload.get("observation_kind") or "")
    if schema in {
        FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
        FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_SCHEMA,
    }:
        return True
    if observation_kind == "family_agnostic_materializer_empirical_observation":
        return True
    observations = payload.get("observations")
    if observations is None:
        observations = payload.get("rows")
    if isinstance(observations, list):
        return any(
            isinstance(row, Mapping) and _is_materializer_feedback_payload(row)
            for row in observations
        )
    return False


def _materializer_feedback_paths(root: Path, *, max_files: int) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        raise FrontierRateAttackFeedbackError(f"feedback root does not exist: {root}")
    candidates: list[Path] = []
    scanned_files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        scanned_files += 1
        if scanned_files > max_files:
            raise FrontierRateAttackFeedbackError(
                f"{root}: materializer feedback discovery exceeded max_files={max_files}"
            )
        if path.name in {"sweep.json", "observations.jsonl"} or (path.suffix in {".json", ".jsonl"} and "materializer" in path.as_posix()):
            candidates.append(path)
    return candidates


def _payload_from_materializer_feedback_path(path: Path) -> dict[str, Any] | None:
    if path.suffix == ".jsonl":
        rows = _load_jsonl(path)
        materializer_rows = [
            row for row in rows if _is_materializer_feedback_payload(row)
        ]
        if not materializer_rows:
            return None
        for index, row in enumerate(materializer_rows):
            require_no_truthy_authority_fields(
                row,
                context=f"frontier_rate_attack_feedback.jsonl[{index}]",
            )
        return {
            "schema": FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
            "source_format": "jsonl_observation_rows",
            "observations": materializer_rows,
            **FALSE_AUTHORITY,
        }
    payload = _load_json(path)
    if not _is_materializer_feedback_payload(payload):
        return None
    require_no_truthy_authority_fields(
        payload,
        context="frontier_rate_attack_feedback.materializer_payload",
    )
    return payload


def _materializer_observation_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("observation_id") or ""),
        str(row.get("candidate_id") or ""),
        str(row.get("target_kind") or ""),
        str(row.get("materializer_id") or ""),
        str(row.get("source_archive_sha256") or ""),
        str(row.get("candidate_archive_sha256") or ""),
        str(row.get("saved_bytes") or ""),
        str(row.get("selected_member_name") or ""),
        ",".join(str(item) for item in row.get("selected_member_names") or []),
    )


def discover_materializer_feedback_payloads(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    materializer_feedback_paths: Sequence[str | Path] = (),
    max_files_per_root: int = 256,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], dict[str, Any]]:
    """Discover family-agnostic materializer feedback under frontier roots."""

    repo = Path(repo_root)
    paths: list[Path] = []
    seen_paths: set[str] = set()
    for value in materializer_feedback_paths:
        path = _resolve_path(value, repo_root=repo)
        if path.as_posix() not in seen_paths:
            seen_paths.add(path.as_posix())
            paths.append(path)
    for value in frontier_artifact_roots:
        root = _resolve_path(value, repo_root=repo)
        for path in _materializer_feedback_paths(root, max_files=max_files_per_root):
            if path.as_posix() in seen_paths:
                continue
            seen_paths.add(path.as_posix())
            paths.append(path)

    payloads: list[dict[str, Any]] = []
    source_paths: list[str] = []
    discovered: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    seen_observation_keys: set[tuple[str, ...]] = set()
    duplicate_observation_count = 0
    for path in paths:
        try:
            payload = _payload_from_materializer_feedback_path(path)
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(f"{path}: {exc}") from exc
        rel_path = _repo_rel(path, repo)
        if payload is None:
            ignored.append(
                {
                    "path": rel_path,
                    "reason": "not_family_agnostic_materializer_feedback",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        try:
            rows = materializer_observation_feedback_rows(payload, source_path=rel_path)
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(f"{path}: {exc}") from exc
        unique_rows: list[dict[str, Any]] = []
        duplicate_rows = 0
        for row in rows:
            key = _materializer_observation_key(row)
            if key in seen_observation_keys:
                duplicate_rows += 1
                duplicate_observation_count += 1
                continue
            seen_observation_keys.add(key)
            unique_rows.append(row)
        if not unique_rows:
            ignored.append(
                {
                    "path": rel_path,
                    "reason": (
                        "duplicate_materializer_observations"
                        if duplicate_rows
                        else "materializer_feedback_has_no_observation_rows"
                    ),
                    "duplicate_observation_count": duplicate_rows,
                    **FALSE_AUTHORITY,
                }
            )
            continue
        target_kinds = sorted(
            {
                str(row.get("target_kind"))
                for row in unique_rows
                if str(row.get("target_kind") or "").strip()
            }
        )
        payloads.append(
            {
                "schema": FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_SWEEP_SCHEMA,
                "source_payload_schema": payload.get("schema"),
                "source_format": payload.get("source_format") or path.suffix.lstrip("."),
                "observations": unique_rows,
                **FALSE_AUTHORITY,
            }
        )
        source_paths.append(rel_path)
        discovered.append(
            {
                "schema": DISCOVERED_MATERIALIZER_FEEDBACK_SCHEMA,
                "path": rel_path,
                "payload_schema": payload.get("schema"),
                "observation_count": len(unique_rows),
                "duplicate_observation_count": duplicate_rows,
                "target_kinds": target_kinds,
                "rate_positive_count": sum(
                    1 for row in unique_rows if row.get("rate_positive") is True
                ),
                "receiver_positive_rate_saving_count": sum(
                    1
                    for row in unique_rows
                    if row.get("rate_positive") is True
                    and (
                        row.get("receiver_contract_satisfied") is True
                        or row.get("inflate_parity_satisfied") is True
                    )
                ),
                **FALSE_AUTHORITY,
            }
        )

    discovery = {
        "schema": MATERIALIZER_FEEDBACK_DISCOVERY_SCHEMA,
        "frontier_artifact_roots": [
            _repo_rel(_resolve_path(root, repo_root=repo), repo)
            for root in frontier_artifact_roots
        ],
        "explicit_materializer_feedback_paths": [
            _repo_rel(_resolve_path(path, repo_root=repo), repo)
            for path in materializer_feedback_paths
        ],
        "scanned_candidate_path_count": len(paths),
        "discovered_feedback_count": len(payloads),
        "duplicate_observation_count": duplicate_observation_count,
        "discovered_feedback": discovered,
        "ignored_feedback_candidates": ignored,
        **FALSE_AUTHORITY,
    }
    return tuple(payloads), tuple(source_paths), discovery


def load_dqs1_observations(
    *,
    repo_root: str | Path,
    observation_paths: Sequence[str | Path],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    """Load and dedupe canonical DQS1 local-first observation JSONL rows."""

    repo = Path(repo_root)
    rows: list[dict[str, Any]] = []
    source_paths: list[str] = []
    seen_rows: set[tuple[tuple[str, str | None], ...]] = set()
    seen_paths: set[str] = set()
    for value in observation_paths:
        path = _resolve_path(value, repo_root=repo)
        if path.suffix != ".jsonl":
            raise FrontierRateAttackFeedbackError(
                f"{path}: DQS1 observations must be JSONL rows"
            )
        if path.as_posix() not in seen_paths:
            seen_paths.add(path.as_posix())
            source_paths.append(_repo_rel(path, repo))
        try:
            loaded = load_observation_rows(path)
        except OSError as exc:
            raise FrontierRateAttackFeedbackError(
                f"{path}: cannot read DQS1 observation JSONL"
            ) from exc
        except MLXDynamicSweepObservationError as exc:
            raise FrontierRateAttackFeedbackError(
                f"{path}: invalid DQS1 observation JSONL: {exc}"
            ) from exc
        for row in loaded:
            if (
                row.get("source_schema") != DQS1_OBSERVATION_SOURCE_SCHEMA
                or row.get("sweep_config_id") != DQS1_OBSERVATION_SWEEP_CONFIG_ID
            ):
                raise FrontierRateAttackFeedbackError(
                    f"{path}: non-local-first DQS1 observation row refused "
                    f"for candidate {row.get('candidate_id')!r}"
                )
            key = observation_duplicate_key(row)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            rows.append(row)
    return tuple(rows), tuple(source_paths)


def _queue_summary(queue: Mapping[str, Any]) -> dict[str, Any]:
    experiments = queue.get("experiments")
    experiment_rows = experiments if isinstance(experiments, list) else []
    return {
        "queue_id": queue.get("queue_id"),
        "experiment_count": len(experiment_rows),
        "step_count": sum(
            len(exp.get("steps", []))
            for exp in experiment_rows
            if isinstance(exp, Mapping)
        ),
        "selected_candidate_ids": [
            str(exp.get("id"))
            for exp in experiment_rows
            if isinstance(exp, Mapping) and exp.get("id")
        ],
        **FALSE_AUTHORITY,
    }


def build_frontier_rate_attack_feedback_refresh(
    *,
    repo_root: str | Path,
    frontier_artifact_roots: Sequence[str | Path] = (),
    materializer_feedback_paths: Sequence[str | Path] = (),
    dqs1_observation_paths: Sequence[str | Path] = (),
    action_summary_path: str | Path | None = None,
    results_root: str = DEFAULT_RESULTS_ROOT,
    queue_id: str = DEFAULT_QUEUE_ID,
    candidate_limit: int = 4,
    skip_observed_dqs1_candidates: bool = True,
    local_cpu_concurrency: int = 1,
    local_io_concurrency: int = 1,
    include_raw_retention_plan: bool = True,
    include_mlx_retention_plan: bool = True,
) -> dict[str, Any]:
    """Build a forest-level feedback refresh and optional DQS1 follow-up queue."""

    repo = Path(repo_root)
    if candidate_limit < 1:
        raise FrontierRateAttackFeedbackError("candidate_limit must be >= 1")
    payloads, source_paths, discovery = discover_materializer_feedback_payloads(
        repo_root=repo,
        frontier_artifact_roots=frontier_artifact_roots,
        materializer_feedback_paths=materializer_feedback_paths,
    )
    dqs1_observations, dqs1_source_paths = load_dqs1_observations(
        repo_root=repo,
        observation_paths=dqs1_observation_paths,
    )
    queue_payload: dict[str, Any] | None = None
    bridge: dict[str, Any] | None = None
    selected_candidate_ids: list[str] = []
    if action_summary_path is not None:
        result = build_queue_from_action_summary(
            _resolve_path(action_summary_path, repo_root=repo),
            repo_root=repo,
            results_root=results_root,
            queue_id=queue_id,
            candidate_limit=candidate_limit,
            materializer_feedback_payloads=payloads,
            materializer_feedback_source_paths=source_paths,
            dqs1_observations=dqs1_observations,
            dqs1_observation_source_paths=dqs1_source_paths,
            skip_observed_dqs1_candidates=skip_observed_dqs1_candidates,
            local_cpu_concurrency=local_cpu_concurrency,
            local_io_concurrency=local_io_concurrency,
            include_raw_retention_plan=include_raw_retention_plan,
            include_mlx_retention_plan=include_mlx_retention_plan,
        )
        queue_payload = normalize_queue_definition(result.queue)
        bridge = result.materializer_feedback_bridge
        selected_candidate_ids = [selection.candidate_id for selection in result.selections]
    else:
        try:
            bridge = build_dqs1_materializer_feedback_bridge(
                materializer_feedback_payloads=payloads,
                materializer_feedback_source_paths=source_paths,
                candidate_limit=candidate_limit,
                dqs1_observations=dqs1_observations,
                dqs1_observation_source_paths=dqs1_source_paths,
            )
        except ValueError as exc:
            raise FrontierRateAttackFeedbackError(str(exc)) from exc

    return {
        "schema": FEEDBACK_REFRESH_SCHEMA,
        "generated_at_utc": _utc_now(),
        "discovery": discovery,
        "materializer_feedback_source_paths": list(source_paths),
        "materializer_feedback_payload_count": len(payloads),
        "dqs1_observation_source_paths": list(dqs1_source_paths),
        "dqs1_observation_count": len(dqs1_observations),
        "action_summary_path": (
            None
            if action_summary_path is None
            else _repo_rel(_resolve_path(action_summary_path, repo_root=repo), repo)
        ),
        "queue_id": queue_id if queue_payload is not None else None,
        "results_root": results_root,
        "selected_candidate_ids": selected_candidate_ids,
        "materializer_feedback_bridge": bridge,
        "queue_summary": None if queue_payload is None else _queue_summary(queue_payload),
        "queue": queue_payload,
        "allowed_use": "queue_owned_frontier_feedback_replanning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


__all__ = [
    "DISCOVERED_MATERIALIZER_FEEDBACK_SCHEMA",
    "FEEDBACK_REFRESH_SCHEMA",
    "FRONTIER_RATE_ATTACK_FEEDBACK_REFRESH_SCHEMA",
    "MATERIALIZER_FEEDBACK_DISCOVERY_SCHEMA",
    "FrontierRateAttackFeedbackError",
    "build_frontier_rate_attack_feedback_refresh",
    "discover_materializer_feedback_payloads",
    "load_dqs1_observations",
]
