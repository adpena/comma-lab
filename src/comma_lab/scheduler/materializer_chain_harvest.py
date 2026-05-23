# SPDX-License-Identifier: MIT
"""Harvest completed materializer chains into optimizer source queues.

The experiment queue tells us what ran; the chain manifest tells us what was
actually produced. This module treats queue state as a filter only, then
revalidates live chain manifests through ``tac.optimizer.materializer_chain_harvest``
before emitting planning-only optimizer queue rows.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_MANIFEST_NAME as BYTE_RANGE_CHAIN_MANIFEST_NAME,
)
from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_SCHEMA as BYTE_RANGE_CHAIN_SCHEMA,
)
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_MANIFEST_NAME as INVERSE_SCORER_CELL_CHAIN_MANIFEST_NAME,
)
from tac.optimization.inverse_scorer_cell_chain import (
    CHAIN_SCHEMA as INVERSE_SCORER_CELL_CHAIN_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimizer.candidate_queue import build_candidate_queue
from tac.optimizer.materializer_chain_harvest import (
    SUPPORTED_CHAIN_SCHEMAS,
    MaterializerChainHarvestError,
    adapt_materializer_chain_manifest_to_candidate,
)

from .byte_shaving_campaign_queue import (
    MATERIALIZER_EXECUTION_STEP_ID,
    MATERIALIZER_WORK_QUEUE_SCHEMA,
)
from .experiment_queue import ExperimentQueueError, connect_state_readonly

HARVEST_SCHEMA = "materializer_chain_harvest_report.v1"
TOOL_NAME = "comma_lab.scheduler.materializer_chain_harvest"
CHAIN_MANIFEST_NAME_BY_SCHEMA = {
    BYTE_RANGE_CHAIN_SCHEMA: BYTE_RANGE_CHAIN_MANIFEST_NAME,
    INVERSE_SCORER_CELL_CHAIN_SCHEMA: INVERSE_SCORER_CELL_CHAIN_MANIFEST_NAME,
}
KNOWN_CHAIN_MANIFEST_NAMES = frozenset(CHAIN_MANIFEST_NAME_BY_SCHEMA.values())


def harvest_materializer_chain_manifests(
    *,
    repo_root: str | Path,
    work_queue_path: str | Path | None = None,
    experiment_queue_state_path: str | Path | None = None,
    experiment_queue_id: str | None = None,
    chain_manifest_paths: Sequence[str | Path] = (),
    chain_roots: Sequence[str | Path] = (),
    require_succeeded_state: bool = True,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Return a harvest report and planning-only optimizer source queue."""

    if top_k is not None and (isinstance(top_k, bool) or top_k < 1):
        raise ExperimentQueueError("top_k must be >= 1 when provided")
    repo = Path(repo_root)
    state_rows = _load_state_rows(
        experiment_queue_state_path,
        experiment_queue_id=experiment_queue_id,
    )
    discoveries = _discover_chain_manifest_candidates(
        repo_root=repo,
        work_queue_path=work_queue_path,
        chain_manifest_paths=chain_manifest_paths,
        chain_roots=chain_roots,
    )

    accepted_paths: list[Path] = []
    inspected_rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for discovery in discoveries:
        path = _resolve_path(discovery["path"], repo_root=repo)
        path_key = path.resolve(strict=False)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        row = {
            "path": _repo_rel(path, repo),
            "source": discovery.get("source"),
            "work_id": discovery.get("work_id"),
            "declared_schema": discovery.get("schema"),
            "state_rows": _state_rows_for_discovery(discovery, state_rows),
            "accepted": False,
            "blockers": [],
        }
        state_blockers = _state_blockers(row, require_succeeded_state=require_succeeded_state)
        if state_blockers:
            row["blockers"] = state_blockers
            inspected_rows.append(row)
            continue
        blockers, observed_schema = _validate_chain_manifest(path, repo_root=repo)
        row["observed_schema"] = observed_schema
        if blockers:
            row["blockers"] = blockers
            inspected_rows.append(row)
            continue
        row["accepted"] = True
        accepted_paths.append(path)
        inspected_rows.append(row)

    source_queue = build_candidate_queue(accepted_paths, repo_root=repo, top_k=top_k)
    accepted_rows = [row for row in inspected_rows if row["accepted"] is True]
    rejected_rows = [row for row in inspected_rows if row["accepted"] is not True]
    report = apply_proxy_evidence_boundary(
        {
            "schema": HARVEST_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "work_queue_path": _repo_rel(_resolve_path(work_queue_path, repo_root=repo), repo)
            if work_queue_path is not None
            else None,
            "experiment_queue_state_path": _repo_rel(
                _resolve_path(experiment_queue_state_path, repo_root=repo),
                repo,
            )
            if experiment_queue_state_path is not None
            else None,
            "experiment_queue_id": experiment_queue_id,
            "require_succeeded_state": require_succeeded_state,
            "discovered_manifest_count": len(discoveries),
            "unique_manifest_count": len(inspected_rows),
            "accepted_manifest_count": len(accepted_rows),
            "rejected_manifest_count": len(rejected_rows),
            "accepted_manifest_paths": [row["path"] for row in accepted_rows],
            "rows": inspected_rows,
            "source_queue_schema": source_queue["schema"],
            "source_queue_candidate_count": source_queue["n_candidates"],
            "source_queue_dispatch_ready_count": source_queue["dispatch_ready_count"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        dispatch_blockers=[
            "materializer_chain_harvest_is_source_queue_only",
            "exact_readiness_promotion_required_before_dispatch",
            "lane_claim_required_before_gpu_or_remote_eval",
        ],
    )
    return {"report": report, "source_queue": source_queue}


def _discover_chain_manifest_candidates(
    *,
    repo_root: Path,
    work_queue_path: str | Path | None,
    chain_manifest_paths: Sequence[str | Path],
    chain_roots: Sequence[str | Path],
) -> list[dict[str, Any]]:
    discoveries: list[dict[str, Any]] = []
    if work_queue_path is not None:
        discoveries.extend(
            _work_queue_manifest_candidates(
                _load_json(_resolve_path(work_queue_path, repo_root=repo_root)),
                work_queue_path=_resolve_path(work_queue_path, repo_root=repo_root),
                repo_root=repo_root,
            )
        )
    for raw_path in chain_manifest_paths:
        discoveries.append(
            {
                "source": "explicit_chain_manifest",
                "path": str(raw_path),
                "schema": None,
                "work_id": None,
            }
        )
    for raw_root in chain_roots:
        discoveries.extend(_chain_root_manifest_candidates(raw_root, repo_root=repo_root))
    return discoveries


def _work_queue_manifest_candidates(
    payload: Any,
    *,
    work_queue_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ExperimentQueueError("materializer work queue must be an object")
    if payload.get("schema") != MATERIALIZER_WORK_QUEUE_SCHEMA:
        raise ExperimentQueueError(f"expected schema {MATERIALIZER_WORK_QUEUE_SCHEMA}")
    discoveries: list[dict[str, Any]] = []
    for index, raw_row in enumerate(payload.get("rows") or []):
        if not isinstance(raw_row, Mapping):
            raise ExperimentQueueError(f"materializer work queue row {index} must be an object")
        try:
            require_no_truthy_authority_fields(
                raw_row,
                context=f"materializer_work_queue.rows.{index}",
            )
        except ValueError as exc:
            raise ExperimentQueueError(str(exc)) from exc
        work_id = str(raw_row.get("work_id") or "")
        for condition in raw_row.get("postconditions") or []:
            if not isinstance(condition, Mapping):
                continue
            if condition.get("type") != "materializer_chain_complete":
                continue
            schema = str(condition.get("schema") or "")
            path = condition.get("path")
            if schema not in SUPPORTED_CHAIN_SCHEMAS:
                continue
            if not isinstance(path, str) or not path.strip():
                continue
            discoveries.append(
                {
                    "source": "materializer_work_queue_postcondition",
                    "work_queue_path": _repo_rel(work_queue_path, repo_root),
                    "path": path,
                    "schema": schema,
                    "work_id": work_id or None,
                    "backlog_key": raw_row.get("backlog_key"),
                }
            )
    return discoveries


def _chain_root_manifest_candidates(
    root: str | Path,
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    root_path = _resolve_path(root, repo_root=repo_root)
    if root_path.is_file():
        return [
            {
                "source": "chain_root_file",
                "path": str(root_path),
                "schema": None,
                "work_id": None,
            }
        ]
    if not root_path.is_dir():
        return [
            {
                "source": "chain_root_missing",
                "path": str(root_path),
                "schema": None,
                "work_id": None,
            }
        ]
    discoveries: list[dict[str, Any]] = []
    for name in sorted(KNOWN_CHAIN_MANIFEST_NAMES):
        for path in sorted(root_path.rglob(name)):
            discoveries.append(
                {
                    "source": "chain_root_scan",
                    "path": str(path),
                    "schema": None,
                    "work_id": None,
                }
            )
    return discoveries


def _load_state_rows(
    state_path: str | Path | None,
    *,
    experiment_queue_id: str | None,
) -> dict[str, list[dict[str, Any]]]:
    if state_path is None:
        return {}
    query = """
        SELECT queue_id, experiment_id, step_id, status, attempts,
               updated_at_utc, last_event_json
        FROM step_state
        WHERE step_id = ?
    """
    params: list[Any] = [MATERIALIZER_EXECUTION_STEP_ID]
    if experiment_queue_id is not None:
        query += " AND queue_id = ?"
        params.append(experiment_queue_id)
    query += " ORDER BY queue_id, experiment_id, step_id"
    with connect_state_readonly(state_path) as conn:
        rows = conn.execute(query, params).fetchall()
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        event = _json_or_empty(row["last_event_json"])
        item = {
            "queue_id": row["queue_id"],
            "experiment_id": row["experiment_id"],
            "step_id": row["step_id"],
            "status": row["status"],
            "attempts": row["attempts"],
            "updated_at_utc": row["updated_at_utc"],
            "last_event": event,
        }
        out.setdefault(str(row["experiment_id"]), []).append(item)
    return out


def _state_rows_for_discovery(
    discovery: Mapping[str, Any],
    state_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    work_id = discovery.get("work_id")
    if not isinstance(work_id, str) or not work_id:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in ordered_unique([work_id, _materializer_execution_experiment_id(work_id)]):
        for row in state_rows.get(key, []):
            identity = (
                str(row.get("queue_id") or ""),
                str(row.get("experiment_id") or ""),
                str(row.get("step_id") or ""),
            )
            if identity in seen:
                continue
            seen.add(identity)
            out.append(dict(row))
    return out


def _materializer_execution_experiment_id(work_id: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", work_id.lower()).strip("_") or "row"


def _state_blockers(
    row: Mapping[str, Any],
    *,
    require_succeeded_state: bool,
) -> list[str]:
    if not require_succeeded_state:
        return []
    state_rows = row.get("state_rows")
    work_id = row.get("work_id")
    if not isinstance(work_id, str) or not work_id:
        return []
    if not isinstance(state_rows, list) or not state_rows:
        return [f"experiment_queue_state_missing:{work_id}"]
    statuses = ordered_unique(
        str(item.get("status") or "") for item in state_rows if isinstance(item, Mapping)
    )
    if "succeeded" not in statuses:
        return [f"experiment_queue_state_not_succeeded:{work_id}:{','.join(statuses)}"]
    return []


def _validate_chain_manifest(path: Path, *, repo_root: Path) -> tuple[list[str], str | None]:
    if not path.is_file():
        return [f"chain_manifest_missing:{path}"], None
    if path.is_symlink():
        return [f"chain_manifest_is_symlink:{path}"], None
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"chain_manifest_json_invalid:{exc}"], None
    if not isinstance(payload, Mapping):
        return ["chain_manifest_not_object"], None
    schema = str(payload.get("schema") or "")
    if schema not in SUPPORTED_CHAIN_SCHEMAS:
        return [f"unsupported_chain_schema:{schema!r}"], schema or None
    if payload.get("status") == "failed":
        return ["chain_manifest_status_failed"], schema
    try:
        adapt_materializer_chain_manifest_to_candidate(
            payload,
            source_path=path,
            repo_root=repo_root,
        )
    except MaterializerChainHarvestError as exc:
        return [str(exc)], schema
    return [], schema


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_or_empty(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "HARVEST_SCHEMA",
    "harvest_materializer_chain_manifests",
    "write_json",
]
