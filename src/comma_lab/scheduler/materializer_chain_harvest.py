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
from tac.optimizer.exact_readiness import (
    ACTIVE_FLOOR_ARCHIVE_BYTES,
    ACTIVE_FLOOR_SCORE,
    ExactReadinessError,
    promote_candidate_for_exact_eval,
)
from tac.optimizer.exact_readiness import (
    json_dumps as exact_readiness_json_dumps,
)
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
EXACT_READINESS_BRIDGE_SCHEMA = "materializer_chain_exact_readiness_bridge_report.v1"
TOOL_NAME = "comma_lab.scheduler.materializer_chain_harvest"
EXACT_READINESS_BRIDGE_TOOL = (
    "comma_lab.scheduler.materializer_chain_harvest.exact_readiness_bridge"
)
MATERIALIZER_HARVEST_CLEARABLE_SOURCE_BLOCKERS = (
    "materializer_chain_is_not_dispatch_authorization",
    "materialized_archive_runtime_custody_required",
    "materializer_chain_harvest_candidate_pending_exact_readiness",
    "exact_readiness_promotion_required",
    "exact_auth_eval_result_required_before_score_claim",
    "byte_range_entropy_recode_chain_is_not_dispatch_authorization",
    "inverse_scorer_cell_candidate_chain_is_not_dispatch_authorization",
)
MATERIALIZER_BRIDGE_OPERATOR_CLEARABLE_SOURCE_BLOCKER_ALLOWLIST: frozenset[str] = (
    frozenset()
)
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


def run_exact_readiness_bridge_for_harvested_queue(
    *,
    repo_root: str | Path,
    source_queue_path: str | Path,
    exact_readiness_out_dir: str | Path,
    candidate_ids: Sequence[str] = (),
    allow_source_blockers: Sequence[str] = (),
    dispatch_claims_path: str | Path | None = None,
    claim_ttl_hours: float = 24.0,
    active_floor_archive_bytes: int | None = ACTIVE_FLOOR_ARCHIVE_BYTES,
    active_floor_score: float | None = ACTIVE_FLOOR_SCORE,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
) -> dict[str, Any]:
    """Run the exact-readiness gate for harvested materializer source rows.

    The returned report is an observation artifact. Only per-candidate
    ``*_exact_ready_queue.json`` outputs from the existing promoter are dispatch
    packets, and those still require a lane claim before provider launch.
    """

    repo = Path(repo_root)
    queue_path = _resolve_path(source_queue_path, repo_root=repo)
    out_dir = _resolve_path(exact_readiness_out_dir, repo_root=repo)
    if allow_above_active_floor_dispatch and not operator_override_reason:
        raise ExperimentQueueError(
            "allow_above_active_floor_dispatch requires operator_override_reason"
        )
    queue_payload = _load_json(queue_path)
    if not isinstance(queue_payload, Mapping):
        raise ExperimentQueueError("source queue must be an object")
    if queue_payload.get("schema") != "optimizer_candidate_queue_v1":
        raise ExperimentQueueError(
            f"expected optimizer_candidate_queue_v1, got {queue_payload.get('schema')!r}"
        )
    dispatch_ready_rows = queue_payload.get("dispatch_ready")
    if isinstance(dispatch_ready_rows, list) and dispatch_ready_rows:
        raise ExperimentQueueError(
            "exact_readiness_bridge_source_queue_must_not_have_dispatch_ready_rows"
        )
    candidate_filter = {str(candidate_id) for candidate_id in candidate_ids if str(candidate_id)}
    rows = [
        row
        for row in queue_payload.get("top_k") or []
        if isinstance(row, Mapping)
        and (
            not candidate_filter
            or str(row.get("candidate_id") or "") in candidate_filter
        )
    ]
    if candidate_filter:
        found = {str(row.get("candidate_id") or "") for row in rows}
        missing = sorted(candidate_filter - found)
        if missing:
            raise ExperimentQueueError(
                "exact_readiness_candidate_id_missing:" + ",".join(missing)
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    extra_clearable_source_blockers = _validated_bridge_extra_clearable_source_blockers(
        allow_source_blockers,
        operator_override_reason=operator_override_reason,
    )
    clearable_source_blockers = ordered_unique(
        [
            *MATERIALIZER_HARVEST_CLEARABLE_SOURCE_BLOCKERS,
            *extra_clearable_source_blockers,
        ]
    )
    _require_bridge_source_queue_identity(rows, candidate_filter=candidate_filter)
    resolved_dispatch_claims_path = (
        _resolve_path(dispatch_claims_path, repo_root=repo)
        if dispatch_claims_path is not None
        else repo / ".omx" / "state" / "active_lane_dispatch_claims.md"
    )
    report_rows: list[dict[str, Any]] = []
    ready_count = 0
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        slug = _safe_slug(candidate_id)
        per_candidate_report_path = out_dir / f"{slug}.exact_readiness_report.json"
        exact_ready_queue_path = out_dir / f"{slug}.exact_ready_queue.json"
        try:
            result = promote_candidate_for_exact_eval(
                queue_path,
                candidate_id,
                repo_root=repo,
                active_floor_archive_bytes=active_floor_archive_bytes,
                active_floor_score=active_floor_score,
                allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
                operator_override_reason=operator_override_reason,
                extra_clearable_source_blockers=clearable_source_blockers,
                dispatch_claims_path=resolved_dispatch_claims_path,
                claim_ttl_hours=claim_ttl_hours,
            )
            readiness_report = result["report"]
            promoted_queue = result["promoted_queue"]
        except ExactReadinessError as exc:
            readiness_report = {
                "schema": "optimizer_candidate_exact_eval_readiness_report_v1",
                "tool": "tools/promote_optimizer_candidate_for_exact_eval.py",
                "generated_at_utc": _utc_now(),
                "source_queue_path": _repo_rel(queue_path, repo),
                "candidate_id": candidate_id,
                "ready_for_exact_eval_dispatch": False,
                "blockers": [str(exc)],
                "facts": {},
            }
            promoted_queue = None
        per_candidate_report_path.write_text(
            exact_readiness_json_dumps(readiness_report),
            encoding="utf-8",
        )
        ready = promoted_queue is not None
        if ready:
            exact_ready_queue_path.write_text(
                exact_readiness_json_dumps(promoted_queue),
                encoding="utf-8",
            )
            ready_count += 1
        report_rows.append(
            {
                "candidate_id": candidate_id,
                "readiness_verdict": "exact_ready_queue_written" if ready else "blocked",
                "exact_ready_queue_written": ready,
                "exact_readiness_report_path": _repo_rel(per_candidate_report_path, repo),
                "exact_ready_queue_path": _repo_rel(exact_ready_queue_path, repo)
                if ready
                else None,
                "blockers": list(readiness_report.get("blockers") or []),
            }
        )

    report = apply_proxy_evidence_boundary(
        {
            "schema": EXACT_READINESS_BRIDGE_SCHEMA,
            "tool": EXACT_READINESS_BRIDGE_TOOL,
            "generated_at_utc": _utc_now(),
            "source_queue_path": _repo_rel(queue_path, repo),
            "exact_readiness_out_dir": _repo_rel(out_dir, repo),
            "candidate_count": len(rows),
            "ready_candidate_count": ready_count,
            "blocked_candidate_count": len(rows) - ready_count,
            "clearable_source_blockers": clearable_source_blockers,
            "operator_clearable_source_blockers": extra_clearable_source_blockers,
            "dispatch_claims_path": _repo_rel(resolved_dispatch_claims_path, repo),
            "rows": report_rows,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        dispatch_blockers=[
            "bridge_report_is_not_dispatch_authority",
            "use_per_candidate_exact_ready_queue_only_when_present",
            "lane_claim_required_before_gpu_or_remote_eval",
        ],
    )
    require_no_truthy_authority_fields(
        report,
        context="materializer_chain_exact_readiness_bridge_report",
    )
    return report


def _validated_bridge_extra_clearable_source_blockers(
    blockers: Sequence[str],
    *,
    operator_override_reason: str | None,
) -> list[str]:
    extras = ordered_unique(str(item) for item in blockers if str(item))
    if not extras:
        return []
    if not operator_override_reason:
        raise ExperimentQueueError(
            "exact_readiness_extra_source_blocker_requires_operator_override_reason"
        )
    not_allowed = [
        blocker
        for blocker in extras
        if blocker not in MATERIALIZER_BRIDGE_OPERATOR_CLEARABLE_SOURCE_BLOCKER_ALLOWLIST
    ]
    if not_allowed:
        raise ExperimentQueueError(
            "exact_readiness_extra_source_blocker_not_allowlisted:"
            + ",".join(sorted(not_allowed))
        )
    return extras


def _require_bridge_source_queue_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_filter: set[str],
) -> None:
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    missing_ids = 0
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            missing_ids += 1
            continue
        if candidate_id in seen:
            duplicate_ids.add(candidate_id)
        seen.add(candidate_id)
    if missing_ids:
        raise ExperimentQueueError("exact_readiness_candidate_id_missing_in_source_row")
    if duplicate_ids:
        raise ExperimentQueueError(
            "exact_readiness_duplicate_candidate_id:"
            + ",".join(sorted(duplicate_ids))
        )
    if candidate_filter and not rows:
        raise ExperimentQueueError("exact_readiness_candidate_filter_matched_no_rows")


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


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:120] or "candidate"


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
    "EXACT_READINESS_BRIDGE_SCHEMA",
    "HARVEST_SCHEMA",
    "harvest_materializer_chain_manifests",
    "run_exact_readiness_bridge_for_harvested_queue",
    "write_json",
]
