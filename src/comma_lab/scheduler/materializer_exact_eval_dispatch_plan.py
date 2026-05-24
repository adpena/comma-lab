# SPDX-License-Identifier: MIT
"""Build queue-owned exact-eval dispatch plans from materializer ready queues."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimizer.exact_dispatch_authority import exact_dispatch_authority
from tac.optimizer.exact_readiness import (
    ACTIVE_FLOOR_ARCHIVE_BYTES,
    ACTIVE_FLOOR_SCORE,
)
from tac.optimizer.exact_readiness import (
    QUEUE_SCHEMA as EXACT_READY_QUEUE_SCHEMA,
)
from tac.optimizer.exact_ready_audit import audit_exact_ready_queue

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition
from .materializer_chain_harvest import EXACT_READINESS_BRIDGE_SCHEMA

DISPATCH_PLAN_SCHEMA = "materializer_exact_eval_dispatch_plan.v1"
TOOL_NAME = "comma_lab.scheduler.materializer_exact_eval_dispatch_plan"
CLAIM_STEP_ID = "claim_lane_dispatch"
DISPATCH_STEP_ID = "dispatch_exact_eval"
DRY_RUN_DISPATCH_STEP_ID = "dispatch_exact_eval_dry_run"
SUPPORTED_DISPATCH_MODES = frozenset({"dry_run", "execute"})
SUPPORTED_PROVIDERS = frozenset({"lightning", "vastai"})
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}


def build_materializer_exact_eval_dispatch_plan(
    *,
    repo_root: str | Path,
    bridge_report_path: str | Path | None = None,
    exact_ready_queue_paths: Sequence[str | Path] = (),
    experiment_queue_id: str = "materializer_exact_eval_dispatch_queue",
    dispatch_mode: str = "dry_run",
    allow_paid_dispatch_queue: bool = False,
    provider: str = "lightning",
    max_concurrency: int = 1,
    estimated_cost_per_dispatch: float = 0.30,
    max_total_cost: float = 5.00,
    label_prefix: str = "materializer_exact_eval",
    agent: str = "codex",
    dispatch_claims_path: str | Path | None = None,
    active_floor_archive_bytes: int | None = ACTIVE_FLOOR_ARCHIVE_BYTES,
    active_floor_score: float | None = ACTIVE_FLOOR_SCORE,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
) -> dict[str, Any]:
    """Return a dispatch plan and an ``experiment_queue.v1`` definition.

    The queue defaults to dry-run dispatch commands. Real paid dispatch queue
    generation requires ``dispatch_mode='execute'`` and
    ``allow_paid_dispatch_queue=True``.
    """

    repo = Path(repo_root)
    mode = str(dispatch_mode)
    if mode not in SUPPORTED_DISPATCH_MODES:
        raise ExperimentQueueError(f"unsupported_dispatch_mode:{mode}")
    if mode == "execute" and not allow_paid_dispatch_queue:
        raise ExperimentQueueError(
            "execute_dispatch_queue_requires_allow_paid_dispatch_queue"
        )
    if provider not in SUPPORTED_PROVIDERS:
        raise ExperimentQueueError(f"unsupported_dispatch_provider:{provider}")
    if max_concurrency < 1:
        raise ExperimentQueueError("max_concurrency must be >= 1")
    if estimated_cost_per_dispatch <= 0:
        raise ExperimentQueueError("estimated_cost_per_dispatch must be > 0")
    if max_total_cost <= 0:
        raise ExperimentQueueError("max_total_cost must be > 0")
    if allow_above_active_floor_dispatch and not operator_override_reason:
        raise ExperimentQueueError(
            "allow_above_active_floor_dispatch requires operator_override_reason"
        )

    claims_path = (
        _resolve_path(dispatch_claims_path, repo_root=repo)
        if dispatch_claims_path is not None
        else repo / ".omx" / "state" / "active_lane_dispatch_claims.md"
    )
    ready_queue_paths, bridge_summary = _collect_ready_queue_paths(
        bridge_report_path=bridge_report_path,
        exact_ready_queue_paths=exact_ready_queue_paths,
        repo_root=repo,
    )

    rows: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    seen_stable_identity: set[str] = set()
    authorized_count = 0
    blocked_count = 0
    duplicate_count = 0
    for queue_path in ready_queue_paths:
        try:
            queue_row = _load_single_dispatch_ready_row(queue_path)
        except ExperimentQueueError as exc:
            rows.append(
                _blocked_plan_row(
                    queue_path,
                    repo,
                    candidate_id=None,
                    stable_identity=None,
                    blockers=[str(exc)],
                )
            )
            blocked_count += 1
            continue
        candidate_id = str(queue_row.get("candidate_id") or "")
        stable_identity, identity_blockers = _stable_candidate_identity(queue_row)
        if identity_blockers:
            rows.append(
                _blocked_plan_row(
                    queue_path,
                    repo,
                    candidate_id=candidate_id,
                    stable_identity=stable_identity,
                    blockers=identity_blockers,
                    lane_id=_row_lane_id(queue_row),
                    archive_sha256=_row_archive_sha(queue_row),
                )
            )
            blocked_count += 1
            continue
        if candidate_id in seen_candidate_ids:
            rows.append(
                _blocked_plan_row(
                    queue_path,
                    repo,
                    candidate_id=candidate_id,
                    stable_identity=stable_identity,
                    blockers=[f"duplicate_candidate_id:{candidate_id}"],
                    lane_id=_row_lane_id(queue_row),
                    archive_sha256=_row_archive_sha(queue_row),
                )
            )
            blocked_count += 1
            duplicate_count += 1
            continue
        if stable_identity in seen_stable_identity:
            rows.append(
                _blocked_plan_row(
                    queue_path,
                    repo,
                    candidate_id=candidate_id,
                    stable_identity=stable_identity,
                    blockers=[f"duplicate_stable_identity:{stable_identity}"],
                    lane_id=_row_lane_id(queue_row),
                    archive_sha256=_row_archive_sha(queue_row),
                )
            )
            blocked_count += 1
            duplicate_count += 1
            continue
        seen_candidate_ids.add(candidate_id)
        seen_stable_identity.add(stable_identity)
        blockers, facts = _exact_ready_queue_blockers(
            queue_path=queue_path,
            row=queue_row,
            repo_root=repo,
            dispatch_claims_path=claims_path,
            active_floor_archive_bytes=active_floor_archive_bytes,
            active_floor_score=active_floor_score,
            allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
            operator_override_reason=operator_override_reason,
        )
        if blockers:
            rows.append(
                _blocked_plan_row(
                    queue_path,
                    repo,
                    candidate_id=candidate_id,
                    stable_identity=stable_identity,
                    blockers=blockers,
                    lane_id=_row_lane_id(queue_row),
                    archive_sha256=_row_archive_sha(queue_row),
                )
            )
            blocked_count += 1
            continue
        lane_id = _row_lane_id(queue_row)
        archive_sha = _row_archive_sha(queue_row)
        job_id = _dispatch_job_id(
            label_prefix=label_prefix,
            candidate_id=candidate_id,
            archive_sha256=archive_sha,
        )
        claim_command = _claim_command(
            lane_id=lane_id,
            provider=provider,
            job_id=job_id,
            agent=agent,
            dispatch_claims_path=claims_path,
            plan_label=label_prefix,
            dry_run=mode == "dry_run",
        )
        dispatch_command = _dispatch_command(
            queue_path=queue_path,
            provider=provider,
            label_prefix=label_prefix,
            estimated_cost_per_dispatch=estimated_cost_per_dispatch,
            max_total_cost=max_total_cost,
            active_floor_archive_bytes=active_floor_archive_bytes,
            active_floor_score=active_floor_score,
            allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
            operator_override_reason=operator_override_reason,
            dispatch_claims_path=claims_path,
            required_claim_job_id=job_id,
            dry_run=mode == "dry_run",
        )
        experiments.append(
            _dispatch_experiment(
                experiment_id=_safe_slug(job_id),
                claim_command=claim_command,
                dispatch_command=dispatch_command,
                dispatch_mode=mode,
                candidate_id=candidate_id,
                stable_identity=stable_identity,
                lane_id=lane_id,
                queue_path=queue_path,
                repo_root=repo,
            )
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "stable_identity": stable_identity,
                "lane_id": lane_id,
                "archive_sha256": archive_sha,
                "runtime_tree_sha256": _row_runtime_tree_sha(queue_row),
                "runtime_content_tree_sha256": _row_runtime_content_sha(queue_row),
                "score_axis": _row_score_axis(queue_row),
                "exact_ready_queue_path": _repo_rel(queue_path, repo),
                "dispatch_job_id": job_id,
                "dispatch_mode": mode,
                "provider": provider,
                "authorized_for_dispatch_plan": True,
                "claim_required_before_dispatch": True,
                "claim_command": claim_command,
                "dispatch_command": dispatch_command,
                "blockers": [],
                "facts": {
                    "audit_stale_ready_row_count": facts.get("audit_stale_ready_row_count"),
                    "authority_source": facts.get("authority_source"),
                },
                **FALSE_AUTHORITY,
            }
        )
        authorized_count += 1

    estimated_total_cost = authorized_count * estimated_cost_per_dispatch
    hard_plan_blockers: list[str] = []
    plan_blockers: list[str] = []
    if estimated_total_cost > max_total_cost:
        hard_plan_blockers.append(
            "estimated_total_cost_exceeds_cap:"
            f"{estimated_total_cost:.2f}>{max_total_cost:.2f}"
        )
    plan_blockers.extend(hard_plan_blockers)
    if mode == "execute":
        plan_blockers.append("execute_dispatch_queue_created_requires_operator_review")
    queue_experiments = [] if hard_plan_blockers else experiments
    queue_mode = "paused"
    dispatch_queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": experiment_queue_id,
            "controls": {
                "mode": queue_mode,
                "max_concurrency": {"local_cpu": max_concurrency},
            },
            "experiments": queue_experiments or [
                {
                    "id": (
                        "frozen_materializer_exact_eval_dispatch"
                        if hard_plan_blockers
                        else "no_authorized_materializer_exact_eval_dispatch"
                    ),
                    "steps": [
                        {
                            "id": "noop",
                            "command": [
                                sys.executable,
                                "-c",
                                (
                                    "print('materializer exact-eval dispatch queue frozen "
                                    "by plan blockers')"
                                    if hard_plan_blockers
                                    else "print('no authorized materializer exact-eval dispatch rows')"
                                ),
                            ],
                            "resources": {"kind": "local_cpu"},
                        }
                    ],
                    "metadata": {
                        "dispatch_plan_schema": DISPATCH_PLAN_SCHEMA,
                        "reason": hard_plan_blockers or "no_authorized_rows",
                        **FALSE_AUTHORITY,
                    },
                }
            ],
        }
    )
    plan = apply_proxy_evidence_boundary(
        {
            "schema": DISPATCH_PLAN_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "bridge_report_path": bridge_summary.get("bridge_report_path"),
            "bridge_ready_candidate_count": bridge_summary.get("ready_candidate_count"),
            "exact_ready_queue_count": len(ready_queue_paths),
            "authorized_candidate_count": authorized_count,
            "blocked_candidate_count": blocked_count,
            "duplicate_candidate_count": duplicate_count,
            "dispatch_mode": mode,
            "provider": provider,
            "estimated_cost_per_dispatch": estimated_cost_per_dispatch,
            "estimated_total_cost": estimated_total_cost,
            "max_total_cost": max_total_cost,
            "dispatch_claims_path": _repo_rel(claims_path, repo),
            "experiment_queue_schema": dispatch_queue["schema"],
            "experiment_queue_id": dispatch_queue["queue_id"],
            "experiment_count": len(queue_experiments),
            "plan_blockers": plan_blockers,
            "hard_plan_blockers": hard_plan_blockers,
            "rows": rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=[
            "dispatch_plan_is_not_score_authority",
            "lane_claim_step_must_succeed_before_dispatch_step",
            "contest_auth_eval_result_required_before_score_claim",
            *plan_blockers,
        ],
    )
    require_no_truthy_authority_fields(
        plan,
        context="materializer_exact_eval_dispatch_plan",
    )
    return {"plan": plan, "experiment_queue": dispatch_queue}


def _collect_ready_queue_paths(
    *,
    bridge_report_path: str | Path | None,
    exact_ready_queue_paths: Sequence[str | Path],
    repo_root: Path,
) -> tuple[list[Path], dict[str, Any]]:
    paths = [_resolve_path(path, repo_root=repo_root) for path in exact_ready_queue_paths]
    summary: dict[str, Any] = {}
    if bridge_report_path is not None:
        bridge_path = _resolve_path(bridge_report_path, repo_root=repo_root)
        bridge = _load_json(bridge_path)
        if not isinstance(bridge, Mapping):
            raise ExperimentQueueError("bridge_report_not_object")
        if bridge.get("schema") != EXACT_READINESS_BRIDGE_SCHEMA:
            raise ExperimentQueueError(
                f"bridge_report_schema_unsupported:{bridge.get('schema')!r}"
            )
        require_no_truthy_authority_fields(
            bridge,
            context="materializer_exact_readiness_bridge_report",
        )
        summary = {
            "bridge_report_path": _repo_rel(bridge_path, repo_root),
            "ready_candidate_count": bridge.get("ready_candidate_count"),
        }
        for row in bridge.get("rows") or []:
            if not isinstance(row, Mapping):
                continue
            if row.get("exact_ready_queue_written") is not True:
                continue
            queue_path = row.get("exact_ready_queue_path")
            if isinstance(queue_path, str) and queue_path.strip():
                paths.append(_resolve_path(queue_path, repo_root=repo_root))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve(strict=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped, summary


def _load_single_dispatch_ready_row(queue_path: Path) -> Mapping[str, Any]:
    payload = _load_json(queue_path)
    if not isinstance(payload, Mapping):
        raise ExperimentQueueError(f"exact_ready_queue_not_object:{queue_path}")
    if payload.get("schema") != EXACT_READY_QUEUE_SCHEMA:
        raise ExperimentQueueError(
            f"exact_ready_queue_schema_unsupported:{payload.get('schema')!r}"
        )
    rows = payload.get("dispatch_ready")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ExperimentQueueError(
            f"exact_ready_queue_must_have_one_dispatch_ready_row:{queue_path}"
        )
    queue_blockers = _exact_ready_queue_shape_blockers(payload, queue_path)
    if queue_blockers:
        raise ExperimentQueueError(",".join(queue_blockers))
    candidate_id = rows[0].get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ExperimentQueueError(f"dispatch_ready_row_candidate_id_missing:{queue_path}")
    return rows[0]


def _exact_ready_queue_shape_blockers(
    payload: Mapping[str, Any],
    queue_path: Path,
) -> list[str]:
    blockers: list[str] = []
    rows = payload.get("dispatch_ready")
    dispatch_ready = rows if isinstance(rows, list) else []
    dispatch_ready_count = payload.get("dispatch_ready_count")
    if dispatch_ready_count != 1:
        blockers.append(
            f"exact_ready_queue_dispatch_ready_count_mismatch:{queue_path}:"
            f"{dispatch_ready_count!r}!=1"
        )
    n_candidates = payload.get("n_candidates")
    if n_candidates is not None and n_candidates != 1:
        blockers.append(
            f"exact_ready_queue_n_candidates_mismatch:{queue_path}:{n_candidates!r}!=1"
        )
    top_k = payload.get("top_k")
    if not isinstance(top_k, list) or len(top_k) != 1 or not isinstance(top_k[0], Mapping):
        blockers.append(f"exact_ready_queue_top_k_must_have_one_row:{queue_path}")
        return blockers
    top_k_count = payload.get("top_k_count")
    if top_k_count is not None and top_k_count != 1:
        blockers.append(
            f"exact_ready_queue_top_k_count_mismatch:{queue_path}:{top_k_count!r}!=1"
        )
    if dispatch_ready and isinstance(dispatch_ready[0], Mapping) and dict(top_k[0]) != dict(dispatch_ready[0]):
        blockers.append(f"exact_ready_queue_top_k_dispatch_ready_mismatch:{queue_path}")
    return blockers


def _stable_candidate_identity(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Return the score-affecting archive/runtime identity for dedupe."""

    archive_sha = _row_archive_sha(row)
    runtime_content_sha = _row_runtime_content_sha(row)
    runtime_tree_sha = _row_runtime_tree_sha(row)
    score_axis = _row_score_axis(row)
    blockers: list[str] = []
    blockers.extend(_archive_sha_alias_blockers(row))
    if archive_sha is None:
        blockers.append("stable_identity_archive_sha256_missing")
    if runtime_content_sha is None:
        blockers.append("stable_identity_runtime_content_tree_sha256_missing")
    if runtime_tree_sha is None:
        blockers.append("stable_identity_runtime_tree_sha256_missing")
    if blockers:
        fallback = str(row.get("candidate_id") or "").strip()
        return (f"unstable:{fallback}" if fallback else "unstable:missing_candidate_id"), blockers
    return (
        "archive="
        f"{archive_sha}:runtime_content={runtime_content_sha}:"
        f"score_axis={score_axis}",
        [],
    )


def _exact_ready_queue_blockers(
    *,
    queue_path: Path,
    row: Mapping[str, Any],
    repo_root: Path,
    dispatch_claims_path: Path,
    active_floor_archive_bytes: int | None,
    active_floor_score: float | None,
    allow_above_active_floor_dispatch: bool,
    operator_override_reason: str | None,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    facts: dict[str, Any] = {}
    candidate_id = str(row.get("candidate_id") or "")
    audit_repo = _audit_repo_root(queue_path, repo_root)
    audit = audit_exact_ready_queue(
        queue_path,
        repo_root=audit_repo,
        dispatch_claims_path=dispatch_claims_path,
        active_floor_score=active_floor_score,
        candidate_ids=[candidate_id],
    )
    stale_rows = audit.get("stale_ready_rows")
    stale_count = len(stale_rows) if isinstance(stale_rows, list) else 0
    facts["audit_stale_ready_row_count"] = stale_count
    if stale_count:
        blockers.append(f"exact_ready_queue_audit_stale_rows:{stale_count}")
    authority = exact_dispatch_authority(
        row,
        repo_root=audit_repo,
        queue_dir=queue_path.parent,
        source=TOOL_NAME,
        active_floor_archive_bytes=active_floor_archive_bytes,
        active_floor_score=active_floor_score,
        allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
        operator_override_reason=operator_override_reason,
        dispatch_claims_path=dispatch_claims_path,
        claim_policy="preclaim_conflict_check",
    )
    facts["authority_source"] = authority.source
    blockers.extend(f"exact_dispatch_authority:{blocker}" for blocker in authority.blockers)
    return ordered_unique(blockers), facts


def _audit_repo_root(queue_path: Path, repo_root: Path) -> Path:
    queue_root = queue_path.parent.resolve()
    return queue_root if (queue_root / "upstream" / "evaluate.py").is_file() else repo_root


def _blocked_plan_row(
    queue_path: Path,
    repo_root: Path,
    *,
    candidate_id: str | None,
    stable_identity: str | None = None,
    blockers: Sequence[str],
    lane_id: str | None = None,
    archive_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "stable_identity": stable_identity,
        "lane_id": lane_id,
        "archive_sha256": archive_sha256,
        "exact_ready_queue_path": _repo_rel(queue_path, repo_root),
        "authorized_for_dispatch_plan": False,
        "claim_required_before_dispatch": True,
        "blockers": list(blockers),
        **FALSE_AUTHORITY,
    }


def _dispatch_experiment(
    *,
    experiment_id: str,
    claim_command: Sequence[str],
    dispatch_command: Sequence[str],
    dispatch_mode: str,
    candidate_id: str,
    stable_identity: str,
    lane_id: str,
    queue_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    dispatch_step_id = (
        DRY_RUN_DISPATCH_STEP_ID if dispatch_mode == "dry_run" else DISPATCH_STEP_ID
    )
    return {
        "id": experiment_id,
        "metadata": {
            "candidate_id": candidate_id,
            "stable_identity": stable_identity,
            "lane_id": lane_id,
            "exact_ready_queue_path": _repo_rel(queue_path, repo_root),
            "dispatch_mode": dispatch_mode,
            **FALSE_AUTHORITY,
        },
        "steps": [
            {
                "id": CLAIM_STEP_ID,
                "command": list(claim_command),
                "resources": {"kind": "local_cpu"},
            },
            {
                "id": dispatch_step_id,
                "requires": [CLAIM_STEP_ID],
                "command": list(dispatch_command),
                "resources": {"kind": "local_cpu"},
            },
        ],
    }


def _claim_command(
    *,
    lane_id: str,
    provider: str,
    job_id: str,
    agent: str,
    dispatch_claims_path: Path,
    plan_label: str,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        "tools/claim_lane_dispatch.py",
        "claim",
        "--claims-path",
        dispatch_claims_path.as_posix(),
        "--lane-id",
        lane_id,
        "--platform",
        provider,
        "--instance-job-id",
        job_id,
        "--agent",
        agent,
        "--status",
        "planned_exact_eval",
        "--notes",
        f"materializer exact-eval dispatch plan {plan_label}",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def _dispatch_command(
    *,
    queue_path: Path,
    provider: str,
    label_prefix: str,
    estimated_cost_per_dispatch: float,
    max_total_cost: float,
    active_floor_archive_bytes: int | None,
    active_floor_score: float | None,
    allow_above_active_floor_dispatch: bool,
    operator_override_reason: str | None,
    dispatch_claims_path: Path,
    required_claim_job_id: str,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        "tools/parallel_dispatch_top_k.py",
        "--ranked-input",
        queue_path.as_posix(),
        "--top-k",
        "1",
        "--max-concurrency",
        "1",
        "--provider",
        provider,
        "--label-prefix",
        label_prefix,
        "--estimated-cost-per-dispatch",
        f"{estimated_cost_per_dispatch:.8g}",
        "--max-total-cost",
        f"{max_total_cost:.8g}",
        "--dispatch-claims-path",
        dispatch_claims_path.as_posix(),
    ]
    if active_floor_archive_bytes is not None:
        command.extend(["--active-floor-archive-bytes", str(active_floor_archive_bytes)])
    if active_floor_score is not None:
        command.extend(["--active-floor-score", f"{active_floor_score:.12g}"])
    if allow_above_active_floor_dispatch:
        command.append("--allow-above-active-floor-dispatch")
        command.extend(["--operator-override-reason", str(operator_override_reason)])
    if dry_run:
        command.append("--dry-run")
    else:
        command.extend(
            [
                "--claim-policy",
                "require_active_claim",
                "--required-claim-platform",
                provider,
                "--required-claim-instance-job-id",
                required_claim_job_id,
            ]
        )
    return command


def _row_lane_id(row: Mapping[str, Any]) -> str:
    lane_id = row.get("lane_id")
    if isinstance(lane_id, str) and lane_id.strip():
        return lane_id.strip()
    candidate_id = str(row.get("candidate_id") or "materializer_candidate")
    return f"materializer_exact_eval::{_safe_slug(candidate_id)}"


def _row_archive_sha(row: Mapping[str, Any]) -> str | None:
    for key in ("candidate_archive_sha256", "archive_sha256", "expected_archive_sha256"):
        value = _row_sha(row, key)
        if value is not None:
            return value
    return None


def _archive_sha_alias_blockers(row: Mapping[str, Any]) -> list[str]:
    values: dict[str, str] = {}
    blockers: list[str] = []
    for key in ("candidate_archive_sha256", "archive_sha256", "expected_archive_sha256"):
        raw = row.get(key)
        if raw is None:
            continue
        value = _row_sha(row, key)
        if value is None:
            blockers.append(f"archive_sha_alias_invalid:{key}")
            continue
        values[key] = value
    if len(set(values.values())) > 1:
        summary = ":".join(f"{key}={value}" for key, value in sorted(values.items()))
        blockers.append(f"archive_sha_alias_mismatch:{summary}")
    return blockers


def _row_runtime_content_sha(row: Mapping[str, Any]) -> str | None:
    for key in ("runtime_content_tree_sha256", "candidate_runtime_content_tree_sha256"):
        value = _row_sha(row, key)
        if value is not None:
            return value
    runtime_manifest = row.get("runtime_manifest")
    if isinstance(runtime_manifest, Mapping):
        return _row_sha(runtime_manifest, "runtime_content_tree_sha256")
    return None


def _row_runtime_tree_sha(row: Mapping[str, Any]) -> str | None:
    for key in ("runtime_tree_sha256", "candidate_runtime_tree_sha256"):
        value = _row_sha(row, key)
        if value is not None:
            return value
    runtime_manifest = row.get("runtime_manifest")
    if isinstance(runtime_manifest, Mapping):
        return _row_sha(runtime_manifest, "runtime_tree_sha256")
    return None


def _row_score_axis(row: Mapping[str, Any]) -> str:
    for key in (
        "score_axis",
        "target_score_axis",
        "target_auth_axis",
        "auth_eval_axis",
        "contest_axis",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "contest_cuda"


def _row_sha(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if isinstance(value, str):
        text = value.strip().lower()
        if len(text) == 64 and all(ch in "0123456789abcdef" for ch in text):
            return text
    return None


def _dispatch_job_id(
    *,
    label_prefix: str,
    candidate_id: str,
    archive_sha256: str | None,
) -> str:
    parts = [label_prefix, candidate_id]
    if archive_sha256:
        parts.append(archive_sha256[:12])
    return _safe_slug("_".join(parts))[:120]


def _resolve_path(path: str | Path | None, *, repo_root: Path) -> Path:
    if path is None:
        raise ExperimentQueueError("path_missing")
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
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value).strip("._:-")
    return slug or "row"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> None:
    output = Path(path)
    if output.exists() and not overwrite:
        raise ExperimentQueueError(f"refusing_to_overwrite_json:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(output)


__all__ = [
    "DISPATCH_PLAN_SCHEMA",
    "build_materializer_exact_eval_dispatch_plan",
    "write_json",
]
