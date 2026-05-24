# SPDX-License-Identifier: MIT
"""Consume exact-ready artifacts into a paused dry-run experiment queue.

This module is intentionally one step after exact-readiness promotion and one
step before any provider fan-out. It never turns exact-ready custody into score,
promotion, or rank/kill authority; it only materializes paused dry-run queue rows
after the shared exact-dispatch authority gate authorizes the row.
"""

from __future__ import annotations

import hashlib
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
from tac.optimizer.exact_readiness import QUEUE_SCHEMA as EXACT_READY_QUEUE_SCHEMA
from tac.optimizer.exact_ready_audit import audit_exact_ready_queue

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition
from .materializer_chain_harvest import EXACT_READINESS_BRIDGE_SCHEMA

CONSUMER_SCHEMA = "materializer_exact_eval_consumer.v1"
TOOL_NAME = "comma_lab.scheduler.materializer_exact_eval_consumer"
CLAIM_STEP_ID = "claim_lane_dispatch"
DRY_RUN_DISPATCH_STEP_ID = "dispatch_exact_eval_dry_run"
SUPPORTED_PROVIDERS = frozenset({"lightning", "vastai"})

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}


def build_materializer_exact_eval_consumer_queue(
    *,
    repo_root: str | Path,
    bridge_report_paths: Sequence[str | Path] = (),
    exact_ready_queue_paths: Sequence[str | Path] = (),
    experiment_queue_id: str = "materializer_exact_eval_consumer_queue",
    provider: str = "lightning",
    max_concurrency: int = 1,
    estimated_cost_per_dispatch: float = 0.30,
    max_total_cost: float = 5.00,
    label_prefix: str = "materializer_exact_eval_consumer",
    agent: str = "codex",
    dispatch_claims_path: str | Path | None = None,
    active_floor_archive_bytes: int | None = ACTIVE_FLOOR_ARCHIVE_BYTES,
    active_floor_score: float | None = ACTIVE_FLOOR_SCORE,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
) -> dict[str, Any]:
    """Return a non-authoritative consumer report and paused dry-run queue."""

    repo = Path(repo_root)
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
    queue_paths, bridge_rows = collect_exact_ready_queue_paths(
        repo_root=repo,
        bridge_report_paths=bridge_report_paths,
        exact_ready_queue_paths=exact_ready_queue_paths,
    )
    if not queue_paths:
        raise ExperimentQueueError("exact_ready_consumer_requires_input_queue")

    seen_identity: set[str] = set()
    rows: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    authorized_count = 0
    blocked_count = 0
    duplicate_count = 0
    for queue_path in queue_paths:
        try:
            source_row = load_single_dispatch_ready_row(queue_path)
        except ExperimentQueueError as exc:
            rows.append(
                _blocked_consumer_row(
                    queue_path,
                    repo,
                    candidate_id=None,
                    stable_identity=None,
                    blockers=[str(exc)],
                )
            )
            blocked_count += 1
            continue

        candidate_id = str(source_row.get("candidate_id") or "")
        stable_identity, identity_blockers = stable_candidate_identity(source_row)
        if identity_blockers:
            rows.append(
                _blocked_consumer_row(
                    queue_path,
                    repo,
                    candidate_id=candidate_id,
                    stable_identity=stable_identity,
                    blockers=identity_blockers,
                    lane_id=_row_lane_id(source_row),
                    archive_sha256=_row_archive_sha(source_row),
                )
            )
            blocked_count += 1
            continue
        if stable_identity in seen_identity:
            rows.append(
                _blocked_consumer_row(
                    queue_path,
                    repo,
                    candidate_id=candidate_id,
                    stable_identity=stable_identity,
                    blockers=[f"duplicate_stable_identity:{stable_identity}"],
                    lane_id=_row_lane_id(source_row),
                    archive_sha256=_row_archive_sha(source_row),
                )
            )
            blocked_count += 1
            duplicate_count += 1
            continue
        seen_identity.add(stable_identity)

        blockers, facts = exact_dispatch_blockers(
            queue_path=queue_path,
            row=source_row,
            repo_root=repo,
            dispatch_claims_path=claims_path,
            active_floor_archive_bytes=active_floor_archive_bytes,
            active_floor_score=active_floor_score,
            allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
            operator_override_reason=operator_override_reason,
        )
        if blockers:
            rows.append(
                _blocked_consumer_row(
                    queue_path,
                    repo,
                    candidate_id=candidate_id,
                    stable_identity=stable_identity,
                    blockers=blockers,
                    lane_id=_row_lane_id(source_row),
                    archive_sha256=_row_archive_sha(source_row),
                )
            )
            blocked_count += 1
            continue

        lane_id = _row_lane_id(source_row)
        archive_sha = _row_archive_sha(source_row)
        job_id = _dispatch_job_id(
            label_prefix=label_prefix,
            stable_identity=stable_identity,
            candidate_id=candidate_id,
        )
        claim_command = _claim_command(
            lane_id=lane_id,
            provider=provider,
            job_id=job_id,
            agent=agent,
            dispatch_claims_path=claims_path,
            plan_label=label_prefix,
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
        )
        experiments.append(
            _dry_run_experiment(
                experiment_id=_safe_slug(job_id),
                claim_command=claim_command,
                dispatch_command=dispatch_command,
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
                "exact_ready_queue_path": _repo_rel(queue_path, repo),
                "dispatch_job_id": job_id,
                "provider": provider,
                "authorized_for_paused_dry_run_queue": True,
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
    if estimated_total_cost > max_total_cost:
        hard_plan_blockers.append(
            "estimated_total_cost_exceeds_cap:"
            f"{estimated_total_cost:.2f}>{max_total_cost:.2f}"
        )

    queue_experiments = [] if hard_plan_blockers else experiments
    experiment_queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": experiment_queue_id,
            "controls": {
                "mode": "paused",
                "max_concurrency": {"local_cpu": max_concurrency},
            },
            "experiments": queue_experiments
            or [
                {
                    "id": (
                        "frozen_materializer_exact_eval_consumer"
                        if hard_plan_blockers
                        else "no_authorized_materializer_exact_eval_consumer_rows"
                    ),
                    "metadata": {
                        "consumer_schema": CONSUMER_SCHEMA,
                        "reason": hard_plan_blockers or "no_authorized_rows",
                        **FALSE_AUTHORITY,
                    },
                    "steps": [
                        {
                            "id": "noop",
                            "command": [
                                sys.executable,
                                "-c",
                                "print('no authorized exact-eval consumer rows')",
                            ],
                            "resources": {"kind": "local_cpu"},
                        }
                    ],
                }
            ],
        }
    )

    report = apply_proxy_evidence_boundary(
        {
            "schema": CONSUMER_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": _utc_now(),
            "bridge_report_count": len(bridge_report_paths),
            "bridge_ready_row_count": len(bridge_rows),
            "exact_ready_queue_count": len(queue_paths),
            "authorized_candidate_count": authorized_count,
            "blocked_candidate_count": blocked_count,
            "duplicate_candidate_count": duplicate_count,
            "estimated_cost_per_dispatch": estimated_cost_per_dispatch,
            "estimated_total_cost": estimated_total_cost,
            "max_total_cost": max_total_cost,
            "hard_plan_blockers": hard_plan_blockers,
            "dispatch_claims_path": _repo_rel(claims_path, repo),
            "experiment_queue_schema": experiment_queue["schema"],
            "experiment_queue_id": experiment_queue["queue_id"],
            "experiment_count": len(queue_experiments),
            "rows": rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=[
            "consumer_queue_is_paused_dry_run_only",
            "lane_claim_step_must_succeed_before_dispatch_step",
            "contest_auth_eval_result_required_before_score_claim",
            *hard_plan_blockers,
        ],
    )
    require_no_truthy_authority_fields(
        report,
        context="materializer_exact_eval_consumer_report",
    )
    return {"report": report, "experiment_queue": experiment_queue}


def collect_exact_ready_queue_paths(
    *,
    repo_root: str | Path,
    bridge_report_paths: Sequence[str | Path] = (),
    exact_ready_queue_paths: Sequence[str | Path] = (),
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Resolve explicit and bridge-discovered exact-ready queue paths."""

    repo = Path(repo_root)
    paths = [_resolve_path(path, repo_root=repo) for path in exact_ready_queue_paths]
    bridge_rows: list[dict[str, Any]] = []
    for bridge_report_path in bridge_report_paths:
        bridge_path = _resolve_path(bridge_report_path, repo_root=repo)
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
        for row in bridge.get("rows") or []:
            if not isinstance(row, Mapping):
                continue
            if row.get("exact_ready_queue_written") is not True:
                continue
            queue_path = row.get("exact_ready_queue_path")
            if isinstance(queue_path, str) and queue_path.strip():
                paths.append(_resolve_path(queue_path, repo_root=repo))
                bridge_rows.append(dict(row))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve(strict=False)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped, bridge_rows


def load_single_dispatch_ready_row(queue_path: str | Path) -> Mapping[str, Any]:
    """Load the single dispatch-ready row from an exact-ready queue artifact."""

    path = Path(queue_path)
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ExperimentQueueError(f"exact_ready_queue_not_object:{path}")
    if payload.get("schema") != EXACT_READY_QUEUE_SCHEMA:
        raise ExperimentQueueError(
            f"exact_ready_queue_schema_unsupported:{payload.get('schema')!r}"
        )
    rows = payload.get("dispatch_ready")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ExperimentQueueError(
            f"exact_ready_queue_must_have_one_dispatch_ready_row:{path}"
        )
    candidate_id = rows[0].get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ExperimentQueueError(f"dispatch_ready_row_candidate_id_missing:{path}")
    return rows[0]


def stable_candidate_identity(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Return archive/runtime identity and fail-closed blockers."""

    archive_sha = _row_archive_sha(row)
    runtime_content_sha = _row_sha(row, "runtime_content_tree_sha256")
    runtime_tree_sha = _row_sha(row, "runtime_tree_sha256")
    blockers: list[str] = []
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
        f"{archive_sha}:runtime_content={runtime_content_sha}:runtime_tree={runtime_tree_sha}",
        [],
    )


def exact_dispatch_blockers(
    *,
    queue_path: str | Path,
    row: Mapping[str, Any],
    repo_root: str | Path,
    dispatch_claims_path: str | Path,
    active_floor_archive_bytes: int | None,
    active_floor_score: float | None,
    allow_above_active_floor_dispatch: bool,
    operator_override_reason: str | None,
) -> tuple[list[str], dict[str, Any]]:
    """Return blockers from live exact-ready audit and exact-dispatch authority."""

    queue = Path(queue_path)
    repo = Path(repo_root)
    candidate_id = str(row.get("candidate_id") or "")
    audit_repo = _audit_repo_root(queue, repo)
    audit = audit_exact_ready_queue(
        queue,
        repo_root=audit_repo,
        dispatch_claims_path=Path(dispatch_claims_path),
        active_floor_score=active_floor_score,
        candidate_ids=[candidate_id],
    )
    stale_rows = audit.get("stale_ready_rows")
    stale_count = len(stale_rows) if isinstance(stale_rows, list) else 0
    blockers: list[str] = []
    facts: dict[str, Any] = {"audit_stale_ready_row_count": stale_count}
    if stale_count:
        blockers.append(f"exact_ready_queue_audit_stale_rows:{stale_count}")

    authority = exact_dispatch_authority(
        row,
        repo_root=audit_repo,
        queue_dir=queue.parent,
        source=TOOL_NAME,
        active_floor_archive_bytes=active_floor_archive_bytes,
        active_floor_score=active_floor_score,
        allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
        operator_override_reason=operator_override_reason,
        dispatch_claims_path=dispatch_claims_path,
        claim_policy="preclaim_conflict_check",
    )
    facts["authority_source"] = authority.source
    facts["authority"] = authority.as_dict()
    blockers.extend(f"exact_dispatch_authority:{blocker}" for blocker in authority.blockers)
    if authority.authorized is not True:
        blockers.append("exact_dispatch_authority:not_authorized")
    return ordered_unique(blockers), facts


def _blocked_consumer_row(
    queue_path: Path,
    repo_root: Path,
    *,
    candidate_id: str | None,
    stable_identity: str | None,
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
        "authorized_for_paused_dry_run_queue": False,
        "claim_required_before_dispatch": True,
        "blockers": list(blockers),
        **FALSE_AUTHORITY,
    }


def _dry_run_experiment(
    *,
    experiment_id: str,
    claim_command: Sequence[str],
    dispatch_command: Sequence[str],
    candidate_id: str,
    stable_identity: str,
    lane_id: str,
    queue_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "id": experiment_id,
        "metadata": {
            "consumer_schema": CONSUMER_SCHEMA,
            "candidate_id": candidate_id,
            "stable_identity": stable_identity,
            "lane_id": lane_id,
            "exact_ready_queue_path": _repo_rel(queue_path, repo_root),
            **FALSE_AUTHORITY,
        },
        "steps": [
            {
                "id": CLAIM_STEP_ID,
                "command": list(claim_command),
                "resources": {"kind": "local_cpu"},
            },
            {
                "id": DRY_RUN_DISPATCH_STEP_ID,
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
) -> list[str]:
    return [
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
        f"materializer exact-eval consumer {plan_label}",
        "--dry-run",
    ]


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
        "--dry-run",
    ]
    if active_floor_archive_bytes is not None:
        command.extend(["--active-floor-archive-bytes", str(active_floor_archive_bytes)])
    if active_floor_score is not None:
        command.extend(["--active-floor-score", f"{active_floor_score:.12g}"])
    if allow_above_active_floor_dispatch:
        command.append("--allow-above-active-floor-dispatch")
        command.extend(["--operator-override-reason", str(operator_override_reason)])
    return command


def _audit_repo_root(queue_path: Path, repo_root: Path) -> Path:
    queue_root = queue_path.parent.resolve()
    return queue_root if (queue_root / "upstream" / "evaluate.py").is_file() else repo_root


def _row_lane_id(row: Mapping[str, Any]) -> str:
    lane_id = row.get("lane_id")
    if isinstance(lane_id, str) and lane_id.strip():
        return lane_id.strip()
    candidate_id = str(row.get("candidate_id") or "materializer_candidate")
    return f"materializer_exact_eval::{_safe_slug(candidate_id)}"


def _row_archive_sha(row: Mapping[str, Any]) -> str | None:
    for key in ("archive_sha256", "candidate_archive_sha256", "expected_archive_sha256"):
        value = _row_sha(row, key)
        if value is not None:
            return value
    return None


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
    stable_identity: str,
    candidate_id: str,
) -> str:
    archive_match = re.search(r"archive=([0-9a-f]{64})", stable_identity)
    identity_suffix = (
        archive_match.group(1)[:12]
        if archive_match
        else hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()[:12]
    )
    return _safe_slug(f"{label_prefix}_{candidate_id}_{identity_suffix}")[:120]


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
    "CONSUMER_SCHEMA",
    "build_materializer_exact_eval_consumer_queue",
    "collect_exact_ready_queue_paths",
    "exact_dispatch_blockers",
    "load_single_dispatch_ready_row",
    "stable_candidate_identity",
    "write_json",
]
