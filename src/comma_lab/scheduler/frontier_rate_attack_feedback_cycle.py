# SPDX-License-Identifier: MIT
"""Queue-owned frontier feedback cycle helpers.

This module keeps the frontier-rate attack loop out of chat glue:

materializer/local observations -> bounded DQS1 queue -> harvest observations ->
refreshed queue.  All artifacts are planning signal only unless a separate exact
auth path later grants authority.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.cross_family_candidate_portfolio import (
    CrossFamilyCandidatePortfolioError,
    build_cross_family_candidate_portfolio,
    render_cross_family_candidate_portfolio_markdown,
    source_artifacts_from_paths,
)
from tac.optimization.dqs1_local_first_harvest_observations import (
    build_harvest_observation_summary,
    build_observation_rows_from_harvests,
    load_pairset_acquisition_index,
    render_markdown_summary,
    write_observation_jsonl,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.mlx_dynamic_sweep_observations import (
    load_observation_rows,
    observation_duplicate_key,
)
from tac.optimization.pairset_component_marginal import (
    PAIRSET_COMPONENT_MARGINAL_MODEL_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.repo_io import (
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
    write_text_artifact,
)

from .experiment_queue import ExperimentQueueError
from .frontier_rate_attack_feedback import (
    OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA,
    OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA,
    attach_frontier_autonomous_chain_optimization,
    build_frontier_autonomous_chain_optimization_queue,
    build_frontier_materializer_execution_queue_if_available,
    build_frontier_operation_chain_compiler_queue,
    build_frontier_receiver_repair_queue,
    build_frontier_repair_budget_waterfill_queue,
    build_frontier_targeted_component_correction_chain_materializer_handoff,
    build_frontier_targeted_component_correction_chain_work_orders,
    build_frontier_targeted_component_correction_materialization_queue,
    build_frontier_targeted_component_correction_materialization_requests,
    build_frontier_targeted_component_correction_queue,
    build_frontier_targeted_component_correction_response_harvest,
)
from .pair_frame_5d_coverage_acquisition_queue import (
    build_pair_frame_5d_coverage_acquisition_queue,
)
from .pair_frame_5d_extended_operator_queue import (
    build_pair_frame_5d_extended_operator_queue,
)
from .repair_campaign_score_queue import build_repair_campaign_score_queue

FRONTIER_RATE_ATTACK_FEEDBACK_CYCLE_SCHEMA = "frontier_rate_attack_feedback_cycle.v1"
FRONTIER_RATE_ATTACK_DQS1_OBSERVATION_BUNDLE_SCHEMA = (
    "frontier_rate_attack_dqs1_observation_bundle.v1"
)
FRONTIER_RATE_ATTACK_PAIRSET_COMPONENT_MARGINAL_BUNDLE_SCHEMA = (
    "frontier_rate_attack_pairset_component_marginal_bundle.v1"
)
PAIRSET_COMPONENT_MARGINAL_ACTION_SUMMARY_SCHEMA = (
    "pairset_component_marginal_canonicalization_summary.v1"
)
PAIR_FRAME_5D_CANVAS_POPULATED_SCHEMA = (
    "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1"
)
AUTOPILOT_RESULT_SCHEMA = "dqs1_local_first_autopilot_result.v1"
DQS1_DROP_MANY_GREEDY_VERDICT_DISCOVERY_SCHEMA = (
    "frontier_rate_attack_dqs1_drop_many_greedy_verdict_discovery.v1"
)
DQS1_DROP_MANY_GREEDY_VERDICT_SCHEMA = (
    "dqs1_drop_many_build_1c_greedy_independent_heuristic_verdict.v1"
)
DQS1_DROP_MANY_GREEDY_VERDICT_GLOB = (
    "dqs1_drop_many_build_1c_greedy_heuristic*/verdict.json"
)


class FrontierRateAttackFeedbackCycleError(ExperimentQueueError):
    """Raised when a feedback cycle would lose signal or imply false authority."""


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def artifact_token(value: object) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def resolve_repo_path(path: str | Path, *, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path(repo_root) / value
    return value.resolve(strict=False)


def _extend_followup_roots(
    roots: list[str | Path],
    values: object,
) -> None:
    if isinstance(values, str | Path):
        roots.append(values)
    elif isinstance(values, Sequence):
        roots.extend(value for value in values if isinstance(value, str | Path))


def _pair_frame_5d_followup_search_roots(
    report: Mapping[str, Any],
    *,
    output_dir: Path,
    repo_root: str | Path,
) -> list[str | Path]:
    roots: list[str | Path] = [
        output_dir,
        Path(str(report.get("results_root") or "experiments/results")),
    ]
    _extend_followup_roots(roots, report.get("component_response_cache_roots"))
    for key in ("discovery", "pair_frame_geometry_discovery"):
        discovery = report.get(key)
        if isinstance(discovery, Mapping):
            _extend_followup_roots(roots, discovery.get("frontier_artifact_roots"))
    submissions = Path(repo_root) / "submissions"
    if submissions.exists():
        roots.append(submissions)
    return roots


def load_json_object(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontierRateAttackFeedbackCycleError(
            f"{target}: could not load JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise FrontierRateAttackFeedbackCycleError(f"{target}: expected JSON object")
    return payload


def _drop_many_greedy_verdict_candidate_paths(
    root: Path,
    *,
    max_files: int,
) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    if not root.is_dir():
        raise FrontierRateAttackFeedbackCycleError(
            f"drop-many verdict root is not a file or directory: {root}"
        )
    paths: list[Path] = []
    seen: set[str] = set()
    for path in sorted(root.glob(DQS1_DROP_MANY_GREEDY_VERDICT_GLOB)):
        key = path.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
        if len(paths) > max_files:
            raise FrontierRateAttackFeedbackCycleError(
                f"{root}: drop-many verdict discovery exceeded max_files={max_files}"
            )
    return paths


def discover_dqs1_drop_many_greedy_verdict_paths(
    *,
    repo_root: str | Path,
    roots: Sequence[str | Path] = (),
    max_files_per_root: int = 64,
) -> dict[str, Any]:
    """Discover safe DQS1 drop-many greedy verdicts for queue-owned replanning."""

    if max_files_per_root < 1:
        raise FrontierRateAttackFeedbackCycleError("max_files_per_root must be >= 1")
    repo = Path(repo_root)
    default_roots = (repo / ".omx" / "research", repo / "experiments" / "results")
    resolved_roots = [
        resolve_repo_path(root, repo_root=repo) for root in (roots or default_roots)
    ]
    discovered: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in resolved_roots:
        for path in _drop_many_greedy_verdict_candidate_paths(
            root,
            max_files=max_files_per_root,
        ):
            key = path.resolve(strict=False).as_posix()
            if key in seen:
                continue
            seen.add(key)
            try:
                payload = load_json_object(path)
            except FrontierRateAttackFeedbackCycleError as exc:
                refusals.append({"path": repo_rel(path, repo), "reason": str(exc)})
                continue
            if payload.get("schema") != DQS1_DROP_MANY_GREEDY_VERDICT_SCHEMA:
                refusals.append(
                    {
                        "path": repo_rel(path, repo),
                        "reason": "schema_mismatch",
                        "schema": payload.get("schema"),
                    }
                )
                continue
            try:
                require_no_truthy_authority_fields(
                    payload,
                    context=f"drop_many_greedy_verdict_discovery[{path}]",
                )
            except ValueError as exc:
                raise FrontierRateAttackFeedbackCycleError(str(exc)) from exc
            discovered.append(
                {
                    "path": repo_rel(path, repo),
                    "schema": payload.get("schema"),
                    "verdict": payload.get("build_1c_final_verdict"),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
    return {
        "schema": DQS1_DROP_MANY_GREEDY_VERDICT_DISCOVERY_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "active": bool(discovered),
        "roots": [repo_rel(root, repo) for root in resolved_roots],
        "discovered_verdict_count": len(discovered),
        "discovered_verdicts": discovered,
        "discovered_verdict_paths": [row["path"] for row in discovered],
        "refusal_count": len(refusals),
        "refusals": refusals,
        **FALSE_AUTHORITY,
    }


def _candidate_ids_from_harvests(harvest_paths: Sequence[Path]) -> set[str]:
    candidate_ids: set[str] = set()
    for path in harvest_paths:
        payload = load_json_object(path)
        try:
            require_no_truthy_authority_fields(
                payload,
                context=f"frontier_rate_attack_feedback_cycle.harvest[{path}]",
            )
        except ValueError as exc:
            raise FrontierRateAttackFeedbackCycleError(str(exc)) from exc
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if candidate_id:
            candidate_ids.add(candidate_id)
    return candidate_ids


def select_pairset_acquisition_for_harvests(
    *,
    harvest_paths: Sequence[str | Path],
    repo_root: str | Path,
    preferred_pairset_acquisition_path: str | Path | None,
    fallback_pairset_acquisition_path: str | Path,
) -> Path:
    """Prefer the queue-selected acquisition sidecar when it covers harvests."""

    repo = Path(repo_root)
    fallback = resolve_repo_path(fallback_pairset_acquisition_path, repo_root=repo)
    if preferred_pairset_acquisition_path is None:
        return fallback
    resolved_harvests = _unique_paths(
        resolve_repo_path(path, repo_root=repo) for path in harvest_paths
    )
    harvested_candidate_ids = _candidate_ids_from_harvests(resolved_harvests)
    if not harvested_candidate_ids:
        return fallback
    preferred = resolve_repo_path(preferred_pairset_acquisition_path, repo_root=repo)
    try:
        preferred_index = load_pairset_acquisition_index(preferred)
    except Exception:
        return fallback
    if harvested_candidate_ids.issubset(set(preferred_index)):
        return preferred
    return fallback


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return tuple(out)


def _targeted_dqs1_child_queue_paths(queue: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, Mapping):
            continue
        metadata = experiment.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        value = metadata.get("targeted_drop_many_dqs1_followup_queue_path")
        if not isinstance(value, str) or not value.strip() or value in seen:
            continue
        seen.add(value)
        paths.append(value)
    return paths


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _component_model_from_portfolio(portfolio: Mapping[str, Any]) -> dict[str, Any]:
    feedback = _mapping(portfolio.get("observation_feedback"))
    model = feedback.get("pairset_component_marginal_model")
    if isinstance(model, Mapping):
        return dict(model)
    return {
        "schema": PAIRSET_COMPONENT_MARGINAL_MODEL_SCHEMA,
        "active": False,
        "inactive_reason": "missing_from_portfolio",
        **FALSE_AUTHORITY,
    }


def _top_component_marginal_actions(
    portfolio: Mapping[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows = portfolio.get("operator_action_rows")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        if not isinstance(row, Mapping):
            continue
        blockers = [str(blocker) for blocker in row.get("dispatch_blockers") or []]
        out.append(
            {
                "operator_action_rank": row.get("operator_action_rank"),
                "bayesian_rank": row.get("rank"),
                "candidate_id": row.get("candidate_id"),
                "source_kind": row.get("source_kind"),
                "operator_next_action": row.get("operator_next_action"),
                "acquisition_value": row.get("acquisition_value"),
                "predicted_score_mean": row.get("predicted_score_mean"),
                "predicted_score_variance": row.get("predicted_score_variance"),
                "dispatch_blocker_count": len(blockers),
                "dispatch_blockers": blockers[:8],
                **FALSE_AUTHORITY,
            }
        )
    return out


def _render_component_marginal_action_summary_markdown(
    summary: Mapping[str, Any],
) -> str:
    component = _mapping(summary.get("pairset_component_marginal_model"))
    lines = [
        "## Pairset Component Marginal Feedback",
        "",
        f"- Schema: `{summary.get('schema')}`",
        f"- Allowed use: `{summary.get('allowed_use')}`",
        f"- Component model active: `{component.get('active')}`",
        f"- Axes: `{component.get('axes')}`",
        f"- Training rows: `{component.get('training_row_count')}`",
        f"- Score claim: `{summary.get('score_claim')}`",
        f"- Ready for exact eval dispatch: `{summary.get('ready_for_exact_eval_dispatch')}`",
        "",
        "### Queue Candidate Actions",
        "",
        "| action rank | bayes rank | candidate | action | acquisition | blockers |",
        "|---:|---:|---|---|---:|---:|",
    ]
    rows = summary.get("top_operator_actions")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                acquisition = float(row.get("acquisition_value", 0.0))
            except (TypeError, ValueError):
                acquisition = 0.0
            lines.append(
                "| {action_rank} | {bayes_rank} | `{candidate}` | `{action}` | "
                "{acquisition:.12g} | {blockers} |".format(
                    action_rank=row.get("operator_action_rank"),
                    bayes_rank=row.get("bayesian_rank"),
                    candidate=row.get("candidate_id"),
                    action=row.get("operator_next_action"),
                    acquisition=acquisition,
                    blockers=row.get("dispatch_blocker_count"),
                )
            )
    lines.append("")
    return "\n".join(lines)


def harvest_paths_from_autopilot_payload(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> tuple[Path, ...]:
    """Extract append-only DQS1 harvest JSON paths from an autopilot result."""

    if payload.get("schema") != AUTOPILOT_RESULT_SCHEMA:
        raise FrontierRateAttackFeedbackCycleError(
            f"autopilot result schema must be {AUTOPILOT_RESULT_SCHEMA}"
        )
    try:
        require_no_truthy_authority_fields(
            payload,
            context="frontier_rate_attack_feedback_cycle.autopilot_result",
        )
    except ValueError as exc:
        raise FrontierRateAttackFeedbackCycleError(str(exc)) from exc
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        raise FrontierRateAttackFeedbackCycleError("autopilot result rounds must be a list")
    paths: list[Path] = []
    for round_index, round_row in enumerate(rounds):
        if not isinstance(round_row, Mapping):
            continue
        harvests = round_row.get("harvests")
        if harvests is None and isinstance(round_row.get("harvest"), Mapping):
            harvests = [round_row["harvest"]]
        if not isinstance(harvests, list):
            continue
        for harvest_index, harvest_row in enumerate(harvests):
            if not isinstance(harvest_row, Mapping):
                continue
            try:
                require_no_truthy_authority_fields(
                    harvest_row,
                    context=(
                        "frontier_rate_attack_feedback_cycle.autopilot_harvest"
                        f"[{round_index}][{harvest_index}]"
                    ),
                )
            except ValueError as exc:
                raise FrontierRateAttackFeedbackCycleError(str(exc)) from exc
            raw_path = harvest_row.get("harvest_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            path = resolve_repo_path(raw_path, repo_root=repo_root)
            if not path.is_file():
                raise FrontierRateAttackFeedbackCycleError(
                    f"autopilot harvest path not found: {path}"
                )
            paths.append(path)
    return _unique_paths(paths)


def harvest_paths_from_autopilot_result_files(
    paths: Sequence[str | Path],
    *,
    repo_root: str | Path,
) -> tuple[Path, ...]:
    """Load autopilot result files and return unique harvest JSON paths."""

    collected: list[Path] = []
    for value in paths:
        path = resolve_repo_path(value, repo_root=repo_root)
        collected.extend(
            harvest_paths_from_autopilot_payload(
                load_json_object(path),
                repo_root=repo_root,
            )
        )
    return _unique_paths(collected)


def _valid_pair_frame_5d_canvas_path(
    value: str | Path,
    *,
    repo_root: str | Path,
) -> Path | None:
    path = resolve_repo_path(value, repo_root=repo_root)
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema") != PAIR_FRAME_5D_CANVAS_POPULATED_SCHEMA:
        return None
    if not isinstance(payload.get("archive_sha256"), str):
        return None
    if not isinstance(payload.get("cells"), list):
        return None
    return path


def _pair_frame_5d_canvas_paths_from_report(
    report: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> list[Path]:
    candidates: list[str | Path] = []
    raw_paths = report.get("pair_frame_5d_canvas_paths")
    if isinstance(raw_paths, Sequence) and not isinstance(raw_paths, str | bytes):
        candidates.extend(raw_paths)
    raw_path = report.get("pair_frame_5d_canvas_path")
    if isinstance(raw_path, str | Path):
        candidates.append(raw_path)
    seen: set[str] = set()
    valid: list[Path] = []
    for candidate in candidates:
        if not isinstance(candidate, str | Path):
            continue
        path = _valid_pair_frame_5d_canvas_path(candidate, repo_root=repo_root)
        if path is None:
            continue
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        valid.append(path)
    return valid


def _pair_frame_5d_queue_coverage_audit(
    queue: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    controls = queue.get("controls")
    if isinstance(controls, Mapping):
        audit = controls.get("canvas_coverage_audit")
        if isinstance(audit, Mapping):
            return audit
    metadata = queue.get("metadata")
    if isinstance(metadata, Mapping):
        audit = metadata.get("canvas_coverage_audit")
        if isinstance(audit, Mapping):
            return audit
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, Mapping):
            continue
        experiment_metadata = experiment.get("metadata")
        if not isinstance(experiment_metadata, Mapping):
            continue
        audit = experiment_metadata.get("canvas_coverage_audit")
        if isinstance(audit, Mapping):
            return audit
    return None


def write_frontier_refresh_artifacts(
    *,
    output_dir: str | Path,
    report: Mapping[str, Any],
    repo_root: str | Path,
    report_filename: str = "feedback_refresh_report.json",
    queue_filename: str = "dqs1_followup_queue.json",
) -> dict[str, str]:
    """Write the standard frontier feedback refresh artifact family."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    target_profile_metadata = (
        dict(report.get("target_optimization_profile_metadata"))
        if isinstance(report.get("target_optimization_profile_metadata"), Mapping)
        else {}
    )
    if target_profile_metadata:
        require_no_truthy_authority_fields(
            target_profile_metadata,
            context="frontier_refresh_artifacts_target_optimization_profile",
        )
    discovery = report.get("discovery")
    if isinstance(discovery, Mapping):
        path = out / "materializer_feedback_discovery.json"
        write_json_artifact(path, dict(discovery))
        artifacts["materializer_feedback_discovery"] = repo_rel(path, repo_root)
    dqs1_discovery = report.get("dqs1_observation_discovery")
    if isinstance(dqs1_discovery, Mapping):
        path = out / "dqs1_observation_discovery.json"
        write_json_artifact(path, dict(dqs1_discovery))
        artifacts["dqs1_observation_discovery"] = repo_rel(path, repo_root)
    pair_frame = report.get("pair_frame_geometry_discovery")
    if isinstance(pair_frame, Mapping):
        path = out / "pair_frame_geometry_discovery.json"
        write_json_artifact(path, dict(pair_frame))
        artifacts["pair_frame_geometry_discovery"] = repo_rel(path, repo_root)
    pair_frame_5d_canvas_paths = _pair_frame_5d_canvas_paths_from_report(
        report,
        repo_root=repo_root,
    )
    if pair_frame_5d_canvas_paths:
        pair_frame_5d_canvas = pair_frame_5d_canvas_paths[0]
        pair_frame_5d_queue = build_pair_frame_5d_extended_operator_queue(
            repo_root=repo_root,
            canvas_path=pair_frame_5d_canvas,
            output_root=out / "pair_frame_5d_extended_operator_outputs",
            queue_id=(
                f"{report.get('queue_id') or 'frontier_feedback'}_"
                "pair_frame_5d_extended_operators"
            ),
            top_n=int(report.get("candidate_limit") or 4),
        )
        pair_frame_5d_queue_path = out / "pair_frame_5d_extended_operator_queue.json"
        write_json_artifact(pair_frame_5d_queue_path, pair_frame_5d_queue)
        coverage_audit = _pair_frame_5d_queue_coverage_audit(pair_frame_5d_queue)
        if coverage_audit is not None:
            coverage_audit_path = out / "pair_frame_5d_canvas_coverage_audit.json"
            write_json_artifact(coverage_audit_path, dict(coverage_audit))
            artifacts["pair_frame_5d_canvas_coverage_audit"] = repo_rel(
                coverage_audit_path,
                repo_root,
            )
            if int(coverage_audit.get("work_order_count") or 0) > 0:
                coverage_followup_search_roots = _pair_frame_5d_followup_search_roots(
                    report,
                    output_dir=out,
                    repo_root=repo_root,
                )
                coverage_acquisition_queue = (
                    build_pair_frame_5d_coverage_acquisition_queue(
                        repo_root=repo_root,
                        coverage_audit_path=coverage_audit_path,
                        canvas_path=pair_frame_5d_canvas,
                        output_root=out / "pair_frame_5d_coverage_acquisition",
                        queue_id=(
                            f"{report.get('queue_id') or 'frontier_feedback'}_"
                            "pair_frame_5d_coverage_acquisition"
                        ),
                        top_n=int(report.get("candidate_limit") or 4),
                        followup_search_roots=coverage_followup_search_roots,
                    )
                )
                coverage_acquisition_queue_path = (
                    out / "pair_frame_5d_coverage_acquisition_queue.json"
                )
                followup_execution_queue_path = (
                    out
                    / "pair_frame_5d_coverage_acquisition"
                    / "followup_execution_queue.json"
                )
                followup_readiness_report_path = (
                    out
                    / "pair_frame_5d_coverage_acquisition"
                    / "followup_readiness_report.json"
                )
                followup_input_binding_report_path = (
                    out
                    / "pair_frame_5d_coverage_acquisition"
                    / "followup_input_binding_report.json"
                )
                followup_execution_worker_result_path = (
                    out
                    / "pair_frame_5d_coverage_acquisition"
                    / "followup_execution_worker_result.json"
                )
                followup_input_binding_report_ref = repo_rel(
                    followup_input_binding_report_path,
                    repo_root,
                )
                followup_readiness_report_ref = repo_rel(
                    followup_readiness_report_path,
                    repo_root,
                )
                followup_execution_queue_ref = repo_rel(
                    followup_execution_queue_path,
                    repo_root,
                )
                followup_execution_worker_result_ref = repo_rel(
                    followup_execution_worker_result_path,
                    repo_root,
                )
                write_json_artifact(
                    coverage_acquisition_queue_path,
                    dict(coverage_acquisition_queue),
                )
                artifacts["pair_frame_5d_coverage_acquisition_queue"] = repo_rel(
                    coverage_acquisition_queue_path,
                    repo_root,
                )
                report["pair_frame_5d_coverage_acquisition_queue_summary"] = {
                    "schema": (
                        "frontier_rate_attack_pair_frame_5d_coverage_"
                        "acquisition_queue_summary.v1"
                    ),
                    "queue_id": coverage_acquisition_queue.get("queue_id"),
                    "coverage_audit_path": repo_rel(coverage_audit_path, repo_root),
                    "experiment_count": len(
                        coverage_acquisition_queue.get("experiments") or []
                    ),
                    "followup_input_binding_report_path": (
                        followup_input_binding_report_ref
                    ),
                    "followup_readiness_report_path": followup_readiness_report_ref,
                    "followup_execution_queue_path": followup_execution_queue_ref,
                    "followup_execution_worker_result_path": (
                        followup_execution_worker_result_ref
                    ),
                    "followup_search_roots": (
                        coverage_acquisition_queue.get("metadata", {}).get(
                            "followup_search_roots",
                            [],
                        )
                    ),
                    "followup_input_binding_planned_by_queue": True,
                    "followup_readiness_refresh_planned_by_queue": True,
                    "followup_execution_queue_planned_by_queue": True,
                    "followup_execution_bounded_local_run_completed": False,
                    "work_order_count": coverage_audit.get("work_order_count"),
                    "coverage_verdict": coverage_audit.get("verdict"),
                    "allowed_use": (
                        "local_encoder_side_coverage_acquisition_and_followup_"
                        "execution_planning_only"
                    ),
                    **FALSE_AUTHORITY,
                }
        artifacts["pair_frame_5d_canvas"] = repo_rel(pair_frame_5d_canvas, repo_root)
        artifacts["pair_frame_5d_extended_operator_queue"] = repo_rel(
            pair_frame_5d_queue_path,
            repo_root,
        )
        report["pair_frame_5d_canvas_path"] = repo_rel(
            pair_frame_5d_canvas,
            repo_root,
        )
        report["pair_frame_5d_extended_operator_queue_summary"] = {
            "schema": "frontier_rate_attack_pair_frame_5d_extended_operator_queue_summary.v1",
            "queue_id": pair_frame_5d_queue.get("queue_id"),
            "canvas_path": repo_rel(pair_frame_5d_canvas, repo_root),
            "experiment_count": len(pair_frame_5d_queue.get("experiments") or []),
            "operator_count": len(pair_frame_5d_queue.get("experiments") or []),
            "coverage_verdict": (
                coverage_audit.get("verdict") if coverage_audit is not None else None
            ),
            "coverage_work_order_count": (
                coverage_audit.get("work_order_count")
                if coverage_audit is not None
                else None
            ),
            "coverage_acquisition_queue": artifacts.get(
                "pair_frame_5d_coverage_acquisition_queue"
            ),
            "allowed_use": "local_encoder_side_5d_extended_operator_planning_only",
            **FALSE_AUTHORITY,
        }
    eureka_planning = report.get("local_cpu_eureka_planning")
    if isinstance(eureka_planning, Mapping):
        path = out / "local_cpu_eureka_planning.json"
        write_json_artifact(path, dict(eureka_planning))
        artifacts["local_cpu_eureka_planning"] = repo_rel(path, repo_root)
    operation_portfolio = report.get("operation_portfolio")
    if isinstance(operation_portfolio, Mapping):
        path = out / "operation_portfolio.json"
        write_json_artifact(path, dict(operation_portfolio))
        artifacts["operation_portfolio"] = repo_rel(path, repo_root)
    rate_budget_preservation_plan = report.get("rate_budget_preservation_plan")
    if isinstance(rate_budget_preservation_plan, Mapping):
        path = out / "rate_budget_preservation_plan.json"
        write_json_artifact(path, dict(rate_budget_preservation_plan))
        artifacts["rate_budget_preservation_plan"] = repo_rel(path, repo_root)
    operation_materializer_bridge = report.get("operation_materializer_bridge")
    operation_work_queue: Mapping[str, Any] | None = None
    operation_execution_queue: Mapping[str, Any] | None = None
    targeted_chain_work_queue: Mapping[str, Any] | None = None
    targeted_execution_queue: Mapping[str, Any] | None = None
    if isinstance(operation_materializer_bridge, Mapping):
        path = out / "operation_materializer_bridge.json"
        write_json_artifact(path, dict(operation_materializer_bridge))
        artifacts["operation_materializer_bridge"] = repo_rel(path, repo_root)
        chain_work_orders: list[dict[str, Any]] = []
        for index, row in enumerate(operation_materializer_bridge.get("rows") or []):
            if not isinstance(row, Mapping):
                continue
            work_order = row.get("chain_compiler_work_order")
            if not isinstance(work_order, Mapping):
                continue
            if work_order.get("schema") != OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA:
                continue
            chain_work_orders.append(
                {
                    **dict(work_order),
                    "source_bridge_row_index": index,
                    "source_bridge_blockers": list(row.get("blockers") or []),
                }
            )
        if chain_work_orders:
            payload = {
                "schema": OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA,
                "generated_at_utc": report.get("generated_at_utc"),
                "operation_materializer_bridge_schema": (
                    operation_materializer_bridge.get("schema")
                ),
                "work_order_count": len(chain_work_orders),
                "work_orders": chain_work_orders,
                "allowed_use": "queue_owned_operation_chain_compiler_work_orders_only",
                "forbidden_use": (
                    "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
                ),
                **FALSE_AUTHORITY,
            }
            require_no_truthy_authority_fields(
                payload,
                context="operation_chain_compiler_work_orders",
            )
            chain_path = out / "operation_chain_compiler_work_orders.json"
            write_json_artifact(chain_path, payload)
            artifacts["operation_chain_compiler_work_orders"] = repo_rel(
                chain_path,
                repo_root,
            )
            chain_queue = build_frontier_operation_chain_compiler_queue(
                repo_root=repo_root,
                operation_chain_compiler_work_orders=payload,
                operation_chain_compiler_work_orders_path=chain_path,
                results_root=str(report.get("results_root") or "experiments/results"),
                queue_id=f"{report.get('queue_id') or 'frontier_feedback'}_chain_compiler",
                dqs1_observation_source_paths=tuple(
                    report.get("dqs1_observation_source_paths") or ()
                ),
            )
            if isinstance(chain_queue, Mapping):
                chain_queue_path = out / "operation_chain_compiler_queue.json"
                write_json_artifact(chain_queue_path, dict(chain_queue))
                artifacts["operation_chain_compiler_queue"] = repo_rel(
                    chain_queue_path,
                    repo_root,
                )
        bridge_artifacts = (
            ("operation_materializer_backlog", "materializer_backlog"),
            ("operation_materializer_contexts", "materializer_contexts"),
            ("operation_materializer_work_queue", "materializer_work_queue"),
        )
        for artifact_name, bridge_key in bridge_artifacts:
            payload = operation_materializer_bridge.get(bridge_key)
            if isinstance(payload, Mapping):
                artifact_path = out / f"{artifact_name}.json"
                write_json_artifact(artifact_path, dict(payload))
                artifacts[artifact_name] = repo_rel(artifact_path, repo_root)
        operation_work_queue = operation_materializer_bridge.get("materializer_work_queue")
        if isinstance(operation_work_queue, Mapping):
            operation_work_queue_path = out / "operation_materializer_work_queue.json"
            operation_execution_queue = (
                build_frontier_materializer_execution_queue_if_available(
                    repo_root=repo_root,
                    materializer_work_queue=operation_work_queue,
                    materializer_work_queue_path=operation_work_queue_path,
                    results_root=str(report.get("results_root") or "experiments/results"),
                    queue_id=(
                        f"{report.get('queue_id') or 'frontier_feedback'}_"
                        "operation_materializer_execution"
                    ),
                    candidate_limit=int(report.get("candidate_limit") or 4),
                )
            )
            if isinstance(operation_execution_queue, Mapping):
                operation_execution_queue_path = (
                    out / "operation_materializer_execution_queue.json"
                )
                write_json_artifact(
                    operation_execution_queue_path,
                    dict(operation_execution_queue),
                )
                artifacts["operation_materializer_execution_queue"] = repo_rel(
                    operation_execution_queue_path,
                    repo_root,
                )
    receiver_repair_backlog = report.get("receiver_repair_backlog")
    if isinstance(receiver_repair_backlog, Mapping):
        path = out / "receiver_repair_backlog.json"
        write_json_artifact(path, dict(receiver_repair_backlog))
        artifacts["receiver_repair_backlog"] = repo_rel(path, repo_root)
        repair_queue = build_frontier_receiver_repair_queue(
            repo_root=repo_root,
            receiver_repair_backlog=receiver_repair_backlog,
            receiver_repair_backlog_path=path,
            results_root=str(report.get("results_root") or "experiments/results"),
            queue_id=f"{report.get('queue_id') or 'frontier_feedback'}_receiver_repair",
        )
        if isinstance(repair_queue, Mapping):
            queue_path = out / "receiver_repair_queue.json"
            write_json_artifact(queue_path, dict(repair_queue))
            artifacts["receiver_repair_queue"] = repo_rel(queue_path, repo_root)
    receiver_closed_budget = report.get("receiver_closed_correction_budget")
    if isinstance(receiver_closed_budget, Mapping):
        path = out / "receiver_closed_correction_budget.json"
        write_json_artifact(path, dict(receiver_closed_budget))
        artifacts["receiver_closed_correction_budget"] = repo_rel(path, repo_root)
    targeted_component_correction = report.get(
        "targeted_component_correction_acquisition"
    )
    if isinstance(targeted_component_correction, Mapping):
        path = out / "targeted_component_correction_acquisition.json"
        write_json_artifact(path, dict(targeted_component_correction))
        artifacts["targeted_component_correction_acquisition"] = repo_rel(
            path,
            repo_root,
        )
        correction_queue = build_frontier_targeted_component_correction_queue(
            repo_root=repo_root,
            targeted_component_correction_acquisition=targeted_component_correction,
            targeted_component_correction_acquisition_path=path,
            results_root=str(report.get("results_root") or "experiments/results"),
            queue_id=(
                f"{report.get('queue_id') or 'frontier_feedback'}_"
                "component_correction"
            ),
            target_optimization_profile_metadata=target_profile_metadata,
        )
        if isinstance(correction_queue, Mapping):
            queue_path = out / "targeted_component_correction_queue.json"
            write_json_artifact(queue_path, dict(correction_queue))
            artifacts["targeted_component_correction_queue"] = repo_rel(
                queue_path,
                repo_root,
            )
            response_harvest = build_frontier_targeted_component_correction_response_harvest(
                repo_root=repo_root,
                targeted_component_correction_queue=correction_queue,
                results_root=str(report.get("results_root") or "experiments/results"),
            )
            response_harvest_path = (
                out / "targeted_component_correction_response_harvest.json"
            )
            write_json_artifact(response_harvest_path, dict(response_harvest))
            artifacts["targeted_component_correction_response_harvest"] = repo_rel(
                response_harvest_path,
                repo_root,
            )
            report["targeted_component_correction_response_harvest"] = (
                response_harvest
            )
            materialization_requests = (
                build_frontier_targeted_component_correction_materialization_requests(
                    targeted_component_correction_response_harvest=response_harvest,
                    candidate_limit=int(report.get("candidate_limit") or 4),
                )
            )
            materialization_requests_path = (
                out / "targeted_component_correction_materialization_requests.json"
            )
            write_json_artifact(
                materialization_requests_path,
                dict(materialization_requests),
            )
            artifacts["targeted_component_correction_materialization_requests"] = (
                repo_rel(materialization_requests_path, repo_root)
            )
            report["targeted_component_correction_materialization_requests"] = (
                materialization_requests
            )
            materialization_queue = (
                build_frontier_targeted_component_correction_materialization_queue(
                    repo_root=repo_root,
                    targeted_component_correction_response_harvest=response_harvest,
                    targeted_component_correction_response_harvest_path=(
                        response_harvest_path
                    ),
                    results_root=str(report.get("results_root") or "experiments/results"),
                    queue_id=(
                        f"{report.get('queue_id') or 'frontier_feedback'}_"
                        "component_materialization"
                    ),
                    candidate_limit=int(report.get("candidate_limit") or 4),
                )
            )
            if isinstance(materialization_queue, Mapping):
                materialization_queue_path = (
                    out / "targeted_component_correction_materialization_queue.json"
                )
                write_json_artifact(materialization_queue_path, dict(materialization_queue))
                artifacts[
                    "targeted_component_correction_materialization_queue"
                ] = repo_rel(materialization_queue_path, repo_root)
            targeted_chain_work_orders = (
                build_frontier_targeted_component_correction_chain_work_orders(
                    targeted_component_correction_materialization_requests=(
                        materialization_requests
                    ),
                    request_limit=int(report.get("candidate_limit") or 4),
                )
            )
            targeted_chain_work_orders_path = (
                out
                / "targeted_component_correction_operation_chain_work_orders.json"
            )
            write_json_artifact(
                targeted_chain_work_orders_path,
                dict(targeted_chain_work_orders),
            )
            artifacts[
                "targeted_component_correction_operation_chain_work_orders"
            ] = repo_rel(targeted_chain_work_orders_path, repo_root)
            report[
                "targeted_component_correction_operation_chain_work_orders"
            ] = targeted_chain_work_orders
            targeted_chain_queue = build_frontier_operation_chain_compiler_queue(
                repo_root=repo_root,
                operation_chain_compiler_work_orders=targeted_chain_work_orders,
                operation_chain_compiler_work_orders_path=(
                    targeted_chain_work_orders_path
                ),
                results_root=str(report.get("results_root") or "experiments/results"),
                queue_id=(
                    f"{report.get('queue_id') or 'frontier_feedback'}_"
                    "component_operation_chain"
                ),
                candidate_limit=int(report.get("candidate_limit") or 4),
                dqs1_observation_source_paths=tuple(
                    report.get("dqs1_observation_source_paths") or ()
                ),
            )
            if isinstance(targeted_chain_queue, Mapping):
                targeted_chain_queue_path = (
                    out / "targeted_component_correction_operation_chain_queue.json"
                )
                write_json_artifact(targeted_chain_queue_path, dict(targeted_chain_queue))
                artifacts[
                    "targeted_component_correction_operation_chain_queue"
                ] = repo_rel(targeted_chain_queue_path, repo_root)
                report["targeted_drop_many_dqs1_child_queue_paths"] = (
                    _targeted_dqs1_child_queue_paths(targeted_chain_queue)
                )
            targeted_chain_materializer_handoff = (
                build_frontier_targeted_component_correction_chain_materializer_handoff(
                    repo_root=repo_root,
                    targeted_component_correction_chain_work_orders=(
                        targeted_chain_work_orders
                    ),
                    default_output_root=(
                        Path(str(report.get("results_root") or "experiments/results"))
                        / "frontier_targeted_component_correction_chain_materializers"
                    ),
                )
            )
            targeted_chain_materializer_handoff_path = (
                out / "targeted_component_correction_chain_materializer_handoff.json"
            )
            write_json_artifact(
                targeted_chain_materializer_handoff_path,
                dict(targeted_chain_materializer_handoff),
            )
            artifacts[
                "targeted_component_correction_chain_materializer_handoff"
            ] = repo_rel(targeted_chain_materializer_handoff_path, repo_root)
            targeted_chain_work_queue = targeted_chain_materializer_handoff.get(
                "materializer_work_queue"
            )
            if isinstance(targeted_chain_work_queue, Mapping):
                targeted_chain_work_queue_path = (
                    out
                    / "targeted_component_correction_chain_materializer_work_queue.json"
                )
                write_json_artifact(
                    targeted_chain_work_queue_path,
                    dict(targeted_chain_work_queue),
                )
                artifacts[
                    "targeted_component_correction_chain_materializer_work_queue"
                ] = repo_rel(targeted_chain_work_queue_path, repo_root)
                targeted_execution_queue = (
                    build_frontier_materializer_execution_queue_if_available(
                        repo_root=repo_root,
                        materializer_work_queue=targeted_chain_work_queue,
                        materializer_work_queue_path=targeted_chain_work_queue_path,
                        results_root=str(
                            report.get("results_root") or "experiments/results"
                        ),
                        queue_id=(
                            f"{report.get('queue_id') or 'frontier_feedback'}_"
                            "targeted_chain_materializer_execution"
                        ),
                        candidate_limit=int(report.get("candidate_limit") or 4),
                    )
                )
                if isinstance(targeted_execution_queue, Mapping):
                    targeted_execution_queue_path = (
                        out
                        / (
                            "targeted_component_correction_chain_materializer_"
                            "execution_queue.json"
                        )
                    )
                    write_json_artifact(
                        targeted_execution_queue_path,
                        dict(targeted_execution_queue),
                    )
                    artifacts[
                        "targeted_component_correction_chain_materializer_execution_queue"
                    ] = repo_rel(targeted_execution_queue_path, repo_root)
            report["targeted_component_correction_chain_materializer_handoff"] = (
                targeted_chain_materializer_handoff
            )
            attach_frontier_autonomous_chain_optimization(
                report,
                targeted_component_correction_chain_materializer_handoff=(
                    targeted_chain_materializer_handoff
                ),
            )
    autonomous_chain_optimization = report.get("autonomous_chain_optimization")
    if isinstance(autonomous_chain_optimization, Mapping):
        path = out / "autonomous_chain_optimization.json"
        write_json_artifact(path, dict(autonomous_chain_optimization))
        artifacts["autonomous_chain_optimization"] = repo_rel(path, repo_root)
        response_harvest = report.get("targeted_component_correction_response_harvest")
        receiver_closed_budget = report.get("receiver_closed_correction_budget")
        repair_materializer_work_queue = (
            targeted_chain_work_queue
            if isinstance(targeted_chain_work_queue, Mapping)
            else operation_work_queue
            if isinstance(operation_work_queue, Mapping)
            else None
        )
        repair_materializer_execution_queue = (
            targeted_execution_queue
            if isinstance(targeted_execution_queue, Mapping)
            else operation_execution_queue
            if isinstance(operation_execution_queue, Mapping)
            else None
        )
        repair_materializer_work_queue_path = (
            artifacts.get("targeted_component_correction_chain_materializer_work_queue")
            or artifacts.get("operation_materializer_work_queue")
        )
        repair_materializer_execution_queue_path = (
            artifacts.get(
                "targeted_component_correction_chain_materializer_execution_queue"
            )
            or artifacts.get("operation_materializer_execution_queue")
        )
        repair_waterfill_queue = build_frontier_repair_budget_waterfill_queue(
            repo_root=repo_root,
            autonomous_chain_optimization=autonomous_chain_optimization,
            autonomous_chain_optimization_path=path,
            targeted_component_correction_response_harvest=(
                response_harvest if isinstance(response_harvest, Mapping) else None
            ),
            targeted_component_correction_response_harvest_path=artifacts.get(
                "targeted_component_correction_response_harvest"
            ),
            receiver_closed_correction_budget=(
                receiver_closed_budget
                if isinstance(receiver_closed_budget, Mapping)
                else None
            ),
            receiver_closed_correction_budget_path=artifacts.get(
                "receiver_closed_correction_budget"
            ),
            materializer_work_queue=repair_materializer_work_queue,
            materializer_work_queue_path=repair_materializer_work_queue_path,
            materializer_execution_queue=repair_materializer_execution_queue,
            materializer_execution_queue_path=repair_materializer_execution_queue_path,
            results_root=str(report.get("results_root") or "experiments/results"),
            queue_id=(
                f"{report.get('queue_id') or 'frontier_feedback'}_"
                "repair_budget_waterfill"
            ),
            chain_limit=int(report.get("candidate_limit") or 4),
            target_optimization_profile_metadata=target_profile_metadata,
        )
        if isinstance(repair_waterfill_queue, Mapping):
            repair_waterfill_queue_path = out / "repair_budget_waterfill_queue.json"
            write_json_artifact(
                repair_waterfill_queue_path,
                dict(repair_waterfill_queue),
            )
            artifacts["repair_budget_waterfill_queue"] = repo_rel(
                repair_waterfill_queue_path,
                repo_root,
            )
            repair_campaign_score_queue = build_repair_campaign_score_queue(
                repo_root=repo_root,
                repair_budget_waterfill_queue=repair_waterfill_queue,
                repair_budget_waterfill_queue_path=repair_waterfill_queue_path,
                results_root=str(report.get("results_root") or "experiments/results"),
                queue_id=(
                    f"{report.get('queue_id') or 'frontier_feedback'}_"
                    "repair_campaign_score"
                ),
                experiment_limit=int(report.get("candidate_limit") or 4),
            )
            repair_campaign_score_queue_path = out / "repair_campaign_score_queue.json"
            write_json_artifact(
                repair_campaign_score_queue_path,
                dict(repair_campaign_score_queue),
            )
            artifacts["repair_campaign_score_queue"] = repo_rel(
                repair_campaign_score_queue_path,
                repo_root,
            )
            report["repair_campaign_score_queue_summary"] = {
                "schema": "frontier_rate_attack_repair_campaign_score_queue_summary.v1",
                "queue_id": repair_campaign_score_queue.get("queue_id"),
                "experiment_count": len(
                    repair_campaign_score_queue.get("experiments") or []
                ),
                "ready_experiment_count": repair_campaign_score_queue.get(
                    "metadata",
                    {},
                ).get("ready_experiment_count"),
                "blocked_experiment_count": repair_campaign_score_queue.get(
                    "metadata",
                    {},
                ).get("blocked_experiment_count"),
                "queue_path": artifacts["repair_campaign_score_queue"],
                "allowed_use": (
                    "default_repair_campaign_scorer_queue_planning_only"
                ),
                **FALSE_AUTHORITY,
            }
        autonomous_queue = build_frontier_autonomous_chain_optimization_queue(
            repo_root=repo_root,
            autonomous_chain_optimization=autonomous_chain_optimization,
            autonomous_chain_optimization_path=path,
            artifact_paths_by_key=artifacts,
            results_root=str(report.get("results_root") or "experiments/results"),
            queue_id=(
                f"{report.get('queue_id') or 'frontier_feedback'}_"
                "autonomous_chain_optimization"
            ),
            chain_limit=int(report.get("candidate_limit") or 4),
        )
        if isinstance(autonomous_queue, Mapping):
            queue_path = out / "autonomous_chain_optimization_queue.json"
            write_json_artifact(queue_path, dict(autonomous_queue))
            artifacts["autonomous_chain_optimization_queue"] = repo_rel(
                queue_path,
                repo_root,
            )
    selected_acquisition = report.get("selected_pairset_acquisition")
    if isinstance(selected_acquisition, Mapping):
        path = out / "dqs1_selected_pairset_acquisition.json"
        write_json_artifact(path, dict(selected_acquisition))
        artifacts["dqs1_selected_pairset_acquisition"] = repo_rel(path, repo_root)
    bridge = report.get("materializer_feedback_bridge")
    if isinstance(bridge, Mapping):
        path = out / "materializer_feedback_bridge.json"
        write_json_artifact(path, dict(bridge))
        artifacts["materializer_feedback_bridge"] = repo_rel(path, repo_root)
    queue = report.get("queue")
    if isinstance(queue, Mapping):
        path = out / queue_filename
        write_json_artifact(path, dict(queue))
        artifacts["dqs1_followup_queue"] = repo_rel(path, repo_root)

    report_path = out / report_filename
    report_to_write = dict(report)
    report_to_write["artifacts"] = dict(artifacts)
    operator_commands: dict[str, Any] = {}
    if "dqs1_followup_queue" in artifacts:
        operator_commands.update(
            {
                "validate_followup_queue": [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    artifacts["dqs1_followup_queue"],
                    "validate",
                ],
                "run_frontier_feedback_cycle": [
                    ".venv/bin/python",
                    "tools/run_frontier_rate_attack_feedback_cycle.py",
                    "--frontier-artifact-root",
                    ".omx/research",
                    "--action-summary",
                    str(report.get("action_summary_path") or "latest"),
                    "--results-root",
                    str(report.get("results_root") or ""),
                ],
            }
        )
    if "receiver_repair_queue" in artifacts:
        operator_commands["validate_receiver_repair_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["receiver_repair_queue"],
            "validate",
        ]
        operator_commands["init_receiver_repair_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["receiver_repair_queue"],
            "init",
        ]
        operator_commands["status_receiver_repair_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["receiver_repair_queue"],
            "status",
        ]
        operator_commands["run_receiver_repair_queue_bounded_local"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["receiver_repair_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "12",
            "--max-experiments",
            "4",
            "--max-parallel",
            "2",
        ]
    if "operation_chain_compiler_queue" in artifacts:
        operator_commands["validate_operation_chain_compiler_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["operation_chain_compiler_queue"],
            "validate",
        ]
        operator_commands["init_operation_chain_compiler_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["operation_chain_compiler_queue"],
            "init",
        ]
        operator_commands["status_operation_chain_compiler_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["operation_chain_compiler_queue"],
            "status",
        ]
        operator_commands["run_operation_chain_compiler_queue_bounded_local"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["operation_chain_compiler_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "16",
            "--max-experiments",
            "2",
            "--max-parallel",
            "2",
        ]
    if "operation_materializer_execution_queue" in artifacts:
        operator_commands["validate_operation_materializer_execution_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["operation_materializer_execution_queue"],
            "validate",
        ]
        operator_commands["run_operation_materializer_execution_queue_bounded_local"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["operation_materializer_execution_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "8",
            "--max-experiments",
            "2",
            "--max-parallel",
            "1",
        ]
    if "targeted_component_correction_queue" in artifacts:
        operator_commands["validate_targeted_component_correction_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_queue"],
            "validate",
        ]
        operator_commands["init_targeted_component_correction_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_queue"],
            "init",
        ]
        operator_commands["status_targeted_component_correction_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_queue"],
            "status",
        ]
        operator_commands[
            "run_targeted_component_correction_queue_bounded_local"
        ] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "21",
            "--max-experiments",
            "2",
            "--max-parallel",
            "3",
        ]
    if "targeted_component_correction_materialization_queue" in artifacts:
        operator_commands[
            "validate_targeted_component_correction_materialization_queue"
        ] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_materialization_queue"],
            "validate",
        ]
        operator_commands[
            "run_targeted_component_correction_materialization_queue_bounded_local"
        ] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_materialization_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "16",
            "--max-experiments",
            "4",
            "--max-parallel",
            "2",
        ]
    if "targeted_component_correction_operation_chain_queue" in artifacts:
        operator_commands[
            "validate_targeted_component_correction_operation_chain_queue"
        ] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_operation_chain_queue"],
            "validate",
        ]
        operator_commands[
            "run_targeted_component_correction_operation_chain_queue_bounded_local"
        ] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_operation_chain_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "24",
            "--max-experiments",
            "4",
            "--max-parallel",
            "2",
        ]
        child_queue_paths = [
            str(path)
            for path in report_to_write.get("targeted_drop_many_dqs1_child_queue_paths")
            or []
        ]
        if child_queue_paths:
            first_child_queue = child_queue_paths[0]
            operator_commands[
                "validate_targeted_drop_many_dqs1_child_queue_after_chain_run"
            ] = [
                ".venv/bin/python",
                "tools/experiment_queue.py",
                "--queue",
                first_child_queue,
                "validate",
            ]
            operator_commands[
                "run_targeted_drop_many_dqs1_child_queue_bounded_local_after_chain_run"
            ] = [
                ".venv/bin/python",
                "tools/experiment_queue.py",
                "--queue",
                first_child_queue,
                "run-worker",
                "--execute",
                "--max-steps",
                "8",
                "--max-experiments",
                "2",
                "--max-parallel",
                "2",
            ]
    if "targeted_component_correction_chain_materializer_handoff" in artifacts:
        operator_commands[
            "inspect_targeted_component_correction_chain_materializer_handoff"
        ] = [
            ".venv/bin/python",
            "-m",
            "json.tool",
            artifacts["targeted_component_correction_chain_materializer_handoff"],
        ]
    if "targeted_component_correction_chain_materializer_execution_queue" in artifacts:
        operator_commands[
            "validate_targeted_component_correction_chain_materializer_execution_queue"
        ] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts[
                "targeted_component_correction_chain_materializer_execution_queue"
            ],
            "validate",
        ]
        operator_commands[
            "run_targeted_component_correction_chain_materializer_execution_queue_bounded_local"
        ] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts[
                "targeted_component_correction_chain_materializer_execution_queue"
            ],
            "run-worker",
            "--execute",
            "--max-steps",
            "8",
            "--max-experiments",
            "2",
            "--max-parallel",
            "1",
        ]
    if "autonomous_chain_optimization" in artifacts:
        operator_commands["inspect_autonomous_chain_optimization"] = [
            ".venv/bin/python",
            "-m",
            "json.tool",
            artifacts["autonomous_chain_optimization"],
        ]
    if "repair_budget_waterfill_queue" in artifacts:
        operator_commands["validate_repair_budget_waterfill_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["repair_budget_waterfill_queue"],
            "validate",
        ]
        operator_commands["init_repair_budget_waterfill_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["repair_budget_waterfill_queue"],
            "init",
        ]
        operator_commands["run_repair_budget_waterfill_queue_bounded_local"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["repair_budget_waterfill_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "8",
            "--max-experiments",
            "2",
            "--max-parallel",
            "1",
        ]
    if "repair_campaign_score_queue" in artifacts:
        operator_commands["validate_repair_campaign_score_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["repair_campaign_score_queue"],
            "validate",
        ]
        operator_commands["init_repair_campaign_score_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["repair_campaign_score_queue"],
            "init",
        ]
        operator_commands["run_repair_campaign_score_queue_bounded_local"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["repair_campaign_score_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "8",
            "--max-experiments",
            "2",
            "--max-parallel",
            "1",
        ]
    if "pair_frame_5d_extended_operator_queue" in artifacts:
        operator_commands["validate_pair_frame_5d_extended_operator_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["pair_frame_5d_extended_operator_queue"],
            "validate",
        ]
        operator_commands["init_pair_frame_5d_extended_operator_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["pair_frame_5d_extended_operator_queue"],
            "init",
        ]
        operator_commands[
            "run_pair_frame_5d_extended_operator_queue_bounded_local"
        ] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["pair_frame_5d_extended_operator_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "8",
            "--max-experiments",
            "8",
            "--max-parallel",
            "1",
        ]
    if "pair_frame_5d_coverage_acquisition_queue" in artifacts:
        operator_commands["validate_pair_frame_5d_coverage_acquisition_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["pair_frame_5d_coverage_acquisition_queue"],
            "validate",
        ]
        operator_commands["init_pair_frame_5d_coverage_acquisition_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["pair_frame_5d_coverage_acquisition_queue"],
            "init",
        ]
        operator_commands[
            "run_pair_frame_5d_coverage_acquisition_queue_bounded_local"
        ] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["pair_frame_5d_coverage_acquisition_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "16",
            "--max-experiments",
            "8",
            "--max-parallel",
            "1",
        ]
        coverage_summary = report_to_write.get(
            "pair_frame_5d_coverage_acquisition_queue_summary"
        )
        followup_execution_queue = (
            coverage_summary.get("followup_execution_queue_path")
            if isinstance(coverage_summary, Mapping)
            else None
        )
        followup_worker_result = (
            coverage_summary.get("followup_execution_worker_result_path")
            if isinstance(coverage_summary, Mapping)
            else None
        )
        followup_input_binding_report = (
            coverage_summary.get("followup_input_binding_report_path")
            if isinstance(coverage_summary, Mapping)
            else None
        )
        followup_readiness_report = (
            coverage_summary.get("followup_readiness_report_path")
            if isinstance(coverage_summary, Mapping)
            else None
        )
        if (
            isinstance(followup_input_binding_report, str)
            and followup_input_binding_report
        ):
            operator_commands[
                "inspect_pair_frame_5d_followup_input_binding_report_after_acquisition"
            ] = [
                ".venv/bin/python",
                "-m",
                "json.tool",
                followup_input_binding_report,
            ]
        if isinstance(followup_readiness_report, str) and followup_readiness_report:
            operator_commands[
                "inspect_pair_frame_5d_followup_readiness_report_after_acquisition"
            ] = [
                ".venv/bin/python",
                "-m",
                "json.tool",
                followup_readiness_report,
            ]
        if isinstance(followup_execution_queue, str) and followup_execution_queue:
            operator_commands[
                "validate_pair_frame_5d_followup_execution_queue_after_acquisition"
            ] = [
                ".venv/bin/python",
                "tools/experiment_queue.py",
                "--queue",
                followup_execution_queue,
                "validate",
            ]
            operator_commands[
                "run_pair_frame_5d_followup_execution_queue_bounded_local_after_acquisition"
            ] = [
                ".venv/bin/python",
                "tools/experiment_queue.py",
                "--queue",
                followup_execution_queue,
                "run-worker",
                "--execute",
                "--max-steps",
                "4",
                "--max-experiments",
                "2",
                "--max-parallel",
                "1",
            ] + (
                ["--output", followup_worker_result]
                if isinstance(followup_worker_result, str) and followup_worker_result
                else []
            )
    if "autonomous_chain_optimization_queue" in artifacts:
        operator_commands["validate_autonomous_chain_optimization_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["autonomous_chain_optimization_queue"],
            "validate",
        ]
        operator_commands["init_autonomous_chain_optimization_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["autonomous_chain_optimization_queue"],
            "init",
        ]
        operator_commands["run_autonomous_chain_optimization_queue_bounded_local"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["autonomous_chain_optimization_queue"],
            "run-worker",
            "--execute",
            "--max-steps",
            "16",
            "--max-experiments",
            "2",
            "--max-parallel",
            "1",
        ]
    if operator_commands:
        report_to_write["operator_commands"] = operator_commands
    write_json_artifact(report_path, report_to_write)
    artifacts["feedback_refresh_report"] = repo_rel(report_path, repo_root)
    return artifacts


def write_targeted_component_correction_post_auxiliary_artifacts(
    *,
    output_dir: str | Path,
    targeted_component_correction_queue: Mapping[str, Any],
    targeted_component_correction_queue_path: str | Path,
    repo_root: str | Path,
    results_root: str | Path,
    queue_id: str,
    candidate_limit: int,
    dqs1_observation_source_paths: Sequence[str | Path] = (),
    artifact_prefix: str = "post_auxiliary",
    target_optimization_profile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-harvest targeted correction responses after bounded local queue work."""

    if candidate_limit < 1:
        raise FrontierRateAttackFeedbackCycleError("candidate_limit must be >= 1")
    if not artifact_prefix:
        raise FrontierRateAttackFeedbackCycleError("artifact_prefix must be non-empty")
    require_no_truthy_authority_fields(
        targeted_component_correction_queue,
        context="post_auxiliary_targeted_component_correction_queue",
    )
    target_profile_metadata = (
        dict(target_optimization_profile_metadata)
        if isinstance(target_optimization_profile_metadata, Mapping)
        and target_optimization_profile_metadata
        else {}
    )
    if not target_profile_metadata:
        queue_metadata = targeted_component_correction_queue.get("metadata")
        if isinstance(queue_metadata, Mapping):
            nested = queue_metadata.get("frontier_target_optimization_profile")
            if isinstance(nested, Mapping) and nested:
                target_profile_metadata = dict(nested)
    if target_profile_metadata:
        require_no_truthy_authority_fields(
            target_profile_metadata,
            context=(
                "post_auxiliary_targeted_component_refresh_"
                "target_optimization_profile"
            ),
        )
    repo = Path(repo_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {
        "targeted_component_correction_queue": repo_rel(
            targeted_component_correction_queue_path,
            repo,
        )
    }
    response_harvest = build_frontier_targeted_component_correction_response_harvest(
        repo_root=repo,
        targeted_component_correction_queue=targeted_component_correction_queue,
        results_root=results_root,
    )
    response_harvest_path = (
        out / f"{artifact_prefix}_targeted_component_correction_response_harvest.json"
    )
    write_json_artifact(response_harvest_path, dict(response_harvest))
    artifacts["targeted_component_correction_response_harvest"] = repo_rel(
        response_harvest_path,
        repo,
    )

    materialization_requests = (
        build_frontier_targeted_component_correction_materialization_requests(
            targeted_component_correction_response_harvest=response_harvest,
            candidate_limit=candidate_limit,
        )
    )
    materialization_requests_path = (
        out
        / f"{artifact_prefix}_targeted_component_correction_materialization_requests.json"
    )
    write_json_artifact(materialization_requests_path, dict(materialization_requests))
    artifacts["targeted_component_correction_materialization_requests"] = repo_rel(
        materialization_requests_path,
        repo,
    )

    materialization_queue = (
        build_frontier_targeted_component_correction_materialization_queue(
            repo_root=repo,
            targeted_component_correction_response_harvest=response_harvest,
            targeted_component_correction_response_harvest_path=response_harvest_path,
            results_root=results_root,
            queue_id=f"{queue_id}_component_materialization",
            candidate_limit=candidate_limit,
        )
    )
    if isinstance(materialization_queue, Mapping):
        materialization_queue_path = (
            out
            / f"{artifact_prefix}_targeted_component_correction_materialization_queue.json"
        )
        write_json_artifact(materialization_queue_path, dict(materialization_queue))
        artifacts["targeted_component_correction_materialization_queue"] = repo_rel(
            materialization_queue_path,
            repo,
        )

    chain_work_orders = build_frontier_targeted_component_correction_chain_work_orders(
        targeted_component_correction_materialization_requests=materialization_requests,
        request_limit=candidate_limit,
    )
    chain_work_orders_path = (
        out
        / f"{artifact_prefix}_targeted_component_correction_operation_chain_work_orders.json"
    )
    write_json_artifact(chain_work_orders_path, dict(chain_work_orders))
    artifacts["targeted_component_correction_operation_chain_work_orders"] = repo_rel(
        chain_work_orders_path,
        repo,
    )

    chain_queue = build_frontier_operation_chain_compiler_queue(
        repo_root=repo,
        operation_chain_compiler_work_orders=chain_work_orders,
        operation_chain_compiler_work_orders_path=chain_work_orders_path,
        results_root=results_root,
        queue_id=f"{queue_id}_component_operation_chain",
        candidate_limit=candidate_limit,
        dqs1_observation_source_paths=dqs1_observation_source_paths,
    )
    targeted_child_queue_paths: list[str] = []
    if isinstance(chain_queue, Mapping):
        chain_queue_path = (
            out
            / f"{artifact_prefix}_targeted_component_correction_operation_chain_queue.json"
        )
        write_json_artifact(chain_queue_path, dict(chain_queue))
        artifacts["targeted_component_correction_operation_chain_queue"] = repo_rel(
            chain_queue_path,
            repo,
        )
        targeted_child_queue_paths = _targeted_dqs1_child_queue_paths(chain_queue)

    chain_materializer_handoff = (
        build_frontier_targeted_component_correction_chain_materializer_handoff(
            repo_root=repo,
            targeted_component_correction_chain_work_orders=chain_work_orders,
            default_output_root=(
                Path(str(results_root))
                / "frontier_targeted_component_correction_chain_materializers"
            ),
        )
    )
    chain_materializer_handoff_path = (
        out
        / f"{artifact_prefix}_targeted_component_correction_chain_materializer_handoff.json"
    )
    write_json_artifact(
        chain_materializer_handoff_path,
        dict(chain_materializer_handoff),
    )
    artifacts["targeted_component_correction_chain_materializer_handoff"] = repo_rel(
        chain_materializer_handoff_path,
        repo,
    )

    summary_path = out / f"{artifact_prefix}_targeted_component_correction_refresh.json"
    artifacts["summary"] = repo_rel(summary_path, repo)
    summary = {
        "schema": "frontier_rate_attack_post_auxiliary_targeted_component_refresh.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_prefix": artifact_prefix,
        "targeted_component_correction_queue_path": repo_rel(
            targeted_component_correction_queue_path,
            repo,
        ),
        "frontier_target_optimization_profile": dict(target_profile_metadata),
        "response_harvest_row_count": response_harvest.get("row_count"),
        "response_harvest_local_acquisition_recommended_count": (
            response_harvest.get("local_acquisition_recommended_count")
        ),
        "materialization_request_row_count": materialization_requests.get("row_count"),
        "operation_chain_work_order_count": chain_work_orders.get("work_order_count"),
        "chain_materializer_handoff_work_queue_row_count": (
            chain_materializer_handoff.get("work_queue_row_count")
        ),
        "targeted_drop_many_dqs1_child_queue_paths": targeted_child_queue_paths,
        "artifacts": artifacts,
        "allowed_use": "post_auxiliary_targeted_correction_reharvest_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        summary,
        context="post_auxiliary_targeted_component_refresh",
    )
    write_json_artifact(summary_path, summary)
    return summary


def write_dqs1_harvest_observation_bundle(
    *,
    harvest_paths: Sequence[str | Path],
    repo_root: str | Path,
    pairset_acquisition_path: str | Path,
    baseline_advisory_path: str | Path,
    baseline_archive_size_bytes: int,
    baseline_candidate_id: str,
    output_dir: str | Path,
    stamp: str | None = None,
) -> dict[str, Any]:
    """Canonicalize harvested DQS1 local results into dynamic observation rows."""

    repo = Path(repo_root)
    resolved_harvests = _unique_paths(
        resolve_repo_path(path, repo_root=repo) for path in harvest_paths
    )
    if not resolved_harvests:
        raise FrontierRateAttackFeedbackCycleError(
            "at least one DQS1 harvest path is required"
        )
    missing = [path.as_posix() for path in resolved_harvests if not path.is_file()]
    if missing:
        raise FrontierRateAttackFeedbackCycleError(
            "missing DQS1 harvest JSON: " + ", ".join(missing)
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_stamp = stamp or utc_stamp()
    jsonl_path = out / f"dqs1_local_first_harvest_observations_{run_stamp}.jsonl"
    summary_path = jsonl_path.with_suffix(".summary.json")
    markdown_path = jsonl_path.with_suffix(".md")
    rows = build_observation_rows_from_harvests(
        resolved_harvests,
        repo_root=repo,
        pairset_acquisition_path=resolve_repo_path(
            pairset_acquisition_path,
            repo_root=repo,
        ),
        baseline_advisory_path=resolve_repo_path(
            baseline_advisory_path,
            repo_root=repo,
        ),
        baseline_archive_size_bytes=baseline_archive_size_bytes,
        baseline_candidate_id=baseline_candidate_id,
    )
    write_observation_jsonl(rows, output_path=jsonl_path)
    summary = build_harvest_observation_summary(
        rows,
        jsonl_path=jsonl_path,
        repo_root=repo,
    )
    write_json_artifact(summary_path, summary)
    write_text_artifact(markdown_path, render_markdown_summary(summary))
    return {
        "schema": FRONTIER_RATE_ATTACK_DQS1_OBSERVATION_BUNDLE_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harvest_paths": [repo_rel(path, repo) for path in resolved_harvests],
        "pairset_acquisition_path": repo_rel(
            resolve_repo_path(pairset_acquisition_path, repo_root=repo),
            repo,
        ),
        "observation_jsonl": repo_rel(jsonl_path, repo),
        "observation_summary_json": repo_rel(summary_path, repo),
        "observation_markdown": repo_rel(markdown_path, repo),
        "row_count": len(rows),
        "best_local_advisory": summary.get("best_local_advisory"),
        **FALSE_AUTHORITY,
    }


def write_pairset_component_marginal_feedback_bundle(
    *,
    repo_root: str | Path,
    pairset_acquisition_paths: Sequence[str | Path],
    observation_paths: Sequence[str | Path],
    incumbent_score: float,
    incumbent_scores_by_axis: Mapping[str, Any] | None,
    output_dir: str | Path,
    drop_many_greedy_verdict_paths: Sequence[str | Path] = (),
    auto_discover_drop_many_greedy_verdicts: bool = True,
    stamp: str | None = None,
    top_k: int = 64,
    top_actions: int = 16,
) -> dict[str, Any]:
    """Canonicalize DQS1 observations into the next queue-owned action summary."""

    repo = Path(repo_root)
    if not pairset_acquisition_paths:
        raise FrontierRateAttackFeedbackCycleError(
            "pairset component-marginal feedback requires a pairset acquisition"
        )
    if not observation_paths:
        raise FrontierRateAttackFeedbackCycleError(
            "pairset component-marginal feedback requires observation JSONL paths"
        )
    if top_k < 1:
        raise FrontierRateAttackFeedbackCycleError("top_k must be >= 1")
    if top_actions < 1:
        raise FrontierRateAttackFeedbackCycleError("top_actions must be >= 1")

    acquisition_paths = _unique_paths(
        resolve_repo_path(path, repo_root=repo) for path in pairset_acquisition_paths
    )
    resolved_observation_paths = _unique_paths(
        resolve_repo_path(path, repo_root=repo) for path in observation_paths
    )
    if auto_discover_drop_many_greedy_verdicts:
        discovery = discover_dqs1_drop_many_greedy_verdict_paths(repo_root=repo)
    else:
        discovery = {
            "schema": DQS1_DROP_MANY_GREEDY_VERDICT_DISCOVERY_SCHEMA,
            "active": False,
            "discovered_verdict_paths": [],
            "discovered_verdict_count": 0,
            "refusals": [],
            **FALSE_AUTHORITY,
        }
    discovered_verdict_paths = [
        resolve_repo_path(path, repo_root=repo)
        for path in discovery.get("discovered_verdict_paths") or []
    ]
    resolved_drop_many_verdict_paths = _unique_paths(
        [
            *(
                resolve_repo_path(path, repo_root=repo)
                for path in drop_many_greedy_verdict_paths
            ),
            *discovered_verdict_paths,
        ]
    )
    missing = [
        path.as_posix()
        for path in (
            *acquisition_paths,
            *resolved_observation_paths,
            *resolved_drop_many_verdict_paths,
        )
        if not path.is_file()
    ]
    if missing:
        raise FrontierRateAttackFeedbackCycleError(
            "missing component-marginal source artifact: " + ", ".join(missing)
        )

    observations: list[dict[str, Any]] = []
    try:
        seen_observations: set[tuple[tuple[str, str | None], ...]] = set()
        for path in resolved_observation_paths:
            for row in load_observation_rows(path):
                key = observation_duplicate_key(row)
                if key in seen_observations:
                    continue
                seen_observations.add(key)
                observations.append(row)
        pairset_acquisitions = [load_json_object(path) for path in acquisition_paths]
        drop_many_greedy_verdicts = [
            load_json_object(path) for path in resolved_drop_many_verdict_paths
        ]
        source_artifacts = source_artifacts_from_paths(
            {
                "pairset_acquisitions": acquisition_paths,
                "observation_jsonl": resolved_observation_paths,
                "dqs1_drop_many_greedy_verdicts": resolved_drop_many_verdict_paths,
            },
            repo_root=repo,
        )
        portfolio = build_cross_family_candidate_portfolio(
            incumbent_score=incumbent_score,
            pairset_acquisitions=pairset_acquisitions,
            observations=observations,
            drop_many_greedy_verdicts=drop_many_greedy_verdicts,
            incumbent_scores_by_axis=incumbent_scores_by_axis or {},
            source_artifacts=source_artifacts,
            source_artifact_paths={
                "pairset_acquisitions": [
                    repo_rel(path, repo) for path in acquisition_paths
                ],
                "dqs1_drop_many_greedy_verdicts": [
                    repo_rel(path, repo) for path in resolved_drop_many_verdict_paths
                ],
            },
            top_k=top_k,
        )
        require_no_truthy_authority_fields(
            portfolio,
            context="frontier_rate_attack_feedback_cycle.component_marginal_portfolio",
        )
    except (CrossFamilyCandidatePortfolioError, ValueError) as exc:
        raise FrontierRateAttackFeedbackCycleError(str(exc)) from exc

    component_model = _component_model_from_portfolio(portfolio)
    active = component_model.get("active") is True
    run_stamp = stamp or utc_stamp()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    portfolio_path = out / "portfolio.json"
    portfolio_md_path = out / "portfolio.md"
    action_summary_path = out / "action_summary.json"
    action_summary_md_path = out / "action_summary.md"

    write_json_artifact(portfolio_path, portfolio)
    write_text_artifact(
        portfolio_md_path,
        render_cross_family_candidate_portfolio_markdown(portfolio),
    )
    summary = {
        "schema": PAIRSET_COMPONENT_MARGINAL_ACTION_SUMMARY_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stamp": run_stamp,
        "producer": "comma_lab.scheduler.frontier_rate_attack_feedback_cycle",
        "allowed_use": "queue_owned_component_marginal_replanning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        "portfolio_json": repo_rel(portfolio_path, repo),
        "portfolio_sha256": sha256_file(portfolio_path),
        "portfolio_md": repo_rel(portfolio_md_path, repo),
        "action_summary_json": repo_rel(action_summary_path, repo),
        "pairset_acquisition_paths": [repo_rel(path, repo) for path in acquisition_paths],
        "observation_jsonl_paths": [
            repo_rel(path, repo) for path in resolved_observation_paths
        ],
        "drop_many_greedy_verdict_paths": [
            repo_rel(path, repo) for path in resolved_drop_many_verdict_paths
        ],
        "drop_many_greedy_verdict_discovery": discovery,
        "observation_row_count": len(observations),
        "pairset_component_marginal_model": component_model,
        "drop_many_greedy_verdict_model": dict(
            _mapping(
                _mapping(portfolio.get("observation_feedback")).get(
                    "drop_many_greedy_verdict_model"
                )
            )
        ),
        "portfolio_summary": dict(_mapping(portfolio.get("portfolio_summary"))),
        "top_action_limit": top_actions,
        "top_operator_actions": _top_component_marginal_actions(
            portfolio,
            limit=top_actions,
        ),
        "dispatch_blockers": [
            str(blocker) for blocker in portfolio.get("dispatch_blockers") or []
        ],
        **FALSE_AUTHORITY,
    }
    try:
        require_no_truthy_authority_fields(
            summary,
            context="frontier_rate_attack_feedback_cycle.component_marginal_summary",
        )
    except ValueError as exc:
        raise FrontierRateAttackFeedbackCycleError(str(exc)) from exc
    write_json_artifact(action_summary_path, summary)
    write_text_artifact(
        action_summary_md_path,
        _render_component_marginal_action_summary_markdown(summary),
    )
    return {
        "schema": FRONTIER_RATE_ATTACK_PAIRSET_COMPONENT_MARGINAL_BUNDLE_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "active": active,
        "inactive_reason": component_model.get("inactive_reason"),
        "component_model_schema": component_model.get("schema"),
        "training_row_count": component_model.get("training_row_count", 0),
        "axes": component_model.get("axes", []),
        "pairset_acquisition_paths": [repo_rel(path, repo) for path in acquisition_paths],
        "observation_jsonl_paths": [
            repo_rel(path, repo) for path in resolved_observation_paths
        ],
        "drop_many_greedy_verdict_paths": [
            repo_rel(path, repo) for path in resolved_drop_many_verdict_paths
        ],
        "drop_many_greedy_verdict_discovery": discovery,
        "drop_many_greedy_verdict_count": len(resolved_drop_many_verdict_paths),
        "observed_candidate_ids": sorted(
            {
                str(row.get("candidate_id"))
                for row in observations
                if row.get("candidate_id")
            }
        ),
        "portfolio_json": repo_rel(portfolio_path, repo),
        "portfolio_md": repo_rel(portfolio_md_path, repo),
        "action_summary_json": repo_rel(action_summary_path, repo),
        "action_summary_md": repo_rel(action_summary_md_path, repo),
        "recommended_next_candidate_id": _mapping(
            portfolio.get("portfolio_summary")
        ).get("recommended_next_candidate_id"),
        "recommended_next_action": _mapping(portfolio.get("portfolio_summary")).get(
            "recommended_next_action"
        ),
        "top_operator_action_candidate_ids": [
            str(row.get("candidate_id"))
            for row in summary["top_operator_actions"]
            if isinstance(row, Mapping) and row.get("candidate_id")
        ],
        "allowed_use": "queue_owned_component_marginal_replanning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def write_cycle_report(
    *,
    output_dir: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write the append-only cycle report."""

    path = Path(output_dir) / "frontier_rate_attack_feedback_cycle.json"
    try:
        write_json_artifact(path, dict(payload))
    except ArtifactWriteError as exc:
        raise FrontierRateAttackFeedbackCycleError(str(exc)) from exc
    return path


__all__ = [
    "AUTOPILOT_RESULT_SCHEMA",
    "DQS1_DROP_MANY_GREEDY_VERDICT_DISCOVERY_SCHEMA",
    "FRONTIER_RATE_ATTACK_DQS1_OBSERVATION_BUNDLE_SCHEMA",
    "FRONTIER_RATE_ATTACK_FEEDBACK_CYCLE_SCHEMA",
    "FRONTIER_RATE_ATTACK_PAIRSET_COMPONENT_MARGINAL_BUNDLE_SCHEMA",
    "FrontierRateAttackFeedbackCycleError",
    "artifact_token",
    "discover_dqs1_drop_many_greedy_verdict_paths",
    "harvest_paths_from_autopilot_payload",
    "harvest_paths_from_autopilot_result_files",
    "json_text",
    "load_json_object",
    "repo_rel",
    "resolve_repo_path",
    "select_pairset_acquisition_for_harvests",
    "utc_stamp",
    "write_cycle_report",
    "write_dqs1_harvest_observation_bundle",
    "write_frontier_refresh_artifacts",
    "write_pairset_component_marginal_feedback_bundle",
]
