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
    build_frontier_receiver_repair_queue,
    build_frontier_targeted_component_correction_queue,
)

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
        )
        if isinstance(correction_queue, Mapping):
            queue_path = out / "targeted_component_correction_queue.json"
            write_json_artifact(queue_path, dict(correction_queue))
            artifacts["targeted_component_correction_queue"] = repo_rel(
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
    if "targeted_component_correction_queue" in artifacts:
        operator_commands["validate_targeted_component_correction_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_queue"],
            "validate",
        ]
    if operator_commands:
        report_to_write["operator_commands"] = operator_commands
    write_json_artifact(report_path, report_to_write)
    artifacts["feedback_refresh_report"] = repo_rel(report_path, repo_root)
    return artifacts


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
