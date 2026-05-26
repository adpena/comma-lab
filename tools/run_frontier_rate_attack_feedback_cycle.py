#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the frontier rate-attack feedback loop as one queue-owned cycle.

The cycle is intentionally local and false-authority:

1. compile materializer/DQS1 observations into a bounded DQS1 follow-up queue;
2. optionally run that queue through the existing local-first autopilot;
3. harvest completed DQS1 results into canonical observation rows;
4. compile a refreshed follow-up queue that suppresses already-observed work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.dqs1_local_first_queue import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    find_latest_cross_family_action_summary,
)
from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.frontier_rate_attack_feedback import (  # noqa: E402
    FrontierRateAttackFeedbackError,
    build_frontier_rate_attack_feedback_refresh,
    discover_dqs1_observation_jsonl_paths,
    discover_local_cpu_eureka_planning_signals,
)
from comma_lab.scheduler.frontier_rate_attack_feedback_cycle import (  # noqa: E402
    FRONTIER_RATE_ATTACK_FEEDBACK_CYCLE_SCHEMA,
    FrontierRateAttackFeedbackCycleError,
    harvest_paths_from_autopilot_payload,
    harvest_paths_from_autopilot_result_files,
    json_text,
    load_json_object,
    repo_rel,
    resolve_repo_path,
    select_pairset_acquisition_for_harvests,
    utc_stamp,
    write_cycle_report,
    write_dqs1_harvest_observation_bundle,
    write_frontier_refresh_artifacts,
    write_pairset_component_marginal_feedback_bundle,
    write_targeted_component_correction_post_auxiliary_artifacts,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY  # noqa: E402
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

DEFAULT_BASELINE_ADVISORY = (
    "experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/"
    "dqs1_top32_cpu_advisory_venv.json"
)
DEFAULT_BASELINE_ARCHIVE_SIZE_BYTES = 178_560
DEFAULT_BASELINE_CANDIDATE_ID = "dqs1_top32_gap_uleb"


def _effective_raw_retention_execute(args: argparse.Namespace) -> bool:
    return (
        bool(args.execute_followup)
        if args.execute_raw_retention is None
        else bool(args.execute_raw_retention)
    )


def _display_path(path: str | Path) -> str:
    return repo_rel(path, REPO_ROOT)


def _is_under_repo(path: str | Path) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=False))
    except ValueError:
        return False
    return True


def _action_summary_path(value: str) -> Path | None:
    if value == "none":
        return None
    if value == "latest":
        return find_latest_cross_family_action_summary(REPO_ROOT)
    return resolve_repo_path(value, repo_root=REPO_ROOT)


def _latest_pairset_acquisition(root: str | Path) -> Path:
    resolved = resolve_repo_path(root, repo_root=REPO_ROOT)
    search_roots = [resolved / "pairset_acquisition", resolved]
    research_root = REPO_ROOT / ".omx" / "research"
    if research_root not in search_roots:
        search_roots.append(research_root)
    candidates: list[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        candidates.extend(search_root.glob("dqs1_pairset_acquisition_full_drop_two_*.json"))
    if not candidates:
        for search_root in search_roots:
            if not search_root.exists():
                continue
            candidates.extend(search_root.glob("dqs1_pairset_acquisition*.json"))
    if not candidates:
        raise FrontierRateAttackFeedbackCycleError(
            f"{resolved}: no DQS1 pairset acquisition JSON found"
        )
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name))[-1]


def _pairset_acquisition_path(value: str, *, root: str | Path) -> Path:
    if value == "latest":
        return _latest_pairset_acquisition(root)
    return resolve_repo_path(value, repo_root=REPO_ROOT)


def _parse_axis_scores(values: tuple[str, ...] | list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise FrontierRateAttackFeedbackCycleError(
                "--component-marginal-incumbent-score-by-axis must use AXIS=SCORE"
            )
        axis, raw = value.split("=", 1)
        axis = axis.strip()
        if not axis:
            raise FrontierRateAttackFeedbackCycleError(
                "--component-marginal-incumbent-score-by-axis axis must be non-empty"
            )
        try:
            out[axis] = float(raw)
        except ValueError as exc:
            raise FrontierRateAttackFeedbackCycleError(
                f"--component-marginal-incumbent-score-by-axis {axis} score must be numeric"
            ) from exc
    return out


def _baseline_advisory_score(path: str | Path) -> float:
    payload_path = resolve_repo_path(path, repo_root=REPO_ROOT)
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontierRateAttackFeedbackCycleError(
            f"{payload_path}: cannot load baseline advisory score"
        ) from exc
    if not isinstance(payload, dict):
        raise FrontierRateAttackFeedbackCycleError(
            f"{payload_path}: baseline advisory must be a JSON object"
        )
    for key in (
        "canonical_score",
        "score_recomputed_from_components",
        "observed_score",
        "local_score",
    ):
        value = payload.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed == parsed and parsed not in (float("inf"), float("-inf")):
            return parsed
    raise FrontierRateAttackFeedbackCycleError(
        f"{payload_path}: no finite baseline score field found"
    )


def _run(command: list[str], *, label: str) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise FrontierRateAttackFeedbackCycleError(
            f"{label} failed ({proc.returncode}): {' '.join(command)}"
        )
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FrontierRateAttackFeedbackCycleError(
            f"{label}: command did not emit JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise FrontierRateAttackFeedbackCycleError(f"{label}: expected JSON object")
    payload.setdefault("elapsed_seconds", time.monotonic() - started)
    return payload


def _validate_queue(queue_path: str | Path) -> dict[str, Any]:
    return _run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            _display_path(queue_path),
            "validate",
        ],
        label="queue validate",
    )


AUXILIARY_QUEUE_ARTIFACT_KEYS = (
    "receiver_repair_queue",
    "operation_chain_compiler_queue",
    "targeted_component_correction_queue",
    "targeted_component_correction_materialization_queue",
    "targeted_component_correction_operation_chain_queue",
)


def _run_auxiliary_queue_execution(
    *,
    args: argparse.Namespace,
    artifacts: dict[str, str],
    stage_label: str,
) -> dict[str, Any]:
    """Run non-DQS1 queues emitted by a refresh in bounded local slices."""

    rows: list[dict[str, Any]] = []
    if not args.execute_auxiliary_queues:
        return {
            "schema": "frontier_rate_attack_auxiliary_queue_execution.v1",
            "stage_label": stage_label,
            "execute_auxiliary_queues": False,
            "queue_count": 0,
            "executed_queue_count": 0,
            "rows": rows,
            **FALSE_AUTHORITY,
        }
    for artifact_key in AUXILIARY_QUEUE_ARTIFACT_KEYS:
        queue_path = artifacts.get(artifact_key)
        if not queue_path:
            continue
        validate = _validate_queue(queue_path)
        command = [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            _display_path(queue_path),
            "run-worker",
            "--execute",
            "--max-steps",
            str(args.auxiliary_queue_max_steps),
            "--max-experiments",
            str(args.auxiliary_queue_max_experiments),
            "--max-parallel",
            str(args.auxiliary_queue_max_parallel),
        ]
        result = _run(command, label=f"{stage_label} auxiliary queue {artifact_key}")
        rows.append(
            {
                "schema": "frontier_rate_attack_auxiliary_queue_execution_row.v1",
                "stage_label": stage_label,
                "artifact_key": artifact_key,
                "queue_path": queue_path,
                "validate": validate,
                "command": command,
                "result": {
                    "started_count": result.get("started_count"),
                    "completed_count": result.get("completed_count"),
                    "success_count": result.get("success_count"),
                    "failure_count": result.get("failure_count"),
                    "skipped_count": result.get("skipped_count"),
                    "pending_count": result.get("pending_count"),
                    "ready_count": result.get("ready_count"),
                    "running_count": result.get("running_count"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "stop_reason": result.get("stop_reason"),
                    **FALSE_AUTHORITY,
                },
                **FALSE_AUTHORITY,
            }
        )
    return {
        "schema": "frontier_rate_attack_auxiliary_queue_execution.v1",
        "stage_label": stage_label,
        "execute_auxiliary_queues": True,
        "queue_count": len(rows),
        "executed_queue_count": sum(
            1
            for row in rows
            if (
                row["result"].get("started_count")
                or row["result"].get("success_count")
                or row["result"].get("failure_count")
            )
        ),
        "failed_queue_count": sum(
            1 for row in rows if row["result"].get("failure_count")
        ),
        "max_steps_per_queue": args.auxiliary_queue_max_steps,
        "max_experiments_per_queue": args.auxiliary_queue_max_experiments,
        "max_parallel_per_queue": args.auxiliary_queue_max_parallel,
        "rows": rows,
        "allowed_use": "bounded_local_auxiliary_queue_execution_trace_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _write_post_auxiliary_targeted_component_refresh(
    *,
    args: argparse.Namespace,
    artifacts: dict[str, str],
    report: dict[str, Any],
    output_dir: Path,
    stage_label: str,
) -> dict[str, Any] | None:
    queue_path_text = artifacts.get("targeted_component_correction_queue")
    if not args.execute_auxiliary_queues or not queue_path_text:
        return None
    queue_path = resolve_repo_path(queue_path_text, repo_root=REPO_ROOT)
    targeted_queue = load_json_object(queue_path)
    return write_targeted_component_correction_post_auxiliary_artifacts(
        output_dir=output_dir,
        targeted_component_correction_queue=targeted_queue,
        targeted_component_correction_queue_path=queue_path,
        repo_root=REPO_ROOT,
        results_root=args.results_root,
        queue_id=f"{report.get('queue_id') or stage_label}_{stage_label}_post_auxiliary",
        candidate_limit=int(report.get("candidate_limit") or args.candidate_limit),
        dqs1_observation_source_paths=tuple(
            report.get("dqs1_observation_source_paths") or ()
        ),
        artifact_prefix=f"{stage_label}_post_auxiliary",
    )


def _build_refresh(
    *,
    args: argparse.Namespace,
    queue_id: str,
    dqs1_observation_paths: tuple[str | Path, ...],
    action_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    raw_retention_execute = _effective_raw_retention_execute(args)
    return build_frontier_rate_attack_feedback_refresh(
        repo_root=REPO_ROOT,
        frontier_artifact_roots=tuple(args.frontier_artifact_root),
        local_cpu_eureka_roots=tuple(args.local_cpu_eureka_root),
        materializer_feedback_paths=tuple(args.materializer_feedback),
        pair_frame_geometry_paths=tuple(args.pair_frame_geometry_lattice),
        dqs1_observation_paths=dqs1_observation_paths,
        action_summary_path=(
            resolve_repo_path(action_summary_path, repo_root=REPO_ROOT)
            if action_summary_path is not None
            else _action_summary_path(args.action_summary)
        ),
        results_root=args.results_root,
        queue_id=queue_id,
        candidate_limit=args.candidate_limit,
        skip_observed_dqs1_candidates=not args.include_observed_dqs1_candidate,
        local_cpu_concurrency=args.local_cpu_concurrency,
        local_io_concurrency=args.local_io_concurrency,
        include_raw_retention_plan=not args.skip_raw_retention_plan,
        raw_retention_execute=raw_retention_execute,
        raw_retention_action=args.raw_retention_action,
        raw_retention_cold_store_roots=tuple(args.raw_retention_cold_store_root),
        raw_retention_cold_store_reserve_gb=args.raw_retention_cold_store_reserve_gb,
        include_mlx_retention_plan=not args.skip_mlx_retention_plan,
        mlx_retention_execute=bool(args.execute_mlx_retention),
        mlx_retention_action=args.mlx_retention_action,
        mlx_retention_cold_store_roots=tuple(args.mlx_retention_cold_store_root),
        mlx_retention_cold_store_reserve_gb=args.mlx_retention_cold_store_reserve_gb,
    )


def _component_marginal_pairset_acquisition(args: argparse.Namespace) -> Path:
    value = args.component_marginal_pairset_acquisition or args.pairset_acquisition
    return _pairset_acquisition_path(value, root=args.pairset_acquisition_root)


def _cycle_dqs1_observation_paths(
    args: argparse.Namespace,
    explicit_paths: tuple[str | Path, ...],
) -> tuple[str | Path, ...]:
    discovery = discover_dqs1_observation_jsonl_paths(
        repo_root=REPO_ROOT,
        frontier_artifact_roots=tuple(
            args.frontier_artifact_root or (".omx/research",)
        ),
    )
    discovered = tuple(
        str(path)
        for path in discovery.get("discovered_observation_jsonl_paths", ())
    )
    return tuple(dict.fromkeys([*explicit_paths, *discovered]))


def _write_component_marginal_bundle(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    stamp: str,
    dqs1_observation_paths: tuple[str | Path, ...],
) -> dict[str, Any] | None:
    if args.skip_component_marginal_refresh or not dqs1_observation_paths:
        return None
    baseline_score = _baseline_advisory_score(args.baseline_advisory)
    axis_scores = _parse_axis_scores(args.component_marginal_incumbent_score_by_axis)
    axis_scores.setdefault("macos_cpu_advisory", baseline_score)
    return write_pairset_component_marginal_feedback_bundle(
        repo_root=REPO_ROOT,
        pairset_acquisition_paths=(_component_marginal_pairset_acquisition(args),),
        observation_paths=dqs1_observation_paths,
        incumbent_score=(
            baseline_score
            if args.component_marginal_incumbent_score is None
            else args.component_marginal_incumbent_score
        ),
        incumbent_scores_by_axis=axis_scores,
        output_dir=output_dir,
        stamp=stamp,
        top_k=args.component_marginal_top_k,
        top_actions=args.component_marginal_top_actions,
    )


def _component_action_summary_path(bundle: dict[str, Any] | None) -> str | None:
    if not bundle or bundle.get("active") is not True:
        return None
    observed = {str(value) for value in bundle.get("observed_candidate_ids") or []}
    top_ids = [
        str(value)
        for value in bundle.get("top_operator_action_candidate_ids") or []
        if value
    ]
    if top_ids and all(candidate_id in observed for candidate_id in top_ids):
        return None
    value = bundle.get("action_summary_json")
    return value if isinstance(value, str) and value.strip() else None


def _autopilot_command(
    *,
    args: argparse.Namespace,
    queue_path: str | Path,
) -> list[str]:
    worker_experiments = args.max_worker_experiments or args.candidate_limit
    max_candidates = args.max_candidates or args.candidate_limit
    max_total_steps = args.max_total_steps or max_candidates * args.max_steps_per_candidate
    command = [
        sys.executable,
        "tools/run_dqs1_local_first_autopilot.py",
        "--queue",
        _display_path(queue_path),
        "--max-candidates",
        str(max_candidates),
        "--max-total-steps",
        str(max_total_steps),
        "--max-steps-per-worker",
        str(max_total_steps),
        "--max-worker-experiments",
        str(worker_experiments),
        "--results-root",
        args.results_root,
        "--min-free-disk-gb",
        str(args.min_free_disk_gb),
    ]
    raw_retention_execute = _effective_raw_retention_execute(args)
    post_harvest_retention = args.post_harvest_retention
    if post_harvest_retention is None:
        post_harvest_retention = not raw_retention_execute
    if post_harvest_retention:
        command.extend(["--retention-action", args.raw_retention_action])
        command.extend(
            [
                "--retention-min-bytes",
                str(args.post_harvest_retention_min_bytes),
            ]
        )
        for root in args.raw_retention_cold_store_root:
            command.extend(["--retention-cold-store-root", root])
        if args.include_mlx_cache_post_harvest_retention:
            command.append("--include-mlx-cache-retention")
    else:
        command.append("--no-post-harvest-retention")
    if args.state is not None:
        command.extend(["--state", args.state])
    if args.execute_followup:
        command.append("--execute")
    return command


def _harvest_payload_has_exact_request(harvest_paths: tuple[Path, ...]) -> bool:
    for path in harvest_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("exact_auth_request_path") or payload.get("exact_auth_request"):
            return True
        if payload.get("recommended_action") == "dispatch_exact_auth_anchor":
            return True
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--queue-id", default=None)
    parser.add_argument("--post-harvest-queue-id", default=None)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--action-summary",
        default="latest",
        help="DQS1 action_summary path, latest, or none.",
    )
    parser.add_argument("--frontier-artifact-root", action="append", default=[])
    parser.add_argument(
        "--local-cpu-eureka-root",
        action="append",
        default=[],
        help=(
            "Root or file for local_cpu_contest_drift_eureka_*.json signals. "
            "Defaults to --frontier-artifact-root, or .omx/research when not supplied."
        ),
    )
    parser.add_argument("--materializer-feedback", action="append", default=[])
    parser.add_argument("--pair-frame-geometry-lattice", action="append", default=[])
    parser.add_argument(
        "--dqs1-observation-jsonl",
        "--dqs1-observations",
        action="append",
        default=[],
        dest="dqs1_observation_jsonl",
    )
    parser.add_argument("--include-observed-dqs1-candidate", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--local-cpu-concurrency", type=int, default=2)
    parser.add_argument("--local-io-concurrency", type=int, default=1)
    parser.add_argument("--skip-raw-retention-plan", action="store_true")
    parser.add_argument(
        "--execute-raw-retention",
        dest="execute_raw_retention",
        action="store_true",
        default=None,
        help=(
            "Execute raw/inflated retention inside the generated queue. Defaults "
            "to enabled when --execute-followup is used."
        ),
    )
    parser.add_argument(
        "--no-execute-raw-retention",
        dest="execute_raw_retention",
        action="store_false",
        help="Keep generated raw/inflated retention planning-only.",
    )
    parser.add_argument(
        "--raw-retention-action",
        choices=("move", "delete"),
        default="move",
        help="Action for queue-owned raw/inflated retention execution.",
    )
    parser.add_argument(
        "--raw-retention-cold-store-root",
        action="append",
        default=[],
        help=(
            "Cold-store root for queue-owned raw retention moves. May repeat; "
            "defaults to currently attached operator storage tiers."
        ),
    )
    parser.add_argument(
        "--raw-retention-cold-store-reserve-gb",
        type=float,
        default=40.0,
    )
    parser.add_argument("--skip-mlx-retention-plan", action="store_true")
    parser.add_argument("--execute-mlx-retention", action="store_true")
    parser.add_argument("--mlx-retention-action", choices=("move", "delete"), default="move")
    parser.add_argument("--mlx-retention-cold-store-root", action="append", default=[])
    parser.add_argument("--mlx-retention-cold-store-reserve-gb", type=float, default=40.0)
    parser.add_argument(
        "--post-harvest-retention",
        dest="post_harvest_retention",
        action="store_true",
        default=None,
        help=(
            "Enable autopilot post-harvest retention. Defaults to off when the "
            "generated queue owns raw retention execution."
        ),
    )
    parser.add_argument(
        "--no-post-harvest-retention",
        dest="post_harvest_retention",
        action="store_false",
        help="Disable autopilot post-harvest retention.",
    )
    parser.add_argument(
        "--post-harvest-retention-min-bytes",
        default="1",
        help="Minimum candidate size forwarded to autopilot post-harvest retention.",
    )
    parser.add_argument(
        "--include-mlx-cache-post-harvest-retention",
        action="store_true",
        help="Forward MLX cache retention to the autopilot post-harvest actuator.",
    )
    parser.add_argument(
        "--execute-followup",
        action="store_true",
        help="Run the generated follow-up queue locally with bounded autopilot.",
    )
    parser.add_argument(
        "--execute-auxiliary-queues",
        action="store_true",
        help=(
            "Run emitted receiver/operation/targeted auxiliary queues in bounded "
            "local slices and record the false-authority execution trace."
        ),
    )
    parser.add_argument("--auxiliary-queue-max-steps", type=int, default=1)
    parser.add_argument("--auxiliary-queue-max-experiments", type=int, default=1)
    parser.add_argument("--auxiliary-queue-max-parallel", type=int, default=1)
    parser.add_argument(
        "--autopilot-result-json",
        action="append",
        default=[],
        help="Existing DQS1 autopilot result JSON to harvest into this cycle.",
    )
    parser.add_argument(
        "--harvest",
        action="append",
        default=[],
        help="Existing dqs1_local_first_harvest JSON path. May repeat.",
    )
    parser.add_argument("--state", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-total-steps", type=int, default=None)
    parser.add_argument("--max-steps-per-candidate", type=int, default=8)
    parser.add_argument("--max-worker-experiments", type=int, default=None)
    parser.add_argument(
        "--campaign-waves",
        "--waves",
        type=int,
        default=1,
        help=(
            "Run repeated queue-owned feedback waves in one invocation. Wave 1 "
            "is the normal initial follow-up; later waves execute each refreshed "
            "post-harvest queue, harvest it, and rebuild the next queue."
        ),
    )
    parser.add_argument("--min-free-disk-gb", type=float, default=40.0)
    parser.add_argument("--pairset-acquisition", default="latest")
    parser.add_argument("--pairset-acquisition-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--skip-component-marginal-refresh",
        action="store_true",
        help=(
            "Do not canonicalize DQS1 observations into a component-marginal "
            "action summary before building the next queue."
        ),
    )
    parser.add_argument(
        "--component-marginal-pairset-acquisition",
        default=None,
        help=(
            "Pairset acquisition JSON for component-marginal canonicalization. "
            "Defaults to --pairset-acquisition."
        ),
    )
    parser.add_argument(
        "--component-marginal-incumbent-score",
        type=float,
        default=None,
        help=(
            "Incumbent score for component-marginal portfolio ranking. Defaults "
            "to the finite canonical score in --baseline-advisory."
        ),
    )
    parser.add_argument(
        "--component-marginal-incumbent-score-by-axis",
        action="append",
        default=[],
        metavar="AXIS=SCORE",
        help=(
            "Axis-specific incumbent score for component-marginal observations. "
            "Defaults macos_cpu_advisory to --baseline-advisory score."
        ),
    )
    parser.add_argument("--component-marginal-top-k", type=int, default=64)
    parser.add_argument("--component-marginal-top-actions", type=int, default=16)
    parser.add_argument("--baseline-advisory", default=DEFAULT_BASELINE_ADVISORY)
    parser.add_argument(
        "--baseline-archive-size-bytes",
        type=int,
        default=DEFAULT_BASELINE_ARCHIVE_SIZE_BYTES,
    )
    parser.add_argument("--baseline-candidate-id", default=DEFAULT_BASELINE_CANDIDATE_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.candidate_limit < 1:
        raise SystemExit("--candidate-limit must be >= 1")
    if args.local_cpu_concurrency < 1:
        raise SystemExit("--local-cpu-concurrency must be >= 1")
    if args.local_io_concurrency < 1:
        raise SystemExit("--local-io-concurrency must be >= 1")
    if args.max_steps_per_candidate < 1:
        raise SystemExit("--max-steps-per-candidate must be >= 1")
    if args.campaign_waves < 1:
        raise SystemExit("--campaign-waves must be >= 1")
    if args.campaign_waves > 1 and not args.execute_followup:
        raise SystemExit("--campaign-waves > 1 requires --execute-followup")
    if args.campaign_waves > 1 and args.state is not None:
        raise SystemExit("--campaign-waves > 1 cannot use a shared explicit --state")
    if args.auxiliary_queue_max_steps < 1:
        raise SystemExit("--auxiliary-queue-max-steps must be >= 1")
    if args.auxiliary_queue_max_experiments < 1:
        raise SystemExit("--auxiliary-queue-max-experiments must be >= 1")
    if args.auxiliary_queue_max_parallel < 1:
        raise SystemExit("--auxiliary-queue-max-parallel must be >= 1")
    stamp = utc_stamp()
    queue_id = args.queue_id or f"frontier_rate_attack_feedback_cycle_{stamp}"
    post_queue_id = args.post_harvest_queue_id or f"{queue_id}_post_harvest"
    output_dir = args.output_dir or (
        REPO_ROOT / ".omx" / "research" / f"frontier_rate_attack_feedback_cycle_{stamp}"
    )
    output_dir = resolve_repo_path(output_dir, repo_root=REPO_ROOT)
    initial_dir = output_dir / "initial_refresh"
    post_dir = output_dir / "post_harvest_refresh"
    observation_dir = output_dir / "dqs1_harvest_observations"
    initial_component_dir = output_dir / "initial_component_marginal_refresh"
    post_component_dir = output_dir / "post_harvest_component_marginal_refresh"

    try:
        initial_observations = tuple(args.dqs1_observation_jsonl)
        initial_observation_context = _cycle_dqs1_observation_paths(
            args,
            initial_observations,
        )
        initial_component_marginal = _write_component_marginal_bundle(
            args=args,
            output_dir=initial_component_dir,
            stamp=stamp,
            dqs1_observation_paths=initial_observation_context,
        )
        initial_action_summary_path = _component_action_summary_path(
            initial_component_marginal
        )
        initial_report = _build_refresh(
            args=args,
            queue_id=queue_id,
            dqs1_observation_paths=initial_observations,
            action_summary_path=initial_action_summary_path,
        )
        initial_artifacts = write_frontier_refresh_artifacts(
            output_dir=initial_dir,
            report=initial_report,
            repo_root=REPO_ROOT,
        )
        initial_validate = None
        queue_path = initial_artifacts.get("dqs1_followup_queue")
        if queue_path is not None:
            initial_validate = _validate_queue(queue_path)
        initial_auxiliary_execution = _run_auxiliary_queue_execution(
            args=args,
            artifacts=initial_artifacts,
            stage_label="initial_refresh",
        )
        initial_post_auxiliary_targeted_refresh = (
            _write_post_auxiliary_targeted_component_refresh(
                args=args,
                artifacts=initial_artifacts,
                report=initial_report,
                output_dir=initial_dir / "post_auxiliary_targeted_component_refresh",
                stage_label="initial_refresh",
            )
        )

        followup: dict[str, Any] = {
            "execute_followup": bool(args.execute_followup),
            "execute_auxiliary_queues": bool(args.execute_auxiliary_queues),
            "queue_raw_retention_execute": _effective_raw_retention_execute(args),
            "post_harvest_retention_policy": (
                "auto_disabled_when_queue_raw_retention_executes"
                if args.post_harvest_retention is None
                and _effective_raw_retention_execute(args)
                else (
                    "auto_enabled_when_queue_raw_retention_is_plan_only"
                    if args.post_harvest_retention is None
                    else "explicit_operator_setting"
                )
            ),
            "autopilot_command": None,
            "autopilot_result": None,
            **FALSE_AUTHORITY,
        }
        live_harvest_paths: tuple[Path, ...] = ()
        if args.execute_followup:
            if queue_path is None:
                raise FrontierRateAttackFeedbackCycleError(
                    "--execute-followup requires an initial refresh queue"
                )
            command = _autopilot_command(args=args, queue_path=queue_path)
            autopilot_payload = _run(command, label="DQS1 local-first autopilot")
            live_harvest_paths = harvest_paths_from_autopilot_payload(
                autopilot_payload,
                repo_root=REPO_ROOT,
            )
            followup["autopilot_command"] = command
            followup["autopilot_result"] = {
                "stop_reason": autopilot_payload.get("stop_reason"),
                "total_steps_started": autopilot_payload.get("total_steps_started"),
                "candidates_harvested": autopilot_payload.get("candidates_harvested"),
                "harvest_path_count": len(live_harvest_paths),
                **FALSE_AUTHORITY,
            }

        external_harvest_paths = harvest_paths_from_autopilot_result_files(
            tuple(args.autopilot_result_json),
            repo_root=REPO_ROOT,
        )
        explicit_harvest_paths = tuple(
            resolve_repo_path(path, repo_root=REPO_ROOT) for path in args.harvest
        )
        harvest_paths = tuple(
            dict.fromkeys([*live_harvest_paths, *external_harvest_paths, *explicit_harvest_paths])
        )

        observation_bundle = None
        post_component_marginal = None
        post_report = None
        post_artifacts: dict[str, str] = {}
        post_validate = None
        post_auxiliary_execution: dict[str, Any] | None = None
        post_auxiliary_targeted_refresh: dict[str, Any] | None = None
        campaign_waves: list[dict[str, Any]] = []
        campaign_stop_reason = "single_wave_only"
        post_observations: tuple[str | Path, ...] = initial_observation_context
        if harvest_paths:
            fallback_pairset_acquisition = _pairset_acquisition_path(
                args.pairset_acquisition,
                root=args.pairset_acquisition_root,
            )
            pairset_acquisition = select_pairset_acquisition_for_harvests(
                harvest_paths=harvest_paths,
                repo_root=REPO_ROOT,
                preferred_pairset_acquisition_path=initial_artifacts.get(
                    "dqs1_selected_pairset_acquisition"
                ),
                fallback_pairset_acquisition_path=fallback_pairset_acquisition,
            )
            observation_bundle = write_dqs1_harvest_observation_bundle(
                harvest_paths=harvest_paths,
                repo_root=REPO_ROOT,
                pairset_acquisition_path=pairset_acquisition,
                baseline_advisory_path=args.baseline_advisory,
                baseline_archive_size_bytes=args.baseline_archive_size_bytes,
                baseline_candidate_id=args.baseline_candidate_id,
                output_dir=observation_dir,
                stamp=stamp,
            )
            post_observations = (
                *initial_observation_context,
                observation_bundle["observation_jsonl"],
            )
            post_component_marginal = _write_component_marginal_bundle(
                args=args,
                output_dir=post_component_dir,
                stamp=stamp,
                dqs1_observation_paths=post_observations,
            )
            post_action_summary_path = _component_action_summary_path(
                post_component_marginal
            )
            post_report = _build_refresh(
                args=args,
                queue_id=post_queue_id,
                dqs1_observation_paths=post_observations,
                action_summary_path=post_action_summary_path,
            )
            post_artifacts = write_frontier_refresh_artifacts(
                output_dir=post_dir,
                report=post_report,
                repo_root=REPO_ROOT,
            )
            post_queue_path = post_artifacts.get("dqs1_followup_queue")
            if post_queue_path is not None:
                post_validate = _validate_queue(post_queue_path)
            post_auxiliary_execution = _run_auxiliary_queue_execution(
                args=args,
                artifacts=post_artifacts,
                stage_label="post_harvest_refresh",
            )
            post_auxiliary_targeted_refresh = (
                _write_post_auxiliary_targeted_component_refresh(
                    args=args,
                    artifacts=post_artifacts,
                    report=post_report,
                    output_dir=(
                        post_dir / "post_auxiliary_targeted_component_refresh"
                    ),
                    stage_label="post_harvest_refresh",
                )
            )
            campaign_stop_reason = "wave_limit_reached"

            current_queue_path = post_queue_path
            current_observations = post_observations
            current_pairset_acquisition = post_artifacts.get(
                "dqs1_selected_pairset_acquisition"
            )
            for wave_index in range(2, args.campaign_waves + 1):
                if current_queue_path is None:
                    campaign_stop_reason = "no_refreshed_queue_available"
                    break
                wave_stamp = f"{stamp}_wave{wave_index:03d}"
                wave_dir = output_dir / f"campaign_wave_{wave_index:03d}"
                wave_observation_dir = wave_dir / "dqs1_harvest_observations"
                wave_component_dir = wave_dir / "component_marginal_refresh"
                wave_refresh_dir = wave_dir / "refresh"
                command = _autopilot_command(args=args, queue_path=current_queue_path)
                autopilot_payload = _run(
                    command,
                    label=f"DQS1 local-first autopilot wave {wave_index}",
                )
                wave_harvest_paths = harvest_paths_from_autopilot_payload(
                    autopilot_payload,
                    repo_root=REPO_ROOT,
                )
                wave_record: dict[str, Any] = {
                    "wave_index": wave_index,
                    "queue_path": _display_path(current_queue_path),
                    "autopilot_command": command,
                    "autopilot_result": {
                        "stop_reason": autopilot_payload.get("stop_reason"),
                        "total_steps_started": autopilot_payload.get(
                            "total_steps_started"
                        ),
                        "candidates_harvested": autopilot_payload.get(
                            "candidates_harvested"
                        ),
                        "harvest_path_count": len(wave_harvest_paths),
                        **FALSE_AUTHORITY,
                    },
                    **FALSE_AUTHORITY,
                }
                if not wave_harvest_paths:
                    wave_record["terminal"] = "no_harvest"
                    campaign_waves.append(wave_record)
                    campaign_stop_reason = "no_harvest"
                    break
                fallback_pairset_acquisition = _pairset_acquisition_path(
                    args.pairset_acquisition,
                    root=args.pairset_acquisition_root,
                )
                pairset_acquisition = select_pairset_acquisition_for_harvests(
                    harvest_paths=wave_harvest_paths,
                    repo_root=REPO_ROOT,
                    preferred_pairset_acquisition_path=current_pairset_acquisition,
                    fallback_pairset_acquisition_path=fallback_pairset_acquisition,
                )
                wave_observation_bundle = write_dqs1_harvest_observation_bundle(
                    harvest_paths=wave_harvest_paths,
                    repo_root=REPO_ROOT,
                    pairset_acquisition_path=pairset_acquisition,
                    baseline_advisory_path=args.baseline_advisory,
                    baseline_archive_size_bytes=args.baseline_archive_size_bytes,
                    baseline_candidate_id=args.baseline_candidate_id,
                    output_dir=wave_observation_dir,
                    stamp=wave_stamp,
                )
                next_observations = (
                    *current_observations,
                    wave_observation_bundle["observation_jsonl"],
                )
                wave_component_marginal = _write_component_marginal_bundle(
                    args=args,
                    output_dir=wave_component_dir,
                    stamp=wave_stamp,
                    dqs1_observation_paths=next_observations,
                )
                wave_action_summary_path = _component_action_summary_path(
                    wave_component_marginal
                )
                wave_report = _build_refresh(
                    args=args,
                    queue_id=f"{post_queue_id}_wave{wave_index:03d}",
                    dqs1_observation_paths=next_observations,
                    action_summary_path=wave_action_summary_path,
                )
                wave_artifacts = write_frontier_refresh_artifacts(
                    output_dir=wave_refresh_dir,
                    report=wave_report,
                    repo_root=REPO_ROOT,
                )
                wave_queue_path = wave_artifacts.get("dqs1_followup_queue")
                wave_validate = (
                    None if wave_queue_path is None else _validate_queue(wave_queue_path)
                )
                wave_record.update(
                    {
                        "harvest_paths": [
                            _display_path(path) for path in wave_harvest_paths
                        ],
                        "observation_bundle": wave_observation_bundle,
                        "component_marginal": wave_component_marginal,
                        "refresh": {
                            "artifacts": wave_artifacts,
                            "selected_candidate_ids": wave_report.get(
                                "selected_candidate_ids"
                            ),
                            "queue_summary": wave_report.get("queue_summary"),
                            "queue_validate": wave_validate,
                            **FALSE_AUTHORITY,
                        },
                    }
                )
                campaign_waves.append(wave_record)
                current_observations = next_observations
                current_queue_path = wave_queue_path
                current_pairset_acquisition = wave_artifacts.get(
                    "dqs1_selected_pairset_acquisition"
                )
                if _harvest_payload_has_exact_request(wave_harvest_paths):
                    campaign_stop_reason = "exact_auth_anchor_request_created"
                    break

        post_followup_eureka_root_values: list[str] = list(args.frontier_artifact_root)
        if _is_under_repo(output_dir):
            post_followup_eureka_root_values.append(str(REPO_ROOT / ".omx" / "research"))
        post_followup_eureka_root_values.append(str(output_dir))
        post_followup_eureka_roots = tuple(dict.fromkeys(post_followup_eureka_root_values))
        post_followup_eureka_planning = discover_local_cpu_eureka_planning_signals(
            repo_root=REPO_ROOT,
            frontier_artifact_roots=post_followup_eureka_roots,
            strict_authority=False,
        )
        post_followup_eureka_path = output_dir / "post_followup_local_cpu_eureka_planning.json"
        write_json_artifact(post_followup_eureka_path, post_followup_eureka_planning)

        payload = {
            "schema": FRONTIER_RATE_ATTACK_FEEDBACK_CYCLE_SCHEMA,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_dir": _display_path(output_dir),
            "initial_refresh": {
                "artifacts": initial_artifacts,
                "selected_candidate_ids": initial_report.get("selected_candidate_ids"),
                "queue_summary": initial_report.get("queue_summary"),
                "retention_policy": initial_report.get("retention_policy"),
                "local_cpu_eureka_planning": initial_report.get(
                    "local_cpu_eureka_planning"
                ),
                "operation_portfolio": initial_report.get("operation_portfolio"),
                "autonomous_chain_optimization": initial_report.get(
                    "autonomous_chain_optimization"
                ),
                "receiver_repair_backlog": initial_report.get(
                    "receiver_repair_backlog"
                ),
                "receiver_closed_correction_budget": initial_report.get(
                    "receiver_closed_correction_budget"
                ),
                "pairset_component_marginal": initial_component_marginal,
                "queue_validate": initial_validate,
                "auxiliary_queue_execution": initial_auxiliary_execution,
                "post_auxiliary_targeted_component_refresh": (
                    initial_post_auxiliary_targeted_refresh
                ),
                **FALSE_AUTHORITY,
            },
            "followup_execution": followup,
            "harvest_signal": {
                "harvest_path_count": len(harvest_paths),
                "harvest_paths": [_display_path(path) for path in harvest_paths],
                "observation_bundle": observation_bundle,
                **FALSE_AUTHORITY,
            },
            "post_followup_eureka_planning": {
                "path": _display_path(post_followup_eureka_path),
                "payload": post_followup_eureka_planning,
                **FALSE_AUTHORITY,
            },
            "post_harvest_refresh": None
            if post_report is None
            else {
                "artifacts": post_artifacts,
                "selected_candidate_ids": post_report.get("selected_candidate_ids"),
                "queue_summary": post_report.get("queue_summary"),
                "retention_policy": post_report.get("retention_policy"),
                "local_cpu_eureka_planning": post_report.get(
                    "local_cpu_eureka_planning"
                ),
                "operation_portfolio": post_report.get("operation_portfolio"),
                "autonomous_chain_optimization": post_report.get(
                    "autonomous_chain_optimization"
                ),
                "receiver_repair_backlog": post_report.get("receiver_repair_backlog"),
                "receiver_closed_correction_budget": post_report.get(
                    "receiver_closed_correction_budget"
                ),
                "pairset_component_marginal": post_component_marginal,
                "queue_validate": post_validate,
                "auxiliary_queue_execution": post_auxiliary_execution,
                "post_auxiliary_targeted_component_refresh": (
                    post_auxiliary_targeted_refresh
                ),
                **FALSE_AUTHORITY,
            },
            "campaign_execution": {
                "requested_wave_count": args.campaign_waves,
                "executed_followup": bool(args.execute_followup),
                "completed_additional_wave_count": len(campaign_waves),
                "stop_reason": campaign_stop_reason,
                "waves": campaign_waves,
                **FALSE_AUTHORITY,
            },
            "integration_edges": [
                "family_materializer_feedback_to_dqs1_bridge",
                "dqs1_batch_followup_queue_to_local_autopilot",
                "local_autopilot_harvest_to_dynamic_observation_jsonl",
                "dynamic_observation_jsonl_to_pairset_component_marginal_model",
                "pairset_component_marginal_model_to_queue_owned_local_first_actions",
                "dynamic_observation_jsonl_to_refreshed_dqs1_queue",
                "local_cpu_eureka_signal_to_default_feedback_refresh",
                "post_followup_eureka_signal_to_next_acquisition_hint",
                "materializer_eureka_component_signals_to_many_operation_portfolio",
                "operation_portfolio_to_chained_materializer_and_receiver_backlog",
                "operation_portfolio_to_materializer_backlog_context_work_queue",
                "exact_readiness_bridge_to_receiver_repair_backlog_and_correction_budget",
                "receiver_closed_rate_budget_to_targeted_segnet_posenet_correction_planning",
                "receiver_closed_correction_acquisition_to_local_component_correction_queue",
                "targeted_component_correction_queue_to_response_harvest_and_materialization_requests",
                "targeted_component_materialization_requests_to_operation_chain_queue",
                "targeted_component_operation_chain_to_materializer_handoff",
                "targeted_operation_chain_queue_to_targeted_drop_many_child_queue",
                "autonomous_chain_optimization_to_queue_owned_many_op_plan",
                "many_op_plan_to_component_replay_and_exact_readiness_bridge",
                "bounded_auxiliary_queue_artifacts_to_local_execution_trace",
                "bounded_auxiliary_targeted_response_reharvest_to_materialization_chain",
            ],
            "allowed_use": "local_queue_owned_frontier_feedback_iteration_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
            **FALSE_AUTHORITY,
        }
        report_path = write_cycle_report(output_dir=output_dir, payload=payload)
    except (
        ExperimentQueueError,
        ArtifactWriteError,
        FrontierRateAttackFeedbackError,
        FrontierRateAttackFeedbackCycleError,
        OSError,
    ) as exc:
        print(f"FATAL: frontier rate-attack feedback cycle failed: {exc}", file=sys.stderr)
        return 2

    print(
        json_text(
            {
                "schema": "frontier_rate_attack_feedback_cycle_cli_result.v1",
                "cycle_report": _display_path(report_path),
                "output_dir": _display_path(output_dir),
                "initial_selected_candidate_ids": initial_report.get(
                    "selected_candidate_ids"
                ),
                "post_harvest_selected_candidate_ids": None
                if post_report is None
                else post_report.get("selected_candidate_ids"),
                "harvest_path_count": len(harvest_paths),
                "observation_jsonl": None
                if observation_bundle is None
                else observation_bundle.get("observation_jsonl"),
                "initial_component_marginal_action_summary": None
                if initial_component_marginal is None
                else initial_component_marginal.get("action_summary_json"),
                "post_harvest_component_marginal_action_summary": None
                if post_component_marginal is None
                else post_component_marginal.get("action_summary_json"),
                "initial_targeted_component_correction_chain_materializer_handoff_summary": (
                    None
                    if not isinstance(
                        initial_report.get(
                            "targeted_component_correction_chain_materializer_handoff"
                        ),
                        dict,
                    )
                    else {
                        "work_queue_row_count": initial_report[
                            "targeted_component_correction_chain_materializer_handoff"
                        ].get("work_queue_row_count"),
                        "executable_work_row_count": initial_report[
                            "targeted_component_correction_chain_materializer_handoff"
                        ].get("executable_work_row_count"),
                        "unregistered_chain_target_count": initial_report[
                            "targeted_component_correction_chain_materializer_handoff"
                        ].get("unregistered_chain_target_count"),
                        **FALSE_AUTHORITY,
                    }
                ),
                "initial_autonomous_chain_optimization_summary": (
                    None
                    if not isinstance(
                        initial_report.get("autonomous_chain_optimization"),
                        dict,
                    )
                    else {
                        "chain_count": initial_report[
                            "autonomous_chain_optimization"
                        ].get("chain_count"),
                        "top_chain_ids": initial_report[
                            "autonomous_chain_optimization"
                        ].get("top_chain_ids"),
                        "target_classes": initial_report[
                            "autonomous_chain_optimization"
                        ].get("target_classes"),
                        "registered_target_count": initial_report[
                            "autonomous_chain_optimization"
                        ].get("registered_target_count"),
                        **FALSE_AUTHORITY,
                    }
                ),
                "post_harvest_targeted_component_correction_chain_materializer_handoff_summary": (
                    None
                    if post_report is None
                    or not isinstance(
                        post_report.get(
                            "targeted_component_correction_chain_materializer_handoff"
                        ),
                        dict,
                    )
                    else {
                        "work_queue_row_count": post_report[
                            "targeted_component_correction_chain_materializer_handoff"
                        ].get("work_queue_row_count"),
                        "executable_work_row_count": post_report[
                            "targeted_component_correction_chain_materializer_handoff"
                        ].get("executable_work_row_count"),
                        "unregistered_chain_target_count": post_report[
                            "targeted_component_correction_chain_materializer_handoff"
                        ].get("unregistered_chain_target_count"),
                        **FALSE_AUTHORITY,
                    }
                ),
                "post_harvest_autonomous_chain_optimization_summary": (
                    None
                    if post_report is None
                    or not isinstance(
                        post_report.get("autonomous_chain_optimization"),
                        dict,
                    )
                    else {
                        "chain_count": post_report[
                            "autonomous_chain_optimization"
                        ].get("chain_count"),
                        "top_chain_ids": post_report[
                            "autonomous_chain_optimization"
                        ].get("top_chain_ids"),
                        "target_classes": post_report[
                            "autonomous_chain_optimization"
                        ].get("target_classes"),
                        "registered_target_count": post_report[
                            "autonomous_chain_optimization"
                        ].get("registered_target_count"),
                        **FALSE_AUTHORITY,
                    }
                ),
                "initial_targeted_drop_many_dqs1_child_queue_paths": list(
                    initial_report.get("targeted_drop_many_dqs1_child_queue_paths")
                    or []
                ),
                "post_harvest_targeted_drop_many_dqs1_child_queue_paths": []
                if post_report is None
                else list(
                    post_report.get("targeted_drop_many_dqs1_child_queue_paths") or []
                ),
                "initial_auxiliary_queue_execution_summary": {
                    "queue_count": initial_auxiliary_execution.get("queue_count"),
                    "executed_queue_count": initial_auxiliary_execution.get(
                        "executed_queue_count"
                    ),
                    "failed_queue_count": initial_auxiliary_execution.get(
                        "failed_queue_count"
                    ),
                    **FALSE_AUTHORITY,
                },
                "initial_post_auxiliary_targeted_component_refresh_summary": (
                    None
                    if initial_post_auxiliary_targeted_refresh is None
                    else {
                        "response_harvest_row_count": (
                            initial_post_auxiliary_targeted_refresh.get(
                                "response_harvest_row_count"
                            )
                        ),
                        "materialization_request_row_count": (
                            initial_post_auxiliary_targeted_refresh.get(
                                "materialization_request_row_count"
                            )
                        ),
                        "operation_chain_work_order_count": (
                            initial_post_auxiliary_targeted_refresh.get(
                                "operation_chain_work_order_count"
                            )
                        ),
                        "chain_materializer_handoff_work_queue_row_count": (
                            initial_post_auxiliary_targeted_refresh.get(
                                "chain_materializer_handoff_work_queue_row_count"
                            )
                        ),
                        **FALSE_AUTHORITY,
                    }
                ),
                "post_harvest_auxiliary_queue_execution_summary": None
                if post_report is None
                else {
                    "queue_count": post_auxiliary_execution.get("queue_count"),
                    "executed_queue_count": post_auxiliary_execution.get(
                        "executed_queue_count"
                    ),
                    "failed_queue_count": post_auxiliary_execution.get(
                        "failed_queue_count"
                    ),
                    **FALSE_AUTHORITY,
                },
                "post_harvest_post_auxiliary_targeted_component_refresh_summary": (
                    None
                    if post_auxiliary_targeted_refresh is None
                    else {
                        "response_harvest_row_count": (
                            post_auxiliary_targeted_refresh.get(
                                "response_harvest_row_count"
                            )
                        ),
                        "materialization_request_row_count": (
                            post_auxiliary_targeted_refresh.get(
                                "materialization_request_row_count"
                            )
                        ),
                        "operation_chain_work_order_count": (
                            post_auxiliary_targeted_refresh.get(
                                "operation_chain_work_order_count"
                            )
                        ),
                        "chain_materializer_handoff_work_queue_row_count": (
                            post_auxiliary_targeted_refresh.get(
                                "chain_materializer_handoff_work_queue_row_count"
                            )
                        ),
                        **FALSE_AUTHORITY,
                    }
                ),
                **FALSE_AUTHORITY,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
