#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile frontier materializer feedback into a queue-owned DQS1 refresh.

The tool discovers family-agnostic materializer sweep observations across
frontier artifact roots, folds in DQS1 local-first harvest observations, and
emits the next bounded follow-up queue. It is false-authority by construction:
the output may steer local queue work, but it cannot claim score, promote,
rank/kill, or dispatch paid exact eval.
"""

from __future__ import annotations

import argparse
import json
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
    OPERATION_CHAIN_COMPILER_WORK_ORDER_SCHEMA,
    OPERATION_CHAIN_COMPILER_WORK_ORDERS_SCHEMA,
    FrontierRateAttackFeedbackError,
    attach_frontier_autonomous_chain_optimization,
    build_frontier_autonomous_chain_optimization_queue,
    build_frontier_materializer_execution_queue_if_available,
    build_frontier_operation_chain_compiler_queue,
    build_frontier_rate_attack_feedback_refresh,
    build_frontier_receiver_repair_queue,
    build_frontier_repair_budget_waterfill_queue,
    build_frontier_targeted_component_correction_chain_materializer_handoff,
    build_frontier_targeted_component_correction_chain_work_orders,
    build_frontier_targeted_component_correction_materialization_queue,
    build_frontier_targeted_component_correction_materialization_requests,
    build_frontier_targeted_component_correction_queue,
    build_frontier_targeted_component_correction_response_harvest,
)
from tac.fec6_selector_operator_space import FEC6_FIXED_K16_MODE_IDS  # noqa: E402
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY  # noqa: E402
from tac.optimization.pair_frame_scorer_geometry_lattice import (  # noqa: E402
    PairFrameScorerGeometryLatticeError,
    build_pair_frame_scorer_geometry_lattice,
)
from tac.optimization.pair_frame_scorer_geometry_lattice import (  # noqa: E402
    load_json_object as load_pair_frame_json_object,
)
from tac.optimization.pair_frame_scorer_geometry_lattice import (  # noqa: E402
    write_json as write_pair_frame_json,
)
from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    require_no_truthy_authority_fields,
)
from tac.repo_io import ArtifactWriteError, json_text, write_json_artifact  # noqa: E402


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve(strict=False).relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _repair_palette_modes_from_args(args: argparse.Namespace) -> list[str]:
    modes = list(args.repair_palette_mode or [])
    for palette_id in args.repair_palette or []:
        if palette_id == "fec6-fixed-k16":
            modes.extend(FEC6_FIXED_K16_MODE_IDS)
    return _dedupe_strings(modes)


def _load_repair_dynamics_priors(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    priors: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise FrontierRateAttackFeedbackError(
                f"repair dynamics prior must be a JSON object: {path}"
            )
        if isinstance(payload.get("repair_dynamics_palette_prior"), dict):
            prior = dict(payload["repair_dynamics_palette_prior"])
        elif isinstance(payload.get("manual_repair_dynamics_palette_prior"), dict):
            prior = dict(payload["manual_repair_dynamics_palette_prior"])
        elif isinstance(payload.get("palette_modes"), list):
            prior = dict(payload)
        else:
            raise FrontierRateAttackFeedbackError(
                "repair dynamics prior JSON must contain palette_modes or a "
                f"repair_dynamics_palette_prior object: {path}"
            )
        prior.setdefault("source", _display_path(path))
        require_no_truthy_authority_fields(
            prior,
            context=f"repair_dynamics_prior:{_display_path(path)}",
        )
        priors.append(prior)
        source_paths.append(_display_path(path))
    return priors, source_paths


def _targeted_dqs1_child_queue_paths(queue: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, dict):
            continue
        metadata = experiment.get("metadata")
        if not isinstance(metadata, dict):
            continue
        value = metadata.get("targeted_drop_many_dqs1_followup_queue_path")
        if not isinstance(value, str) or not value.strip() or value in seen:
            continue
        seen.add(value)
        paths.append(value)
    return paths


def _parse_csv_ints(text: str | None) -> list[int] | None:
    if text is None:
        return None
    try:
        parsed = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise PairFrameScorerGeometryLatticeError(
            "--pair-frame-drop-counts must be a comma-separated integer list"
        ) from exc
    return parsed or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--action-summary",
        default="latest",
        help=(
            "DQS1 action_summary path, 'latest', or 'none' to emit only the "
            "materializer/DQS1 bridge without a follow-up queue."
        ),
    )
    parser.add_argument(
        "--frontier-artifact-root",
        action="append",
        default=[],
        help="Root to scan for frontier materializer sweep.json/observations.jsonl artifacts.",
    )
    parser.add_argument(
        "--max-files-per-root",
        type=int,
        default=4096,
        help=(
            "Maximum candidate files to inspect per discovered artifact root. "
            "Raise this for noisy .omx/research refreshes instead of forcing "
            "operators to enumerate every root by hand."
        ),
    )
    parser.add_argument(
        "--local-cpu-eureka-root",
        action="append",
        default=[],
        help="Root or file for local_cpu_contest_drift_eureka_*.json planning signals.",
    )
    parser.add_argument(
        "--materializer-feedback",
        action="append",
        default=[],
        help="Explicit family-agnostic materializer sweep/observation JSON or JSONL. May repeat.",
    )
    parser.add_argument(
        "--pair-frame-geometry-lattice",
        action="append",
        default=[],
        help=(
            "Explicit pair-frame scorer-geometry lattice JSON whose "
            "queue-executable requests should seed the DQS1 follow-up queue. "
            "May repeat."
        ),
    )
    parser.add_argument(
        "--pair-frame-pairset-acquisition",
        type=Path,
        default=None,
        help=(
            "decoder_q_pairset_acquisition.v1 JSON to compile into a "
            "pair-frame scorer-geometry lattice inside this refresh before "
            "building the DQS1 follow-up queue."
        ),
    )
    parser.add_argument(
        "--pair-frame-curriculum",
        type=Path,
        default=None,
        help="Optional frame-pair curriculum JSON for generated pair-frame geometry.",
    )
    parser.add_argument(
        "--pair-component-xray",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional pair_component_error_xray_v1 JSON for generated "
            "pair-frame geometry. May repeat."
        ),
    )
    parser.add_argument(
        "--pair-frame-drop-counts",
        default="3,4,6,8,12,16",
        help="Comma-separated drop counts for generated pair-frame geometry requests.",
    )
    parser.add_argument(
        "--pair-frame-max-requests",
        type=int,
        default=32,
        help="Maximum generated pair-frame geometry queue requests.",
    )
    parser.add_argument(
        "--dqs1-observation-jsonl",
        "--dqs1-observations",
        action="append",
        default=[],
        dest="dqs1_observation_jsonl",
        help="DQS1 local-first harvest observation JSONL. May repeat; rows are deduped.",
    )
    parser.add_argument(
        "--repair-palette",
        choices=("fec6-fixed-k16",),
        action="append",
        default=[],
        help=(
            "Named repair-dynamics palette prior to inject as local planning "
            "signal. Currently supports the fixed PR110/FEC6 K=16 palette."
        ),
    )
    parser.add_argument(
        "--repair-dynamics-prior",
        type=Path,
        action="append",
        default=[],
        help=(
            "False-authority repair-dynamics prior JSON to inject into targeted "
            "component acquisition. May repeat."
        ),
    )
    parser.add_argument(
        "--repair-palette-mode",
        action="append",
        default=[],
        help=(
            "Manual repair-dynamics palette mode to inject as local planning "
            "signal. May repeat; never grants score or dispatch authority."
        ),
    )
    parser.add_argument(
        "--include-observed-dqs1-candidate",
        action="store_true",
        help="Allow replaying DQS1 candidates already present in observation JSONLs.",
    )
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--local-cpu-concurrency", type=int, default=1)
    parser.add_argument("--local-io-concurrency", type=int, default=1)
    parser.add_argument(
        "--component-response-cache-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Root containing prior false-authority local CPU advisory/cache-hash "
            "artifacts for targeted component correction queue reuse. May repeat."
        ),
    )
    parser.add_argument(
        "--skip-raw-retention-plan",
        action="store_true",
        help="Do not add the raw/inflated artifact retention planning step.",
    )
    parser.add_argument(
        "--execute-raw-retention",
        action="store_true",
        help=(
            "Make the raw/inflated retention step execute inside the generated "
            "queue instead of only writing a plan."
        ),
    )
    parser.add_argument(
        "--raw-retention-action",
        choices=("move", "delete"),
        default="move",
        help="Action for --execute-raw-retention.",
    )
    parser.add_argument(
        "--raw-retention-cold-store-root",
        action="append",
        default=[],
        help=(
            "Cold-store root for executed raw retention moves. May repeat; "
            "defaults to currently attached operator storage tiers."
        ),
    )
    parser.add_argument(
        "--raw-retention-cold-store-reserve-gb",
        type=float,
        default=40.0,
        help="Free GiB to preserve on each cold-store tier for raw retention moves.",
    )
    parser.add_argument(
        "--skip-mlx-retention-plan",
        action="store_true",
        help="Do not add the mlx_delta_cache retention planning step.",
    )
    parser.add_argument(
        "--execute-mlx-retention",
        action="store_true",
        help="Make the MLX cache retention step execute inside the generated queue.",
    )
    parser.add_argument(
        "--mlx-retention-action",
        choices=("move", "delete"),
        default="move",
        help="Action for --execute-mlx-retention.",
    )
    parser.add_argument(
        "--mlx-retention-cold-store-root",
        action="append",
        default=[],
        help=(
            "Cold-store root for executed MLX cache moves. May repeat; defaults "
            "to currently attached operator storage tiers."
        ),
    )
    parser.add_argument(
        "--mlx-retention-cold-store-reserve-gb",
        type=float,
        default=40.0,
        help="Free GiB to preserve on each cold-store tier for MLX cache moves.",
    )
    return parser.parse_args(argv)


def _action_summary_path(value: str) -> Path | None:
    if value == "none":
        return None
    if value == "latest":
        return find_latest_cross_family_action_summary(REPO_ROOT)
    return Path(value)


def _build_generated_pair_frame_lattice(
    *,
    output_dir: Path,
    pairset_acquisition_path: Path | None,
    frame_pair_curriculum_path: Path | None,
    pair_component_xray_paths: tuple[Path, ...],
    drop_counts: str | None,
    max_requests: int,
) -> tuple[Path | None, dict[str, Any] | None]:
    if pairset_acquisition_path is None:
        return None, None
    if max_requests < 1:
        raise PairFrameScorerGeometryLatticeError(
            "--pair-frame-max-requests must be >= 1"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_pair_frame_scorer_geometry_lattice(
        load_pair_frame_json_object(pairset_acquisition_path),
        frame_pair_curriculum=(
            None
            if frame_pair_curriculum_path is None
            else load_pair_frame_json_object(frame_pair_curriculum_path)
        ),
        pair_component_xrays=tuple(
            load_pair_frame_json_object(path) for path in pair_component_xray_paths
        ),
        drop_counts=_parse_csv_ints(drop_counts),
        max_requests=max_requests,
    )
    path = output_dir / "pair_frame_scorer_geometry_lattice.json"
    write_pair_frame_json(path, payload)
    summary = {
        "schema": "frontier_rate_attack_generated_pair_frame_geometry_lattice.v1",
        "pairset_acquisition_path": _display_path(pairset_acquisition_path),
        "frame_pair_curriculum_path": (
            None
            if frame_pair_curriculum_path is None
            else _display_path(frame_pair_curriculum_path)
        ),
        "pair_component_xray_paths": [
            _display_path(path) for path in pair_component_xray_paths
        ],
        "lattice_path": _display_path(path),
        "row_count": payload.get("summary", {}).get("row_count")
        if isinstance(payload.get("summary"), dict)
        else None,
        "queue_executable_request_count": payload.get("summary", {}).get(
            "queue_executable_request_count"
        )
        if isinstance(payload.get("summary"), dict)
        else None,
        "geometry_coverage": payload.get("coverage", {}).get("geometry_coverage")
        if isinstance(payload.get("coverage"), dict)
        else None,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }
    require_no_truthy_authority_fields(
        summary,
        context="generated_pair_frame_geometry_lattice_summary",
    )
    return path, summary


def _write_outputs(output_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    generated_lattice_path = report.get("generated_pair_frame_geometry_lattice_path")
    if isinstance(generated_lattice_path, str) and generated_lattice_path:
        artifacts["generated_pair_frame_geometry_lattice"] = generated_lattice_path
    discovery = report.get("discovery")
    if isinstance(discovery, dict):
        path = output_dir / "materializer_feedback_discovery.json"
        write_json_artifact(path, discovery)
        artifacts["materializer_feedback_discovery"] = _display_path(path)
    dqs1_discovery = report.get("dqs1_observation_discovery")
    if isinstance(dqs1_discovery, dict):
        path = output_dir / "dqs1_observation_discovery.json"
        write_json_artifact(path, dqs1_discovery)
        artifacts["dqs1_observation_discovery"] = _display_path(path)
    pair_frame = report.get("pair_frame_geometry_discovery")
    if isinstance(pair_frame, dict):
        path = output_dir / "pair_frame_geometry_discovery.json"
        write_json_artifact(path, pair_frame)
        artifacts["pair_frame_geometry_discovery"] = _display_path(path)
    eureka_planning = report.get("local_cpu_eureka_planning")
    if isinstance(eureka_planning, dict):
        path = output_dir / "local_cpu_eureka_planning.json"
        write_json_artifact(path, eureka_planning)
        artifacts["local_cpu_eureka_planning"] = _display_path(path)
    operation_portfolio = report.get("operation_portfolio")
    if isinstance(operation_portfolio, dict):
        path = output_dir / "operation_portfolio.json"
        write_json_artifact(path, operation_portfolio)
        artifacts["operation_portfolio"] = _display_path(path)
    rate_budget_preservation_plan = report.get("rate_budget_preservation_plan")
    if isinstance(rate_budget_preservation_plan, dict):
        path = output_dir / "rate_budget_preservation_plan.json"
        write_json_artifact(path, rate_budget_preservation_plan)
        artifacts["rate_budget_preservation_plan"] = _display_path(path)
    operation_materializer_bridge = report.get("operation_materializer_bridge")
    operation_work_queue: dict[str, object] | None = None
    operation_execution_queue: dict[str, object] | None = None
    if isinstance(operation_materializer_bridge, dict):
        path = output_dir / "operation_materializer_bridge.json"
        write_json_artifact(path, operation_materializer_bridge)
        artifacts["operation_materializer_bridge"] = _display_path(path)
        chain_work_orders: list[dict[str, Any]] = []
        for index, row in enumerate(operation_materializer_bridge.get("rows") or []):
            if not isinstance(row, dict):
                continue
            work_order = row.get("chain_compiler_work_order")
            if not isinstance(work_order, dict):
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
            chain_path = output_dir / "operation_chain_compiler_work_orders.json"
            write_json_artifact(chain_path, payload)
            artifacts["operation_chain_compiler_work_orders"] = _display_path(
                chain_path
            )
            chain_queue = build_frontier_operation_chain_compiler_queue(
                repo_root=REPO_ROOT,
                operation_chain_compiler_work_orders=payload,
                operation_chain_compiler_work_orders_path=chain_path,
                results_root=str(report.get("results_root") or DEFAULT_RESULTS_ROOT),
                queue_id=f"{report.get('queue_id') or 'frontier_feedback'}_chain_compiler",
                dqs1_observation_source_paths=tuple(
                    report.get("dqs1_observation_source_paths") or ()
                ),
            )
            if isinstance(chain_queue, dict):
                chain_queue_path = output_dir / "operation_chain_compiler_queue.json"
                write_json_artifact(chain_queue_path, chain_queue)
                artifacts["operation_chain_compiler_queue"] = _display_path(
                    chain_queue_path
                )
        bridge_artifacts = (
            ("operation_materializer_backlog", "materializer_backlog"),
            ("operation_materializer_contexts", "materializer_contexts"),
            ("operation_materializer_work_queue", "materializer_work_queue"),
        )
        for artifact_name, bridge_key in bridge_artifacts:
            payload = operation_materializer_bridge.get(bridge_key)
            if isinstance(payload, dict):
                artifact_path = output_dir / f"{artifact_name}.json"
                write_json_artifact(artifact_path, payload)
                artifacts[artifact_name] = _display_path(artifact_path)
        operation_work_queue = operation_materializer_bridge.get("materializer_work_queue")
        if isinstance(operation_work_queue, dict):
            operation_work_queue_path = output_dir / "operation_materializer_work_queue.json"
            operation_execution_queue = (
                build_frontier_materializer_execution_queue_if_available(
                    repo_root=REPO_ROOT,
                    materializer_work_queue=operation_work_queue,
                    materializer_work_queue_path=operation_work_queue_path,
                    results_root=str(report.get("results_root") or DEFAULT_RESULTS_ROOT),
                    queue_id=(
                        f"{report.get('queue_id') or 'frontier_feedback'}_"
                        "operation_materializer_execution"
                    ),
                    candidate_limit=int(report.get("candidate_limit") or 4),
                )
            )
            if isinstance(operation_execution_queue, dict):
                operation_execution_queue_path = (
                    output_dir / "operation_materializer_execution_queue.json"
                )
                write_json_artifact(
                    operation_execution_queue_path,
                    operation_execution_queue,
                )
                artifacts["operation_materializer_execution_queue"] = _display_path(
                    operation_execution_queue_path
                )
    receiver_repair_backlog = report.get("receiver_repair_backlog")
    if isinstance(receiver_repair_backlog, dict):
        path = output_dir / "receiver_repair_backlog.json"
        write_json_artifact(path, receiver_repair_backlog)
        artifacts["receiver_repair_backlog"] = _display_path(path)
        repair_queue = build_frontier_receiver_repair_queue(
            repo_root=REPO_ROOT,
            receiver_repair_backlog=receiver_repair_backlog,
            receiver_repair_backlog_path=path,
            results_root=str(report.get("results_root") or DEFAULT_RESULTS_ROOT),
            queue_id=f"{report.get('queue_id') or 'frontier_feedback'}_receiver_repair",
        )
        if isinstance(repair_queue, dict):
            queue_path = output_dir / "receiver_repair_queue.json"
            write_json_artifact(queue_path, repair_queue)
            artifacts["receiver_repair_queue"] = _display_path(queue_path)
    repair_dynamics_prior = report.get("repair_dynamics_palette_prior")
    if isinstance(repair_dynamics_prior, dict) and repair_dynamics_prior:
        path = output_dir / "repair_dynamics_palette_prior.json"
        write_json_artifact(path, repair_dynamics_prior)
        artifacts["repair_dynamics_palette_prior"] = _display_path(path)
    receiver_closed_budget = report.get("receiver_closed_correction_budget")
    if isinstance(receiver_closed_budget, dict):
        path = output_dir / "receiver_closed_correction_budget.json"
        write_json_artifact(path, receiver_closed_budget)
        artifacts["receiver_closed_correction_budget"] = _display_path(path)
    targeted_component_correction = report.get(
        "targeted_component_correction_acquisition"
    )
    if isinstance(targeted_component_correction, dict):
        path = output_dir / "targeted_component_correction_acquisition.json"
        write_json_artifact(path, targeted_component_correction)
        artifacts["targeted_component_correction_acquisition"] = _display_path(path)
        correction_queue = build_frontier_targeted_component_correction_queue(
            repo_root=REPO_ROOT,
            targeted_component_correction_acquisition=targeted_component_correction,
            targeted_component_correction_acquisition_path=path,
            results_root=str(report.get("results_root") or DEFAULT_RESULTS_ROOT),
            queue_id=(
                f"{report.get('queue_id') or 'frontier_feedback'}_"
                "component_correction"
            ),
            candidate_limit=int(report.get("candidate_limit") or 4),
            component_response_cache_roots=tuple(
                report.get("component_response_cache_roots") or ()
            ),
        )
        if isinstance(correction_queue, dict):
            queue_path = output_dir / "targeted_component_correction_queue.json"
            write_json_artifact(queue_path, correction_queue)
            artifacts["targeted_component_correction_queue"] = _display_path(
                queue_path
            )
            response_harvest = (
                build_frontier_targeted_component_correction_response_harvest(
                    repo_root=REPO_ROOT,
                    targeted_component_correction_queue=correction_queue,
                    results_root=str(report.get("results_root") or DEFAULT_RESULTS_ROOT),
                )
            )
            report["targeted_component_correction_response_harvest"] = response_harvest
            response_harvest_path = (
                output_dir / "targeted_component_correction_response_harvest.json"
            )
            write_json_artifact(response_harvest_path, response_harvest)
            artifacts["targeted_component_correction_response_harvest"] = (
                _display_path(response_harvest_path)
            )
            materialization_requests = (
                build_frontier_targeted_component_correction_materialization_requests(
                    targeted_component_correction_response_harvest=response_harvest,
                    candidate_limit=int(report.get("candidate_limit") or 4),
                )
            )
            report[
                "targeted_component_correction_materialization_requests"
            ] = materialization_requests
            materialization_requests_path = (
                output_dir
                / "targeted_component_correction_materialization_requests.json"
            )
            write_json_artifact(materialization_requests_path, materialization_requests)
            artifacts[
                "targeted_component_correction_materialization_requests"
            ] = _display_path(materialization_requests_path)
            materialization_queue = (
                build_frontier_targeted_component_correction_materialization_queue(
                    repo_root=REPO_ROOT,
                    targeted_component_correction_response_harvest=response_harvest,
                    targeted_component_correction_response_harvest_path=(
                        response_harvest_path
                    ),
                    results_root=str(report.get("results_root") or DEFAULT_RESULTS_ROOT),
                    queue_id=(
                        f"{report.get('queue_id') or 'frontier_feedback'}_"
                        "component_materialization"
                    ),
                    candidate_limit=int(report.get("candidate_limit") or 4),
                )
            )
            if isinstance(materialization_queue, dict):
                materialization_queue_path = (
                    output_dir
                    / "targeted_component_correction_materialization_queue.json"
                )
                write_json_artifact(materialization_queue_path, materialization_queue)
                artifacts[
                    "targeted_component_correction_materialization_queue"
                ] = _display_path(materialization_queue_path)
            targeted_chain_work_orders = (
                build_frontier_targeted_component_correction_chain_work_orders(
                    targeted_component_correction_materialization_requests=(
                        materialization_requests
                    ),
                    request_limit=int(report.get("candidate_limit") or 4),
                )
            )
            report[
                "targeted_component_correction_operation_chain_work_orders"
            ] = targeted_chain_work_orders
            targeted_chain_work_orders_path = (
                output_dir
                / "targeted_component_correction_operation_chain_work_orders.json"
            )
            write_json_artifact(
                targeted_chain_work_orders_path,
                targeted_chain_work_orders,
            )
            artifacts[
                "targeted_component_correction_operation_chain_work_orders"
            ] = _display_path(targeted_chain_work_orders_path)
            targeted_chain_queue = build_frontier_operation_chain_compiler_queue(
                repo_root=REPO_ROOT,
                operation_chain_compiler_work_orders=targeted_chain_work_orders,
                operation_chain_compiler_work_orders_path=(
                    targeted_chain_work_orders_path
                ),
                results_root=str(report.get("results_root") or DEFAULT_RESULTS_ROOT),
                queue_id=(
                    f"{report.get('queue_id') or 'frontier_feedback'}_"
                    "component_operation_chain"
                ),
                candidate_limit=int(report.get("candidate_limit") or 4),
                dqs1_observation_source_paths=tuple(
                    report.get("dqs1_observation_source_paths") or ()
                ),
            )
            if isinstance(targeted_chain_queue, dict):
                targeted_chain_queue_path = (
                    output_dir / "targeted_component_correction_operation_chain_queue.json"
                )
                write_json_artifact(targeted_chain_queue_path, targeted_chain_queue)
                artifacts[
                    "targeted_component_correction_operation_chain_queue"
                ] = _display_path(targeted_chain_queue_path)
                report["targeted_drop_many_dqs1_child_queue_paths"] = (
                    _targeted_dqs1_child_queue_paths(targeted_chain_queue)
                )
            targeted_chain_materializer_handoff = (
                build_frontier_targeted_component_correction_chain_materializer_handoff(
                    repo_root=REPO_ROOT,
                    targeted_component_correction_chain_work_orders=(
                        targeted_chain_work_orders
                    ),
                    default_output_root=(
                        Path(str(report.get("results_root") or DEFAULT_RESULTS_ROOT))
                        / "frontier_targeted_component_correction_chain_materializers"
                    ),
                )
            )
            report[
                "targeted_component_correction_chain_materializer_handoff"
            ] = targeted_chain_materializer_handoff
            targeted_chain_materializer_handoff_path = (
                output_dir
                / "targeted_component_correction_chain_materializer_handoff.json"
            )
            write_json_artifact(
                targeted_chain_materializer_handoff_path,
                targeted_chain_materializer_handoff,
            )
            artifacts[
                "targeted_component_correction_chain_materializer_handoff"
            ] = _display_path(targeted_chain_materializer_handoff_path)
            targeted_chain_work_queue = targeted_chain_materializer_handoff.get(
                "materializer_work_queue"
            )
            if isinstance(targeted_chain_work_queue, dict):
                targeted_chain_work_queue_path = (
                    output_dir
                    / "targeted_component_correction_chain_materializer_work_queue.json"
                )
                write_json_artifact(
                    targeted_chain_work_queue_path,
                    targeted_chain_work_queue,
                )
                artifacts[
                    "targeted_component_correction_chain_materializer_work_queue"
                ] = _display_path(targeted_chain_work_queue_path)
                targeted_execution_queue = (
                    build_frontier_materializer_execution_queue_if_available(
                        repo_root=REPO_ROOT,
                        materializer_work_queue=targeted_chain_work_queue,
                        materializer_work_queue_path=targeted_chain_work_queue_path,
                        results_root=str(
                            report.get("results_root") or DEFAULT_RESULTS_ROOT
                        ),
                        queue_id=(
                            f"{report.get('queue_id') or 'frontier_feedback'}_"
                            "targeted_chain_materializer_execution"
                        ),
                        candidate_limit=int(report.get("candidate_limit") or 4),
                    )
                )
                if isinstance(targeted_execution_queue, dict):
                    targeted_execution_queue_path = (
                        output_dir
                        / (
                            "targeted_component_correction_chain_materializer_"
                            "execution_queue.json"
                        )
                    )
                    write_json_artifact(
                        targeted_execution_queue_path,
                        targeted_execution_queue,
                    )
                    artifacts[
                        "targeted_component_correction_chain_materializer_execution_queue"
                    ] = _display_path(targeted_execution_queue_path)
            attach_frontier_autonomous_chain_optimization(
                report,
                targeted_component_correction_chain_materializer_handoff=(
                    targeted_chain_materializer_handoff
                ),
            )
    autonomous_chain_optimization = report.get("autonomous_chain_optimization")
    if isinstance(autonomous_chain_optimization, dict):
        path = output_dir / "autonomous_chain_optimization.json"
        write_json_artifact(path, autonomous_chain_optimization)
        artifacts["autonomous_chain_optimization"] = _display_path(path)
        response_harvest = report.get("targeted_component_correction_response_harvest")
        receiver_closed_budget = report.get("receiver_closed_correction_budget")
        repair_waterfill_queue = build_frontier_repair_budget_waterfill_queue(
            repo_root=REPO_ROOT,
            autonomous_chain_optimization=autonomous_chain_optimization,
            autonomous_chain_optimization_path=path,
            targeted_component_correction_response_harvest=(
                response_harvest if isinstance(response_harvest, dict) else None
            ),
            targeted_component_correction_response_harvest_path=artifacts.get(
                "targeted_component_correction_response_harvest"
            ),
            receiver_closed_correction_budget=(
                receiver_closed_budget
                if isinstance(receiver_closed_budget, dict)
                else None
            ),
            receiver_closed_correction_budget_path=artifacts.get(
                "receiver_closed_correction_budget"
            ),
            materializer_work_queue=(
                operation_work_queue if isinstance(operation_work_queue, dict) else None
            ),
            materializer_work_queue_path=artifacts.get(
                "operation_materializer_work_queue"
            ),
            materializer_execution_queue=(
                operation_execution_queue
                if isinstance(operation_execution_queue, dict)
                else None
            ),
            materializer_execution_queue_path=artifacts.get(
                "operation_materializer_execution_queue"
            ),
            repair_dynamics_palette_prior=(
                report.get("repair_dynamics_palette_prior")
                if isinstance(report.get("repair_dynamics_palette_prior"), dict)
                else None
            ),
            results_root=str(report.get("results_root") or DEFAULT_RESULTS_ROOT),
            queue_id=(
                f"{report.get('queue_id') or 'frontier_feedback'}_"
                "repair_budget_waterfill"
            ),
            chain_limit=int(report.get("candidate_limit") or 4),
        )
        if isinstance(repair_waterfill_queue, dict):
            repair_waterfill_queue_path = output_dir / "repair_budget_waterfill_queue.json"
            write_json_artifact(
                repair_waterfill_queue_path,
                repair_waterfill_queue,
            )
            artifacts["repair_budget_waterfill_queue"] = _display_path(
                repair_waterfill_queue_path
            )
        autonomous_queue = build_frontier_autonomous_chain_optimization_queue(
            repo_root=REPO_ROOT,
            autonomous_chain_optimization=autonomous_chain_optimization,
            autonomous_chain_optimization_path=path,
            artifact_paths_by_key=artifacts,
            results_root=str(report.get("results_root") or DEFAULT_RESULTS_ROOT),
            queue_id=(
                f"{report.get('queue_id') or 'frontier_feedback'}_"
                "autonomous_chain_optimization"
            ),
            chain_limit=int(report.get("candidate_limit") or 4),
        )
        if isinstance(autonomous_queue, dict):
            queue_path = output_dir / "autonomous_chain_optimization_queue.json"
            write_json_artifact(queue_path, autonomous_queue)
            artifacts["autonomous_chain_optimization_queue"] = _display_path(
                queue_path
            )
    bridge = report.get("materializer_feedback_bridge")
    if isinstance(bridge, dict):
        path = output_dir / "materializer_feedback_bridge.json"
        write_json_artifact(path, bridge)
        artifacts["materializer_feedback_bridge"] = _display_path(path)
    queue = report.get("queue")
    if isinstance(queue, dict):
        path = output_dir / "dqs1_followup_queue.json"
        write_json_artifact(path, queue)
        artifacts["dqs1_followup_queue"] = _display_path(path)

    report_path = output_dir / "feedback_refresh_report.json"
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
                "init_followup_queue": [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    artifacts["dqs1_followup_queue"],
                    "init",
                ],
                "run_frontier_feedback_cycle": [
                    ".venv/bin/python",
                    "tools/run_frontier_rate_attack_feedback_cycle.py",
                    "--action-summary",
                    str(report.get("action_summary_path") or "latest"),
                    "--results-root",
                    str(report.get("results_root") or DEFAULT_RESULTS_ROOT),
                    "--output-dir",
                    _display_path(output_dir.parent / f"{output_dir.name}_cycle"),
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
    if "targeted_component_correction_response_harvest" in artifacts:
        operator_commands["inspect_targeted_component_correction_response_harvest"] = [
            ".venv/bin/python",
            "-m",
            "json.tool",
            artifacts["targeted_component_correction_response_harvest"],
        ]
    if "targeted_component_correction_materialization_requests" in artifacts:
        operator_commands[
            "inspect_targeted_component_correction_materialization_requests"
        ] = [
            ".venv/bin/python",
            "-m",
            "json.tool",
            artifacts["targeted_component_correction_materialization_requests"],
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
        operator_commands["init_targeted_component_correction_materialization_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_materialization_queue"],
            "init",
        ]
        operator_commands["status_targeted_component_correction_materialization_queue"] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_materialization_queue"],
            "status",
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
            "6",
            "--max-experiments",
            "2",
            "--max-parallel",
            "2",
        ]
    if "targeted_component_correction_operation_chain_work_orders" in artifacts:
        operator_commands[
            "inspect_targeted_component_correction_operation_chain_work_orders"
        ] = [
            ".venv/bin/python",
            "-m",
            "json.tool",
            artifacts["targeted_component_correction_operation_chain_work_orders"],
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
            "init_targeted_component_correction_operation_chain_queue"
        ] = [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            artifacts["targeted_component_correction_operation_chain_queue"],
            "init",
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
            "16",
            "--max-experiments",
            "2",
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
    artifacts["feedback_refresh_report"] = _display_path(report_path)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = _utc_stamp()
    queue_id = args.queue_id or f"frontier_rate_attack_feedback_dqs1_followup_{stamp}"
    output_dir = args.output_dir or (
        REPO_ROOT / ".omx" / "research" / f"frontier_rate_attack_feedback_refresh_{stamp}"
    )
    try:
        action_summary_path = _action_summary_path(args.action_summary)
        generated_pair_frame_lattice, generated_pair_frame_lattice_summary = (
            _build_generated_pair_frame_lattice(
                output_dir=output_dir,
                pairset_acquisition_path=args.pair_frame_pairset_acquisition,
                frame_pair_curriculum_path=args.pair_frame_curriculum,
                pair_component_xray_paths=tuple(args.pair_component_xray),
                drop_counts=args.pair_frame_drop_counts,
                max_requests=args.pair_frame_max_requests,
            )
        )
        pair_frame_geometry_paths = [
            *args.pair_frame_geometry_lattice,
            *(
                [generated_pair_frame_lattice]
                if generated_pair_frame_lattice is not None
                else []
            ),
        ]
        repair_dynamics_priors, repair_dynamics_source_paths = (
            _load_repair_dynamics_priors(args.repair_dynamics_prior)
        )
        repair_palette_modes = _repair_palette_modes_from_args(args)
        report = build_frontier_rate_attack_feedback_refresh(
            repo_root=REPO_ROOT,
            frontier_artifact_roots=tuple(args.frontier_artifact_root),
            local_cpu_eureka_roots=tuple(args.local_cpu_eureka_root),
            materializer_feedback_paths=tuple(args.materializer_feedback),
            pair_frame_geometry_paths=tuple(pair_frame_geometry_paths),
            dqs1_observation_paths=tuple(args.dqs1_observation_jsonl),
            max_files_per_root=args.max_files_per_root,
            action_summary_path=action_summary_path,
            results_root=args.results_root,
            queue_id=queue_id,
            candidate_limit=args.candidate_limit,
            skip_observed_dqs1_candidates=not args.include_observed_dqs1_candidate,
            local_cpu_concurrency=args.local_cpu_concurrency,
            local_io_concurrency=args.local_io_concurrency,
            include_raw_retention_plan=not args.skip_raw_retention_plan,
            raw_retention_execute=args.execute_raw_retention,
            raw_retention_action=args.raw_retention_action,
            raw_retention_cold_store_roots=tuple(args.raw_retention_cold_store_root),
            raw_retention_cold_store_reserve_gb=(
                args.raw_retention_cold_store_reserve_gb
            ),
            include_mlx_retention_plan=not args.skip_mlx_retention_plan,
            mlx_retention_execute=args.execute_mlx_retention,
            mlx_retention_action=args.mlx_retention_action,
            mlx_retention_cold_store_roots=tuple(args.mlx_retention_cold_store_root),
            mlx_retention_cold_store_reserve_gb=args.mlx_retention_cold_store_reserve_gb,
            repair_palette_modes=tuple(repair_palette_modes),
            repair_dynamics_palette_priors=tuple(repair_dynamics_priors),
            repair_dynamics_prior_source_paths=tuple(repair_dynamics_source_paths),
            component_response_cache_roots=tuple(args.component_response_cache_root),
        )
        if generated_pair_frame_lattice is not None:
            report["generated_pair_frame_geometry_lattice_path"] = _display_path(
                generated_pair_frame_lattice
            )
            report["generated_pair_frame_geometry_lattice"] = (
                generated_pair_frame_lattice_summary
            )
        artifacts = _write_outputs(output_dir, report)
    except (
        ArtifactWriteError,
        ExperimentQueueError,
        FrontierRateAttackFeedbackError,
        PairFrameScorerGeometryLatticeError,
        OSError,
    ) as exc:
        print(f"FATAL: frontier rate-attack feedback refresh failed: {exc}", file=sys.stderr)
        return 2

    print(
        json_text(
            {
                "schema": "frontier_rate_attack_feedback_refresh_cli_result.v1",
                "queue_id": report.get("queue_id"),
                "output_dir": _display_path(output_dir),
                "artifacts": artifacts,
                "materializer_feedback_payload_count": report.get(
                    "materializer_feedback_payload_count"
                ),
                "pair_frame_geometry_queue_request_count": report.get(
                    "pair_frame_geometry_queue_request_count"
                ),
                "generated_pair_frame_geometry_lattice": report.get(
                    "generated_pair_frame_geometry_lattice"
                ),
                "dqs1_observation_count": report.get("dqs1_observation_count"),
                "selected_candidate_ids": report.get("selected_candidate_ids"),
                "queue_summary": report.get("queue_summary"),
                "operation_portfolio_summary": {
                    "operation_count": (
                        report.get("operation_portfolio", {}).get("operation_count")
                        if isinstance(report.get("operation_portfolio"), dict)
                        else None
                    ),
                    "queue_executable_operation_count": (
                        report.get("operation_portfolio", {}).get(
                            "queue_executable_operation_count"
                        )
                        if isinstance(report.get("operation_portfolio"), dict)
                        else None
                    ),
                    "followup_signal_operation_count": (
                        report.get("operation_portfolio", {}).get(
                            "followup_signal_operation_count"
                        )
                        if isinstance(report.get("operation_portfolio"), dict)
                        else None
                    ),
                    "top_operation_ids": (
                        report.get("operation_portfolio", {}).get("top_operation_ids")
                        if isinstance(report.get("operation_portfolio"), dict)
                        else None
                    ),
                    "top_queue_executable_operation_ids": (
                        report.get("operation_portfolio", {}).get(
                            "top_queue_executable_operation_ids"
                        )
                        if isinstance(report.get("operation_portfolio"), dict)
                        else None
                    ),
                },
                "receiver_repair_backlog_summary": {
                    "row_count": (
                        report.get("receiver_repair_backlog", {}).get("row_count")
                        if isinstance(report.get("receiver_repair_backlog"), dict)
                        else None
                    ),
                    "queue_actionable_repair_count": (
                        report.get("receiver_repair_backlog", {}).get(
                            "queue_actionable_repair_count"
                        )
                        if isinstance(report.get("receiver_repair_backlog"), dict)
                        else None
                    ),
                    "top_repair_ids": (
                        report.get("receiver_repair_backlog", {}).get(
                            "top_repair_ids"
                        )
                        if isinstance(report.get("receiver_repair_backlog"), dict)
                        else None
                    ),
                    "top_repair_families": (
                        report.get("receiver_repair_backlog", {}).get(
                            "top_repair_families"
                        )
                        if isinstance(report.get("receiver_repair_backlog"), dict)
                        else None
                    ),
                },
                "receiver_closed_correction_budget_summary": {
                    "active": (
                        report.get("receiver_closed_correction_budget", {}).get("active")
                        if isinstance(
                            report.get("receiver_closed_correction_budget"), dict
                        )
                        else None
                    ),
                    "receiver_closed_candidate_count": (
                        report.get("receiver_closed_correction_budget", {}).get(
                            "receiver_closed_candidate_count"
                        )
                        if isinstance(
                            report.get("receiver_closed_correction_budget"), dict
                        )
                        else None
                    ),
                    "receiver_closed_saved_bytes_total": (
                        report.get("receiver_closed_correction_budget", {}).get(
                            "receiver_closed_saved_bytes_total"
                        )
                        if isinstance(
                            report.get("receiver_closed_correction_budget"), dict
                        )
                        else None
                    ),
                },
                "repair_dynamics_prior_summary": {
                    "repair_dynamics_palette_prior_present": (
                        bool(report.get("repair_dynamics_palette_prior"))
                        if isinstance(report.get("repair_dynamics_palette_prior"), dict)
                        else False
                    ),
                    "repair_dynamics_prior_source_count": len(
                        report.get("repair_dynamics_prior_source_paths") or []
                    ),
                    "repair_dynamics_prior_defaulted": False,
                    "mode_count": (
                        report.get("repair_dynamics_palette_prior", {}).get("mode_count")
                        if isinstance(report.get("repair_dynamics_palette_prior"), dict)
                        else None
                    ),
                    "zero_frame1_modes": (
                        report.get("repair_dynamics_palette_prior", {}).get(
                            "zero_frame1_modes"
                        )
                        if isinstance(report.get("repair_dynamics_palette_prior"), dict)
                        else None
                    ),
                },
                "targeted_component_correction_acquisition_summary": {
                    "active": (
                        report.get(
                            "targeted_component_correction_acquisition", {}
                        ).get("active")
                        if isinstance(
                            report.get("targeted_component_correction_acquisition"),
                            dict,
                        )
                        else None
                    ),
                    "row_count": (
                        report.get(
                            "targeted_component_correction_acquisition", {}
                        ).get("row_count")
                        if isinstance(
                            report.get("targeted_component_correction_acquisition"),
                            dict,
                        )
                        else None
                    ),
                    "queue_actionable_acquisition_count": (
                        report.get(
                            "targeted_component_correction_acquisition", {}
                        ).get("queue_actionable_acquisition_count")
                        if isinstance(
                            report.get("targeted_component_correction_acquisition"),
                            dict,
                        )
                        else None
                    ),
                    "receiver_closed_saved_bytes_total": (
                        report.get(
                            "targeted_component_correction_acquisition", {}
                        ).get("receiver_closed_saved_bytes_total")
                        if isinstance(
                            report.get("targeted_component_correction_acquisition"),
                            dict,
                        )
                        else None
                    ),
                    "top_acquisition_ids": (
                        report.get(
                            "targeted_component_correction_acquisition", {}
                        ).get("top_acquisition_ids")
                        if isinstance(
                            report.get("targeted_component_correction_acquisition"),
                            dict,
                        )
                        else None
                    ),
                    "repair_dynamics_prior_active": (
                        report.get(
                            "targeted_component_correction_acquisition", {}
                        ).get("repair_dynamics_prior_active")
                        if isinstance(
                            report.get("targeted_component_correction_acquisition"),
                            dict,
                        )
                        else None
                    ),
                    "repair_dynamics_palette_probe_count": (
                        report.get(
                            "targeted_component_correction_acquisition", {}
                        ).get("repair_dynamics_palette_probe_count")
                        if isinstance(
                            report.get("targeted_component_correction_acquisition"),
                            dict,
                        )
                        else None
                    ),
                },
                "targeted_component_correction_response_harvest_summary": {
                    "active": (
                        report.get(
                            "targeted_component_correction_response_harvest", {}
                        ).get("active")
                        if isinstance(
                            report.get("targeted_component_correction_response_harvest"),
                            dict,
                        )
                        else None
                    ),
                    "row_count": (
                        report.get(
                            "targeted_component_correction_response_harvest", {}
                        ).get("row_count")
                        if isinstance(
                            report.get("targeted_component_correction_response_harvest"),
                            dict,
                        )
                        else None
                    ),
                    "local_acquisition_recommended_count": (
                        report.get(
                            "targeted_component_correction_response_harvest", {}
                        ).get("local_acquisition_recommended_count")
                        if isinstance(
                            report.get("targeted_component_correction_response_harvest"),
                            dict,
                        )
                        else None
                    ),
                    "blocked_response_count": (
                        report.get(
                            "targeted_component_correction_response_harvest", {}
                        ).get("blocked_response_count")
                        if isinstance(
                            report.get("targeted_component_correction_response_harvest"),
                            dict,
                        )
                        else None
                    ),
                },
                "targeted_component_correction_materialization_request_summary": {
                    "active": (
                        report.get(
                            "targeted_component_correction_materialization_requests",
                            {},
                        ).get("active")
                        if isinstance(
                            report.get(
                                "targeted_component_correction_materialization_requests"
                            ),
                            dict,
                        )
                        else None
                    ),
                    "row_count": (
                        report.get(
                            "targeted_component_correction_materialization_requests",
                            {},
                        ).get("row_count")
                        if isinstance(
                            report.get(
                                "targeted_component_correction_materialization_requests"
                            ),
                            dict,
                        )
                        else None
                    ),
                    "accepted_response_count": (
                        report.get(
                            "targeted_component_correction_materialization_requests",
                            {},
                        ).get("accepted_response_count")
                        if isinstance(
                            report.get(
                                "targeted_component_correction_materialization_requests"
                            ),
                            dict,
                        )
                        else None
                    ),
                    "ready_for_budget_spend_count": (
                        report.get(
                            "targeted_component_correction_materialization_requests",
                            {},
                        ).get("ready_for_budget_spend_count")
                        if isinstance(
                            report.get(
                                "targeted_component_correction_materialization_requests"
                            ),
                            dict,
                        )
                        else None
                    ),
                },
                "targeted_component_correction_operation_chain_summary": {
                    "active": (
                        report.get(
                            "targeted_component_correction_operation_chain_work_orders",
                            {},
                        ).get("active")
                        if isinstance(
                            report.get(
                                "targeted_component_correction_operation_chain_work_orders"
                            ),
                            dict,
                        )
                        else None
                    ),
                    "work_order_count": (
                        report.get(
                            "targeted_component_correction_operation_chain_work_orders",
                            {},
                        ).get("work_order_count")
                        if isinstance(
                            report.get(
                                "targeted_component_correction_operation_chain_work_orders"
                            ),
                            dict,
                        )
                        else None
                    ),
                    "request_count": (
                        report.get(
                            "targeted_component_correction_operation_chain_work_orders",
                            {},
                        ).get("request_count")
                        if isinstance(
                            report.get(
                                "targeted_component_correction_operation_chain_work_orders"
                            ),
                            dict,
                        )
                        else None
                    ),
                },
                "targeted_component_correction_chain_materializer_handoff_summary": {
                    "work_queue_row_count": (
                        report.get(
                            "targeted_component_correction_chain_materializer_handoff",
                            {},
                        ).get("work_queue_row_count")
                        if isinstance(
                            report.get(
                                "targeted_component_correction_chain_materializer_handoff"
                            ),
                            dict,
                        )
                        else None
                    ),
                    "executable_work_row_count": (
                        report.get(
                            "targeted_component_correction_chain_materializer_handoff",
                            {},
                        ).get("executable_work_row_count")
                        if isinstance(
                            report.get(
                                "targeted_component_correction_chain_materializer_handoff"
                            ),
                            dict,
                        )
                        else None
                    ),
                    "unregistered_chain_target_count": (
                        report.get(
                            "targeted_component_correction_chain_materializer_handoff",
                            {},
                        ).get("unregistered_chain_target_count")
                        if isinstance(
                            report.get(
                                "targeted_component_correction_chain_materializer_handoff"
                            ),
                            dict,
                        )
                        else None
                    ),
                },
                "autonomous_chain_optimization_summary": {
                    "chain_count": (
                        report.get("autonomous_chain_optimization", {}).get(
                            "chain_count"
                        )
                        if isinstance(
                            report.get("autonomous_chain_optimization"),
                            dict,
                        )
                        else None
                    ),
                    "top_chain_ids": (
                        report.get("autonomous_chain_optimization", {}).get(
                            "top_chain_ids"
                        )
                        if isinstance(
                            report.get("autonomous_chain_optimization"),
                            dict,
                        )
                        else None
                    ),
                    "target_classes": (
                        report.get("autonomous_chain_optimization", {}).get(
                            "target_classes"
                        )
                        if isinstance(
                            report.get("autonomous_chain_optimization"),
                            dict,
                        )
                        else None
                    ),
                    "registered_target_count": (
                        report.get("autonomous_chain_optimization", {}).get(
                            "registered_target_count"
                        )
                        if isinstance(
                            report.get("autonomous_chain_optimization"),
                            dict,
                        )
                        else None
                    ),
                },
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
