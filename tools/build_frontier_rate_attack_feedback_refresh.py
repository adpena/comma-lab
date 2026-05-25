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
        "--dqs1-observation-jsonl",
        "--dqs1-observations",
        action="append",
        default=[],
        dest="dqs1_observation_jsonl",
        help="DQS1 local-first harvest observation JSONL. May repeat; rows are deduped.",
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


def _write_outputs(output_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
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
    if "dqs1_followup_queue" in artifacts:
        report_to_write["operator_commands"] = {
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
        report = build_frontier_rate_attack_feedback_refresh(
            repo_root=REPO_ROOT,
            frontier_artifact_roots=tuple(args.frontier_artifact_root),
            materializer_feedback_paths=tuple(args.materializer_feedback),
            pair_frame_geometry_paths=tuple(args.pair_frame_geometry_lattice),
            dqs1_observation_paths=tuple(args.dqs1_observation_jsonl),
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
        )
        artifacts = _write_outputs(output_dir, report)
    except (
        ArtifactWriteError,
        ExperimentQueueError,
        FrontierRateAttackFeedbackError,
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
                "dqs1_observation_count": report.get("dqs1_observation_count"),
                "selected_candidate_ids": report.get("selected_candidate_ids"),
                "queue_summary": report.get("queue_summary"),
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
