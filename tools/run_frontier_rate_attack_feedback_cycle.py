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
)
from comma_lab.scheduler.frontier_rate_attack_feedback_cycle import (  # noqa: E402
    FRONTIER_RATE_ATTACK_FEEDBACK_CYCLE_SCHEMA,
    FrontierRateAttackFeedbackCycleError,
    harvest_paths_from_autopilot_payload,
    harvest_paths_from_autopilot_result_files,
    json_text,
    repo_rel,
    resolve_repo_path,
    utc_stamp,
    write_cycle_report,
    write_dqs1_harvest_observation_bundle,
    write_frontier_refresh_artifacts,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY  # noqa: E402

DEFAULT_BASELINE_ADVISORY = (
    "experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/"
    "dqs1_top32_cpu_advisory_venv.json"
)
DEFAULT_BASELINE_ARCHIVE_SIZE_BYTES = 178_560
DEFAULT_BASELINE_CANDIDATE_ID = "dqs1_top32_gap_uleb"


def _display_path(path: str | Path) -> str:
    return repo_rel(path, REPO_ROOT)


def _action_summary_path(value: str) -> Path | None:
    if value == "none":
        return None
    if value == "latest":
        return find_latest_cross_family_action_summary(REPO_ROOT)
    return resolve_repo_path(value, repo_root=REPO_ROOT)


def _latest_pairset_acquisition(root: str | Path) -> Path:
    resolved = resolve_repo_path(root, repo_root=REPO_ROOT)
    search_roots = [resolved / "pairset_acquisition", resolved]
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


def _build_refresh(
    *,
    args: argparse.Namespace,
    queue_id: str,
    dqs1_observation_paths: tuple[str | Path, ...],
) -> dict[str, Any]:
    return build_frontier_rate_attack_feedback_refresh(
        repo_root=REPO_ROOT,
        frontier_artifact_roots=tuple(args.frontier_artifact_root),
        materializer_feedback_paths=tuple(args.materializer_feedback),
        dqs1_observation_paths=dqs1_observation_paths,
        action_summary_path=_action_summary_path(args.action_summary),
        results_root=args.results_root,
        queue_id=queue_id,
        candidate_limit=args.candidate_limit,
        skip_observed_dqs1_candidates=not args.include_observed_dqs1_candidate,
        local_cpu_concurrency=args.local_cpu_concurrency,
        local_io_concurrency=args.local_io_concurrency,
        include_raw_retention_plan=not args.skip_raw_retention_plan,
        include_mlx_retention_plan=not args.skip_mlx_retention_plan,
    )


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
        "--no-post-harvest-retention",
    ]
    if args.state is not None:
        command.extend(["--state", args.state])
    if args.execute_followup:
        command.append("--execute")
    return command


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
    parser.add_argument("--materializer-feedback", action="append", default=[])
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
    parser.add_argument("--local-io-concurrency", type=int, default=2)
    parser.add_argument("--skip-raw-retention-plan", action="store_true")
    parser.add_argument("--skip-mlx-retention-plan", action="store_true")
    parser.add_argument(
        "--execute-followup",
        action="store_true",
        help="Run the generated follow-up queue locally with bounded autopilot.",
    )
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
    parser.add_argument("--min-free-disk-gb", type=float, default=40.0)
    parser.add_argument("--pairset-acquisition", default="latest")
    parser.add_argument("--pairset-acquisition-root", default=DEFAULT_RESULTS_ROOT)
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

    try:
        initial_observations = tuple(args.dqs1_observation_jsonl)
        initial_report = _build_refresh(
            args=args,
            queue_id=queue_id,
            dqs1_observation_paths=initial_observations,
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

        followup: dict[str, Any] = {
            "execute_followup": bool(args.execute_followup),
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
        post_report = None
        post_artifacts: dict[str, str] = {}
        post_validate = None
        if harvest_paths:
            pairset_acquisition = _pairset_acquisition_path(
                args.pairset_acquisition,
                root=args.pairset_acquisition_root,
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
                *initial_observations,
                observation_bundle["observation_jsonl"],
            )
            post_report = _build_refresh(
                args=args,
                queue_id=post_queue_id,
                dqs1_observation_paths=post_observations,
            )
            post_artifacts = write_frontier_refresh_artifacts(
                output_dir=post_dir,
                report=post_report,
                repo_root=REPO_ROOT,
            )
            post_queue_path = post_artifacts.get("dqs1_followup_queue")
            if post_queue_path is not None:
                post_validate = _validate_queue(post_queue_path)

        payload = {
            "schema": FRONTIER_RATE_ATTACK_FEEDBACK_CYCLE_SCHEMA,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_dir": _display_path(output_dir),
            "initial_refresh": {
                "artifacts": initial_artifacts,
                "selected_candidate_ids": initial_report.get("selected_candidate_ids"),
                "queue_summary": initial_report.get("queue_summary"),
                "queue_validate": initial_validate,
                **FALSE_AUTHORITY,
            },
            "followup_execution": followup,
            "harvest_signal": {
                "harvest_path_count": len(harvest_paths),
                "harvest_paths": [_display_path(path) for path in harvest_paths],
                "observation_bundle": observation_bundle,
                **FALSE_AUTHORITY,
            },
            "post_harvest_refresh": None
            if post_report is None
            else {
                "artifacts": post_artifacts,
                "selected_candidate_ids": post_report.get("selected_candidate_ids"),
                "queue_summary": post_report.get("queue_summary"),
                "queue_validate": post_validate,
                **FALSE_AUTHORITY,
            },
            "integration_edges": [
                "family_materializer_feedback_to_dqs1_bridge",
                "dqs1_batch_followup_queue_to_local_autopilot",
                "local_autopilot_harvest_to_dynamic_observation_jsonl",
                "dynamic_observation_jsonl_to_refreshed_dqs1_queue",
            ],
            "allowed_use": "local_queue_owned_frontier_feedback_iteration_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
            **FALSE_AUTHORITY,
        }
        report_path = write_cycle_report(output_dir=output_dir, payload=payload)
    except (
        ExperimentQueueError,
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
                **FALSE_AUTHORITY,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
