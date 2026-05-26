#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest a targeted component-correction response into queue-owned signal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.frontier_rate_attack_feedback import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    FrontierRateAttackFeedbackError,
    build_frontier_targeted_component_correction_materialization_queue,
    build_frontier_targeted_component_correction_materialization_requests,
    build_frontier_targeted_component_correction_response_harvest,
    build_frontier_targeted_component_correction_response_harvest_from_artifacts,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-order", type=Path)
    parser.add_argument("--local-cpu-advisory", type=Path)
    parser.add_argument(
        "--targeted-component-correction-queue",
        type=Path,
        default=None,
        help=(
            "Aggregate response rows from an executed targeted component "
            "correction queue. Mutually exclusive with --work-order."
        ),
    )
    parser.add_argument("--reference-local-cpu-advisory", type=Path, default=None)
    parser.add_argument(
        "--reference-role",
        choices=("receiver_closed_source_reference", "correction_spend_reference"),
        default="receiver_closed_source_reference",
    )
    parser.add_argument("--local-mlx-response", type=Path, default=None)
    parser.add_argument("--reference-local-mlx-response", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--materialization-requests-output", type=Path, default=None)
    parser.add_argument("--materialization-queue-output", type=Path, default=None)
    parser.add_argument(
        "--materialization-queue-id",
        default="frontier_targeted_component_correction_materialization_queue",
    )
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--materialization-candidate-limit", type=int, default=4)
    parser.add_argument("--materialization-family-limit-per-candidate", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-existing-sha256", default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    if args.targeted_component_correction_queue is not None:
        if args.work_order is not None or args.local_cpu_advisory is not None:
            parser.error(
                "--targeted-component-correction-queue is mutually exclusive "
                "with --work-order/--local-cpu-advisory"
            )
        if (
            args.reference_local_cpu_advisory is not None
            or args.local_mlx_response is not None
            or args.reference_local_mlx_response is not None
        ):
            parser.error(
                "--reference-local-cpu-advisory/--local-mlx-response/"
                "--reference-local-mlx-response apply only to single-row "
                "--work-order harvests"
            )
        return args
    if args.work_order is None or args.local_cpu_advisory is None:
        parser.error(
            "single-row harvest requires --work-order and --local-cpu-advisory; "
            "queue aggregate harvest requires --targeted-component-correction-queue"
        )
    return args


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FrontierRateAttackFeedbackError(f"{path}: expected JSON object")
    return payload


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(
            repo_root.resolve(strict=False)
        ).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json_with_overwrite_guard(
    path: Path,
    payload: dict[str, Any],
    *,
    allow_overwrite: bool,
    expected_existing_sha256: str | None = None,
) -> None:
    expected_sha = expected_existing_sha256
    if path.exists() and allow_overwrite and expected_sha is None:
        expected_sha = sha256_file(path)
    write_json_artifact(
        path,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=expected_sha,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve(strict=False)
    materialization_requests: dict[str, Any] | None = None
    materialization_queue: dict[str, Any] | None = None
    try:
        if args.targeted_component_correction_queue is not None:
            queue = _load_json(args.targeted_component_correction_queue)
            harvest = build_frontier_targeted_component_correction_response_harvest(
                repo_root=repo_root,
                targeted_component_correction_queue=queue,
            )
        else:
            work_order = _load_json(args.work_order)
            local_cpu_advisory = _load_json(args.local_cpu_advisory)
            reference_local_cpu_advisory = (
                None
                if args.reference_local_cpu_advisory is None
                else _load_json(args.reference_local_cpu_advisory)
            )
            local_mlx_response = (
                None
                if args.local_mlx_response is None
                else _load_json(args.local_mlx_response)
            )
            reference_local_mlx_response = (
                None
                if args.reference_local_mlx_response is None
                else _load_json(args.reference_local_mlx_response)
            )
            row = (
                build_frontier_targeted_component_correction_response_harvest_from_artifacts(
                    work_order=work_order,
                    local_cpu_advisory=local_cpu_advisory,
                    reference_local_cpu_advisory=reference_local_cpu_advisory,
                    local_mlx_response=local_mlx_response,
                    reference_local_mlx_response=reference_local_mlx_response,
                    work_order_path=_display_path(args.work_order, repo_root=repo_root),
                    local_cpu_advisory_path=_display_path(
                        args.local_cpu_advisory,
                        repo_root=repo_root,
                    ),
                    reference_local_cpu_advisory_path=(
                        None
                        if args.reference_local_cpu_advisory is None
                        else _display_path(
                            args.reference_local_cpu_advisory,
                            repo_root=repo_root,
                        )
                    ),
                    local_mlx_response_path=(
                        None
                        if args.local_mlx_response is None
                        else _display_path(args.local_mlx_response, repo_root=repo_root)
                    ),
                    reference_local_mlx_response_path=(
                        None
                        if args.reference_local_mlx_response is None
                        else _display_path(
                            args.reference_local_mlx_response,
                            repo_root=repo_root,
                        )
                    ),
                    response_artifact_path=_display_path(
                        args.output,
                        repo_root=repo_root,
                    ),
                    reference_role=args.reference_role,
                )
            )
            harvest = build_frontier_targeted_component_correction_response_harvest(
                repo_root=repo_root,
                response_rows=(row,),
            )
        _write_json_with_overwrite_guard(
            args.output,
            harvest,
            allow_overwrite=args.overwrite,
            expected_existing_sha256=args.expected_existing_sha256,
        )
        if args.materialization_requests_output is not None:
            materialization_requests = (
                build_frontier_targeted_component_correction_materialization_requests(
                    targeted_component_correction_response_harvest=harvest,
                    candidate_limit=args.materialization_candidate_limit,
                    family_limit_per_candidate=(
                        args.materialization_family_limit_per_candidate
                    ),
                )
            )
            _write_json_with_overwrite_guard(
                args.materialization_requests_output,
                materialization_requests,
                allow_overwrite=args.overwrite,
            )
        if args.materialization_queue_output is not None:
            materialization_queue = (
                build_frontier_targeted_component_correction_materialization_queue(
                    repo_root=repo_root,
                    targeted_component_correction_response_harvest=harvest,
                    targeted_component_correction_response_harvest_path=args.output,
                    results_root=args.results_root,
                    queue_id=args.materialization_queue_id,
                    candidate_limit=args.materialization_candidate_limit,
                    family_limit_per_candidate=(
                        args.materialization_family_limit_per_candidate
                    ),
                )
            )
            if materialization_queue is not None:
                _write_json_with_overwrite_guard(
                    args.materialization_queue_output,
                    materialization_queue,
                    allow_overwrite=args.overwrite,
                )
    except (
        ArtifactWriteError,
        FrontierRateAttackFeedbackError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"FATAL: targeted component-correction response harvest failed: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        json_text(
            {
                "schema": "frontier_targeted_component_correction_response_harvest_cli_result.v1",
                "output": _display_path(args.output, repo_root=repo_root),
                "row_count": harvest.get("row_count"),
                "local_acquisition_recommended_count": harvest.get(
                    "local_acquisition_recommended_count"
                ),
                "ready_for_budget_spend_count": harvest.get(
                    "ready_for_budget_spend_count"
                ),
                "materialization_requests_output": (
                    None
                    if args.materialization_requests_output is None
                    else _display_path(
                        args.materialization_requests_output,
                        repo_root=repo_root,
                    )
                ),
                "materialization_request_count": (
                    None
                    if materialization_requests is None
                    else materialization_requests.get("row_count")
                ),
                "materialization_queue_output": (
                    None
                    if args.materialization_queue_output is None
                    else _display_path(
                        args.materialization_queue_output,
                        repo_root=repo_root,
                    )
                ),
                "materialization_queue_experiment_count": (
                    None
                    if materialization_queue is None
                    else len(materialization_queue.get("experiments") or [])
                ),
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
