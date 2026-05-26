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
    FrontierRateAttackFeedbackError,
    build_frontier_targeted_component_correction_response_harvest,
    build_frontier_targeted_component_correction_response_harvest_from_artifacts,
)
from tac.repo_io import ArtifactWriteError, json_text, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-order", required=True, type=Path)
    parser.add_argument("--local-cpu-advisory", required=True, type=Path)
    parser.add_argument("--local-mlx-response", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args(argv)


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve(strict=False)
    try:
        work_order = _load_json(args.work_order)
        local_cpu_advisory = _load_json(args.local_cpu_advisory)
        local_mlx_response = (
            None
            if args.local_mlx_response is None
            else _load_json(args.local_mlx_response)
        )
        row = build_frontier_targeted_component_correction_response_harvest_from_artifacts(
            work_order=work_order,
            local_cpu_advisory=local_cpu_advisory,
            local_mlx_response=local_mlx_response,
            work_order_path=_display_path(args.work_order, repo_root=repo_root),
            local_cpu_advisory_path=_display_path(
                args.local_cpu_advisory,
                repo_root=repo_root,
            ),
            local_mlx_response_path=(
                None
                if args.local_mlx_response is None
                else _display_path(args.local_mlx_response, repo_root=repo_root)
            ),
            response_artifact_path=_display_path(args.output, repo_root=repo_root),
        )
        harvest = build_frontier_targeted_component_correction_response_harvest(
            repo_root=repo_root,
            response_rows=(row,),
        )
        write_json_artifact(args.output, harvest)
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
