#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bind repair-budget parent/child rows to materializer execution manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.frontier_rate_attack_feedback import (  # noqa: E402
    FrontierRateAttackFeedbackError,
    build_frontier_repair_budget_materializer_binding_report,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-plan", required=True, type=Path)
    parser.add_argument("--binding-report-out", required=True, type=Path)
    parser.add_argument("--materializer-work-queue", type=Path)
    parser.add_argument("--materializer-execution-queue", type=Path)
    parser.add_argument("--materializer-manifest", action="append", type=Path, default=[])
    parser.add_argument("--repair-palette-mode", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FrontierRateAttackFeedbackError(f"{path} must contain a JSON object")
    return payload


def _optional_json(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = _resolve_repo_path(path)
    if not resolved.is_file():
        raise FrontierRateAttackFeedbackError(f"{path} does not exist")
    return _load_json(resolved)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        materialization_plan_path = _resolve_repo_path(args.materialization_plan)
        binding_report = build_frontier_repair_budget_materializer_binding_report(
            repo_root=REPO_ROOT,
            repair_budget_materialization_plan=_load_json(materialization_plan_path),
            repair_budget_materialization_plan_path=args.materialization_plan,
            materializer_work_queue=_optional_json(args.materializer_work_queue),
            materializer_work_queue_path=args.materializer_work_queue,
            materializer_execution_queue=_optional_json(args.materializer_execution_queue),
            materializer_execution_queue_path=args.materializer_execution_queue,
            materializer_manifest_paths=tuple(args.materializer_manifest),
            repair_palette_modes=tuple(args.repair_palette_mode),
        )
        binding_report_out = _resolve_repo_path(args.binding_report_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if binding_report_out.exists() and args.overwrite:
            existing_text = binding_report_out.read_text(encoding="utf-8")
            next_text = json_text(binding_report)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(binding_report_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                binding_report_out,
                binding_report,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        ExperimentQueueError,
        FrontierRateAttackFeedbackError,
        OSError,
        ValueError,
    ) as exc:
        print(f"FATAL: repair materializer binding failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "frontier_rate_attack_repair_budget_materializer_binding_report_cli_result.v1",
                "materialization_plan": str(args.materialization_plan),
                "binding_report_out": str(args.binding_report_out),
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
                ),
                "candidate_archive_materialized": (
                    binding_report.get("candidate_archive_materialized") is True
                ),
                "repair_dynamics_palette_prior_present": bool(
                    binding_report.get("repair_dynamics_palette_prior")
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
