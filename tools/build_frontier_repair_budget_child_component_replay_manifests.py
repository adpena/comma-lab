#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit repair-budget child component replay manifests from MLX harvests."""

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

from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.frontier_rate_attack_feedback import (  # noqa: E402
    FrontierRateAttackFeedbackError,
    build_frontier_repair_budget_child_component_replay_manifests,
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
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--candidate-chain-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FrontierRateAttackFeedbackError(f"{path} must contain a JSON object")
    return payload


def _load_response_harvests(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    harvests: dict[str, dict[str, Any]] = {}
    for row in plan.get("candidate_chain_rows") or []:
        if not isinstance(row, dict):
            continue
        if row.get("candidate_kind") != "spent_budget_repair_child":
            continue
        raw_path = str(row.get("source_response_artifact_path") or "").strip()
        if not raw_path or raw_path in harvests:
            continue
        path = _resolve_repo_path(Path(raw_path))
        if not path.is_file():
            continue
        harvests[raw_path] = _load_json(path)
    return harvests


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        materialization_plan_path = _resolve_repo_path(args.materialization_plan)
        plan = _load_json(materialization_plan_path)
        manifests = build_frontier_repair_budget_child_component_replay_manifests(
            repair_budget_materialization_plan=plan,
            repair_budget_materialization_plan_path=args.materialization_plan,
            response_harvests_by_path=_load_response_harvests(plan),
            candidate_chain_ids=tuple(args.candidate_chain_id),
        )
        output_manifest = _resolve_repo_path(args.output_manifest)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if output_manifest.exists() and args.overwrite:
            existing_text = output_manifest.read_text(encoding="utf-8")
            next_text = json_text(manifests)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(output_manifest)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                output_manifest,
                manifests,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        ExperimentQueueError,
        FrontierRateAttackFeedbackError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: repair child replay manifest build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "frontier_rate_attack_repair_budget_child_component_replay_manifests_cli_result.v1",
                "materialization_plan": str(args.materialization_plan),
                "output_manifest": str(args.output_manifest),
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
                ),
                "manifest_count": manifests.get("manifest_count"),
                "component_response_replayed_count": manifests.get(
                    "component_response_replayed_count"
                ),
                "byte_closed_candidate_emitted_count": manifests.get(
                    "byte_closed_candidate_emitted_count"
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
