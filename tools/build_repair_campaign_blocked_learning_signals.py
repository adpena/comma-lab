#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build blocked repair-campaign learning signals from score or activation artifacts."""

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

from tac.optimization.repair_campaign_learning_signal import (  # noqa: E402
    RepairCampaignLearningSignalError,
    build_repair_campaign_activation_plan_learning_signal_report,
    build_repair_campaign_blocked_learning_signal_report,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--score-report", type=Path)
    source.add_argument("--activation-plan", type=Path)
    parser.add_argument(
        "--blocked-signal-report-out",
        "--blocked-learning-signal-report-out",
        dest="blocked_signal_report_out",
        required=True,
        type=Path,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_label: str
        source_path: Path
        if args.score_report is not None:
            source_label = "score_report"
            source_path = _resolve(args.score_report)
            score_report = json.loads(source_path.read_text(encoding="utf-8"))
            if not isinstance(score_report, dict):
                raise RepairCampaignLearningSignalError(
                    "score report must be a JSON object"
                )
            report = build_repair_campaign_blocked_learning_signal_report(
                score_report_path=args.score_report,
                score_report=score_report,
                repo_root=REPO_ROOT,
            )
        else:
            source_label = "activation_plan"
            source_path = _resolve(args.activation_plan)
            activation_plan = json.loads(source_path.read_text(encoding="utf-8"))
            if not isinstance(activation_plan, dict):
                raise RepairCampaignLearningSignalError(
                    "activation plan must be a JSON object"
                )
            report = build_repair_campaign_activation_plan_learning_signal_report(
                activation_plan_path=args.activation_plan,
                activation_plan=activation_plan,
                repo_root=REPO_ROOT,
            )
        report_out = _resolve(args.blocked_signal_report_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if report_out.exists() and args.overwrite:
            existing_text = report_out.read_text(encoding="utf-8")
            next_text = json_text(report)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(report_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                report_out,
                report,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        OSError,
        RepairCampaignLearningSignalError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"FATAL: repair campaign blocked learning signal failed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_blocked_learning_signal_cli_result.v1",
                "source_kind": source_label,
                "source_path": str(
                    args.score_report
                    if args.score_report is not None
                    else args.activation_plan
                ),
                "blocked_signal_report_out": str(args.blocked_signal_report_out),
                "blocked_signal_count": report["blocked_signal_count"],
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
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
