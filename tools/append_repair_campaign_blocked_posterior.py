#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Append blocked repair-campaign learning signals to the posterior."""

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

from tac.optimization.repair_campaign_posterior import (  # noqa: E402
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    RepairCampaignPosteriorError,
    append_repair_campaign_blocked_learning_signal_report,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocked-learning-signal-report", required=True, type=Path)
    parser.add_argument(
        "--posterior-path",
        type=Path,
        default=DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    )
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        signal_report_path = _resolve(args.blocked_learning_signal_report)
        signal_report = json.loads(signal_report_path.read_text(encoding="utf-8"))
        if not isinstance(signal_report, dict):
            raise RepairCampaignPosteriorError(
                "blocked learning signal report must be a JSON object"
            )
        report = append_repair_campaign_blocked_learning_signal_report(
            blocked_learning_signal_report_path=args.blocked_learning_signal_report,
            blocked_learning_signal_report=signal_report,
            posterior_path=args.posterior_path,
            lock_path=args.lock_path,
            repo_root=REPO_ROOT,
        )
        report_out = _resolve(args.report_out)
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
        RepairCampaignPosteriorError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"FATAL: repair campaign blocked posterior append failed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_blocked_posterior_append_cli_result.v1",
                "blocked_learning_signal_report": str(
                    args.blocked_learning_signal_report
                ),
                "posterior_path": str(args.posterior_path),
                "report_out": str(args.report_out),
                "signal_count": report["signal_count"],
                "appended_count": report["appended_count"],
                "skipped_duplicate_count": report["skipped_duplicate_count"],
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
