#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Append a repair campaign learning signal to the stackability posterior."""

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
    append_repair_campaign_stackability_posterior_signal,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-signal", required=True, type=Path)
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


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairCampaignPosteriorError("learning signal must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        learning_signal_path = _resolve(args.learning_signal)
        learning_signal = _load_json_object(learning_signal_path)
        report = append_repair_campaign_stackability_posterior_signal(
            learning_signal_path=args.learning_signal,
            learning_signal=learning_signal,
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
        print(f"FATAL: repair campaign posterior append failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_stackability_posterior_append_cli_result.v1",
                "learning_signal": str(args.learning_signal),
                "posterior_path": str(args.posterior_path),
                "report_out": str(args.report_out),
                "row_id": report["row_id"],
                "appended": report["appended"],
                "skipped_duplicate": report["skipped_duplicate"],
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
