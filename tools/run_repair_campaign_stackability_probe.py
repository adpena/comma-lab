#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit a local repair-campaign stackability probe from a score report."""

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

from tac.optimization.repair_campaign_scorer import (  # noqa: E402
    RepairCampaignScorerError,
    build_repair_campaign_stackability_probe,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-report", required=True, type=Path)
    parser.add_argument("--typed-response-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        score_report_path = _resolve(args.score_report)
        score_report = json.loads(score_report_path.read_text(encoding="utf-8"))
        if not isinstance(score_report, dict):
            raise RepairCampaignScorerError("score report must be a JSON object")
        probe = build_repair_campaign_stackability_probe(
            score_report=score_report,
            typed_response_id=args.typed_response_id,
            repo_root=REPO_ROOT,
        )
        output_path = _resolve(args.output)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if output_path.exists() and args.overwrite:
            existing_text = output_path.read_text(encoding="utf-8")
            next_text = json_text(probe)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(output_path)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                output_path,
                probe,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (ArtifactWriteError, OSError, RepairCampaignScorerError, ValueError) as exc:
        print(f"FATAL: repair stackability probe failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_stackability_probe_cli_result.v1",
                "score_report": str(args.score_report),
                "typed_response_id": args.typed_response_id,
                "output": str(args.output),
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
                ),
                "stackability_ready": probe["stackability_ready"],
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
