#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the next acquisition policy from a grouped cascade campaign report."""

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

from comma_lab.scheduler.scorer_region_selector_cascade_campaign_queue import (  # noqa: E402
    DEFAULT_MASTER_GRADIENT_TENSOR_PATH,
    DEFAULT_PIXEL_GRADIENT_CACHE_PATH,
    ScorerRegionSelectorCascadeCampaignQueueError,
    build_scorer_region_selector_cascade_acquisition_policy,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--master-gradient-tensor",
        type=Path,
        default=DEFAULT_MASTER_GRADIENT_TENSOR_PATH,
        help="Per-byte/per-pair master-gradient tensor; pass 'none' to disable.",
    )
    parser.add_argument(
        "--archive-master-gradient-hydration",
        type=Path,
        help="Archive-bound master-gradient hydration artifact; pass 'none' to disable.",
    )
    parser.add_argument(
        "--pixel-gradient-cache",
        type=Path,
        default=DEFAULT_PIXEL_GRADIENT_CACHE_PATH,
        help="UNIWARD per-pixel scorer-gradient cache; pass 'none' to disable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _optional_path(path: Path) -> Path | None:
    return None if str(path).strip().lower() == "none" else path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report_path = _resolve(args.campaign_report)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ScorerRegionSelectorCascadeCampaignQueueError(
                f"campaign report must be a JSON object: {report_path}"
            )
        payload = build_scorer_region_selector_cascade_acquisition_policy(
            repo_root=REPO_ROOT,
            campaign_report=report,
            master_gradient_tensor_path=_optional_path(args.master_gradient_tensor),
            archive_master_gradient_hydration_path=_optional_path(
                args.archive_master_gradient_hydration
            )
            if args.archive_master_gradient_hydration is not None
            else None,
            pixel_gradient_cache_path=_optional_path(args.pixel_gradient_cache),
        )
        output = _resolve(args.output)
        expected_existing_sha256 = (
            sha256_file(output) if output.is_file() and args.overwrite else None
        )
        write = write_json_artifact(
            output,
            payload,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_existing_sha256,
        )
    except (
        ArtifactWriteError,
        OSError,
        json.JSONDecodeError,
        ScorerRegionSelectorCascadeCampaignQueueError,
        ValueError,
    ) as exc:
        print(f"FATAL: scorer-region cascade acquisition policy failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "scorer_region_selector_cascade_acquisition_policy_cli_result.v1",
                "output": str(args.output),
                "bytes_written": write.bytes_written,
                "next_queue_mode": payload["next_queue_policy"]["mode"],
                "blocker_count": len(payload["blockers"]),
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
