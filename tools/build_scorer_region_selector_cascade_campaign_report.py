#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest grouped scorer-region cascade campaign outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.scorer_region_selector_cascade_campaign_queue import (  # noqa: E402
    ScorerRegionSelectorCascadeCampaignQueueError,
    build_scorer_region_selector_cascade_campaign_report,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def _variant_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--variant-root must be variant_id=path")
    key, raw_path = value.split("=", 1)
    key = key.strip()
    raw_path = raw_path.strip()
    if not key or not raw_path:
        raise argparse.ArgumentTypeError("--variant-root requires non-empty id and path")
    return key, Path(raw_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-root", action="append", type=_variant_root, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        variant_roots = dict(args.variant_root)
        payload = build_scorer_region_selector_cascade_campaign_report(
            repo_root=REPO_ROOT,
            variant_roots=variant_roots,
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
        ScorerRegionSelectorCascadeCampaignQueueError,
        ValueError,
    ) as exc:
        print(f"FATAL: scorer-region cascade campaign report failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "scorer_region_selector_cascade_campaign_report_cli_result.v1",
                "output": str(args.output),
                "variant_count": payload["variant_count"],
                "completed_learning_variant_count": payload[
                    "completed_learning_variant_count"
                ],
                "bytes_written": write.bytes_written,
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
