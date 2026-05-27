#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Diff two repair stackability replay bundles without granting authority."""

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

from tac.optimization.repair_campaign_replay_bundle import (  # noqa: E402
    RepairCampaignReplayBundleError,
    diff_repair_campaign_stackability_replay_bundles,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--diff-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairCampaignReplayBundleError(f"{label} must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        left = _load_json_object(_resolve(args.left), label="left bundle")
        right = _load_json_object(_resolve(args.right), label="right bundle")
        diff = diff_repair_campaign_stackability_replay_bundles(left, right)
        diff_out = _resolve(args.diff_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if diff_out.exists() and args.overwrite:
            existing_text = diff_out.read_text(encoding="utf-8")
            next_text = json_text(diff)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(diff_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                diff_out,
                diff,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        OSError,
        RepairCampaignReplayBundleError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: repair stackability replay diff failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_stackability_replay_bundle_diff_cli_result.v1",
                "left": str(args.left),
                "right": str(args.right),
                "diff_out": str(args.diff_out),
                "matched": diff["matched"],
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
