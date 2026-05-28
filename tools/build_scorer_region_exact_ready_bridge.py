#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build fail-closed exact-ready bridge inputs for scorer-region cascades."""

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

from comma_lab.scheduler.scorer_region_exact_ready_bridge import (  # noqa: E402
    ScorerRegionExactReadyBridgeError,
    build_scorer_region_exact_ready_bridge,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-report", required=True, type=Path)
    parser.add_argument("--receiver-patch-manifest", required=True, type=Path)
    parser.add_argument("--source-queue-out", required=True, type=Path)
    parser.add_argument("--blocked-exact-ready-queue-out", required=True, type=Path)
    parser.add_argument("--bridge-report-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _write(path: Path, payload: object, *, overwrite: bool) -> int:
    target = _resolve(path)
    expected = sha256_file(target) if target.exists() and overwrite else None
    return write_json_artifact(
        target,
        payload,
        allow_overwrite=overwrite,
        expected_existing_sha256=expected,
    ).bytes_written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bridge = build_scorer_region_exact_ready_bridge(
            chain_report_path=args.chain_report,
            receiver_patch_manifest_path=args.receiver_patch_manifest,
            repo_root=REPO_ROOT,
        )
        bytes_written = 0
        bytes_written += _write(
            args.source_queue_out,
            bridge["source_optimizer_queue"],
            overwrite=bool(args.overwrite),
        )
        bytes_written += _write(
            args.blocked_exact_ready_queue_out,
            bridge["blocked_exact_ready_queue"],
            overwrite=bool(args.overwrite),
        )
        bytes_written += _write(
            args.bridge_report_out,
            bridge["bridge_report"],
            overwrite=bool(args.overwrite),
        )
    except (
        ArtifactWriteError,
        OSError,
        ScorerRegionExactReadyBridgeError,
        ValueError,
    ) as exc:
        print(f"FATAL: scorer-region exact-ready bridge failed: {exc}", file=sys.stderr)
        return 2
    report = bridge["bridge_report"]
    print(
        json_text(
            {
                "schema": "scorer_region_exact_ready_bridge_cli_result.v1",
                "candidate_count": report["candidate_count"],
                "archive_custody_proven_count": report[
                    "archive_custody_proven_count"
                ],
                "runtime_content_tree_custody_proven_count": report[
                    "runtime_content_tree_custody_proven_count"
                ],
                "bytes_written": bytes_written,
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
