#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the composite P18/P19 -> P11 -> P15 chain report."""

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

from comma_lab.scheduler.scorer_region_selector_chain_queue import (  # noqa: E402
    ScorerRegionSelectorChainQueueError,
    build_scorer_region_selector_chain_report,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-context", required=True, type=Path)
    parser.add_argument("--selector-manifest", required=True, type=Path)
    parser.add_argument("--repack-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScorerRegionSelectorChainQueueError(f"{path} must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_scorer_region_selector_chain_report(
            repo_root=REPO_ROOT,
            chain_context=_load_json(args.chain_context),
            chain_context_path=args.chain_context,
            selector_manifest=_load_json(args.selector_manifest),
            selector_manifest_path=args.selector_manifest,
            repack_manifest=_load_json(args.repack_manifest),
            repack_manifest_path=args.repack_manifest,
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
        ScorerRegionSelectorChainQueueError,
        ValueError,
    ) as exc:
        print(f"FATAL: scorer-region selector chain report failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "scorer_region_selector_chain_report_cli_result.v1",
                "output": str(args.output),
                "bytes_written": write.bytes_written,
                "selected_local_survivor_stage": payload["selected_local_survivor_stage"],
                "cumulative_rate_saved_bytes_vs_source": payload[
                    "cumulative_rate_saved_bytes_vs_source"
                ],
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
