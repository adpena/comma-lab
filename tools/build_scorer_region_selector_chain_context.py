#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build P18/P19 upstream custody context for a selector/repack chain."""

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

from comma_lab.scheduler.scorer_region_selector_chain_queue import (  # noqa: E402
    ScorerRegionSelectorChainQueueError,
    build_scorer_region_selector_chain_context,
)
from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-submission-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-waterfill-work-order", type=Path)
    parser.add_argument("--full-frame-inflate-parity-proof", type=Path)
    parser.add_argument("--posenet-null-pairs", type=Path)
    parser.add_argument("--segnet-region-masks", type=Path)
    parser.add_argument("--selector-region-bits", type=Path)
    parser.add_argument(
        "--chain-label",
        default="cascade_c_p19_p18_to_p11_selector_context_then_p15_repack",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_scorer_region_selector_chain_context(
            repo_root=REPO_ROOT,
            source_submission_dir=args.source_submission_dir,
            source_waterfill_work_order=args.source_waterfill_work_order,
            full_frame_inflate_parity_proof=args.full_frame_inflate_parity_proof,
            posenet_null_pairs=args.posenet_null_pairs,
            segnet_region_masks=args.segnet_region_masks,
            selector_region_bits=args.selector_region_bits,
            chain_label=args.chain_label,
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
        print(f"FATAL: scorer-region selector chain context failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "scorer_region_selector_chain_context_cli_result.v1",
                "output": str(args.output),
                "bytes_written": write.bytes_written,
                "p18_p19_upstream_ready": payload["p18_p19_upstream_ready"],
                "blockers": payload["blockers"],
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
