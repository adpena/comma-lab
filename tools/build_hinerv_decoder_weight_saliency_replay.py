#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build HiNeRV decoder-weight saliency replay rows from a real ladder."""

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

from tac.analysis.hinerv_decoder_weight_saliency_replay import (  # noqa: E402
    DEFAULT_MAX_MEAN_SCORE_LOSS_PROXY_FOR_ALLOCATOR,
    write_hinerv_decoder_weight_saliency_replay,
)


def _csv_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(value).split(",") if token.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-ladder-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--row-id",
        action="append",
        default=[],
        help="Selected archive ladder row id. May be repeated. Defaults to all rows.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
    )
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--segmentation-surrogate", default="soft_cosine")
    parser.add_argument("--segmentation-temperature", type=float, default=1.0)
    parser.add_argument(
        "--max-mean-score-loss-proxy-for-allocator",
        type=float,
        default=DEFAULT_MAX_MEAN_SCORE_LOSS_PROXY_FOR_ALLOCATOR,
        help=(
            "Fail-closed allocator-basin gate. Saliency replay still writes, but "
            "rows above this mean score-loss proxy carry a blocker."
        ),
    )
    parser.add_argument(
        "--include-substrings",
        default="latent_embed,blocks,feature_grids,head,decoder,injector",
    )
    parser.add_argument(
        "--exclude-substrings",
        default="latents,codebook,selector,ema,teacher,student",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive_ladder_path = args.archive_ladder_json.expanduser().resolve(strict=False)
    report = json.loads(archive_ladder_path.read_text(encoding="utf-8"))
    output_json = args.output_json.expanduser().resolve(strict=False)
    output_md = (
        None
        if args.output_md is None
        else args.output_md.expanduser().resolve(strict=False)
    )
    result = write_hinerv_decoder_weight_saliency_replay(
        archive_ladder_report=report,
        row_ids=tuple(args.row_id),
        video_path=args.video_path,
        upstream_dir=args.upstream_dir,
        device=str(args.device),
        max_pairs=int(args.max_pairs),
        start_pair=int(args.start_pair),
        pair_stride=int(args.pair_stride),
        include_substrings=_csv_tokens(args.include_substrings),
        exclude_substrings=_csv_tokens(args.exclude_substrings),
        segmentation_surrogate=str(args.segmentation_surrogate),
        segmentation_temperature=float(args.segmentation_temperature),
        max_mean_score_loss_proxy_for_allocator=(
            float(args.max_mean_score_loss_proxy_for_allocator)
        ),
        output_json=output_json,
        output_md=output_md,
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "report_path": output_json.as_posix(),
                "row_count": result["row_count"],
                "full_video_coverage": result["full_video_coverage"],
                "blockers": result["blockers"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
