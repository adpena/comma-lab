#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a PR101 pose-axis master-gradient operator candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.master_gradient_pr101_operator_candidate import (  # noqa: E402
    MasterGradientPR101OperatorError,
    build_pr101_pose_axis_decoder_recompression_candidate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--operator-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument(
        "--quality",
        type=int,
        action="append",
        dest="qualities",
        help="Brotli quality to try. Repeatable. Defaults to 0..11.",
    )
    parser.add_argument(
        "--lgwin",
        type=int,
        action="append",
        dest="lgwins",
        help="Brotli lgwin to try. Repeatable. Defaults to 10..24.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        operator_manifest = json.loads(args.operator_manifest.read_text(encoding="utf-8"))
        manifest = build_pr101_pose_axis_decoder_recompression_candidate(
            source_archive=args.source_archive,
            operator_manifest=operator_manifest,
            output_dir=args.output_dir,
            candidate_id=args.candidate_id,
            candidate_rank=args.candidate_rank,
            operator_manifest_path=args.operator_manifest,
            qualities=tuple(args.qualities) if args.qualities else tuple(range(12)),
            lgwin_values=tuple(args.lgwins) if args.lgwins else tuple(range(10, 25)),
        )
    except (json.JSONDecodeError, MasterGradientPR101OperatorError, OSError) as exc:
        raise SystemExit(f"PR101 pose-axis operator build failed: {exc}") from None

    archive = manifest["candidate_archive"]
    stream = manifest["selected_stream"]
    replacement = manifest["replacement_stream"]
    print(
        f"wrote {archive['path']} ({archive['bytes']} bytes, "
        f"sha256={archive['sha256']})"
    )
    print(
        f"rank={args.candidate_rank} stream={stream['stream_index']} "
        f"offset={stream['compressed_start']}:{stream['compressed_end']} "
        f"quality={replacement['quality']} lgwin={replacement['lgwin']} "
        f"archive_delta={archive['archive_byte_delta']}"
    )
    print(
        "score_claim=false promotion_eligible=false "
        f"ready_for_exact_eval_dispatch={manifest['ready_for_exact_eval_dispatch']}"
    )
    print(f"operator_manifest={args.output_dir / 'operator_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
