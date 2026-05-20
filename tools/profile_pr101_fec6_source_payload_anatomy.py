#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile PR101/FEC6 source-payload anatomy and magic-codec byte probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.pr101_fec6_source_anatomy import (  # noqa: E402
    profile_pr101_fec6_source_payload_anatomy,
    render_source_anatomy_markdown,
)


DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--null-indices", type=Path)
    parser.add_argument("--no-magic", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        null_indices = None
        if args.null_indices is not None:
            null_indices = np.load(args.null_indices)
        profile = profile_pr101_fec6_source_payload_anatomy(
            archive_path=args.archive,
            null_indices=null_indices,
            include_magic=not args.no_magic,
        )
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_source_anatomy_markdown(profile),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": profile["schema"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "ranked_next_target": profile["ranked_next_targets"][0]["target_id"],
                "score_claim": profile["score_claim"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
