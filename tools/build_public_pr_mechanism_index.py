#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a local public-PR mechanism index from report/writeup text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.analysis.public_pr_mechanism_index import (  # noqa: E402
    build_public_pr_mechanism_index,
    render_markdown_summary,
    write_index_outputs,
)

DEFAULT_ROOTS = (
    "experiments/results/public_pr_archive_release_view",
    "experiments/results/public_pr_archive_kaggle_mirror",
    "experiments/results/public_pr_intake_full",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, help="Corpus root to scan. May be repeated.")
    parser.add_argument("--min-pr", type=int, default=95)
    parser.add_argument("--max-pr", type=int)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    roots = args.root or [REPO_ROOT / rel for rel in DEFAULT_ROOTS]
    index = build_public_pr_mechanism_index(
        roots,
        min_pr=args.min_pr,
        max_pr=args.max_pr,
        repo_root=REPO_ROOT,
    )
    write_index_outputs(index, json_out=args.json_out, md_out=args.md_out)
    if not args.json_out and not args.md_out:
        print(render_markdown_summary(index))
    else:
        print(f"indexed {index['file_count']} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
