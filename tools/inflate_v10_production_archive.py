#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: positional-only argv passed through to the production inflate entry; no flags of its own
"""Scorer-free V10 production inflater using the contest three-argument API."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tac.witness_dsl.v10_production_receiver import (
    ProductionReceiverError,
    inflate_archive,
)


def main(argv: Sequence[str] | None = None) -> int:
    active = list(sys.argv[1:] if argv is None else argv)
    if len(active) != 3:
        raise SystemExit("usage: inflate_v10_production_archive.py <archive_dir> <output_dir> <video_names_file>")
    archive_dir, output_dir, video_names_file = map(Path, active)
    try:
        result = inflate_archive(archive_dir, output_dir, video_names_file)
    except ProductionReceiverError as exc:
        raise SystemExit(f"V10 production inflate refused: {exc}") from exc
    if not result.completed or result.raw_path is None or result.raw_sha256 is None:
        raise SystemExit("V10 production inflate stopped before final raw promotion")
    print(
        json.dumps(
            {
                "raw_path": str(result.raw_path),
                "raw_bytes": result.raw_bytes,
                "raw_sha256": result.raw_sha256,
                "pair_stages_preserved": result.pair_stages_preserved,
                "numerator_values_verified": result.numerator_values_verified,
                "tree_sha256": result.tree_sha256,
                "launch_ready": False,
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
