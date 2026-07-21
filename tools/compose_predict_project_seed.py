#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compose the real n600 seed curve for delegated Task seed_compose_b2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.optimization.seed_compose_b2 import compose_seed_curve  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"),
    )
    parser.add_argument(
        "--s2-packet",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/s2_compose_20260721/partition_seed/s2_partition_event_seed.bin"),
    )
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        default=Path(
            "/Volumes/VertigoDataTier/pact/evidence/r2b_sparse_target_selection_20260720T1621Z/"
            "baseline_stages_a7192f938785_31d77be9ab9f_107a7d3a179d"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721"),
    )
    args = parser.parse_args()
    receipt = compose_seed_curve(
        repository_root=REPO,
        gt_cache=args.gt_cache.resolve(),
        s2_packet=args.s2_packet.resolve(),
        inventory_dir=args.inventory_dir.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps({"verdict": receipt["verdict"], "receipt": str(args.output_root / "receipt.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
