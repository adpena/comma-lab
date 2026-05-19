# SPDX-License-Identifier: MIT
"""CLI wrapper around :func:`train_on_mps_real_frames`.

Invoked by ``tools/run_mps_gap_experiment.sh`` Phase 1. Thin argparse
front-end; the heavy lifting lives in :mod:`tac.mps_gap_experiment.train_on_mps`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tac.mps_gap_experiment.train_on_mps import train_on_mps_real_frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--include-scorer-loss", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = train_on_mps_real_frames(
        output_dir=args.output_dir,
        upstream_dir=args.upstream_dir,
        epochs=args.epochs,
        num_pairs=args.num_pairs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        include_scorer_loss=args.include_scorer_loss,
    )
    print(
        f"[mps_gap_experiment] trained {metrics.total_epochs} epochs on "
        f"{metrics.device} in {metrics.total_seconds:.1f}s "
        f"(final pixel_l1 {metrics.final_pixel_loss:.4f}) "
        f"[MPS-research-signal]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
