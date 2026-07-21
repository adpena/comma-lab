#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Thin CLI for Task #578's resumable n64/n600 cell-description measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.optimization.predictor_upgrade_xi_chart import build_final_receipt, run_measurement_stage


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--stage", choices=("n64", "n600", "final"), required=True)
    value.add_argument("--cache", type=Path, required=True)
    value.add_argument("--work-dir", type=Path, required=True)
    value.add_argument("--repository-root", type=Path, default=Path.cwd())
    value.add_argument("--predecessor-seed-dir", type=Path, required=True)
    value.add_argument("--lane-chart", type=Path, required=True)
    value.add_argument("--output", type=Path)
    value.add_argument("--chunk-size", type=int, default=16)
    value.add_argument("--no-resume", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    seeds = {name: args.predecessor_seed_dir / f"seed_compose_b2_{name}.ppcs" for name in ("loose", "knee", "tight")}
    if args.stage == "final":
        if args.output is None:
            raise SystemExit("--output is required for --stage final")
        result = build_final_receipt(work_dir=args.work_dir, output_path=args.output, predecessor_seeds=seeds)
    else:
        n_pairs = 64 if args.stage == "n64" else 600
        result = run_measurement_stage(
            repository_root=args.repository_root, cache=args.cache, work_dir=args.work_dir, n_pairs=n_pairs,
            chunk_size=args.chunk_size, predecessor_seeds=seeds, lane_chart=args.lane_chart,
            resume=not args.no_resume,
        )
    print(json.dumps({"schema": result["schema"], "verdict": result.get("verdict"), "pointer": "0.1910828242 UNMOVED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
