#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Thin CLI for Task #578's resumable n64/n600 cell-description measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.optimization.predictor_r2_missdelta import (
    build_final_receipt as build_r2_final_receipt,
)
from tac.optimization.predictor_r2_missdelta import (
    run_measurement_stage as run_r2_measurement_stage,
)
from tac.optimization.predictor_r3_causal import (
    build_final_receipt as build_r3_final_receipt,
)
from tac.optimization.predictor_r4_tailrace import (
    build_final_receipt as build_r4_final_receipt,
)
from tac.optimization.predictor_r4_tailrace import run_n64_stage as run_r4_n64_stage
from tac.optimization.predictor_upgrade_xi_chart import build_final_receipt, run_measurement_stage


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--stage",
        choices=("n64", "n600", "final", "r2-n64", "r2-n600", "r2-final", "r3-final", "r4-n64", "r4-final"),
        required=True,
    )
    value.add_argument("--cache", type=Path, required=True)
    value.add_argument("--work-dir", type=Path, required=True)
    value.add_argument("--repository-root", type=Path, default=Path.cwd())
    value.add_argument("--predecessor-seed-dir", type=Path, required=True)
    value.add_argument(
        "--round1-work-dir",
        type=Path,
        help="Required for r2-n64/r2-n600; custody root containing round-1 charts.",
    )
    value.add_argument("--r2-work-dir", type=Path, help="Required for r3-final; completed R2 custody root.")
    value.add_argument("--r2-receipt", type=Path, help="Required for r4-final; committed R2 receipt.")
    value.add_argument("--r3-receipt", type=Path, help="Required for r4-n64/r4-final; committed R3 receipt.")
    value.add_argument("--lane-chart", type=Path, required=True)
    value.add_argument("--output", type=Path)
    value.add_argument("--chunk-size", type=int, default=16)
    value.add_argument("--no-resume", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.stage == "r4-final":
        if args.output is None or args.r2_receipt is None or args.r2_work_dir is None or args.r3_receipt is None:
            raise SystemExit("--output, --r2-receipt, --r2-work-dir and --r3-receipt are required for --stage r4-final")
        result = build_r4_final_receipt(
            repository_root=args.repository_root,
            cache=args.cache,
            r2_work_dir=args.r2_work_dir,
            r2_receipt_path=args.r2_receipt,
            r3_receipt_path=args.r3_receipt,
            work_dir=args.work_dir,
            output_path=args.output,
        )
        print(
            json.dumps(
                {"schema": result["schema"], "verdict": result["verdict"], "pointer": "0.1910828242 UNMOVED"},
                sort_keys=True,
            )
        )
        return 0
    if args.stage == "r4-n64":
        if args.r2_work_dir is None or args.r3_receipt is None:
            raise SystemExit("--r2-work-dir and --r3-receipt are required for --stage r4-n64")
        result = run_r4_n64_stage(
            repository_root=args.repository_root,
            cache=args.cache,
            r2_work_dir=args.r2_work_dir,
            r3_receipt_path=args.r3_receipt,
            work_dir=args.work_dir,
        )
        print(
            json.dumps(
                {
                    "schema": result["schema"],
                    "generator_winning_streams": result["generator_winning_streams"],
                    "pointer": "0.1910828242 UNMOVED",
                },
                sort_keys=True,
            )
        )
        return 0
    if args.stage == "r3-final":
        if args.output is None or args.round1_work_dir is None or args.r2_work_dir is None:
            raise SystemExit("--output, --round1-work-dir and --r2-work-dir are required for --stage r3-final")
        result = build_r3_final_receipt(
            repository_root=args.repository_root,
            cache=args.cache,
            r2_work_dir=args.r2_work_dir,
            round1_work_dir=args.round1_work_dir,
            lane_chart=args.lane_chart,
            work_dir=args.work_dir,
            output_path=args.output,
        )
        print(
            json.dumps(
                {"schema": result["schema"], "verdict": result["verdict"], "pointer": "0.1910828242 UNMOVED"},
                sort_keys=True,
            )
        )
        return 0
    if args.stage == "r2-final":
        if args.output is None:
            raise SystemExit("--output is required for --stage r2-final")
        result = build_r2_final_receipt(cache=args.cache, work_dir=args.work_dir, output_path=args.output)
        print(
            json.dumps(
                {"schema": result["schema"], "verdict": result.get("verdict"), "pointer": "0.1910828242 UNMOVED"},
                sort_keys=True,
            )
        )
        return 0
    if args.stage.startswith("r2-"):
        if args.round1_work_dir is None:
            raise SystemExit("--round1-work-dir is required for round-2 measurement stages")
        n_pairs = 64 if args.stage == "r2-n64" else 600
        result = run_r2_measurement_stage(
            repository_root=args.repository_root,
            cache=args.cache,
            work_dir=args.work_dir,
            round1_work_dir=args.round1_work_dir,
            predecessor_seed_dir=args.predecessor_seed_dir,
            lane_chart=args.lane_chart,
            n_pairs=n_pairs,
            chunk_size=args.chunk_size,
            resume=not args.no_resume,
        )
        print(
            json.dumps(
                {"schema": result["schema"], "verdict": result.get("verdict"), "pointer": "0.1910828242 UNMOVED"},
                sort_keys=True,
            )
        )
        return 0
    seeds = {name: args.predecessor_seed_dir / f"seed_compose_b2_{name}.ppcs" for name in ("loose", "knee", "tight")}
    if args.stage == "final":
        if args.output is None:
            raise SystemExit("--output is required for --stage final")
        result = build_final_receipt(work_dir=args.work_dir, output_path=args.output, predecessor_seeds=seeds)
    else:
        n_pairs = 64 if args.stage == "n64" else 600
        result = run_measurement_stage(
            repository_root=args.repository_root,
            cache=args.cache,
            work_dir=args.work_dir,
            n_pairs=n_pairs,
            chunk_size=args.chunk_size,
            predecessor_seeds=seeds,
            lane_chart=args.lane_chart,
            resume=not args.no_resume,
        )
    print(
        json.dumps(
            {"schema": result["schema"], "verdict": result.get("verdict"), "pointer": "0.1910828242 UNMOVED"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
