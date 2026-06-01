#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Run a bounded local SNeRV scorer-loop decoder/QAT smoke."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (  # noqa: E402
    run_snerv_scorer_loop_decoder_qat_smoke,
)


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_scorer_loop_decoder_qat_smoke_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-pairs", type=int, default=1)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--wavelet", default="db2")
    parser.add_argument("--target-bits-per-coeff", type=float, default=5.0)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--upstream-dir", default="upstream")
    parser.add_argument("--video-path", default="upstream/videos/0.mkv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--step-map-bins", type=int, default=16)
    parser.add_argument("--qat-bits", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=2)
    parser.add_argument(
        "--search-mode",
        choices=("random_signed", "top_weight_coordinate"),
        default="random_signed",
    )
    parser.add_argument("--perturb-scale", type=float, default=0.02)
    parser.add_argument("--pose-slack", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    result = run_snerv_scorer_loop_decoder_qat_smoke(
        n_pairs=args.n_pairs,
        levels=args.levels,
        wavelet=args.wavelet,
        target_bits_per_coeff=args.target_bits_per_coeff,
        pair_stride=args.pair_stride,
        start_pair=args.start_pair,
        upstream_dir=args.upstream_dir,
        video_path=args.video_path,
        device=args.device,
        step_map_bins=args.step_map_bins,
        qat_bits=args.qat_bits,
        max_trials=args.max_trials,
        search_mode=args.search_mode,
        perturb_scale=args.perturb_scale,
        pose_slack=args.pose_slack,
        seed=args.seed,
    )

    out_path = Path(args.out) if args.out else _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result.as_jsonable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[SNeRV scorer-loop decoder/QAT smoke] false-authority")
    print(f"  n_pairs: {result.n_pairs}")
    print(f"  evaluations: {result.scorer_loop_evaluations}")
    print(f"  baseline_score_linf: {result.baseline.score_linf}")
    print(f"  best_score_linf: {result.best.score_linf}")
    print(f"  accepted_improvement: {result.accepted_improvement}")
    print(f"  ready_for_pose_guard_gate: {result.ready_for_pose_guard_gate}")
    print(f"  ready_for_exact_eval_dispatch: {result.ready_for_exact_eval_dispatch}")
    if result.blockers:
        print(f"  blockers: {list(result.blockers)}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
