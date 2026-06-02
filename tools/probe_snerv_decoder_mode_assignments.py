#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Probe SNeRV mixed decoder mode assignments on a local advisory axis.

This is a NON-PROMOTABLE [macOS-CPU advisory] probe. It runs the existing
SNeRV receiver-decoded advisory path for one or more mixed decoder mode plans
and writes a fail-closed JSON artifact. It does not launch exact eval, CUDA, or
full-video work.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.analysis.snerv_decoder_mode_assignment_probe import (  # noqa: E402
    run_snerv_decoder_mode_assignment_probe,
)


def _default_out() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f".omx/research/snerv_decoder_mode_assignment_probe_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode-plan",
        action="append",
        default=None,
        help=(
            "Mode plan to probe. Use magnitude_heuristic/auto, or a comma-separated "
            "explicit plan with one mode per level/subband kernel."
        ),
    )
    ap.add_argument("--n-pairs", type=int, default=1)
    ap.add_argument("--levels", type=int, default=1)
    ap.add_argument("--bits-per-coeff", type=float, default=2.0)
    ap.add_argument("--wavelet", type=str, default="db2")
    ap.add_argument("--pair-stride", type=int, default=1)
    ap.add_argument("--start-pair", type=int, default=0)
    ap.add_argument("--pr101-frontier-bytes", type=int, default=178_493)
    ap.add_argument("--upstream-dir", type=str, default="upstream")
    ap.add_argument("--video-path", type=str, default="upstream/videos/0.mkv")
    ap.add_argument("--step-map-coder-bins", type=int, default=4)
    ap.add_argument(
        "--step-map-coder-mode",
        choices=("uniform", "adaptive", "waterfill"),
        default="uniform",
    )
    ap.add_argument("--step-map-adaptive-bin-choices", default="128,16,4")
    ap.add_argument("--step-map-constant-importance-quantile", type=float, default=None)
    ap.add_argument("--step-map-waterfill-bits-per-coeff", type=float, default=4.0)
    ap.add_argument(
        "--hf-decoder-fit-mode",
        choices=("least_squares", "score_weighted"),
        default="least_squares",
    )
    ap.add_argument("--hf-decoder-saliency-gain", type=float, default=1.0)
    ap.add_argument(
        "--hf-decoder-saliency-component",
        choices=("combined", "seg", "pose"),
        default="combined",
    )
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--receiver-packet-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for exported raw receiver .snar packets. "
            "When omitted, the probe remains JSON-only and cannot feed "
            "trained-ladder archive custody directly."
        ),
    )
    args = ap.parse_args(argv)

    mode_plans = args.mode_plan
    if not mode_plans:
        mode_plans = ["magnitude_heuristic"]
        if args.levels == 1:
            mode_plans.append("fp16,int4,int4")

    payload = run_snerv_decoder_mode_assignment_probe(
        mode_plans=mode_plans,
        n_pairs=args.n_pairs,
        levels=args.levels,
        bits_per_coeff=args.bits_per_coeff,
        wavelet=args.wavelet,
        pair_stride=args.pair_stride,
        start_pair=args.start_pair,
        pr101_frontier_bytes=args.pr101_frontier_bytes,
        upstream_dir=args.upstream_dir,
        video_path=args.video_path,
        step_map_coder_bins=args.step_map_coder_bins,
        step_map_coder_mode=args.step_map_coder_mode,
        step_map_adaptive_bin_choices=_parse_bins(args.step_map_adaptive_bin_choices),
        step_map_constant_importance_quantile=(
            args.step_map_constant_importance_quantile
        ),
        step_map_waterfill_bits_per_coeff=args.step_map_waterfill_bits_per_coeff,
        hf_decoder_fit_mode=args.hf_decoder_fit_mode,
        hf_decoder_saliency_gain=args.hf_decoder_saliency_gain,
        hf_decoder_saliency_component=args.hf_decoder_saliency_component,
        receiver_packet_dir=args.receiver_packet_dir,
    )
    out_path = Path(args.out or _default_out())
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("[SNeRV decoder mode probe] [macOS-CPU advisory] NON-PROMOTABLE")
    print(
        f"  plans={payload['mode_plan_count']} n_pairs={payload['n_pairs']} "
        f"levels={payload['levels']} best={payload['best_plan_label']}"
    )
    if payload["best_plan_score_linf_advisory"] is not None:
        print(f"  best advisory score_linf={payload['best_plan_score_linf_advisory']:.6f}")
    print(f"  blockers={', '.join(payload['blockers'])}")
    print(f"  wrote {out_path}")
    return 0


def _parse_bins(raw: str) -> tuple[int, ...]:
    values = []
    for chunk in str(raw or "").split(","):
        if chunk.strip():
            value = int(chunk.strip())
            if value < 2 or value > 256:
                raise ValueError("step-map adaptive bin choices must be in [2, 256]")
            values.append(value)
    if not values:
        raise ValueError("at least one step-map adaptive bin choice is required")
    return tuple(values)


if __name__ == "__main__":
    raise SystemExit(main())
