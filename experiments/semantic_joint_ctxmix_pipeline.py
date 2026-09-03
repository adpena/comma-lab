#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Thin CLI for the resumable semantic-joint-ctxmix stage graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.semantic_pipeline import FullPipelineConfig, PipelineBlocked, SemanticPipeline
from tac.semantic_pipeline.contracts import TargetLineage
from tac.semantic_pipeline.pipeline import DEFAULT_STORE, DEFAULT_VIDEO
from tac.subset_selection import (
    DEFAULT_STRATIFIED_BLOCKS,
    MODE_SEEDED_RANDOM,
    MODE_STRATIFIED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replay", "full"), required=True)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), required=True)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--smoke-pairs", type=int, default=2)
    parser.add_argument("--pairs", type=int, help="alias for --smoke-pairs used by governed launch tickets")
    parser.add_argument("--smoke-steps", "--updates", dest="smoke_steps", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="authorize only the bounded n<=8 local smoke")
    parser.add_argument("--verdict-batch", "--verdict-batch-size", dest="verdict_batch", type=int, default=32)
    parser.add_argument("--chunk-pairs", type=int, default=16)
    parser.add_argument(
        "--selection-mode",
        choices=(MODE_SEEDED_RANDOM, MODE_STRATIFIED),
        default=MODE_STRATIFIED,
    )
    parser.add_argument("--stratified-blocks", type=int, default=DEFAULT_STRATIFIED_BLOCKS)
    parser.add_argument("--scorer-claim-id")
    parser.add_argument("--semantic-lineage", choices=("av", "dali"), default="av")
    parser.add_argument("--carrier-lineage", choices=("dali",), default="dali")
    parser.add_argument("--hpac-lineage", choices=("dali",), default="dali")
    parser.add_argument("--token-lineage", choices=("dali",), default="dali")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--prepare-launch-ticket",
        action="store_true",
        help="write numeric n600 preflight/ticket receipts without launching",
    )
    parser.add_argument("--from-scratch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pairs is not None and args.smoke_pairs != 2 and args.pairs != args.smoke_pairs:
        raise ValueError("--pairs and --smoke-pairs disagree")
    pair_count = args.smoke_pairs if args.pairs is None else args.pairs
    resume = args.resume_from is not None
    resume_checkpoint = None
    if args.resume_from is not None and str(args.resume_from) != "latest":
        run_store = (args.store / args.mode).resolve()
        resolved = args.resume_from.resolve()
        if resolved == run_store:
            resume_checkpoint = None
        elif resolved.parent == (run_store / "train" / "checkpoints").resolve():
            resume_checkpoint = resolved
        else:
            raise ValueError("--resume-from must be 'latest', this mode store, or one of its chunk checkpoints")
    config = FullPipelineConfig(
        mode=args.mode,
        device=args.device,
        video=args.video,
        store=args.store,
        seed=args.seed,
        smoke_pairs=pair_count,
        smoke_steps=args.smoke_steps,
        verdict_batch_size=args.verdict_batch,
        chunk_pairs=args.chunk_pairs,
        selection_mode=args.selection_mode,
        stratified_blocks=args.stratified_blocks,
        resume=resume,
        resume_from=resume_checkpoint,
        scorer_claim_id=args.scorer_claim_id,
        from_scratch=args.from_scratch,
        smoke=args.smoke,
        target_lineage=TargetLineage(
            semantic=args.semantic_lineage,
            carrier=args.carrier_lineage,
            hpac=args.hpac_lineage,
            token=args.token_lineage,
        ),
    )
    try:
        pipeline = SemanticPipeline(config)
        result = (
            pipeline.prepare_population_launch()
            if args.prepare_launch_ticket
            else pipeline.run()
        )
    except PipelineBlocked as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
