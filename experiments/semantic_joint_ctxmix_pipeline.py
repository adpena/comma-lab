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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replay", "full"), required=True)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), required=True)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--smoke-pairs", type=int, default=2)
    parser.add_argument("--pairs", type=int, help="alias for --smoke-pairs used by governed launch tickets")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="authorize only the bounded n<=8 local smoke")
    parser.add_argument("--verdict-batch-size", type=int, default=32)
    parser.add_argument("--semantic-lineage", choices=("av", "dali"), default="av")
    parser.add_argument("--carrier-lineage", choices=("dali",), default="dali")
    parser.add_argument("--hpac-lineage", choices=("dali",), default="dali")
    parser.add_argument("--token-lineage", choices=("dali",), default="dali")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--from-scratch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pairs is not None and args.smoke_pairs != 2 and args.pairs != args.smoke_pairs:
        raise ValueError("--pairs and --smoke-pairs disagree")
    pair_count = args.smoke_pairs if args.pairs is None else args.pairs
    if args.resume_from is not None and args.resume_from.resolve() != (args.store / args.mode).resolve():
        raise ValueError("--resume-from must name this mode's durable stage store")
    config = FullPipelineConfig(
        mode=args.mode,
        device=args.device,
        video=args.video,
        store=args.store,
        seed=args.seed,
        smoke_pairs=pair_count,
        smoke_steps=args.smoke_steps,
        verdict_batch_size=args.verdict_batch_size,
        resume=args.resume_from is not None,
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
        result = SemanticPipeline(config).run()
    except PipelineBlocked as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
