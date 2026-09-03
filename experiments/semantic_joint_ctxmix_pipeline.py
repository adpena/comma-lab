#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Thin CLI for the resumable semantic-joint-ctxmix stage graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.semantic_pipeline import FullPipelineConfig, PipelineBlocked, SemanticPipeline
from tac.semantic_pipeline.pipeline import DEFAULT_STORE, DEFAULT_VIDEO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replay", "full"), required=True)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), required=True)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--smoke-pairs", type=int, default=2)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--verdict-batch-size", type=int, default=32)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--from-scratch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume_from is not None and args.resume_from.resolve() != (args.store / args.mode).resolve():
        raise ValueError("--resume-from must name this mode's durable stage store")
    config = FullPipelineConfig(
        mode=args.mode,
        device=args.device,
        video=args.video,
        store=args.store,
        seed=args.seed,
        smoke_pairs=args.smoke_pairs,
        smoke_steps=args.smoke_steps,
        verdict_batch_size=args.verdict_batch_size,
        resume=args.resume_from is not None,
        from_scratch=args.from_scratch,
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
