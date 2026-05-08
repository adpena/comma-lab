#!/usr/bin/env python3
"""Emit a deterministic HStack/VStack/multipass hyperprior repair plan.

This is a planning tool only. It produces no score claim, performs no dispatch,
and does not mutate Ballé or shared-PMF implementation files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.codec_stack_planner import (  # noqa: E402
    DEFAULT_STATIC_PMF_DELTA_BYTES,
    build_hstack_vstack_multipass_plan,
    summarize_plan,
)


def _write_json(payload: dict, output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-id",
        default="current_frontier_archive",
        help="Human-readable anchor/archive ID for the plan.",
    )
    parser.add_argument(
        "--static-pmf-delta-bytes",
        type=int,
        default=DEFAULT_STATIC_PMF_DELTA_BYTES,
        help="Negative-control delta for static shared PMF K=12 after charged bytes.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=3,
        help="Number of compress-time planning passes to describe, capped at 5.",
    )
    parser.add_argument(
        "--learned-model-overhead-bytes",
        type=int,
        default=None,
        help="Optional measured Ballé/full hyperprior model overhead, if known.",
    )
    parser.add_argument(
        "--without-balle-hyperprior",
        action="store_true",
        help="Emit the stack plan without the learned Ballé/full hyperprior candidate.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Emit the compact summary instead of the full manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path; omit to print to stdout.",
    )
    args = parser.parse_args(argv)

    plan = build_hstack_vstack_multipass_plan(
        anchor_id=args.anchor_id,
        static_pmf_delta_bytes=args.static_pmf_delta_bytes,
        max_passes=args.max_passes,
        include_balle_hyperprior=not args.without_balle_hyperprior,
        learned_model_overhead_bytes=args.learned_model_overhead_bytes,
    )
    payload = summarize_plan(plan) if args.summary else plan.to_manifest()
    _write_json(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

