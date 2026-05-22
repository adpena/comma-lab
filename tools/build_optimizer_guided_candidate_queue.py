#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a deterministic offline queue for optimizer-guided candidates.

The output is a ranked planning queue for local, MLX, Modal, Kaggle, M5, and
custom representation-substrate prefilters. It does not dispatch, create
archives, or claim exact scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.optimizer_guided_candidate_generation import (  # noqa: E402
    DEFAULT_GENERATED_AT_UTC,
    CandidateGenerationError,
    default_profiles,
    generate_candidate_queue,
    load_profile,
    profile_from_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(default_profiles()),
        default="pr101_bias_sidecar",
        help="Built-in low-dimensional search profile.",
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=None,
        help="Custom profile JSON. Overrides --profile when supplied.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("grid", "random", "cmaes", "optuna"),
        default="cmaes",
        help="Proposal strategy. All strategies are stdlib and offline.",
    )
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--generated-at-utc",
        default=DEFAULT_GENERATED_AT_UTC,
        help=(
            "Metadata timestamp. Default is stable so same profile/seed emits "
            "byte-identical JSON."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        profile = (
            profile_from_json(args.profile_json)
            if args.profile_json is not None
            else load_profile(args.profile)
        )
        queue = generate_candidate_queue(
            profile=profile,
            optimizer=args.optimizer,
            max_candidates=args.max_candidates,
            top_k=args.top_k,
            seed=args.seed,
            generated_at_utc=args.generated_at_utc,
        )
    except CandidateGenerationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(queue, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} "
        f"(profile={queue['profile']}, optimizer={queue['optimizer_status']}, "
        f"n_candidates={queue['n_candidates']}, top_k={queue['top_k_count']}, "
        "dispatch_ready=0, score_claim=false)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
