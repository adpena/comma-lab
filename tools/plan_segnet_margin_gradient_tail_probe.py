#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create the source-closed D24a n600 probe plan; do not run the scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.scorer_surrogate.segnet_margin_gradient_tail_probe import (  # noqa: E402
    ArtifactBinding,
    MarginGradientTailProbePlan,
    write_plan,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = MarginGradientTailProbePlan(
        scorer=ArtifactBinding.from_path(role="frozen_segnet_scorer", path=args.scorer),
        source=ArtifactBinding.from_path(role="n600_source", path=args.source),
        cache=ArtifactBinding.from_path(role="scorer_cache", path=args.cache),
        seed=args.seed,
    )
    write_plan(args.output, plan)
    print(json.dumps(plan.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
