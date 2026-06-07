#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rank the worst HiNeRV target-class regions into a durable mining plan.

Thin CLI over ``tac.analysis.hinerv_hard_region_miner``.  It reads
label/argmax plus either compact target-margin maps or full logits arrays
(``.npy`` or a key inside an ``.npz``), produces a deterministic,
class-diverse ranking of the hardest target regions, and writes the plan JSON
to a durable path (default under ``experiments/results/`` — never ``/tmp`` per
CLAUDE.md custody discipline).

The plan is planning evidence only: it carries the
``planning_control_false_authority`` marker and no score/promotion keys.  It is
the *input* a scoped birth actuator works through to produce the
``hi_nerv_representative_region_coverage.v1`` coverage rows the long-run launch
gate requires.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - exercised only outside package
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hinerv_hard_region_miner import (  # noqa: E402
    build_hard_region_mining_plan,
    mine_hard_regions,
    mine_hard_regions_from_margin_map,
)


def _load_array(spec: str) -> np.ndarray:
    """Load ``path`` or ``path::key`` (npz key) into an ndarray."""

    if "::" in spec:
        path_str, key = spec.split("::", 1)
    else:
        path_str, key = spec, None
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"array file not found: {path}")
    if path.suffix == ".npz":
        with np.load(path) as bundle:
            if key is None:
                names = list(bundle.files)
                if len(names) != 1:
                    raise ValueError(
                        f"{path} is an npz with {len(names)} keys {names}; "
                        f"specify one as {path}::<key>"
                    )
                key = names[0]
            if key not in bundle.files:
                raise KeyError(f"key {key!r} not in {path} (keys={list(bundle.files)})")
            return np.asarray(bundle[key])
    return np.asarray(np.load(path, allow_pickle=False))


def _default_output(repo_root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "experiments" / "results" / f"hinerv_hard_region_plan_{stamp}" / "plan.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-labels",
        required=True,
        help="BHW int label map (.npy or path.npz::key).",
    )
    parser.add_argument(
        "--candidate-argmax",
        required=True,
        help="BHW int candidate argmax map (.npy or path.npz::key).",
    )
    parser.add_argument(
        "--logits",
        default=None,
        help="BHWC float logits (.npy or path.npz::key).",
    )
    parser.add_argument(
        "--target-margin",
        default=None,
        help=(
            "BHW compact PR95 target margin map (.npy or path.npz::key). "
            "Use this instead of --logits for durable receipts."
        ),
    )
    parser.add_argument(
        "--pose-coupling",
        default=None,
        help="Optional BHW float pose-Jacobian magnitude map (.npy or path.npz::key).",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Max regions after diversity enforcement.")
    parser.add_argument(
        "--min-region-pixels",
        type=int,
        default=1,
        help="Drop connected components smaller than this before ranking.",
    )
    parser.add_argument(
        "--include-solved-regions",
        action="store_true",
        help="Keep regions with zero unsolved pixels (diagnostic only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Durable plan JSON path (default: experiments/results/hinerv_hard_region_plan_<utc>/plan.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    target = _load_array(args.target_labels)
    candidate = _load_array(args.candidate_argmax)
    pose = _load_array(args.pose_coupling) if args.pose_coupling else None
    if bool(args.logits) == bool(args.target_margin):
        raise ValueError("exactly one of --target-margin or --logits is required")

    if args.target_margin:
        regions = mine_hard_regions_from_margin_map(
            target,
            candidate,
            _load_array(args.target_margin),
            top_k=args.top_k,
            pose_coupling=pose,
            min_region_pixels=args.min_region_pixels,
            include_solved_regions=args.include_solved_regions,
        )
    else:
        regions = mine_hard_regions(
            target,
            candidate,
            _load_array(args.logits),
            top_k=args.top_k,
            pose_coupling=pose,
            min_region_pixels=args.min_region_pixels,
            include_solved_regions=args.include_solved_regions,
        )
    plan = build_hard_region_mining_plan(
        regions,
        source=str(Path(args.target_labels)),
        top_k=args.top_k,
    )

    output = args.output if args.output is not None else _default_output(REPO_ROOT)
    output = output.expanduser()
    posix = output.resolve().as_posix()
    # Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact": this plan
    # is durable planning evidence, so it must not land under any transient
    # tmp tier (matches tools/build_composition_ranking_json.py discipline).
    if posix.startswith(("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")):
        raise ValueError(
            f"refusing to write durable plan to a forbidden /tmp path: {output} "
            "(per CLAUDE.md Forbidden /tmp paths non-negotiable)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {plan['region_count']} ranked regions -> {output}")
    print(f"distinct_classes={plan['distinct_classes']}")
    print(f"distinct_class_size_buckets={plan['distinct_class_size_buckets']}")
    print(f"size_class_histogram={plan['size_class_histogram']}")
    for region in plan["regions"]:
        pose_risk = region["pose_coupling_risk_mean"]
        pose_str = "n/a" if pose_risk is None else f"{pose_risk:.4g}"
        print(
            f"  rank={region['rank']} class={region['class_index']} "
            f"size={region['size_class']} px={region['region_pixel_count']} "
            f"debt_local={region['score_debt_units_local']:.4g} "
            f"margin_mean={region['margin_mean']:.4g} pose_risk={pose_str}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
