#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan deterministic CPU scorer-oracle candidates from a contest atom lattice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.contest_oracle_search import build_lfv1_pair_queue  # noqa: E402


def _float_grid(raw: str) -> list[float]:
    out: list[float] = []
    for token in raw.split(","):
        text = token.strip()
        if text:
            out.append(float(text))
    if not out:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-archive-delta-bytes", type=int, default=320)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--top-pair-pool", type=int, default=32)
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--alpha-grid", type=_float_grid, default="0.000005,0.00001,0.00002,0.00004")
    parser.add_argument("--radius-scale-grid", type=_float_grid, default="0.45,0.70,0.95")
    parser.add_argument("--power-grid", type=_float_grid, default="0.8,1.3,2.0")
    parser.add_argument("--origin-y-frac-grid", type=_float_grid, default="0.38,0.45,0.52")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lattice = json.loads(args.lattice.read_text(encoding="utf-8"))
    if not isinstance(lattice, dict):
        raise SystemExit("--lattice must be a JSON object")
    plan = build_lfv1_pair_queue(
        lattice,
        max_archive_delta_bytes=args.max_archive_delta_bytes,
        max_candidates=args.max_candidates,
        alpha_grid=args.alpha_grid,
        top_pair_pool=args.top_pair_pool,
        min_pairs=args.min_pairs,
        max_pairs=args.max_pairs,
        radius_scale_grid=args.radius_scale_grid,
        power_grid=args.power_grid,
        origin_y_frac_grid=args.origin_y_frac_grid,
    )
    plan["source_lattice_path"] = _repo_relative(args.lattice)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": plan["candidate_count"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
