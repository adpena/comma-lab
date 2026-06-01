#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build executable bounded-runner rows for HPRC pair-scoped residual candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.substrates.hprc.pair_scoped_residual_runner import (  # noqa: E402
    build_pair_scoped_residual_bounded_runner_plan,
    write_pair_scoped_residual_bounded_runner_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-plan", required=True, type=Path)
    parser.add_argument("--reuse-baseline-profile", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=600)
    parser.add_argument("--window-pairs", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--no-allow-large-tensor-cache",
        action="store_true",
        help="Omit --allow-large-tensor-cache from generated profile commands.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Plan JSON path; defaults to <output-dir>/hprc_pair_scoped_residual_bounded_runner_plan.json.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    output_dir = _resolve(args.output_dir, base=repo_root)
    plan = build_pair_scoped_residual_bounded_runner_plan(
        pair_plan_path=args.pair_plan,
        reuse_baseline_profile_path=args.reuse_baseline_profile,
        candidate_dir=args.candidate_dir,
        output_dir=output_dir,
        repo_root=repo_root,
        max_candidates=int(args.max_candidates),
        max_pairs=int(args.max_pairs),
        window_pairs=int(args.window_pairs),
        device=str(args.device),
        allow_large_tensor_cache=not bool(args.no_allow_large_tensor_cache),
    )
    output = (
        _resolve(args.output, base=repo_root)
        if args.output is not None
        else output_dir / "hprc_pair_scoped_residual_bounded_runner_plan.json"
    )
    write_pair_scoped_residual_bounded_runner_plan(
        output_path=output,
        plan=plan,
        allow_overwrite=bool(args.force),
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "runner_rows": len(plan["runner_rows"]),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve(path: Path, *, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (base / expanded).resolve(strict=False)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
