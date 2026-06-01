#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare singleton and batched HPRC MLX profile evidence."""

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

from tac.substrates.hprc.batch_profile_compare import (  # noqa: E402
    compare_hprc_mlx_batch_profiles,
    write_hprc_mlx_batch_profile_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--singleton-profile", required=True, type=Path)
    parser.add_argument("--batched-profile", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    output = _resolve(args.output, base=repo_root)
    comparison = compare_hprc_mlx_batch_profiles(
        singleton_profile_path=args.singleton_profile,
        batched_profile_path=args.batched_profile,
        repo_root=repo_root,
    )
    write_hprc_mlx_batch_profile_comparison(
        output_path=output,
        comparison=comparison,
        allow_overwrite=bool(args.force),
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "raw_speedup_singleton_over_batched": comparison["wall_clock"][
                    "raw_speedup_singleton_over_batched"
                ],
                "max_abs_response_drift": comparison["max_abs_response_drift"],
                "max_abs_delta_drift": comparison["max_abs_delta_drift"],
                "score_claim": False,
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
