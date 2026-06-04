#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a prioritized hard-pair hitlist from MLX scorer-response components."""

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

from tac.adaptation.hard_pair_hitlist import (  # noqa: E402
    write_hard_pair_hitlist_from_mlx_response,
)
from tac.repo_io import read_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-response", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-fraction", type=float)
    parser.add_argument("--min-pairs", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    response_path = _resolve(args.mlx_response)
    payload = write_hard_pair_hitlist_from_mlx_response(
        mlx_response=read_json(response_path),
        mlx_response_path=response_path,
        output_json=_resolve(args.output_json),
        top_k=args.top_k,
        top_fraction=args.top_fraction,
        min_pairs=args.min_pairs,
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "output_json": str(_resolve(args.output_json)),
                "pair_count": payload["pair_count"],
                "top_k": payload["top_k"],
                "first_pair_indices": payload["pair_indices"][:10],
                "score_claim": payload["score_claim"],
                "ready_for_exact_eval_dispatch": payload[
                    "ready_for_exact_eval_dispatch"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (REPO_ROOT / expanded).resolve(strict=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
