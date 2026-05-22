#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a non-authoritative MLX scorer-response dataset with per-window baselines."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.scorer_response_dataset import (  # noqa: E402
    ScorerResponseDatasetError,
    build_windowed_mlx_response_dataset,
    render_markdown,
)


def _expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value, recursive=True))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        unique.append(path)
        seen.add(resolved)
    return unique


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-response",
        action="append",
        default=[],
        help="MLX scorer-response JSON path or glob; may repeat.",
    )
    parser.add_argument(
        "--baseline-response",
        action="append",
        default=[],
        help="Baseline MLX scorer-response JSON path or glob; may repeat.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty dataset instead of failing when no usable candidate rows survive.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.candidate_response:
        print("FATAL: at least one --candidate-response is required", file=sys.stderr)
        return 2
    if not args.baseline_response:
        print("FATAL: at least one --baseline-response is required", file=sys.stderr)
        return 2
    try:
        dataset = build_windowed_mlx_response_dataset(
            candidate_paths=_expand_inputs(args.candidate_response),
            baseline_paths=_expand_inputs(args.baseline_response),
        )
    except ScorerResponseDatasetError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    row_count = int(dataset.get("summary", {}).get("row_count") or 0)
    if row_count == 0 and not args.allow_empty:
        print("FATAL: no usable MLX window response rows produced", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(dataset), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "summary": dataset["summary"],
                "score_claim": dataset["score_claim"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
