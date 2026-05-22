#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Merge non-authoritative scorer-response datasets."""

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
    merge_scorer_response_datasets,
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
        seen.add(resolved)
        unique.append(path)
    return unique


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], help="Dataset JSON path or glob; may repeat.")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _expand_inputs(args.input)
    if not paths:
        print("FATAL: at least one --input dataset is required", file=sys.stderr)
        return 2
    try:
        datasets = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ScorerResponseDatasetError(f"{path}: expected JSON object")
            datasets.append((str(path), payload))
        merged = merge_scorer_response_datasets(datasets)
    except (OSError, ValueError, ScorerResponseDatasetError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(merged), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "summary": merged["summary"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
