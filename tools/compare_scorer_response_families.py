#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare matched families inside a non-authoritative scorer-response dataset."""

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

from tac.optimization.scorer_response_dataset import ScorerResponseDatasetError  # noqa: E402
from tac.optimization.scorer_response_family_delta import (  # noqa: E402
    DEFAULT_DELTA_FIELDS,
    build_family_delta,
    render_family_delta_markdown,
)


def _parse_fields(value: str) -> tuple[str, ...]:
    fields = tuple(item.strip() for item in value.split(",") if item.strip())
    if not fields:
        raise argparse.ArgumentTypeError("at least one field is required")
    return fields


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-family", required=True)
    parser.add_argument("--candidate-family", required=True)
    parser.add_argument("--match-key", default="source_start_pair")
    parser.add_argument("--fields", type=_parse_fields, default=DEFAULT_DELTA_FIELDS)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        if not isinstance(dataset, dict):
            raise ScorerResponseDatasetError("dataset must be a JSON object")
        delta = build_family_delta(
            dataset,
            reference_family=args.reference_family,
            candidate_family=args.candidate_family,
            match_key=args.match_key,
            fields=args.fields,
            top_k=args.top_k,
        )
    except (OSError, ValueError, ScorerResponseDatasetError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(delta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_family_delta_markdown(delta), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "summary": delta["summary"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
