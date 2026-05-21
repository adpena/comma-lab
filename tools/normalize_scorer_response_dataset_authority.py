#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Normalize legacy scorer-response datasets to explicit false authority fields."""

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

from tac.optimization.scorer_response_dataset import (  # noqa: E402
    ScorerResponseDatasetError,
    normalize_legacy_response_dataset_authority,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--source-label",
        help="Optional provenance label recorded in authority_normalization.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        normalized = normalize_legacy_response_dataset_authority(
            payload,
            source_label=args.source_label or str(args.input),
        )
    except (OSError, json.JSONDecodeError, ScorerResponseDatasetError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = normalized["authority_normalization"]
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "score_claim": False,
                "backfilled_missing_false_field_count": metadata[
                    "backfilled_missing_false_field_count"
                ],
                "source_label": metadata["source_label"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
