#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Refresh MLX scorer-response source identity fields from source payloads."""

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
    refresh_mlx_scorer_response_source_identity,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--allow-blocked-output",
        action="store_true",
        help="Write the refreshed dataset even when the refresh gate is blocked.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _render_markdown(dataset: dict, *, source_path: Path, output_path: Path) -> str:
    refresh = dataset.get("source_identity_refresh")
    if not isinstance(refresh, dict):
        refresh = {}
    return "\n".join(
        [
            "# Scorer Response Source Identity Refresh",
            "",
            "## Authority",
            "",
            "- Score claim: `False`",
            "- Promotion eligible: `False`",
            "- Ready for exact-eval dispatch: `False`",
            "- Rank/kill eligible: `False`",
            "",
            "## Summary",
            "",
            f"- Source dataset: `{source_path}`",
            f"- Output dataset: `{output_path}`",
            f"- Passed: `{refresh.get('passed')}`",
            f"- MLX rows: `{refresh.get('mlx_row_count')}`",
            f"- Refreshed rows: `{refresh.get('refreshed_row_count')}`",
            f"- Updated rows: `{refresh.get('updated_row_count')}`",
            f"- Changed fields: `{refresh.get('changed_field_count')}`",
            f"- Updated row sample: `{refresh.get('updated_row_ids_sample')}`",
            f"- Blockers: `{refresh.get('blockers')}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = refresh_mlx_scorer_response_source_identity(
            _load_json_object(args.dataset)
        )
    except (OSError, json.JSONDecodeError, ValueError, ScorerResponseDatasetError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    refresh = dataset["source_identity_refresh"]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            _render_markdown(dataset, source_path=args.dataset, output_path=args.json_out),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "passed": refresh["passed"],
                "mlx_row_count": refresh["mlx_row_count"],
                "refreshed_row_count": refresh["refreshed_row_count"],
                "updated_row_count": refresh["updated_row_count"],
                "changed_field_count": refresh["changed_field_count"],
                "blocker_count": len(refresh["blockers"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if refresh["passed"] or args.allow_blocked_output:
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
