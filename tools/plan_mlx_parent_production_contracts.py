#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan strict parent-window MLX production contracts for a response dataset."""

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
    build_mlx_parent_production_contract_plan,
    render_mlx_parent_production_contract_plan_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--production-contract",
        type=Path,
        help="Optional strict MLX production contract or bundle JSON to check coverage.",
    )
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used to discover sibling parent-response artifacts.",
    )
    parser.add_argument(
        "--allow-blocked-output",
        action="store_true",
        help="Return success after writing a blocked planning artifact.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        production_contract = (
            None
            if args.production_contract is None
            else _load_json_object(args.production_contract)
        )
        plan = build_mlx_parent_production_contract_plan(
            _load_json_object(args.dataset),
            production_contract=production_contract,
            repo_root=args.repo_root,
        )
    except (OSError, json.JSONDecodeError, ValueError, ScorerResponseDatasetError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            render_mlx_parent_production_contract_plan_markdown(plan),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "status": plan["status"],
                "mlx_row_count": plan["summary"]["mlx_row_count"],
                "required_parent_contract_count": plan["summary"][
                    "required_parent_contract_count"
                ],
                "covered_parent_contract_group_count": plan["summary"][
                    "covered_parent_contract_group_count"
                ],
                "missing_parent_contract_group_count": plan["summary"][
                    "missing_parent_contract_group_count"
                ],
                "blocker_count": len(plan["blockers"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if plan["status"] == "strict_pass" or args.allow_blocked_output:
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
