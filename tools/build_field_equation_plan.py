#!/usr/bin/env python3
"""Build a planning-only field-equation plan from a meta-Lagrangian ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.field_equation_planner import build_field_equation_plan  # noqa: E402
from tac.repo_io import json_text, read_json  # noqa: E402


def _read_optional_list(path: Path | None) -> list[dict]:
    if path is None:
        return []
    payload = read_json(path)
    if not isinstance(payload, list):
        raise SystemExit("--interactions-json must contain a JSON list")
    return [dict(item) for item in payload if isinstance(item, dict)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-ledger", type=Path, required=True)
    parser.add_argument("--interactions-json", type=Path)
    parser.add_argument("--source", default="manual")
    parser.add_argument("--base-score", type=float)
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument("--max-byte-delta", type=float)
    parser.add_argument(
        "--research-basis-id",
        action="append",
        default=[],
        help="Optional explicit research-basis id to include in the plan manifest.",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    constraints = {}
    if args.min_confidence is not None:
        constraints["min_confidence"] = args.min_confidence
    if args.max_byte_delta is not None:
        constraints["max_byte_delta"] = args.max_byte_delta
    plan = build_field_equation_plan(
        read_json(args.atom_ledger),
        source=args.source,
        constraints=constraints,
        interactions=_read_optional_list(args.interactions_json),
        base_score=args.base_score,
        research_basis_ids=args.research_basis_id,
    )
    text = json_text(plan)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
