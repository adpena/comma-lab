#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate a JSON artifact contract without rewriting the artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--schema-key")
    parser.add_argument("--schema-equals")
    parser.add_argument("--false-authority", action="store_true")
    return parser.parse_args(argv)


def _json_pointer(payload: Any, key: str) -> Any:
    current = payload
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"missing JSON key: {key}")
        current = current[part]
    return current


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = json.loads(args.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object: {args.path}")
        if args.schema_key is not None or args.schema_equals is not None:
            if not args.schema_key or args.schema_equals is None:
                raise ValueError("--schema-key and --schema-equals must be provided together")
            actual = _json_pointer(payload, args.schema_key)
            if actual != args.schema_equals:
                raise ValueError(
                    f"{args.schema_key} mismatch: expected {args.schema_equals!r}, got {actual!r}"
                )
        if args.false_authority:
            require_no_truthy_authority_fields(
                payload,
                context=f"validate_json_artifact_contract:{args.path}",
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: JSON artifact contract validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": "json_artifact_contract_validation.v1",
                "path": str(args.path),
                "valid": True,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
