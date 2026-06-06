#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate thin ``tac.action_effect.v1`` rows.

This is a contract checker, not a score authority.  It accepts JSONL ledgers
written by ``append_action_effect`` and JSON files containing either one row or
a list of rows.  Every row is validated by the same deserializer the launch gate
and commutator consumers use, so stale deltas, hidden score-claim keys,
authority/scope mistakes, and survival action-id mismatches fail closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.action_effect import validate_action_effect_payload  # noqa: E402


def _iter_rows(path: Path) -> Iterable[tuple[int, Mapping[str, Any] | Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return
    if stripped[0] in "[{":
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if "Extra data" not in str(exc):
                raise
        else:
            if isinstance(payload, list):
                for index, row in enumerate(payload, start=1):
                    yield index, row
            else:
                yield 1, payload
            return
    for index, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        yield index, json.loads(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, help="ActionEffect JSONL/JSON file to validate.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary output path.")
    args = parser.parse_args(argv)

    if not args.rows.is_file():
        parser.error(f"rows file not found: {args.rows}")

    validations: list[dict[str, Any]] = []
    try:
        for line_number, row in _iter_rows(args.rows):
            status = validate_action_effect_payload(row if isinstance(row, Mapping) else {})
            status["line_number"] = line_number
            validations.append(status)
    except json.JSONDecodeError as exc:
        validations.append(
            {
                "schema": "tac.action_effect_validation.v1",
                "passed": False,
                "blockers": [f"action_effect_malformed_json:{exc.lineno}:{exc.colno}"],
                "line_number": exc.lineno,
            }
        )

    summary = {
        "schema": "tac.action_effect_validation_summary.v1",
        "rows_path": args.rows.as_posix(),
        "row_count": len(validations),
        "passed_count": sum(1 for row in validations if row.get("passed") is True),
        "failed_count": sum(1 for row in validations if row.get("passed") is not True),
        "rows": validations,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
