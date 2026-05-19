#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit substrate trainers for canonical optimization-helper consumption."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.trainer_optimization_helper_audit import (  # noqa: E402
    audit_trainer_optimization_helpers,
)


def _render_table(payload: dict[str, object]) -> str:
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        return "(no train_substrate_*.py files found)"
    out = [
        "STATUS             TRAINER                                      REASON",
        "-----------------  -------------------------------------------  --------------------------------------------",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "?"))
        trainer = str(row.get("trainer_id", "?"))
        reason = str(row.get("reason", "?"))
        out.append(f"{status[:17]:17}  {trainer[:43]:43}  {reason}")
    return "\n".join(out)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any trainer is missing helper usage or waiver.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    audit = audit_trainer_optimization_helpers(args.repo_root)
    payload = audit.to_json_obj()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_table(payload))
        print()
        print(
            "summary: "
            f"scanned={audit.scanned_trainers} "
            f"accepted={audit.accepted_trainers} "
            f"missing={audit.missing_trainers} "
            f"waived={audit.waived_trainers}"
        )
    return 1 if args.strict and audit.violations else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
