#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Select a compact-carrier model-size budget from measured ladder rows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import write_json  # noqa: E402
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (  # noqa: E402
    MODEL_SIZE_BUDGET_PLAN_SCHEMA,
    build_modelsize_budget_plan,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-json", required=True, type=Path)
    parser.add_argument("--rows-key", default="modelsize_budget_rows")
    parser.add_argument("--carrier-id", default="unknown")
    parser.add_argument("--baseline-id", default="pr95_hnerv")
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = _load_rows(args.rows_json, rows_key=str(args.rows_key))
    report = build_modelsize_budget_plan(
        rows,
        carrier_id=str(args.carrier_id),
        baseline_id=str(args.baseline_id),
    )
    out = args.output_json.expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = out.as_posix()
    write_json(out, report)
    print(
        json.dumps(
            {
                "schema": MODEL_SIZE_BUDGET_PLAN_SCHEMA,
                "report_path": out.as_posix(),
                "carrier_id": report["carrier_id"],
                "status": report["status"],
                "selected_archive_bytes": report.get("selected_archive_bytes"),
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def _load_rows(path: Path, *, rows_key: str) -> list[Mapping[str, Any]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        value = payload.get(rows_key)
        if value is None:
            value = payload.get("rows")
        if value is None:
            value = payload.get("variant_rows")
        rows = value
    else:
        raise SystemExit(f"rows JSON must be array or object: {path}")
    if not isinstance(rows, list):
        raise SystemExit(f"rows JSON does not contain list rows: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise SystemExit(f"rows JSON contains non-object rows: {path}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
