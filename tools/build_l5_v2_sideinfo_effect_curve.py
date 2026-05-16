#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the paired TT5L L5-v2 side-info effect-curve artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_measurement_schedule import (  # noqa: E402
    L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH,
)
from tac.optimization.l5_v2_sideinfo_effect_curve import (  # noqa: E402
    build_l5_v2_sideinfo_effect_curve,
    sideinfo_effect_curve_json,
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(f"refusing to write L5 v2 side-info effect curve to tmp: {text!r}")


def _read_cells(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        cells = payload
    elif isinstance(payload, dict):
        rows = payload.get("cells") or payload.get("observed_cells")
        cells = rows if isinstance(rows, list) else [payload]
    else:
        raise ValueError(f"cell JSON must be an object or list: {path}")
    out: list[dict[str, Any]] = []
    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(f"cell #{idx} in {path} is not an object")
        out.append(cell)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-json",
        action="append",
        type=Path,
        required=True,
        help=(
            "Input JSON object/list for one or more cells. Each cell must include "
            "axis, variant, and exact-eval evidence fields or an evidence object."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH),
        help="Output side-info effect-curve JSON path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used for custody path validation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cells: list[dict[str, Any]] = []
        for path in args.cell_json:
            cells.extend(_read_cells(path))
        output_json = args.output_json
        _refuse_tmp(output_json)
        payload = build_l5_v2_sideinfo_effect_curve(
            cells,
            repo_root=args.repo_root,
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(sideinfo_effect_curve_json(payload), encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[l5-v2-sideinfo-effect-curve] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[l5-v2-sideinfo-effect-curve] "
        f"predicate_passed={str(payload['predicate_passed']).lower()} "
        f"contract_blockers={len(payload['contract_blockers'])} "
        f"effect_blockers={len(payload['effect_blockers'])} "
        f"output={output_json} "
        "score_claim=false promotion_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
