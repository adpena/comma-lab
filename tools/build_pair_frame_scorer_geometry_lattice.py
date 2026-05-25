#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a pair-frame scorer-geometry lattice for DQS1 start selection."""

# ruff: noqa: E402

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

from tac.optimization.pair_frame_scorer_geometry_lattice import (
    PairFrameScorerGeometryLatticeError,
    build_pair_frame_scorer_geometry_lattice,
    load_json_object,
    render_markdown,
    write_json,
)


def _parse_csv_ints(text: str | None) -> list[int] | None:
    if text is None:
        return None
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return values or None


def _load_optional(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return load_json_object(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairset-acquisition", type=Path, required=True)
    parser.add_argument("--frame-pair-curriculum", type=Path)
    parser.add_argument("--pair-component-xray", type=Path, action="append", default=[])
    parser.add_argument("--drop-counts", default="3,4,6,8,12,16")
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_pair_frame_scorer_geometry_lattice(
            load_json_object(args.pairset_acquisition),
            frame_pair_curriculum=_load_optional(args.frame_pair_curriculum),
            pair_component_xrays=tuple(
                load_json_object(path) for path in args.pair_component_xray
            ),
            drop_counts=_parse_csv_ints(args.drop_counts),
            max_requests=args.max_requests,
        )
        write_json(args.json_out, payload)
        if args.md_out is not None:
            args.md_out.parent.mkdir(parents=True, exist_ok=True)
            args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    except (
        OSError,
        json.JSONDecodeError,
        PairFrameScorerGeometryLatticeError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "row_count": payload["summary"]["row_count"],
                "queue_executable_request_count": payload["summary"][
                    "queue_executable_request_count"
                ],
                "geometry_coverage": payload["coverage"]["geometry_coverage"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
