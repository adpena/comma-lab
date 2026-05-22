#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a decoder-q preserve/suppress response surface from family deltas."""

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

from tac.optimization.decoder_q_response_surface import (  # noqa: E402
    DecoderQResponseSurfaceError,
    build_decoder_q_response_surface,
    render_decoder_q_response_surface_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-delta", type=Path, required=True)
    parser.add_argument("--improvement-threshold", type=float, default=0.0)
    parser.add_argument("--regression-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        family_delta = json.loads(args.family_delta.read_text(encoding="utf-8"))
        surface = build_decoder_q_response_surface(
            family_delta,
            improvement_threshold=args.improvement_threshold,
            regression_threshold=args.regression_threshold,
            top_k=args.top_k,
        )
    except (OSError, ValueError, DecoderQResponseSurfaceError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(surface, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_decoder_q_response_surface_markdown(surface), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "summary": surface["summary"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
