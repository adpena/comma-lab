#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Audit 5D canvas coverage and emit queue-consumable densification orders."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_coverage import (  # noqa: E402
    CanvasCoverageAuditError,
    audit_5d_canvas_coverage_json,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_5d_canvas_coverage",
        description=(
            "Audit a populated pair-frame scorer-geometry 5D canvas and emit "
            "false-authority work orders for the next acquisition wave."
        ),
    )
    parser.add_argument("--canvas-path", required=True, type=Path)
    parser.add_argument(
        "--min-pair-coverage",
        type=float,
        default=0.05,
        help="Grouped-search pair coverage floor in [0, 1] (default 0.05).",
    )
    parser.add_argument(
        "--min-frame-coverage",
        type=float,
        default=0.05,
        help="Grouped-search frame coverage floor in [0, 1] (default 0.05).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to stdout.",
    )
    return parser


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = audit_5d_canvas_coverage_json(
            args.canvas_path,
            min_pair_coverage_for_grouped_search=args.min_pair_coverage,
            min_frame_coverage_for_grouped_search=args.min_frame_coverage,
        )
    except CanvasCoverageAuditError as exc:
        print(f"[audit_5d_canvas_coverage] FATAL: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        output = args.output if args.output.is_absolute() else _REPO_ROOT / args.output
        _write_text_atomic(output, text)
        print(f"[audit_5d_canvas_coverage] OK: wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
