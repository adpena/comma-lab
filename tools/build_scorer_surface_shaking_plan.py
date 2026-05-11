#!/usr/bin/env python3
"""Build the default local scorer-surface shaking plan.

The plan is CPU-only, deterministic, and planning-only. It ranks pixel/frame/
scorer perturbation families by local score economics, then records the
PacketIR/materialization gates required before any row can affect score claims.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.scorer_surface_shaking import (  # noqa: E402
    OperatingPoint,
    ScorerSurfacePlanError,
    build_scorer_surface_shaking_plan,
    render_markdown,
)
from tac.repo_io import json_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None, help="override operating-point label")
    parser.add_argument(
        "--device-axis",
        choices=("contest_cuda", "contest_cpu", "diagnostic_cuda", "diagnostic_cpu"),
        default=None,
    )
    parser.add_argument("--score", type=float, default=None)
    parser.add_argument("--archive-bytes", type=int, default=None)
    parser.add_argument("--archive-sha256", default=None)
    parser.add_argument("--seg-dist", type=float, default=None)
    parser.add_argument("--pose-dist", type=float, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser


def _operating_point_from_args(args: argparse.Namespace) -> OperatingPoint | None:
    values = {
        "label": args.label,
        "device_axis": args.device_axis,
        "score": args.score,
        "archive_bytes": args.archive_bytes,
        "archive_sha256": args.archive_sha256,
        "seg_dist": args.seg_dist,
        "pose_dist": args.pose_dist,
    }
    supplied = {key: value for key, value in values.items() if value is not None}
    if not supplied:
        return None
    required = ("label", "device_axis", "score", "archive_bytes", "seg_dist", "pose_dist")
    missing = [key for key in required if values[key] is None]
    if missing:
        raise ScorerSurfacePlanError(
            "custom operating point requires: " + ", ".join(missing)
        )
    return OperatingPoint(
        label=str(values["label"]),
        device_axis=str(values["device_axis"]),
        score=float(values["score"]),
        archive_bytes=int(values["archive_bytes"]),
        archive_sha256=str(values["archive_sha256"]) if values["archive_sha256"] else None,
        seg_dist=float(values["seg_dist"]),
        pose_dist=float(values["pose_dist"]),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_scorer_surface_shaking_plan(
            operating_point=_operating_point_from_args(args),
            max_rows=args.max_rows,
        )
    except ScorerSurfacePlanError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(plan), encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(plan), encoding="utf-8")
    if args.json_out is None and args.md_out is None:
        sys.stdout.write(json_text(plan) if args.format == "json" else render_markdown(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
