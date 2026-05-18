#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a RATE-OP-3 decoy/mosaic residual-basis route entropy report.

The output is a planning-only authority packet for Cathedral autopilot.  It
does not rewrite archives, load scorer code, dispatch work, or claim score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.contest_exploits.decoy_mosaic_residual_basis import (  # noqa: E402
    DecoyMosaicArchiveInput,
    RouteProbeConfig,
    build_decoy_mosaic_residual_basis_probe,
    write_decoy_mosaic_residual_basis_probe,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        action="append",
        required=True,
        help="Charged contest archive ZIP. Repeat for multi-archive feasibility.",
    )
    parser.add_argument(
        "--axis-tag",
        action="append",
        required=True,
        help="Axis tag for each --archive, e.g. [contest-CPU] or [contest-CUDA].",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label for each --archive. Defaults to archive stem.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--route-features-json",
        type=Path,
        help="Optional per-pair feature report for the 10-50 pair cheap probe.",
    )
    parser.add_argument(
        "--single-monolith-control-json",
        type=Path,
        help="Optional monolith-control report on the same archive family.",
    )
    parser.add_argument(
        "--specialist-head-bytes",
        type=int,
        help="Measured specialist-head payload bytes, if already known.",
    )
    parser.add_argument("--pair-count", type=int, default=600)
    parser.add_argument("--route-count", type=int, default=4)
    parser.add_argument("--cheap-probe-pair-cap", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    archives = _archive_inputs(args.archive, args.axis_tag, args.label)
    config = RouteProbeConfig(
        pair_count=args.pair_count,
        route_count=args.route_count,
        cheap_probe_pair_cap=args.cheap_probe_pair_cap,
    )
    manifest = build_decoy_mosaic_residual_basis_probe(
        archives,
        route_features_path=args.route_features_json,
        config=config,
        generated_at_utc=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        single_monolith_control_path=args.single_monolith_control_json,
        specialist_head_bytes=args.specialist_head_bytes,
    )
    paths = write_decoy_mosaic_residual_basis_probe(manifest, args.output_dir)
    print(json.dumps({"paths": paths, "manifest": manifest}, indent=2, sort_keys=True))
    return 0


def _archive_inputs(
    archives: list[Path],
    axis_tags: list[str],
    labels: list[str],
) -> list[DecoyMosaicArchiveInput]:
    if len(axis_tags) != len(archives):
        raise SystemExit("--axis-tag must be supplied once per --archive")
    if labels and len(labels) != len(archives):
        raise SystemExit("--label must be supplied zero times or once per --archive")
    return [
        DecoyMosaicArchiveInput(
            path=archive,
            axis_tag=axis_tags[index],
            label=labels[index] if labels else "",
        )
        for index, archive in enumerate(archives)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
