#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a RATE-OP-2 tropical argmax boundary feasibility report.

The output is a planning-only authority packet for Cathedral autopilot. It
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

from tac.contest_exploits.tropical_argmax_boundary_grammar import (  # noqa: E402
    BoundaryArchiveInput,
    build_tropical_argmax_boundary_feasibility,
    write_tropical_argmax_boundary_feasibility,
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
        "--segnet-boundary-marginals-json",
        type=Path,
        help="Optional analysis artifact from tools/build_segnet_boundary_marginals.py.",
    )
    parser.add_argument(
        "--sabor-stable-capacity-json",
        type=Path,
        help="Optional analysis artifact from tools/measure_segnet_argmax_stable_interior.py.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    archives = _archive_inputs(args.archive, args.axis_tag, args.label)
    features = {
        key: path
        for key, path in {
            "segnet_boundary_marginals": args.segnet_boundary_marginals_json,
            "sabor_stable_capacity": args.sabor_stable_capacity_json,
        }.items()
        if path is not None
    }
    manifest = build_tropical_argmax_boundary_feasibility(
        archives,
        boundary_features=features,
        generated_at_utc=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    )
    paths = write_tropical_argmax_boundary_feasibility(manifest, args.output_dir)
    print(json.dumps({"paths": paths, "manifest": manifest}, indent=2, sort_keys=True))
    return 0


def _archive_inputs(
    archives: list[Path],
    axis_tags: list[str],
    labels: list[str],
) -> list[BoundaryArchiveInput]:
    if len(axis_tags) != len(archives):
        raise SystemExit("--axis-tag must be supplied once per --archive")
    if labels and len(labels) != len(archives):
        raise SystemExit("--label must be supplied zero times or once per --archive")
    return [
        BoundaryArchiveInput(
            path=archive,
            axis_tag=axis_tags[index],
            label=labels[index] if labels else "",
        )
        for index, archive in enumerate(archives)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
