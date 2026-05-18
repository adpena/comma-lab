#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a RATE-OP-1 stable-orbit packet-diet xray manifest.

The output is byte-custody planning evidence for Cathedral autopilot. It never
rewrites archives, never loads scorer code, never dispatches work, and never
claims a score.
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

from tac.contest_exploits.stable_orbit_packet_diet import (  # noqa: E402
    ArchiveInput,
    build_stable_orbit_packet_diet_xray,
    write_xray_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        action="append",
        required=True,
        help="Charged contest archive ZIP. Repeat for multi-archive xray.",
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
    parser.add_argument("--master-gradient-json", type=Path)
    parser.add_argument("--xray-json", type=Path)
    parser.add_argument("--hard-pair-json", type=Path)
    parser.add_argument("--sensitive-byte-json", type=Path)
    parser.add_argument("--boundary-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    archives = _archive_inputs(args.archive, args.axis_tag, args.label)
    overlays = {
        key: path
        for key, path in {
            "master_gradient": args.master_gradient_json,
            "xray": args.xray_json,
            "hard_pair": args.hard_pair_json,
            "sensitive_byte": args.sensitive_byte_json,
            "boundary": args.boundary_json,
        }.items()
        if path is not None
    }
    manifest = build_stable_orbit_packet_diet_xray(
        archives,
        overlays=overlays,
        generated_at_utc=dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    )
    paths = write_xray_manifest(manifest, args.output_dir)
    print(json.dumps({"paths": paths, "manifest": manifest}, indent=2, sort_keys=True))
    return 0


def _archive_inputs(
    archives: list[Path],
    axis_tags: list[str],
    labels: list[str],
) -> list[ArchiveInput]:
    if len(axis_tags) != len(archives):
        raise SystemExit("--axis-tag must be supplied once per --archive")
    if labels and len(labels) != len(archives):
        raise SystemExit("--label must be supplied zero times or once per --archive")
    return [
        ArchiveInput(
            path=archive,
            axis_tag=axis_tags[index],
            label=labels[index] if labels else "",
        )
        for index, archive in enumerate(archives)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
