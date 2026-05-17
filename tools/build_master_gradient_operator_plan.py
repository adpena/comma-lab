#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build grammar-aware master-gradient operator rows from an archive layout."""

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

from tac.frontier_archive_layout import inspect_frontier_archive_layout  # noqa: E402
from tac.master_gradient_operator_plan import build_master_gradient_operator_plan_payload  # noqa: E402
from tac.repo_io import json_text, read_json, write_json  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--archive",
        type=Path,
        help="Archive ZIP to inspect before building operator rows.",
    )
    source.add_argument(
        "--layout-json",
        type=Path,
        help="Existing tac_frontier_archive_layout_v1 JSON manifest.",
    )
    parser.add_argument(
        "--axis-label",
        choices=["contest_cpu", "contest_cuda", "paired_contest_cpu_cuda", "diagnostic"],
        default="paired_contest_cpu_cuda",
    )
    parser.add_argument(
        "--packet-proofs-available",
        action="store_true",
        help=(
            "Mark rows probe-ready only when a concrete mutation builder has already "
            "proved repack, ZIP metadata/CRC, inflate, and byte-consumption closure."
        ),
    )
    parser.add_argument("--output-json", type=Path, help="Optional destination for JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layout = inspect_frontier_archive_layout(args.archive) if args.archive else read_json(args.layout_json)
    plan = build_master_gradient_operator_plan_payload(
        layout,
        axis_label=args.axis_label,
        packet_proofs_available=args.packet_proofs_available,
    )
    if args.output_json:
        write_json(args.output_json, plan)
    else:
        sys.stdout.write(json_text(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
