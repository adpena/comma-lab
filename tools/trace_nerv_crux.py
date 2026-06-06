#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Trace where scorer-space signal dies in a NeRV training artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_crux_trace import (  # noqa: E402
    write_trace_rows_for_training_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-direct-live-posenet",
        action="store_true",
        help="Do not emit fail-closed blockers when PoseNet direct-live metrics are absent.",
    )
    parser.add_argument(
        "--allow-missing-direct-live-segnet",
        action="store_true",
        help="Do not emit fail-closed blockers when SegNet direct-live metrics are absent.",
    )
    parser.add_argument(
        "--require-direct-live-posenet",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--require-direct-live-segnet",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    write_trace_rows_for_training_artifact(
        args.training_artifact,
        output_path=args.out,
        require_direct_live_posenet=bool(args.require_direct_live_posenet)
        or not bool(args.allow_missing_direct_live_posenet),
        require_direct_live_segnet=bool(args.require_direct_live_segnet)
        or not bool(args.allow_missing_direct_live_segnet),
    )


if __name__ == "__main__":
    main()
