#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a DP1 live Modal status packet from synced baseline/procedural logs."""

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

from tac.optimization.dp1_live_modal_status import (  # noqa: E402
    build_dp1_live_modal_status,
    build_dp1_modal_call_status,
    render_markdown,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--procedural-log", type=Path)
    parser.add_argument("--baseline-metadata", type=Path)
    parser.add_argument("--procedural-metadata", type=Path)
    parser.add_argument(
        "--get-timeout-seconds",
        type=float,
        default=2.0,
        help="Short Modal FunctionCall.get timeout for metadata polling mode.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args(argv)

    if args.baseline_metadata and args.procedural_metadata:
        status = build_dp1_modal_call_status(
            baseline_metadata=args.baseline_metadata,
            procedural_metadata=args.procedural_metadata,
            repo_root=REPO_ROOT,
            timeout_seconds=args.get_timeout_seconds,
        )
    elif args.baseline_log and args.procedural_log:
        status = build_dp1_live_modal_status(
            baseline_log=args.baseline_log,
            procedural_log=args.procedural_log,
            repo_root=REPO_ROOT,
        )
    else:
        parser.error(
            "provide either --baseline-log/--procedural-log or "
            "--baseline-metadata/--procedural-metadata"
        )
    if args.json_out:
        write_json(args.json_out, status)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(status), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status.get("status") in {"running", "finished"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
