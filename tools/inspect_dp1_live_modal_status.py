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
    render_markdown,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--procedural-log", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args(argv)

    status = build_dp1_live_modal_status(
        baseline_log=args.baseline_log,
        procedural_log=args.procedural_log,
        repo_root=REPO_ROOT,
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
