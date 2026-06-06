#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Fail-closed launch arbiter for NeRV-family long MLX training runs.

Thin CLI over :mod:`tac.analysis.nerv_long_run_launch_gate`.  Exit codes:
0 = approved; 3 = blocked (default fail-closed); 0 with blocked verdict only
under ``--advisory``.  The JSON verdict is printed to stdout and optionally
written to ``--output-json``.

Example:

    .venv/bin/python tools/validate_nerv_long_run_gate.py \
        --family hinerv \
        --run-root /Volumes/VertigoDataTier/pact/experiments/results/<run> \
        --frontier-pointer .omx/state/canonical_frontier_pointer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.analysis.nerv_long_run_launch_gate import (  # noqa: E402
    SUPPORTED_FAMILIES,
    evaluate_nerv_long_run_launch_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=SUPPORTED_FAMILIES)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--frontier-pointer",
        default=".omx/state/canonical_frontier_pointer.json",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Report-only: exit 0 even when blocked (default is fail-closed).",
    )
    args = parser.parse_args()

    verdict = evaluate_nerv_long_run_launch_gate(
        family=args.family,
        run_root=args.run_root,
        frontier_pointer=args.frontier_pointer,
    )
    rendered = json.dumps(verdict, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        out = Path(args.output_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
    if verdict["approved"]:
        return 0
    return 0 if args.advisory else 3


if __name__ == "__main__":
    raise SystemExit(main())
