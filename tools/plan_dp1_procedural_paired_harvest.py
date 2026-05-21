#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan DP1 procedural-codebook paired CPU/CUDA auth-eval harvests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tac.optimization.dp1_procedural_paired_harvest_plan import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    build_dp1_procedural_paired_harvest_plan,
    render_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-output-dir", type=Path)
    parser.add_argument("--procedural-output-dir", type=Path)
    parser.add_argument("--null-control-output-dir", type=Path)
    parser.add_argument(
        "--include-null-control",
        action="store_true",
        help="Require/report the optional null-exploit control arm even when no output dir is supplied.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root passed to tools/dispatch_modal_paired_auth_eval.py command templates.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    plan = build_dp1_procedural_paired_harvest_plan(
        output_dirs={
            "baseline": args.baseline_output_dir,
            "procedural": args.procedural_output_dir,
            "null_control": args.null_control_output_dir,
        },
        repo_root=REPO_ROOT,
        output_root=args.output_root,
        include_null_control=args.include_null_control,
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(plan), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
