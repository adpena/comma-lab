#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed CPU/MPS architecture sweep recommendation manifest.

The output is a planner artifact only. It never marks rows dispatchable or
promotion eligible; exact score use still requires a built archive and full
contest CUDA auth eval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.architecture_sweep_recommendations import (  # noqa: E402
    build_architecture_sweep_recommendations,
    json_text,
    load_dispatch_claims_text,
    load_lightning_active_jobs,
    load_manifest_sources,
    render_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Input JSON manifest. Repeat for multiple CPU/MPS planning artifacts.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Recommendation manifest JSON output")
    parser.add_argument("--markdown-output", type=Path, help="Optional Markdown summary output")
    parser.add_argument("--run-id", required=True, help="Stable run id for provenance")
    parser.add_argument(
        "--lightning-active-jobs",
        type=Path,
        default=REPO_ROOT / ".omx/state/lightning_active_jobs.json",
        help="Optional local Lightning active-jobs state JSON",
    )
    parser.add_argument(
        "--dispatch-claims",
        type=Path,
        default=REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md",
        help="Optional dispatch claim ledger Markdown",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_architecture_sweep_recommendations(
        load_manifest_sources(args.input),
        run_id=args.run_id,
        lightning_active_jobs=load_lightning_active_jobs(args.lightning_active_jobs),
        dispatch_claims_markdown=load_dispatch_claims_text(args.dispatch_claims),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json_text(manifest), encoding="utf-8")
    print(f"wrote {args.output}")

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(manifest), encoding="utf-8")
        print(f"wrote {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
