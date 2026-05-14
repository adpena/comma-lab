#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write the cooperative-receiver solver integration manifest.

This is the operator-visible bridge from research campaigns into autopilot,
meta-Lagrangian, Pareto, continual-learning, xray, magic-codec, and deterministic
packet-compiler surfaces. It is planning-only and creates no score authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.optimization.cooperative_receiver_integration import (  # noqa: E402
    build_integration_manifest,
    render_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cooperative_receiver/integration_manifest.json"),
        help="JSON integration manifest output path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/cooperative_receiver/integration_manifest.md"),
        help="Markdown summary output path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if manifest construction or proxy-safety checks fail.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_integration_manifest()
    except Exception as exc:
        if args.strict:
            raise
        print(f"failed to build cooperative receiver integration manifest: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(manifest), encoding="utf-8")

    print(
        "wrote cooperative_receiver_integration "
        f"campaigns={manifest['campaign_count']} "
        f"autopilot_rows={len(manifest['autopilot_dispatch_hook']['rows'])} "
        f"xray_grammars={len(manifest['xray_hook']['cooperative_receiver_packet_grammars'])} "
        f"score_claim={str(manifest['score_claim']).lower()} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
