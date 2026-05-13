#!/usr/bin/env python3
"""Write the cooperative-receiver campaign queue.

This is a planning artifact, not a dispatch launcher.  Rows are forced through
the proxy false-authority contract so cross-domain predictions cannot leak into
score authority or exact-eval readiness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.optimization.cooperative_receiver_campaigns import (
    build_campaign_queue,
    render_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cooperative_receiver/campaign_queue.json"),
        help="JSON manifest output path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/cooperative_receiver/campaign_queue.md"),
        help="Markdown summary output path.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_campaign_queue(top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(manifest), encoding="utf-8")

    print(
        "wrote cooperative_receiver_campaign_queue "
        f"rows={len(manifest['top_k'])} dispatch_ready=0 "
        f"score_claim={str(manifest['score_claim']).lower()} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
