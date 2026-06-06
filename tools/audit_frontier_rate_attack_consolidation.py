#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit the legacy final-rate stack as the single score-program compiler surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from comma_lab.scheduler.frontier_rate_attack_consolidation import (  # noqa: E402
    build_frontier_rate_attack_consolidation_audit,
    render_frontier_rate_attack_consolidation_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to audit.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Accepted by the shared all-lanes gate contract; audit is always strict.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for a JSON copy of the audit.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        help="Optional path for a Markdown/text copy of the audit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = build_frontier_rate_attack_consolidation_audit(args.repo_root)
    text = render_frontier_rate_attack_consolidation_audit(audit)
    json_text = json.dumps(audit, indent=2, sort_keys=True) + "\n"

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text, encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(text, encoding="utf-8")

    if args.format == "json":
        sys.stdout.write(json_text)
    else:
        sys.stdout.write(text)
    return 0 if audit.get("status") == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
