#!/usr/bin/env python
"""Journal an operator's disqualification or reinstatement; then refresh explicitly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.frontier_disqualifications import REASON_CLASSES, append_disqualification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--reason-class", choices=REASON_CLASSES)
    parser.add_argument("--evidence")
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--who")
    parser.add_argument("--reinstate", action="store_true")
    args = parser.parse_args(argv)
    try:
        row = append_disqualification(
            args.repo_root, lane_id=args.lane, archive_sha256=args.archive_sha256,
            reason_class=args.reason_class, evidence=args.evidence,
            rationale=args.rationale, who=args.who, reinstate=args.reinstate,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"REFUSED: {exc}\n")
    print(json.dumps({"event": row, "next": "Run tools/refresh_canonical_frontier.py"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
