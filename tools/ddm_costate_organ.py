#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build or render the read-only live DDM costate organ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.ddm_costate_organ import (  # noqa: E402
    build_live_ddm_costate,
    digest_lines,
    write_receipt_atomic,
)


def _resume_payload(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("resume_state"), dict):
        raise ValueError("--resume-from must name a DDM costate receipt with resume_state")
    return payload["resume_state"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the complete advisory cycle")
    parser.add_argument(
        "--write-receipt",
        type=Path,
        help="atomically write a new durable advisory receipt; refuses overwrite",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="resume from a prior receipt only when all source hashes still match",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        report = build_live_ddm_costate(
            repo_root=args.repo_root.resolve(),
            resume_state=_resume_payload(args.resume_from),
        )
        if args.write_receipt is not None:
            target = args.write_receipt
            if not target.is_absolute():
                target = args.repo_root / target
            if target.exists():
                raise FileExistsError(f"refusing to overwrite append-only receipt: {target}")
            write_receipt_atomic(target, report)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("\n".join(digest_lines(report)))
            if args.write_receipt is not None:
                print(f"receipt: {args.write_receipt}")
        return 0 if report.get("available") else 2
    except Exception as exc:
        print(f"ddm-costate-organ: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
