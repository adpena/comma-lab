#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build a fail-closed SNeRV score-aware decoder-fit work order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.analysis.snerv_score_aware_decoder_fit_work_order import (  # noqa: E402
    build_snerv_decoder_fit_work_order,
)


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_score_aware_decoder_fit_work_order_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "adjudication",
        help="SNeRV rate/advisory adjudication JSON.",
    )
    parser.add_argument("--out", default=None, help="Output work-order JSON path.")
    parser.add_argument(
        "--lane-id",
        default="lane_snerv_score_aware_decoder_fit_20260601",
        help="Lane id to embed in the work order.",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.adjudication)
    if not in_path.is_absolute():
        in_path = REPO_ROOT / in_path
    source_bytes = in_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    adjudication = json.loads(source_bytes.decode("utf-8"))
    work_order = build_snerv_decoder_fit_work_order(
        adjudication,
        source_path=str(in_path),
        source_sha256=source_sha,
        lane_id=args.lane_id,
    )

    out_path = Path(args.out) if args.out else _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(work_order.as_jsonable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[SNeRV score-aware decoder-fit work order] false-authority")
    print(f"  source: {in_path}")
    print(f"  source_sha256: {source_sha}")
    print(f"  ready_for_local_decoder_fit_smoke: {work_order.ready_for_local_decoder_fit_smoke}")
    print(f"  ready_for_exact_eval_dispatch: {work_order.ready_for_exact_eval_dispatch}")
    print(f"  next: {work_order.next_action}")
    if work_order.blockers:
        print(f"  blockers: {list(work_order.blockers)}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
