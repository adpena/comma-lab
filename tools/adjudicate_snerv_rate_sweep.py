#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Adjudicate SNeRV advisory/sweep JSON without granting score authority."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.analysis.snerv_rate_adjudication import (  # noqa: E402
    DEFAULT_PR101_FRONTIER_BYTES,
    build_snerv_rate_adjudication_payload,
)


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_rate_sweep_adjudication_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default=".omx/research/snerv_rate_sweep_20260601.json",
        help="SNeRV advisory/sweep JSON to adjudicate.",
    )
    parser.add_argument("--out", default=None, help="Output adjudication JSON path.")
    parser.add_argument(
        "--pr101-frontier-bytes",
        type=int,
        default=DEFAULT_PR101_FRONTIER_BYTES,
        help="Reference archive byte count for rate-only comparison.",
    )
    parser.add_argument(
        "--pose-preservation-ceiling",
        type=float,
        default=0.10,
        help="Advisory d_pose ceiling for distortion-promising classification.",
    )
    parser.add_argument(
        "--seg-preservation-ceiling",
        type=float,
        default=0.02,
        help="Advisory d_seg ceiling for distortion-promising classification.",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = REPO_ROOT / in_path
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    report = build_snerv_rate_adjudication_payload(
        payload,
        source_path=str(in_path),
        pr101_frontier_bytes=args.pr101_frontier_bytes,
        pose_preservation_ceiling=args.pose_preservation_ceiling,
        seg_preservation_ceiling=args.seg_preservation_ceiling,
    )

    out_path = Path(args.out) if args.out else _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    summary = report["summary"]
    print("[SNeRV rate adjudication] [macOS-CPU advisory] false-authority")
    print(f"  input: {in_path}")
    print(f"  rows: {summary['row_count']}")
    print(f"  classifications: {summary['classification_counts']}")
    print("  frontier_score_claim: false")
    print(f"  ready_for_exact_eval_dispatch: {report['ready_for_exact_eval_dispatch']}")
    print(f"  next: {summary['actionable_next_code_move']}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
