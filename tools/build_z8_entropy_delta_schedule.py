#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a materializer-ready Z8 per-subband detail Δ schedule from RD rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.entropy_delta_schedule import (
    build_entropy_delta_schedule_from_headroom_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headroom-json", required=True, type=Path)
    parser.add_argument(
        "--max-subband-mse",
        required=True,
        type=float,
        help="Per-aggregate-subband quantization distortion ceiling used for RD selection.",
    )
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help=(
            "Permit schedules from sampled headroom reports to be marked materializer-ready. "
            "Default is fail-closed when pairs_measured < total_pairs_in_archive."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = json.loads(args.headroom_json.read_text())
    schedule = build_entropy_delta_schedule_from_headroom_report(
        report,
        max_subband_mse=float(args.max_subband_mse),
        require_full_archive_coverage=not args.allow_partial_coverage,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(schedule, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "schema": schedule["schema"],
                "ready_for_materializer": schedule["ready_for_materializer"],
                "schedule_sha256": schedule["schedule_sha256"],
                "step_count": len(schedule["entropy_detail_quantization_steps"]),
                "blockers": schedule["blockers"],
                "out_json": args.out_json.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
