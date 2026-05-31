#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark a Z8 candidate through its full ``inflate.sh`` runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.substrates.z8_hierarchical_predictive_coding.inflate_runtime_benchmark import (
    benchmark_z8_submission_inflate_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inflate-sh", required=True, type=Path)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Directory passed as inflate.sh argv[1]; defaults to inflate.sh parent.",
    )
    parser.add_argument("--file-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--auth-eval-window-seconds", type=float, default=1800.0)
    parser.add_argument("--inflate-device", default="cpu")
    parser.add_argument("--out-json", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    archive_dir = args.archive_dir if args.archive_dir is not None else args.inflate_sh.parent
    report = benchmark_z8_submission_inflate_runtime(
        inflate_sh=args.inflate_sh,
        archive_dir=archive_dir,
        file_list=args.file_list,
        output_dir=args.output_dir,
        repeat=int(args.repeat),
        timeout_seconds=float(args.timeout_seconds),
        auth_eval_window_seconds=float(args.auth_eval_window_seconds),
        inflate_device=str(args.inflate_device),
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
