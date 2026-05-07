#!/usr/bin/env python3
"""Normalize HDM3 HNeRV payloads before delegating to public PR106 inflate."""

from __future__ import annotations

import argparse
from pathlib import Path

from tac.hnerv_hdm3_runtime_adapter import restore_hdm3_file_to_legacy_brotli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument(
        "--legacy-brotli-quality",
        type=int,
        default=10,
        choices=range(12),
        metavar="{0..11}",
        help="Brotli quality for the restored legacy decoder section.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    restore_hdm3_file_to_legacy_brotli(
        input_path=args.input_path,
        output_path=args.output_path,
        json_out=args.json_out,
        brotli_quality=args.legacy_brotli_quality,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
