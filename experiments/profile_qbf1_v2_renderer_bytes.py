#!/usr/bin/env python3
"""CPU-only QBF1-v2 renderer byte feasibility profile.

This script emits empirical/readiness JSON only.  It does not build an archive,
dispatch remote work, run CUDA, or make a score claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.qbf1_renderer_codec import (  # noqa: E402
    QBF1_V2_DEFAULT_BLOCK_SIZES,
    QBF1_V2_REFERENCE_QZS3_NBYTES,
    profile_qbf1_v2_renderer_bytes,
)


def parse_block_sizes(value: str) -> tuple[int, ...]:
    """Parse a comma-separated block-size list."""

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("block size list must not be empty")
    try:
        return tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid block size list: {value!r}") from exc


def build_profile(
    *,
    block_sizes: tuple[int, ...] = QBF1_V2_DEFAULT_BLOCK_SIZES,
    reference_qzs3_nbytes: int = QBF1_V2_REFERENCE_QZS3_NBYTES,
) -> dict[str, Any]:
    """Build the default renderer byte profile as JSON-compatible data."""

    return profile_qbf1_v2_renderer_bytes(
        block_sizes=block_sizes,
        reference_qzs3_nbytes=reference_qzs3_nbytes,
    )


def write_profile(
    output_json: Path,
    *,
    block_sizes: tuple[int, ...] = QBF1_V2_DEFAULT_BLOCK_SIZES,
    reference_qzs3_nbytes: int = QBF1_V2_REFERENCE_QZS3_NBYTES,
) -> dict[str, Any]:
    """Write the profile JSON and return it for tests/callers."""

    profile = build_profile(
        block_sizes=block_sizes,
        reference_qzs3_nbytes=reference_qzs3_nbytes,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block-sizes",
        type=parse_block_sizes,
        default=QBF1_V2_DEFAULT_BLOCK_SIZES,
        help="comma-separated block sizes to screen",
    )
    parser.add_argument(
        "--reference-qzs3-nbytes",
        type=int,
        default=QBF1_V2_REFERENCE_QZS3_NBYTES,
        help="current local QZS3 renderer raw byte reference",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional path to write the profile JSON",
    )
    args = parser.parse_args(argv)

    profile = build_profile(
        block_sizes=args.block_sizes,
        reference_qzs3_nbytes=args.reference_qzs3_nbytes,
    )
    text = json.dumps(profile, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
