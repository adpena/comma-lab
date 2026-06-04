#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write a SNeRV skip-high mode comparison artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.analysis.snerv_skip_high_mode_compare import (  # noqa: E402
    DEFAULT_HARD_BYTE_CEILING,
    write_skip_high_mode_comparison,
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    binary_profiles = _labeled_paths(args.binary_profile)
    prefilter_profiles = _labeled_paths(args.prefilter_profile)
    payload = write_skip_high_mode_comparison(
        output_json=args.output_json,
        output_md=args.output_md,
        binary_profiles=binary_profiles,
        prefilter_profiles=prefilter_profiles,
        hard_byte_ceiling=int(args.hard_byte_ceiling),
    )
    print(
        "wrote "
        f"{args.output_json} verdict={payload['verdict']} "
        f"rows={len(payload['binary_profile_rows'])} "
        f"prefilters={len(payload['prefilter_profile_rows'])}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary-profile",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="SNeRV snerv_binary_profile.json with a human label.",
    )
    parser.add_argument(
        "--prefilter-profile",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional local MLX prefilter profile with a human label.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--hard-byte-ceiling",
        type=int,
        default=DEFAULT_HARD_BYTE_CEILING,
    )
    return parser


def _labeled_paths(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"empty label in {value!r}")
        out[label] = Path(raw_path).expanduser()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
