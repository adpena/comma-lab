#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build, parse, and bounded-decode counted C2 integer-plane archives."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tac.boundary_math.integer_plane_emitter_byte_close import (
    C2ByteCloseError,
    archive_receipt,
    build_counted_archive,
    compare_capped_archives,
    decode_counted_archive,
    parse_counted_archive,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--base-archive", type=Path, required=True)
    build.add_argument("--checkpoint", type=Path, required=True)
    build.add_argument("--pdw2-packet", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--authority", choices=("ema", "live"), default="ema")
    build.add_argument("--receipt", type=Path, required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--archive", type=Path, required=True)
    inspect.add_argument("--receipt", type=Path, required=True)
    decode = subparsers.add_parser("decode")
    decode.add_argument("--archive", type=Path, required=True)
    decode.add_argument("--base-decoder", type=Path, required=True)
    decode.add_argument("--scratch-root", type=Path, required=True)
    decode.add_argument("--pair-cap", type=int, required=True)
    decode.add_argument("--output-raw", type=Path, required=True)
    decode.add_argument("--workers", type=int, default=1)
    decode.add_argument("--receipt", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--pre-archive", type=Path, required=True)
    compare.add_argument("--post-archive", type=Path, required=True)
    compare.add_argument("--base-decoder", type=Path, required=True)
    compare.add_argument("--cache", type=Path, required=True)
    compare.add_argument("--upstream", type=Path, required=True)
    compare.add_argument("--scratch-root", type=Path, required=True)
    compare.add_argument("--output-root", type=Path, required=True)
    compare.add_argument("--pair-cap", type=int, required=True)
    compare.add_argument("--cpu-threads", type=int, default=1)
    compare.add_argument("--receipt", type=Path, required=True)
    return parser


def _write(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise C2ByteCloseError(f"receipt overwrite refused: {path}")
    path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="ascii")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            receipt = build_counted_archive(
                base_archive=args.base_archive,
                checkpoint_path=args.checkpoint,
                output=args.output,
                pdw2_packet=args.pdw2_packet.expanduser().resolve(strict=True).read_bytes(),
                authority=args.authority,
            )
        elif args.command == "inspect":
            receipt = archive_receipt(parse_counted_archive(args.archive))
        elif args.command == "decode":
            receipt = decode_counted_archive(
                archive=args.archive,
                base_decoder=args.base_decoder,
                scratch_root=args.scratch_root,
                pair_cap=args.pair_cap,
                output_raw=args.output_raw,
                workers=args.workers,
            )
        else:
            receipt = compare_capped_archives(
                pre_archive=args.pre_archive,
                post_archive=args.post_archive,
                base_decoder=args.base_decoder,
                cache=args.cache,
                upstream=args.upstream,
                scratch_root=args.scratch_root,
                output_root=args.output_root,
                pair_cap=args.pair_cap,
                cpu_threads=args.cpu_threads,
            )
        _write(args.receipt, receipt)
    except (C2ByteCloseError, OSError, ValueError) as exc:
        print(f"C2 byte-close refusal: {exc}", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
