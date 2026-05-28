#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a rate-only FP11 source split-Brotli recode candidate packet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    _TOOL_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _TOOL_DIR.parent
    for _path in (str(_REPO_ROOT), str(_TOOL_DIR)):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.fp11_source_brotli_recode import (  # noqa: E402
    DEFAULT_BROTLI_LGWINS,
    DEFAULT_BROTLI_QUALITIES,
    Fp11SourceBrotliRecodeError,
    build_fp11_source_brotli_recode_candidate,
)
from tac.repo_io import json_text  # noqa: E402


def _ints(values: list[str] | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return default
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return tuple(out)


def _lgwins(
    values: list[str] | None,
    *,
    default: tuple[int | None, ...],
) -> tuple[int | None, ...]:
    if not values:
        return default
    out: list[int | None] = []
    for value in values:
        for part in value.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if part in {"none", "default"}:
                out.append(None)
            else:
                out.append(int(part))
    return tuple(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-submission-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quality", action="append", default=[])
    parser.add_argument("--lgwin", action="append", default=[])
    parser.add_argument("--full-frame-inflate-parity-proof", type=Path)
    parser.add_argument(
        "--allow-nonpositive-candidate",
        action="store_true",
        help=(
            "Emit a non-authoritative zero-delta manifest when the sweep finds no "
            "rate-positive recode. Safety and roundtrip failures remain fatal."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_fp11_source_brotli_recode_candidate(
            source_submission_dir=args.source_submission_dir,
            output_dir=args.output_dir,
            qualities=_ints(args.quality, default=DEFAULT_BROTLI_QUALITIES),
            lgwins=_lgwins(args.lgwin, default=DEFAULT_BROTLI_LGWINS),
            full_frame_inflate_parity_proof=args.full_frame_inflate_parity_proof,
            allow_nonpositive_candidate=args.allow_nonpositive_candidate,
            allow_overwrite=args.overwrite,
        )
    except (Fp11SourceBrotliRecodeError, OSError, ValueError) as exc:
        print(f"FATAL: FP11 source Brotli recode candidate build failed: {exc}", file=sys.stderr)
        return 2
    print(json_text(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
