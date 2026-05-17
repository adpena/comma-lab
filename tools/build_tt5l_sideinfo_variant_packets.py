#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build byte-closed TT5L side-info variant archive packets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.tt5l_sideinfo_variant_packets import (  # noqa: E402
    DEFAULT_TT5L_SIDEINFO_VARIANT_JSON_PATH,
    DEFAULT_TT5L_SIDEINFO_VARIANT_OUTPUT_ROOT,
    DEFAULT_TT5L_SIDEINFO_VARIANT_REPORT_PATH,
    DEFAULT_TT5L_SIDEINFO_VARIANT_SUBMISSION_DIR,
    DEFAULT_TT5L_SOURCE_ARCHIVE_PATH,
    build_tt5l_sideinfo_variant_packets,
    render_tt5l_sideinfo_variant_packets_markdown,
    tt5l_sideinfo_variant_packets_json,
)


def _refuse_tmp(path: Path) -> None:
    text = str(path)
    if text.startswith("/tmp/") or "/private/tmp/" in text or "/var/tmp/" in text:
        raise ValueError(f"refusing to write TT5L side-info packet artifact to tmp: {text!r}")


def _write(path: Path, text: str) -> None:
    _refuse_tmp(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=Path(DEFAULT_TT5L_SOURCE_ARCHIVE_PATH),
        help="Source TT5L archive.zip containing a monolithic 0.bin or x member.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_TT5L_SIDEINFO_VARIANT_OUTPUT_ROOT),
        help="Directory where variant archive.zip packets will be written.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(DEFAULT_TT5L_SIDEINFO_VARIANT_JSON_PATH),
        help="Output variant-packet custody JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(DEFAULT_TT5L_SIDEINFO_VARIANT_REPORT_PATH),
        help="Output variant-packet custody markdown path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used for relative custody paths.",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=Path(DEFAULT_TT5L_SIDEINFO_VARIANT_SUBMISSION_DIR),
        help="Runtime submission directory shared by the variant packets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260517,
        help="Deterministic seed for random_lsb and shuffled controls.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        for path in (args.output_root, args.output_json, args.output_md):
            _refuse_tmp(path)
        payload = build_tt5l_sideinfo_variant_packets(
            source_archive=args.source_archive,
            output_root=args.output_root,
            repo_root=args.repo_root,
            seed=args.seed,
            submission_dir=args.submission_dir,
            command_argv=sys.argv,
        )
        _write(args.output_json, tt5l_sideinfo_variant_packets_json(payload))
        _write(args.output_md, render_tt5l_sideinfo_variant_packets_markdown(payload))
    except (OSError, ValueError) as exc:
        print(f"[tt5l-sideinfo-variants] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[tt5l-sideinfo-variants] "
        f"variant_count={payload['variant_count']} "
        f"blockers={payload['blockers']} "
        f"output_json={args.output_json} "
        f"output_md={args.output_md} "
        "score_claim=false promotion_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
