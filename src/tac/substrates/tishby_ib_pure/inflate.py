# SPDX-License-Identifier: MIT
"""Inflate stub for the Tishby IB-pure L1 scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from tac.substrates.tishby_ib_pure.archive import parse_archive


def inflate_one_video(archive_bytes: bytes, output_path: str | Path) -> Path:
    """Write a deterministic proof artifact after parsing TIBP1 bytes.

    This is not a contest video decoder. It exists so archive bytes are consumed
    by an importable inflate surface while the lane remains research-only.
    """

    parsed = parse_archive(archive_bytes)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = parsed.meta
    marker = (
        "TIBP1 research_only inflate proof\n"
        f"sha256={parsed.content_sha256}\n"
        f"score_claim={meta.get('score_claim', False)}\n"
    )
    out.write_text(marker, encoding="utf-8")
    return out


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    inflate_one_video(args.archive.read_bytes(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
