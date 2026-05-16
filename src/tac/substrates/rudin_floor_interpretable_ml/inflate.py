# SPDX-License-Identifier: MIT
"""Pure-Python inflate proof for the Rudin floor research scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

from tac.substrates.rudin_floor_interpretable_ml.archive import parse_archive


def inflate_one_video(
    archive_bytes: bytes,
    output_path: Path,
    *,
    features: dict[str, object] | None = None,
) -> Path:
    """Parse RDIF bytes and write a deterministic one-pixel PNG/PPM proof."""

    archive = parse_archive(archive_bytes)
    rgb = archive.rule_list.evaluate(features)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target = output_path if output_path.suffix else output_path.with_suffix(".png")
    try:
        from PIL import Image

        Image.new("RGB", (1, 1), rgb).save(target)
    except Exception:
        target = target.with_suffix(".ppm")
        target.write_bytes("P6\n1 1\n255\n".encode("ascii") + bytes(rgb))
    return target


def main() -> int:
    """Contest-shape CLI shim for research-only package smoke tests."""

    if len(sys.argv) != 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list = Path(sys.argv[3])
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    for line in file_list.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if clean:
            inflate_one_video(archive_bytes, output_dir / Path(clean).stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["inflate_one_video", "main"]
