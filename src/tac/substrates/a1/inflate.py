# SPDX-License-Identifier: MIT
"""Canonical A1 inflate adapter for composed substrate runtimes.

The source of truth for A1 contest decode remains ``submissions/a1/inflate.py``.
This adapter provides the package-level ``main(argv)`` contract expected by
D1 and other composition lanes: ``archive_dir output_dir file_list``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_file_list(file_list_path: Path) -> list[str]:
    if not file_list_path.is_file():
        raise FileNotFoundError(f"file_list not found: {file_list_path}")
    names = [
        line.strip()
        for line in file_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not names:
        raise ValueError(f"file_list {file_list_path} is empty")
    return names


def _locate_a1_blob(archive_dir: Path) -> Path:
    for name in ("a1.bin", "x"):
        candidate = archive_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"A1 source blob not found in {archive_dir}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        raise SystemExit("Usage: inflate.py <archive_dir> <output_dir> <file_list>")

    archive_dir = Path(args[0]).resolve()
    output_dir = Path(args[1]).resolve()
    file_list = Path(args[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    a1_blob = _locate_a1_blob(archive_dir)
    a1_inflate = _repo_root() / "submissions" / "a1" / "inflate.py"
    if not a1_inflate.is_file():
        raise FileNotFoundError(f"canonical A1 inflate.py missing: {a1_inflate}")

    for video_name in _read_file_list(file_list):
        stem = video_name.rsplit(".", 1)[0] if "." in video_name else video_name
        dst_raw = output_dir / f"{stem}.raw"
        subprocess.run(
            [
                os.environ.get("PYTHON", sys.executable),
                str(a1_inflate),
                str(a1_blob),
                str(dst_raw),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
