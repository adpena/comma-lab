# SPDX-License-Identifier: MIT
"""NSCS02 substrate inflate adapter (composition-friendly).

Substrate composition lanes (NSCS02 + NSCS01 + NSCS07 stack candidate
per audit composition matrix) call the substrate's package-level
``main(argv)``. This adapter delegates to the standalone submission
inflate at ``submissions/nscs02_downsampled_renderer/inflate.py``.

Per CLAUDE.md "Strict scorer rule" non-negotiable: NO PoseNet/SegNet
loaded at inflate time. The inflate path renders frames at (192, 256),
upsamples to (1164, 874), writes raw RGB. The contest scorer
re-interpolates to (384, 512) inside its own preprocess.
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


def _locate_nscs02_blob(archive_dir: Path) -> Path:
    """Locate the NSCS02 single-file payload inside an archive_dir.

    Per HNeRV parity discipline lesson 3 the NSCS02 archive ships as a
    single-file ``0.bin`` (or the legacy ``x`` member name carried by
    older PR101-derived adapters).
    """
    for name in ("0.bin", "nscs02.bin", "x"):
        candidate = archive_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"NSCS02 source blob not found in {archive_dir}")


def main(argv: list[str] | None = None) -> int:
    """Composition-friendly entry point: ``inflate.py archive_dir output_dir file_list``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        raise SystemExit("Usage: inflate.py <archive_dir> <output_dir> <file_list>")

    archive_dir = Path(args[0]).resolve()
    output_dir = Path(args[1]).resolve()
    file_list = Path(args[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nscs02_blob = _locate_nscs02_blob(archive_dir)
    nscs02_inflate = (
        _repo_root() / "submissions" / "nscs02_downsampled_renderer" / "inflate.py"
    )
    if not nscs02_inflate.is_file():
        raise FileNotFoundError(
            f"canonical NSCS02 inflate.py missing: {nscs02_inflate}"
        )

    for video_name in _read_file_list(file_list):
        stem = video_name.rsplit(".", 1)[0] if "." in video_name else video_name
        dst_raw = output_dir / f"{stem}.raw"
        subprocess.run(
            [
                os.environ.get("PYTHON", sys.executable),
                str(nscs02_inflate),
                str(nscs02_blob),
                str(dst_raw),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
