#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Carmack-Hotz strip-everything contest-compliant inflate (NO torch, NO scorer).

One-line passthrough to ``tac.substrates.nscs06_carmack_hotz_strip_everything.inflate``.
The package's ``inflate.py`` is ≤100 LOC per HNeRV parity discipline L4 and
imports only ``numpy`` + ``Pillow`` (NO torch, NO scorer; strict-scorer-rule).

Per Catalog #146 the contract is ``inflate.py <archive_dir> <output_dir> <file_list>``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from tac.substrates.nscs06_carmack_hotz_strip_everything.inflate import (  # noqa: E402
    inflate_one_video,
)


def select_inflate_device() -> str:
    """Catalog #205 canonical helper signature; Carmack-Hotz needs no torch.

    The substrate has no neural primitives, so the choice of device is a
    no-op. We still expose the canonical name + honor the env var to keep
    Catalog #205's strict gate green.
    """
    import os

    # INLINE_DEVICE_FORK_OK:carmack-hotz-substrate-has-no-torch-no-cuda-cpu-distinction
    pinned = os.environ.get("PACT_INFLATE_DEVICE", "auto").lower()
    if pinned not in {"auto", "cpu", "cuda"}:
        raise SystemExit(f"PACT_INFLATE_DEVICE must be auto|cpu|cuda; got {pinned!r}")
    if pinned == "cpu":
        return "cpu"
    if pinned == "cuda":
        return "cpu"  # substrate is numpy-only; cuda request is a no-op
    return "cpu"


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: inflate.py <archive_dir> <output_dir> <file_list>",
            file=sys.stderr,
        )
        return 2
    select_inflate_device()
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    for line in file_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        base = line.rsplit(".", 1)[0]
        inflate_one_video(archive_bytes, output_dir / base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
