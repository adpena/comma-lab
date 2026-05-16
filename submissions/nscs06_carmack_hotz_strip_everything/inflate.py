#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Carmack-Hotz contest-compliant inflate runtime (NO torch, NO scorer).

Self-contained per HNeRV parity discipline L4 + L9. The codec package
(archive parser + arithmetic decoder + per-pair render) is vendored into
the sibling ``_nscs06_codec/`` directory by
``experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py::_write_runtime``.
ZERO ``tac.*`` imports at inflate time; ZERO PACT repo dependency.

Per Catalog #146 the contract is
``inflate.py <archive_dir> <output_dir> <file_list>``.
Per Catalog #205 the canonical ``select_inflate_device`` helper is
exposed even though the substrate is numpy+Pillow only (no torch).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _nscs06_codec.inflate import inflate_one_video  # noqa: E402


def select_inflate_device() -> str:
    """Catalog #205 canonical helper; Carmack-Hotz needs no torch.

    The substrate has no neural primitives, so cuda vs cpu is a no-op.
    We still expose the canonical name + honor the env var to keep
    Catalog #205's strict gate green.
    """
    # INLINE_DEVICE_FORK_OK:carmack-hotz-substrate-has-no-torch-no-cuda-cpu-distinction
    pinned = os.environ.get("PACT_INFLATE_DEVICE", "auto").lower()
    if pinned not in {"auto", "cpu", "cuda"}:
        raise SystemExit(
            f"PACT_INFLATE_DEVICE must be auto|cpu|cuda; got {pinned!r}"
        )
    return "cpu"  # substrate is numpy-only; cuda is a no-op


def main() -> int:
    if len(sys.argv) != 4:
        print(
            'usage: inflate.py <archive_dir> <output_dir> <file_list>',
            file=sys.stderr,
        )
        return 2
    select_inflate_device()
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = (archive_dir / '0.bin').read_bytes()
    for line in file_list_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        base = line.rsplit('.', 1)[0]
        inflate_one_video(archive_bytes, output_dir / base)
    return 0


if __name__ == '__main__':
    sys.exit(main())
