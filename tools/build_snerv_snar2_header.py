#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compatibility wrapper for the canonical SNeRV SNAR2 minimizer route.

The executable implementation lives in ``tools/minimize_snerv_snar_header.py``.
Keeping this wrapper avoids losing old operator commands while preventing a
second packet builder from drifting out of the train/export/runtime pipeline.
"""

from __future__ import annotations

import sys

try:
    from tools.minimize_snerv_snar_header import main as _minimize_main
except ModuleNotFoundError:  # pragma: no cover
    from minimize_snerv_snar_header import main as _minimize_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--wire-format" not in args:
        args = ["--wire-format", "snar2", *args]
    return _minimize_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
