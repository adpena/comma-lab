#!/usr/bin/env python3
"""Thin shell that forwards to the canonical tac CLI."""
from __future__ import annotations

import sys
from pathlib import Path


_script_dir = Path(__file__).resolve().parent
_repo = _script_dir.parent
if (_repo / "src" / "tac").exists():
    sys.path.insert(0, str(_repo))
    sys.path.insert(0, str(_repo / "src"))

from tac import cli as tac_cli  # noqa: E402


def main(argv: list[str] | None = None):
    args = list(sys.argv[1:] if argv is None else argv)
    return tac_cli.main(["lossy", *args])


if __name__ == "__main__":
    result = main()
    raise SystemExit(result if isinstance(result, int) else 0)
