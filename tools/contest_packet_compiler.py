#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Contest packet compiler/checker CLI.

This is a stable operator-facing alias for
``tac.packet_compiler.deterministic_compiler``. It emits byte-custody and
conformance manifests only; it never runs scorers, claims score, or dispatches.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.deterministic_compiler_cli import main as _main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return _main(argv, label="contest-packet-compiler")


if __name__ == "__main__":
    raise SystemExit(main())
