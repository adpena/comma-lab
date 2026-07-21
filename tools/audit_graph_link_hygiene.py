#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rebuild and measure graph-memory link hygiene over the canonical 8k corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tac.graph_memory import load_or_build  # noqa: E402
from tac.graph_memory.link_hygiene import measure_link_hygiene  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="force a cache + Obsidian refresh")
    parser.add_argument("--output", type=Path, help="optional JSON receipt path")
    args = parser.parse_args(argv)
    payload = measure_link_hygiene(load_or_build(rebuild=args.rebuild))
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
