#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""C1/Z5/TT5L L5-v2 staircase probe-disambiguator CLI.

This is a planning-only arbitration surface. It consumes JSON observations from
paired exact probe artifacts and returns a fail-closed verdict. It never
authorizes dispatch, never claims score, and never promotes an archive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_probe_disambiguator import (  # noqa: E402
    build_probe_template,
    evaluate_l5_v2_probe,
    load_observations_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="JSON observations file; accepts a list or {'observations': [...]}.",
    )
    parser.add_argument(
        "--emit-template",
        action="store_true",
        help="Emit an input template instead of evaluating observations.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for the verdict/template JSON.",
    )
    return parser.parse_args(argv)


def _write_or_print(payload: dict[str, object], output_json: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_json is None:
        sys.stdout.write(text)
        return
    output_str = str(output_json)
    if (
        output_str.startswith("/tmp/")
        or "/private/tmp/" in output_str
        or "/var/tmp/" in output_str
    ):
        raise ValueError(f"refusing to write L5 v2 probe output to tmp: {output_str!r}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.emit_template:
        payload = build_probe_template()
    else:
        observations = (
            load_observations_json(args.input_json)
            if args.input_json is not None
            else ()
        )
        payload = evaluate_l5_v2_probe(observations)
    _write_or_print(payload, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
