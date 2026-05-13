#!/usr/bin/env python3
"""Prove PR106/R2 sidecar bytes are consumed by the submission runtime decoder.

This is a no-score, no-promotion proof. It imports the selected submission
``inflate.py`` and runs only its sidecar parser/decoder on the source archive
and on a valid one-correction mutation. It does not run full HNeRV inflate and
does not evaluate a score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler import (  # noqa: E402
    dumps_runtime_consumption_manifest,
    prove_pr106_sidecar_runtime_decode_consumption,
)
from tac.repo_io import json_text  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to archive.zip containing a single stored 0.bin or x member.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        required=True,
        help="Submission directory containing inflate.py and src/.",
    )
    parser.add_argument(
        "--member-name",
        default=None,
        help=(
            "Expected ZIP member name. Omit to auto-detect the known single-member "
            "packet names: 0.bin or x."
        ),
    )
    parser.add_argument(
        "--expected-archive-sha256",
        help="Optional expected archive SHA-256. A mismatch fails closed.",
    )
    parser.add_argument(
        "--expected-runtime-source-tree-sha256",
        help=(
            "Optional expected SHA-256 over inflate.py plus recognized src/*.py "
            "runtime files. A mismatch fails closed."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for the proof manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prove_pr106_sidecar_runtime_decode_consumption(
        archive_path=args.archive,
        runtime_dir=args.runtime_dir,
        expected_member_name=args.member_name,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_runtime_source_tree_sha256=args.expected_runtime_source_tree_sha256,
    )
    text = dumps_runtime_consumption_manifest(manifest)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text(manifest), encoding="utf-8")
    sys.stdout.write(text)
    if manifest.get("blockers") or not manifest.get("runtime_sidecar_decode_consumption_claim"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
