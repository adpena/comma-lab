#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove PR106/R2 sidecar PacketIR identity without claiming score.

This is the operator-facing wrapper around
``tac.packet_compiler.prove_pr106_sidecar_packet_ir_identity``. It proves that
the PR106 sidecar wrapper can be parsed into typed PacketIR sections and
re-emitted byte-for-byte, with every payload byte accounted for by the parser.
It does not import the submission runtime, render frames, or evaluate a score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler import prove_pr106_sidecar_packet_ir_identity  # noqa: E402
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
        "--output-json",
        type=Path,
        help="Optional path for the proof manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prove_pr106_sidecar_packet_ir_identity(
        archive_path=args.archive,
        expected_member_name=args.member_name,
        expected_archive_sha256=args.expected_archive_sha256,
    )
    text = json_text(manifest)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0 if manifest.get("packet_ir_identity_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
