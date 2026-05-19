#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove PR101/FEC6 FP11 PacketIR identity without claiming score."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.pr101_fec6_packetir import (  # noqa: E402
    PR101_FEC6_DEFAULT_MEMBER_NAME,
    prove_pr101_fec6_packetir_identity,
    render_pr101_fec6_packetir_identity_markdown,
)
from tac.repo_io import json_text  # noqa: E402

DEFAULT_FEC6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_FEC6_ARCHIVE,
        help="Path to archive.zip containing the single PR101/FEC6 x member.",
    )
    parser.add_argument(
        "--member-name",
        default=PR101_FEC6_DEFAULT_MEMBER_NAME,
        help="Expected ZIP member name. Defaults to x for the PR101/FEC6 packet.",
    )
    parser.add_argument(
        "--expected-archive-sha256",
        help="Optional expected archive SHA-256. A mismatch fails closed.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for the JSON proof manifest. JSON is always written to stdout.",
    )
    parser.add_argument(
        "--output-md",
        "--output-markdown",
        dest="output_md",
        type=Path,
        help="Optional path for a markdown summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    proof = prove_pr101_fec6_packetir_identity(
        archive_path=args.archive,
        expected_member_name=args.member_name,
        expected_archive_sha256=args.expected_archive_sha256,
    )
    text = json_text(proof)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            render_pr101_fec6_packetir_identity_markdown(proof),
            encoding="utf-8",
        )
    sys.stdout.write(text)
    return 0 if proof.get("packet_ir_identity_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
