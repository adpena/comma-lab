#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove PR101/FEC6 packet bytes are consumed by the submission runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.pr101_fec6_runtime_consumption import (  # noqa: E402
    dumps_pr101_fec6_runtime_consumption_proof,
    prove_pr101_fec6_runtime_consumption,
)
from tac.repo_io import repo_relative, write_json  # noqa: E402

DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "submission_dir/archive.zip"
)
DEFAULT_RUNTIME_DIR = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "submission_dir"
)
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "runtime_consumption_proof_20260520_codex.json"
)
DEFAULT_ARCHIVE_SHA256 = (
    "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--expected-archive-sha256",
        default=DEFAULT_ARCHIVE_SHA256,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    proof = prove_pr101_fec6_runtime_consumption(
        archive_path=args.archive,
        runtime_dir=args.runtime_dir,
        expected_archive_sha256=args.expected_archive_sha256,
        repo_root=REPO_ROOT,
    )
    proof["runtime_consumption_proof_path"] = repo_relative(args.output_json, REPO_ROOT)
    output_parent = args.output_json.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_json, proof)
    sys.stdout.write(dumps_pr101_fec6_runtime_consumption_proof(proof))
    return 0 if proof.get("no_op_detector_passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
