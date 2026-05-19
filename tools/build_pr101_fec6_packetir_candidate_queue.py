#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the non-promotional PR101/FEC6 PacketIR candidate queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.pr101_fec6_candidate_queue import (  # noqa: E402
    build_pr101_fec6_packetir_candidate_queue,
    render_pr101_fec6_packetir_candidate_queue_markdown,
)
from tac.repo_io import json_text, write_json  # noqa: E402

DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "archive.zip"
)
DEFAULT_OPERATOR_MANIFEST = (
    REPO_ROOT
    / "experiments/results/fec6_selector_operator_space_20260517_codex/"
    "operator_space_manifest.json"
)
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "packetir_candidate_queue.json"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / ".omx/research/pr101_fec6_packetir_candidate_queue_20260519_codex.md"
)
DEFAULT_ARCHIVE_SHA256 = (
    "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--operator-space-manifest",
        type=Path,
        default=DEFAULT_OPERATOR_MANIFEST,
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--expected-archive-sha256",
        default=DEFAULT_ARCHIVE_SHA256,
    )
    parser.add_argument(
        "--runtime-consumption-proof",
        type=Path,
        default=None,
        help=(
            "Optional path to a Catalog #105 no-op-detector proof JSON. When "
            "supplied (and valid) the queue's runtime_consumption_proven flips "
            "True; otherwise top-level blockers surface "
            "`runtime_byte_consumption_noop_detector_missing` per codex "
            "adversarial review 2026-05-19 F1."
        ),
    )
    return parser.parse_args(argv)


_BLOCKERS_ONLY_RUNTIME_NOOP = (
    "runtime_byte_consumption_noop_detector_missing",
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue = build_pr101_fec6_packetir_candidate_queue(
        archive_path=args.archive,
        operator_space_manifest_path=args.operator_space_manifest,
        expected_archive_sha256=args.expected_archive_sha256,
        runtime_consumption_proof_path=args.runtime_consumption_proof,
    )
    write_json(args.output_json, queue)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        render_pr101_fec6_packetir_candidate_queue_markdown(queue),
        encoding="utf-8",
    )
    summary = {
        "schema": queue["schema"],
        "archive_sha256": queue["archive_sha256"],
        "candidate_count": queue["candidate_count"],
        "operator_candidate_count": queue["operator_candidate_count"],
        "materialized_new_archive_count": queue["materialized_new_archive_count"],
        # v2 split: parser_byte_accounting_passed (parser-domain truth) vs
        # runtime_consumption_proven (runtime-authority).
        "parser_byte_accounting_passed": queue["byte_accounting"][
            "parser_byte_accounting_passed"
        ],
        "runtime_consumption_proven": queue["byte_accounting"][
            "runtime_consumption_proven"
        ],
        # Legacy alias preserved for backward-compat dashboards; rebound to
        # the runtime-authority value per codex adversarial review 2026-05-19 F1.
        "byte_accounting_passed": queue["byte_accounting"][
            "runtime_consumed_byte_accounting_passed"
        ],
        "score_claim": queue["score_claim"],
        "promotion_eligible": queue["promotion_eligible"],
        "ready_for_exact_eval_dispatch": queue["ready_for_exact_eval_dispatch"],
        "blockers": queue["blockers"],
        "outputs": {
            "json": args.output_json.relative_to(REPO_ROOT).as_posix()
            if args.output_json.is_relative_to(REPO_ROOT)
            else args.output_json.as_posix(),
            "markdown": args.output_md.relative_to(REPO_ROOT).as_posix()
            if args.output_md.is_relative_to(REPO_ROOT)
            else args.output_md.as_posix(),
        },
    }
    sys.stdout.write(json_text(summary))
    # Exit codes:
    #   0 = no blockers (full runtime proof supplied + parser passed)
    #   3 = ONLY runtime-proof-missing (queue itself is well-formed; operator
    #       can re-run with --runtime-consumption-proof to clear)
    #   2 = other blockers present (true error state)
    if not queue["blockers"]:
        return 0
    blockers = set(queue["blockers"])
    if blockers.issubset(set(_BLOCKERS_ONLY_RUNTIME_NOOP)):
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
