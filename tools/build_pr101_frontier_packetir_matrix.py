#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the PR101/FEC6 frontier PacketIR authority matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.pr101_frontier_packetir_matrix import (  # noqa: E402
    PR101_FRONTIER_PACKETIR_MATRIX_DEFAULT_JSON,
    PR101_FRONTIER_PACKETIR_MATRIX_DEFAULT_MD,
    write_pr101_frontier_packetir_matrix,
)
from tac.repo_io import json_text  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(PR101_FRONTIER_PACKETIR_MATRIX_DEFAULT_JSON),
        help="Repo-relative or absolute JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(PR101_FRONTIER_PACKETIR_MATRIX_DEFAULT_MD),
        help="Repo-relative or absolute Markdown output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = write_pr101_frontier_packetir_matrix(
        output_json=args.output_json,
        output_md=args.output_md,
        repo_root=REPO_ROOT,
    )
    summary = matrix["authority_summary"]
    sys.stdout.write(
        json_text(
            {
                "schema": matrix["schema"],
                "status": matrix["status"],
                "score_claim": matrix["score_claim"],
                "promotion_eligible": matrix["promotion_eligible"],
                "ready_for_exact_eval_dispatch": matrix[
                    "ready_for_exact_eval_dispatch"
                ],
                "fec6_has_contest_cpu_evidence": summary[
                    "fec6_has_contest_cpu_evidence"
                ],
                "fec6_has_contest_cuda_evidence": summary[
                    "fec6_has_contest_cuda_evidence"
                ],
                "fec6_has_paired_exact_same_archive_runtime": summary[
                    "fec6_has_paired_exact_same_archive_runtime"
                ],
                "fec6_has_parser_profile_evidence": summary[
                    "fec6_has_parser_profile_evidence"
                ],
                "fec6_has_packetir_identity_evidence": summary[
                    "fec6_has_packetir_identity_evidence"
                ],
                "fec6_has_deterministic_compiler_identity_evidence": summary[
                    "fec6_has_deterministic_compiler_identity_evidence"
                ],
                "fec6_has_pr106_style_packetir_candidate_queue": summary[
                    "fec6_has_pr106_style_packetir_candidate_queue"
                ],
                "artifact_paths": matrix["artifact_paths"],
                "artifact_sha256": matrix["artifact_sha256"],
                "written_artifact_sha256": matrix["written_artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
