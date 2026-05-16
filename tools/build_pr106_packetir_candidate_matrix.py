#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the PR106 PacketIR candidate evidence matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.pr106_candidate_matrix import (  # noqa: E402
    PR106_PACKETIR_CANDIDATE_MATRIX_DEFAULT_JSON,
    PR106_PACKETIR_CANDIDATE_MATRIX_DEFAULT_MD,
    write_pr106_packetir_candidate_matrix,
)
from tac.repo_io import json_text  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(PR106_PACKETIR_CANDIDATE_MATRIX_DEFAULT_JSON),
        help="Repo-relative or absolute JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(PR106_PACKETIR_CANDIDATE_MATRIX_DEFAULT_MD),
        help="Repo-relative or absolute Markdown output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = write_pr106_packetir_candidate_matrix(
        output_json=args.output_json,
        output_md=args.output_md,
        repo_root=REPO_ROOT,
    )
    sys.stdout.write(
        json_text(
            {
                "schema": matrix["schema"],
                "candidate_count": matrix["candidate_count"],
                "status_counts": matrix["status_counts"],
                "artifact_paths": matrix["artifact_paths"],
                "artifact_sha256": matrix["artifact_sha256"],
                "score_claim": matrix["score_claim"],
                "promotion_eligible": matrix["promotion_eligible"],
                "ready_for_exact_eval_dispatch": matrix[
                    "ready_for_exact_eval_dispatch"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
