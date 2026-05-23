#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a byte-closed decoder-q selective runtime submission directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.decoder_q_selective_runtime_materializer import (  # noqa: E402
    materialize_selective_runtime_candidate,
    write_json,
)

DEFAULT_FEC6_SUBMISSION_DIR = Path(
    "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "submission_dir"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="DQS1 packet plan JSON or decoder-q selective bridge-plan JSON.",
    )
    parser.add_argument(
        "--base-submission-dir",
        type=Path,
        default=DEFAULT_FEC6_SUBMISSION_DIR,
        help="FEC6 submission_dir to copy before patching inflate.py and archive.zip.",
    )
    parser.add_argument(
        "--base-archive",
        type=Path,
        help="Base FEC6 archive.zip. Defaults to --base-submission-dir/archive.zip.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output submission directory to create.",
    )
    parser.add_argument(
        "--frame-policy",
        choices=("pair_all_frames", "segnet_last_frame_only"),
        default="pair_all_frames",
        help="Frame mapping policy when deriving a packet from a bridge plan.",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        help="Limit bridge-plan work units; omit to consume the full bridge plan.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Optional extra copy of the materialization manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace --output-dir if it already exists and --expected-output-tree-sha256 matches.",
    )
    parser.add_argument(
        "--expected-output-tree-sha256",
        default=None,
        help="required tree sha256 of existing --output-dir when --force replaces it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_archive = args.base_archive or args.base_submission_dir / "archive.zip"
    manifest = materialize_selective_runtime_candidate(
        plan_path=args.plan,
        base_submission_dir=args.base_submission_dir,
        base_archive=base_archive,
        output_dir=args.output_dir,
        repo_root=REPO_ROOT,
        frame_policy=args.frame_policy,
        max_units=args.max_units,
        force=args.force,
        expected_output_tree_sha256=args.expected_output_tree_sha256,
    )
    if args.manifest_output:
        write_json(args.manifest_output, manifest)
    print(
        json.dumps(
            {
                "output_submission_dir": manifest["output_submission_dir"],
                "archive_zip_sha256": manifest["materialized_archive"]["zip_sha256"],
                "member_bytes": manifest["materialized_archive"]["member_bytes"],
                "dqs1_payload_bytes": manifest["dqs1_payload"]["payload_bytes"],
                "selected_pair_count": manifest["dqs1_payload"]["pair_count"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
