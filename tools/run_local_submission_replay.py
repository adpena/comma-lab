#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run cleanup-aware local replay for an arbitrary byte-closed submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from comma_lab.local_submission_replay import (
    run_local_submission_replay,
    stage_local_replay_submission,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-submission-dir", required=True, type=Path)
    parser.add_argument("--archive-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--upstream-root", type=Path, default=None)
    parser.add_argument("--video-names-file", type=Path, default=None)
    parser.add_argument("--keep-inflated", action="store_true")
    parser.add_argument(
        "--cleanup-failed-scratch",
        action="store_true",
        help=(
            "delete failed replay scratch only when paired with "
            "--certify-failed-scratch-rebuildable"
        ),
    )
    parser.add_argument(
        "--certify-failed-scratch-rebuildable",
        action="store_true",
        help="certify failed replay scratch can be regenerated from manifest inputs",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    submission_dir = stage_local_replay_submission(
        runtime_submission_dir=args.runtime_submission_dir,
        archive_zip_path=args.archive_zip,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    summary = run_local_submission_replay(
        submission_dir=submission_dir,
        source_runtime_submission_dir=args.runtime_submission_dir,
        archive_zip_path=args.archive_zip,
        device=args.device,
        upstream_root=args.upstream_root,
        video_names_file=args.video_names_file,
        keep_inflated=bool(args.keep_inflated),
        cleanup_failed_scratch=bool(args.cleanup_failed_scratch),
        certify_failed_scratch_rebuildable=bool(
            args.certify_failed_scratch_rebuildable
        ),
    )
    out = args.summary_json or (args.output_dir / "local_submission_replay_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(summary.to_json() + "\n", encoding="utf-8")
    print(summary.to_json())
    return 0 if summary.evaluation_passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
