#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the C0B-ABI0 mechanical source-identity control and receiver."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tac.witness_dsl.c0b_identity_receiver import (
    IdentityReceiverError,
    build_identity_archive,
    emit_standalone_runtime,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_FRAME_UTILS = REPO_ROOT / "upstream" / "frame_utils.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--frame-utils", type=Path, default=DEFAULT_FRAME_UTILS)
    parser.add_argument("--stage-pairs", type=int, default=12)
    parser.add_argument(
        "--source-origin",
        default="upstream/videos/0.mkv",
        help="Logical source identity written into the charged header.",
    )
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="Admit a tiny codec fixture while retaining false scientific/score authority labels.",
    )
    parser.add_argument(
        "--allow-local-spill",
        action="store_true",
        help="Explicitly opt into local-disk output instead of the SSD waterfall.",
    )
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    submission_dir = args.submission_dir
    archive_path = submission_dir / "archive.zip"
    runtime_source = REPO_ROOT / "src" / "tac" / "witness_dsl" / "c0b_identity_receiver.py"
    try:
        result = build_identity_archive(
            args.source_video,
            archive_path=archive_path,
            frame_utils_path=args.frame_utils,
            source_origin=args.source_origin,
            stage_pairs=args.stage_pairs,
            fixture_only=args.fixture_only,
            manifest_path=args.manifest,
            runtime_source_path=runtime_source,
            allow_local_spill=args.allow_local_spill,
        )
        runtime = emit_standalone_runtime(
            submission_dir,
            runtime_source_path=runtime_source,
        )
        if runtime.inflate_python_sha256 != result.manifest["runtime_source_sha256"]:
            raise IdentityReceiverError("emitted inflate.py differs from the charged runtime identity")
        if runtime.inflate_shell_sha256 != result.manifest["inflate_sh_sha256"]:
            raise IdentityReceiverError("emitted inflate.sh differs from the charged runtime identity")
    except (OSError, IdentityReceiverError) as exc:
        raise SystemExit(f"C0B-ABI0 identity-control archive build refused: {exc}") from exc
    print(
        json.dumps(
            {
                "archive_path": str(result.archive_path),
                "archive_bytes": result.archive_bytes,
                "archive_sha256": result.archive_sha256,
                "source_bytes": result.source_bytes,
                "source_sha256": result.source_sha256,
                "charged_state_bytes": result.state_bytes,
                "charged_state_sha256": result.state_sha256,
                "pair_count": result.pair_count,
                "manifest_path": str(result.manifest_path),
                "inflate_python_path": str(runtime.inflate_python_path),
                "inflate_python_sha256": runtime.inflate_python_sha256,
                "inflate_shell_path": str(runtime.inflate_shell_path),
                "inflate_shell_sha256": runtime.inflate_shell_sha256,
                "fixture_only": args.fixture_only,
                "role": "mechanical_identity_control",
                "identity_control_only": True,
                "research_only": True,
                "scientific_evidence": False,
                "scientific_state_composed": False,
                "c0b_gate_complete": False,
                "launch_ready": False,
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
