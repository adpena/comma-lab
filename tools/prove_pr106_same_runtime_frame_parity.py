#!/usr/bin/env python3
"""Streaming same-runtime PR106/R2 frame parity proof.

This compares two PR106 sidecar archives through one selected submission
runtime, hashing rendered uint8 frame bytes as they are produced. It does not
write `.raw` files and does not evaluate a score. Use `--max-pairs` only for
prefix smoke tests; omit it for full-frame parity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler import (  # noqa: E402
    dumps_runtime_consumption_manifest,
    prove_pr106_same_runtime_full_frame_parity,
)
from tac.repo_io import json_text  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--candidate-archive", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument(
        "--member-name",
        default=None,
        help=(
            "Expected ZIP member name. Omit to auto-detect the known single-member "
            "packet names: 0.bin or x."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-pairs", type=int)
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Prefix-smoke pair limit. Omit for full-frame parity.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prove_pr106_same_runtime_full_frame_parity(
        source_archive_path=args.source_archive,
        candidate_archive_path=args.candidate_archive,
        runtime_dir=args.runtime_dir,
        expected_member_name=args.member_name,
        device=args.device,
        batch_pairs=args.batch_pairs,
        max_pairs=args.max_pairs,
    )
    text = dumps_runtime_consumption_manifest(manifest)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text(manifest), encoding="utf-8")
    sys.stdout.write(text)
    if args.max_pairs is not None:
        return 0 if manifest.get("prefix_parity_claim") else 1
    return 0 if manifest.get("full_frame_inflate_output_parity_claim") else 1


if __name__ == "__main__":
    raise SystemExit(main())
