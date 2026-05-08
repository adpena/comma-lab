#!/usr/bin/env python3
"""Extract the PR101 HNeRV decoder state_dict from a public archive.zip.

This is a thin operator-facing CLI over
``tac.pr101_archive_state_loader.load_pr101_archive_state``. It performs no
scorer loads and no GPU work; it only validates the PR101 archive layout,
decodes the fixed decoder slice, and optionally materializes a torch
``state_dict`` plus JSON custody metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_archive_state_loader import (
    Pr101ArchiveStateLoaderError,
    load_pr101_archive_state,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="PR101 archive.zip path")
    parser.add_argument(
        "--output-state-dict",
        type=Path,
        default=None,
        help="Optional output .pt path for the decoded decoder state_dict",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional output JSON path for archive/member/slice custody metadata",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        loaded = load_pr101_archive_state(args.archive)
    except Pr101ArchiveStateLoaderError as exc:
        raise SystemExit(f"PR101 archive state load failed: {exc}") from exc

    if args.output_state_dict is not None:
        args.output_state_dict.parent.mkdir(parents=True, exist_ok=True)
        torch.save(loaded.state_dict, args.output_state_dict)

    metadata_json = json.dumps(loaded.metadata, indent=2, sort_keys=True)
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(metadata_json + "\n", encoding="utf-8")
    if args.output_state_dict is None and args.metadata_output is None:
        print(metadata_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
