#!/usr/bin/env python3
"""Build a PR106 HLM1 fixed-latent archive candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.hnerv_hlm1_archive_candidate import (  # noqa: E402
    build_hlm1_latent_archive_candidate,
)
from tac.repo_io import write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--candidate-member-name",
        choices=("0.bin", "x"),
        help=(
            "Optional single ZIP member name for the emitted archive. Use x for "
            "the PR101-style one-byte member-name rate repack; default preserves "
            "the source member name."
        ),
    )
    parser.add_argument(
        "--allow-rate-regression",
        action="store_true",
        help="Materialize even when HLM1 does not reduce section bytes.",
    )
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit nonzero if the archive candidate is blocked.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_hlm1_latent_archive_candidate(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        source_label=args.source_label,
        candidate_member_name=args.candidate_member_name,
        allow_rate_regression=args.allow_rate_regression,
        repo_root=REPO,
    )
    if args.json_out:
        write_json(args.json_out, manifest)
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.fail_if_blocked and not manifest["ready_for_archive_preflight"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
