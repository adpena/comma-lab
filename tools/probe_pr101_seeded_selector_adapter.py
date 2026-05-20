#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe PR101/FEC6 seeded selector adapter feasibility.

This is a no-score, no-dispatch profiler. It parses the current FEC6 selector
stream, searches deterministic archive-charged seeds for simple selector
priors, charges residual override bytes, and reports whether the adapter beats
the current FEC6 selector payload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.pr101_seeded_selector_adapter import (  # noqa: E402
    profile_archive_seeded_selector_adapter,
    render_seeded_selector_profile_markdown,
)
from tac.repo_io import repo_relative  # noqa: E402

DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--seed-lengths", type=_parse_int_csv, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--search-seeds-per-length", type=int, default=256)
    parser.add_argument("--generator-kind", choices=("xorshift", "lcg", "pcg64"), default="pcg64")
    parser.add_argument("--target-saving-bytes", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    profile = profile_archive_seeded_selector_adapter(
        args.archive.read_bytes(),
        seed_lengths=args.seed_lengths,
        search_seeds_per_length=args.search_seeds_per_length,
        generator_kind=args.generator_kind,
        target_saving_bytes=args.target_saving_bytes,
    )
    profile["source_archive"] = {
        "path": repo_relative(args.archive, REPO_ROOT),
        "bytes": args.archive.stat().st_size,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_seeded_selector_profile_markdown(profile) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": profile["schema"],
                "best_candidate": profile["best_candidate"]["candidate_id"],
                "best_payload_bytes": profile["best_candidate"]["payload_bytes"],
                "fec6_selector_payload_bytes": profile["fec6_selector_payload_bytes"],
                "can_meet_target": profile[
                    "can_meet_target_with_seeded_selector_adapter"
                ],
                "score_claim": profile["score_claim"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
