#!/usr/bin/env python3
"""Probe whether PR85/QMA9 semantic teacher-forcing advances PR91/HPM1 replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.pr91_hpm1_codec import (  # noqa: E402
    DEFAULT_HPAC_PROBABILITY_VARIANT,
    DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
    DEFAULT_PR91_ARCHIVE,
    run_pr91_hpm1_reference_teacher_forcing_probe,
)
from tac.repo_io import json_text  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

DEFAULT_CANDIDATES = "tile_major_row_major,phase_major_row_major"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR91_ARCHIVE)
    parser.add_argument(
        "--reference-tokens",
        type=Path,
        default=DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
    )
    parser.add_argument(
        "--reference-layout",
        default="legacy_assume_nhw",
        choices=("legacy_assume_nhw", "nhw_render_order", "qma9_storage_wh_to_render_hw"),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--probability-variant", default=DEFAULT_HPAC_PROBABILITY_VARIANT)
    parser.add_argument("--prob-eps", type=float, default=1e-7)
    parser.add_argument("--max-frames", type=int, default=1)
    parser.add_argument(
        "--allow-unexpected-reference-sha",
        action="store_true",
        help=(
            "Run exploratory noncanonical reference-token probes instead of "
            "failing closed when the PR85/QMA9 SHA-256 does not match."
        ),
    )
    parser.add_argument(
        "--spatial-order-candidates",
        default=DEFAULT_CANDIDATES,
        help="Comma-separated spatial traversal candidates to teacher-force.",
    )
    return parser.parse_args(argv)


def _parse_candidates(text: str) -> tuple[str, ...]:
    candidates = tuple(part.strip() for part in text.split(",") if part.strip())
    if not candidates:
        raise SystemExit("--spatial-order-candidates must not be empty")
    return candidates


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    payload = run_pr91_hpm1_reference_teacher_forcing_probe(
        args.archive,
        reference_tokens_path=args.reference_tokens,
        reference_layout=args.reference_layout,
        probability_variant=args.probability_variant,
        prob_eps=args.prob_eps,
        max_frames=args.max_frames,
        candidates=_parse_candidates(args.spatial_order_candidates),
        require_expected_reference_sha=not args.allow_unexpected_reference_sha,
        write_json=False,
    )
    input_paths = [
        path for path in (args.archive, args.reference_tokens) if path.is_file()
    ]
    payload = attach_tool_run_manifest(
        payload,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    text = json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
