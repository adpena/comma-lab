#!/usr/bin/env python3
"""Recover the deterministic local PR91/HPM1 submitted token prefix."""

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
    run_pr91_hpm1_submitted_prefix_token_recovery_probe,
)
from tac.repo_io import json_text  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


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
    parser.add_argument(
        "--max-symbols",
        type=int,
        help="Optional positive submitted prefix length cap; default runs to first failure.",
    )
    parser.add_argument(
        "--row-preview-limit",
        type=int,
        default=16,
        help="Maximum recovered symbol rows to include in the JSON preview.",
    )
    parser.add_argument(
        "--mismatch-limit",
        type=int,
        default=16,
        help="Maximum PR85/QMA9 reference mismatch rows to include.",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Recover submitted symbols without loading the PR85/QMA9 reference tensor.",
    )
    parser.add_argument(
        "--allow-unexpected-reference-sha",
        action="store_true",
        help=(
            "Run exploratory noncanonical reference-token probes instead of "
            "failing closed when the PR85/QMA9 SHA-256 does not match."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    reference_path = None if args.skip_reference else args.reference_tokens
    payload = run_pr91_hpm1_submitted_prefix_token_recovery_probe(
        args.archive,
        reference_tokens_path=reference_path,
        reference_layout=args.reference_layout,
        probability_variant=args.probability_variant,
        prob_eps=args.prob_eps,
        max_symbols=args.max_symbols,
        row_preview_limit=args.row_preview_limit,
        mismatch_limit=args.mismatch_limit,
        require_expected_reference_sha=not args.allow_unexpected_reference_sha,
        write_json=False,
    )
    input_paths = [args.archive] if args.archive.is_file() else []
    if reference_path is not None and reference_path.is_file():
        input_paths.append(reference_path)
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
