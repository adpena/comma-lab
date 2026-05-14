#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inventory PR91/HPM1 semantic decode readiness without dispatching eval."""

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
    DEFAULT_PR91_ARCHIVE,
    run_pr91_hpm1_semantic_decode_trench,
)
from tac.repo_io import json_text  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR91_ARCHIVE)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--probability-row-count", type=int, default=8)
    parser.add_argument("--prob-eps", type=float, default=1e-7)
    parser.add_argument(
        "--probability-variants",
        default=DEFAULT_HPAC_PROBABILITY_VARIANT,
        help="Comma-separated HPAC probability variants to inventory.",
    )
    parser.add_argument("--prefix-max-frames", type=int, default=1)
    parser.add_argument(
        "--skip-prefix-decode",
        action="store_true",
        help="Load model and probability rows only; do not consume token stream.",
    )
    return parser.parse_args(argv)


def _parse_variants(text: str) -> tuple[str, ...]:
    variants = tuple(part.strip() for part in text.split(",") if part.strip())
    if not variants:
        raise SystemExit("--probability-variants must not be empty")
    return variants


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    payload = run_pr91_hpm1_semantic_decode_trench(
        args.archive,
        probability_variants=_parse_variants(args.probability_variants),
        probability_row_count=args.probability_row_count,
        prob_eps=args.prob_eps,
        prefix_max_frames=args.prefix_max_frames,
        attempt_prefix_decode=not args.skip_prefix_decode,
        write_json=False,
    )
    input_paths = [args.archive] if args.archive.is_file() else []
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
