#!/usr/bin/env python3
"""Classify the PR91/HPM1 first entropy failure with a suffix row scan."""

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
    PR91_HPM1_SPATIAL_ORDER_CANDIDATES,
    run_pr91_hpm1_next_row_suffix_scan_probe,
)
from tac.repo_io import json_text  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR91_ARCHIVE)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--probability-variant", default=DEFAULT_HPAC_PROBABILITY_VARIANT)
    parser.add_argument("--prob-eps", type=float, default=1e-7)
    parser.add_argument(
        "--spatial-order-candidate",
        default="tile_major_row_major",
        choices=PR91_HPM1_SPATIAL_ORDER_CANDIDATES,
        help=(
            "Spatial traversal hypothesis to replay before scanning the failed "
            "group suffix. Non-source values are local forensic probes only."
        ),
    )
    parser.add_argument(
        "--valid-preview-limit",
        type=int,
        default=16,
        help="Maximum valid alternate rows to include if any suffix row decodes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    payload = run_pr91_hpm1_next_row_suffix_scan_probe(
        args.archive,
        probability_variant=args.probability_variant,
        prob_eps=args.prob_eps,
        spatial_order_candidate=args.spatial_order_candidate,
        valid_preview_limit=args.valid_preview_limit,
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
