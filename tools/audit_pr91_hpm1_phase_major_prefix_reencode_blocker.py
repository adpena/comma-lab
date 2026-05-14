#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the fail-closed PR91/HPM1 phase-major prefix re-encode blocker."""

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
    run_pr91_hpm1_phase_major_prefix_reencode_blocker_probe,
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
        "--range-prefix-seed-symbol-counts",
        default="1,8,64",
        help="Comma-separated positive seed prefix lengths to replay.",
    )
    parser.add_argument(
        "--range-prefix-replay-symbol-limit",
        type=int,
        default=64,
        help="Maximum prefix length to replay against the submitted stream.",
    )
    parser.add_argument(
        "--range-prefix-max-target-decoded-before",
        type=int,
        default=20000,
        help=(
            "Maximum target decoded-symbol prefix for this explicit forensic "
            "phase-major run."
        ),
    )
    return parser.parse_args(argv)


def _parse_positive_int_csv(text: str, *, option: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            value = int(stripped)
        except ValueError as exc:
            raise SystemExit(f"{option} must contain only integers") from exc
        if value <= 0:
            raise SystemExit(f"{option} values must be positive")
        values.append(value)
    if not values:
        raise SystemExit(f"{option} must not be empty")
    return tuple(values)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    payload = run_pr91_hpm1_phase_major_prefix_reencode_blocker_probe(
        args.archive,
        reference_tokens_path=args.reference_tokens,
        reference_layout=args.reference_layout,
        probability_variant=args.probability_variant,
        prob_eps=args.prob_eps,
        range_prefix_seed_symbol_counts=_parse_positive_int_csv(
            args.range_prefix_seed_symbol_counts,
            option="--range-prefix-seed-symbol-counts",
        ),
        range_prefix_replay_symbol_limit=args.range_prefix_replay_symbol_limit,
        range_prefix_max_target_decoded_before=(
            args.range_prefix_max_target_decoded_before
        ),
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
