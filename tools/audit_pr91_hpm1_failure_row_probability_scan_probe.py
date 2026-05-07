#!/usr/bin/env python3
"""Scan PR91/HPM1 first-failure probability contracts without dispatch."""

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

from tac.pr86_hpac_codec import supported_hpac_probability_variant_names  # noqa: E402
from tac.pr91_hpm1_codec import (  # noqa: E402
    DEFAULT_HPAC_PROBABILITY_VARIANT,
    DEFAULT_PR91_ARCHIVE,
    DEFAULT_PR91_HPM1_FAILURE_ROW_PROB_EPS_VALUES,
    DEFAULT_PR91_HPM1_FAILURE_ROW_UNIFORM_MIX_MASSES,
    PR91_HPM1_SPATIAL_ORDER_CANDIDATES,
    run_pr91_hpm1_failure_row_probability_scan_probe,
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
    )
    parser.add_argument(
        "--scan-variants",
        default=",".join(supported_hpac_probability_variant_names()),
        help="Comma-separated HPAC probability variants to test at the failure row.",
    )
    parser.add_argument(
        "--scan-prob-eps-values",
        default=",".join(
            f"{value:g}" for value in DEFAULT_PR91_HPM1_FAILURE_ROW_PROB_EPS_VALUES
        ),
        help="Comma-separated positive prob_eps values for the failure-row scan.",
    )
    parser.add_argument(
        "--uniform-mix-masses",
        default=",".join(
            f"{value:g}" for value in DEFAULT_PR91_HPM1_FAILURE_ROW_UNIFORM_MIX_MASSES
        ),
        help=(
            "Comma-separated [0,1) uniform-mix masses to apply to the failed "
            "probability row before categorical construction."
        ),
    )
    parser.add_argument(
        "--decodable-preview-limit",
        type=int,
        default=16,
        help="Maximum decodable numeric mutation rows to include.",
    )
    return parser.parse_args(argv)


def _parse_csv(text: str, *, option: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise SystemExit(f"{option} must not be empty")
    return values


def _parse_float_csv(text: str, *, option: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError as exc:
            raise SystemExit(f"{option} must contain only floats") from exc
    if not values:
        raise SystemExit(f"{option} must not be empty")
    return tuple(values)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    payload = run_pr91_hpm1_failure_row_probability_scan_probe(
        args.archive,
        probability_variant=args.probability_variant,
        prob_eps=args.prob_eps,
        spatial_order_candidate=args.spatial_order_candidate,
        scan_variants=_parse_csv(args.scan_variants, option="--scan-variants"),
        scan_prob_eps_values=_parse_float_csv(
            args.scan_prob_eps_values,
            option="--scan-prob-eps-values",
        ),
        uniform_mix_masses=_parse_float_csv(
            args.uniform_mix_masses,
            option="--uniform-mix-masses",
        ),
        decodable_preview_limit=args.decodable_preview_limit,
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
