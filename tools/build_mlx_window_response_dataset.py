#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a non-authoritative MLX scorer-response dataset with per-window baselines."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.scorer_response_dataset import (  # noqa: E402
    ScorerResponseDatasetError,
    build_windowed_mlx_response_dataset,
    render_markdown,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact, write_text_artifact  # noqa: E402


def _expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value, recursive=True))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        unique.append(path)
        seen.add(resolved)
    return unique


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-response",
        action="append",
        default=[],
        help="MLX scorer-response JSON path or glob; may repeat.",
    )
    parser.add_argument(
        "--baseline-response",
        action="append",
        default=[],
        help="Baseline MLX scorer-response JSON path or glob; may repeat.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty dataset instead of failing when no usable candidate rows survive.",
    )
    parser.add_argument(
        "--expected-row-count",
        type=int,
        help="Fail unless the generated dataset has exactly this many rows.",
    )
    parser.add_argument(
        "--require-no-skipped",
        action="store_true",
        help="Fail if any candidate or baseline window was skipped.",
    )
    parser.add_argument(
        "--require-auth-audited-windows",
        action="store_true",
        default=True,
        help=(
            "Require candidate and baseline windows to carry passing auth-cache "
            "identity audits and matching reference tensor identities."
        ),
    )
    parser.add_argument(
        "--allow-unaudited-mlx-debug-dataset",
        action="store_false",
        dest="require_auth_audited_windows",
        help=(
            "Opt out of auth-cache window audits for explicit research/debug datasets. "
            "Rows remain non-authoritative and are not spend-triage or promotion inputs."
        ),
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256", default=None)
    parser.add_argument("--expected-md-sha256", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.candidate_response:
        print("FATAL: at least one --candidate-response is required", file=sys.stderr)
        return 2
    if not args.baseline_response:
        print("FATAL: at least one --baseline-response is required", file=sys.stderr)
        return 2
    try:
        dataset = build_windowed_mlx_response_dataset(
            candidate_paths=_expand_inputs(args.candidate_response),
            baseline_paths=_expand_inputs(args.baseline_response),
            require_auth_audited_windows=args.require_auth_audited_windows,
        )
    except ScorerResponseDatasetError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    row_count = int(dataset.get("summary", {}).get("row_count") or 0)
    if row_count == 0 and not args.allow_empty:
        print("FATAL: no usable MLX window response rows produced", file=sys.stderr)
        return 2
    if args.expected_row_count is not None and row_count != args.expected_row_count:
        print(
            "FATAL: MLX window response row count mismatch: "
            f"expected={args.expected_row_count}:actual={row_count}",
            file=sys.stderr,
        )
        return 2
    skipped_count = len(dataset.get("skipped") or [])
    if args.require_no_skipped and skipped_count:
        print(
            f"FATAL: MLX window response dataset has skipped rows: {skipped_count}",
            file=sys.stderr,
        )
        return 2
    try:
        write_json_artifact(
            args.json_out,
            dataset,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_output_sha256,
        )
        if args.md_out is not None:
            write_text_artifact(
                args.md_out,
                render_markdown(dataset),
                allow_overwrite=args.allow_overwrite,
                expected_existing_sha256=args.expected_md_sha256,
            )
    except ArtifactWriteError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "summary": dataset["summary"],
                "score_claim": dataset["score_claim"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
