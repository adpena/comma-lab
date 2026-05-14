#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe one PR103 arithmetic histogram retarget without emitting an archive."""

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

from tac.pr103_arithmetic_transform_plan import (  # noqa: E402
    DEFAULT_STRATEGY,
    Pr103ArithmeticTransformPlanError,
    build_pr103_arithmetic_histogram_beam_probe,
    build_pr103_arithmetic_histogram_coordinate_probe,
    build_pr103_arithmetic_histogram_global_combo_probe,
    build_pr103_arithmetic_retarget_probe,
    render_beam_markdown,
    render_coordinate_markdown,
    render_global_combo_markdown,
    render_retarget_markdown,
    resolve_pr103_combo_worker_count,
)
from tac.repo_io import json_text, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-manifest", required=True, type=Path)
    parser.add_argument(
        "--source-archive",
        type=Path,
        help="Override source archive path. Defaults to source_archive.path in the manifest.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target-label", help="Exact target stream label.")
    group.add_argument("--target-rank", type=int, help="One-based target rank.")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument(
        "--probe-mode",
        choices=(
            "baseline-retarget",
            "coordinate-search",
            "beam-search",
            "global-combo-search",
        ),
        default="baseline-retarget",
    )
    parser.add_argument("--top-symbols", type=int, default=32)
    parser.add_argument(
        "--top-per-stream",
        type=int,
        default=20,
        help="Per-stream frontier depth for global-combo-search.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument(
        "--beam-probe-report",
        action="append",
        type=Path,
        default=[],
        help="Beam probe JSON input. Required for global-combo-search; may be repeated.",
    )
    parser.add_argument(
        "--deltas",
        default="-2,-1,1,2",
        help="Comma-separated q8 weight deltas for coordinate and beam search modes.",
    )
    parser.add_argument(
        "--max-global-combo-states",
        type=int,
        default=25_000,
        help=(
            "Fail closed before a serial global-combo probe if the estimated "
            "state count exceeds this budget. Use --allow-slow-global-combo "
            "only after adding parallel/vectorized execution or accepting the "
            "wall-clock cost."
        ),
    )
    parser.add_argument(
        "--allow-slow-global-combo",
        action="store_true",
        help="Allow a global-combo probe above --max-global-combo-states.",
    )
    parser.add_argument(
        "--global-combo-workers",
        type=int,
        default=0,
        help=(
            "Worker count for exact global-combo candidate scoring. Zero uses "
            "PACT_PR103_COMBO_WORKERS or the local CPU count, capped by the "
            "library worker budget."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit 1 when the probe is not archive-preflight ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    input_paths = [
        args.schema_manifest,
        *([args.source_archive] if args.source_archive is not None else []),
        *args.beam_probe_report,
    ]
    try:
        if args.probe_mode == "global-combo-search":
            combo_workers = resolve_pr103_combo_worker_count(args.global_combo_workers)
            estimated_states = _estimate_global_combo_state_count(
                stream_count=len(args.beam_probe_report),
                top_per_stream=args.top_per_stream,
                beam_width=args.beam_width,
            )
            if (
                estimated_states > args.max_global_combo_states * combo_workers
                and not args.allow_slow_global_combo
            ):
                print(
                    "FATAL: PR103 global-combo probe would evaluate about "
                    f"{estimated_states} serial states; budget is "
                    f"{args.max_global_combo_states} states per worker "
                    f"across {combo_workers} worker(s). Lower "
                    "--top-per-stream/--beam-width, increase "
                    "--global-combo-workers, or pass --allow-slow-global-combo "
                    "after accepting the wall-clock cost.",
                    file=sys.stderr,
                )
                return 2
            report = build_pr103_arithmetic_histogram_global_combo_probe(
                schema_manifest=args.schema_manifest,
                source_archive=args.source_archive,
                beam_probe_reports=args.beam_probe_report,
                top_per_stream=args.top_per_stream,
                beam_width=args.beam_width,
                combo_workers=combo_workers,
                repo_root=REPO_ROOT,
            )
            renderer = render_global_combo_markdown
        elif args.probe_mode == "beam-search":
            report = build_pr103_arithmetic_histogram_beam_probe(
                schema_manifest=args.schema_manifest,
                source_archive=args.source_archive,
                target_label=args.target_label,
                target_rank=args.target_rank,
                top_symbols=args.top_symbols,
                deltas=_parse_deltas(args.deltas),
                rounds=args.rounds,
                beam_width=args.beam_width,
                repo_root=REPO_ROOT,
            )
            renderer = render_beam_markdown
        elif args.probe_mode == "coordinate-search":
            report = build_pr103_arithmetic_histogram_coordinate_probe(
                schema_manifest=args.schema_manifest,
                source_archive=args.source_archive,
                target_label=args.target_label,
                target_rank=args.target_rank,
                top_symbols=args.top_symbols,
                deltas=_parse_deltas(args.deltas),
                repo_root=REPO_ROOT,
            )
            renderer = render_coordinate_markdown
        else:
            report = build_pr103_arithmetic_retarget_probe(
                schema_manifest=args.schema_manifest,
                source_archive=args.source_archive,
                target_label=args.target_label,
                target_rank=args.target_rank,
                strategy=args.strategy,
                repo_root=REPO_ROOT,
            )
            renderer = render_retarget_markdown
    except (OSError, Pr103ArithmeticTransformPlanError) as exc:
        print(f"FATAL: PR103 arithmetic retarget probe failed: {exc}", file=sys.stderr)
        return 2

    report = attach_tool_run_manifest(
        report,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out is not None:
        write_json(args.json_out, report)
    else:
        print(json_text(report), end="")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(renderer(report), encoding="utf-8")
    if args.fail_if_blocked and report.get("ready_for_archive_preflight") is not True:
        return 1
    return 0


def _parse_deltas(value: str) -> list[int]:
    try:
        deltas = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise Pr103ArithmeticTransformPlanError(f"invalid --deltas value: {value}") from exc
    if not deltas:
        raise Pr103ArithmeticTransformPlanError("--deltas must include at least one integer")
    return deltas


def _estimate_global_combo_state_count(
    *,
    stream_count: int,
    top_per_stream: int,
    beam_width: int,
) -> int:
    """Return the serial state count for a global-combo frontier probe."""

    if stream_count <= 0 or top_per_stream <= 0 or beam_width <= 0:
        return 0
    option_count = top_per_stream + 1  # include source option
    evaluated = 0
    kept = 1
    for _ in range(stream_count):
        expanded = kept * option_count
        evaluated += expanded
        kept = min(beam_width, expanded)
    return evaluated


if __name__ == "__main__":
    raise SystemExit(main())
