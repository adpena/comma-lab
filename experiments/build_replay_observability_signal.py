#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fast offline signal table from archive bytes and component traces.

This CLI performs no inflate, scorer, CUDA, remote, or dispatch work. It joins
static public PR81/PR82 profiles, PR79/S2 exact component baseline JSON, archive
byte profiles, and diagnostic component_trace JSON into a ranked planning table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.archive_signal import build_signal_table, write_signal_outputs

DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/replay_observability_signal_20260503_codex"
DEFAULT_BASELINE_EXACT = (
    REPO_ROOT
    / "experiments/results/lightning_batch/exact_eval_pr79_s2_fixed_adaptive_actions_t4_20260503T173023Z/contest_auth_eval.json"
)
DEFAULT_PR81_PROFILE = (
    REPO_ROOT
    / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/pr81_qma9_semantic_range_mask_profile.json"
)
DEFAULT_PR82_PROFILE = (
    REPO_ROOT
    / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/pr82_henosis_frontier_static_profile.json"
)
DEFAULT_ARCHIVES = (
    REPO_ROOT / "experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/pr79_s2_fixed_adaptive_actions/archive.zip",
    REPO_ROOT / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/archive.zip",
    REPO_ROOT / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/archive.zip",
)
DEFAULT_COMPONENT_TRACE_GLOBS = (
    "experiments/results/lightning_batch/exact_eval_pr79*/component_trace.json",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-exact-json", type=Path, default=DEFAULT_BASELINE_EXACT)
    parser.add_argument("--baseline-archive-bytes", type=int, default=277_321)
    parser.add_argument("--baseline-score", type=float, default=0.31453355357318635)
    parser.add_argument("--archive", type=Path, action="append", default=[])
    parser.add_argument("--archive-profile-json", type=Path, action="append", default=[])
    parser.add_argument(
        "--no-default-archives",
        action="store_true",
        help="Do not auto-profile local PR79/S2, PR81, and PR82 ZIP containers when present.",
    )
    parser.add_argument("--pr81-profile-json", type=Path, default=DEFAULT_PR81_PROFILE)
    parser.add_argument("--pr82-profile-json", type=Path, default=DEFAULT_PR82_PROFILE)
    parser.add_argument("--component-trace-json", type=Path, action="append", default=[])
    parser.add_argument(
        "--component-trace-glob",
        action="append",
        default=[],
        help="Glob for diagnostic component_trace JSON. Repeatable. Relative globs are repo-root relative.",
    )
    parser.add_argument(
        "--no-default-component-traces",
        action="store_true",
        help="Do not auto-ingest PR79 component_trace JSON under experiments/results/lightning_batch.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT_DIR / "signal_table.json")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_DIR / "signal_table.md")
    parser.add_argument("--top-k", type=int, default=16)
    return parser


def _repo_relative_glob(pattern: str) -> str:
    path = Path(pattern)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    archive_paths = list(args.archive)
    if not args.no_default_archives:
        archive_paths.extend(path for path in DEFAULT_ARCHIVES if path.exists())
    component_trace_globs = [_repo_relative_glob(pattern) for pattern in args.component_trace_glob]
    if not args.no_default_component_traces:
        component_trace_globs.extend(_repo_relative_glob(pattern) for pattern in DEFAULT_COMPONENT_TRACE_GLOBS)

    table = build_signal_table(
        baseline_exact_json=args.baseline_exact_json if args.baseline_exact_json.exists() else None,
        baseline_archive_bytes=args.baseline_archive_bytes,
        baseline_score=args.baseline_score,
        archive_paths=archive_paths,
        archive_profile_jsons=args.archive_profile_json,
        pr81_profile_json=args.pr81_profile_json,
        pr82_profile_json=args.pr82_profile_json,
        component_trace_jsons=args.component_trace_json,
        component_trace_globs=component_trace_globs,
        top_k=args.top_k,
    )
    write_signal_outputs(
        table,
        json_out=args.output_json,
        markdown_out=args.output_md,
        markdown_top_k=args.top_k,
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"score_claim={table['score_claim']} dispatch_performed={table['dispatch_performed']}")
    print(f"rows={len(table['ranked_signal_rows'])} sources={len(table['sources'])}")
    for index, row in enumerate(table["top_dispatch_guidance"][: min(5, args.top_k)], start=1):
        print(
            f"{index}. {row['source_label']} {row['row_kind']} {row['name']} "
            f"priority={float(row.get('priority_score', 0.0)):.9f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
