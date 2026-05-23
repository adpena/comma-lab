#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a non-promotional scorer-response dataset from advisory artifacts."""

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
    ResponseBaseline,
    ScorerResponseDatasetError,
    build_response_dataset,
    build_scorer_response_consumer_routing,
    render_markdown,
)


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
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], help="JSON path or glob; may repeat")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--consumer-routing-json-out",
        type=Path,
        help=(
            "Optional observability artifact that routes emitted rows through "
            "CONSUMES_SCORER_RESPONSE_DATASET cathedral consumers."
        ),
    )
    parser.add_argument("--baseline-score", type=float, required=True)
    parser.add_argument("--baseline-archive-bytes", type=int, required=True)
    parser.add_argument("--baseline-pose", type=float)
    parser.add_argument("--baseline-seg", type=float)
    parser.add_argument(
        "--include-distilled-vs-direct-rows",
        action="store_true",
        help=(
            "Opt in to PACT-NERV-DistilledScorer paired-smoke rows with "
            "family=distilled_vs_direct_scorer_paired_smoke."
        ),
    )
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help=(
            "Allow requested inputs to be skipped. Without this, skipped rows "
            "make the CLI fail closed after writing the diagnostic dataset."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input:
        print("FATAL: at least one --input path or glob is required", file=sys.stderr)
        return 2
    baseline = ResponseBaseline(
        score=args.baseline_score,
        archive_bytes=args.baseline_archive_bytes,
        avg_posenet_dist=args.baseline_pose,
        avg_segnet_dist=args.baseline_seg,
    )
    try:
        dataset = build_response_dataset(
            _expand_inputs(args.input),
            baseline=baseline,
            include_distilled_vs_direct_rows=args.include_distilled_vs_direct_rows,
        )
    except ScorerResponseDatasetError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(dataset), encoding="utf-8")
    consumer_routing_out = None
    consumer_routing_summary = None
    if args.consumer_routing_json_out is not None:
        routing = build_scorer_response_consumer_routing(dataset)
        args.consumer_routing_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.consumer_routing_json_out.write_text(
            json.dumps(routing, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        consumer_routing_out = str(args.consumer_routing_json_out)
        consumer_routing_summary = {
            "consumer_count": routing["consumer_count"],
            "row_count": routing["row_count"],
            "verdict_count": routing["verdict_count"],
        }
    skipped = dataset.get("skipped") if isinstance(dataset.get("skipped"), list) else []
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "consumer_routing_json_out": consumer_routing_out,
                "consumer_routing_summary": consumer_routing_summary,
                "summary": dataset["summary"],
                "skipped_count": len(skipped),
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if skipped and not args.allow_skipped:
        print(
            "FATAL: scorer-response dataset skipped requested input(s); "
            "rerun with --allow-skipped only for an audited diagnostic build",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
