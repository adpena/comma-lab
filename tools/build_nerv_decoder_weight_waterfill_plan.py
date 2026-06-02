#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a false-authority NeRV decoder-weight waterfill plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_decoder_weight_waterfill import (  # noqa: E402
    DEFAULT_EXCLUDE_SUBSTRINGS,
    DEFAULT_INCLUDE_SUBSTRINGS,
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
    build_nerv_decoder_weight_waterfill_plan,
    load_saliency_json,
    load_state_npz,
    load_state_npz_from_manifest,
    render_nerv_decoder_weight_waterfill_markdown,
)
from tac.repo_io import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument("--state-npz", type=Path)
    state_group.add_argument("--state-npz-manifest", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument("--saliency-json", default=None, type=Path)
    parser.add_argument("--family", default="hi_nerv")
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--include", action="append", default=None)
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--action-bits", default="0,2,4,8,16,32")
    parser.add_argument("--archive-sha256", default=None)
    parser.add_argument("--receiver-proof-status", default="missing")
    parser.add_argument("--full-video-coverage", action="store_true")
    parser.add_argument("--zero-run-overhead-bytes", default=2, type=int)
    args = parser.parse_args(argv)

    state = (
        load_state_npz(args.state_npz)
        if args.state_npz is not None
        else load_state_npz_from_manifest(args.state_npz_manifest)
    )
    saliency = (
        None if args.saliency_json is None else load_saliency_json(args.saliency_json)
    )
    report = build_nerv_decoder_weight_waterfill_plan(
        state,
        saliency_by_name=saliency,
        family=str(args.family),
        candidate_id=args.candidate_id,
        include_substrings=tuple(args.include) if args.include else DEFAULT_INCLUDE_SUBSTRINGS,
        exclude_substrings=tuple(args.exclude) if args.exclude else DEFAULT_EXCLUDE_SUBSTRINGS,
        action_bits=_parse_action_bits(args.action_bits),
        archive_sha256=args.archive_sha256,
        receiver_proof_status=str(args.receiver_proof_status),
        full_video_coverage=bool(args.full_video_coverage),
        zero_run_overhead_bytes=int(args.zero_run_overhead_bytes),
    )
    output = args.output_json.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output.as_posix()
    write_json(output, report)
    if args.output_md is not None:
        md_output = args.output_md.expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
        md_output.write_text(
            render_nerv_decoder_weight_waterfill_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _parse_action_bits(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(value).split(",") if part.strip())


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
        "report_path": report.get("report_path"),
        "group_count": report["group_count"],
        "total_selected_byte_delta": report["total_selected_byte_delta"],
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
        "blockers": report["blockers"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
