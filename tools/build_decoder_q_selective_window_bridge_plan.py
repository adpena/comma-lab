#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed decoder-q selective-window bridge work order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.decoder_q_selective_window_bridge import (  # noqa: E402
    DecoderQSelectiveWindowBridgeError,
    build_decoder_q_selective_window_bridge_plan,
    dumps_json,
    load_json_object,
    render_decoder_q_selective_window_bridge_markdown,
    source_artifact_metadata,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Use only the top N selected windows from the strict MLX selector.",
    )
    parser.add_argument(
        "--coalesce-gap",
        type=int,
        default=0,
        help=(
            "Coalesce adjacent work units into non-authoritative run probes when "
            "the next start <= current end + gap. Default only coalesces touching "
            "singleton windows."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selection = load_json_object(args.selection)
        candidate_manifest = load_json_object(args.candidate_manifest)
        plan = build_decoder_q_selective_window_bridge_plan(
            selection,
            candidate_manifest,
            repo_root=REPO_ROOT,
            lane_id=args.lane_id,
            max_windows=args.max_windows,
            coalesce_gap=args.coalesce_gap,
            source_artifacts=source_artifact_metadata(
                {
                    "selection": args.selection,
                    "candidate_manifest": args.candidate_manifest,
                }
            ),
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        DecoderQSelectiveWindowBridgeError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(dumps_json(plan), encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            render_decoder_q_selective_window_bridge_markdown(plan),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "bridge_status": plan["bridge_status"],
                "selected_window_count": plan["summary"]["selected_window_count"],
                "coalesced_run_count": plan["summary"]["coalesced_run_count"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
