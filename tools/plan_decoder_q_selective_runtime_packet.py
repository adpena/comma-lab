#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan a byte-closed selective decoder-q runtime packet."""

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

from tac.optimization.decoder_q_selective_runtime_packet import (  # noqa: E402
    DecoderQSelectiveRuntimePacketError,
    build_decoder_q_selective_runtime_packet_plan,
    dumps_json,
    load_json_object,
    render_decoder_q_selective_runtime_packet_markdown,
)

DEFAULT_BASE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
    / "submission_dir/archive.zip"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-plan", type=Path, required=True)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--frame-policy",
        choices=("pair_all_frames", "segnet_last_frame_only"),
        default="pair_all_frames",
        help="Which decoded frames receive the alternate decoder output.",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        help="Use only the top N bridge work units; useful for singleton proof planning.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bridge_plan = load_json_object(args.bridge_plan)
        plan = build_decoder_q_selective_runtime_packet_plan(
            bridge_plan,
            base_archive=args.base_archive,
            repo_root=REPO_ROOT,
            frame_policy=args.frame_policy,
            max_units=args.max_units,
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        DecoderQSelectiveRuntimePacketError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(dumps_json(plan), encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            render_decoder_q_selective_runtime_packet_markdown(plan),
            encoding="utf-8",
        )
    packet = plan["selective_packet"]
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "packet_status": plan["packet_status"],
                "frame_policy": packet["frame_policy"],
                "selected_pair_count": packet["selected_pair_count"],
                "affected_frame_count": packet["affected_frame_count"],
                "payload_bytes": packet["payload_bytes"],
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
