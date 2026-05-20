#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build LL scorer-surrogate frame/pair curriculum and masked knob plan."""

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

from tac.optimization.frame_pair_curriculum import (  # noqa: E402
    CurriculumConfig,
    FramePairCurriculumError,
    build_frame_pair_curriculum,
    load_frame_axis_npy,
    render_markdown,
)


def _read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FramePairCurriculumError(f"{path}: expected JSON object")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-axis-npy", type=Path, required=True)
    parser.add_argument("--decomposition-json", type=Path)
    parser.add_argument("--response-plan-json", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--topology", choices=("non_overlapping", "sliding"), default="non_overlapping")
    parser.add_argument("--top-k-frames", type=int, default=16)
    parser.add_argument("--top-k-pairs", type=int, default=8)
    parser.add_argument("--sampling-floor", type=float, default=1e-12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_frame_pair_curriculum(
            load_frame_axis_npy(args.frame_axis_npy),
            config=CurriculumConfig(
                topology=args.topology,
                top_k_frames=args.top_k_frames,
                top_k_pairs=args.top_k_pairs,
                sampling_floor=args.sampling_floor,
            ),
            decomposition_metadata=_read_json(args.decomposition_json),
            response_plan=_read_json(args.response_plan_json),
        )
    except (OSError, ValueError, FramePairCurriculumError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    payload["source_frame_axis_npy"] = str(args.frame_axis_npy)
    if args.decomposition_json is not None:
        payload["source_decomposition_json"] = str(args.decomposition_json)
    if args.response_plan_json is not None:
        payload["source_response_plan_json"] = str(args.response_plan_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "n_frames": payload["n_frames"],
                "n_pairs": payload["n_pairs"],
                "adjustment_layers": len(payload["adjustment_layers"]),
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
