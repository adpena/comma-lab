#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a non-promotional per-frame view of a per-pair master-gradient tensor."""

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

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from tac.master_gradient import OperatingPoint, compute_marginal_coefficients  # noqa: E402
from tac.master_gradient_frame_decomposition import (  # noqa: E402
    FrameDecompositionConfig,
    MasterGradientFrameDecompositionError,
    axis_coefficients_from_anchor,
    decompose_per_pair_gradient_to_frames,
    json_ready_decomposition,
    load_anchor_for_gradient_path,
    render_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-npy", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--frame-axis-npy-out", type=Path)
    parser.add_argument("--ledger", type=Path, default=Path(".omx/state/master_gradient_anchors.jsonl"))
    parser.add_argument("--topology", choices=("non_overlapping", "sliding"), default="non_overlapping")
    parser.add_argument("--pose-first-frame-share", type=float, default=0.5)
    parser.add_argument("--rate-first-frame-share", type=float, default=0.5)
    parser.add_argument("--top-k-frames", type=int, default=20)
    parser.add_argument("--d-seg", type=float)
    parser.add_argument("--d-pose", type=float)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--score", type=float)
    parser.add_argument("--seg-coeff", type=float)
    parser.add_argument("--pose-coeff", type=float)
    parser.add_argument("--rate-coeff", type=float)
    return parser.parse_args(argv)


def _axis_coefficients(args: argparse.Namespace) -> tuple[tuple[float, float, float], str]:
    explicit_coeffs = (args.seg_coeff, args.pose_coeff, args.rate_coeff)
    if any(value is not None for value in explicit_coeffs):
        if not all(value is not None for value in explicit_coeffs):
            raise MasterGradientFrameDecompositionError(
                "--seg-coeff, --pose-coeff, and --rate-coeff must be provided together"
            )
        return (float(args.seg_coeff), float(args.pose_coeff), float(args.rate_coeff)), "explicit_coefficients"

    explicit_op = (args.d_seg, args.d_pose, args.rate, args.score)
    if any(value is not None for value in explicit_op):
        if not all(value is not None for value in explicit_op):
            raise MasterGradientFrameDecompositionError(
                "--d-seg, --d-pose, --rate, and --score must be provided together"
            )
        op = OperatingPoint(
            d_seg=float(args.d_seg),
            d_pose=float(args.d_pose),
            rate=float(args.rate),
            score=float(args.score),
        )
        return compute_marginal_coefficients(op), "explicit_operating_point"

    anchor = load_anchor_for_gradient_path(gradient_path=args.gradient_npy, ledger_path=args.ledger)
    if anchor is None:
        raise MasterGradientFrameDecompositionError(
            "no ledger anchor found for --gradient-npy; provide explicit coefficients or operating point"
        )
    return axis_coefficients_from_anchor(anchor), "ledger_anchor"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if np is None:
        print("FATAL: numpy required", file=sys.stderr)
        return 2
    try:
        coeffs, coefficient_source = _axis_coefficients(args)
        arr = np.load(args.gradient_npy, mmap_mode="r")
        payload = decompose_per_pair_gradient_to_frames(
            arr,
            axis_coefficients=coeffs,
            config=FrameDecompositionConfig(
                topology=args.topology,
                pose_first_frame_share=args.pose_first_frame_share,
                rate_first_frame_share=args.rate_first_frame_share,
                top_k_frames=args.top_k_frames,
            ),
        )
    except (OSError, ValueError, MasterGradientFrameDecompositionError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    payload["source_gradient_npy"] = str(args.gradient_npy)
    payload["coefficient_source"] = coefficient_source
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(json_ready_decomposition(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    if args.frame_axis_npy_out is not None:
        args.frame_axis_npy_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.frame_axis_npy_out, payload["frame_axis_l1"])
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "md_out": None if args.md_out is None else str(args.md_out),
                "frame_axis_npy_out": None
                if args.frame_axis_npy_out is None
                else str(args.frame_axis_npy_out),
                "topology": payload["topology"],
                "n_pairs": payload["n_pairs"],
                "n_frames": payload["n_frames"],
                "coefficient_source": coefficient_source,
                "conservation_ok": payload["conservation_ok"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
