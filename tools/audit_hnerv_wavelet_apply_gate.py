#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compute break-even readiness for WR01 HNeRV wavelet apply transforms."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_wavelet_apply_gate import build_wavelet_apply_gate_from_paths  # noqa: E402
from tac.repo_io import json_text  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidechannel-manifest", type=Path, required=True)
    parser.add_argument("--stacked-metadata", type=Path)
    parser.add_argument("--baseline-pose-dist", type=float)
    parser.add_argument("--required-component-margin", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-dispatch-blocked", action="store_true")
    args = parser.parse_args(argv)

    payload = build_wavelet_apply_gate_from_paths(
        sidechannel_manifest_path=args.sidechannel_manifest,
        stacked_metadata_path=args.stacked_metadata,
        baseline_pose_dist=args.baseline_pose_dist,
        required_component_margin=args.required_component_margin,
    )
    text = json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.fail_if_dispatch_blocked and payload["dispatch_blockers"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
