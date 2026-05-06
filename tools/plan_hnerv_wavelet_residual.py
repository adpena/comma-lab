#!/usr/bin/env python3
"""Emit a planning-only wavelet residual atom manifest for HNeRV payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_wavelet_residual import build_wavelet_residual_plan, plan_digest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--target-section", action="append", dest="target_sections")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--quant-step", type=float, default=1.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-blocked", action="store_true")
    args = parser.parse_args()

    scorecard = json.loads(args.scorecard.read_text(encoding="utf-8"))
    plan = build_wavelet_residual_plan(
        source_archive=str(args.source_archive),
        scorecard=scorecard,
        source_label=args.source_label,
        target_sections=tuple(args.target_sections or ("latents_and_sidecar_brotli",)),
        top_k=args.top_k,
        block_size=args.block_size,
        quant_step=args.quant_step,
    )
    plan["plan_sha256"] = plan_digest(plan)
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.fail_if_blocked and not plan["ready_for_wavelet_candidate_build"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
