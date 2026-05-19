#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe B1 contest-video patch density.

By default this measures held-out contest-video self-density only.  A B1
substrate dispatch still requires a rendered-frontier query source with archive
and runtime custody; use ``--query-is-rendered-frontier`` only when that is true.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.contest_exploits.contest_video_codebook import (  # noqa: E402
    build_b1_patch_density_probe,
    write_json_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-video",
        type=Path,
        default=REPO_ROOT / "upstream/videos/0.mkv",
        help="Contest video used as decoder-side codebook.",
    )
    parser.add_argument(
        "--query-video",
        type=Path,
        help="Optional query video. Defaults to held-out frames from upstream video.",
    )
    parser.add_argument("--codebook-frame-count", type=int, default=8)
    parser.add_argument("--query-frame-count", type=int, default=4)
    parser.add_argument("--codebook-start-frame", type=int, default=0)
    parser.add_argument("--query-start-frame", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max-codebook-patches", type=int, default=4096)
    parser.add_argument("--max-query-patches", type=int, default=1024)
    parser.add_argument("--threshold-rmse", type=float, default=10.0)
    parser.add_argument(
        "--nn-backend",
        choices=("numpy", "faiss"),
        default="numpy",
        help=(
            "Nearest-neighbor backend. Faiss is opt-in because local evaluator "
            "decode can already load OpenMP through torch."
        ),
    )
    parser.add_argument(
        "--query-is-rendered-frontier",
        action="store_true",
        help="Assert query frames are inflated frontier output with custody.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--register-probe-outcome",
        action="store_true",
        help="Append the summarized result to .omx/state/probe_outcomes.jsonl.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    report = build_b1_patch_density_probe(
        upstream_video=args.upstream_video,
        query_video=args.query_video,
        codebook_frame_count=args.codebook_frame_count,
        query_frame_count=args.query_frame_count,
        codebook_start_frame=args.codebook_start_frame,
        query_start_frame=args.query_start_frame,
        patch_size=args.patch_size,
        stride=args.stride,
        max_codebook_patches=args.max_codebook_patches,
        max_query_patches=args.max_query_patches,
        threshold_rmse=args.threshold_rmse,
        nn_backend=args.nn_backend,
        query_is_rendered_frontier=args.query_is_rendered_frontier,
        generated_at_utc=generated_at,
    )
    if args.output_json:
        report["evidence_path"] = str(write_json_report(report, args.output_json))
    if args.register_probe_outcome:
        _register_probe_outcome(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _register_probe_outcome(report: dict) -> None:
    from tac.probe_outcomes_ledger import register_probe_outcome

    kwargs = dict(report["probe_outcome_kwargs"])
    kwargs["probe_id"] = f"b1_patch_density_{_stamp()}"
    kwargs["evidence_path"] = report.get("evidence_path")
    kwargs["agent"] = "codex"
    kwargs["subagent_id"] = "codex_session_019de465_b1_phase1"
    kwargs["session_id"] = "019de465"
    kwargs["notes"] = (
        "B1 Phase 1 patch-density probe. Held-out upstream self-density is "
        "machinery evidence only unless query_is_rendered_frontier is true."
    )
    register_probe_outcome(**kwargs)


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
