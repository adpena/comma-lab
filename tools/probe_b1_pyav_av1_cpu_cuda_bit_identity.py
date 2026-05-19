#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe B1 evaluator CPU/DALI decode identity for the contest video.

Default mode is local-first and CPU-only.  It deliberately does NOT claim AV1
or CUDA authority unless the actual video codec and evaluator CUDA/DALI leg
prove it.  The filename preserves the original routing-directive slug.
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
    build_b1_decode_identity_probe,
    write_json_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-video",
        type=Path,
        default=REPO_ROOT / "upstream/videos/0.mkv",
        help="Contest video path. Defaults to upstream/videos/0.mkv.",
    )
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--enable-cuda-decode",
        action="store_true",
        help="Attempt evaluator DaliVideoDataset CUDA decode; required for CPU/CUDA authority.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for the JSON report.",
    )
    parser.add_argument(
        "--register-probe-outcome",
        action="store_true",
        help="Append the summarized result to .omx/state/probe_outcomes.jsonl.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    report = build_b1_decode_identity_probe(
        video_path=args.upstream_video,
        frame_count=args.frame_count,
        start_frame=args.start_frame,
        enable_cuda_decode=args.enable_cuda_decode,
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
    kwargs["probe_id"] = f"b1_pyav_decode_identity_{_stamp()}"
    kwargs["evidence_path"] = report.get("evidence_path")
    kwargs["agent"] = "codex"
    kwargs["subagent_id"] = "codex_session_019de465_b1_phase1"
    kwargs["session_id"] = "019de465"
    kwargs["notes"] = (
        "B1 Phase 1 decode identity probe. This is not score authority; "
        "authority flags are in the JSON report."
    )
    register_probe_outcome(**kwargs)


def _stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
