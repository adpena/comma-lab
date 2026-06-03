#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit active SNeRV/HiNeRV campaign telemetry ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_active_campaign_feedback_audit import (  # noqa: E402
    build_nerv_active_campaign_feedback_audit,
    render_nerv_active_campaign_feedback_audit_markdown,
)
from tac.repo_io import write_json_artifact, write_text_artifact  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claims",
        type=Path,
        default=REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md",
    )
    parser.add_argument(
        "--research-dir",
        type=Path,
        default=REPO_ROOT / ".omx/research",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--stale-epoch-tolerance", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_nerv_active_campaign_feedback_audit(
        claims_path=args.claims,
        repo_root=REPO_ROOT,
        research_dir=args.research_dir,
        stale_epoch_tolerance=args.stale_epoch_tolerance,
    )
    write_json_artifact(args.output_json, report)
    if args.output_md is not None:
        write_text_artifact(
            args.output_md,
            render_nerv_active_campaign_feedback_audit_markdown(report),
        )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "active_claim_count": report["active_claim_count"],
                "audited_claim_count": report["audited_claim_count"],
                "artifact_count": report["artifact_count"],
                "blocker_count": report["blocker_count"],
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report[
                    "ready_for_exact_eval_dispatch"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
