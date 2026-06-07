#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare a proven HiNeRV target-region sidecar action with backend fit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hinerv_target_region_action_comparison import (  # noqa: E402
    HI_NERV_TARGET_REGION_ACTION_COMPARISON_SCHEMA,
    build_hinerv_target_region_action_comparison_from_archive,
    write_hinerv_target_region_action_comparison,
)

DEFAULT_OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/experiments/results")


def _default_output_dir() -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / f"hinerv_target_region_action_comparison_{stamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="HIV1 archive.zip with target-region action")
    parser.add_argument(
        "--survival-receipt",
        type=Path,
        required=True,
        help="hi_nerv_target_region_action_parseback_survival/inflate_survival JSON",
    )
    parser.add_argument(
        "--runner-report",
        type=Path,
        default=None,
        help="compact runner report containing target_region_wall_normal_lift and sidecar candidate details",
    )
    parser.add_argument("--action-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = (args.output_dir or _default_output_dir()).expanduser().resolve(strict=False)
    report = build_hinerv_target_region_action_comparison_from_archive(
        args.archive,
        survival_receipt=args.survival_receipt,
        runner_report=args.runner_report,
        action_id=args.action_id,
    )
    written = write_hinerv_target_region_action_comparison(report, out_dir)
    summary = {
        "schema": HI_NERV_TARGET_REGION_ACTION_COMPARISON_SCHEMA,
        "output_dir": out_dir.as_posix(),
        **written,
        "action_id": report.get("action_id"),
        "support_sha256": report.get("support_sha256"),
        "next_blocker": (report.get("comparison") or {}).get("next_blocker"),
        "sidecar_current_inflate_survived": (report.get("comparison") or {}).get(
            "sidecar_current_inflate_survived"
        ),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
